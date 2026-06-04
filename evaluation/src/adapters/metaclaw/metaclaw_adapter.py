"""MetaClaw memory-system adapter for SubtleMemory evaluation.

This adapter uses MetaClaw's native Python memory subsystem as a SubtleMemory
evaluation memory-system backend. The add stage replays SubtleMemory sessions
through ``MemoryManager.ingest_session_turns``. The search stage is intentionally
skipped, and the answer stage calls MetaClaw's own chat-completions path so
MetaClaw performs recall and memory injection internally.
"""

from __future__ import annotations

import json
import asyncio
import os
import queue
import re
import sys
import threading
import time
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.utils.prompts import get_prompt
from evaluation.src.run_artifacts.models import (
    ImportManifestRecord,
    dataclass_to_dict,
)
from evaluation.src.core.data_models import Conversation, Message, SearchResult
from evaluation.src.core.readback import NormalizedStorageObject, StorageReadbackResult


METACLAW_QA_PROMPT_CATEGORY = "metaclaw"
METACLAW_DEFAULT_QA_PROMPT_KEY = "answer_prompt_subtlememory_metaclaw_v2"
METACLAW_EMPTY_QA_PROMPT_KEYS = {"", "none", "null", "off", "disabled", "empty"}
_ACTIVE_METACLAW_QUESTION_ID: ContextVar[str] = ContextVar(
    "active_metaclaw_question_id", default=""
)
_ACTIVE_METACLAW_FORMATTED_CONTEXT: ContextVar[str] = ContextVar(
    "active_metaclaw_formatted_context", default=""
)
_ACTIVE_METACLAW_RAW_QUERY: ContextVar[str] = ContextVar(
    "active_metaclaw_raw_query", default=""
)
_MUTATING_STORE_METHODS = {
    "add_memories",
    "import_memories_json",
    "expire_stale",
    "set_ttl",
    "set_type_ttl",
    "share_to_scope",
    "update_content",
    "merge_memories",
    "supersede",
    "compact",
    "archive",
    "bulk_archive",
    "add_tags",
    "remove_tags",
    "record_feedback",
    "update_reinforcement",
}


def _safe_component(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text or fallback


def _utc_iso(value: Optional[datetime]) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _session_sort_key(session_key: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", session_key)
    if match:
        return int(match.group(1)), session_key
    return 10**9, session_key


def _required_path_config(*values: Any, env_name: str, purpose: str) -> Path:
    for value in values:
        text = str(value or "").strip()
        if text:
            return Path(text).expanduser()
    raise ValueError(f"{purpose} is not configured. Set {env_name} in .env or the shell.")


@register_adapter("metaclaw")
class MetaClawAdapter(BaseAdapter):
    """MetaClaw native-memory adapter for the evaluation pipeline."""

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        metaclaw_cfg = config.get("metaclaw", {})
        self.metaclaw_root = _required_path_config(
            metaclaw_cfg.get("root")
            or os.environ.get("METACLAW_ROOT"),
            env_name="METACLAW_ROOT",
            purpose="MetaClaw root",
        )
        if str(self.metaclaw_root) not in sys.path:
            sys.path.insert(0, str(self.metaclaw_root))

        self.runtime_dir = Path(
            metaclaw_cfg.get("run_root") or self.output_dir / "metaclaw_runtime"
        )
        self.memory_dir = self.runtime_dir / "memory_data"
        self.memory_db_path = self.memory_dir / "memory.db"
        self.policy_path = self.memory_dir / "policy.json"
        self.telemetry_path = self.memory_dir / "telemetry.jsonl"
        self.memory_import_path = self.runtime_dir / "memory_import.json"

        memory_cfg = config.get("memory", {})
        search_cfg = config.get("search", {})
        self.top_k = int(search_cfg.get("top_k", memory_cfg.get("search_top_k", 20)))
        self.max_context_memories = int(
            search_cfg.get("max_context_memories", self.top_k)
        )
        self.max_context_chars = int(search_cfg.get("max_context_chars", 18000))
        self.memory_scope_prefix = str(
            memory_cfg.get("scope_prefix") or "evaluation"
        ).strip()
        supported = config.get("supported_datasets", ["subtlememory"])
        self.supported_datasets = {
            str(dataset).strip()
            for dataset in (supported if isinstance(supported, list) else [supported])
            if str(dataset).strip()
        }
        self._validate_supported_dataset()
        self.retrieval_mode = str(memory_cfg.get("retrieval_mode", "hybrid"))
        self.memory_use_embeddings = bool(memory_cfg.get("use_embeddings", False))
        self.memory_embedding_mode = str(memory_cfg.get("embedding_mode", "hashing"))
        self.memory_auto_consolidate = bool(memory_cfg.get("auto_consolidate", False))
        self.memory_flush_every = int(memory_cfg.get("flush_every", 50))

        llm_config = config.get("llm", {})
        self.llm_provider = str(llm_config.get("provider", "custom") or "custom")
        self.llm_model = str(llm_config.get("model", "gpt-4o-mini") or "")
        self.llm_api_key = str(llm_config.get("api_key", "") or "")
        self.llm_base_url = str(llm_config.get("base_url", "") or "")
        self.llm_temperature = float(llm_config.get("temperature", 0.0))
        self.llm_max_tokens = int(llm_config.get("max_tokens", 16384))
        self.num_workers = int(config.get("num_workers", 1))

        self._conversations: Dict[str, Conversation] = {}
        self._import_manifest_records: List[Dict[str, Any]] = []
        self._manager = None
        self._server = None
        self._scope_map: Dict[str, Dict[str, str]] = {}
        self._runtime_recall_payloads: Dict[str, Dict[str, Any]] = {}
        self._readback_answer_cache: Dict[str, Dict[str, Any]] = {}

        print("MetaClawAdapter initialized")
        print(f"   MetaClaw root: {self.metaclaw_root}")
        print(f"   Output Dir: {self.output_dir}")
        print(f"   Runtime Dir: {self.runtime_dir}")
        print(f"   Retrieval mode: {self.retrieval_mode}")
        print(f"   Num Workers: {self.num_workers}")

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "metaclaw",
            "config": self.config,
            "metaclaw_root": str(self.metaclaw_root),
            "runtime_dir": str(self.runtime_dir),
            "memory_db_path": str(self.memory_db_path),
            "telemetry_path": str(self.telemetry_path),
            "isolation": {
                "mode": "conversation_scope",
                "physical_store": "shared_run_sqlite",
            },
        }

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        del kwargs
        self._cache_conversations(conversations)
        self._reset_runtime()
        manager = self._get_manager()
        records: List[Dict[str, Any]] = []

        for conversation in conversations:
            scope_map = self._build_scope_map(conversation)
            self._scope_map[conversation.conversation_id] = scope_map
            session_groups = self._group_messages_by_session(conversation)
            for session_key, messages in session_groups:
                timestamp = self._session_timestamp(messages)
                scope_id = scope_map["all"]
                turns = self._messages_to_turns(messages=messages)
                if not turns:
                    continue
                session_id = self._build_session_id(
                    conversation.conversation_id, session_key
                )
                added = manager.ingest_session_turns(
                    session_id,
                    turns,
                    scope_id=scope_id,
                    timestamp_override=self._timestamp_override(timestamp),
                )
                memory_refs = self._memory_refs_for_session(
                    scope_id=scope_id, source_session_id=session_id
                )
                records.append(
                    self._build_manifest_record(
                        conversation=conversation,
                        session_key=session_key,
                        view_name="all",
                        scope_id=scope_id,
                        session_id=session_id,
                        message_count=len(messages),
                        added_count=added,
                        memory_refs=memory_refs,
                    )
                )

        self._import_manifest_records = records
        export_rows = self._export_active_memories()
        self.memory_import_path.write_text(
            json.dumps(export_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        return {
            "type": "metaclaw_memory_index",
            "conversation_ids": list(self._conversations.keys()),
            "conversation_count": len(self._conversations),
            "scope_map": self._scope_map,
            "isolation": {
                "mode": "conversation_scope",
                "physical_store": "shared_run_sqlite",
                "note": (
                    "All conversations in one evaluation run share one MetaClaw "
                    "SQLite store, but each conversation uses a distinct "
                    "run-scoped memory_scope."
                ),
            },
            "memory_db_path": str(self.memory_db_path),
            "policy_path": str(self.policy_path),
            "telemetry_path": str(self.telemetry_path),
            "memory_import_path": str(self.memory_import_path),
            "memory_count": len(export_rows),
        }

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any]:
        del output_dir
        self._cache_conversations(conversations)
        for conversation in conversations:
            self._scope_map[conversation.conversation_id] = self._build_scope_map(
                conversation
            )
        self._get_manager()
        return {
            "type": "metaclaw_memory_index",
            "conversation_ids": list(self._conversations.keys()),
            "conversation_count": len(self._conversations),
            "scope_map": self._scope_map,
            "isolation": {
                "mode": "conversation_scope",
                "physical_store": "shared_run_sqlite",
            },
            "memory_db_path": str(self.memory_db_path),
        }

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        del index
        started_at = time.perf_counter()
        question_id = str(kwargs.get("question_id") or "")
        scope_map = self._scope_map.get(conversation_id) or self._build_scope_map(
            self._conversations.get(conversation_id)
        )
        self._scope_map[conversation_id] = scope_map
        scope_id = self._scope_for_answer(conversation_id)
        retrieval_metadata = {
            "formatted_context": "",
            "retrieval_mode": "skipped_direct_metaclaw_answer",
            "source": "metaclaw_answer_stage",
            "note": (
                "Search is intentionally skipped. The answer stage calls "
                "MetaClaw's own chat-completions path, which recalls memories "
                "and injects them internally."
            ),
            "metaclaw_retrieval_mode": self.retrieval_mode,
            "metaclaw_isolation": "conversation_scope",
            "metaclaw_runtime": {
                "runtime_dir": str(self.runtime_dir),
                "memory_db_path": str(self.memory_db_path),
                "telemetry_path": str(self.telemetry_path),
            },
            "scope_id": scope_id,
            "top_k": self.top_k,
            "memory_db_path": str(self.memory_db_path),
            "memory_import_path": str(self.memory_import_path),
            "retrieved_count": 0,
        }
        return SearchResult(
            question_id=question_id,
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata=retrieval_metadata,
            retrieval_status="skipped",
            timing_ms=(time.perf_counter() - started_at) * 1000,
        )

    async def search_from_readback(
        self,
        query: str,
        conversation_id: str,
        index: Any = None,
        *,
        conversation: Optional[Conversation] = None,
        question_id: Optional[str] = None,
        question_metadata: Optional[Dict[str, Any]] = None,
        import_manifest_records: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> SearchResult:
        """Build prompt-ready context by reading MetaClaw memory units."""
        del index, kwargs
        started_at = time.perf_counter()
        session_ids = self._extract_question_session_ids(question_metadata or {})
        if not session_ids:
            return self._unsupported_readback_search_result(
                query=query,
                conversation_id=conversation_id,
                question_id=question_id,
                session_ids=[],
                reason="Readback search requires question_metadata.session_ids.",
            )

        conversation = conversation or self._conversations.get(conversation_id)
        if conversation is not None:
            self._conversations[conversation.conversation_id] = conversation
        metaclaw_session_ids, missing_raw_session_ids = (
            self._metaclaw_session_ids_for_question_sessions(
                conversation=conversation,
                conversation_id=conversation_id,
                session_ids=session_ids,
            )
        )
        if not metaclaw_session_ids:
            readback = StorageReadbackResult(
                status="empty",
                checked_session_ids=list(session_ids),
                missing_session_ids=missing_raw_session_ids,
                objects=[],
                metadata={
                    "provider": "metaclaw",
                    "question_id": question_id,
                    "conversation_id": conversation_id,
                    "metaclaw_session_ids": [],
                    "readback_scope": "question_sessions",
                },
                errors=[],
            )
            return SearchResult(
                question_id=question_id or "",
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "system": str(self.config.get("name") or "metaclaw"),
                    "search_mode": "readback",
                    "session_ids": session_ids,
                    "metaclaw_session_ids": [],
                    "formatted_context": "",
                    "readback": readback.to_dict(),
                },
                retrieval_status="empty",
                timing_ms=(time.perf_counter() - started_at) * 1000,
            )

        scope_id = self._scope_for_answer(conversation_id)
        session_set = set(metaclaw_session_ids)
        manager = self._get_manager()
        units = [
            unit
            for unit in manager.store.list_active(scope_id, limit=10000)
            if getattr(unit, "source_session_id", None) in session_set
        ]
        if not units:
            units = self._units_from_import_manifest_records(
                import_manifest_records or [],
                conversation_id=conversation_id,
                scope_id=scope_id,
                source_session_ids=session_set,
            )
        units_by_session: Dict[str, List[Any]] = {
            session_id: [] for session_id in metaclaw_session_ids
        }
        for unit in units:
            units_by_session.setdefault(unit.source_session_id, []).append(unit)

        ordered_units: List[Any] = []
        for session_id in metaclaw_session_ids:
            ordered_units.extend(units_by_session.get(session_id, []))
        ordered_units = ordered_units[: self.max_context_memories]

        results = [
            self._unit_to_result(unit, scope_id=scope_id, rank=rank)
            for rank, unit in enumerate(ordered_units, start=1)
        ]
        results = self._dedupe_results(results)
        formatted_context = self._render_units_for_prompt(ordered_units)
        missing_memory_sessions = [
            raw_session_id
            for raw_session_id, metaclaw_session_id in zip(
                session_ids, metaclaw_session_ids
            )
            if not units_by_session.get(metaclaw_session_id)
        ]
        missing_session_ids = self._dedupe_nonempty_strings(
            [*missing_raw_session_ids, *missing_memory_sessions]
        )
        readback_objects = [
            NormalizedStorageObject(
                session_id=str(result.get("metadata", {}).get("source_session_id") or ""),
                kind=str(result.get("metadata", {}).get("memory_type") or "memory"),
                id=str(result.get("metadata", {}).get("memory_id") or ""),
                content=str(result.get("content") or ""),
                metadata=dict(result.get("metadata") or {}),
            )
            for result in results
        ]
        readback = StorageReadbackResult(
            status="ok",
            checked_session_ids=list(session_ids),
            missing_session_ids=missing_session_ids,
            objects=readback_objects,
            metadata={
                "provider": "metaclaw",
                "question_id": question_id,
                "conversation_id": conversation_id,
                "scope_id": scope_id,
                "metaclaw_session_ids": metaclaw_session_ids,
                "readback_scope": "question_sessions",
            },
            errors=[],
        )
        if question_id:
            self._readback_answer_cache[str(question_id)] = {
                "conversation_id": conversation_id,
                "session_ids": list(session_ids),
                "metaclaw_session_ids": list(metaclaw_session_ids),
                "formatted_context": formatted_context,
                "results": list(results),
                "units": list(ordered_units),
                "scope_id": scope_id,
                "readback": readback.to_dict(),
            }
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": str(self.config.get("name") or "metaclaw"),
                "search_mode": "readback",
                "session_ids": session_ids,
                "metaclaw_session_ids": metaclaw_session_ids,
                "formatted_context": formatted_context,
                "scope_id": scope_id,
                "top_k": self.max_context_memories,
                "readback": readback.to_dict(),
            },
            retrieval_status="ok" if formatted_context else "empty",
            timing_ms=(time.perf_counter() - started_at) * 1000,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        conversation_id = str(kwargs.get("conversation_id") or "").strip()
        question_id = str(kwargs.get("question_id") or "").strip()
        prompt = self._get_metaclaw_answer_prompt(query)
        max_retries = int(self.config.get("answer", {}).get("max_retries", 3))
        question_token = _ACTIVE_METACLAW_QUESTION_ID.set(question_id)
        context_token = _ACTIVE_METACLAW_FORMATTED_CONTEXT.set(context or "")
        raw_query_token = _ACTIVE_METACLAW_RAW_QUERY.set(query)
        try:
            for attempt in range(max_retries):
                try:
                    answer = await self._call_metaclaw_answer(
                        prompt=prompt,
                        conversation_id=conversation_id,
                        question_id=question_id,
                        attempt=attempt + 1,
                    )
                    answer = self._clean_answer(answer)
                    if answer:
                        return answer
                except Exception:
                    if attempt == max_retries - 1:
                        raise
            return ""
        finally:
            _ACTIVE_METACLAW_QUESTION_ID.reset(question_token)
            _ACTIVE_METACLAW_FORMATTED_CONTEXT.reset(context_token)
            _ACTIVE_METACLAW_RAW_QUERY.reset(raw_query_token)

    def render_answer_prompt(
        self, query: str, context: str, **kwargs: Any
    ) -> Optional[str]:
        del context, kwargs
        return self._get_metaclaw_answer_prompt(query)

    def get_import_manifest_records(self) -> List[Dict[str, Any]]:
        return list(self._import_manifest_records)

    def pop_runtime_recall_payload(
        self, question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        key = str(question_id or "").strip()
        if key and key in self._runtime_recall_payloads:
            return self._runtime_recall_payloads.pop(key)
        if key:
            return {}
        if not self._runtime_recall_payloads:
            return {}
        first_key = next(iter(self._runtime_recall_payloads))
        return self._runtime_recall_payloads.pop(first_key)

    def pop_runtime_logger_payload(
        self, question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.pop_runtime_recall_payload(question_id)

    async def get_memory(self, memory_ref: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        del kwargs
        manager = self._get_manager()
        memory_id = str(memory_ref.get("memory_id") or "")
        unit = manager.store.get(memory_id) if memory_id else None
        if unit is None:
            return {
                "memory_ref": memory_ref,
                "storage_kind": "metaclaw_memory_unit",
                "content": "",
                "metadata": {"provider": "metaclaw"},
                "status": "not_found",
            }
        return {
            "memory_ref": memory_ref,
            "storage_kind": "metaclaw_memory_unit",
            "content": unit.content,
            "metadata": {
                "provider": "metaclaw",
                "memory_type": unit.memory_type.value,
                "scope_id": unit.scope_id,
                "source_session_id": unit.source_session_id,
                "created_at": unit.created_at,
                "updated_at": unit.updated_at,
                "summary": unit.summary,
                "entities": unit.entities,
                "topics": unit.topics,
                "importance": unit.importance,
            },
            "status": "ok",
        }

    async def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None
        self._server = None

    def _get_answer_prompt(self) -> str:
        answer_cfg = self.config.get("answer", {})
        prompt_key = answer_cfg.get(
            "metaclaw_prompt_key",
            answer_cfg.get("prompt_key", METACLAW_DEFAULT_QA_PROMPT_KEY),
        )
        if prompt_key is None:
            return ""
        prompt_key = str(prompt_key).strip()
        if prompt_key.lower() in METACLAW_EMPTY_QA_PROMPT_KEYS:
            return ""
        return get_prompt(METACLAW_QA_PROMPT_CATEGORY, prompt_key)

    def _get_metaclaw_answer_prompt(self, query: str) -> str:
        prompt = self._get_answer_prompt()
        if prompt:
            try:
                return prompt.format(context="", question=query)
            except KeyError:
                return prompt.format(question=query)
        return query

    def _is_readback_search_mode(self) -> bool:
        return (
            str((self.config.get("search") or {}).get("mode") or "")
            .strip()
            .lower()
            == "readback"
        )

    @staticmethod
    def _clean_answer(answer: str) -> str:
        answer = str(answer or "").strip()
        answer = re.sub(r"^Answer:\s*", "", answer, flags=re.IGNORECASE).strip()
        if "FINAL ANSWER:" in answer:
            answer = answer.split("FINAL ANSWER:", 1)[1].strip()
        return answer

    def _get_manager(self):
        if self._manager is not None:
            return self._manager
        from metaclaw.config import MetaClawConfig
        from metaclaw.memory.manager import MemoryManager

        cfg = MetaClawConfig(
            memory_enabled=True,
            memory_dir=str(self.memory_dir),
            memory_store_path=str(self.memory_db_path),
            memory_scope="evaluation",
            memory_policy_path=str(self.policy_path),
            memory_telemetry_path=str(self.telemetry_path),
            memory_auto_extract=True,
            memory_auto_consolidate=self.memory_auto_consolidate,
            memory_retrieval_mode=self.retrieval_mode,
            memory_use_embeddings=self.memory_use_embeddings,
            memory_embedding_mode=self.memory_embedding_mode,
            memory_max_injected_units=self.top_k,
            memory_max_injected_tokens=int(
                self.config.get("memory", {}).get("max_injected_tokens", 2400)
            ),
            memory_flush_every=self.memory_flush_every,
            record_enabled=False,
        )
        self._manager = MemoryManager.from_config(cfg)
        return self._manager

    def _get_server(self):
        if self._server is not None:
            return self._server
        from metaclaw.api_server import MetaClawAPIServer
        from metaclaw.config import MetaClawConfig

        adapter = self

        class EvaluationMetaClawAPIServer(MetaClawAPIServer):
            def _load_tokenizer(self):
                return None

            async def _inject_memory(self, messages, scope_id: str = ""):
                if not self.memory_manager:
                    adapter._last_metaclaw_runtime_capture = {
                        "messages": list(messages),
                        "retrieved_units": [],
                        "injected_context": "",
                        "status": "no_memory_manager",
                    }
                    return messages

                user_msgs = [m for m in messages if m.get("role") == "user"]
                task_desc = _ACTIVE_METACLAW_RAW_QUERY.get().strip()
                if not task_desc and user_msgs:
                    task_desc = MetaClawAdapter._flatten_message_content(
                        user_msgs[-1].get("content", "")
                    )
                if not task_desc:
                    adapter._last_metaclaw_runtime_capture = {
                        "messages": list(messages),
                        "retrieved_units": [],
                        "injected_context": "",
                        "status": "no_task_description",
                    }
                    return messages

                units, retrieval_source, staged_context = await adapter._resolve_recall_units_for_answer(
                    self.memory_manager,
                    task_desc=task_desc,
                    scope_id=scope_id,
                )
                if not units:
                    adapter._last_metaclaw_runtime_capture = {
                        "messages": list(messages),
                        "retrieved_units": [],
                        "injected_context": staged_context,
                        "status": f"no_memory_retrieved:{retrieval_source}",
                        "recall_source": retrieval_source,
                    }
                    if staged_context:
                        next_messages = list(messages)
                        sys_indices = [
                            index
                            for index, message in enumerate(next_messages)
                            if message.get("role") == "system"
                        ]
                        if sys_indices:
                            idx = sys_indices[0]
                            existing = MetaClawAdapter._flatten_message_content(
                                next_messages[idx].get("content", "")
                            )
                            next_messages[idx] = {
                                **next_messages[idx],
                                "content": existing + "\n\n" + staged_context,
                            }
                        else:
                            next_messages.insert(
                                0, {"role": "system", "content": staged_context}
                            )
                        adapter._last_metaclaw_runtime_capture = {
                            "messages": next_messages,
                            "retrieved_units": [],
                            "injected_context": staged_context,
                            "status": "captured_context_only",
                            "recall_source": retrieval_source,
                        }
                        return next_messages
                    return messages

                if retrieval_source == "readback_question_sessions" and staged_context:
                    memory_text = staged_context
                else:
                    memory_text = self.memory_manager.render_for_prompt(units)
                next_messages = list(messages)
                sys_indices = [
                    index
                    for index, message in enumerate(next_messages)
                    if message.get("role") == "system"
                ]
                if sys_indices:
                    idx = sys_indices[0]
                    existing = MetaClawAdapter._flatten_message_content(
                        next_messages[idx].get("content", "")
                    )
                    next_messages[idx] = {
                        **next_messages[idx],
                        "content": existing + "\n\n" + memory_text,
                    }
                else:
                    next_messages.insert(0, {"role": "system", "content": memory_text})

                adapter._last_metaclaw_runtime_capture = {
                    "messages": next_messages,
                    "retrieved_units": list(units),
                    "injected_context": memory_text,
                    "status": "captured",
                    "recall_source": retrieval_source,
                }
                return next_messages

        answer_cfg = self.config.get("answer", {})
        llm_config = self.config.get("llm", {})
        cfg = MetaClawConfig(
            mode="skills_only",
            api_key="",
            record_enabled=False,
            record_dir=str(self.runtime_dir / "records"),
            use_prm=False,
            use_opd=False,
            use_skills=False,
            enable_skill_evolution=False,
            memory_enabled=True,
            memory_dir=str(self.memory_dir),
            memory_store_path=str(self.memory_db_path),
            memory_scope="evaluation",
            memory_policy_path=str(self.policy_path),
            memory_telemetry_path=str(self.telemetry_path),
            memory_auto_extract=False,
            memory_manual_trigger=True,
            memory_auto_consolidate=self.memory_auto_consolidate,
            memory_retrieval_mode=self.retrieval_mode,
            memory_use_embeddings=self.memory_use_embeddings,
            memory_embedding_mode=self.memory_embedding_mode,
            memory_max_injected_units=self.top_k,
            memory_max_injected_tokens=int(
                self.config.get("memory", {}).get("max_injected_tokens", 2400)
            ),
            memory_flush_every=self.memory_flush_every,
            llm_provider=self.llm_provider,
            llm_auth_method=str(llm_config.get("auth_method", "api_key") or "api_key"),
            llm_api_base=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            llm_model_id=self.llm_model,
            served_model_name=self.llm_model or "metaclaw-qa",
            max_context_tokens=int(answer_cfg.get("max_context_tokens", 20000)),
        )
        submission_enabled = threading.Event()
        submission_enabled.set()
        manager = self._get_manager()
        self._make_manager_read_only(manager)
        self._server = EvaluationMetaClawAPIServer(
            config=cfg,
            output_queue=queue.Queue(),
            submission_enabled=submission_enabled,
            memory_manager=manager,
        )
        return self._server

    async def _resolve_recall_units_for_answer(
        self,
        manager: Any,
        *,
        task_desc: str,
        scope_id: str,
    ) -> Tuple[List[Any], str, str]:
        if self._is_readback_search_mode():
            question_id = _ACTIVE_METACLAW_QUESTION_ID.get()
            payload = self._readback_answer_cache.get(question_id, {})
            units = list(payload.get("units") or [])
            formatted_context = str(
                payload.get("formatted_context")
                or _ACTIVE_METACLAW_FORMATTED_CONTEXT.get()
                or ""
            )
            return units, "readback_question_sessions", formatted_context

        from metaclaw.memory.scope import base_scope

        retrieval_scope = base_scope(scope_id) if scope_id else None
        units = await asyncio.to_thread(
            manager.retrieve_for_prompt,
            task_desc,
            scope_id=retrieval_scope,
        )
        return list(units or []), "api_conversation_scope", ""

    async def _call_metaclaw_answer(
        self,
        *,
        prompt: str,
        conversation_id: str,
        question_id: str,
        attempt: int,
    ) -> str:
        server = self._get_server()
        scope_id = self._scope_for_answer(conversation_id)
        session_stem = _safe_component(
            question_id or f"{conversation_id}-{hash(prompt)}", "question"
        )
        session_id = f"answer-{session_stem}-{attempt}"
        payload = {
            "model": self.llm_model or "metaclaw-qa",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "session_id": session_id,
            "turn_type": "main",
            "session_done": False,
            "memory_scope": scope_id,
        }
        self._last_metaclaw_runtime_capture = {}
        try:
            response_payload = await server._handle_request(
                payload,
                session_id=session_id,
                turn_type="main",
                session_done=False,
                memory_scope=scope_id,
            )
        finally:
            self._cleanup_metaclaw_session(server, session_id)
        runtime_capture = getattr(self, "_last_metaclaw_runtime_capture", {}) or {}
        self._capture_metaclaw_runtime_recall_payload(
            question_id=question_id,
            conversation_id=conversation_id,
            session_id=session_id,
            scope_id=scope_id,
            prompt=prompt,
            original_messages=payload["messages"],
            final_messages=runtime_capture.get("messages", payload["messages"]),
            retrieved_units=runtime_capture.get("retrieved_units", []),
            injected_context=runtime_capture.get("injected_context", ""),
            status=runtime_capture.get("status", "no_memory_retrieved"),
            recall_source=runtime_capture.get(
                "recall_source",
                (
                    "readback_question_sessions"
                    if self._is_readback_search_mode()
                    else "api_conversation_scope"
                ),
            ),
        )
        response = response_payload.get("response", response_payload)
        answer = self._extract_answer_text(response)
        if not answer:
            raise RuntimeError(
                "MetaClaw answer response did not contain assistant content: "
                f"{json.dumps(response, ensure_ascii=False, default=str)[:1000]}"
            )
        return answer

    @staticmethod
    def _make_manager_read_only(manager: Any) -> None:
        store = getattr(manager, "store", None)
        if store is None or getattr(store, "_metaclaw_eval_read_only", False):
            return

        def mark_accessed(memory_ids: Iterable[str], accessed_at: str) -> None:
            del memory_ids, accessed_at
            return None

        def update_importance(
            memory_id: str, importance: float, updated_at: str
        ) -> None:
            del memory_id, importance, updated_at
            return None

        def forbid(action: str):
            def _raise(*args: Any, **kwargs: Any) -> None:
                del args, kwargs
                raise RuntimeError(
                    f"MetaClaw evaluation answer path is read-only; blocked {action}."
                )

            return _raise

        store.mark_accessed = mark_accessed
        store.update_importance = update_importance
        for method_name in _MUTATING_STORE_METHODS:
            if hasattr(store, method_name):
                setattr(store, method_name, forbid(method_name))
        store._metaclaw_eval_read_only = True

    @staticmethod
    def _cleanup_metaclaw_session(server: Any, session_id: str) -> None:
        for attr in (
            "_session_memory_turns",
            "_session_memory_scopes",
            "_session_turns",
            "_turn_counts",
            "_pending_turn_data",
            "_prm_tasks",
            "_teacher_tasks",
            "_pending_records",
            "_session_effective",
        ):
            mapping = getattr(server, attr, None)
            if isinstance(mapping, dict):
                mapping.pop(session_id, None)

    @staticmethod
    def _extract_answer_text(response_payload: Dict[str, Any]) -> str:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            return ""
        return MetaClawAdapter._flatten_message_content(message.get("content"))

    @staticmethod
    def _flatten_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part.strip() for part in parts if part.strip()).strip()
        return str(content or "").strip()

    def _capture_metaclaw_runtime_recall_payload(
        self,
        *,
        question_id: str,
        conversation_id: str,
        session_id: str,
        scope_id: str,
        prompt: str,
        original_messages: List[Dict[str, Any]],
        final_messages: List[Dict[str, Any]],
        retrieved_units: List[Any],
        injected_context: str,
        status: str,
        recall_source: str = "",
    ) -> Dict[str, Any]:
        del prompt
        retrieved_items = [
            self._unit_to_result(unit, scope_id=getattr(unit, "scope_id", scope_id), rank=rank)
            for rank, unit in enumerate(retrieved_units or [], start=1)
        ]
        payload = {
            "runtime_answer_session_id": session_id,
            "runtime_prompt_messages": list(final_messages or original_messages or []),
            "runtime_injected_context": str(injected_context or ""),
            "runtime_retrieved_items": retrieved_items,
            "runtime_logger_status": status or (
                "captured" if retrieved_items else "no_memory_retrieved"
            ),
        }
        if question_id:
            self._runtime_recall_payloads[question_id] = payload
        return payload

    def _capture_metaclaw_runtime_logger_payload(
        self,
        *,
        question_id: str,
        conversation_id: str,
        session_id: str,
        scope_id: str,
        prompt: str,
        original_messages: List[Dict[str, Any]],
        final_messages: List[Dict[str, Any]],
        retrieved_units: List[Any],
        injected_context: str,
        status: str,
        recall_source: str = "",
    ) -> Dict[str, Any]:
        return self._capture_metaclaw_runtime_recall_payload(
            question_id=question_id,
            conversation_id=conversation_id,
            session_id=session_id,
            scope_id=scope_id,
            prompt=prompt,
            original_messages=original_messages,
            final_messages=final_messages,
            retrieved_units=retrieved_units,
            injected_context=injected_context,
            status=status,
            recall_source=recall_source,
        )

    def _reset_runtime(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None
        self._server = None
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.memory_db_path,
            Path(f"{self.memory_db_path}-wal"),
            Path(f"{self.memory_db_path}-shm"),
            self.policy_path,
            Path(f"{self.policy_path}.history.jsonl"),
            self.telemetry_path,
            self.memory_import_path,
        ):
            if path.exists():
                path.unlink()

    def _cache_conversations(self, conversations: List[Conversation]) -> None:
        self._conversations = {conv.conversation_id: conv for conv in conversations}

    def _validate_supported_dataset(self) -> None:
        dataset_name = str(self.config.get("dataset_name") or "").strip()
        if not dataset_name or not self.supported_datasets:
            return
        if dataset_name in self.supported_datasets:
            return
        if any(dataset_name.startswith(f"{name}_") for name in self.supported_datasets):
            return
        allowed = ", ".join(sorted(self.supported_datasets))
        raise ValueError(
            f"MetaClaw adapter is configured for SubtleMemory-only evaluation; "
            f"dataset {dataset_name!r} is not supported. Allowed dataset(s): {allowed}."
        )

    def _build_scope_map(self, conversation: Optional[Conversation]) -> Dict[str, str]:
        if conversation is None:
            return {"all": self._scope_for("unknown")}
        conv_id = conversation.conversation_id
        return {"all": self._scope_for(conv_id)}

    def _scope_for(self, conversation_id: str) -> str:
        prefix = _safe_component(
            f"{self.memory_scope_prefix}_{self.run_context.get('run_id', 'run')}"
        )
        return f"{prefix}_{_safe_component(conversation_id)}"

    def _scope_for_answer(self, conversation_id: str) -> str:
        scope_map = self._scope_map.get(conversation_id)
        if not scope_map:
            scope_map = self._build_scope_map(self._conversations.get(conversation_id))
            self._scope_map[conversation_id] = scope_map
        return scope_map.get("all") or next(iter(scope_map.values()))

    @staticmethod
    def _extract_question_session_ids(metadata: Dict[str, Any]) -> List[str]:
        raw_session_ids = metadata.get("session_ids")
        if isinstance(raw_session_ids, list):
            values = raw_session_ids
        elif raw_session_ids:
            values = [raw_session_ids]
        else:
            values = []
        return MetaClawAdapter._dedupe_nonempty_strings(values)

    @staticmethod
    def _dedupe_nonempty_strings(values: Iterable[Any]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _unsupported_readback_search_result(
        self,
        *,
        query: str,
        conversation_id: str,
        question_id: Optional[str],
        session_ids: List[str],
        reason: str,
    ) -> SearchResult:
        readback = StorageReadbackResult(
            status="unsupported",
            checked_session_ids=list(session_ids),
            missing_session_ids=[],
            objects=[],
            metadata={
                "provider": "metaclaw",
                "question_id": question_id,
                "conversation_id": conversation_id,
                "reason": reason,
                "readback_scope": "question_sessions",
            },
            errors=[],
        )
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={
                "system": str(self.config.get("name") or "metaclaw"),
                "search_mode": "readback",
                "session_ids": list(session_ids),
                "metaclaw_session_ids": [],
                "formatted_context": "",
                "readback": readback.to_dict(),
            },
            retrieval_status="unsupported",
        )

    def _metaclaw_session_ids_for_question_sessions(
        self,
        *,
        conversation: Optional[Conversation],
        conversation_id: str,
        session_ids: List[str],
    ) -> Tuple[List[str], List[str]]:
        source_to_session_key: Dict[str, str] = {}
        if conversation is not None:
            for session_key, messages in self._group_messages_by_session(conversation):
                for message in messages:
                    source_session_id = str(
                        message.metadata.get("source_session_id") or ""
                    ).strip()
                    if source_session_id:
                        source_to_session_key.setdefault(
                            source_session_id, session_key
                        )
                source_to_session_key.setdefault(session_key, session_key)

        metaclaw_session_ids: List[str] = []
        missing_session_ids: List[str] = []
        for raw_session_id in session_ids:
            session_key = source_to_session_key.get(raw_session_id)
            if session_key is None:
                missing_session_ids.append(raw_session_id)
                session_key = raw_session_id
            metaclaw_session_ids.append(
                self._build_session_id(conversation_id, session_key)
            )
        return metaclaw_session_ids, missing_session_ids

    def _units_from_import_manifest_records(
        self,
        records: List[Dict[str, Any]],
        *,
        conversation_id: str,
        scope_id: str,
        source_session_ids: set[str],
    ) -> List[Any]:
        if not records:
            return []

        try:
            from metaclaw.memory.models import MemoryStatus, MemoryType, MemoryUnit
        except Exception:
            return []

        units: List[Any] = []
        seen_memory_ids: set[str] = set()
        for row in records:
            if str(row.get("conversation_id") or "") != conversation_id:
                continue
            for ref in row.get("memory_refs") or []:
                if not isinstance(ref, dict):
                    continue
                if str(ref.get("provider") or "") != "metaclaw":
                    continue
                source_session_id = str(ref.get("source_session_id") or "")
                if source_session_id not in source_session_ids:
                    continue
                content = str(ref.get("content") or "").strip()
                if not content:
                    continue
                memory_id = str(ref.get("memory_id") or "")
                if memory_id and memory_id in seen_memory_ids:
                    continue
                if memory_id:
                    seen_memory_ids.add(memory_id)
                try:
                    memory_type = MemoryType(str(ref.get("memory_type") or "episodic"))
                except ValueError:
                    memory_type = MemoryType.EPISODIC
                try:
                    status = MemoryStatus(str(ref.get("status") or "active"))
                except ValueError:
                    status = MemoryStatus.ACTIVE
                units.append(
                    MemoryUnit(
                        memory_id=memory_id,
                        scope_id=str(ref.get("scope_id") or scope_id),
                        memory_type=memory_type,
                        content=content,
                        summary=str(ref.get("summary") or ""),
                        source_session_id=source_session_id,
                        source_turn_start=int(ref.get("source_turn_start") or 0),
                        source_turn_end=int(ref.get("source_turn_end") or 0),
                        entities=list(ref.get("entities") or []),
                        topics=list(ref.get("topics") or []),
                        importance=float(ref.get("importance", 0.5) or 0.5),
                        confidence=float(ref.get("confidence", 0.7) or 0.7),
                        access_count=int(ref.get("access_count") or 0),
                        reinforcement_score=float(
                            ref.get("reinforcement_score", 0.0) or 0.0
                        ),
                        status=status,
                        supersedes=list(ref.get("supersedes") or []),
                        superseded_by=str(ref.get("superseded_by") or ""),
                        embedding=list(ref.get("embedding") or []),
                        created_at=str(ref.get("created_at") or ""),
                        updated_at=str(ref.get("updated_at") or ""),
                        last_accessed_at=str(ref.get("last_accessed_at") or ""),
                        expires_at=str(ref.get("expires_at") or ""),
                        tags=list(ref.get("tags") or []),
                    )
                )
        return units

    def _group_messages_by_session(
        self, conversation: Conversation
    ) -> List[Tuple[str, List[Message]]]:
        grouped: Dict[str, List[Message]] = {}
        for index, message in enumerate(conversation.messages):
            session_key = str(
                message.metadata.get("session")
                or message.metadata.get("source_session_id")
                or f"session_{index + 1}"
            )
            grouped.setdefault(session_key, []).append(message)
        return sorted(grouped.items(), key=lambda item: _session_sort_key(item[0]))

    def _messages_to_turns(self, *, messages: List[Message]) -> List[Dict[str, str]]:
        return [
            {
                "prompt_text": (
                    f"{message.sender_name} said on {_utc_iso(message.timestamp)}:\n"
                    f"{message.content}"
                ),
                "response_text": "",
            }
            for message in messages
            if str(message.content or "").strip()
        ]

    @staticmethod
    def _session_timestamp(messages: List[Message]) -> str:
        timestamps = [msg.timestamp for msg in messages if msg.timestamp is not None]
        return _utc_iso(min(timestamps) if timestamps else None)

    @staticmethod
    def _timestamp_override(timestamp_iso: str):
        from metaclaw.memory.models import MemoryTimestampOverride

        return MemoryTimestampOverride(
            created_at=timestamp_iso, updated_at=timestamp_iso
        )

    @staticmethod
    def _build_session_id(conversation_id: str, session_key: str) -> str:
        return (
            f"eval-{_safe_component(conversation_id)}-"
            f"{_safe_component(session_key)}"
        )

    def _memory_refs_for_session(
        self, *, scope_id: str, source_session_id: str
    ) -> List[Dict[str, Any]]:
        manager = self._get_manager()
        units = [
            unit
            for unit in manager.store.list_active(scope_id, limit=10000)
            if unit.source_session_id == source_session_id
        ]
        return [
            {
                "provider": "metaclaw",
                "type": "metaclaw_memory_unit",
                "memory_id": unit.memory_id,
                "scope_id": unit.scope_id,
                "source_session_id": unit.source_session_id,
                "memory_type": unit.memory_type.value,
                "content": unit.content,
                "summary": unit.summary,
                "created_at": unit.created_at,
                "updated_at": unit.updated_at,
                "source_turn_start": unit.source_turn_start,
                "source_turn_end": unit.source_turn_end,
                "entities": unit.entities,
                "topics": unit.topics,
                "importance": unit.importance,
                "confidence": unit.confidence,
            }
            for unit in units
        ]

    def _build_manifest_record(
        self,
        *,
        conversation: Conversation,
        session_key: str,
        view_name: str,
        scope_id: str,
        session_id: str,
        message_count: int,
        added_count: int,
        memory_refs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chunk_id = f"{conversation.conversation_id}:{session_key}:{view_name}"
        record = ImportManifestRecord(
            run_id=str(self.run_context.get("run_id", "")),
            system_id=str(self.run_context.get("system_id", "metaclaw")),
            conversation_id=conversation.conversation_id,
            view_id=view_name,
            chunk_id=chunk_id,
            source_unit_ids=[],
            write_request_summary={
                "adapter": "metaclaw",
                "session_key": session_key,
                "view_name": view_name,
                "scope_id": scope_id,
                "session_id": session_id,
                "message_count": message_count,
            },
            write_receipt={
                "provider_status": "written",
                "added_count": added_count,
                "memory_ref_count": len(memory_refs),
            },
            memory_refs=memory_refs,
            write_status="written",
            errors=[],
        )
        return dataclass_to_dict(record)

    def _export_active_memories(self) -> List[Dict[str, Any]]:
        manager = self._get_manager()
        rows: List[Dict[str, Any]] = []
        for scope_id in sorted(
            {
                scope
                for mapping in self._scope_map.values()
                for scope in mapping.values()
            }
        ):
            for unit in manager.store.list_active(scope_id, limit=10000):
                rows.append(asdict(unit))
                rows[-1]["memory_type"] = unit.memory_type.value
                rows[-1]["status"] = unit.status.value
        return rows

    @staticmethod
    def _unit_to_result(unit: Any, *, scope_id: str, rank: int) -> Dict[str, Any]:
        return {
            "content": unit.content,
            "score": float(unit.importance),
            "metadata": {
                "provider": "metaclaw",
                "rank": rank,
                "memory_id": unit.memory_id,
                "scope_id": scope_id,
                "memory_type": unit.memory_type.value,
                "summary": unit.summary,
                "source_session_id": unit.source_session_id,
                "source_turn_start": unit.source_turn_start,
                "source_turn_end": unit.source_turn_end,
                "created_at": unit.created_at,
                "updated_at": unit.updated_at,
                "entities": unit.entities,
                "topics": unit.topics,
                "importance": unit.importance,
                "confidence": unit.confidence,
            },
        }

    @staticmethod
    def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for result in results:
            memory_id = result.get("metadata", {}).get("memory_id")
            if memory_id in seen:
                continue
            seen.add(memory_id)
            deduped.append(result)
        return deduped

    def _format_search_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = ["## Retrieved MetaClaw Memories"]
        used_chars = len(lines[0])
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            text = str(result.get("content") or "").strip()
            if not text:
                continue
            block = (
                f"\n[{index}] type={metadata.get('memory_type')} "
                f"scope={metadata.get('scope_id')} "
                f"session={metadata.get('source_session_id')} "
                f"updated={metadata.get('updated_at')}\n"
                f"{text}"
            )
            if used_chars + len(block) > self.max_context_chars:
                break
            lines.append(block)
            used_chars += len(block)
        return "\n".join(lines).strip()

    def _render_units_for_prompt(self, units: List[Any]) -> str:
        if not units:
            return ""
        manager = self._get_manager()
        renderer = getattr(manager, "render_for_prompt", None)
        if callable(renderer):
            return str(renderer(units) or "")
        return self._format_search_context(
            [
                self._unit_to_result(unit, scope_id=getattr(unit, "scope_id", ""), rank=rank)
                for rank, unit in enumerate(units, start=1)
            ]
        )
