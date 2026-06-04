"""A-Mem adapter for the standalone evaluation framework."""

from __future__ import annotations

import json
import os
import pickle
import sys
import importlib.util
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.shared.subtlememory_prompts import (
    get_subtlememory_unified_answer_prompt,
)
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.core.readback import NormalizedStorageObject, StorageReadbackResult
from evaluation.src.utils.answer_cleaner import clean_answer_text
from evaluation.src.utils.config import load_yaml

from memory_layer.llm.llm_provider import LLMProvider


AMEM_MEMORY_CACHE_FILENAME = "amem_memory_cache.pkl"
AMEM_MEMORY_MANIFEST_FILENAME = "amem_memory_manifest.jsonl"


@dataclass
class _CachedMemory:
    memory_id: str
    conversation_id: str
    source_session_id: str
    source_unit_id: str
    speaker: str
    timestamp: str
    content: str
    metadata: Dict[str, Any]


class _KeywordMemoryNote:
    def __init__(
        self,
        *,
        memory_id: str,
        content: str,
        timestamp: str,
        context: str = "General",
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        links: Optional[List[int]] = None,
        category: Any = None,
    ) -> None:
        self.id = memory_id
        self.content = content
        self.timestamp = timestamp
        self.context = context
        self.keywords = keywords or []
        self.tags = tags or []
        self.links = links or []
        self.category = category


class _KeywordMemorySystem:
    """Small local fallback when A-Mem's sentence-transformer stack is unavailable."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.memories: Dict[str, _KeywordMemoryNote] = {}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        import re

        return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)}

    def add_note(self, content: str, time: str = None, **kwargs: Any) -> str:
        memory_id = str(kwargs.get("id") or f"amem-local-{len(self.memories) + 1}")
        keywords = list(kwargs.get("keywords") or list(self._tokens(content))[:12])
        note = _KeywordMemoryNote(
            memory_id=memory_id,
            content=content,
            timestamp=str(time or ""),
            context=str(kwargs.get("context") or "General"),
            keywords=keywords,
            tags=list(kwargs.get("tags") or []),
            links=list(kwargs.get("links") or []),
            category=kwargs.get("category"),
        )
        self.memories[memory_id] = note
        return memory_id

    def find_related_memories(self, query: str, k: int = 5) -> tuple[str, List[int]]:
        query_tokens = self._tokens(query)
        scored: List[tuple[float, int, _KeywordMemoryNote]] = []
        for idx, note in enumerate(self.memories.values()):
            content_tokens = self._tokens(note.content)
            overlap = len(query_tokens & content_tokens)
            score = float(overlap) / max(len(query_tokens), 1)
            scored.append((score, idx, note))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [(idx, note) for score, idx, note in scored[:k] if score > 0]
        if not selected:
            selected = [(idx, note) for _, idx, note in scored[:k]]
        text = "\n".join(
            f"memory index:{idx}\t talk start time:{note.timestamp}\t memory content: {note.content}"
            for idx, note in selected
        )
        return text, [idx for idx, _note in selected]


@register_adapter("amem")
class AMemAdapter(BaseAdapter):
    """Adapter that ingests SubtleMemory messages into A-Mem and exposes api/readback search."""

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(
            config.get("cache_dir") or "."
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.output_dir / AMEM_MEMORY_CACHE_FILENAME
        self.manifest_path = self.output_dir / AMEM_MEMORY_MANIFEST_FILENAME
        self._memory_system = None
        self._memory_system_cls = config.get("_test_memory_system_cls")
        self._runtime_backend = "uninitialized"
        self._runtime_backend_error = ""
        self._memory_cache: Dict[str, _CachedMemory] = {}
        self._import_manifest_records: List[Dict[str, Any]] = []
        self._native_memories: Dict[str, Any] = {}

        llm_config = config.get("llm", {}) or {}
        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0),
            max_tokens=llm_config.get("max_tokens", 16384),
        )
        evaluation_root = Path(__file__).resolve().parents[3]
        self._prompts = load_yaml(str(evaluation_root / "config" / "prompts.yaml"))
        self.num_workers = int(config.get("num_workers", 1))

    def set_run_context(self, run_context: Dict[str, Any]) -> None:
        super().set_run_context(run_context)
        self._memory_cache = {}
        self._import_manifest_records = []
        self._native_memories = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def add(self, conversations: List[Conversation], **kwargs: Any) -> Dict[str, Any]:
        del kwargs
        self._memory_system = self._create_memory_system()
        self._memory_cache = {}
        self._import_manifest_records = []
        self._native_memories = {}

        for conversation in conversations:
            for message_index, message in enumerate(conversation.messages):
                content = str(message.content or "").strip()
                if not content:
                    continue
                timestamp = self._format_timestamp(message.timestamp)
                source_session_id = str(
                    message.metadata.get("source_session_id")
                    or message.metadata.get("session")
                    or ""
                )
                source_unit_id = str(
                    message.metadata.get("source_unit_id")
                    or f"{conversation.conversation_id}:{message_index}"
                )
                speaker = str(message.sender_name or message.sender_id or "")
                note_content = self._format_message_for_amem_note(speaker, content)
                memory_id = str(
                    self._memory_system.add_note(
                        note_content,
                        time=timestamp,
                    )
                )
                note_metadata = self._amem_note_metadata(
                    self._get_memory_note(self._memory_system, memory_id)
                )
                cached = _CachedMemory(
                    memory_id=memory_id,
                    conversation_id=str(conversation.conversation_id),
                    source_session_id=source_session_id,
                    source_unit_id=source_unit_id,
                    speaker=speaker,
                    timestamp=timestamp,
                    content=note_content,
                    metadata={
                        "sender_id": message.sender_id,
                        "sender_name": message.sender_name,
                        "session_key": message.metadata.get("session"),
                        "source_case_id": message.metadata.get("source_case_id"),
                        "session_order": message.metadata.get("session_order"),
                        **note_metadata,
                    },
                )
                self._memory_cache[memory_id] = cached
                self._import_manifest_records.append(
                    self._build_manifest_record(cached)
                )

        self._persist_cache()
        return {
            "type": "local_memory",
            "system": "amem",
            "conversation_ids": [conversation.conversation_id for conversation in conversations],
            "memory_count": len(self._memory_cache),
            "cache_path": str(self.cache_path),
            "manifest_path": str(self.manifest_path),
            "embedding_backend": self._embedding_backend_label(),
            "embedding_config": self._embedding_metadata(),
            "import_manifest_records": self.get_import_manifest_records(),
        }

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs: Any
    ) -> SearchResult:
        del index
        started_at = perf_counter()
        self._ensure_cache_loaded()
        memory_system = self._ensure_memory_system()
        top_k = int(kwargs.get("top_k") or self.config.get("search", {}).get("top_k", 5))
        retrieval_query = await self._generate_retrieval_query(query, memory_system)
        selected_ids = self._amem_related_memory_ids(
            memory_system, retrieval_query, top_k
        )
        formatted_context, context_format = self._format_api_context(
            memory_system=memory_system,
            retrieval_query=retrieval_query,
            top_k=top_k,
            selected_ids=selected_ids,
        )
        if not selected_ids and context_format == "amem_raw_fallback":
            selected_ids = [
                memory_id
                for memory_id, memory in self._memory_cache.items()
                if memory.conversation_id == str(conversation_id)
            ][:top_k]

        results = [
            self._cached_memory_to_search_result(self._memory_cache[memory_id])
            for memory_id in selected_ids
            if memory_id in self._memory_cache
        ]

        return SearchResult(
            question_id=str(kwargs.get("question_id") or ""),
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "amem",
                "search_mode": "api",
                "top_k": top_k,
                "original_query": query,
                "retrieval_query": retrieval_query,
                "retrieval_query_source": "llm_keywords",
                "context_format": context_format,
                "formatted_context": formatted_context,
                "results": results,
                "embedding_backend": self._embedding_backend_label(),
                "embedding_config": self._embedding_metadata(),
                "amem_repo_path": str(self._amem_repo_path()),
            },
            retrieval_status="ok" if results else "empty",
            timing_ms=(perf_counter() - started_at) * 1000,
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
        **kwargs: Any,
    ) -> SearchResult:
        del index, conversation, import_manifest_records, kwargs
        session_ids = self._extract_session_ids(question_metadata or {})
        if not session_ids:
            readback = StorageReadbackResult(
                status="unsupported",
                checked_session_ids=[],
                missing_session_ids=[],
                objects=[],
                metadata={
                    "provider": "amem",
                    "question_id": question_id,
                    "reason": "missing question session_ids",
                },
                errors=[],
            )
            return SearchResult(
                question_id=question_id or "",
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "system": "amem",
                    "search_mode": "readback",
                    "session_ids": [],
                    "formatted_context": "",
                    "results": [],
                    "readback": readback.to_dict(),
                },
                retrieval_status="unsupported",
            )

        readback = await self.get_storage_readback(
            session_ids=session_ids,
            question_id=question_id,
            context={
                "question_id": question_id,
                "conversation_id": conversation_id,
                "session_ids": session_ids,
            },
        )
        results, formatted_context = self._build_readback_search_payload(
            readback,
            session_ids=session_ids,
        )
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "amem",
                "search_mode": "readback",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "results": results,
                "readback": readback.to_dict(),
                "embedding_backend": self._embedding_backend_label(),
                "embedding_config": self._embedding_metadata(),
            },
            retrieval_status=self._readback_search_status(readback, bool(results)),
        )

    async def get_storage_readback(
        self,
        *,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        question_id: Optional[str] = None,
        evidence_texts: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> StorageReadbackResult:
        del user_id, run_id, evidence_texts, context, kwargs
        self._ensure_cache_loaded()
        requested_sessions = self._dedupe_nonempty(session_ids or [])
        session_set = set(requested_sessions)
        objects: List[NormalizedStorageObject] = []
        seen_sessions = set()

        for memory in self._memory_cache.values():
            if requested_sessions and memory.source_session_id not in session_set:
                continue
            if memory.source_session_id:
                seen_sessions.add(memory.source_session_id)
            readback_content = self._format_cached_memory_as_amem_note(memory)
            objects.append(
                NormalizedStorageObject(
                    session_id=memory.source_session_id,
                    kind="memory",
                    id=memory.memory_id,
                    content=readback_content,
                    metadata={
                        "provider": "amem",
                        "conversation_id": memory.conversation_id,
                        "source_unit_id": memory.source_unit_id,
                        "source_session_id": memory.source_session_id,
                        "speaker": memory.speaker,
                        "timestamp": memory.timestamp,
                        **(memory.metadata or {}),
                    },
                    raw={
                        "memory_id": memory.memory_id,
                        "content": readback_content,
                        "original_content": memory.content,
                    },
                )
            )

        missing = [
            session_id for session_id in requested_sessions if session_id not in seen_sessions
        ]
        status = "ok"
        return StorageReadbackResult(
            status=status,
            checked_session_ids=requested_sessions,
            missing_session_ids=missing,
            objects=objects,
            metadata={
                "provider": "amem",
                "question_id": question_id,
                "session_ids": requested_sessions,
                "cache_path": str(self.cache_path),
                "manifest_path": str(self.manifest_path),
            },
            errors=[],
        )

    async def finalize_imports(
        self,
        import_manifest_rows: List[Dict[str, Any]],
        *,
        add_result: Any = None,
        budget_seconds: int = 0,
        poll_interval_seconds: float | None = None,
        dataset: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del add_result, budget_seconds, poll_interval_seconds, dataset, kwargs
        rows = list(import_manifest_rows or self.get_import_manifest_records())
        for row in rows:
            row["write_status"] = "completed"
            receipt = dict(row.get("write_receipt") or {})
            receipt["provider_status"] = "completed"
            row["write_receipt"] = receipt
        self._import_manifest_records = rows
        return {
            "import_manifest_records": rows,
            "ready": True,
            "status": "finalized",
            "provider_status_counts": {"completed": len(rows)},
            "updated_rows": len(rows),
            "missing_memory_refs": [
                str(row.get("chunk_id") or "") for row in rows if not row.get("memory_refs")
            ],
            "warnings": [],
            "finalize_budget_exhausted": False,
            "finalized_at": None,
        }

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any] | None:
        del conversations
        self.output_dir = Path(output_dir)
        self.cache_path = self.output_dir / AMEM_MEMORY_CACHE_FILENAME
        self.manifest_path = self.output_dir / AMEM_MEMORY_MANIFEST_FILENAME
        self._ensure_cache_loaded()
        return {
            "system": "amem",
            "cache_path": str(self.cache_path),
            "manifest_path": str(self.manifest_path),
            "memory_count": len(self._memory_cache),
            "embedding_backend": self._embedding_backend_label(),
            "embedding_config": self._embedding_metadata(),
        }

    def get_import_manifest_records(self) -> List[Dict[str, Any]]:
        if not self._import_manifest_records and self.manifest_path.exists():
            self._import_manifest_records = self._read_manifest_file(self.manifest_path)
        if not self._import_manifest_records:
            standard_manifest_path = self.output_dir / "import_manifest.jsonl"
            if standard_manifest_path.exists():
                self._import_manifest_records = self._read_manifest_file(
                    standard_manifest_path
                )
        return list(self._import_manifest_records)

    async def answer(self, query: str, context: str, **kwargs: Any) -> str:
        del kwargs
        prompt = self._get_answer_prompt().format(context=context, question=query)
        max_retries = int(self.config.get("answer", {}).get("max_retries", 3))
        for attempt in range(max_retries):
            try:
                answer = await self.llm_provider.generate(prompt=prompt, temperature=0)
                answer = clean_answer_text(answer)
                if answer:
                    return answer
            except Exception:
                if attempt == max_retries - 1:
                    raise
        return ""

    def render_answer_prompt(self, query: str, context: str, **kwargs: Any) -> str:
        del kwargs
        return self._get_answer_prompt().format(context=context, question=query)

    def _get_answer_prompt(self) -> str:
        return get_subtlememory_unified_answer_prompt(
            prompts=self._prompts,
            config=self.config,
        )

    def _create_memory_system(self) -> Any:
        cls = self._memory_system_cls or self._load_amem_memory_system_cls()
        kwargs = {
            "model_name": self._amem_retriever_model(),
            "llm_backend": self.config.get("llm_backend", "openai"),
            "llm_model": self.config.get("llm_model", "gpt-4o-mini"),
            "api_key": self.config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            "api_base": self.config.get("api_base") or os.getenv("OPENAI_BASE_URL"),
            "evo_threshold": int(self.config.get("evo_threshold", 100)),
            "check_connection": bool(self.config.get("check_connection", False)),
        }
        try:
            instance = cls(**kwargs)
            self._configure_native_openai_client(instance)
            self._runtime_backend = "a-mem-robust-local-retriever"
            self._runtime_backend_error = ""
            return instance
        except Exception as exc:
            if self.config.get("allow_keyword_fallback", True):
                self._runtime_backend = "keyword-fallback"
                self._runtime_backend_error = f"{type(exc).__name__}: {exc}"
                return _KeywordMemorySystem(**kwargs)
            raise

    def _configure_native_openai_client(self, memory_system: Any) -> None:
        options: Dict[str, Any] = {}
        timeout_seconds = self._optional_float_config(
            "openai_timeout_seconds", "AMEM_OPENAI_TIMEOUT_SECONDS"
        )
        if timeout_seconds is not None and timeout_seconds > 0:
            options["timeout"] = timeout_seconds

        max_retries = self._optional_int_config(
            "openai_max_retries", "AMEM_OPENAI_MAX_RETRIES"
        )
        if max_retries is not None and max_retries >= 0:
            options["max_retries"] = min(max_retries, 5)

        if not options:
            return

        llm = getattr(getattr(memory_system, "llm_controller", None), "llm", None)
        client = getattr(llm, "client", None)
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            llm.client = with_options(**options)

    def _optional_float_config(self, key: str, env_key: str) -> Optional[float]:
        raw = self.config.get(key)
        if raw in (None, ""):
            raw = os.getenv(env_key)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _optional_int_config(self, key: str, env_key: str) -> Optional[int]:
        raw = self.config.get(key)
        if raw in (None, ""):
            raw = os.getenv(env_key)
        if raw in (None, ""):
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _load_amem_memory_system_cls(self) -> Any:
        repo_path = self._amem_repo_path()
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
        try:
            return self._load_amem_robust_class_from_path(repo_path)
        except Exception:
            if self.config.get("allow_keyword_fallback", True):
                return _KeywordMemorySystem
            raise

    @staticmethod
    def _load_amem_robust_class_from_path(repo_path: Path) -> Any:
        memory_layer_path = repo_path / "memory_layer.py"
        robust_path = repo_path / "memory_layer_robust.py"
        if not memory_layer_path.exists() or not robust_path.exists():
            raise ImportError(f"A-Mem memory layer files not found under {repo_path}")

        previous_memory_layer = sys.modules.get("memory_layer")
        previous_robust = sys.modules.get("memory_layer_robust")
        try:
            memory_spec = importlib.util.spec_from_file_location(
                "memory_layer", memory_layer_path
            )
            if memory_spec is None or memory_spec.loader is None:
                raise ImportError(f"Cannot load {memory_layer_path}")
            memory_module = importlib.util.module_from_spec(memory_spec)
            sys.modules["memory_layer"] = memory_module
            memory_spec.loader.exec_module(memory_module)

            robust_spec = importlib.util.spec_from_file_location(
                "memory_layer_robust", robust_path
            )
            if robust_spec is None or robust_spec.loader is None:
                raise ImportError(f"Cannot load {robust_path}")
            robust_module = importlib.util.module_from_spec(robust_spec)
            sys.modules["memory_layer_robust"] = robust_module
            robust_spec.loader.exec_module(robust_module)
            return robust_module.RobustAgenticMemorySystem
        finally:
            if previous_memory_layer is None:
                sys.modules.pop("memory_layer", None)
            else:
                sys.modules["memory_layer"] = previous_memory_layer
            if previous_robust is None:
                sys.modules.pop("memory_layer_robust", None)
            else:
                sys.modules["memory_layer_robust"] = previous_robust

    def _amem_repo_path(self) -> Path:
        raw = self.config.get("amem_repo_path") or os.getenv("AMEM_REPO_PATH")
        if not raw:
            raise ValueError(
                "A-Mem repository path is required. Set AMEM_REPO_PATH in .env "
                "or configure amem_repo_path in the system YAML."
            )
        return Path(str(raw)).expanduser().resolve()

    def _amem_retriever_model(self) -> str:
        return str(
            self.config.get("amem_retriever_model")
            or self.config.get("embedding_model")
            or "all-MiniLM-L6-v2"
        )

    async def _generate_retrieval_query(self, question: str, memory_system: Any) -> str:
        del memory_system
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
        response = await self.llm_provider.generate(prompt=prompt, temperature=0)
        return self._parse_keywords_response(response)

    @staticmethod
    def _parse_keywords_response(response: str) -> str:
        try:
            cleaned = AMemAdapter._strip_markdown_fences(str(response or ""))
            data = json.loads(cleaned)
            if isinstance(data, dict) and "keywords" in data:
                return str(data["keywords"])
        except (json.JSONDecodeError, ValueError):
            pass
        return str(response or "").strip()

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    def _ensure_memory_system(self) -> Any:
        if self._memory_system is not None:
            return self._memory_system
        memory_system = self._create_memory_system()
        if self._restore_native_memory_system(memory_system):
            self._memory_system = memory_system
            return memory_system
        for memory in self._memory_cache.values():
            note_kwargs = self._restored_note_kwargs(memory)
            try:
                returned_id = str(
                    memory_system.add_note(
                        memory.content,
                        time=memory.timestamp,
                        id=memory.memory_id,
                        **note_kwargs,
                    )
                )
                if returned_id != memory.memory_id and returned_id in getattr(
                    memory_system, "memories", {}
                ):
                    memories = getattr(memory_system, "memories", {})
                    memories[memory.memory_id] = memories.pop(returned_id)
                    memories[memory.memory_id].id = memory.memory_id
            except TypeError:
                memory_system.add_note(memory.content, time=memory.timestamp)
        self._memory_system = memory_system
        return memory_system

    def _amem_related_memory_ids(self, memory_system: Any, query: str, top_k: int) -> List[str]:
        try:
            _memory_text, indices = memory_system.find_related_memories(query, k=top_k)
        except Exception:
            return []
        all_memory_ids = list(getattr(memory_system, "memories", {}).keys())
        selected: List[str] = []
        for index in [] if indices is None else indices:
            try:
                memory_id = all_memory_ids[int(index)]
            except (IndexError, TypeError, ValueError):
                continue
            if memory_id in self._memory_cache and memory_id not in selected:
                selected.append(memory_id)
        return selected

    def _format_api_context(
        self,
        *,
        memory_system: Any,
        retrieval_query: str,
        top_k: int,
        selected_ids: Optional[List[str]] = None,
    ) -> tuple[str, str]:
        if isinstance(memory_system, _KeywordMemorySystem):
            results = [
                self._cached_memory_to_search_result(memory)
                for memory in list(self._memory_cache.values())[:top_k]
            ]
            return self._format_amem_raw_fallback_context(results), "amem_raw_fallback"

        raw_retriever = getattr(memory_system, "find_related_memories_raw", None)
        if callable(raw_retriever):
            raw_context = self._format_native_raw_context(
                memory_system=memory_system,
                selected_ids=selected_ids or [],
                top_k=top_k,
            )
            if not raw_context:
                raw_context = str(raw_retriever(retrieval_query, k=top_k) or "").strip()
            if raw_context:
                return raw_context, "amem_raw"

        results = [
            self._cached_memory_to_search_result(memory)
            for memory in list(self._memory_cache.values())[:top_k]
        ]
        return self._format_amem_raw_fallback_context(results), "amem_raw_fallback"

    @staticmethod
    def _format_message_for_amem_note(speaker: str, content: str) -> str:
        return "Speaker " + str(speaker or "") + "says : " + str(content or "")

    @staticmethod
    def _format_amem_raw_fallback_context(results: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for result in results:
            content = str(result.get("content") or "").strip()
            if not content:
                continue
            metadata = result.get("metadata") or {}
            timestamp = str(metadata.get("timestamp") or "")
            context = str(metadata.get("amem_context") or "")
            keywords = metadata.get("amem_keywords") or []
            tags = metadata.get("amem_tags") or []
            parts.append(
                "talk start time:"
                + timestamp
                + "memory content: "
                + content
                + "memory context: "
                + context
                + "memory keywords: "
                + str(keywords)
                + "memory tags: "
                + str(tags)
            )
        return "\n".join(parts)

    @staticmethod
    def _format_cached_memory_as_amem_note(memory: _CachedMemory) -> str:
        metadata = memory.metadata or {}
        return (
            "talk start time:"
            + str(memory.timestamp or "")
            + "memory content: "
            + str(memory.content or "")
            + "memory context: "
            + str(metadata.get("amem_context") or "")
            + "memory keywords: "
            + str(metadata.get("amem_keywords") or [])
            + "memory tags: "
            + str(metadata.get("amem_tags") or [])
        )

    def _persist_cache(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as handle:
            pickle.dump(
                {
                    "memory_cache": {
                        memory_id: memory.__dict__
                        for memory_id, memory in self._memory_cache.items()
                    },
                    "native_memories": self._copy_native_memories(),
                    "embedding_backend": self._embedding_backend_label(),
                    "embedding_config": self._embedding_metadata(),
                },
                handle,
            )
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            for row in self._import_manifest_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _ensure_cache_loaded(self) -> None:
        if self._memory_cache:
            return
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as handle:
                payload = pickle.load(handle)
            raw_cache = payload.get("memory_cache", {}) if isinstance(payload, dict) else {}
            self._memory_cache = {
                str(memory_id): _CachedMemory(**memory)
                for memory_id, memory in raw_cache.items()
            }
            native_memories = payload.get("native_memories", {}) if isinstance(payload, dict) else {}
            self._native_memories = dict(native_memories) if isinstance(native_memories, dict) else {}
        if not self._import_manifest_records and self.manifest_path.exists():
            self._import_manifest_records = self._read_manifest_file(self.manifest_path)
        if not self._import_manifest_records:
            standard_manifest_path = self.output_dir / "import_manifest.jsonl"
            if standard_manifest_path.exists():
                self._import_manifest_records = self._read_manifest_file(
                    standard_manifest_path
                )
        if not self._memory_cache and self._import_manifest_records:
            self._memory_cache = self._memory_cache_from_manifest(
                self._import_manifest_records
            )

    @staticmethod
    def _read_manifest_file(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def _build_manifest_record(self, memory: _CachedMemory) -> Dict[str, Any]:
        run_context = getattr(self, "run_context", {}) or {}
        run_id = str(run_context.get("run_id") or "run")
        system_id = str(run_context.get("system_id") or self.config.get("name") or "amem")
        chunk_id = f"{memory.conversation_id}:{memory.source_unit_id}"
        namespace_scope = {
            "provider": "amem",
            "conversation_id": memory.conversation_id,
            "source_session_id": memory.source_session_id,
            "namespace_id": memory.conversation_id,
            "view_id": "shared",
            "dataset_id": run_context.get("dataset_id"),
        }
        memory_ref = {
            "provider": "amem",
            "memory_id": memory.memory_id,
            "conversation_id": memory.conversation_id,
            "source_session_id": memory.source_session_id,
            "session_id": memory.source_session_id,
            "source_unit_id": memory.source_unit_id,
            "timestamp": memory.timestamp,
            "speaker": memory.speaker,
            "content": memory.content,
            "metadata": memory.metadata or {},
        }
        return {
            "run_id": run_id,
            "system_id": system_id,
            "conversation_id": memory.conversation_id,
            "view_id": "shared",
            "chunk_id": chunk_id,
            "source_session_id": memory.source_session_id,
            "source_unit_ids": [memory.source_unit_id] if memory.source_unit_id else [],
            "write_request_summary": {
                "message_count": 1,
                "content_chars": len(memory.content),
                "speaker": memory.speaker,
            },
            "write_receipt": {
                "system_id": system_id,
                "namespace_scope": namespace_scope,
                "chunk_id": chunk_id,
                "source_unit_ids": [memory.source_unit_id] if memory.source_unit_id else [],
                "provider_receipt": {
                    "memory_id": memory.memory_id,
                    "embedding_backend": self._embedding_backend_label(),
                    "embedding_config": self._embedding_metadata(),
                },
                "provider_status": "completed",
                "memory_refs": [memory_ref],
                "errors": [],
            },
            "memory_refs": [memory_ref],
            "write_status": "completed",
            "errors": [],
        }

    def _cached_memory_to_search_result(self, memory: _CachedMemory) -> Dict[str, Any]:
        return {
            "content": memory.content,
            "score": 1.0,
            "metadata": {
                "provider": "amem",
                "memory_id": memory.memory_id,
                "conversation_id": memory.conversation_id,
                "source_session_id": memory.source_session_id,
                "session_id": memory.source_session_id,
                "source_unit_id": memory.source_unit_id,
                "speaker": memory.speaker,
                "timestamp": memory.timestamp,
                **(memory.metadata or {}),
            },
        }

    @staticmethod
    def _get_memory_note(memory_system: Any, memory_id: str) -> Any:
        memories = getattr(memory_system, "memories", {}) or {}
        if isinstance(memories, dict):
            return memories.get(memory_id)
        return None

    @staticmethod
    def _amem_note_metadata(note: Any) -> Dict[str, Any]:
        if note is None:
            return {}
        metadata: Dict[str, Any] = {}
        if hasattr(note, "context"):
            metadata["amem_context"] = str(getattr(note, "context") or "")
        if hasattr(note, "keywords"):
            metadata["amem_keywords"] = list(getattr(note, "keywords") or [])
        if hasattr(note, "tags"):
            metadata["amem_tags"] = list(getattr(note, "tags") or [])
        if hasattr(note, "links"):
            links = getattr(note, "links") or []
            if isinstance(links, dict):
                links = list(links.values())
            metadata["amem_links"] = list(links)
        if hasattr(note, "category"):
            metadata["amem_category"] = getattr(note, "category")
        return metadata

    def _copy_native_memories(self) -> Dict[str, Any]:
        memories = getattr(self._memory_system, "memories", {}) or {}
        if isinstance(memories, dict):
            return {
                str(memory_id): self._native_memory_state(note)
                for memory_id, note in memories.items()
            }
        return {}

    def _restore_native_memory_system(self, memory_system: Any) -> bool:
        if not self._native_memories:
            return False
        memories = getattr(memory_system, "memories", None)
        if not isinstance(memories, dict):
            return False
        memory_system.memories = {
            str(memory_id): self._native_note_from_state(memory_id, state)
            for memory_id, state in self._native_memories.items()
        }
        retriever = getattr(memory_system, "retriever", None)
        loader = getattr(retriever, "load_from_local_memory", None)
        if callable(loader):
            try:
                memory_system.retriever = loader(
                    memory_system.memories,
                    self._amem_retriever_model(),
                )
            except TypeError:
                pass
        elif retriever is not None:
            add_documents = getattr(retriever, "add_documents", None)
            if callable(add_documents):
                documents = [
                    self._amem_retriever_document(note)
                    for note in memory_system.memories.values()
                ]
                if documents:
                    add_documents(documents)
        return True

    @staticmethod
    def _native_memory_state(note: Any) -> Dict[str, Any]:
        return {
            "content": str(getattr(note, "content", "") or ""),
            "id": str(getattr(note, "id", "") or ""),
            "keywords": list(getattr(note, "keywords", []) or []),
            "links": list(getattr(note, "links", []) or []),
            "importance_score": getattr(note, "importance_score", 1.0),
            "retrieval_count": getattr(note, "retrieval_count", 0),
            "timestamp": str(getattr(note, "timestamp", "") or ""),
            "last_accessed": str(getattr(note, "last_accessed", "") or ""),
            "context": str(getattr(note, "context", "") or ""),
            "evolution_history": list(getattr(note, "evolution_history", []) or []),
            "category": getattr(note, "category", None),
            "tags": list(getattr(note, "tags", []) or []),
        }

    @staticmethod
    def _native_note_from_state(memory_id: str, state: Any) -> Any:
        if not isinstance(state, dict):
            return state
        note_id = str(state.get("id") or memory_id)
        return SimpleNamespace(
            content=str(state.get("content") or ""),
            id=note_id,
            keywords=list(state.get("keywords") or []),
            links=list(state.get("links") or []),
            importance_score=state.get("importance_score", 1.0),
            retrieval_count=state.get("retrieval_count", 0),
            timestamp=str(state.get("timestamp") or ""),
            last_accessed=str(state.get("last_accessed") or ""),
            context=str(state.get("context") or "General"),
            evolution_history=list(state.get("evolution_history") or []),
            category=state.get("category"),
            tags=list(state.get("tags") or []),
        )

    @staticmethod
    def _amem_retriever_document(note: Any) -> str:
        content = str(getattr(note, "content", "") or "")
        context = str(getattr(note, "context", "") or "")
        keywords = " ".join(str(value) for value in getattr(note, "keywords", []) or [])
        tags = " ".join(str(value) for value in getattr(note, "tags", []) or [])
        return f"{content} , {context} {keywords} {tags}"

    def _format_native_raw_context(
        self,
        *,
        memory_system: Any,
        selected_ids: List[str],
        top_k: int,
    ) -> str:
        memories = getattr(memory_system, "memories", {}) or {}
        if not isinstance(memories, dict):
            return ""
        notes = list(memories.values())
        chunks: List[str] = []
        for memory_id in selected_ids:
            note = memories.get(memory_id)
            if note is None:
                continue
            chunks.append(self._format_amem_note_for_context(note))
            neighbor_count = 0
            for neighbor in getattr(note, "links", []) or []:
                try:
                    neighbor_note = notes[int(neighbor)]
                except (IndexError, TypeError, ValueError):
                    continue
                chunks.append(self._format_amem_note_for_context(neighbor_note))
                if neighbor_count >= top_k:
                    break
                neighbor_count += 1
        return "\n".join(chunks)

    @staticmethod
    def _format_amem_note_for_context(note: Any) -> str:
        return (
            "talk start time:"
            + str(getattr(note, "timestamp", "") or "")
            + "memory content: "
            + str(getattr(note, "content", "") or "")
            + "memory context: "
            + str(getattr(note, "context", "") or "")
            + "memory keywords: "
            + str(getattr(note, "keywords", []) or [])
            + "memory tags: "
            + str(getattr(note, "tags", []) or [])
        )

    def _restored_note_kwargs(self, memory: _CachedMemory) -> Dict[str, Any]:
        metadata = memory.metadata or {}
        kwargs: Dict[str, Any] = {}
        if "amem_context" in metadata:
            kwargs["context"] = str(metadata.get("amem_context") or "")
        if "amem_keywords" in metadata:
            kwargs["keywords"] = list(metadata.get("amem_keywords") or [])
        if "amem_tags" in metadata:
            kwargs["tags"] = list(metadata.get("amem_tags") or [])
        if "amem_links" in metadata:
            kwargs["links"] = list(metadata.get("amem_links") or [])
        if "amem_category" in metadata:
            kwargs["category"] = metadata.get("amem_category")
        return kwargs

    @staticmethod
    def _memory_cache_from_manifest(
        rows: List[Dict[str, Any]]
    ) -> Dict[str, _CachedMemory]:
        cache: Dict[str, _CachedMemory] = {}
        for row in rows:
            refs = row.get("memory_refs") or []
            if not isinstance(refs, list):
                continue
            for ref in refs:
                if not isinstance(ref, dict) or ref.get("provider") != "amem":
                    continue
                memory_id = str(ref.get("memory_id") or "").strip()
                content = str(ref.get("content") or "").strip()
                if not memory_id or not content:
                    continue
                cache[memory_id] = _CachedMemory(
                    memory_id=memory_id,
                    conversation_id=str(
                        ref.get("conversation_id") or row.get("conversation_id") or ""
                    ),
                    source_session_id=str(
                        ref.get("source_session_id")
                        or ref.get("session_id")
                        or row.get("source_session_id")
                        or ""
                    ),
                    source_unit_id=str(
                        ref.get("source_unit_id")
                        or (row.get("source_unit_ids") or [""])[0]
                        or ""
                    ),
                    speaker=str(ref.get("speaker") or ""),
                    timestamp=str(ref.get("timestamp") or ""),
                    content=content,
                    metadata=dict(ref.get("metadata") or {}),
                )
        return cache

    @staticmethod
    def _build_readback_search_payload(
        readback: StorageReadbackResult, *, session_ids: List[str]
    ) -> tuple[List[Dict[str, Any]], str]:
        buckets: Dict[str, List[NormalizedStorageObject]] = {
            session_id: [] for session_id in session_ids
        }
        extras: List[NormalizedStorageObject] = []
        for obj in readback.objects:
            if not isinstance(obj, NormalizedStorageObject):
                continue
            if obj.session_id in buckets:
                buckets[obj.session_id].append(obj)
            else:
                extras.append(obj)

        ordered_objects: List[NormalizedStorageObject] = []
        for session_id in session_ids:
            ordered_objects.extend(buckets.get(session_id, []))
        ordered_objects.extend(extras)

        results: List[Dict[str, Any]] = []
        for obj in ordered_objects:
            content = str(obj.content or "").strip()
            if not content:
                continue
            results.append(
                {
                    "content": content,
                    "score": 1.0,
                    "metadata": {
                        "provider": "amem",
                        "memory_id": obj.id,
                        "session_id": obj.session_id,
                        "source_session_id": obj.session_id,
                        "kind": obj.kind,
                        "provider_metadata": obj.metadata or {},
                        **(obj.metadata or {}),
                    },
                }
            )
        return results, AMemAdapter._format_readback_context(results)

    @staticmethod
    def _format_readback_context(results: List[Dict[str, Any]]) -> str:
        return "\n".join(
            str(result.get("content") or "").strip()
            for result in results
            if str(result.get("content") or "").strip()
        )

    @staticmethod
    def _format_search_context(results: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"{idx}. {str(result.get('content') or '').strip()}"
            for idx, result in enumerate(results, start=1)
            if str(result.get("content") or "").strip()
        )

    @staticmethod
    def _readback_search_status(readback: StorageReadbackResult, has_results: bool) -> str:
        if readback.status in {"unsupported", "error", "rate_limited", "timeout"}:
            return readback.status
        return "ok" if has_results else "empty"

    @staticmethod
    def _extract_session_ids(metadata: Dict[str, Any]) -> List[str]:
        raw = (
            metadata.get("session_ids")
            or metadata.get("source_session_ids")
            or metadata.get("source_session_id")
        )
        if isinstance(raw, (str, int, float)):
            values: Iterable[Any] = [raw]
        else:
            values = raw or []
        return AMemAdapter._dedupe_nonempty(values)

    @staticmethod
    def _dedupe_nonempty(values: Iterable[Any]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @staticmethod
    def _format_timestamp(value: Any) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        if value:
            return str(value)
        return ""

    def _embedding_backend_label(self) -> str:
        return str(
            self.config.get("embedding_backend")
            or "a-mem-local-retriever"
        )

    def _embedding_metadata(self) -> Dict[str, Any]:
        return {
            "configured_embedding_model": self.config.get(
                "configured_embedding_model", "text-embedding-3-small"
            ),
            "amem_retriever_model": self._amem_retriever_model(),
            "embedding_backend": self._embedding_backend_label(),
            "runtime_backend": self._runtime_backend,
            "runtime_backend_error": self._runtime_backend_error,
        }
