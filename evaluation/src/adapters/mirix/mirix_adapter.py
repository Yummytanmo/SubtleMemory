"""MIRIX native-agent adapter for SubtleMemory-only evaluation."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.mirix.native_backend import MirixNativeBackend
from evaluation.src.core.data_models import Conversation, Message, SearchResult
from evaluation.src.core.readback import NormalizedStorageObject, StorageReadbackResult
from evaluation.src.run_artifacts.models import ImportManifestRecord, dataclass_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MIRIX_ROOT = PROJECT_ROOT / "third_party" / "mirix"
DEFAULT_MIRIX_CONFIG_PATH = Path("configs/mirix_gpt4o-mini.yaml")


def _resolve_repo_path(value: Any, *, default: Path) -> Path:
    path = Path(str(value or default))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_mirix_config_path(value: Any, *, mirix_root: Path) -> Path:
    path = Path(str(value or DEFAULT_MIRIX_CONFIG_PATH))
    if not path.is_absolute():
        path = mirix_root / path
    return path.resolve()


def _safe_component(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text or fallback


def _session_sort_key(session_key: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", str(session_key))
    if match:
        return int(match.group(1)), str(session_key)
    return 10**9, str(session_key)


def _iso_timestamp(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso_timestamp(value)
    return str(value)


@dataclass
class _RuntimePaths:
    runtime_path: Path
    agent_state_path: Path
    sqlite_path: Path


@dataclass
class _StoredMemoryRow:
    id: str
    kind: str
    content: str
    metadata: Dict[str, Any]
    tree_path: List[str]
    created_at: str = ""


@register_adapter("mirix")
class MirixAdapter(BaseAdapter):
    """MIRIX native ingestion with MIRIX-owned answering."""

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        mirix_cfg = config.get("mirix", {})
        self.mirix_root = _resolve_repo_path(
            mirix_cfg.get("root"), default=DEFAULT_MIRIX_ROOT
        )
        self.config_path = _resolve_mirix_config_path(
            mirix_cfg.get("config_path"), mirix_root=self.mirix_root
        )

        run_root = Path(str(mirix_cfg.get("run_root") or "mirix_runtime"))
        if not run_root.is_absolute():
            run_root = self.output_dir / run_root
        self.run_root = run_root.resolve()
        self.runtime_dir_template = str(
            mirix_cfg.get("runtime_dir_template") or "{conversation_id}"
        )
        self.agent_state_dirname = str(
            mirix_cfg.get("agent_state_dirname") or "agent_state"
        )
        self.sqlite_filename = str(mirix_cfg.get("sqlite_filename") or "sqlite.db")
        self.supported_datasets = {
            str(value).strip()
            for value in config.get("supported_datasets", ["subtlememory"])
            if str(value).strip()
        }
        self.num_workers = int(config.get("num_workers", 1))

        self._runtime_paths: Dict[str, _RuntimePaths] = {}
        self._native_backends: Dict[str, MirixNativeBackend] = {}
        self._conversations: Dict[str, Conversation] = {}
        self._import_manifest_records: List[Dict[str, Any]] = []
        self._conversation_session_cache: Dict[str, List[str]] = {}
        self._answer_trace_payloads: Dict[str, Dict[str, Any]] = {}

        self._validate_supported_dataset()

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        del kwargs
        self._conversations = {
            conversation.conversation_id: conversation for conversation in conversations
        }
        self._runtime_paths = {}
        self._native_backends = {}
        self._import_manifest_records = []
        self._conversation_session_cache = {}
        self._answer_trace_payloads = {}

        for conversation in conversations:
            runtime = self._runtime_for(conversation.conversation_id)
            self._reset_runtime(runtime)
            sessions = self._build_native_sessions(conversation)
            self._conversation_session_cache[conversation.conversation_id] = [
                str(session["source_session_id"]) for session in sessions
            ]
            backend = self._backend_for(conversation.conversation_id)
            ingest_result = backend.ingest_conversation(
                conversation_id=conversation.conversation_id,
                sessions=sessions,
            )
            runtime = self._runtime_from_backend_result(
                conversation.conversation_id, runtime, ingest_result
            )
            result_sessions = self._merge_session_results(
                sessions, ingest_result.get("sessions") or []
            )
            self._import_manifest_records.extend(
                self._build_manifest_rows(
                    conversation=conversation,
                    runtime=runtime,
                    sessions=result_sessions,
                    ingest_result=ingest_result,
                )
            )

        return {
            "type": "mirix_native_runtime",
            "conversation_ids": list(self._conversations.keys()),
            "conversation_count": len(self._conversations),
            "runtime_root": str(self.run_root),
            "import_manifest_records": self.get_import_manifest_records(),
        }

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any]:
        del output_dir
        self._conversations = {
            conversation.conversation_id: conversation for conversation in conversations
        }
        for conversation in conversations:
            self._runtime_for(conversation.conversation_id)
        return {
            "type": "mirix_native_runtime",
            "conversation_ids": list(self._conversations.keys()),
            "conversation_count": len(self._conversations),
            "runtime_root": str(self.run_root),
        }

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        del index
        started_at = time.perf_counter()
        question_id = str(kwargs.get("question_id") or "")
        qa_session_ids = self._extract_session_ids(kwargs.get("question_metadata") or {})
        runtime = self._runtime_from_manifest(
            conversation_id,
            list(kwargs.get("import_manifest_records") or []),
        )
        if runtime is None:
            runtime = self._runtime_for(conversation_id)
        session_ids = self._conversation_session_ids(conversation_id)

        return SearchResult(
            question_id=question_id,
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={
                "system": "mirix",
                "search_mode": "api",
                "search_backend": "mirix_agent_main_loop",
                "search_scope": "conversation",
                "session_ids": session_ids,
                "qa_session_ids": qa_session_ids,
                "native_agent_answer_deferred": True,
                "formatted_context": "",
                "runtime_path": str(runtime.runtime_path),
                "agent_state_path": str(runtime.agent_state_path),
                "sqlite_path": str(runtime.sqlite_path),
            },
            retrieval_status="ok",
            timing_ms=(time.perf_counter() - started_at) * 1000,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        question_id = str(kwargs.get("question_id") or "")
        conversation_id = str(kwargs.get("conversation_id") or "").strip()
        if not conversation_id:
            raise ValueError("mirix answer requires conversation_id")

        runtime = self._runtime_for_answer(conversation_id, kwargs)
        readback_context = None
        if str(context or "").strip():
            readback_context = {
                "formatted_context": str(context or ""),
                "results": list(kwargs.get("search_results") or []),
            }
        backend = self._backend_for(conversation_id)
        payload = await asyncio.to_thread(
            backend.answer_question,
            question=query,
            readback_context=readback_context,
        )
        answer = str(payload.get("native_agent_answer") or "").strip()
        trace = {
            "answer_owner": "mirix",
            "answer_prompt": None,
            "answer_prompt_available": False,
            "native_agent_answer": answer,
            "native_agent_answer_status": payload.get("status", "ok"),
            "native_agent_answer_deferred": True,
            "readback_context_char_count": len(str(context or "")),
            "runtime_path": str(payload.get("runtime_path") or runtime.runtime_path),
            "agent_state_path": str(
                payload.get("agent_state_path") or runtime.agent_state_path
            ),
            "sqlite_path": str(payload.get("sqlite_path") or runtime.sqlite_path),
            "native_output": payload.get("native_output", {}),
        }
        if question_id:
            self._answer_trace_payloads[question_id] = trace
        return answer

    async def finalize_imports(
        self,
        import_manifest_rows: List[Dict[str, Any]],
        *,
        add_result: Any = None,
        budget_seconds: int = 0,
        poll_interval_seconds: float | None = None,
        dataset: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        del add_result, budget_seconds, poll_interval_seconds, dataset, kwargs
        provider_status_counts: Dict[str, int] = {}
        missing_memory_refs: List[str] = []
        warnings: List[str] = []

        for row in import_manifest_rows:
            status = str(row.get("write_status", "unknown"))
            provider_status_counts[status] = provider_status_counts.get(status, 0) + 1
            chunk_id = str(row.get("chunk_id") or "")
            refs = row.get("memory_refs") or []
            if not refs:
                missing_memory_refs.append(chunk_id)

            runtime = self._runtime_from_manifest_row(row)
            if runtime is None:
                warnings.append(f"missing runtime metadata for {chunk_id}")
                continue
            if not runtime.runtime_path.exists():
                warnings.append(f"missing runtime path for {chunk_id}: {runtime.runtime_path}")
            if not runtime.agent_state_path.exists():
                warnings.append(
                    f"missing agent state path for {chunk_id}: {runtime.agent_state_path}"
                )
            if not runtime.sqlite_path.exists():
                warnings.append(f"missing sqlite path for {chunk_id}: {runtime.sqlite_path}")
                continue
            expected_ids = [
                str(ref.get("memory_id") or "") for ref in refs if ref.get("memory_id")
            ]
            if expected_ids:
                present_ids = set(self._fetch_memory_ids(runtime.sqlite_path, expected_ids))
                missing_ids = [
                    memory_id for memory_id in expected_ids if memory_id not in present_ids
                ]
                if missing_ids:
                    warnings.append(
                        f"missing sqlite memory refs for {chunk_id}: "
                        f"{', '.join(missing_ids[:5])}"
                    )

        ready = not missing_memory_refs and not any(
            warning.startswith("missing ") for warning in warnings
        )
        return {
            "import_manifest_records": list(import_manifest_rows),
            "ready": ready,
            "status": "finalized" if ready else "incomplete",
            "provider_status_counts": provider_status_counts,
            "updated_rows": len(import_manifest_rows),
            "missing_memory_refs": missing_memory_refs,
            "warnings": warnings,
            "finalize_budget_exhausted": False,
            "finalized_at": None,
        }

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
        del index, conversation, kwargs
        session_ids = self._extract_session_ids(question_metadata or {})
        if not session_ids:
            readback = StorageReadbackResult(
                status="unsupported",
                checked_session_ids=[],
                missing_session_ids=[],
                objects=[],
                metadata={
                    "provider": "mirix",
                    "question_id": question_id,
                    "reason": "Readback search requires question_metadata.session_ids.",
                },
                errors=[],
            )
            return SearchResult(
                question_id=question_id or "",
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "system": "mirix",
                    "search_mode": "readback",
                    "search_backend": "mirix_storage_readback_for_native_agent",
                    "session_ids": [],
                    "formatted_context": "",
                    "native_agent_answer_deferred": True,
                    "answer_owner": "mirix",
                    "readback": readback.to_dict(),
                },
                retrieval_status="unsupported",
            )

        readback = await self.get_storage_readback(
            session_ids=session_ids,
            question_id=question_id,
            context={
                "conversation_id": conversation_id,
                "import_manifest_rows": import_manifest_records or [],
            },
        )
        objects = [
            obj for obj in readback.objects if isinstance(obj, NormalizedStorageObject)
        ]
        results = [
            {
                "content": obj.content,
                "score": 1.0,
                "metadata": {
                    "memory_id": obj.id,
                    "memory_type": obj.kind,
                    "source_session_id": obj.session_id,
                    "provider_metadata": obj.metadata,
                },
            }
            for obj in objects
            if obj.content
        ]
        formatted_context = "\n\n".join(
            f"{idx}. {obj.content}"
            for idx, obj in enumerate(objects, start=1)
            if obj.content
        )
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "mirix",
                "search_mode": "readback",
                "search_backend": "mirix_storage_readback_for_native_agent",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "native_agent_answer_deferred": True,
                "answer_owner": "mirix",
                "runtime_path": str(readback.metadata.get("runtime_path") or ""),
                "agent_state_path": str(readback.metadata.get("agent_state_path") or ""),
                "sqlite_path": str(readback.metadata.get("sqlite_path") or ""),
                "readback": readback.to_dict(),
            },
            retrieval_status="ok" if results else "empty",
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
        **kwargs,
    ) -> StorageReadbackResult:
        del user_id, run_id, evidence_texts, kwargs
        checked_session_ids = [str(value) for value in (session_ids or []) if value]
        conversation_id = str((context or {}).get("conversation_id") or "").strip()
        if not conversation_id:
            for row in (context or {}).get("import_manifest_rows") or []:
                row_conv = str(row.get("conversation_id") or "").strip()
                if row_conv:
                    conversation_id = row_conv
                    break
        if not conversation_id:
            return StorageReadbackResult(
                status="error",
                checked_session_ids=checked_session_ids,
                missing_session_ids=checked_session_ids,
                objects=[],
                metadata={
                    "provider": "mirix",
                    "question_id": question_id,
                    "reason": "missing conversation_id",
                },
                errors=[],
            )

        runtime = self._runtime_from_manifest(
            conversation_id, (context or {}).get("import_manifest_rows") or []
        )
        if runtime is None:
            runtime = self._runtime_for(conversation_id)
        if not runtime.sqlite_path.exists():
            return StorageReadbackResult(
                status="error",
                checked_session_ids=checked_session_ids,
                missing_session_ids=checked_session_ids,
                objects=[],
                metadata={
                    "provider": "mirix",
                    "question_id": question_id,
                    "reason": "sqlite runtime not found",
                    "runtime_path": str(runtime.runtime_path),
                    "agent_state_path": str(runtime.agent_state_path),
                    "sqlite_path": str(runtime.sqlite_path),
                },
                errors=[],
            )

        objects: List[NormalizedStorageObject] = []
        for row in self._read_memory_rows(runtime):
            matching_session_id = self._matching_source_session_id(
                row.metadata, checked_session_ids
            )
            if not matching_session_id:
                continue
            provider_metadata = {
                **row.metadata,
                "tree_path": row.tree_path,
                "source_session_id": matching_session_id,
            }
            objects.append(
                NormalizedStorageObject(
                    session_id=matching_session_id,
                    kind=row.kind,
                    id=row.id,
                    content=row.content,
                    metadata=provider_metadata,
                    raw={"id": row.id, "kind": row.kind, "tree_path": row.tree_path},
                )
            )

        objects.sort(
            key=lambda obj: (
                checked_session_ids.index(obj.session_id)
                if obj.session_id in checked_session_ids
                else 10**9,
                self._memory_type_rank(obj.kind),
                obj.metadata.get("created_at", ""),
                obj.id,
            )
        )
        found_sessions = {obj.session_id for obj in objects}
        missing_session_ids = [
            session_id for session_id in checked_session_ids if session_id not in found_sessions
        ]
        return StorageReadbackResult(
            status="ok",
            checked_session_ids=checked_session_ids,
            missing_session_ids=missing_session_ids,
            objects=objects,
            metadata={
                "provider": "mirix",
                "question_id": question_id,
                "runtime_path": str(runtime.runtime_path),
                "agent_state_path": str(runtime.agent_state_path),
                "sqlite_path": str(runtime.sqlite_path),
            },
            errors=[],
        )

    def render_answer_prompt(
        self, query: str, context: str, **kwargs: Any
    ) -> Optional[str]:
        del query, context, kwargs
        return None

    def consume_answer_trace(
        self, question_id: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        del kwargs
        if question_id:
            return self._answer_trace_payloads.pop(str(question_id), {})
        if not self._answer_trace_payloads:
            return {}
        key = next(iter(self._answer_trace_payloads))
        return self._answer_trace_payloads.pop(key)

    def get_import_manifest_records(self) -> List[Dict[str, Any]]:
        return list(self._import_manifest_records)

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "mirix",
            "config": self.config,
            "runtime_root": str(self.run_root),
            "supported_datasets": sorted(self.supported_datasets),
        }

    def _validate_supported_dataset(self) -> None:
        dataset_name = str(self.config.get("dataset_name") or "").strip()
        if not dataset_name:
            return
        if dataset_name in self.supported_datasets:
            return
        if any(dataset_name.startswith(f"{name}_") for name in self.supported_datasets):
            return
        allowed = ", ".join(sorted(self.supported_datasets))
        raise ValueError(
            f"MIRIX adapter is configured for SubtleMemory-only evaluation; "
            f"dataset {dataset_name!r} is not supported. Allowed dataset(s): {allowed}."
        )

    def _runtime_for(self, conversation_id: str) -> _RuntimePaths:
        cached = self._runtime_paths.get(conversation_id)
        if cached is not None:
            return cached
        runtime_name = self.runtime_dir_template.format(
            conversation_id=_safe_component(conversation_id)
        )
        runtime_path = self.run_root / runtime_name
        runtime = _RuntimePaths(
            runtime_path=runtime_path,
            agent_state_path=runtime_path / self.agent_state_dirname,
            sqlite_path=runtime_path / self.agent_state_dirname / self.sqlite_filename,
        )
        self._runtime_paths[conversation_id] = runtime
        return runtime

    def _runtime_from_manifest(
        self, conversation_id: str, manifest_rows: List[Dict[str, Any]]
    ) -> Optional[_RuntimePaths]:
        for row in manifest_rows:
            row_conversation_id = str(row.get("conversation_id") or "").strip()
            if row_conversation_id and row_conversation_id != conversation_id:
                continue
            runtime = self._runtime_from_manifest_row(row)
            if runtime is not None:
                self._runtime_paths[conversation_id] = runtime
                return runtime
        return None

    def _runtime_from_manifest_row(self, row: Dict[str, Any]) -> Optional[_RuntimePaths]:
        summary = row.get("write_request_summary") or {}
        receipt = row.get("write_receipt") or {}
        namespace_scope = receipt.get("namespace_scope") or {}
        runtime_path = (
            summary.get("runtime_path")
            or receipt.get("runtime_path")
            or namespace_scope.get("runtime_path")
        )
        agent_state_path = (
            summary.get("agent_state_path")
            or receipt.get("agent_state_path")
            or namespace_scope.get("agent_state_path")
        )
        sqlite_path = (
            summary.get("sqlite_path")
            or receipt.get("sqlite_path")
            or namespace_scope.get("sqlite_path")
        )
        if not runtime_path and agent_state_path:
            runtime_path = str(Path(str(agent_state_path)).parent)
        if not agent_state_path and runtime_path:
            agent_state_path = str(Path(str(runtime_path)) / self.agent_state_dirname)
        if not sqlite_path and agent_state_path:
            sqlite_path = str(Path(str(agent_state_path)) / self.sqlite_filename)
        if runtime_path and agent_state_path and sqlite_path:
            return _RuntimePaths(
                runtime_path=Path(str(runtime_path)),
                agent_state_path=Path(str(agent_state_path)),
                sqlite_path=Path(str(sqlite_path)),
            )
        return None

    def _runtime_from_backend_result(
        self,
        conversation_id: str,
        fallback: _RuntimePaths,
        result: Dict[str, Any],
    ) -> _RuntimePaths:
        runtime_path = Path(str(result.get("runtime_path") or fallback.runtime_path))
        agent_state_path = Path(
            str(result.get("agent_state_path") or runtime_path / self.agent_state_dirname)
        )
        sqlite_path = Path(
            str(result.get("sqlite_path") or agent_state_path / self.sqlite_filename)
        )
        runtime = _RuntimePaths(
            runtime_path=runtime_path,
            agent_state_path=agent_state_path,
            sqlite_path=sqlite_path,
        )
        self._runtime_paths[conversation_id] = runtime
        return runtime

    def _runtime_for_answer(
        self, conversation_id: str, kwargs: Dict[str, Any]
    ) -> _RuntimePaths:
        metadata = kwargs.get("retrieval_metadata")
        if isinstance(metadata, dict):
            runtime_path = metadata.get("runtime_path")
            agent_state_path = metadata.get("agent_state_path")
            sqlite_path = metadata.get("sqlite_path")
            if runtime_path and agent_state_path and sqlite_path:
                runtime = _RuntimePaths(
                    runtime_path=Path(str(runtime_path)),
                    agent_state_path=Path(str(agent_state_path)),
                    sqlite_path=Path(str(sqlite_path)),
                )
                self._runtime_paths[conversation_id] = runtime
                self._native_backends.pop(conversation_id, None)
                return runtime
        return self._runtime_for(conversation_id)

    def _backend_for(self, conversation_id: str) -> MirixNativeBackend:
        cached = self._native_backends.get(conversation_id)
        if cached is not None:
            return cached
        runtime = self._runtime_for(conversation_id)
        backend = self._create_native_backend(runtime)
        self._native_backends[conversation_id] = backend
        return backend

    def _create_native_backend(self, runtime: _RuntimePaths) -> MirixNativeBackend:
        llm_cfg = self.config.get("llm") or {}
        embedding_cfg = self.config.get("embedding") or {}
        return MirixNativeBackend(
            mirix_root=self.mirix_root,
            runtime_path=runtime.runtime_path,
            agent_state_path=runtime.agent_state_path,
            config_path=self.config_path,
            openai_api_key=llm_cfg.get("api_key") or "",
            openai_base_url=(
                self.config.get("mirix", {}).get("openai_base_url")
                or llm_cfg.get("base_url")
                or ""
            ),
            embedding_api_key=embedding_cfg.get("api_key") or "",
            embedding_base_url=embedding_cfg.get("base_url") or "",
        )

    def _reset_runtime(self, runtime: _RuntimePaths) -> None:
        if runtime.runtime_path.exists():
            for child in runtime.runtime_path.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
        runtime.agent_state_path.mkdir(parents=True, exist_ok=True)

    def _build_native_sessions(self, conversation: Conversation) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        for session_key, messages in self._group_messages_by_session(conversation):
            source_session_id = self._source_session_id(messages, session_key)
            metadata = self._session_source_metadata(
                conversation=conversation,
                session_key=session_key,
                session_id=source_session_id,
                messages=messages,
            )
            sessions.append(
                {
                    "session_key": session_key,
                    "source_session_id": source_session_id,
                    "messages": messages,
                    "text": self._format_session_chunk(
                        source_session_timestamp=str(
                            metadata.get("source_session_timestamp") or ""
                        ),
                        messages=messages,
                    ),
                    "source_metadata": metadata,
                    "source_unit_ids": list(metadata.get("source_unit_ids") or []),
                    "source_case_id": str(metadata.get("source_case_id") or ""),
                    "session_order": metadata.get("session_order"),
                    "source_session_source": str(
                        metadata.get("source_session_source") or ""
                    ),
                    "source_session_timestamp": str(
                        metadata.get("source_session_timestamp") or ""
                    ),
                }
            )
        return sessions

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

    def _source_session_id(self, messages: List[Message], session_key: str) -> str:
        for message in messages:
            value = str(message.metadata.get("source_session_id") or "").strip()
            if value:
                return value
        return session_key

    def _session_source_metadata(
        self,
        *,
        conversation: Conversation,
        session_key: str,
        session_id: str,
        messages: List[Message],
    ) -> Dict[str, Any]:
        source_unit_ids: List[str] = []
        for message in messages:
            value = str(
                message.metadata.get("source_unit_id")
                or message.metadata.get("dia_id")
                or ""
            ).strip()
            if value and value not in source_unit_ids:
                source_unit_ids.append(value)
        metadata: Dict[str, Any] = {
            "conversation_id": conversation.conversation_id,
            "source_session_id": session_id,
            "session_key": session_key,
            "source_unit_ids": source_unit_ids,
        }
        for field in (
            "source_case_id",
            "session_order",
            "source_session_source",
            "source_session_timestamp",
        ):
            for message in messages:
                value = message.metadata.get(field)
                if value not in (None, ""):
                    metadata[field] = value
                    break
        return metadata

    def _format_session_chunk(
        self, *, source_session_timestamp: str, messages: List[Message]
    ) -> str:
        timestamp = source_session_timestamp or self._first_message_timestamp(messages)
        header = f"""You are ingesting one SubtleMemory history session for memory extraction.

Instructions:
1. Treat the following user/assistant turns as historical conversation memory.
2. Anchor relative time references to the session timestamp when available.
3. Extract event, fact, semantic, resource, and procedural memories from the utterances.
4. Do not treat metadata identifiers as facts unless the conversation content states them.

Session timestamp: {timestamp}

Conversation turns:

"""
        lines = []
        for message in messages:
            speaker = str(message.sender_name or message.sender_id or "unknown").strip()
            content = str(message.content or "").strip()
            if content:
                lines.append(f"{speaker}: {content}")
        return header + "\n".join(lines)

    def _first_message_timestamp(self, messages: List[Message]) -> str:
        for message in messages:
            value = str(message.metadata.get("source_session_timestamp") or "").strip()
            if value:
                return value
        for message in messages:
            if message.timestamp is not None:
                return _iso_timestamp(message.timestamp)
        return ""

    def _build_manifest_rows(
        self,
        *,
        conversation: Conversation,
        runtime: _RuntimePaths,
        sessions: List[Dict[str, Any]],
        ingest_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for session in sessions:
            session_id = str(session.get("source_session_id") or "")
            memory_refs = list(session.get("memory_refs") or [])
            source_unit_ids = [
                str(value)
                for value in (session.get("source_unit_ids") or [])
                if str(value).strip()
            ]
            record = ImportManifestRecord(
                run_id=str(self.run_context.get("run_id") or "run"),
                system_id=str(self.run_context.get("system_id") or "mirix"),
                conversation_id=conversation.conversation_id,
                view_id="shared",
                chunk_id=f"{conversation.conversation_id}:{session_id}",
                source_unit_ids=source_unit_ids,
                write_request_summary={
                    "source_session_id": session_id,
                    "session_key": session.get("session_key"),
                    "source_case_id": session.get("source_case_id"),
                    "session_order": session.get("session_order"),
                    "source_session_source": session.get("source_session_source"),
                    "source_session_timestamp": session.get("source_session_timestamp"),
                    "message_count": len(session.get("messages") or []),
                    "runtime_path": str(runtime.runtime_path),
                    "agent_state_path": str(runtime.agent_state_path),
                    "sqlite_path": str(runtime.sqlite_path),
                },
                write_receipt={
                    "provider_status": "written",
                    "namespace_scope": {
                        "namespace_id": conversation.conversation_id,
                        "view_id": "shared",
                        "dataset_id": str(
                            self.run_context.get("dataset_id") or "subtlememory"
                        ),
                        "runtime_path": str(runtime.runtime_path),
                        "agent_state_path": str(runtime.agent_state_path),
                        "sqlite_path": str(runtime.sqlite_path),
                    },
                    "runtime_path": str(runtime.runtime_path),
                    "agent_state_path": str(runtime.agent_state_path),
                    "sqlite_path": str(runtime.sqlite_path),
                    "memory_refs": memory_refs,
                    "ingest_status": ingest_result.get("status", "ok"),
                    "native_output": ingest_result.get("native_output", {}),
                    "errors": ingest_result.get("errors", []),
                },
                memory_refs=memory_refs,
                write_status="written" if not ingest_result.get("errors") else "error",
                errors=list(ingest_result.get("errors") or []),
            )
            rows.append(dataclass_to_dict(record))
        return rows

    def _merge_session_results(
        self,
        original_sessions: List[Dict[str, Any]],
        result_sessions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result_by_session_id = {
            str(session.get("source_session_id") or ""): session
            for session in result_sessions
        }
        merged_sessions: List[Dict[str, Any]] = []
        for original in original_sessions:
            session_id = str(original.get("source_session_id") or "")
            merged = dict(original)
            returned = result_by_session_id.get(session_id) or {}
            for key, value in returned.items():
                if key == "messages" and "messages" in merged:
                    continue
                merged[key] = value
            merged_sessions.append(merged)
        return merged_sessions

    def _conversation_session_ids(self, conversation_id: str) -> List[str]:
        cached = self._conversation_session_cache.get(conversation_id)
        if cached:
            return list(cached)
        manifest_ids: List[str] = []
        for row in self._import_manifest_records:
            if str(row.get("conversation_id") or "") != conversation_id:
                continue
            session_id = str(
                (row.get("write_request_summary") or {}).get("source_session_id") or ""
            ).strip()
            if session_id and session_id not in manifest_ids:
                manifest_ids.append(session_id)
        if manifest_ids:
            manifest_ids.sort(key=_session_sort_key)
            self._conversation_session_cache[conversation_id] = manifest_ids
            return manifest_ids

        runtime = self._runtime_for(conversation_id)
        session_ids: List[str] = []
        for row in self._read_memory_rows(runtime):
            session_id = self._source_session_id_from_metadata(row.metadata)
            if session_id and session_id not in session_ids:
                session_ids.append(session_id)
        session_ids.sort(key=_session_sort_key)
        self._conversation_session_cache[conversation_id] = session_ids
        return session_ids

    def _extract_session_ids(self, metadata: Dict[str, Any]) -> List[str]:
        raw_session_ids = metadata.get("session_ids") or []
        values = raw_session_ids if isinstance(raw_session_ids, list) else [raw_session_ids]
        session_ids: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in session_ids:
                session_ids.append(text)
        return session_ids

    def _connect(self, sqlite_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _read_memory_rows(self, runtime: _RuntimePaths) -> List[_StoredMemoryRow]:
        rows: List[_StoredMemoryRow] = []
        if not runtime.sqlite_path.exists():
            return rows
        with self._connect(runtime.sqlite_path) as conn:
            specs = (
                ("episodic", "episodic_memory", "details", "summary"),
                ("semantic", "semantic_memory", "details", "summary"),
                ("resource", "resource_memory", "content", "summary"),
                ("procedural", "procedural_memory", "steps", "summary"),
            )
            for kind, table, primary_content_column, fallback_column in specs:
                try:
                    records = conn.execute(
                        f"""
                        SELECT id, {primary_content_column} AS content,
                               {fallback_column} AS fallback, tree_path, metadata_,
                               created_at
                        FROM {table}
                        """
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                for record in records:
                    metadata = self._json_load_dict(record["metadata_"])
                    created_at = str(record["created_at"] or "")
                    if created_at:
                        metadata.setdefault("created_at", created_at)
                    rows.append(
                        _StoredMemoryRow(
                            id=str(record["id"]),
                            kind=kind,
                            content=self._content_from_record(
                                kind, record["content"], record["fallback"]
                            ),
                            metadata=metadata,
                            tree_path=self._json_load_list(record["tree_path"]),
                            created_at=created_at,
                        )
                    )
        return rows

    def _fetch_memory_ids(self, sqlite_path: Path, memory_ids: List[str]) -> List[str]:
        if not memory_ids or not sqlite_path.exists():
            return []
        found: List[str] = []
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect(sqlite_path) as conn:
            for table in (
                "episodic_memory",
                "semantic_memory",
                "resource_memory",
                "procedural_memory",
            ):
                try:
                    rows = conn.execute(
                        f"SELECT id FROM {table} WHERE id IN ({placeholders})",
                        memory_ids,
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                found.extend(str(row["id"]) for row in rows)
        return found

    def _matching_source_session_id(
        self, metadata: Dict[str, Any], allowed_session_ids: List[str]
    ) -> str:
        direct = str(metadata.get("source_session_id") or "").strip()
        if direct in allowed_session_ids:
            return direct
        for key in ("source_metadata_history", "source_metadata_batch"):
            values = metadata.get(key) or []
            if isinstance(values, dict):
                values = [values]
            if isinstance(values, list):
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    session_id = str(item.get("source_session_id") or "").strip()
                    if session_id in allowed_session_ids:
                        return session_id
        return ""

    def _source_session_id_from_metadata(self, metadata: Dict[str, Any]) -> str:
        direct = str(metadata.get("source_session_id") or "").strip()
        if direct:
            return direct
        for key in ("source_metadata_history", "source_metadata_batch"):
            values = metadata.get(key) or []
            if isinstance(values, dict):
                values = [values]
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        session_id = str(item.get("source_session_id") or "").strip()
                        if session_id:
                            return session_id
        return ""

    def _json_load_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if value in (None, ""):
            return {}
        try:
            parsed = json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _json_load_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value in (None, ""):
            return []
        try:
            parsed = json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return []

    def _content_from_record(self, kind: str, content: Any, fallback: Any) -> str:
        if kind == "procedural":
            steps = self._json_load_list(content)
            if steps:
                return " ".join(step.strip() for step in steps if step.strip())
        primary = str(content or "").strip()
        secondary = str(fallback or "").strip()
        return primary or secondary

    def _memory_type_rank(self, kind: str) -> int:
        order = {
            "episodic": 0,
            "semantic": 1,
            "resource": 2,
            "procedural": 3,
        }
        return order.get(kind, 999)
