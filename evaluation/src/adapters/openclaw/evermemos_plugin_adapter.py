"""OpenClaw + EverMemOS ContextEngine plugin adapter."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiohttp

from evaluation.src.adapters.openclaw.plugin_base import (
    OpenClawContextEnginePluginAdapterBase,
)
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.shared.evermemos_readback import (
    get_evermemos_storage_readback,
)
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.core.readback import NormalizedStorageObject, StorageReadbackResult


@register_adapter("openclaw_evermemos_plugin")
class OpenClawEverMemOSPluginAdapter(OpenClawContextEnginePluginAdapterBase):
    adapter_id = "openclaw_evermemos_plugin"
    provider_name = "openclaw_evermemos_plugin"
    default_plugin_path = (
        Path(__file__).parent / "plugins" / "evermemos_runtime"
    )
    qa_prompt_default_key = "answer_prompt_empty"

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir=output_dir)
        plugin_cfg = config.get("plugin", {}) or {}
        if os.environ.get("OPENCLAW_EVERMEMOS_PLUGIN_PATH") and not plugin_cfg.get("path"):
            self.plugin_path = Path(os.environ["OPENCLAW_EVERMEMOS_PLUGIN_PATH"]).expanduser()
        self._evermemos_readiness_session: Optional[aiohttp.ClientSession] = None

    def _build_plugin_config(
        self, conversation_id: str, *, add_enabled: bool, recall_enabled: bool
    ) -> Dict[str, Any]:
        plugin_cfg = dict(self.plugin_config_overrides)
        cfg = self.config.get("plugin", {}) or {}
        configured_user_id = self._resolve_env_string(cfg.get("user_id") or "")
        manifest_user_id = ""
        if recall_enabled and not add_enabled:
            manifest_user_id = self._manifest_user_id_for_conversation(conversation_id)
        user_id = (
            configured_user_id
            or manifest_user_id
            or self._provider_namespace_id(conversation_id)
        )
        plugin_cfg.setdefault("topK", self.search_top_k)
        plugin_cfg.setdefault("memoryTypes", self._memory_types())
        plugin_cfg.setdefault("retrieveMethod", self._retrieve_method())
        plugin_cfg.update(
            {
                "baseUrl": cfg.get("base_url", "${EVERMEMOS_API_URL:https://api.evermind.ai}"),
                "apiKey": cfg.get("api_key", "${EVERMEMOS_API_KEY}"),
                "userId": user_id,
                "conversationId": conversation_id,
                "sessionScope": "source_session",
                "captureStrategy": cfg.get("capture_strategy", "full_session"),
                "asyncMode": bool(cfg.get("async_mode", True)),
                "requestIntervalMs": int(cfg.get("request_interval_ms", 1000)),
                "maxRetries": int(cfg.get("max_retries", 5)),
                "retryBaseMs": int(cfg.get("retry_base_ms", 1000)),
                "retryMaxMs": int(cfg.get("retry_max_ms", 8000)),
                "personalBatchMaxMessages": int(
                    cfg.get("personal_batch_max_messages", 5)
                ),
                "personalBatchMaxChars": int(
                    cfg.get("personal_batch_max_chars", 5500)
                ),
                "saveEnabled": bool(add_enabled),
                "readbackEnabled": bool(
                    recall_enabled and self._is_readback_search_mode()
                ),
            }
        )
        plugin_cfg.pop("scope", None)
        plugin_cfg.pop("groupId", None)
        plugin_cfg.pop("group_id", None)
        return plugin_cfg

    def _manifest_user_id_for_conversation(self, conversation_id: str) -> str:
        rows = self._readback_import_manifest_records(None)
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_conversation_id = str(row.get("conversation_id") or "").strip()
            if row_conversation_id and row_conversation_id != conversation_id:
                continue
            user_id = self._user_id_from_manifest_row(row)
            if user_id:
                return user_id
        return ""

    @staticmethod
    def _user_id_from_manifest_row(row: Dict[str, Any]) -> str:
        for value in (
            row.get("user_id"),
            ((row.get("write_receipt") or {}).get("namespace_scope") or {}).get(
                "user_id"
            ),
            (row.get("write_request_summary") or {}).get("user_id"),
        ):
            text = str(value or "").strip()
            if text:
                return text
        for ref in row.get("memory_refs") or []:
            if not isinstance(ref, dict):
                continue
            text = str(ref.get("user_id") or "").strip()
            if text:
                return text
        return ""

    def _provider_namespace_id(
        self, conversation_id: str, *, speaker: str = "user"
    ) -> str:
        del speaker
        base = f"{self._normalize_namespace_component(conversation_id)}_user"
        vision = self._namespace_vision()
        return f"{base}__{vision}" if vision else base

    def _namespace_vision(self) -> str:
        run_context = self.run_context or {}
        raw = run_context.get("vision") or run_context.get("run_name")
        if raw in (None, "", "default"):
            return ""
        return self._normalize_namespace_component(raw)

    @staticmethod
    def _normalize_namespace_component(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "unknown"
        return text.replace(" ", "_").replace("/", "_")

    def _build_plugin_manifest_record(self, **kwargs: Any) -> Dict[str, Any]:
        row = super()._build_plugin_manifest_record(**kwargs)
        conversation = kwargs["conversation"]
        session_key = str(kwargs["session_key"])
        messages = list(kwargs["messages"])
        conversation_id = str(conversation.conversation_id)
        plugin_cfg = self._build_plugin_config(
            conversation_id, add_enabled=True, recall_enabled=False
        )
        source_session_id = self._source_session_id(
            conversation, session_key, messages
        )
        user_id = str(
            self._resolve_env_string(plugin_cfg.get("userId"))
            or self._namespace_id(conversation_id)
        ).strip()

        row["session_id"] = session_key
        row["source_session_id"] = source_session_id
        row["user_id"] = user_id

        write_request = dict(row.get("write_request_summary") or {})
        write_request.update(
            {
                "session_id": session_key,
                "source_session_id": source_session_id,
                "user_id": user_id,
            }
        )
        row["write_request_summary"] = write_request

        receipt = dict(row.get("write_receipt") or {})
        namespace_scope = dict(receipt.get("namespace_scope") or {})
        namespace_scope["user_id"] = user_id
        receipt["namespace_scope"] = namespace_scope
        row["write_receipt"] = receipt

        memory_refs = []
        for ref in row.get("memory_refs") or []:
            next_ref = dict(ref)
            next_ref["session_id"] = source_session_id
            next_ref["source_session_id"] = source_session_id
            next_ref["user_id"] = user_id
            memory_refs.append(next_ref)
        row["memory_refs"] = memory_refs
        return row

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
        """Build a normal SearchResult from staged EverMemOS storage readback."""
        del index, conversation, kwargs
        started_at = time.perf_counter()
        session_ids = self._extract_question_session_ids(question_metadata or {})
        if not session_ids:
            return self._unsupported_readback_search_result(
                query=query,
                conversation_id=conversation_id,
                question_id=question_id,
                session_ids=[],
                reason="Readback search requires question_metadata.session_ids.",
                error_type="missing_session_ids",
            )

        manifest_rows = self._readback_import_manifest_records(import_manifest_records)
        user_id = self._readback_user_id_for_sessions(
            manifest_rows=manifest_rows,
            conversation_id=conversation_id,
            session_ids=session_ids,
        )
        context = {
            "question_id": question_id,
            "conversation_id": conversation_id,
            "session_ids": session_ids,
            "import_manifest_rows": manifest_rows,
            "qa_row": {"metadata": question_metadata or {}},
        }
        readback = await self.get_storage_readback(
            user_id=user_id,
            session_ids=session_ids,
            question_id=question_id,
            context=context,
        )
        results, formatted_context = self._build_readback_search_payload(
            readback, session_ids=session_ids
        )
        retrieval_status = self._readback_result_status(
            readback, has_context=bool(formatted_context)
        )
        readback_dict = readback.to_dict()
        if question_id:
            self._readback_answer_cache[str(question_id)] = {
                "conversation_id": conversation_id,
                "session_ids": list(session_ids),
                "formatted_context": formatted_context,
                "results": list(results),
                "readback": readback_dict,
            }
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": str(
                    self.config.get("name") or "openclaw-evermemos-plugin"
                ),
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "hook_name": self.recall_hook_name,
                "search_mode": "readback",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "user_ids": readback.metadata.get("user_ids", []),
                "readback": readback_dict,
            },
            retrieval_status=retrieval_status,
            timing_ms=(time.perf_counter() - started_at) * 1000,
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
        del run_id, evidence_texts, kwargs
        readback_config = (self.config.get("readback", {}) or {})
        return await get_evermemos_storage_readback(
            request_json=self._evermemos_request_json,
            provider=self.provider_name,
            scope="personal",
            user_id=user_id,
            session_ids=session_ids,
            question_id=question_id,
            context=context,
            page_size=int(readback_config.get("page_size", 100)),
            max_pages=int(readback_config.get("max_pages", 20)),
        )

    def export_runtime_search_results(
        self,
        answer_results: List[Any],
        search_results: Optional[List[SearchResult]] = None,
    ) -> List[SearchResult]:
        exported = super().export_runtime_search_results(answer_results, search_results)
        fallback_by_question = {
            result.question_id: result
            for result in (search_results or [])
            if isinstance(result, SearchResult) and result.question_id
        }
        return [
            self._merge_runtime_and_fallback_search_result(
                result,
                fallback_by_question.get(result.question_id),
                preserve_fallback_results=False,
                preserve_fallback_user_ids=True,
                preserve_fallback_formatted_context=False,
            )
            for result in exported
        ]

    def _unsupported_readback_search_result(
        self,
        *,
        query: str,
        conversation_id: str,
        question_id: Optional[str],
        session_ids: List[str],
        reason: str,
        error_type: str,
    ) -> SearchResult:
        readback = StorageReadbackResult(
            status="unsupported",
            checked_session_ids=list(session_ids),
            missing_session_ids=list(session_ids),
            objects=[],
            metadata={
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "question_id": question_id,
                "conversation_id": conversation_id,
                "reason": reason,
                "readback_scope": "question_sessions",
            },
            errors=[
                {
                    "error_type": error_type,
                    "error_message": reason,
                }
            ],
        )
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={
                "system": str(
                    self.config.get("name") or "openclaw-evermemos-plugin"
                ),
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "hook_name": self.recall_hook_name,
                "search_mode": "readback",
                "session_ids": list(session_ids),
                "formatted_context": "",
                "readback": readback.to_dict(),
            },
            retrieval_status="unsupported",
        )

    @staticmethod
    def _readback_result_status(
        readback: StorageReadbackResult, *, has_context: bool
    ) -> str:
        if readback.status in {"unsupported", "error", "no_user_id"}:
            return readback.status
        if not has_context:
            return "empty"
        return "ok"

    def _build_readback_search_payload(
        self,
        readback: StorageReadbackResult,
        *,
        session_ids: List[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        session_order = {session_id: index for index, session_id in enumerate(session_ids)}
        buckets: Dict[str, List[NormalizedStorageObject]] = {
            session_id: [] for session_id in session_ids
        }
        for obj in readback.objects:
            if not isinstance(obj, NormalizedStorageObject):
                continue
            if obj.session_id not in session_order:
                continue
            buckets.setdefault(obj.session_id, []).append(obj)

        ordered_objects: List[NormalizedStorageObject] = []
        for session_id in session_ids:
            ordered_objects.extend(
                sorted(
                    buckets.get(session_id, []),
                    key=self._readback_object_sort_key,
                )
            )

        results: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        for obj in ordered_objects:
            content = str(obj.content or "").strip()
            if not content:
                continue
            metadata = {
                "memory_id": obj.id,
                "session_id": obj.session_id,
                "source_session_id": obj.session_id,
                "kind": obj.kind,
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "provider_metadata": obj.metadata or {},
            }
            results.append({"content": content, "score": 1.0, "metadata": metadata})
            context_parts.append(f"{len(context_parts) + 1}. {content}")

        formatted_context = "\n\n".join(context_parts)
        if formatted_context:
            formatted_context = (
                "<recalled-memories>\n"
                f"{formatted_context}\n"
                "</recalled-memories>"
            )
        return results, formatted_context

    @staticmethod
    def _readback_object_sort_key(obj: NormalizedStorageObject) -> Tuple[float, str, str]:
        timestamp_ms = OpenClawEverMemOSPluginAdapter._readback_object_timestamp_ms(obj)
        return (
            timestamp_ms if timestamp_ms is not None else float("inf"),
            str(obj.id or ""),
            str(obj.content or ""),
        )

    @staticmethod
    def _readback_object_timestamp_ms(
        obj: NormalizedStorageObject,
    ) -> Optional[float]:
        values: List[Any] = []
        metadata = obj.metadata or {}
        raw = obj.raw or {}
        for container in (metadata, raw):
            for key in (
                "timestamp",
                "create_time",
                "created_at",
                "message_create_time",
                "updated_at",
            ):
                if key in container:
                    values.append(container.get(key))
        prefix = re.match(
            r"^\s*(\d{4}-\d{2}-\d{2}T[0-9:.+-]+Z?)",
            str(obj.content or ""),
        )
        if prefix:
            values.append(prefix.group(1))

        for value in values:
            timestamp_ms = OpenClawEverMemOSPluginAdapter._parse_timestamp_ms(value)
            if timestamp_ms is not None:
                return timestamp_ms
        return None

    @staticmethod
    def _parse_timestamp_ms(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if numeric > 0:
                return numeric
            return None
        text = str(value or "").strip()
        if not text:
            return None
        try:
            numeric = float(text)
            if numeric > 0:
                return numeric
        except ValueError:
            pass
        iso_text = text
        if iso_text.endswith("Z"):
            iso_text = f"{iso_text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(iso_text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp() * 1000

    def _readback_user_id_for_sessions(
        self,
        *,
        manifest_rows: List[Dict[str, Any]],
        conversation_id: str,
        session_ids: List[str],
    ) -> Optional[str]:
        target_set = set(session_ids)
        for row in manifest_rows:
            row_conversation_id = str(row.get("conversation_id") or "")
            if row_conversation_id and row_conversation_id != conversation_id:
                continue
            for ref in row.get("memory_refs") or []:
                if not isinstance(ref, dict):
                    continue
                ref_session_id = str(
                    ref.get("session_id")
                    or ref.get("source_session_id")
                    or ref.get("run_id")
                    or ""
                ).strip()
                if ref_session_id not in target_set:
                    continue
                user_id = str(ref.get("user_id") or row.get("user_id") or "").strip()
                if user_id:
                    return user_id
            row_session_id = str(
                row.get("session_id") or row.get("source_session_id") or ""
            ).strip()
            if row_session_id in target_set:
                user_id = str(row.get("user_id") or "").strip()
                if user_id:
                    return user_id
        cfg = self.config.get("plugin", {}) or {}
        configured = str(
            self._resolve_env_string(cfg.get("user_id") or "") or ""
        ).strip()
        return configured or None

    def _get_openclaw_agent_prompt(
        self,
        query: str,
        conversation_id: str = "",
        question_id: str = "",
    ) -> str:
        return super()._get_openclaw_agent_prompt(query, conversation_id, question_id)

    async def finalize_imports(
        self,
        import_manifest_rows: List[Dict[str, Any]],
        *,
        budget_seconds: int = 0,
        poll_interval_seconds: Optional[float] = None,
        dataset: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        rows = [dict(row) for row in import_manifest_rows]
        target_session_ids = self._target_session_ids_from_dataset(dataset)
        probe_rows = self._rows_for_target_sessions(rows, target_session_ids)
        warnings: List[str] = []
        errors: List[Dict[str, Any]] = []
        provider_status_counts: Dict[str, int] = {}
        flush_status_counts: Dict[str, int] = {}
        missing_memory_refs: List[str] = []
        readiness_attempts = 0
        probe_supported = bool(probe_rows)

        for scope in self._collect_flush_scopes(probe_rows):
            try:
                payload = await self._flush_evermemos_personal_session(
                    user_id=scope["user_id"], session_id=scope["session_id"]
                )
                status = self._provider_status(payload, default="submitted")
                flush_status_counts[status] = flush_status_counts.get(status, 0) + 1
            except Exception as exc:  # noqa: BLE001
                flush_status_counts["error"] = flush_status_counts.get("error", 0) + 1
                errors.append(
                    {
                        "stage": "finalize_flush",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "user_id": scope["user_id"],
                        "session_id": scope["session_id"],
                    }
                )
                warnings.append(
                    f"{scope['user_id']} / {scope['session_id']}: EverMemOS personal flush failed ({type(exc).__name__})"
                )

        deadline = asyncio.get_running_loop().time() + max(0, float(budget_seconds))
        poll_interval = float(
            5.0 if poll_interval_seconds is None else poll_interval_seconds
        )
        if poll_interval < 0:
            poll_interval = 0.0

        refreshed: List[Dict[str, Any]] = rows
        while True:
            row_updates: Dict[str, Dict[str, Any]] = {}
            all_visible = True
            readiness_attempts += 1
            for row in probe_rows:
                next_row = dict(row)
                row_errors = list(next_row.get("errors") or [])
                memory_refs = [
                    dict(ref)
                    for ref in (next_row.get("memory_refs") or [])
                    if isinstance(ref, dict)
                ]
                if not memory_refs:
                    missing_memory_refs.append(str(next_row.get("chunk_id") or ""))
                    status = "missing_memory_ref"
                    all_visible = False
                else:
                    try:
                        visible, status = await self._probe_row_searchable(memory_refs)
                    except Exception as exc:  # noqa: BLE001
                        visible = False
                        status = "probe_error"
                        error = {
                            "stage": "finalize_probe",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "chunk_id": str(next_row.get("chunk_id") or ""),
                        }
                        errors.append(error)
                        row_errors.append(error)
                        warnings.append(
                            f"{next_row.get('chunk_id', '')}: EverMemOS runtime search probe failed ({type(exc).__name__}: {str(exc)[:240]})"
                        )
                    if not visible:
                        all_visible = False
                normalized_status = (
                    "ready" if status in {"completed", "ready"} else status
                )
                next_row["write_status"] = normalized_status
                receipt = dict(next_row.get("write_receipt") or {})
                receipt["provider_status"] = normalized_status
                next_row["write_receipt"] = receipt
                next_row["errors"] = row_errors
                row_updates[str(next_row.get("chunk_id") or "")] = next_row

            if all_visible or budget_seconds <= 0:
                break
            if not any(
                self._probe_status_is_retryable(
                    str(row.get("write_status") or "unknown")
                )
                for row in row_updates.values()
            ):
                break
            now = asyncio.get_running_loop().time()
            if now >= deadline:
                break
            await asyncio.sleep(min(poll_interval, max(0.0, deadline - now)))

        refreshed = []
        for row in rows:
            key = str(row.get("chunk_id") or "")
            refreshed.append(row_updates.get(key, row))

        for row in refreshed:
            status = str(row.get("write_status") or "unknown")
            provider_status_counts[status] = provider_status_counts.get(status, 0) + 1

        probed_ready = bool(probe_rows) and all(
            str(row.get("write_status")) == "ready"
            for row in refreshed
            if self._row_is_targeted(row, target_session_ids)
        )
        non_target_rows = len(refreshed) - len(probe_rows)
        soft_ready = bool(non_target_rows) or not probed_ready
        if not probed_ready:
            warnings.append(
                "EverMemOS plugin finalize could not confirm runtime search visibility for target QA sessions."
            )
        elif non_target_rows:
            warnings.append(
                f"EverMemOS plugin finalize probed {len(probe_rows)} target rows and left {non_target_rows} non-target rows at their import status."
            )

        self._import_manifest_records = refreshed
        ready = probed_ready
        return {
            "import_manifest_records": refreshed,
            "ready": ready,
            "soft_ready": soft_ready,
            "status": "soft_ready" if ready and soft_ready else ("ready" if ready else "not_ready"),
            "provider_status_counts": provider_status_counts,
            "flush_status_counts": flush_status_counts,
            "updated_rows": len(refreshed),
            "missing_memory_refs": sorted(set(missing_memory_refs)),
            "warnings": warnings,
            "errors": errors,
            "finalize_budget_exhausted": bool(not probed_ready and budget_seconds > 0),
            "readiness_probe": {
                "supported": probe_supported,
                "ready": probed_ready,
                "mode": "evermemos_runtime_search_probe",
                "attempts": readiness_attempts,
                "budget_seconds": budget_seconds,
                "poll_interval_seconds": poll_interval,
                "target_session_ids": target_session_ids,
                "probed_rows": len(probe_rows),
                "total_rows": len(rows),
            },
        }

    async def _probe_row_searchable(
        self, memory_refs: Iterable[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        last_status = "processing"
        for memory_ref in memory_refs:
            visible, status = await self._probe_evermemos_runtime_searchable(
                memory_ref
            )
            last_status = status
            if visible:
                return True, "completed"
        return False, last_status

    def _collect_flush_scopes(
        self, rows: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        seen: set[Tuple[str, str]] = set()
        scopes: List[Dict[str, str]] = []
        for row in rows:
            for ref in row.get("memory_refs") or []:
                if not isinstance(ref, dict):
                    continue
                user_id = str(ref.get("user_id") or row.get("user_id") or "").strip()
                session_id = str(
                    ref.get("session_id")
                    or ref.get("source_session_id")
                    or row.get("source_session_id")
                    or row.get("session_id")
                    or ""
                ).strip()
                if not user_id or not session_id:
                    continue
                key = (user_id, session_id)
                if key in seen:
                    continue
                seen.add(key)
                scopes.append({"user_id": user_id, "session_id": session_id})
        return scopes

    async def _flush_evermemos_personal_session(
        self, *, user_id: str, session_id: str
    ) -> Dict[str, Any]:
        return await self._evermemos_request_json(
            "POST",
            "/memories/flush",
            {"user_id": user_id, "session_id": session_id},
        )

    async def _probe_evermemos_runtime_searchable(
        self, memory_ref: Dict[str, Any]
    ) -> Tuple[bool, str]:
        filters = self._evermemos_filters(memory_ref)
        if not filters:
            return False, "contract_mismatch"
        runtime_filters = dict(filters)
        runtime_filters.pop("session_id", None)
        last_error: Optional[Exception] = None
        saw_response = False
        for query in self._probe_queries(memory_ref):
            payload = {
                "query": query,
                "method": self._retrieve_method(),
                "retrieve_method": self._retrieve_method(),
                "memory_types": self._memory_types(),
                "top_k": 3,
                "filters": runtime_filters,
            }
            if "user_id" in runtime_filters:
                payload["user_id"] = runtime_filters["user_id"]
            try:
                data = await self._evermemos_request_json(
                    "POST", "/memories/search", payload
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            saw_response = True
            if self._search_payload_has_memories(data):
                return True, "completed"
        if saw_response:
            return False, "processing"
        if last_error is not None:
            raise last_error
        return False, "processing"

    async def _evermemos_request_json(
        self, method: str, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        base_url = self._evermemos_api_base_url()
        if not base_url:
            raise ValueError("EverMemOS plugin base_url is empty")
        url = f"{base_url.rstrip('/')}{path}"
        session = await self._get_evermemos_readiness_session()
        plugin_cfg = self.config.get("plugin", {}) or {}
        max_retries = max(1, int(plugin_cfg.get("readiness_max_retries", 3)))
        base_delay = max(
            0.0, float(plugin_cfg.get("readiness_retry_base_seconds", 1.0))
        )
        max_delay = max(
            0.0, float(plugin_cfg.get("readiness_retry_max_seconds", 8.0))
        )
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                async with session.request(method.upper(), url, json=payload) as response:
                    text = await response.text()
                    if response.status >= 400:
                        error = RuntimeError(
                            f"{method.upper()} {url} -> {response.status}: {text[:800]}"
                        )
                        if (
                            response.status not in {429}
                            and not 500 <= response.status < 600
                        ):
                            raise error
                        raise error
                    if not text:
                        return {}
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return await response.json(content_type=None)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= max_retries - 1 or not self._readiness_error_is_retryable(exc):
                    raise
                backoff = base_delay * (2 ** attempt)
                delay = min(max_delay, backoff) if max_delay else backoff
                if delay > 0:
                    await asyncio.sleep(delay)
        raise last_error or RuntimeError(f"{method.upper()} {url} failed")

    @staticmethod
    def _readiness_error_is_retryable(error: Exception) -> bool:
        message = str(error)
        match = re.search(r"->\s*(\d{3})", message)
        if not match:
            return True
        status = int(match.group(1))
        return status == 429 or 500 <= status < 600

    async def _get_evermemos_readiness_session(self) -> aiohttp.ClientSession:
        if (
            self._evermemos_readiness_session is not None
            and not self._evermemos_readiness_session.closed
        ):
            return self._evermemos_readiness_session

        headers = {"Content-Type": "application/json"}
        api_key = self._evermemos_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout = aiohttp.ClientTimeout(
            total=float(
                (self.config.get("plugin", {}) or {}).get(
                    "readiness_timeout_seconds", 60
                )
            )
        )
        connector = None
        if not self._readiness_ssl_verify():
            connector = aiohttp.TCPConnector(ssl=False)
        self._evermemos_readiness_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers,
        )
        return self._evermemos_readiness_session

    async def close(self) -> None:
        if (
            self._evermemos_readiness_session is not None
            and not self._evermemos_readiness_session.closed
        ):
            await self._evermemos_readiness_session.close()
        parent_close = getattr(super(), "close", None)
        if callable(parent_close):
            result = parent_close()
            if hasattr(result, "__await__"):
                await result

    def _evermemos_api_base_url(self) -> str:
        cfg = self.config.get("plugin", {}) or {}
        raw = self._resolve_env_string(
            cfg.get("base_url")
            or self.plugin_config_overrides.get("baseUrl")
            or "https://api.evermind.ai"
        )
        return self._normalize_api_base_url(str(raw or ""))

    def _evermemos_api_key(self) -> str:
        cfg = self.config.get("plugin", {}) or {}
        raw = self._resolve_env_string(
            cfg.get("api_key") or self.plugin_config_overrides.get("apiKey") or ""
        )
        return str(raw or "")

    def _readiness_ssl_verify(self) -> bool:
        cfg = self.config.get("plugin", {}) or {}
        if "ssl_verify" in cfg:
            return bool(cfg.get("ssl_verify"))
        if "readiness_ssl_verify" in cfg:
            return bool(cfg.get("readiness_ssl_verify"))
        return bool(self.config.get("ssl_verify", True))

    @staticmethod
    def _normalize_api_base_url(base_url: str) -> str:
        url = (base_url or "").rstrip("/")
        if not url:
            return ""
        for suffix in (
            "/api/v0/memories/search",
            "/api/v0/memories",
            "/api/v1/memories/search",
            "/api/v1/memories",
        ):
            if url.endswith(suffix):
                return url[: -len(suffix)] + "/api/v1"
        if url.endswith("/api/v0"):
            return url[: -len("/api/v0")] + "/api/v1"
        if url.endswith("/api/v1"):
            return url
        return url + "/api/v1"

    def _evermemos_filters(self, memory_ref: Dict[str, Any]) -> Dict[str, str]:
        session_id = str(
            memory_ref.get("session_id") or memory_ref.get("source_session_id") or ""
        ).strip()
        user_id = str(
            memory_ref.get("user_id")
            or self.config.get("plugin", {}).get("user_id")
            or ""
        ).strip()
        if not user_id:
            return {}
        filters = {"user_id": user_id}
        if session_id and session_id != "-1":
            filters["session_id"] = session_id
        return filters

    def _retrieve_method(self) -> str:
        search_cfg = self.config.get("search", {}) or {}
        return str(search_cfg.get("retrieve_method") or "hybrid")

    def _memory_types(self) -> List[str]:
        search_cfg = self.config.get("search", {}) or {}
        values = search_cfg.get("memory_types") or ["episodic_memory"]
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values if str(value or "").strip()]

    @staticmethod
    def _probe_queries(memory_ref: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()

        def add(value: Any) -> None:
            text = str(value or "").strip()
            if len(text) < 3 or text in seen:
                return
            seen.add(text)
            candidates.append(text)

        content = str(memory_ref.get("content") or "").strip()
        if content:
            first = re.split(r"[.!?]\s+", content, maxsplit=1)[0].strip()
            add(first)
            if len(content) > 128:
                add(content[:128].rsplit(" ", 1)[0].strip() or content[:128])
        add(memory_ref.get("session_id"))
        if content and len(content) <= 128:
            add(content)
        return candidates[:4]

    @staticmethod
    def _probe_status_is_retryable(status: str) -> bool:
        return str(status or "").strip().lower() in {
            "submitted",
            "queued",
            "pending",
            "processing",
        }

    @staticmethod
    def _provider_status(payload: Dict[str, Any], default: str = "submitted") -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        candidates = [
            payload.get("status") if isinstance(payload, dict) else None,
            payload.get("message") if isinstance(payload, dict) else None,
            data.get("status") if isinstance(data, dict) else None,
            data.get("message") if isinstance(data, dict) else None,
        ]
        aliases = {
            "done": "completed",
            "complete": "completed",
            "ready": "completed",
            "success": "completed",
            "extracted": "completed",
            "queued": "queued",
            "pending": "pending",
            "processing": "processing",
            "submitted": "submitted",
        }
        for candidate in candidates:
            value = str(candidate or "").strip().lower()
            if value:
                return aliases.get(value, value)
        return default

    @staticmethod
    def _search_payload_has_memories(payload: Dict[str, Any]) -> bool:
        def buckets(value: Any) -> Iterable[Dict[str, Any]]:
            if isinstance(value, dict):
                yield value
                for key in ("data", "result"):
                    child = value.get(key)
                    if isinstance(child, dict):
                        yield child

        for bucket in buckets(payload):
            for key in ("episodes", "memories", "raw_messages", "profiles"):
                values = bucket.get(key)
                if isinstance(values, list) and values:
                    return True
            agent_memory = bucket.get("agent_memory")
            if isinstance(agent_memory, dict) and agent_memory:
                return True
        return False
