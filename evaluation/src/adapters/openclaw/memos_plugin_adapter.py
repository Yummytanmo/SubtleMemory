"""OpenClaw + MemOS plugin adapter."""

from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from evaluation.src.adapters.openclaw.plugin_base import OpenClawPluginAdapterBase
from evaluation.src.adapters.openclaw.session_memory_adapter import OpenClawRuntime
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.providers.memos_adapter import MemosAdapter
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.core.readback import NormalizedStorageObject, StorageReadbackResult


MEMOS_PLUGIN_STATIC_RECALL_SYSTEM_PROMPT = "\n".join(
    [
        "# Role",
        "",
        "You are an intelligent assistant with long-term memory capabilities (MemOS Assistant). Your goal is to combine retrieved memory fragments to provide highly personalized, accurate, and logically rigorous responses.",
        "",
        "# System Context",
        "",
        "* Current Time: Use the runtime-provided current time as the baseline for freshness checks.",
        "* Additional memory context for the current turn may be prepended before the original user query as a structured `<memories>` block.",
        "",
        "# Memory Data",
        "",
        'Below is the information retrieved by MemOS, categorized into "Facts" and "Preferences".',
        "* **Facts**: May include user attributes, historical conversations, or third-party details.",
        "* **Preferences**: The user's explicit or implicit requirements on response style, format, or reasoning.",
        "",
        "# Instructions",
        "",
        "1. Use only memories that pass filtering as context.",
        "2. Answer directly. Never mention internal terms such as memory store, retrieval, or AI opinions.",
    ]
)

MEMOS_PLUGIN_USER_QUERY_MARKER = "user\u200b原\u200b始\u200bquery\u200b：\u200b\u200b\u200b\u200b"


@register_adapter("openclaw_memos_plugin")
class OpenClawMemosPluginAdapter(OpenClawPluginAdapterBase):
    adapter_id = "openclaw_memos_plugin"
    provider_name = "openclaw_memos_plugin"
    default_plugin_id = "memos-cloud-openclaw-plugin"
    default_plugin_path = ""
    plugin_kind = "lifecycle"
    add_hook_name = "agent_end"
    recall_hook_name = "before_agent_start"
    hook_mode = "lifecycle"
    qa_prompt_default_key = "answer_prompt_empty"
    readback_plugin_id = "memos-cloud-openclaw-plugin-readback"

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir=output_dir)
        plugin_cfg = config.get("plugin", {}) or {}
        if os.environ.get("OPENCLAW_MEMOS_PLUGIN_PATH") and not (
            plugin_cfg.get("entry_path") or plugin_cfg.get("path")
        ):
            self.plugin_path = Path(
                os.environ["OPENCLAW_MEMOS_PLUGIN_PATH"]
            ).expanduser()
        elif not str(plugin_cfg.get("entry_path") or plugin_cfg.get("path") or "").strip():
            raise ValueError(
                "OpenClaw MemOS plugin path is not configured. "
                "Set OPENCLAW_MEMOS_PLUGIN_PATH in .env or the shell."
            )
        self._memos_readback_session: Optional[aiohttp.ClientSession] = None
        self._memos_readback_cache: Dict[str, Dict[str, Any]] = {}
        self._memos_readback_locks: Dict[str, asyncio.Lock] = {}

    def _build_plugin_config(
        self, conversation_id: str, *, add_enabled: bool, recall_enabled: bool
    ) -> Dict[str, Any]:
        plugin_cfg = dict(self.plugin_config_overrides)
        cfg = self.config.get("plugin", {}) or {}
        search_top_k = self.config.get("search", {}).get(
            "top_k", self.config.get("memory", {}).get("search_top_k")
        )
        if search_top_k is not None:
            top_k = int(search_top_k)
            plugin_cfg.setdefault("memoryLimitNumber", top_k)
            plugin_cfg.setdefault("preferenceLimitNumber", top_k)
        plugin_cfg.update(
            {
                "baseUrl": cfg.get("base_url", "${MEMOS_BASE_URL:https://memos.memtensor.cn/api/openmem/v1}"),
                "apiKey": (
                    cfg.get("api_key")
                    or os.environ.get("MEMOS_API_KEY")
                    or os.environ.get("MEMOS_KEY")
                    or "${MEMOS_API_KEY}"
                ),
                "userId": cfg.get("user_id", self._namespace_id(conversation_id)),
                "conversationId": self._namespace_id(conversation_id),
                "addEnabled": bool(add_enabled),
                "recallEnabled": bool(recall_enabled),
                "captureStrategy": cfg.get("capture_strategy", "full_session"),
                "asyncMode": bool(cfg.get("async_mode", True)),
                "timeoutMs": int(cfg.get("timeout_ms", 8000)),
                "retries": int(cfg.get("retries", 0)),
            }
        )
        return plugin_cfg

    def _readback_question_payload_from_answer_kwargs(
        self,
        *,
        question_id: str,
        conversation_id: str,
        context: str,
        search_results: Any,
        retrieval_metadata: Any,
    ) -> Dict[str, Any]:
        if not (
            self._is_readback_search_mode()
            and str(question_id or "").strip()
            and isinstance(retrieval_metadata, dict)
            and retrieval_metadata.get("search_mode") == "readback"
        ):
            return {}

        formatted_context = str(
            retrieval_metadata.get("formatted_context") or context or ""
        )
        raw_results = search_results if isinstance(search_results, list) else []
        results = [item for item in raw_results if isinstance(item, dict)]
        runtime_injected_context = retrieval_metadata.get("runtime_injected_context")
        if not isinstance(runtime_injected_context, dict):
            runtime_injected_context = {
                "prependContext": formatted_context,
                "appendSystemContext": (
                    MEMOS_PLUGIN_STATIC_RECALL_SYSTEM_PROMPT
                    if formatted_context
                    else ""
                ),
            }

        return {
            "conversation_id": conversation_id,
            "session_ids": list(retrieval_metadata.get("session_ids") or []),
            "formatted_context": formatted_context,
            "results": list(results),
            "readback": retrieval_metadata.get("readback", {}),
            "runtime_injected_context": runtime_injected_context,
        }

    async def answer(self, query: str, context: str, **kwargs: Any) -> str:
        question_id = str(kwargs.get("question_id") or "").strip()
        if (
            self._is_readback_search_mode()
            and question_id
            and not self._readback_answer_cache.get(question_id)
        ):
            payload = self._readback_question_payload_from_answer_kwargs(
                question_id=question_id,
                conversation_id=str(kwargs.get("conversation_id") or "").strip(),
                context=context,
                search_results=kwargs.get("search_results"),
                retrieval_metadata=kwargs.get("retrieval_metadata"),
            )
            if payload:
                self._readback_answer_cache[question_id] = payload
        return await super().answer(query, context, **kwargs)

    def _readback_plugin_path(self) -> Path:
        return Path(__file__).parent / "plugins" / "memos_readback_lifecycle"

    def _derive_openclaw_config(
        self,
        cfg: Dict[str, Any],
        runtime: OpenClawRuntime,
        *,
        use_readback_plugin: bool = False,
    ) -> Dict[str, Any]:
        next_cfg = super()._derive_openclaw_config(
            cfg,
            runtime,
            use_readback_plugin=use_readback_plugin,
        )
        if not (self._is_readback_search_mode() and use_readback_plugin):
            return next_cfg

        plugins_cfg = next_cfg.setdefault("plugins", {})
        entries = plugins_cfg.setdefault("entries", {})
        entries.pop(self.plugin_id, None)
        entries.pop("openclaw-session-memory-readback", None)
        entries[self.readback_plugin_id] = {"enabled": True, "config": {}}

        load_paths = plugins_cfg.setdefault("load", {}).setdefault("paths", [])
        blocked_paths = {
            str(self._plugin_load_path()),
            str((Path(__file__).parent / "plugins" / "session_memory_readback").resolve()),
        }
        load_paths = [item for item in load_paths if str(item) not in blocked_paths]
        readback_path = str(self._readback_plugin_path().resolve())
        if readback_path not in load_paths:
            load_paths.append(readback_path)
        plugins_cfg.setdefault("load", {})["paths"] = load_paths

        allow = [
            item
            for item in plugins_cfg.setdefault("allow", [])
            if item not in {self.plugin_id, "openclaw-session-memory-readback"}
        ]
        if self.readback_plugin_id not in allow:
            allow.append(self.readback_plugin_id)
        plugins_cfg["allow"] = allow
        plugins_cfg.setdefault("slots", {})["memory"] = self.readback_plugin_id
        return next_cfg

    def _initialize_qa_workspace(self, runtime: OpenClawRuntime) -> None:
        original_search = self.config.get("search")
        if self._is_readback_search_mode():
            self.config["search"] = {**(original_search or {}), "mode": "api"}
        try:
            super()._initialize_qa_workspace(runtime)
        finally:
            if original_search is None:
                self.config.pop("search", None)
            else:
                self.config["search"] = original_search

    def _get_openclaw_agent_prompt(
        self,
        query: str,
        conversation_id: str = "",
        question_id: str = "",
    ) -> str:
        return super()._get_openclaw_agent_prompt(query, conversation_id, question_id)

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
        del index, conversation, kwargs
        question_metadata = question_metadata or {}
        session_ids = self._extract_question_session_ids(question_metadata)
        if not session_ids:
            return self._unsupported_readback_search_result(
                query=query,
                conversation_id=conversation_id,
                question_id=question_id,
                session_ids=[],
                reason="Readback search requires question_metadata.session_ids.",
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
            "qa_row": {"metadata": question_metadata},
        }
        readback = await self.get_storage_readback(
            user_id=user_id,
            session_ids=session_ids,
            question_id=question_id,
            context=context,
        )
        results, formatted_context = self._build_memos_readback_search_payload(
            readback, session_ids=session_ids
        )
        runtime_injected_context = {
            "prependContext": formatted_context,
            "appendSystemContext": (
                MEMOS_PLUGIN_STATIC_RECALL_SYSTEM_PROMPT if formatted_context else ""
            ),
        }
        readback_dict = readback.to_dict()
        if question_id:
            self._readback_answer_cache[str(question_id)] = {
                "conversation_id": conversation_id,
                "session_ids": list(session_ids),
                "formatted_context": formatted_context,
                "results": list(results),
                "readback": readback_dict,
                "runtime_injected_context": runtime_injected_context,
            }
        status = "ok"
        if readback.status in {"unsupported", "error"}:
            status = readback.status
        elif not results and not formatted_context:
            status = "empty"
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": str(self.config.get("name") or "openclaw-memos-plugin"),
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "hook_name": self.recall_hook_name,
                "search_mode": "readback",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "user_ids": readback.metadata.get("user_ids", []),
                "readback": readback_dict,
            },
            retrieval_status=status,
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
        target_session_ids = MemosAdapter._dedupe_nonempty(list(session_ids or []))
        user_ids = self._find_memos_user_ids_for_readback(
            user_id=user_id,
            context=context,
        )

        if not user_ids:
            return StorageReadbackResult(
                status="unsupported",
                checked_session_ids=target_session_ids,
                objects=[],
                metadata={
                    "provider": self.provider_name,
                    "plugin_id": self.plugin_id,
                    "question_id": question_id,
                    "readback_scope": "session" if target_session_ids else "user",
                    "reason": "missing user_id/namespace",
                },
                errors=[
                    {
                        "error_type": "missing_user_id",
                        "error_message": "Cannot determine MemOS user_id for plugin readback.",
                    }
                ],
            )

        all_objects: List[NormalizedStorageObject] = []
        fetched_payloads: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        target_session_set = set(target_session_ids)

        for memos_user_id in user_ids:
            try:
                user_readback = await self._get_cached_memos_readback_for_user(
                    memos_user_id
                )
                fetched_payloads.append(user_readback)
                for normalized in user_readback.get("objects", []) or []:
                    if target_session_set and normalized.session_id not in target_session_set:
                        continue
                    all_objects.append(normalized)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "stage": "get_storage_readback",
                        "user_id": memos_user_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

        object_session_ids = {
            obj.session_id
            for obj in all_objects
            if obj.session_id and obj.session_id != "unknown"
        }
        missing_session_ids = [
            session_id
            for session_id in target_session_ids
            if session_id not in object_session_ids
        ]
        raw_counts = MemosAdapter._merge_memos_raw_counts(
            payload.get("raw_counts", {}) for payload in fetched_payloads
        )
        page_count = sum(
            int(payload.get("page_count", 0) or 0) for payload in fetched_payloads
        )
        status = "error" if errors and not fetched_payloads else "ok"

        return StorageReadbackResult(
            status=status,
            checked_session_ids=target_session_ids,
            missing_session_ids=missing_session_ids,
            objects=all_objects,
            metadata={
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "question_id": question_id,
                "user_ids": user_ids,
                "readback_scope": "session" if target_session_ids else "user",
                "page_count": page_count,
                "raw_memory_count": raw_counts["memory_detail_list"],
                "raw_preference_count": raw_counts["preference_detail_list"],
                "raw_tool_memory_count": raw_counts["tool_memory_detail_list"],
                "raw_skill_count": raw_counts["skill_detail_list"],
                "raw_count_by_list": raw_counts,
                "object_count": len(all_objects),
            },
            errors=errors,
        )

    def _build_memos_readback_search_payload(
        self,
        readback: StorageReadbackResult,
        *,
        session_ids: List[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        results, context = MemosAdapter.build_readback_search_payload_from_config(
            self.config, readback, session_ids=session_ids
        )
        for item in results:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                item["metadata"] = metadata
            metadata["provider"] = self.provider_name
            metadata["plugin_id"] = self.plugin_id
            if metadata.get("session_id"):
                metadata.setdefault("source_session_id", metadata.get("session_id"))
            if "provider_metadata" not in metadata:
                provider_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if key
                    not in {
                        "memory_id",
                        "kind",
                        "session_id",
                        "source_session_id",
                        "provider",
                        "plugin_id",
                    }
                }
                metadata["provider_metadata"] = provider_metadata
                for key in provider_metadata:
                    metadata.pop(key, None)
        return results, self._format_memos_prepend_context_from_search_context(context)

    def _format_memos_prepend_context_from_search_context(self, context: str) -> str:
        text = str(context or "").strip()
        if not text:
            return ""
        facts: List[str] = []
        preferences: List[str] = []
        in_preference_block = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Explicit Preference:") or line.startswith(
                "Implicit Preference:"
            ) or line.startswith("Preference:"):
                in_preference_block = True
                continue
            normalized = re.sub(r"^\d+\.\s*", "", line).strip()
            normalized = re.sub(r"^-\s*", "", normalized).strip()
            if not normalized:
                continue
            target = preferences if in_preference_block else facts
            target.append(f"   - {normalized}")

        block = [
            "<memories>",
            "  <facts>",
            *facts,
            "  </facts>",
            "  <preferences>",
            *preferences,
            "  </preferences>",
            "</memories>",
            "",
            MEMOS_PLUGIN_USER_QUERY_MARKER,
        ]
        return "\n".join(block)

    def _write_readback_stage_manifest(
        self,
        *,
        runtime: OpenClawRuntime,
        answer_session_id: str,
        question_id: str,
        question_payload: Dict[str, Any],
    ) -> Path:
        manifest_path = super()._write_readback_stage_manifest(
            runtime=runtime,
            answer_session_id=answer_session_id,
            question_id=question_id,
            question_payload=question_payload,
        )
        staged = json.loads(manifest_path.read_text(encoding="utf-8"))
        injected = question_payload.get("runtime_injected_context")
        if isinstance(injected, dict):
            staged.update(
                {
                    "appendSystemContext": injected.get("appendSystemContext", ""),
                    "prependContext": injected.get("prependContext", ""),
                }
            )
        staged.setdefault("appendSystemContext", "")
        staged.setdefault("prependContext", staged.get("formattedContext", ""))
        self._write_json_private(manifest_path, staged)
        return manifest_path

    def _capture_openclaw_runtime_recall_payload(
        self,
        *,
        question_id: str,
        conversation_id: str,
        session_id: str,
        prompt: str,
        recall_query: str = "",
        runtime: OpenClawRuntime,
        agent_payload: Any,
    ) -> Dict[str, Any]:
        payload = super()._capture_openclaw_runtime_recall_payload(
            question_id=question_id,
            conversation_id=conversation_id,
            session_id=session_id,
            prompt=prompt,
            recall_query=recall_query,
            runtime=runtime,
            agent_payload=agent_payload,
        )
        if not self._is_readback_search_mode():
            return payload

        staged = self._read_memos_readback_stage_manifest(runtime, session_id)
        context = self._combined_memos_hook_context(staged)
        if context:
            payload["runtime_injected_context"] = context
        question_payload = self._readback_question_payload(
            str(staged.get("questionId") or question_id)
        )
        items = self._runtime_items_from_readback_payload(question_payload)
        if items:
            payload["runtime_retrieved_items"] = items
        if payload.get("runtime_injected_context") or payload.get("runtime_retrieved_items"):
            payload["runtime_logger_status"] = "fallback_memos_readback_stage_manifest"
        if question_id:
            self._runtime_recall_payloads[question_id] = payload
        return payload

    def _capture_plugin_recall_artifact(
        self,
        *,
        conversation_id: str,
        session_id: str,
        prompt: str,
        runtime: OpenClawRuntime,
    ) -> Dict[str, Any]:
        if not self._is_readback_search_mode():
            return super()._capture_plugin_recall_artifact(
                conversation_id=conversation_id,
                session_id=session_id,
                prompt=prompt,
                runtime=runtime,
            )
        staged = self._read_memos_readback_stage_manifest(runtime, session_id)
        context = self._combined_memos_hook_context(staged)
        if not context:
            return {}
        question_id = str(staged.get("questionId") or "").strip()
        question_payload = self._readback_question_payload(question_id)
        items = self._runtime_items_from_readback_payload(question_payload)
        return {
            "context": context,
            "items": items,
            "artifact_path": str(self._readback_stage_root(runtime) / f"{session_id}.json"),
        }

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
                preserve_fallback_results=True,
                preserve_fallback_user_ids=False,
                preserve_fallback_formatted_context=True,
            )
            for result in exported
        ]

    def _read_memos_readback_stage_manifest(
        self, runtime: OpenClawRuntime, session_id: str
    ) -> Dict[str, Any]:
        path = self._readback_stage_root(runtime) / f"{session_id}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _combined_memos_hook_context(staged: Dict[str, Any]) -> str:
        if not isinstance(staged, dict):
            return ""
        parts = []
        for key in ("appendSystemContext", "prependContext"):
            value = str(staged.get(key) or "").strip()
            if value:
                parts.append(value)
        return "\n\n".join(parts).strip()

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
                    ref.get("session_id") or ref.get("source_session_id") or ""
                ).strip()
                if ref_session_id not in target_set:
                    continue
                user_id = str(
                    ref.get("user_id")
                    or ref.get("namespace_id")
                    or row.get("user_id")
                    or row.get("namespace_id")
                    or ""
                ).strip()
                if user_id:
                    return user_id
            receipt = row.get("write_receipt") or {}
            namespace = receipt.get("namespace_scope") if isinstance(receipt, dict) else {}
            if isinstance(namespace, dict):
                namespace_id = str(namespace.get("namespace_id") or "").strip()
                if namespace_id:
                    return namespace_id
        cfg = self.config.get("plugin", {}) or {}
        configured = str(
            self._resolve_env_string(cfg.get("user_id") or "") or ""
        ).strip()
        return configured or None

    def _find_memos_user_ids_for_readback(
        self,
        *,
        user_id: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> List[str]:
        context = context or {}
        candidates: List[Any] = [user_id]
        manifest_rows = context.get("import_manifest_rows") or []
        candidates.extend(MemosAdapter._collect_memos_user_ids_from_manifest_rows(manifest_rows))
        search_result = context.get("search_result", {}) or {}
        retrieval_metadata = search_result.get("retrieval_metadata", {}) or {}
        if isinstance(retrieval_metadata, dict):
            candidates.extend(retrieval_metadata.get("user_ids") or [])
        return MemosAdapter._dedupe_nonempty(candidates)

    def _memos_api_base_url(self) -> str:
        cfg = self.config.get("plugin", {}) or {}
        raw = self._resolve_env_string(
            cfg.get("base_url")
            or self.plugin_config_overrides.get("baseUrl")
            or "https://memos.memtensor.cn/api/openmem/v1"
        )
        return str(raw or "").rstrip("/")

    def _memos_api_key(self) -> str:
        cfg = self.config.get("plugin", {}) or {}
        raw = self._resolve_env_string(
            cfg.get("api_key")
            or self.plugin_config_overrides.get("apiKey")
            or os.environ.get("MEMOS_API_KEY")
            or os.environ.get("MEMOS_KEY")
            or ""
        )
        return str(raw or "")

    async def _get_memos_readback_session(self) -> aiohttp.ClientSession:
        if self._memos_readback_session is None or self._memos_readback_session.closed:
            timeout_ms = int((self.config.get("plugin", {}) or {}).get("timeout_ms", 8000))
            ssl_context = self._create_ssl_context()
            self._memos_readback_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=max(1, timeout_ms / 1000)),
                connector=aiohttp.TCPConnector(ssl=ssl_context),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Token {self._memos_api_key()}",
                },
            )
        return self._memos_readback_session

    def _create_ssl_context(self) -> ssl.SSLContext:
        cfg = self.config.get("plugin", {}) or {}
        ca_file = (
            cfg.get("ssl_ca_file")
            or self.config.get("ssl_ca_file")
            or os.environ.get("SSL_CERT_FILE")
        )
        if ca_file:
            return ssl.create_default_context(cafile=str(ca_file))
        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    async def close(self) -> None:
        if self._memos_readback_session and not self._memos_readback_session.closed:
            await self._memos_readback_session.close()
        parent_close = getattr(super(), "close", None)
        if callable(parent_close):
            result = parent_close()
            if hasattr(result, "__await__"):
                await result

    async def _get_memories_for_user(
        self,
        user_id: str,
        *,
        page: int = 1,
        size: int = 50,
        include_preference: bool = True,
        include_tool_memory: bool = True,
    ) -> Dict[str, Any]:
        session = await self._get_memos_readback_session()
        url = f"{self._memos_api_base_url()}/get/memory"
        normalized_size = max(1, min(int(size), 50))
        payload = {
            "user_id": user_id,
            "page": page,
            "size": normalized_size,
            "include_preference": include_preference,
            "include_tool_memory": include_tool_memory,
        }
        query_payload = {
            key: ("true" if value is True else "false" if value is False else value)
            for key, value in payload.items()
        }
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
        async with session.get(url, params=query_payload) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {text}")
            return await response.json()

    async def _get_all_memories_for_user(
        self,
        user_id: str,
        *,
        size: int = 50,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        pages: List[Dict[str, Any]] = []
        page = 1
        while True:
            payload = await self._get_memories_for_user(
                user_id,
                page=page,
                size=size,
                include_preference=True,
                include_tool_memory=True,
            )
            if not MemosAdapter._memos_envelope_ok(payload):
                raise RuntimeError(f"MemOS /get/memory API error: {payload}")
            pages.append(payload)
            data = payload.get("data", {}) or {}
            try:
                provider_page_count = int(data.get("pages") or 0)
            except (TypeError, ValueError):
                provider_page_count = 0
            current_rows = sum(
                len(data.get(list_key, []) or [])
                for list_key in MemosAdapter.MEMOS_DETAIL_LIST_KINDS
            )
            if max_pages is not None and page >= max_pages:
                break
            if provider_page_count and page >= provider_page_count:
                break
            if not provider_page_count and current_rows < max(1, min(int(size), 50)):
                break
            if current_rows == 0:
                break
            page += 1
        return {"user_id": user_id, "pages": pages}

    async def _get_cached_memos_readback_for_user(
        self, user_id: str
    ) -> Dict[str, Any]:
        cached = self._memos_readback_cache.get(user_id)
        if cached is not None:
            return cached

        lock = self._memos_readback_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            cached = self._memos_readback_cache.get(user_id)
            if cached is not None:
                return cached

            readback_cfg = ((self.config.get("search") or {}).get("readback") or {})
            raw_max_pages = readback_cfg.get("max_pages")
            max_pages = int(raw_max_pages) if raw_max_pages not in (None, "") else None
            pages_payload = await self._get_all_memories_for_user(
                user_id,
                size=int(readback_cfg.get("page_size", 50)),
                max_pages=max_pages,
            )
            objects = [
                MemosAdapter._normalize_memos_storage_object(row, kind=kind)
                for kind, row in MemosAdapter._iter_memos_detail_rows(pages_payload)
            ]
            raw_counts = MemosAdapter._memos_raw_counts([pages_payload])
            cache_entry = {
                "user_id": user_id,
                "objects": objects,
                "raw_counts": raw_counts,
                "page_count": len(pages_payload.get("pages", []) or []),
            }
            self._memos_readback_cache[user_id] = cache_entry
            return cache_entry
