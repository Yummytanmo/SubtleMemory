"""OpenClaw + Mem0 plugin adapter."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional

from evaluation.src.adapters.openclaw.plugin_base import OpenClawPluginAdapterBase
from evaluation.src.adapters.openclaw.session_memory_adapter import (
    OpenClawRuntime,
)
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.providers.mem0_adapter import Mem0Adapter
from evaluation.src.core.data_models import AnswerResult, Conversation, SearchResult


@register_adapter("openclaw_mem0_plugin")
class OpenClawMem0PluginAdapter(OpenClawPluginAdapterBase):
    adapter_id = "openclaw_mem0_plugin"
    provider_name = "openclaw_mem0_plugin"
    default_plugin_path = ""
    plugin_kind = "lifecycle"
    add_hook_name = "agent_end"
    recall_hook_name = "before_prompt_build"
    hook_mode = "lifecycle"
    qa_prompt_default_key = "answer_prompt_empty"
    readback_facade_id = "openclaw-mem0-readback-facade"

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir=output_dir)
        plugin_cfg = config.get("plugin", {}) or {}
        if os.environ.get("OPENCLAW_MEM0_PLUGIN_PATH") and not plugin_cfg.get("path"):
            self.plugin_path = Path(os.environ["OPENCLAW_MEM0_PLUGIN_PATH"]).expanduser()
        elif not str(plugin_cfg.get("path") or "").strip():
            raise ValueError(
                "OpenClaw Mem0 plugin path is not configured. "
                "Set OPENCLAW_MEM0_PLUGIN_PATH in .env or the shell."
            )
        self._mem0_provider_adapter: Optional[Mem0Adapter] = None

    def _build_plugin_config(
        self, conversation_id: str, *, add_enabled: bool, recall_enabled: bool
    ) -> Dict[str, Any]:
        plugin_cfg = dict(self.plugin_config_overrides)
        cfg = self.config.get("plugin", {}) or {}
        plugin_cfg.update(
            {
                "mode": cfg.get("mode", "platform"),
                "apiKey": cfg.get("api_key", "${MEM0_API_KEY}"),
                "baseUrl": cfg.get("base_url", "${MEM0_HOST:https://api.mem0.ai}"),
                "userId": cfg.get("user_id", self._namespace_id(conversation_id)),
                "topK": self.search_top_k,
                "autoCapture": bool(add_enabled),
                "autoRecall": bool(recall_enabled),
            }
        )
        return plugin_cfg

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
        if not (use_readback_plugin and self._is_readback_search_mode()):
            return next_cfg

        plugins_cfg = next_cfg.setdefault("plugins", {})
        entries = plugins_cfg.setdefault("entries", {})
        entries.pop(self.plugin_id, None)
        entries.pop("openclaw-session-memory-readback", None)
        entries[self.readback_facade_id] = {
            "enabled": True,
            "config": {"topK": self.search_top_k},
        }

        allow = plugins_cfg.setdefault("allow", [])
        plugins_cfg["allow"] = [
            item
            for item in allow
            if item not in {self.plugin_id, "openclaw-session-memory-readback"}
        ]
        if self.readback_facade_id not in plugins_cfg["allow"]:
            plugins_cfg["allow"].append(self.readback_facade_id)

        load_paths = plugins_cfg.setdefault("load", {}).setdefault("paths", [])
        filtered_paths = [
            path
            for path in load_paths
            if "session_memory_readback" not in str(path)
            and str(self._plugin_load_path()) != str(path)
        ]
        facade_path = str(
            (
                Path(__file__).parent / "plugins" / "mem0_readback_facade"
            ).resolve()
        )
        if facade_path not in filtered_paths:
            filtered_paths.append(facade_path)
        plugins_cfg["load"]["paths"] = filtered_paths

        plugins_cfg.setdefault("slots", {})["memory"] = self.readback_facade_id
        tools = next_cfg.setdefault("tools", {})
        tools["alsoAllow"] = [
            item for item in tools.get("alsoAllow", []) if item != "memory_search"
        ]
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

    def _make_mem0_provider_adapter(self) -> Mem0Adapter:
        if self._mem0_provider_adapter is not None:
            return self._mem0_provider_adapter

        plugin_cfg = self.config.get("plugin", {}) or {}
        provider_cfg = {
            "name": "mem0",
            "api_key": self._resolve_env_string(
                plugin_cfg.get("api_key", "${MEM0_API_KEY}")
            ),
            "host": self._resolve_env_string(
                plugin_cfg.get("host")
                or plugin_cfg.get("base_url")
                or "${MEM0_HOST:https://api.mem0.ai}"
            ),
            "batch_size": int(self.config.get("batch_size", 2)),
            "max_retries": int(self.config.get("max_retries", 5)),
            "max_content_length": int(self.config.get("max_content_length", 12000)),
            "readback": dict(self.config.get("readback") or {}),
            "search": dict(self.config.get("search") or {}),
            "llm": dict(self.config.get("llm") or {}),
        }
        provider = Mem0Adapter(provider_cfg, output_dir=self.output_dir)
        provider.set_run_context(dict(self.run_context or {}))
        self._mem0_provider_adapter = provider
        return provider

    @staticmethod
    def _iter_mem0_add_fetch_records(row: Dict[str, Any]):
        seen: set[int] = set()

        def visit(value: Any):
            if isinstance(value, dict):
                marker = id(value)
                if marker in seen:
                    return
                seen.add(marker)
                url = str(value.get("url") or "")
                response = value.get("response_payload_summary")
                if (
                    "/v3/memories/add" in url
                    or (
                        isinstance(response, dict)
                        and (response.get("event_id") or response.get("id"))
                    )
                ):
                    yield value
                for child in value.values():
                    yield from visit(child)
            elif isinstance(value, list):
                for child in value:
                    yield from visit(child)

        yield from visit(
            {
                "fetch_records": row.get("fetch_records") or [],
                "write_request_summary": row.get("write_request_summary") or {},
                "write_receipt": row.get("write_receipt") or {},
                "memory_refs": row.get("memory_refs") or [],
            }
        )

    @staticmethod
    def _mem0_add_response_from_fetch_record(record: Dict[str, Any]) -> Dict[str, Any]:
        response = record.get("response_payload_summary")
        return dict(response) if isinstance(response, dict) else {}

    @staticmethod
    def _row_source_session_id(row: Dict[str, Any]) -> str:
        return str(
            row.get("source_session_id")
            or row.get("session_id")
            or row.get("run_id")
            or ""
        ).strip()

    @staticmethod
    def _row_namespace_user_id(row: Dict[str, Any]) -> str:
        receipt = row.get("write_receipt") if isinstance(row.get("write_receipt"), dict) else {}
        namespace_scope = (
            receipt.get("namespace_scope", {}) if isinstance(receipt, dict) else {}
        )
        request_summary = (
            row.get("write_request_summary")
            if isinstance(row.get("write_request_summary"), dict)
            else {}
        )
        for candidate in (
            row.get("user_id"),
            namespace_scope.get("namespace_id") if isinstance(namespace_scope, dict) else None,
            request_summary.get("user_id") if isinstance(request_summary, dict) else None,
            request_summary.get("namespace_id") if isinstance(request_summary, dict) else None,
        ):
            text = str(candidate or "").strip()
            if text:
                return text
        for ref in row.get("memory_refs") or []:
            if not isinstance(ref, dict):
                continue
            text = str(ref.get("user_id") or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _merge_memory_refs(
        existing_refs: List[Dict[str, Any]],
        new_refs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def add_ref(ref: Dict[str, Any]) -> None:
            key = (
                str(ref.get("provider") or ""),
                str(ref.get("memory_id") or ""),
                str(ref.get("event_id") or ""),
                str(ref.get("source_session_id") or ref.get("run_id") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            merged.append(dict(ref))

        for ref in new_refs:
            add_ref(ref)
        for ref in existing_refs:
            add_ref(ref)
        return merged

    async def _wait_for_mem0_add_event(
        self,
        provider: Mem0Adapter,
        *,
        event_id: str,
        budget_seconds: float,
        poll_interval_seconds: float,
    ) -> Dict[str, Any]:
        deadline = monotonic() + max(float(budget_seconds), 0.0)
        last_status = "UNKNOWN"
        while True:
            event_data = await provider._get_add_event_status(event_id)
            last_status = str(event_data.get("status") or "UNKNOWN").upper()
            if last_status == "SUCCEEDED":
                return event_data
            if last_status in {"FAILED", "ERROR", "CANCELLED", "CANCELED"}:
                return {
                    **event_data,
                    "status": last_status,
                    "error": event_data.get("error")
                    or event_data.get("message")
                    or event_data.get("detail")
                    or last_status,
                }
            if monotonic() >= deadline:
                return {
                    **event_data,
                    "status": last_status,
                    "timeout": True,
                }
            remaining = max(deadline - monotonic(), 0.0)
            sleep_for = min(max(float(poll_interval_seconds), 0.0), remaining)
            await asyncio.sleep(sleep_for)

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
        del add_result, kwargs
        readiness_cfg = self.config.get("readiness", {}) or {}
        budget = float(budget_seconds or readiness_cfg.get("budget_seconds") or 0)
        poll_interval = float(
            poll_interval_seconds
            if poll_interval_seconds is not None
            else readiness_cfg.get("poll_interval_seconds", 2)
        )
        provider = self._make_mem0_provider_adapter()
        refreshed_rows: List[Dict[str, Any]] = []
        warnings: List[str] = []
        missing_memory_refs: List[str] = []
        event_reports: List[Dict[str, Any]] = []
        any_blocked = False
        deadline = monotonic() + max(budget, 0.0)
        rows = [dict(row) for row in import_manifest_rows]
        target_session_ids = self._target_session_ids_from_dataset(dataset)
        probe_rows = self._rows_for_target_sessions(rows, target_session_ids)
        probed_chunk_ids = {str(row.get("chunk_id") or "") for row in probe_rows}

        for row in rows:
            updated_row = dict(row)
            receipt = dict(updated_row.get("write_receipt") or {})
            memory_refs = [
                dict(ref)
                for ref in updated_row.get("memory_refs") or []
                if isinstance(ref, dict)
            ]
            errors = list(updated_row.get("errors") or [])
            user_id = self._row_namespace_user_id(updated_row)
            source_session_id = self._row_source_session_id(updated_row)
            source_unit_ids = list(updated_row.get("source_unit_ids") or [])
            row_event_reports: List[Dict[str, Any]] = []
            provider_status = Mem0Adapter._normalize_provider_status(
                receipt.get("provider_status") or updated_row.get("write_status"),
                default="submitted",
            )
            is_probe_row = str(updated_row.get("chunk_id") or "") in probed_chunk_ids

            for ref in memory_refs:
                if user_id and not ref.get("user_id"):
                    ref["user_id"] = user_id
                if source_session_id:
                    ref.setdefault("run_id", source_session_id)
                    ref.setdefault("source_session_id", source_session_id)
                    ref.setdefault("session_id", source_session_id)

            if updated_row.get("write_status") in {"error", "failed"}:
                if is_probe_row:
                    any_blocked = True
                provider_status = "error"
                refreshed_rows.append({**updated_row, "write_status": "error"})
                continue

            if not is_probe_row:
                refreshed_rows.append(
                    {
                        **updated_row,
                        "write_receipt": receipt,
                        "memory_refs": memory_refs,
                        "errors": errors,
                    }
                )
                continue

            add_responses = [
                self._mem0_add_response_from_fetch_record(record)
                for record in self._iter_mem0_add_fetch_records(updated_row)
            ]
            deduped_responses: List[Dict[str, Any]] = []
            seen_responses: set[tuple[str, str]] = set()
            for response in add_responses:
                if not response:
                    continue
                event_id = Mem0Adapter._extract_event_id_from_add_response(response)
                response_key = (
                    event_id,
                    str(response.get("id") or response.get("memory_id") or ""),
                )
                if response_key in seen_responses:
                    continue
                seen_responses.add(response_key)
                deduped_responses.append(response)
            add_responses = deduped_responses

            if not add_responses:
                warnings.append(
                    f"{updated_row.get('chunk_id', '')}: no Mem0 add receipt found in plugin hook artifacts"
                )

            for batch_index, response in enumerate(add_responses):
                event_id = Mem0Adapter._extract_event_id_from_add_response(response)
                completed_response = dict(response)
                event_status = ""
                event_data: Dict[str, Any] = {}
                if event_id:
                    remaining_budget = max(deadline - monotonic(), 0.0)
                    event_data = await self._wait_for_mem0_add_event(
                        provider,
                        event_id=event_id,
                        budget_seconds=remaining_budget,
                        poll_interval_seconds=poll_interval,
                    )
                    event_status = str(event_data.get("status") or "").upper()
                    event_report = {
                        "event_id": event_id,
                        "status": event_status,
                    }
                    if event_data.get("timeout"):
                        event_report["timeout"] = True
                    if event_data.get("error"):
                        event_report["error"] = event_data.get("error")
                    row_event_reports.append(event_report)
                    event_reports.append(
                        {
                            **event_report,
                            "chunk_id": updated_row.get("chunk_id", ""),
                        }
                    )
                    if event_status == "SUCCEEDED":
                        completed_response = Mem0Adapter._merge_add_response_with_event(
                            response, event_data
                        )
                    else:
                        if is_probe_row:
                            any_blocked = True
                        provider_status = (
                            "timeout" if event_data.get("timeout") else "error"
                        )
                        errors.append(
                            {
                                "stage": "finalize",
                                "event_id": event_id,
                                "status": event_status,
                                "error": event_data.get("error")
                                or "Mem0 add event did not finish successfully",
                            }
                        )
                        continue

                if user_id:
                    extracted_refs = provider._extract_memory_refs_from_add_response(
                        response=completed_response,
                        user_id=user_id,
                        batch_index=batch_index,
                        source_unit_ids=source_unit_ids,
                        run_id=source_session_id,
                    )
                    memory_refs = self._merge_memory_refs(memory_refs, extracted_refs)
                else:
                    warnings.append(
                        f"{updated_row.get('chunk_id', '')}: unable to determine Mem0 user_id during finalize"
                    )

                response_status = Mem0Adapter._normalize_provider_status(
                    provider._extract_provider_status_from_add_response(
                        completed_response
                    ),
                    default=provider_status,
                )
                if response_status:
                    provider_status = response_status

            if provider_status in {"completed", "ready", "success", "succeeded"}:
                write_status = "ready"
            elif provider_status in {"submitted"} and not errors:
                write_status = "ready"
            else:
                write_status = provider_status or "ready"

            if any(ref.get("memory_id") for ref in memory_refs):
                missing = False
            else:
                missing = bool(add_responses)
            if missing:
                missing_memory_refs.append(str(updated_row.get("chunk_id") or ""))

            receipt["provider_status"] = provider_status
            if row_event_reports:
                receipt["mem0_add_events"] = row_event_reports
            if memory_refs:
                receipt["memory_refs"] = memory_refs
            updated_row["write_receipt"] = receipt
            updated_row["memory_refs"] = memory_refs
            updated_row["write_status"] = write_status
            updated_row["errors"] = errors
            refreshed_rows.append(updated_row)

        probe_supported = bool(probe_rows)
        ready = probe_supported and not any_blocked
        non_target_rows = len(refreshed_rows) - len(probe_rows)
        soft_ready = ready and bool(non_target_rows)
        status = "soft_ready" if soft_ready else ("ready" if ready else "not_ready")
        if soft_ready:
            warnings.append(
                f"Mem0 plugin finalize probed {len(probe_rows)} target rows and left {non_target_rows} non-target rows at their import status."
            )
        elif target_session_ids and not probe_rows:
            warnings.append(
                "Mem0 plugin finalize could not find import rows for target QA sessions."
            )
        self._import_manifest_records = refreshed_rows
        return {
            "import_manifest_records": refreshed_rows,
            "ready": ready,
            "soft_ready": soft_ready,
            "status": status,
            "provider_status_counts": self._status_counts(refreshed_rows),
            "updated_rows": len(refreshed_rows),
            "missing_memory_refs": missing_memory_refs,
            "warnings": warnings,
            "finalize_budget_exhausted": any(
                bool(event.get("timeout")) for event in event_reports
            ),
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "readiness_probe": {
                "supported": True,
                "ready": ready,
                "mode": "mem0_plugin_add_event",
                "events": event_reports,
                "target_session_ids": target_session_ids,
                "probed_rows": len(probe_rows),
                "total_rows": len(rows),
            },
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
        provider = self._make_mem0_provider_adapter()
        result = await provider.search_from_readback(
            query,
            conversation_id,
            index,
            conversation=conversation,
            question_id=question_id,
            question_metadata=question_metadata,
            import_manifest_records=import_manifest_records,
            **kwargs,
        )
        if question_id:
            self._readback_answer_cache[str(question_id)] = {
                "conversation_id": conversation_id,
                "session_ids": list(
                    (result.retrieval_metadata or {}).get("session_ids") or []
                ),
                "formatted_context": (result.retrieval_metadata or {}).get(
                    "formatted_context", ""
                ),
                "results": list(result.results or []),
                "readback": (result.retrieval_metadata or {}).get("readback", {}),
            }
        return result

    def _write_readback_stage_manifest(
        self,
        *,
        runtime: OpenClawRuntime,
        answer_session_id: str,
        question_id: str,
        question_payload: Dict[str, Any],
    ) -> Path:
        stage_root = self._readback_stage_root(runtime)
        stage_root.mkdir(parents=True, exist_ok=True)
        manifest_path = stage_root / f"{answer_session_id}.json"
        memories = []
        for item in question_payload.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            metadata = (
                dict(item.get("metadata") or {})
                if isinstance(item.get("metadata"), dict)
                else {}
            )
            memories.append(
                {
                    "id": str(
                        metadata.get("memory_id")
                        or metadata.get("id")
                        or item.get("id")
                        or ""
                    ),
                    "content": str(item.get("content") or ""),
                    "score": item.get("score", 1.0),
                    "metadata": metadata,
                }
            )
        self._write_json_private(
            manifest_path,
            {
                "sessionId": answer_session_id,
                "questionId": question_id,
                "conversationId": question_payload.get(
                    "conversation_id", runtime.conversation_id
                ),
                "sessionIds": list(question_payload.get("session_ids") or []),
                "formattedContext": question_payload.get("formatted_context", ""),
                "readback": question_payload.get("readback", {}),
                "memories": memories,
            },
        )
        return manifest_path

    def export_runtime_search_results(
        self,
        answer_results: List[AnswerResult],
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
                preserve_fallback_formatted_context=True,
            )
            for result in exported
        ]

    def _capture_plugin_recall_artifact(
        self,
        *,
        conversation_id: str,
        session_id: str,
        prompt: str,
        runtime: OpenClawRuntime,
    ) -> Dict[str, Any]:
        if not self._should_use_readback_answer_payload():
            return super()._capture_plugin_recall_artifact(
                conversation_id=conversation_id,
                session_id=session_id,
                prompt=prompt,
                runtime=runtime,
            )
        hook_payload = {
            "mode": "lifecycle",
            "action": "recall",
            "hookName": self.recall_hook_name,
            "pluginId": self.readback_facade_id,
            "pluginPath": str(
                (
                    Path(__file__).parent / "plugins" / "mem0_readback_facade"
                ).resolve()
            ),
            "pluginConfig": {},
            "openclawConfig": self._hook_openclaw_config(runtime),
            "ctx": {
                "sessionId": session_id,
                "sessionKey": f"agent:main:explicit:{session_id}",
                "conversationId": conversation_id,
                "agentId": "main",
            },
            "prompt": prompt,
            "stateDir": str(runtime.state_dir),
            "workspaceDir": str(runtime.workspace_dir),
            "configPath": str(runtime.config_path),
        }
        result = self._run_hook_runner(
            hook_payload, f"{conversation_id}-{session_id}-{self.recall_hook_name}"
        )
        calls = result.get("calls") or []
        call = calls[0] if calls and isinstance(calls[0], dict) else {}
        context = str(call.get("formatted_context") or "").strip()
        if not context:
            return {}
        return {
            "context": context,
            "items": self._runtime_items_from_readback_payload(
                self._readback_question_payload(
                    str((call.get("result") or {}).get("questionId") or "")
                )
            )
            or [{"content": context, "score": 1.0, "metadata": {"source": "mem0_readback_facade"}}],
            "artifact_path": str(result.get("artifact_path") or ""),
        }
