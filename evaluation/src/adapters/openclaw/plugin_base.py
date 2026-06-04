"""Shared helpers for OpenClaw plugin-backed evaluation adapters."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evaluation.src.adapters.openclaw.session_memory_adapter import (
    OPENCLAW_EMPTY_QA_PROMPT_KEYS,
    OpenClawRuntime,
    OpenClawSessionMemoryAdapter,
    _absolute_path,
    _json_default,
    _safe_component,
    _short_hash,
    _utc_iso,
)
from evaluation.src.core.data_models import AnswerResult, Conversation, Message, SearchResult
from evaluation.src.run_artifacts.models import ImportManifestRecord, dataclass_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class OpenClawPluginAdapterBase(OpenClawSessionMemoryAdapter):
    """OpenClaw local-agent adapter where provider memory is handled by a plugin."""

    adapter_id = "openclaw_plugin"
    plugin_kind = "lifecycle"
    add_hook_name = "agent_end"
    recall_hook_name = "before_prompt_build"
    hook_mode = "lifecycle"
    qa_prompt_default_key = "answer_prompt_empty"
    provider_name = "openclaw_plugin"
    default_plugin_id = ""
    default_plugin_path = "."
    uses_runtime_recall_artifacts = True

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir=output_dir)
        plugin_cfg = config.get("plugin", {}) or {}
        self.plugin_id = str(
            plugin_cfg.get("id") or self.default_plugin_id or self.adapter_id
        )
        if plugin_cfg.get("entry_path") is not None:
            plugin_path_value = plugin_cfg.get("entry_path")
            plugin_path_field = "plugin.entry_path"
        elif plugin_cfg.get("path") is not None:
            plugin_path_value = plugin_cfg.get("path")
            plugin_path_field = "plugin.path"
        else:
            plugin_path_value = self.default_plugin_path
            plugin_path_field = "plugin.path"
        self.plugin_path = self._config_path(
            plugin_path_value,
            field=plugin_path_field,
        )
        self.hook_runner_path = self._config_path(
            plugin_cfg.get("hook_runner_path")
            or Path(__file__).parent / "assets" / "plugin_hook_runner.mjs",
            field="plugin.hook_runner_path",
        )
        self.plugin_config_overrides = dict(plugin_cfg.get("config") or {})
        self.plugin_env = dict(plugin_cfg.get("env") or {})
        self.add_wait_after_hook_ms = int(plugin_cfg.get("add_wait_after_hook_ms", 0))
        self.plugin_timeout_seconds = int(plugin_cfg.get("timeout_seconds", 180))
        self.run_root = _absolute_path(
            plugin_cfg.get("run_root")
            or self.config.get("openclaw", {}).get("run_root")
            or self.output_dir / f"{self.adapter_id}_runtime"
        )
        self.conversation_runtime_root = self.run_root / "conversations"
        self.hook_payload_dir = self.run_root / "hook_payloads"
        self.hook_result_dir = self.run_root / "hook_results"
        self.default_runtime = self._runtime_for("default")
        self._runtime_by_conversation = {}
        self._runtime_recall_payloads: Dict[str, Dict[str, Any]] = {}
        self._runtime_search_results: Dict[str, SearchResult] = {}
        self._import_manifest_records: List[Dict[str, Any]] = []

    @staticmethod
    def _config_path(path: Any, *, field: str = "path") -> Path:
        if isinstance(path, (dict, list, tuple, set)):
            raise ValueError(f"{field} must be a path string, got {type(path).__name__}")
        if path is None:
            raise ValueError(f"{field} must be a path string, got None")
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        return resolved.resolve(strict=False)

    @staticmethod
    def _resolve_env_string(value: Any) -> str:
        text = str(value or "")

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            fallback = match.group(2) or ""
            return os.environ.get(key) or fallback

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}", repl, text)

    def set_run_context(self, run_context: Dict[str, Any]) -> None:
        super().set_run_context(run_context)
        self._ensure_subtlememory_supported()

    def _ensure_subtlememory_supported(self) -> None:
        dataset_id = str((self.run_context or {}).get("dataset_id") or self.config.get("dataset_name") or "")
        if dataset_id and not dataset_id.startswith("subtlememory"):
            raise ValueError(f"{self.adapter_id} only supports subtlememory datasets, got {dataset_id!r}")

    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        self._ensure_subtlememory_supported()
        await super().prepare(conversations, **kwargs)

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        del kwargs
        self._ensure_subtlememory_supported()
        self._cache_conversations(conversations)
        self._import_manifest_records = []
        summary = []
        for conversation in conversations:
            runtime = self._runtime_for(conversation.conversation_id)
            self._prepare_runtime_dirs(runtime)
            submitted = 0
            for session_key, messages in self._group_messages_by_session(conversation):
                if not messages:
                    continue
                hook_result = await asyncio.to_thread(
                    self._run_add_hook, conversation, session_key, messages, runtime
                )
                self._import_manifest_records.append(
                    self._build_plugin_manifest_record(
                        conversation=conversation,
                        session_key=session_key,
                        messages=messages,
                        hook_result=hook_result,
                    )
                )
                submitted += 1
            summary.append({"conversation_id": conversation.conversation_id, "sessions_submitted": submitted})
        self._write_json(
            self.output_dir / f"{self.adapter_id}_import_summary.json",
            {
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "plugin_path": str(self.plugin_path),
                "conversation_count": len(conversations),
                "summary": summary,
            },
        )
        return self._build_index_metadata()

    async def finalize_imports(
        self,
        import_manifest_rows: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        del kwargs
        rows = []
        warnings = []
        for row in import_manifest_rows:
            next_row = dict(row)
            if next_row.get("write_status") in {"error", "failed"}:
                warnings.append(f"plugin hook import error for {next_row.get('chunk_id')}")
            else:
                next_row["write_status"] = "ready"
            rows.append(next_row)
        self._import_manifest_records = rows
        return {
            "import_manifest_records": rows,
            "ready": True,
            "soft_ready": bool(warnings),
            "status": "soft_ready" if warnings else "ready",
            "provider_status_counts": self._status_counts(rows),
            "updated_rows": len(rows),
            "missing_memory_refs": [str(row.get("chunk_id") or "") for row in rows if not row.get("memory_refs")],
            "warnings": warnings,
            "finalize_budget_exhausted": False,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "readiness_probe": {"supported": True, "ready": True, "mode": "plugin_hook_artifact"},
        }

    def _build_index_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.adapter_id,
            "provider": self.provider_name,
            "plugin_id": self.plugin_id,
            "conversation_ids": list(self._conversations.keys()),
            "memory_count": len(self._import_manifest_records),
            "runtime_root": str(self.run_root),
            "conversation_runtimes": self._conversation_runtime_summary(
                self._conversations.keys()
            ),
        }

    async def search(self, query: str, conversation_id: str, index: Any, **kwargs) -> SearchResult:
        del index
        return self.build_runtime_placeholder_search_result(
            question_id=str(kwargs.get("question_id") or ""),
            query=query,
            conversation_id=conversation_id,
        )

    def build_runtime_placeholder_search_result(
        self, *, question_id: str, query: str, conversation_id: str
    ) -> SearchResult:
        return SearchResult(
            question_id=question_id,
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={
                "provider": self.provider_name,
                "deferred_to_answer_runtime": True,
                "formatted_context": "",
                "plugin_id": self.plugin_id,
                "hook_name": self.recall_hook_name,
            },
            retrieval_status="deferred",
            timing_ms=0.0,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        answer = await super().answer(query, context, **kwargs)
        question_id = str(kwargs.get("question_id") or "")
        if question_id:
            payload = self._runtime_recall_payloads.get(question_id)
            if payload:
                self._runtime_search_results[question_id] = self._runtime_payload_to_search_result(
                    question_id=question_id,
                    query=query,
                    conversation_id=str(kwargs.get("conversation_id") or ""),
                    payload=payload,
                )
        return answer

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
        payload["runtime_adapter"] = self.adapter_id
        payload["runtime_provider"] = self.provider_name
        payload["runtime_plugin_id"] = self.plugin_id
        payload["runtime_hook_name"] = self.recall_hook_name
        if payload.get("runtime_injected_context") or payload.get("runtime_retrieved_items"):
            return payload

        try:
            fallback = self._capture_plugin_recall_artifact(
                conversation_id=conversation_id,
                session_id=session_id,
                prompt=(recall_query or prompt),
                runtime=runtime,
            )
        except Exception as exc:  # noqa: BLE001
            payload["runtime_recall_fallback_error_type"] = type(exc).__name__
            return payload

        if not fallback:
            return payload

        payload["runtime_injected_context"] = fallback["context"]
        payload["runtime_retrieved_items"] = fallback["items"]
        payload["runtime_logger_status"] = "fallback_plugin_hook_runner_recall"
        return payload

    def _capture_openclaw_runtime_logger_payload(
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
        return self._capture_openclaw_runtime_recall_payload(
            question_id=question_id,
            conversation_id=conversation_id,
            session_id=session_id,
            prompt=prompt,
            recall_query=recall_query,
            runtime=runtime,
            agent_payload=agent_payload,
        )

    def _capture_plugin_recall_artifact(
        self,
        *,
        conversation_id: str,
        session_id: str,
        prompt: str,
        runtime: OpenClawRuntime,
    ) -> Dict[str, Any]:
        hook_payload = {
            "mode": self.hook_mode,
            "action": "recall",
            "hookName": self.recall_hook_name,
            "pluginId": self.plugin_id,
            "pluginPath": str(self.plugin_path),
            "pluginConfig": self._build_plugin_config(
                conversation_id, add_enabled=False, recall_enabled=True
            ),
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
        artifact_path = str(result.get("artifact_path") or "")
        item = {
            "content": context,
            "score": 1.0,
            "metadata": {
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "hook_name": self.recall_hook_name,
                "source": "plugin_hook_runner_recall",
                "artifact_path": artifact_path,
            },
        }
        return {
            "context": context,
            "items": [item],
            "artifact_path": artifact_path,
        }

    def export_runtime_search_results(
        self,
        answer_results: List[AnswerResult],
        search_results: Optional[List[SearchResult]] = None,
    ) -> List[SearchResult]:
        existing = {result.question_id: result for result in (search_results or [])}
        exported = []
        for answer in answer_results:
            runtime_result = self._runtime_search_results.get(answer.question_id)
            if runtime_result is None:
                payload = answer.metadata or {}
                runtime_result = self._runtime_payload_to_search_result(
                    question_id=answer.question_id,
                    query=answer.question,
                    conversation_id=answer.conversation_id,
                    payload=payload,
                    fallback=existing.get(answer.question_id),
                )
            exported.append(runtime_result)
        return exported

    def _merge_runtime_and_fallback_search_result(
        self,
        runtime_result: SearchResult,
        fallback: Optional[SearchResult],
        *,
        preserve_fallback_results: bool = False,
        preserve_fallback_user_ids: bool = True,
        preserve_fallback_formatted_context: bool = True,
    ) -> SearchResult:
        if fallback is None:
            return runtime_result

        fallback_metadata = dict(fallback.retrieval_metadata or {})
        runtime_metadata = dict(runtime_result.retrieval_metadata or {})

        for key in ("search_mode", "session_ids", "readback"):
            if key in fallback_metadata and key not in runtime_metadata:
                runtime_metadata[key] = fallback_metadata[key]
        if preserve_fallback_user_ids and "user_ids" in fallback_metadata and "user_ids" not in runtime_metadata:
            runtime_metadata["user_ids"] = fallback_metadata["user_ids"]
        if (
            preserve_fallback_formatted_context
            and fallback_metadata.get("formatted_context")
            and not runtime_metadata.get("formatted_context")
        ):
            runtime_metadata["formatted_context"] = fallback_metadata["formatted_context"]

        if fallback_metadata.get("search_mode") == "readback":
            runtime_metadata["search_mode"] = "readback"
            for key in ("readback", "session_ids"):
                if key in fallback_metadata:
                    runtime_metadata[key] = fallback_metadata[key]
            if preserve_fallback_user_ids and "user_ids" in fallback_metadata:
                runtime_metadata["user_ids"] = fallback_metadata["user_ids"]
            if preserve_fallback_formatted_context and fallback_metadata.get("formatted_context"):
                runtime_metadata["formatted_context"] = fallback_metadata["formatted_context"]
            if preserve_fallback_results and not runtime_result.results and fallback.results:
                runtime_result.results = list(fallback.results)
            if runtime_result.retrieval_status == "empty" and fallback.retrieval_status:
                runtime_result.retrieval_status = fallback.retrieval_status

        runtime_result.retrieval_metadata = runtime_metadata
        return runtime_result

    def _runtime_payload_to_search_result(
        self,
        *,
        question_id: str,
        query: str,
        conversation_id: str,
        payload: Dict[str, Any],
        fallback: Optional[SearchResult] = None,
    ) -> SearchResult:
        items = payload.get("runtime_retrieved_items") or []
        context = str(payload.get("runtime_injected_context") or "")
        fallback_metadata = (fallback.retrieval_metadata if fallback else {}) or {}
        if not items and fallback is not None:
            items = list(fallback.results or [])
        if not context and fallback_metadata.get("search_mode") == "readback":
            context = str(fallback_metadata.get("formatted_context") or "")
        fallback_metadata = {
            key: value
            for key, value in fallback_metadata.items()
            if key != "runtime_raw_artifact_refs"
        }
        metadata = {
            **fallback_metadata,
            "provider": self.provider_name,
            "plugin_id": self.plugin_id,
            "hook_name": self.recall_hook_name,
            "source": "openclaw_agent_answer_runtime",
            "formatted_context": context,
            "runtime_answer_session_id": payload.get("runtime_answer_session_id", ""),
            "runtime_logger_status": payload.get("runtime_logger_status", ""),
        }
        status = "ok" if items or context else "empty"
        if not items and not context and fallback is not None:
            status = fallback.retrieval_status
        return SearchResult(
            question_id=question_id,
            query=query,
            conversation_id=conversation_id,
            results=list(items),
            retrieval_metadata=metadata,
            retrieval_status=status,
            timing_ms=(fallback.timing_ms if fallback else 0.0),
        )

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
            use_readback_plugin=(
                use_readback_plugin and self.plugin_kind != "context_engine"
            ),
        )
        plugins_cfg = next_cfg.setdefault("plugins", {})
        plugin_entries = plugins_cfg.setdefault("entries", {})
        plugin_entries[self.plugin_id] = {
            "enabled": True,
            "config": self._build_plugin_config(runtime.conversation_id, add_enabled=False, recall_enabled=True),
        }
        load_paths = plugins_cfg.setdefault("load", {}).setdefault("paths", [])
        plugin_load_path = str(self._plugin_load_path())
        if plugin_load_path not in load_paths:
            load_paths.append(plugin_load_path)
        allow = plugins_cfg.setdefault("allow", [])
        if self.plugin_id not in allow:
            allow.append(self.plugin_id)
        slots = plugins_cfg.setdefault("slots", {})
        if self.plugin_kind == "context_engine":
            slots["memory"] = "none"
            slots["contextEngine"] = self.plugin_id
        else:
            slots["memory"] = self.plugin_id
        return next_cfg

    def _plugin_load_path(self) -> Path:
        if self.plugin_path.is_file() or self.plugin_path.suffix.lower() in {
            ".js",
            ".mjs",
            ".cjs",
            ".ts",
            ".mts",
            ".cts",
        }:
            return self.plugin_path.parent
        return self.plugin_path

    def _get_openclaw_agent_prompt_template(self) -> str:
        answer_cfg = self.config.get("answer", {})
        prompt_key = answer_cfg.get("openclaw_agent_prompt_key", self.qa_prompt_default_key)
        if prompt_key is None:
            return ""
        prompt_key = str(prompt_key).strip()
        if prompt_key.lower() in OPENCLAW_EMPTY_QA_PROMPT_KEYS:
            return ""
        raise KeyError(
            "OpenClaw plugin answer prompts are no longer loaded from prompts.yaml; "
            "put answer rules in AGENTS.md and set answer.openclaw_agent_prompt_key "
            "to answer_prompt_empty so --message receives the raw question."
        )

    def _build_plugin_config(self, conversation_id: str, *, add_enabled: bool, recall_enabled: bool) -> Dict[str, Any]:
        return {
            **self.plugin_config_overrides,
            "conversationId": conversation_id,
            "runId": self._run_id(),
            "autoCapture": bool(add_enabled),
            "autoRecall": bool(recall_enabled),
        }

    def _group_messages_by_session(
        self, conversation: Conversation
    ) -> Iterable[tuple[str, List[Message]]]:
        groups: Dict[str, List[Message]] = {}
        for message in conversation.messages:
            session_key = str(
                message.metadata.get("source_session_id")
                or message.metadata.get("session")
                or "session_0"
            )
            groups.setdefault(session_key, []).append(message)

        def sort_key(item: tuple[str, List[Message]]) -> tuple[Any, str]:
            messages = item[1]
            order = messages[0].metadata.get("session_order") if messages else None
            try:
                return (int(order), item[0])
            except Exception:
                return (999999, item[0])

        yield from sorted(groups.items(), key=sort_key)

    def _run_add_hook(
        self,
        conversation: Conversation,
        session_key: str,
        messages: List[Message],
        runtime: OpenClawRuntime,
    ) -> Dict[str, Any]:
        payload = {
            "mode": self.hook_mode,
            "action": "add",
            "hookName": self.add_hook_name,
            "pluginId": self.plugin_id,
            "pluginPath": str(self.plugin_path),
            "pluginConfig": self._build_plugin_config(
                conversation.conversation_id, add_enabled=True, recall_enabled=False
            ),
            "openclawConfig": self._hook_openclaw_config(runtime),
            "event": self._build_add_event(conversation, session_key, messages),
            "ctx": self._build_hook_context(conversation, session_key, messages),
            "messages": [self._hook_message(message) for message in messages],
            "turns": [{"turnIndex": 0, "messages": [self._hook_message(message) for message in messages]}],
            "waitAfterHookMs": self.add_wait_after_hook_ms,
            "stateDir": str(runtime.state_dir),
            "workspaceDir": str(runtime.workspace_dir),
        }
        return self._run_hook_runner(payload, f"{conversation.conversation_id}-{session_key}-{self.add_hook_name}")

    def _build_add_event(
        self, conversation: Conversation, session_key: str, messages: List[Message]
    ) -> Dict[str, Any]:
        timestamp = _utc_iso(messages[-1].timestamp if messages else None)
        capture_date = timestamp.split("T", 1)[0] if timestamp else ""
        return {
            "success": True,
            "messages": [self._hook_message(message) for message in messages],
            "sessionId": self._source_session_id(conversation, session_key, messages),
            "sessionKey": f"agent:main:{self._source_session_id(conversation, session_key, messages)}",
            "timestamp": timestamp,
            "sessionTimestamp": timestamp,
            "currentDate": capture_date,
            "captureDate": capture_date,
        }

    def _build_hook_context(
        self, conversation: Conversation, session_key: str, messages: List[Message]
    ) -> Dict[str, Any]:
        return {
            "sessionId": self._source_session_id(conversation, session_key, messages),
            "sessionKey": f"agent:main:{self._source_session_id(conversation, session_key, messages)}",
            "agentId": "main",
            "conversationId": conversation.conversation_id,
        }

    @staticmethod
    def _hook_message(message: Message) -> Dict[str, Any]:
        timestamp = _utc_iso(message.timestamp)
        return {
            "role": OpenClawSessionMemoryAdapter._openclaw_message_role(message),
            "content": [{"type": "text", "text": OpenClawSessionMemoryAdapter._openclaw_message_text(message)}],
            "timestamp": timestamp,
            "metadata": dict(message.metadata or {}),
        }

    def _run_hook_runner(self, payload: Dict[str, Any], label: str) -> Dict[str, Any]:
        self.hook_payload_dir.mkdir(parents=True, exist_ok=True)
        self.hook_result_dir.mkdir(parents=True, exist_ok=True)
        payload_path = self.hook_payload_dir / f"{_safe_component(label)}-{uuid.uuid4().hex[:8]}.json"
        result_path = self.hook_result_dir / f"{payload_path.stem}.json"
        self._write_json_private(payload_path, payload)
        env = {**os.environ, "OPENCLAW_ROOT": str(self.openclaw_root), **{k: str(v) for k, v in self.plugin_env.items()}}
        env.setdefault("OPENCLAW_PLUGIN_EVAL_HOOK_RUNNER", "1")
        proc = subprocess.run(
            ["node", str(self.hook_runner_path), str(payload_path)],
            cwd=str(self.openclaw_root),
            env=env,
            text=True,
            capture_output=True,
            timeout=self.plugin_timeout_seconds,
        )
        result = self._parse_json_output(
            {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
        )
        if not isinstance(result, dict):
            result = {"ok": False, "raw": result}
        result.update(
            {
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
                "artifact_path": str(result_path),
            }
        )
        self._write_json(result_path, result)
        if proc.returncode != 0 or not result.get("ok"):
            raise RuntimeError(f"OpenClaw plugin hook runner failed: {json.dumps(result, ensure_ascii=False)[:4000]}")
        return result

    def _build_plugin_manifest_record(
        self,
        *,
        conversation: Conversation,
        session_key: str,
        messages: List[Message],
        hook_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_session_id = self._source_session_id(conversation, session_key, messages)
        source_unit_ids = [
            str(message.metadata.get("source_unit_id") or message.metadata.get("dia_id") or "")
            for message in messages
            if message.metadata.get("source_unit_id") or message.metadata.get("dia_id")
        ]
        hook_ok = self._hook_result_succeeded(hook_result)
        hook_errors = self._hook_result_errors(hook_result)
        memory_ref = {
            "provider": self.provider_name,
            "plugin_id": self.plugin_id,
            "hook_name": self.add_hook_name,
            "conversation_id": conversation.conversation_id,
            "session_id": session_key,
            "source_session_id": source_session_id,
            "message_count": len(messages),
            "content": "\n".join(self._message_content(message) for message in messages),
            "fetch_records": hook_result.get("fetch_records") or [],
            "hook_runner_artifact": hook_result.get("artifact_path"),
        }
        record = ImportManifestRecord(
            run_id=self._run_id(),
            system_id=self._system_id(),
            conversation_id=conversation.conversation_id,
            view_id="shared",
            chunk_id=f"{conversation.conversation_id}:shared:{session_key}",
            source_unit_ids=source_unit_ids,
            write_request_summary={
                "provider": self.provider_name,
                "transport": "openclaw_plugin_hook",
                "plugin_id": self.plugin_id,
                "hook_name": self.add_hook_name,
                "session_id": session_key,
                "source_session_id": source_session_id,
                "message_count": len(messages),
                "fetch_records": hook_result.get("fetch_records") or [],
            },
            write_receipt={
                "provider_status": "submitted" if hook_ok else "error",
                "hook_runner_artifact": hook_result.get("artifact_path"),
                "calls": hook_result.get("calls") or [],
                "fetch_records": hook_result.get("fetch_records") or [],
                "namespace_scope": {
                    "run_id": self._run_id(),
                    "system_id": self._system_id(),
                    "dataset_id": self._dataset_id(),
                    "conversation_id": conversation.conversation_id,
                    "namespace_id": self._namespace_id(conversation.conversation_id),
                    "view_id": "shared",
                },
            },
            memory_refs=[memory_ref],
            write_status="submitted" if hook_ok else "error",
            errors=hook_errors,
        )
        row = dataclass_to_dict(record)
        row.update(
            {
                "provider": self.provider_name,
                "plugin_id": self.plugin_id,
                "hook_name": self.add_hook_name,
                "session_id": session_key,
                "source_session_id": source_session_id,
                "fetch_records": hook_result.get("fetch_records") or [],
            }
        )
        return row

    @classmethod
    def _hook_result_succeeded(cls, hook_result: Dict[str, Any]) -> bool:
        if not hook_result.get("ok"):
            return False
        saw_call = False
        for call in hook_result.get("calls") or []:
            if not isinstance(call, dict):
                continue
            saw_call = True
            result = call.get("result")
            if isinstance(result, dict) and result.get("ok") is False:
                return False
        if not saw_call:
            for record in hook_result.get("fetch_records") or []:
                if isinstance(record, dict) and record.get("ok") is False:
                    return False
        return True

    @classmethod
    def _hook_result_errors(cls, hook_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        if not hook_result.get("ok"):
            errors.append(
                {
                    "stage": "plugin_hook",
                    "error": hook_result.get("error", "plugin hook failed"),
                }
            )
        for call in hook_result.get("calls") or []:
            if not isinstance(call, dict):
                continue
            result = call.get("result")
            if isinstance(result, dict) and result.get("ok") is False:
                errors.append(
                    {
                        "stage": "plugin_hook_call",
                        "hook_name": call.get("hook_name"),
                        "turn_index": call.get("turn_index"),
                        "error": result.get("error") or "plugin hook returned ok=false",
                    }
                )
        if errors or not hook_result.get("calls"):
            for record in hook_result.get("fetch_records") or []:
                if not (isinstance(record, dict) and record.get("ok") is False):
                    continue
                errors.append(
                    {
                        "stage": "plugin_fetch",
                        "method": record.get("method"),
                        "url": record.get("url"),
                        "status": record.get("status"),
                        "error": record.get("error")
                        or record.get("response_payload_summary")
                        or "plugin fetch failed",
                    }
                )
        return errors

    async def get_memory(self, memory_ref: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        del kwargs
        return {
            "memory_ref": memory_ref,
            "storage_kind": f"{self.provider_name}_hook_artifact",
            "content": str(memory_ref.get("content") or ""),
            "metadata": {
                "provider": self.provider_name,
                "plugin_id": memory_ref.get("plugin_id"),
                "conversation_id": memory_ref.get("conversation_id"),
                "source_session_id": memory_ref.get("source_session_id"),
            },
            "status": "ok" if memory_ref.get("content") else "empty",
            "errors": [],
        }

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": self.adapter_id,
            "config": self.config,
            "runtime": {
                "openclaw_cli": self.openclaw_cli,
                "openclaw_root": str(self.openclaw_root),
                "plugin_id": self.plugin_id,
                "plugin_path": str(self.plugin_path),
                "hook_runner_path": str(self.hook_runner_path),
                "runtime_root": str(self.run_root),
            },
        }

    def _system_id(self) -> str:
        return str((self.run_context or {}).get("system_id") or self.config.get("name") or self.adapter_id)

    def _run_id(self) -> str:
        return str((self.run_context or {}).get("run_id") or "")

    def _dataset_id(self) -> str:
        return str((self.run_context or {}).get("dataset_id") or self.config.get("dataset_name") or "")

    def _namespace_id(self, conversation_id: str) -> str:
        run_id = _safe_component(self._run_id() or self.adapter_id)
        return f"{run_id}:{_safe_component(conversation_id)}"

    def _provider_namespace_id(
        self, conversation_id: str, *, speaker: str = "user"
    ) -> str:
        return self._namespace_id(conversation_id)

    def _hook_openclaw_config(self, runtime: OpenClawRuntime) -> Dict[str, Any]:
        return self._derive_openclaw_config({}, runtime)

    @staticmethod
    def _source_session_id(conversation: Conversation, session_key: str, messages: List[Message]) -> str:
        del conversation
        if messages:
            return str(messages[0].metadata.get("source_session_id") or session_key)
        return session_key

    @staticmethod
    def _message_content(message: Message) -> str:
        text = str(message.content or "")
        sender = str(message.sender_name or "").strip()
        return f"{sender}: {text}" if sender and sender.lower() not in {"user", "assistant"} else text

    @staticmethod
    def _status_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in rows:
            status = str(row.get("write_status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

    @staticmethod
    def _target_session_ids_from_dataset(dataset: Any) -> List[str]:
        if dataset is None:
            return []
        seen: set[str] = set()
        values: List[str] = []
        for qa in getattr(dataset, "qa_pairs", []) or []:
            metadata = getattr(qa, "metadata", {}) or {}
            raw = metadata.get("session_ids") or []
            if isinstance(raw, (str, int, float)):
                raw_values = [raw]
            else:
                raw_values = list(raw) if isinstance(raw, list) else []
            for item in raw_values:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                values.append(text)
        return values

    def _rows_for_target_sessions(
        self, rows: List[Dict[str, Any]], target_session_ids: List[str]
    ) -> List[Dict[str, Any]]:
        if not target_session_ids:
            return rows
        return [row for row in rows if self._row_is_targeted(row, target_session_ids)]

    @staticmethod
    def _row_is_targeted(row: Dict[str, Any], target_session_ids: List[str]) -> bool:
        if not target_session_ids:
            return True
        target_set = set(target_session_ids)
        candidates = {
            str(row.get("session_id") or ""),
            str(row.get("source_session_id") or ""),
        }
        for ref in row.get("memory_refs") or []:
            if not isinstance(ref, dict):
                continue
            candidates.add(str(ref.get("session_id") or ""))
            candidates.add(str(ref.get("source_session_id") or ""))
        return bool(
            target_set.intersection(candidate for candidate in candidates if candidate)
        )


class OpenClawContextEnginePluginAdapterBase(OpenClawPluginAdapterBase):
    """Base for ContextEngine-style OpenClaw plugins."""

    plugin_kind = "context_engine"
    hook_mode = "context_engine"
    add_hook_name = "afterTurn"
    recall_hook_name = "assemble"

    @property
    def context_engine_capture_strategy(self) -> str:
        return str(
            (self.config.get("plugin", {}) or {}).get(
                "capture_strategy", "incremental_turns"
            )
            or "incremental_turns"
        ).strip().lower()

    def _run_add_hook(
        self,
        conversation: Conversation,
        session_key: str,
        messages: List[Message],
        runtime: OpenClawRuntime,
    ) -> Dict[str, Any]:
        hook_messages = [self._hook_message(message) for message in messages]
        if self.context_engine_capture_strategy == "full_session":
            turns = [
                {
                    "turnIndex": 0,
                    "messages": hook_messages,
                    "prePromptMessageCount": None,
                    "captureStrategy": "full_session",
                }
            ]
        else:
            turns = [
                {
                    "turnIndex": index,
                    "messages": hook_messages[: index + 1],
                    "prePromptMessageCount": index,
                }
                for index in range(len(hook_messages))
            ]
        payload = {
            "mode": "context_engine",
            "action": "add",
            "pluginId": self.plugin_id,
            "pluginPath": str(self.plugin_path),
            "pluginConfig": self._build_plugin_config(
                conversation.conversation_id, add_enabled=True, recall_enabled=False
            ),
            "openclawConfig": self._hook_openclaw_config(runtime),
            "event": {"messages": hook_messages},
            "messages": hook_messages,
            "turns": turns or [{"turnIndex": 0, "messages": hook_messages, "prePromptMessageCount": None}],
            "ctx": self._build_hook_context(conversation, session_key, messages),
            "stateDir": str(runtime.state_dir),
            "workspaceDir": str(runtime.workspace_dir),
        }
        return self._run_hook_runner(payload, f"{conversation.conversation_id}-{session_key}-afterTurn")
