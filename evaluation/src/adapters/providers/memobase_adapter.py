"""Memobase REST adapter for the evaluation framework."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter
from time import sleep
from typing import Any, Dict, Iterable, List, Optional

import requests
from rich.console import Console

from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.shared.online_base import OnlineAPIAdapter
from evaluation.src.adapters.shared.subtlememory_prompts import (
    get_subtlememory_unified_answer_prompt,
)
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.core.readback import (
    NormalizedStorageObject,
    StorageReadbackResult,
)


@register_adapter("memobase")
class MemobaseAdapter(OnlineAPIAdapter):
    """Adapter for Memobase's REST API.

    This intentionally uses ``requests`` directly instead of the Memobase SDK so
    the evaluation harness controls request payloads, audit fields, and readback
    pagination.
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        api_url = config.get("api_url") or config.get("project_url") or ""
        if not api_url:
            raise ValueError("Memobase API URL is required. Set 'api_url' in config.")
        self.api_url = self._normalize_api_url(api_url)

        api_key = config.get("api_key", "")
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        self.batch_size = int(config.get("batch_size", 20))
        self.max_retries = int(config.get("max_retries", 3))
        self.request_timeout_seconds = float(
            config.get("request_timeout_seconds", 60)
        )
        self.add_wait_process = self._coerce_bool(
            config.get("add_wait_process", False)
        )
        self.incremental_flush_every_chunks = int(
            config.get("incremental_flush_every_chunks", 0) or 0
        )
        self.incremental_flush_budget_seconds = int(
            config.get("incremental_flush_budget_seconds", 0) or 0
        )
        self.incremental_flush_poll_interval_seconds = float(
            config.get("incremental_flush_poll_interval_seconds", 0) or 0
        )
        search_config = config.get("search") or {}
        self.search_backend = (
            str(search_config.get("backend", "context")).strip().lower().replace("-", "_")
        )
        self.context_max_token_size = int(
            search_config.get("context_max_token_size", 3000)
        )
        self.event_similarity_threshold = float(
            search_config.get("event_similarity_threshold", 0.2)
        )
        self.fill_window_with_events = self._coerce_bool(
            search_config.get("fill_window_with_events", True)
        )
        self.profile_event_ratio = self._optional_float(
            search_config.get("profile_event_ratio")
        )
        self.full_profile_and_only_search_event = self._optional_bool(
            search_config.get("full_profile_and_only_search_event")
        )
        self.time_range_in_days = self._optional_int(
            search_config.get("time_range_in_days")
        )
        self.require_event_summary = self._optional_bool(
            search_config.get("require_event_summary")
        )
        self.readback_include_profiles = self._coerce_bool(
            search_config.get("readback_include_profiles", True)
        )
        self.event_gist_similarity_threshold = float(
            search_config.get(
                "event_gist_similarity_threshold",
                search_config.get("event_similarity_threshold", 0.2),
            )
        )
        self.console = Console()
        self._readback_user_cache: Dict[str, Dict[str, Any]] = {}
        self._ensured_user_ids: set[str] = set()

        print(f"   API URL: {self.api_url}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Add wait_process: {self.add_wait_process}")

    @staticmethod
    def _normalize_api_url(api_url: str) -> str:
        base = str(api_url).rstrip("/")
        if not base.endswith("/api/v1"):
            base = f"{base}/api/v1"
        return base

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        return bool(value)

    @classmethod
    def _optional_bool(cls, value: Any) -> Optional[bool]:
        if value is None or str(value).strip() == "":
            return None
        return cls._coerce_bool(value)

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None or str(value).strip() == "":
            return None
        return float(value)

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None or str(value).strip() == "":
            return None
        return int(value)

    @staticmethod
    def _string_to_uuid(value: str, salt: str = "memobase_client") -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value) + salt))

    def _memobase_user_id_for_namespace(self, namespace_id: str) -> str:
        if bool(self.config.get("uuid_user_ids", True)):
            return self._string_to_uuid(namespace_id)
        return namespace_id

    def _build_namespace_scope(
        self, conversation: Conversation, speaker: str = "speaker_a"
    ) -> Dict[str, Any]:
        scope = super()._build_namespace_scope(conversation, speaker=speaker)
        logical_id = str(scope.get("namespace_id") or "")
        user_id = self._memobase_user_id_for_namespace(logical_id)
        scope["logical_namespace_id"] = logical_id
        scope["namespace_id"] = user_id
        scope["memobase_user_id"] = user_id
        namespace_cache = getattr(self, "_namespace_cache", {})
        namespace_cache[(conversation.conversation_id, speaker)] = user_id
        self._namespace_cache = namespace_cache
        return scope

    def _extract_user_id(
        self, conversation: Conversation, speaker: str = "speaker_a"
    ) -> str:
        cached_namespace_id = self._namespace_cache.get(
            (conversation.conversation_id, speaker)
        )
        if cached_namespace_id:
            return cached_namespace_id
        return self._build_namespace_scope(conversation, speaker)["namespace_id"]

    @staticmethod
    def _dedupe_nonempty(values: Iterable[Any]) -> List[str]:
        result: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if text and text not in seen:
                result.append(text)
                seen.add(text)
        return result

    @staticmethod
    def _message_session_id(message: Any) -> str:
        metadata = getattr(message, "metadata", {}) or {}
        return str(
            metadata.get("source_session_id")
            or metadata.get("session_id")
            or metadata.get("session")
            or ""
        ).strip()

    @staticmethod
    def _message_timestamp(message: Any) -> str:
        timestamp = getattr(message, "timestamp", None)
        if timestamp is None:
            return ""
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return str(timestamp)

    def _get_format_type(self) -> str:
        return "memobase"

    def _conversation_to_messages(
        self,
        conversation: Conversation,
        format_type: str = "basic",
        perspective: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        messages = super()._conversation_to_messages(
            conversation,
            format_type=format_type,
            perspective=perspective,
        )
        for payload, source_message in zip(messages, conversation.messages):
            timestamp = self._message_timestamp(source_message)
            if timestamp:
                payload["created_at"] = timestamp
            alias = str(getattr(source_message, "sender_name", "") or "").strip()
            if alias:
                payload["alias"] = alias
        return messages

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        user_id = self._extract_user_id(conv, speaker=speaker)
        namespace_scope = dict(
            kwargs.get("namespace_scope")
            or self._build_namespace_scope(conv, speaker)
        )
        if "logical_namespace_id" not in namespace_scope:
            namespace_scope["logical_namespace_id"] = namespace_scope.get(
                "namespace_id", user_id
            )
        namespace_scope["namespace_id"] = user_id
        namespace_scope["memobase_user_id"] = user_id
        await asyncio.to_thread(
            self._ensure_user_exists,
            user_id,
            namespace_scope.get("logical_namespace_id"),
        )

        sender_name = conv.metadata.get(speaker, speaker)
        console = getattr(self, "console", Console())
        console.print(
            f"   📤 Adding for {sender_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )

        grouped: Dict[str, List[tuple[Dict[str, Any], Any]]] = {}
        for payload_message, source_message in zip(messages, conv.messages):
            session_id = self._message_session_id(source_message) or conv.conversation_id
            grouped.setdefault(session_id, []).append((payload_message, source_message))

        receipts: List[Dict[str, Any]] = []
        for session_id, pairs in grouped.items():
            for offset in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[offset : offset + self.batch_size]
                batch_messages = [payload for payload, _source in batch_pairs]
                batch_sources = [source for _payload, source in batch_pairs]
                source_unit_ids = self._dedupe_nonempty(
                    [
                        (getattr(source, "metadata", {}) or {}).get("source_unit_id")
                        for source in batch_sources
                    ]
                )
                fields = {
                    "session_id": session_id,
                    "source_session_id": session_id,
                    "conversation_id": conv.conversation_id,
                    "speaker_view": speaker,
                    "source_unit_ids": source_unit_ids,
                    "run_id": str(self.run_context.get("run_id", "")),
                }
                payload = {
                    "blob_type": "chat",
                    "blob_data": {"messages": batch_messages},
                    "fields": fields,
                }
                response = await asyncio.to_thread(
                    self._post_json,
                    f"/blobs/insert/{user_id}",
                    payload,
                    {"wait_process": self.add_wait_process},
                )
                blob_id = str(response.get("data", {}).get("id") or "").strip()
                status = (
                    self._extract_provider_status(response)
                    if self.add_wait_process
                    else "submitted"
                )
                chunk_id = (
                    f"{conv.conversation_id}:{speaker}:{session_id}:"
                    f"chunk{offset // self.batch_size}"
                )
                memory_ref = {
                    "provider": "memobase",
                    "user_id": user_id,
                    "logical_namespace_id": namespace_scope.get(
                        "logical_namespace_id"
                    ),
                    "blob_id": blob_id,
                    "session_id": session_id,
                    "source_session_id": session_id,
                    "conversation_id": conv.conversation_id,
                    "source_unit_ids": source_unit_ids,
                }
                receipts.append(
                    {
                        "chunk_id": chunk_id,
                        "namespace_scope": namespace_scope,
                        "source_unit_ids": source_unit_ids,
                        "provider_receipt": response,
                        "provider_status": status,
                        "memory_refs": [memory_ref],
                    }
                )
                if self._should_incrementally_flush(len(receipts)):
                    await self._flush_and_wait_for_user_ready(user_id)
            if self.add_wait_process:
                await self._flush_user(user_id, wait_process=True)
        return receipts

    def _should_incrementally_flush(self, receipt_count: int) -> bool:
        interval = int(getattr(self, "incremental_flush_every_chunks", 0) or 0)
        return (
            interval > 0
            and not self.add_wait_process
            and receipt_count > 0
            and receipt_count % interval == 0
        )

    async def _flush_and_wait_for_user_ready(self, user_id: str) -> None:
        await self._flush_user(user_id, wait_process=False)
        await self._wait_for_user_ready(
            user_id,
            budget_seconds=getattr(self, "incremental_flush_budget_seconds", 0),
            poll_interval_seconds=getattr(
                self, "incremental_flush_poll_interval_seconds", 0
            ),
        )

    async def _flush_user(self, user_id: str, *, wait_process: bool) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._post_without_json,
            f"/users/buffer/{user_id}/chat",
            {"wait_process": wait_process},
        )

    def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._request_with_retries(
            "post",
            path,
            json=payload,
            params=params,
        )

    def _post_without_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._request_with_retries(
            "post",
            path,
            params=params,
        )

    def _ensure_user_exists(self, user_id: str, logical_namespace_id: Any = None) -> None:
        if user_id in getattr(self, "_ensured_user_ids", set()):
            return
        try:
            self._get_json(f"/users/{user_id}")
        except Exception:
            self._post_json(
                "/users",
                {
                    "data": {
                        "logical_namespace_id": str(logical_namespace_id or user_id),
                        "system": "memobase",
                    },
                    "id": user_id,
                },
            )
        ensured = getattr(self, "_ensured_user_ids", set())
        ensured.add(user_id)
        self._ensured_user_ids = ensured

    def _get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._request_with_retries(
            "get",
            path,
            params=params,
        )

    def _request_with_retries(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        attempts = max(1, int(getattr(self, "max_retries", 1) or 1))
        last_error: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                request = getattr(self.session, method)
                kwargs: Dict[str, Any] = {
                    "params": params,
                    "timeout": self.request_timeout_seconds,
                }
                if json is not None:
                    kwargs["json"] = json
                response = request(f"{self.api_url}{path}", **kwargs)
                return self._parse_response(response)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    raise
                sleep(1)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Memobase request retry loop exited unexpectedly")

    def _get_ids(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[str]:
        payload = self._get_json(path, params)
        ids = (payload.get("data", {}) or {}).get("ids", []) or []
        return [str(item) for item in ids if item]

    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Memobase response: {payload!r}")
        errno = payload.get("errno", 0)
        if errno not in (0, None):
            message = payload.get("errmsg") or payload.get("message") or payload
            raise RuntimeError(f"Memobase API error: {message}")
        return payload

    @classmethod
    def _extract_provider_status(cls, payload: Dict[str, Any]) -> str:
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        candidates: List[Any] = [data.get("status"), payload.get("status")]
        for chat_result in data.get("chat_results", []) or []:
            if isinstance(chat_result, dict):
                candidates.extend(
                    [
                        chat_result.get("status"),
                        chat_result.get("state"),
                        chat_result.get("processing_status"),
                    ]
                )
        for candidate in candidates:
            normalized = cls._normalize_status(candidate, default="")
            if normalized:
                return normalized
        return "completed"

    @staticmethod
    def _normalize_status(value: Any, default: str = "submitted") -> str:
        text = str(value or "").strip().lower()
        if not text:
            return default
        aliases = {
            "done": "completed",
            "complete": "completed",
            "success": "completed",
            "succeeded": "completed",
            "ready": "completed",
            "running": "processing",
            "pending": "pending",
            "queued": "pending",
            "submitted": "submitted",
        }
        return aliases.get(text, text)

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        del conversation_id, kwargs
        backend = getattr(self, "search_backend", "context")
        if backend == "context":
            return await self._search_context_backend(query, user_id)
        if backend == "event_gist":
            return await self._search_event_gist_backend(query, user_id, top_k)
        if backend == "hybrid_context_event_gist":
            context_results, event_gist_results = await asyncio.gather(
                self._search_context_backend(query, user_id),
                self._search_event_gist_backend(query, user_id, top_k),
            )
            return context_results + event_gist_results
        raise ValueError(
            "Unsupported Memobase search.backend "
            f"{backend!r}; expected context, event_gist, or hybrid_context_event_gist."
        )

    async def _search_context_backend(
        self, query: str, user_id: str
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "max_token_size": getattr(self, "context_max_token_size", 3000),
            "event_similarity_threshold": getattr(
                self, "event_similarity_threshold", 0.2
            ),
            "fill_window_with_events": getattr(
                self, "fill_window_with_events", True
            ),
        }
        if query:
            params["chats"] = [{"role": "user", "content": query}]
        context = await asyncio.to_thread(
            self._get_context_for_user,
            user_id,
            params,
        )
        if not context:
            return []
        return [
            {
                "content": context,
                "score": 1.0,
                "user_id": user_id,
                "metadata": {"source": "memobase_context"},
            }
        ]

    async def _search_event_gist_backend(
        self, query: str, user_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_event_gists_for_user,
            user_id,
            query,
            top_k,
        )

    def _get_context_for_user(self, user_id: str, params: Dict[str, Any]) -> str:
        request_params: Dict[str, Any] = {
            "max_token_size": params.get("max_token_size")
        }
        chats = params.get("chats")
        if chats:
            request_params["chats_str"] = json.dumps(chats, ensure_ascii=False)
        if params.get("event_similarity_threshold") is not None:
            request_params["event_similarity_threshold"] = params[
                "event_similarity_threshold"
            ]
        if params.get("fill_window_with_events") is not None:
            request_params["fill_window_with_events"] = (
                "true"
                if self._coerce_bool(params.get("fill_window_with_events"))
                else "false"
            )
        if getattr(self, "profile_event_ratio", None) is not None:
            request_params["profile_event_ratio"] = self.profile_event_ratio
        if getattr(self, "full_profile_and_only_search_event", None) is not None:
            request_params["full_profile_and_only_search_event"] = (
                "true" if self.full_profile_and_only_search_event else "false"
            )
        if getattr(self, "time_range_in_days", None) is not None:
            request_params["time_range_in_days"] = self.time_range_in_days
        if getattr(self, "require_event_summary", None) is not None:
            request_params["require_event_summary"] = (
                "true" if self.require_event_summary else "false"
            )
        payload = self._get_json(f"/users/context/{user_id}", request_params)
        data = payload.get("data", {}) or {}
        context = data.get("context", data if isinstance(data, str) else "")
        return str(context or "")

    def _search_event_gists_for_user(
        self, user_id: str, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        request_params: Dict[str, Any] = {
            "query": query,
            "topk": top_k,
            "similarity_threshold": getattr(
                self, "event_gist_similarity_threshold", 0.2
            ),
        }
        if getattr(self, "time_range_in_days", None) is not None:
            request_params["time_range_in_days"] = self.time_range_in_days
        payload = self._get_json(
            f"/users/event_gist/search/{user_id}", request_params
        )
        data = payload.get("data", {}) or {}
        gists = data.get("gists", []) if isinstance(data, dict) else []
        results: List[Dict[str, Any]] = []
        for gist in gists or []:
            if not isinstance(gist, dict):
                continue
            gist_data = gist.get("gist_data", {}) or {}
            content = (
                gist_data.get("content")
                if isinstance(gist_data, dict)
                else str(gist_data or "")
            )
            content = str(content or "").strip()
            if not content:
                continue
            results.append(
                {
                    "content": content,
                    "score": float(gist.get("similarity") or gist.get("score") or 0.0),
                    "user_id": user_id,
                    "metadata": {
                        "source": "memobase_event_gist",
                        "gist_id": gist.get("id"),
                        "created_at": gist.get("created_at"),
                        "updated_at": gist.get("updated_at"),
                    },
                }
            )
        return results

    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        backend = getattr(self, "search_backend", "context")
        formatted_context = self._format_backend_context(results, backend=backend)
        context_count = self._count_results_by_source(results, "memobase_context")
        event_gist_count = self._count_results_by_source(
            results, "memobase_event_gist"
        )
        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "memobase",
                "search_mode": "api",
                "search_backend": backend,
                "leakage_guard": "query_only",
                "formatted_context": formatted_context,
                "top_k": top_k,
                "user_ids": [user_id],
                "context_count": context_count,
                "event_gist_count": event_gist_count,
            },
            retrieval_status="ok" if formatted_context or results else "empty",
        )

    def _build_dual_search_result(
        self,
        query: str,
        conversation_id: str,
        all_results: List[Dict[str, Any]],
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        backend = getattr(self, "search_backend", "context")
        speaker_a_context = (
            self._format_backend_context(results_a, backend=backend)
            or "(No memories found)"
        )
        speaker_b_context = (
            self._format_backend_context(results_b, backend=backend)
            or "(No memories found)"
        )
        template = self._prompts["online_api"].get("templates", {}).get(
            "default", "{speaker_1}:\n{speaker_1_memories}\n{speaker_2}:\n{speaker_2_memories}"
        )
        formatted_context = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_context,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_context,
        )
        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            retrieval_metadata={
                "system": "memobase",
                "search_mode": "api",
                "search_backend": backend,
                "leakage_guard": "query_only",
                "dual_perspective": True,
                "formatted_context": formatted_context,
                "top_k": top_k,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "context_count": self._count_results_by_source(
                    all_results, "memobase_context"
                ),
                "event_gist_count": self._count_results_by_source(
                    all_results, "memobase_event_gist"
                ),
            },
            retrieval_status="ok" if all_results else "empty",
        )

    @staticmethod
    def _count_results_by_source(results: List[Dict[str, Any]], source: str) -> int:
        return sum(
            1
            for item in results
            if ((item.get("metadata") or {}).get("source") == source)
        )

    @classmethod
    def _format_backend_context(
        cls, results: List[Dict[str, Any]], *, backend: str
    ) -> str:
        context_items = [
            item
            for item in results
            if (item.get("metadata") or {}).get("source") == "memobase_context"
        ]
        event_gist_items = [
            item
            for item in results
            if (item.get("metadata") or {}).get("source") == "memobase_event_gist"
        ]

        if backend == "hybrid_context_event_gist":
            parts: List[str] = []
            context_text = cls._join_result_contents(context_items)
            event_gist_text = cls._format_event_gists(event_gist_items)
            if context_text:
                parts.append(f"# Memobase Context\n{context_text}")
            if event_gist_text:
                parts.append(f"# Memobase Event Gists\n{event_gist_text}")
            return "\n\n".join(parts)
        if backend == "event_gist":
            return cls._format_event_gists(event_gist_items)
        return cls._join_result_contents(results)

    @staticmethod
    def _join_result_contents(results: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            str(item.get("content") or "").strip()
            for item in results
            if str(item.get("content") or "").strip()
        )

    @staticmethod
    def _format_event_gists(results: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for idx, item in enumerate(results, start=1):
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            metadata = item.get("metadata") or {}
            gist_id = str(metadata.get("gist_id") or "").strip()
            prefix = f"{idx}. "
            suffix = f" [gist_id={gist_id}]" if gist_id else ""
            lines.append(f"{prefix}{content}{suffix}")
        return "\n".join(lines)

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
        question_metadata = question_metadata or {}
        session_ids = self._dedupe_nonempty(question_metadata.get("session_ids") or [])
        if not session_ids:
            readback = StorageReadbackResult(
                status="unsupported",
                checked_session_ids=[],
                objects=[],
                metadata={
                    "provider": "memobase",
                    "question_id": question_id,
                    "reason": "missing question session_ids",
                },
                errors=[
                    {
                        "error_type": "missing_session_ids",
                        "error_message": (
                            "Readback search requires question_metadata.session_ids."
                        ),
                    }
                ],
            )
            return SearchResult(
                question_id=question_id or "",
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "system": "memobase",
                    "search_mode": "readback",
                    "session_ids": [],
                    "formatted_context": "",
                    "readback": readback.to_dict(),
                },
                retrieval_status="unsupported",
            )

        readback = await self.get_storage_readback(
            session_ids=session_ids,
            question_id=question_id,
            context={
                "conversation_id": conversation_id,
                "session_ids": session_ids,
                "import_manifest_rows": import_manifest_records or [],
                "question_id": question_id,
            },
        )
        results, formatted_context = self._build_readback_search_payload(readback)
        status = "ok"
        if readback.status in {"unsupported", "error"}:
            status = readback.status
        elif not formatted_context and not results:
            status = "empty"

        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "memobase",
                "search_mode": "readback",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "user_ids": readback.metadata.get("user_ids", []),
                "readback": readback.to_dict(),
            },
            retrieval_status=status,
        )

    @staticmethod
    def _build_readback_search_payload(
        readback: StorageReadbackResult,
    ) -> tuple[List[Dict[str, Any]], str]:
        objects = [
            obj for obj in readback.objects if isinstance(obj, NormalizedStorageObject)
        ]
        results: List[Dict[str, Any]] = []
        profile_parts: List[str] = []
        event_gist_parts: List[str] = []
        profile_idx = 0
        event_gist_idx = 0
        for obj in objects:
            content = str(obj.content or "").strip()
            if not content:
                continue
            metadata = obj.metadata or {}
            item_metadata = {
                "blob_id": obj.id,
                "kind": obj.kind,
                "session_id": obj.session_id,
                "source_session_id": metadata.get("source_session_id"),
                "conversation_id": metadata.get("conversation_id"),
                "event_id": metadata.get("event_id"),
                "source_blob_ids": metadata.get("source_blob_ids", []),
                "source_unit_ids": metadata.get("source_unit_ids", []),
            }
            results.append(
                {
                    "content": content,
                    "score": 1.0,
                    "metadata": item_metadata,
                }
            )
            if obj.kind == "memobase_profile":
                profile_idx += 1
                suffix_parts = []
                topic = str(metadata.get("topic") or "").strip()
                sub_topic = str(metadata.get("sub_topic") or "").strip()
                profile_id = str(metadata.get("profile_id") or obj.id or "").strip()
                if topic or sub_topic:
                    suffix_parts.append(f"topic={topic}::{sub_topic}")
                if profile_id:
                    suffix_parts.append(f"profile_id={profile_id}")
                suffix = f" [{'; '.join(suffix_parts)}]" if suffix_parts else ""
                profile_parts.append(f"{profile_idx}. {content}{suffix}")
            elif obj.kind == "memobase_event_gist":
                event_gist_idx += 1
                gist_id = str(metadata.get("gist_id") or obj.id or "").strip()
                suffix = f" [gist_id={gist_id}]" if gist_id else ""
                event_gist_parts.append(f"{event_gist_idx}. {content}{suffix}")

        sections: List[str] = []
        if profile_parts:
            sections.append(f"# Memobase Profile\n{chr(10).join(profile_parts)}")
        if event_gist_parts:
            sections.append(
                f"# Memobase Session Event Gists\n{chr(10).join(event_gist_parts)}"
            )
        formatted_context = "\n\n".join(sections)
        return results, formatted_context

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
        del run_id, evidence_texts, kwargs
        target_session_ids = self._dedupe_nonempty(session_ids or [])
        user_ids = self._find_user_ids_for_readback(user_id=user_id, context=context)
        if not user_ids:
            return StorageReadbackResult(
                status="unsupported",
                checked_session_ids=target_session_ids,
                objects=[],
                metadata={
                    "provider": "memobase",
                    "question_id": question_id,
                    "reason": "missing user_id/namespace",
                },
                errors=[
                    {
                        "error_type": "missing_user_id",
                        "error_message": "Cannot determine Memobase user_id for readback.",
                    }
                ],
            )

        all_objects: List[NormalizedStorageObject] = []
        errors: List[Dict[str, Any]] = []
        for candidate_user_id in user_ids:
            try:
                user_payload = await asyncio.to_thread(
                    self._get_cached_profile_and_session_gists_for_user,
                    candidate_user_id,
                    target_session_ids,
                )
                for obj in user_payload.get("objects", []) or []:
                    all_objects.append(obj)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "stage": "get_storage_readback",
                        "user_id": candidate_user_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

        object_sessions = {
            obj.session_id for obj in all_objects if obj.session_id and obj.session_id != "unknown"
        }
        missing_session_ids = [
            session_id
            for session_id in target_session_ids
            if session_id not in object_sessions
        ]
        return StorageReadbackResult(
            status="error" if errors and not all_objects else "ok",
            checked_session_ids=target_session_ids,
            missing_session_ids=missing_session_ids,
            objects=all_objects,
            metadata={
                "provider": "memobase",
                "question_id": question_id,
                "user_ids": user_ids,
                "readback_scope": "session" if target_session_ids else "user",
                "profile_count": sum(
                    1 for obj in all_objects if obj.kind == "memobase_profile"
                ),
                "event_gist_count": sum(
                    1 for obj in all_objects if obj.kind == "memobase_event_gist"
                ),
                "object_count": len(all_objects),
                "readback_include_profiles": self.readback_include_profiles,
            },
            errors=errors,
        )

    def _find_user_ids_for_readback(
        self,
        *,
        user_id: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> List[str]:
        candidates: List[Any] = [user_id]
        for row in (context or {}).get("import_manifest_rows", []) or []:
            receipt = row.get("write_receipt") if isinstance(row, dict) else {}
            if isinstance(receipt, dict):
                namespace_scope = receipt.get("namespace_scope") or {}
                if isinstance(namespace_scope, dict):
                    candidates.append(namespace_scope.get("memobase_user_id"))
                    candidates.append(namespace_scope.get("namespace_id"))
                for ref in receipt.get("memory_refs", []) or []:
                    if isinstance(ref, dict):
                        candidates.append(ref.get("user_id"))
            for ref in row.get("memory_refs", []) or []:
                if isinstance(ref, dict):
                    candidates.append(ref.get("user_id"))
        return self._dedupe_nonempty(candidates)

    def _readback_cache_key(self, user_id: str, session_ids: List[str]) -> str:
        profile_flag = "profiles" if self.readback_include_profiles else "gists_only"
        return f"{user_id}::{profile_flag}::{'|'.join(session_ids)}"

    def _get_cached_profile_and_session_gists_for_user(
        self, user_id: str, session_ids: List[str]
    ) -> Dict[str, Any]:
        cache_key = self._readback_cache_key(user_id, session_ids)
        cached = self._readback_user_cache.get(cache_key)
        if cached is not None:
            return cached

        objects: List[NormalizedStorageObject] = []
        if self.readback_include_profiles:
            profile_payload = self._get_json(f"/users/profile/{user_id}")
            for profile in (profile_payload.get("data", {}) or {}).get(
                "profiles", []
            ) or []:
                obj = self._normalize_profile_payload(profile, user_id=user_id)
                if obj is not None:
                    objects.append(obj)

        gist_payload = self._get_json(
            f"/users/event_gist/session_readback/{user_id}",
            {
                "session_ids": session_ids,
                "time_range_in_days": getattr(self, "time_range_in_days", None)
                or 180,
            },
        )

        gist_data = gist_payload.get("data", {}) or {}
        event_by_id = {
            str(event.get("id")): event
            for event in gist_data.get("events", []) or []
            if isinstance(event, dict)
        }
        for gist in gist_data.get("gists", []) or []:
            obj = self._normalize_event_gist_payload(
                gist,
                user_id=user_id,
                target_session_ids=session_ids,
                event_by_id=event_by_id,
            )
            if obj is not None:
                objects.append(obj)

        result = {
            "objects": objects,
        }
        self._readback_user_cache[cache_key] = result
        return result

    @staticmethod
    def _normalize_profile_payload(
        data: Dict[str, Any], *, user_id: str
    ) -> Optional[NormalizedStorageObject]:
        if not isinstance(data, dict):
            return None
        content = str(data.get("content") or "").strip()
        if not content:
            return None
        attributes = data.get("attributes") or {}
        profile_id = str(data.get("id") or "").strip()
        return NormalizedStorageObject(
            session_id="profile",
            kind="memobase_profile",
            id=profile_id,
            content=content,
            metadata={
                "provider": "memobase",
                "user_id": user_id,
                "profile_id": profile_id,
                "scope": "user",
                "session_scoped": False,
                "topic": attributes.get("topic") if isinstance(attributes, dict) else None,
                "sub_topic": (
                    attributes.get("sub_topic") if isinstance(attributes, dict) else None
                ),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            },
            raw=dict(data),
        )

    @classmethod
    def _normalize_event_gist_payload(
        cls,
        data: Dict[str, Any],
        *,
        user_id: str,
        target_session_ids: List[str],
        event_by_id: Dict[str, Dict[str, Any]],
    ) -> Optional[NormalizedStorageObject]:
        if not isinstance(data, dict):
            return None
        gist_data = data.get("gist_data", {}) or {}
        content = (
            gist_data.get("content") if isinstance(gist_data, dict) else str(gist_data)
        )
        content = str(content or "").strip()
        if not content:
            return None
        event_id = str(data.get("event_id") or "").strip()
        event = event_by_id.get(event_id)
        event_match_status = "matched"
        if event is None and not event_id and len(event_by_id) == 1:
            event = next(iter(event_by_id.values()))
            event_match_status = "legacy_single_event_fallback"
        elif event is None and event_id:
            event_match_status = "unmatched_event_id"
        elif event is None:
            event_match_status = "unmatched_missing_event_id"
        event_data = (event or {}).get("event_data", {}) or {}
        source_session_ids = cls._dedupe_nonempty(event_data.get("source_session_ids") or [])
        source_unit_ids = cls._dedupe_nonempty(event_data.get("source_unit_ids") or [])
        source_blob_ids = cls._dedupe_nonempty(event_data.get("source_blob_ids") or [])
        session_id = source_session_ids[0] if source_session_ids else "unknown"
        gist_id = str(data.get("id") or "").strip()
        return NormalizedStorageObject(
            session_id=session_id,
            kind="memobase_event_gist",
            id=gist_id,
            content=content,
            metadata={
                "provider": "memobase",
                "user_id": user_id,
                "gist_id": gist_id,
                "event_id": event_id or (event or {}).get("id"),
                "event_match_status": event_match_status,
                "source_session_id": session_id,
                "source_session_ids": source_session_ids,
                "source_blob_ids": source_blob_ids,
                "source_unit_ids": source_unit_ids,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            },
            raw=dict(data),
        )

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
        del add_result, dataset, kwargs
        flush_results = await self._flush_manifest_users(import_manifest_rows)
        buffer_status = await self._wait_for_manifest_users_ready(
            import_manifest_rows,
            budget_seconds=budget_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        refreshed_rows: List[Dict[str, Any]] = []
        provider_status_counts: Dict[str, int] = {}
        missing_memory_refs: List[str] = []
        ready = bool(buffer_status.get("ready")) and all(
            result.get("status") != "error" for result in flush_results
        )
        for row in import_manifest_rows:
            updated_row = dict(row)
            receipt = dict(updated_row.get("write_receipt", {}) or {})
            status = self._normalize_status(
                updated_row.get("write_status")
                or receipt.get("provider_status")
                or "completed"
            )
            if ready and status == "submitted":
                status = "completed"
            receipt["provider_status"] = status
            updated_row["write_receipt"] = receipt
            updated_row["write_status"] = status
            provider_status_counts[status] = provider_status_counts.get(status, 0) + 1
            if not updated_row.get("memory_refs"):
                missing_memory_refs.append(str(updated_row.get("chunk_id", "")))
            refreshed_rows.append(updated_row)

        return {
            "import_manifest_records": refreshed_rows,
            "ready": ready,
            "status": "finalized" if ready else "processing",
            "provider_status_counts": provider_status_counts,
            "updated_rows": len(refreshed_rows),
            "missing_memory_refs": missing_memory_refs,
            "flush_results": flush_results,
            "buffer_status": buffer_status,
            "warnings": [],
            "finalize_budget_exhausted": not ready,
        }

    async def _flush_manifest_users(
        self,
        import_manifest_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        user_ids = self._user_ids_from_manifest(import_manifest_rows)
        flush_results: List[Dict[str, Any]] = []
        for user_id in user_ids:
            try:
                payload = await asyncio.to_thread(
                    self._post_without_json,
                    f"/users/buffer/{user_id}/chat",
                    {"wait_process": False},
                )
                flush_results.append(
                    {
                        "user_id": user_id,
                        "status": "submitted",
                        "provider_response": payload,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                flush_results.append(
                    {
                        "user_id": user_id,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
        return flush_results

    async def _wait_for_manifest_users_ready(
        self,
        import_manifest_rows: List[Dict[str, Any]],
        *,
        budget_seconds: int = 0,
        poll_interval_seconds: float | None = None,
    ) -> Dict[str, Any]:
        user_ids = self._user_ids_from_manifest(import_manifest_rows)
        poll_interval = 0.0 if poll_interval_seconds is None else float(poll_interval_seconds)
        deadline = perf_counter() + max(0, int(budget_seconds or 0))
        attempts = 0
        last_status: Dict[str, Any] = {
            "ready": not user_ids,
            "users": {},
            "attempts": 0,
        }

        while True:
            attempts += 1
            user_statuses: Dict[str, Any] = {}
            all_ready = True
            for user_id in user_ids:
                status = await asyncio.to_thread(self._get_user_buffer_status, user_id)
                user_statuses[user_id] = status
                if not self._is_user_buffer_ready(status):
                    all_ready = False

            last_status = {
                "ready": all_ready,
                "users": user_statuses,
                "attempts": attempts,
            }
            if all_ready:
                return last_status
            if budget_seconds <= 0 or perf_counter() >= deadline:
                return last_status
            if poll_interval > 0:
                await asyncio.to_thread(sleep, poll_interval)

    async def _wait_for_user_ready(
        self,
        user_id: str,
        *,
        budget_seconds: int = 0,
        poll_interval_seconds: float | None = None,
    ) -> Dict[str, Any]:
        poll_interval = 0.0 if poll_interval_seconds is None else float(poll_interval_seconds)
        deadline = perf_counter() + max(0, int(budget_seconds or 0))
        attempts = 0
        last_status: Dict[str, Any] = {"ready": False, "user_id": user_id, "attempts": 0}
        while True:
            attempts += 1
            status = await asyncio.to_thread(self._get_user_buffer_status, user_id)
            ready = self._is_user_buffer_ready(status)
            last_status = {
                "ready": ready,
                "user_id": user_id,
                "status": status,
                "attempts": attempts,
            }
            if ready:
                return last_status
            if budget_seconds <= 0 or perf_counter() >= deadline:
                return last_status
            if poll_interval > 0:
                await asyncio.to_thread(sleep, poll_interval)

    @staticmethod
    def _is_user_buffer_ready(status: Dict[str, List[str]]) -> bool:
        return not (
            status.get("idle")
            or status.get("processing")
            or status.get("failed")
        )

    def _get_user_buffer_status(self, user_id: str) -> Dict[str, List[str]]:
        statuses: Dict[str, List[str]] = {}
        for status in ("idle", "processing", "failed"):
            statuses[status] = self._get_ids(
                f"/users/buffer/capacity/{user_id}/chat",
                {"status": status},
            )
        return statuses

    def _user_ids_from_manifest(
        self,
        import_manifest_rows: List[Dict[str, Any]],
    ) -> List[str]:
        candidates: List[Any] = []
        for row in import_manifest_rows:
            if not isinstance(row, dict):
                continue
            receipt = row.get("write_receipt") or {}
            if isinstance(receipt, dict):
                namespace_scope = receipt.get("namespace_scope") or {}
                if isinstance(namespace_scope, dict):
                    candidates.append(namespace_scope.get("memobase_user_id"))
                    candidates.append(namespace_scope.get("namespace_id"))
                for ref in receipt.get("memory_refs", []) or []:
                    if isinstance(ref, dict):
                        candidates.append(ref.get("user_id"))
            for ref in row.get("memory_refs", []) or []:
                if isinstance(ref, dict):
                    candidates.append(ref.get("user_id"))
        return self._dedupe_nonempty(candidates)

    def _get_answer_prompt(self) -> str:
        dataset_id = str((self.run_context or {}).get("dataset_id", "")).strip()
        if dataset_id.startswith("subtlememory"):
            return get_subtlememory_unified_answer_prompt(
                self._prompts,
                config=self.config,
            )
        return self._prompts["online_api"]["default"]["answer_prompt_memos"]

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "Memobase",
            "type": "online_api",
            "description": "Memobase REST adapter",
            "adapter": "MemobaseAdapter",
        }
