"""
Mem0 Adapter - adapt Mem0 online API for evaluation framework.
Reference: https://mem0.ai/

Key features:
- Dual-perspective handling: separate storage and retrieval for speaker_a and speaker_b
- Supports custom instructions
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Dict, List, Optional

from rich.console import Console

from evaluation.src.adapters.shared.online_base import OnlineAPIAdapter
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.shared.subtlememory_prompts import (
    get_subtlememory_unified_answer_prompt,
)
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.core.readback import (
    NormalizedStorageObject,
    StorageReadbackResult,
)


@register_adapter("mem0")
class Mem0Adapter(OnlineAPIAdapter):
    """
    Mem0 online API adapter.

    Supports:
    - Standard memory storage and retrieval

    Config example:
    ```yaml
    adapter: "mem0"
    api_key: "${MEM0_API_KEY}"
    batch_size: 2
    display_timezone_offset: 8  # Optional: convert timestamps to UTC+8 for display
    ```
    """

    READY_MEMORY_STATES = {"done", "ready", "completed", "complete", "success"}
    NON_READY_MEMORY_STATES = {
        "pending",
        "queued",
        "processing",
        "in_progress",
        "running",
    }

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # Import Mem0 async client
        try:
            from mem0 import AsyncMemoryClient
        except ImportError:
            raise ImportError(
                "Mem0 client not installed. Please install: pip install mem0ai"
            )

        # Initialize Mem0 async client
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Mem0 API key is required. Set 'api_key' in config.")

        host = config.get("host") or None
        self.client = AsyncMemoryClient(api_key=api_key, host=host)
        self.batch_size = config.get("batch_size", 2)
        self.max_retries = config.get("max_retries", 5)
        self.max_content_length = config.get("max_content_length", 12000)
        self.add_interval = config.get("add_interval", 0.0)
        self.search_interval = config.get("search", {}).get("search_interval", 0.0)
        self.add_event_wait = bool(config.get("add_event_wait", False))
        self.add_event_poll_interval_seconds = float(
            config.get("add_event_poll_interval_seconds", 2)
        )
        self.add_event_poll_timeout_seconds = float(
            config.get("add_event_poll_timeout_seconds", 180)
        )
        add_event_wait_mode = (
            str(config.get("add_event_wait_mode", "inline")).strip().lower()
        )
        self.add_event_wait_mode = {
            "post_add": "deferred",
            "after_add": "deferred",
            "after_all_adds": "deferred",
        }.get(add_event_wait_mode, add_event_wait_mode)
        if self.add_event_wait_mode not in {"inline", "deferred"}:
            raise ValueError(
                "add_event_wait_mode must be either 'inline' or 'deferred'"
            )
        self.add_event_wait_num_workers = max(
            int(config.get("add_event_wait_num_workers", self.num_workers * 4)), 1
        )
        self.post_add_wait_strategy = config.get(
            "post_add_wait_strategy", "fixed_sleep"
        )
        self.post_add_poll_interval_seconds = config.get(
            "post_add_poll_interval_seconds", 15
        )
        self.post_add_stable_polls = max(int(config.get("post_add_stable_polls", 2)), 1)
        self._post_add_wait_user_ids: set[str] = set()
        self._readback_session_cache: Dict[str, Dict[str, Any]] = {}
        self._readback_session_locks: Dict[str, asyncio.Lock] = {}
        self._readback_request_lock = asyncio.Lock()
        self._readback_next_allowed_at = 0.0
        self.console = Console()

        print(f"   Batch Size: {self.batch_size}")
        print(f"   Max Content Length: {self.max_content_length}")
        if self.add_interval > 0:
            print(f"   Add Interval: {self.add_interval}s (rate limiting)")
        if self.search_interval > 0:
            print(f"   Search Interval: {self.search_interval}s (rate limiting)")
        if self.add_event_wait:
            print(
                f"   Add Event Wait: enabled "
                f"(mode={self.add_event_wait_mode}, "
                f"timeout={self.add_event_poll_timeout_seconds}s, "
                f"interval={self.add_event_poll_interval_seconds}s, "
                f"workers={self.add_event_wait_num_workers})"
            )
        if self.post_add_wait_strategy != "fixed_sleep":
            print(f"   Post-Add Wait Strategy: {self.post_add_wait_strategy}")

    def _convert_timestamp_to_display_timezone(self, timestamp_str: str) -> str:
        """
        Convert mem0's created_at timestamp to display timezone.

        Default behavior (if display_timezone_offset not set):
        - Convert to system local timezone for display only. Add-stage naive timestamps
          are treated as UTC when constructing Mem0 request timestamps.

        Optional behavior (if display_timezone_offset is set):
        - Convert to specified timezone (e.g., UTC for explicit UTC handling)

        Args:
            timestamp_str: ISO format timestamp string with timezone (e.g., "2023-05-07T22:56:00-07:00")

        Returns:
            Formatted timestamp string in display timezone or original if conversion fails
        """
        if not timestamp_str:
            return timestamp_str

        try:
            # Parse ISO format timestamp (with timezone)
            dt = datetime.fromisoformat(timestamp_str)

            dt_display = dt.astimezone(None)

            # Format as readable string (YYYY-MM-DD HH:MM:SS)
            return dt_display.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            # If conversion fails, return original string
            self.console.print(
                f"⚠️  Failed to convert timestamp '{timestamp_str}': {e}", style="yellow"
            )
            return timestamp_str

    def _format_metadata_timestamp(self, timestamp: datetime) -> str:
        """
        Format a conversation timestamp in the same human-readable style used by
        the official Mem0 LoCoMo evaluation.

        Example: "6:07 pm on 13 January, 2023"
        """
        hour = timestamp.strftime("%I").lstrip("0") or "0"
        minute = timestamp.strftime("%M")
        meridiem = timestamp.strftime("%p").lower()
        month = timestamp.strftime("%B")
        return (
            f"{hour}:{minute} {meridiem} on {timestamp.day} {month}, {timestamp.year}"
        )

    @staticmethod
    def _to_utc_unix_seconds(timestamp: datetime) -> int:
        """Convert a message timestamp to reproducible UTC Unix seconds."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        return int(timestamp.timestamp())

    def _iter_session_batches(
        self, messages: List[Dict[str, Any]], source_messages: List[Any]
    ):
        """Yield add batches without crossing LoCoMo session boundaries."""
        current_messages: List[Dict[str, Any]] = []
        current_sources: List[Any] = []
        current_session = None

        for idx, message in enumerate(messages):
            source_message = source_messages[idx] if idx < len(source_messages) else None
            session = (
                source_message.metadata.get("session")
                if source_message and source_message.metadata
                else None
            )

            if current_messages and (
                len(current_messages) >= self.batch_size or session != current_session
            ):
                yield current_messages, current_sources
                current_messages = []
                current_sources = []

            if not current_messages:
                current_session = session
            current_messages.append(message)
            current_sources.append(source_message)

        if current_messages:
            yield current_messages, current_sources

    @staticmethod
    def _extract_batch_session_id(source_messages: List[Any]) -> str:
        """Return one source session id when a batch maps cleanly to one session."""
        session_ids = []
        for source_message in source_messages:
            metadata = source_message.metadata if source_message else None
            if not metadata:
                continue
            session_id = str(metadata.get("source_session_id") or "").strip()
            if session_id and session_id not in session_ids:
                session_ids.append(session_id)
        return session_ids[0] if len(session_ids) == 1 else ""

    @staticmethod
    def _build_session_filters(user_id: str, run_id: str) -> Dict[str, Any]:
        return {"AND": [{"user_id": user_id}, {"run_id": run_id}]}

    @staticmethod
    def _get_memory_text(memory: Dict[str, Any]) -> str:
        return memory.get("memory") or memory.get("text") or memory.get("data") or ""

    @staticmethod
    def _get_memory_timestamp(memory: Dict[str, Any]) -> str:
        metadata = memory.get("metadata") or {}
        return (
            metadata.get("timestamp")
            or memory.get("timestamp")
            or memory.get("created_at")
            or ""
        )

    def _uses_adapter_post_add_wait(self) -> bool:
        """Whether Mem0 handles post-add readiness checks inside the adapter."""
        return self.post_add_wait_strategy == "adapter_poll"

    @staticmethod
    def _extract_event_id_from_add_response(response: Any) -> str:
        if not isinstance(response, dict):
            return ""
        return str(response.get("event_id") or response.get("id") or "").strip()

    @staticmethod
    def _merge_add_response_with_event(
        response: Dict[str, Any], event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged = dict(response)
        merged["event"] = event_data
        if event_data.get("status"):
            merged["status"] = event_data["status"]
        if "results" in event_data and "results" not in merged:
            merged["results"] = event_data["results"]
        return merged

    async def _get_add_event_status(self, event_id: str) -> Dict[str, Any]:
        response = await self.client.async_client.get(f"/v1/event/{event_id}/")
        response.raise_for_status()
        result = response.json()
        return result if isinstance(result, dict) else {"result": result}

    async def _wait_for_add_event_response(self, response: Any) -> Any:
        """Wait for a Mem0 v3 add event and fail fast on failed events."""
        event_id = self._extract_event_id_from_add_response(response)
        if not event_id or not isinstance(response, dict):
            return response

        start_time = monotonic()
        last_status = "UNKNOWN"
        while monotonic() - start_time < self.add_event_poll_timeout_seconds:
            event_data = await self._get_add_event_status(event_id)
            last_status = str(event_data.get("status") or "UNKNOWN").upper()

            if last_status == "SUCCEEDED":
                return self._merge_add_response_with_event(response, event_data)
            if last_status in {"FAILED", "ERROR", "CANCELLED", "CANCELED"}:
                error_message = (
                    event_data.get("error")
                    or event_data.get("message")
                    or event_data.get("detail")
                    or last_status
                )
                raise RuntimeError(
                    f"Mem0 add event {event_id} failed: {error_message}"
                )

            await asyncio.sleep(self.add_event_poll_interval_seconds)

        raise TimeoutError(
            f"Mem0 add event {event_id} did not finish within "
            f"{self.add_event_poll_timeout_seconds}s; last status={last_status}"
        )

    async def _wait_for_add_event(self, response: Any) -> Any:
        """Wait for a Mem0 v3 add event immediately when inline mode is selected."""
        if not self.add_event_wait or self._effective_add_event_wait_mode() != "inline":
            return response
        return await self._wait_for_add_event_response(response)

    def _effective_add_event_wait_mode(self) -> str:
        """Return add-event wait mode with backward-compatible inline default."""
        raw_mode = getattr(self, "add_event_wait_mode", None)
        if raw_mode in (None, ""):
            raw_mode = self.config.get("add_event_wait_mode", "inline")
        mode = str(raw_mode).strip().lower()
        return {
            "post_add": "deferred",
            "after_add": "deferred",
            "after_all_adds": "deferred",
        }.get(mode, mode or "inline")

    def _event_id_from_manifest_row(self, row: Dict[str, Any]) -> str:
        receipt = row.get("write_receipt") or {}
        event_id = self._extract_event_id_from_add_response(
            receipt.get("provider_receipt") or {}
        )
        if event_id:
            return event_id

        for ref in row.get("memory_refs") or receipt.get("memory_refs") or []:
            if isinstance(ref, dict) and ref.get("event_id"):
                return str(ref.get("event_id")).strip()
        return ""

    @staticmethod
    def _first_ref_value(refs: List[Dict[str, Any]], key: str, default: Any = "") -> Any:
        for ref in refs:
            if isinstance(ref, dict) and ref.get(key) not in (None, ""):
                return ref.get(key)
        return default

    def _update_manifest_row_from_add_event(
        self, row: Dict[str, Any], event_id: str, merged_response: Any
    ) -> None:
        receipt = dict(row.get("write_receipt") or {})
        provider_receipt = (
            merged_response
            if isinstance(merged_response, dict)
            else {"result": merged_response, "event_id": event_id}
        )
        receipt["provider_receipt"] = provider_receipt

        provider_status = self._extract_provider_status_from_add_response(
            provider_receipt
        )
        receipt["provider_status"] = provider_status
        row["write_status"] = provider_status

        existing_refs = [
            dict(item)
            for item in (row.get("memory_refs") or receipt.get("memory_refs") or [])
            if isinstance(item, dict)
        ]
        user_id = self._first_ref_value(existing_refs, "user_id")
        batch_index = self._first_ref_value(existing_refs, "batch_index", 0)
        run_id = (
            receipt.get("run_id")
            or receipt.get("source_session_id")
            or self._first_ref_value(existing_refs, "run_id")
        )
        source_unit_ids = (
            receipt.get("source_unit_ids") or row.get("source_unit_ids") or []
        )

        parsed_refs = self._extract_memory_refs_from_add_response(
            response=provider_receipt,
            user_id=str(user_id or ""),
            batch_index=int(batch_index or 0),
            source_unit_ids=source_unit_ids,
            run_id=str(run_id or ""),
        )
        if parsed_refs:
            receipt["memory_refs"] = parsed_refs
            row["memory_refs"] = parsed_refs
        else:
            for ref in existing_refs:
                ref.setdefault("event_id", event_id)
            if existing_refs:
                receipt["memory_refs"] = existing_refs
                row["memory_refs"] = existing_refs

        row["write_receipt"] = receipt

    def _mark_manifest_row_add_event_failed(
        self, row: Dict[str, Any], event_id: str, error: Exception
    ) -> None:
        receipt = dict(row.get("write_receipt") or {})
        errors = list(row.get("errors") or receipt.get("errors") or [])
        errors.append(
            {
                "stage": "add_event_wait",
                "event_id": event_id,
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            }
        )
        receipt["provider_status"] = "failed"
        receipt["errors"] = errors
        row["write_receipt"] = receipt
        row["write_status"] = "failed"
        row["errors"] = errors

    async def _wait_for_deferred_add_events(self) -> None:
        event_rows = [
            (row, event_id)
            for row in self._import_manifest_records
            if (event_id := self._event_id_from_manifest_row(row))
        ]
        if not event_rows:
            self.console.print(
                "[yellow]⚠️  Mem0 deferred add-event wait is enabled, but no event_id values were found.[/yellow]"
            )
            return

        self.console.print(
            f"\n[yellow]⏳ Waiting for {len(event_rows)} Mem0 add events "
            f"(concurrency={self.add_event_wait_num_workers})...[/yellow]"
        )
        semaphore = asyncio.Semaphore(self.add_event_wait_num_workers)

        async def wait_one(row: Dict[str, Any], event_id: str) -> None:
            async with semaphore:
                receipt = row.get("write_receipt") or {}
                provider_receipt = receipt.get("provider_receipt") or {}
                if not isinstance(provider_receipt, dict):
                    provider_receipt = {"result": provider_receipt}
                provider_receipt.setdefault("event_id", event_id)
                try:
                    merged_response = await self._wait_for_add_event_response(
                        provider_receipt
                    )
                    self._update_manifest_row_from_add_event(
                        row, event_id, merged_response
                    )
                except Exception as exc:
                    self._mark_manifest_row_add_event_failed(row, event_id, exc)
                    raise

        results = await asyncio.gather(
            *(wait_one(row, event_id) for row, event_id in event_rows),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            first_failure = failures[0]
            raise RuntimeError(
                f"{len(failures)}/{len(event_rows)} Mem0 add events failed or timed out; "
                f"first failure: {first_failure}"
            ) from first_failure

        self.console.print(
            "[green]✅ All Mem0 add events succeeded; continuing[/green]"
        )

    def _extract_results_list(self, response: Any) -> List[Dict[str, Any]]:
        """Normalize Mem0 list/get_all responses to a result list."""
        if isinstance(response, dict):
            results = response.get("results", [])
            return results if isinstance(results, list) else []
        if isinstance(response, list):
            return response
        return []

    @staticmethod
    def _dedupe_nonempty(values: List[Any]) -> List[str]:
        deduped: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if text and text not in seen:
                deduped.append(text)
                seen.add(text)
        return deduped

    def _count_non_ready_results(self, results: List[Dict[str, Any]]) -> int:
        """Best-effort detection for pending/processing memories."""
        non_ready_count = 0
        for item in results:
            for field in ("status", "state", "processing_status"):
                raw_value = item.get(field)
                if raw_value is None:
                    continue

                normalized = str(raw_value).strip().lower()
                if not normalized:
                    continue
                if normalized in self.READY_MEMORY_STATES:
                    break
                if normalized in self.NON_READY_MEMORY_STATES or normalized not in {
                    "failed",
                    "error",
                    "cancelled",
                    "canceled",
                }:
                    non_ready_count += 1
                    break
        return non_ready_count

    @staticmethod
    def _normalize_provider_status(status: Any, default: str = "submitted") -> str:
        value = str(status or "").strip().lower()
        if not value:
            return default
        aliases = {
            "done": "completed",
            "complete": "completed",
            "ready": "completed",
            "success": "completed",
            "succeeded": "completed",
            "pending": "pending",
            "queued": "pending",
            "processing": "processing",
            "running": "processing",
            "submitted": "submitted",
        }
        return aliases.get(value, value)

    def _extract_provider_status_from_add_response(self, response: Any) -> str:
        """Best-effort parsing of Mem0 add status from raw payload."""
        candidate_rows: List[Dict[str, Any]] = []
        candidates: List[Any] = []

        if isinstance(response, dict):
            candidates.extend(
                [
                    response.get("status"),
                    response.get("state"),
                    response.get("processing_status"),
                ]
            )
            results = response.get("results")
            if isinstance(results, list):
                candidate_rows.extend(
                    item for item in results if isinstance(item, dict)
                )
        elif isinstance(response, list):
            candidate_rows.extend(item for item in response if isinstance(item, dict))

        for row in candidate_rows:
            candidates.extend(
                [row.get("status"), row.get("state"), row.get("processing_status")]
            )

        for candidate in candidates:
            normalized = self._normalize_provider_status(candidate, default="")
            if normalized:
                return normalized
        return "submitted"

    async def _list_memories_for_wait(
        self, user_id: str, run_id: str | None = None
    ) -> Dict[str, Any]:
        """
        Poll the same listing surface the official TS integration helper relies on.

        The current Python SDK routes get_all() through POST /v2/memories/, but
        the official client-side waiting helper uses GET /v1/memories/. In this
        environment the v2 path returns 400, so readiness checks need the v1 list
        endpoint instead.
        """
        if run_id:
            page_size = 200
            page = 1
            page_count = 0
            merged_results: List[Dict[str, Any]] = []
            next_value = None
            previous_value = None

            while True:
                response = await self.client.get_all(
                    filters=self._build_session_filters(user_id, run_id),
                    page=page,
                    page_size=page_size,
                )
                page_count += 1
                page_results = self._extract_results_list(response)
                merged_results.extend(page_results)

                is_paginated_envelope = (
                    isinstance(response, dict)
                    and isinstance(response.get("results"), list)
                    and any(key in response for key in ("count", "next", "previous"))
                )
                if isinstance(response, dict):
                    next_value = response.get("next")
                    previous_value = response.get("previous")
                else:
                    next_value = None
                    previous_value = None

                if (
                    not is_paginated_envelope
                    or not next_value
                    or len(page_results) < page_size
                ):
                    break
                page += 1

            return {
                "results": merged_results,
                "pagination": {
                    "page_count": page_count,
                    "page_size": page_size,
                    "count": len(merged_results),
                    "next": next_value,
                    "previous": previous_value,
                    "run_id": run_id,
                },
            }

        params = {"user_id": user_id}
        prepare_params = getattr(self.client, "_prepare_params", None)
        if callable(prepare_params):
            params = prepare_params(params)

        response = await self.client.async_client.get("/v1/memories/", params=params)
        response.raise_for_status()

        result = response.json()
        if isinstance(result, list):
            return {"results": result}
        return result

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """Track user_ids so Mem0 can poll readiness after add."""
        self._post_add_wait_user_ids = set()
        return await super().add(conversations, **kwargs)

    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        """
        Preparation stage: update project configuration and clean existing data.

        Args:
            conversations: Standard format conversation list
            **kwargs: Extra parameters
        """
        # Check if need to clean existing data
        clean_before_add = self.config.get("clean_before_add", False)

        if not clean_before_add:
            self.console.print(
                "   ⏭️  Skipping data cleanup (clean_before_add=false)", style="dim"
            )
            return

        self.console.print(f"\n{'=' * 60}", style="bold yellow")
        self.console.print(f"Preparation: Cleaning existing data", style="bold yellow")
        self.console.print(f"{'=' * 60}", style="bold yellow")

        # Collect all user_ids to clean
        user_ids_to_clean = set()

        for conv in conversations:
            # Get user_id for speaker_a and speaker_b
            speaker_a = conv.metadata.get("speaker_a", "")
            speaker_b = conv.metadata.get("speaker_b", "")

            need_dual = self._need_dual_perspective(speaker_a, speaker_b)

            user_ids_to_clean.add(self._extract_user_id(conv, speaker="speaker_a"))

            if need_dual:
                user_ids_to_clean.add(self._extract_user_id(conv, speaker="speaker_b"))

        # Clean all user data
        self.console.print(
            f"\n🗑️  Cleaning data for {len(user_ids_to_clean)} user(s)...",
            style="yellow",
        )

        cleaned_count = 0
        failed_count = 0

        for user_id in user_ids_to_clean:
            try:
                # Use async client for delete operation
                await self.client.delete_all(user_id=user_id)
                cleaned_count += 1
                self.console.print(f"   ✅ Cleaned: {user_id}", style="green")
            except Exception as e:
                failed_count += 1
                self.console.print(
                    f"   ⚠️  Failed to clean {user_id}: {e}", style="yellow"
                )

        self.console.print(
            f"\n✅ Cleanup completed: {cleaned_count} succeeded, {failed_count} failed",
            style="bold green",
        )

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user to Mem0.

        Args:
            conv: Original conversation object
            messages: Formatted message list
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters

        Returns:
            None
        """
        # Extract user_id
        user_id = self._extract_user_id(conv, speaker=speaker)
        self._post_add_wait_user_ids.add(user_id)

        # Handle content truncation (Mem0 specific)
        truncated_count = 0
        for msg in messages:
            if len(msg["content"]) > self.max_content_length:
                msg["content"] = msg["content"][: self.max_content_length]
                truncated_count += 1

        # Log info
        sender_name = conv.metadata.get(speaker, speaker)
        is_fake_timestamp = (
            conv.messages[0].metadata.get("is_fake_timestamp", False)
            if conv.messages
            else False
        )

        self.console.print(
            f"   📤 Adding for {sender_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )
        if is_fake_timestamp:
            self.console.print(f"   ⚠️  Using fake timestamp", style="yellow")
        if truncated_count > 0:
            self.console.print(
                f"   ⚠️  Truncated {truncated_count} messages (>{self.max_content_length} chars)",
                style="yellow",
            )

        # Add messages in batches with retry. Keep LoCoMo session boundaries intact
        # so each request has one observation timestamp.
        receipts = []
        for batch_index, (batch_messages, batch_source_messages) in enumerate(
            self._iter_session_batches(messages, conv.messages)
        ):
            source_session_id = self._extract_batch_session_id(batch_source_messages)

            # Use the first available message timestamp in this batch.
            timestamp = None
            metadata = None
            batch_timestamp = None
            for source_message in batch_source_messages:
                if source_message and source_message.timestamp:
                    batch_timestamp = source_message.timestamp
                    break
            if batch_timestamp:
                timestamp = self._to_utc_unix_seconds(batch_timestamp)
                metadata = {
                    "timestamp": self._format_metadata_timestamp(batch_timestamp)
                }
            else:
                self.console.print(
                    f"   ⚠️  [{sender_name} (user_id={user_id})] Missing timestamp "
                    f"for batch {batch_index}; sending without metadata.timestamp",
                    style="yellow",
                )
            if source_session_id:
                metadata = metadata or {}
                metadata["source_session_id"] = source_session_id

            for attempt in range(self.max_retries):
                try:
                    # Use async client for add operation
                    add_kwargs = {
                        "messages": batch_messages,
                        "user_id": user_id,
                    }

                    if timestamp is not None:
                        add_kwargs["timestamp"] = timestamp
                    if metadata:
                        add_kwargs["metadata"] = metadata
                    if source_session_id:
                        add_kwargs["run_id"] = source_session_id

                    response = await self.client.add(**add_kwargs)
                    response = await self._wait_for_add_event(response)
                    source_unit_ids = [
                        msg.metadata.get("source_unit_id")
                        for msg in batch_source_messages
                        if msg and msg.metadata and msg.metadata.get("source_unit_id")
                    ]
                    memory_refs = self._extract_memory_refs_from_add_response(
                        response=response,
                        user_id=user_id,
                        batch_index=batch_index,
                        source_unit_ids=source_unit_ids,
                        run_id=source_session_id,
                    )
                    if not memory_refs:
                        memory_refs = [
                            {
                                "provider": "mem0",
                                "user_id": user_id,
                                "run_id": source_session_id,
                                "source_session_id": source_session_id,
                                "batch_index": batch_index,
                                "event_id": self._extract_event_id_from_add_response(
                                    response
                                ),
                                "source_unit_ids": source_unit_ids,
                                "inspection_scope": "coarse",
                            }
                        ]
                    receipts.append(
                        {
                            "chunk_id": f"{user_id}:batch{batch_index}",
                            "run_id": source_session_id,
                            "source_session_id": source_session_id,
                            "source_unit_ids": source_unit_ids,
                            "provider_receipt": (
                                response
                                if isinstance(response, dict)
                                else {"result": response}
                            ),
                            "provider_status": self._extract_provider_status_from_add_response(
                                response
                            ),
                            "memory_refs": memory_refs,
                        }
                    )
                    # Wait between add requests to avoid rate limits
                    if self.add_interval > 0:
                        await asyncio.sleep(self.add_interval)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.console.print(
                            f"   ⚠️  [{sender_name} (user_id={user_id})] Retry {attempt + 1}/{self.max_retries}: {e}",
                            style="yellow",
                        )
                        await asyncio.sleep(2**attempt)  # Use async sleep
                    else:
                        self.console.print(
                            f"   ❌ [{sender_name} (user_id={user_id})] Failed after {self.max_retries} retries: {e}",
                            style="red",
                        )
                        raise e

        return receipts

    async def _post_add_process(self, add_results: List[Any], **kwargs) -> None:
        """
        Run Mem0 post-add readiness checks before the pipeline continues.

        Deferred event waiting confirms every submitted add event has succeeded after
        all add requests have been sent. Adapter polling is the older optional
        visibility check.
        """
        if self.add_event_wait and self._effective_add_event_wait_mode() == "deferred":
            await self._wait_for_deferred_add_events()

        if not self._uses_adapter_post_add_wait():
            return

        max_wait_seconds = int(self.config.get("post_add_wait_seconds", 0))
        if max_wait_seconds <= 0 or not self._post_add_wait_user_ids:
            return

        last_snapshot: Dict[str, tuple[int, int]] = {}
        stable_polls: Dict[str, int] = {
            user_id: 0 for user_id in self._post_add_wait_user_ids
        }
        start_time = monotonic()

        self.console.print(
            "\n[yellow]⏳ Polling Mem0 until newly added memories are visible...[/yellow]"
        )

        while True:
            all_ready = True

            for user_id in sorted(self._post_add_wait_user_ids):
                try:
                    response = await self._list_memories_for_wait(user_id)
                except Exception as e:
                    all_ready = False
                    stable_polls[user_id] = 0
                    self.console.print(
                        f"   ⚠️  Failed to poll memories for {user_id}: {e}",
                        style="yellow",
                    )
                    continue

                results = self._extract_results_list(response)
                visible_count = len(results)
                non_ready_count = self._count_non_ready_results(results)
                snapshot = (visible_count, non_ready_count)

                if snapshot == last_snapshot.get(user_id) and visible_count > 0:
                    stable_polls[user_id] += 1
                else:
                    stable_polls[user_id] = 0
                    last_snapshot[user_id] = snapshot

                user_ready = (
                    visible_count > 0
                    and non_ready_count == 0
                    and stable_polls[user_id] >= self.post_add_stable_polls
                )
                if not user_ready:
                    all_ready = False

            if all_ready:
                self.console.print(
                    "[green]✅ Mem0 memories look stable; continuing to search[/green]"
                )
                return

            elapsed = monotonic() - start_time
            if elapsed >= max_wait_seconds:
                self.console.print(
                    "[yellow]⚠️  Timed out while waiting for Mem0 readiness; continuing with the latest visible memories[/yellow]"
                )
                return

            sleep_seconds = min(
                self.post_add_poll_interval_seconds, max_wait_seconds - elapsed
            )
            await asyncio.sleep(sleep_seconds)

    def _extract_memory_refs_from_add_response(
        self,
        response: Any,
        user_id: str,
        batch_index: int,
        source_unit_ids: List[str] | None = None,
        run_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Best-effort parsing of Mem0 add receipts."""
        refs: List[Dict[str, Any]] = []
        candidate_rows = []
        event_id = ""
        if isinstance(response, dict):
            event_id = str(response.get("event_id") or "").strip()
            if isinstance(response.get("results"), list):
                candidate_rows.extend(response["results"])
            elif isinstance(response.get("event"), dict) and isinstance(
                response["event"].get("results"), list
            ):
                candidate_rows.extend(response["event"]["results"])
            else:
                candidate_rows.append(response)
        elif isinstance(response, list):
            candidate_rows.extend(response)

        for result_index, row in enumerate(candidate_rows):
            if not isinstance(row, dict):
                continue
            memory_id = row.get("id") or row.get("memory_id")
            if memory_id:
                refs.append(
                    {
                        "provider": "mem0",
                        "memory_id": str(memory_id),
                        "user_id": user_id,
                        "run_id": run_id,
                        "source_session_id": run_id,
                        "batch_index": batch_index,
                        "result_index": result_index,
                        "event_id": event_id,
                        "source_unit_ids": list(source_unit_ids or []),
                        "inspection_scope": "object",
                    }
                )
        return refs

    def _best_effort_backfill_memory_refs(
        self,
        memory_refs: List[Dict[str, Any]],
        visible_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Backfill memory ids only when the mapping is unambiguous."""
        updated_refs = [dict(ref) for ref in memory_refs]
        visible_ids = [
            str(item.get("id") or item.get("memory_id"))
            for item in visible_results
            if item.get("id") or item.get("memory_id")
        ]

        unresolved_refs = [ref for ref in updated_refs if not ref.get("memory_id")]
        if len(unresolved_refs) == 1 and len(visible_ids) == 1:
            unresolved_refs[0]["memory_id"] = visible_ids[0]

        for ref in updated_refs:
            if ref.get("memory_id"):
                ref["inspection_scope"] = "object"
            else:
                ref["inspection_scope"] = "coarse"
        return updated_refs

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
        """Finalize Mem0 writes with best-effort enrichment under manual confirmation."""
        del add_result, budget_seconds, poll_interval_seconds, dataset, kwargs

        provider_status_counts: Dict[str, int] = {}
        missing_memory_refs: List[str] = []
        warnings: List[str] = [
            "Mem0 finalize skips automatic readiness gating; continue only after manual provider confirmation that add has completed."
        ]
        updated_rows = 0
        user_snapshots: Dict[str, Any] = {}
        refreshed_rows: List[Dict[str, Any]] = []

        for row in import_manifest_rows:
            updated_row = dict(row)
            receipt = dict(updated_row.get("write_receipt", {}) or {})
            memory_refs = [
                dict(item) for item in (updated_row.get("memory_refs") or [])
            ]
            row_errors = list(updated_row.get("errors", []) or [])
            user_id = next(
                (str(ref.get("user_id")) for ref in memory_refs if ref.get("user_id")),
                "",
            )
            run_id = next(
                (
                    str(ref.get("run_id") or ref.get("source_session_id"))
                    for ref in memory_refs
                    if ref.get("run_id") or ref.get("source_session_id")
                ),
                str(
                    updated_row.get("run_id")
                    or updated_row.get("source_session_id")
                    or ""
                ),
            )

            status = self._normalize_provider_status(
                updated_row.get("write_status")
                or receipt.get("provider_status")
                or self._extract_provider_status_from_add_response(
                    receipt.get("provider_receipt", {}) or {}
                ),
                default="submitted",
            )

            if user_id:
                snapshot_key = f"{user_id}\0{run_id}" if run_id else user_id
                if snapshot_key not in user_snapshots:
                    try:
                        response = await self._list_memories_for_wait(user_id, run_id)
                        user_snapshots[snapshot_key] = (
                            self._extract_results_list(response),
                            None,
                        )
                    except Exception as exc:  # noqa: BLE001
                        user_snapshots[snapshot_key] = ([], exc)
                visible_results, snapshot_exc = user_snapshots[snapshot_key]
                if snapshot_exc is not None:
                    row_errors.append(
                        {
                            "stage": "finalize",
                            "error_type": type(snapshot_exc).__name__,
                            "error_message": str(snapshot_exc),
                        }
                    )
                    warnings.append(
                        f"{updated_row.get('chunk_id', '')}: visibility probe failed ({type(snapshot_exc).__name__}); keeping manual confirmation flow"
                    )
                else:
                    non_ready_count = self._count_non_ready_results(visible_results)
                    memory_refs = self._best_effort_backfill_memory_refs(
                        memory_refs,
                        visible_results,
                    )
                    if visible_results and non_ready_count == 0:
                        status = "completed"
                    elif visible_results and status in {
                        "submitted",
                        "queued",
                        "pending",
                        "processing",
                    }:
                        status = "pending"
            else:
                warnings.append(
                    f"{updated_row.get('chunk_id', '')}: unable to determine Mem0 user_id during finalize"
                )

            if not any(ref.get("memory_id") for ref in memory_refs):
                missing_memory_refs.append(str(updated_row.get("chunk_id", "")))
                warnings.append(
                    f"{updated_row.get('chunk_id', '')}: finalize could not backfill a stable memory_id; "
                    "readback search may remain coarse/user-level"
                )

            receipt["provider_status"] = status
            updated_row["write_receipt"] = receipt
            updated_row["memory_refs"] = memory_refs
            updated_row["write_status"] = status
            updated_row["errors"] = row_errors
            provider_status_counts[status] = provider_status_counts.get(status, 0) + 1
            updated_rows += 1
            refreshed_rows.append(updated_row)

        return {
            "import_manifest_records": refreshed_rows,
            "ready": True,
            "status": "finalized",
            "provider_status_counts": provider_status_counts,
            "updated_rows": updated_rows,
            "missing_memory_refs": missing_memory_refs,
            "warnings": warnings,
            "finalize_budget_exhausted": False,
        }

    async def probe_readiness(self, add_result: Any = None, **kwargs) -> Dict[str, Any]:
        """Mem0 readiness uses visible memories or adapter-side polling."""
        if self._uses_adapter_post_add_wait():
            return {
                "supported": True,
                "ready": True,
                "status": "ready",
                "details": {"strategy": "adapter_poll"},
            }

        ready_users = {}
        for user_id in sorted(self._post_add_wait_user_ids):
            try:
                response = await self._list_memories_for_wait(user_id)
                ready_users[user_id] = len(self._extract_results_list(response))
            except Exception as exc:  # noqa: BLE001
                ready_users[user_id] = f"error:{exc}"
        return {
            "supported": True,
            "ready": all(
                isinstance(value, int) and value > 0 for value in ready_users.values()
            ),
            "status": "ready" if ready_users else "no_users",
            "details": {"visible_counts": ready_users},
        }

    @staticmethod
    def _iter_manifest_memory_refs(row: Dict[str, Any]):
        for ref in row.get("memory_refs", []) or []:
            if isinstance(ref, dict):
                yield ref
        write_receipt = row.get("write_receipt", {}) or {}
        if isinstance(write_receipt, dict):
            for ref in write_receipt.get("memory_refs", []) or []:
                if isinstance(ref, dict):
                    yield ref

    @staticmethod
    def _session_id_from_ref(ref: Dict[str, Any]) -> str:
        return str(
            ref.get("run_id")
            or ref.get("source_session_id")
            or ref.get("session_id")
            or ""
        ).strip()

    def _find_mem0_user_ids_for_readback(
        self,
        *,
        user_id: Optional[str],
        context: Optional[Dict[str, Any]],
        session_ids: List[str],
    ) -> Dict[str, List[str]]:
        """Resolve Mem0 user ids from selected manifest rows, grouped by session."""
        target_sessions = set(session_ids)
        context = context or {}
        context_conversation_id = str(context.get("conversation_id") or "").strip()
        rows = [
            row
            for row in (context.get("import_manifest_rows", []) or [])
            if isinstance(row, dict)
            and (
                not context_conversation_id
                or not str(row.get("conversation_id") or "").strip()
                or str(row.get("conversation_id") or "").strip() == context_conversation_id
            )
        ]
        user_ids_by_session: Dict[str, List[str]] = {sid: [] for sid in session_ids}
        fallback_user_ids: List[str] = []

        def add_user(target: str, candidate: Any) -> None:
            uid = str(candidate or "").strip()
            if not uid:
                return
            if target and target in user_ids_by_session:
                if uid not in user_ids_by_session[target]:
                    user_ids_by_session[target].append(uid)
            elif uid not in fallback_user_ids:
                fallback_user_ids.append(uid)

        for row in rows:
            row_session = str(
                row.get("source_session_id")
                or row.get("session_id")
                or row.get("run_id")
                or ""
            ).strip()
            write_receipt = row.get("write_receipt", {}) or {}
            namespace_scope = {}
            if isinstance(write_receipt, dict):
                namespace_scope = write_receipt.get("namespace_scope", {}) or {}
                for key in ("run_id", "source_session_id", "session_id"):
                    if not row_session and write_receipt.get(key):
                        row_session = str(write_receipt[key]).strip()
            write_request_summary = row.get("write_request_summary", {}) or {}

            for ref in self._iter_manifest_memory_refs(row):
                ref_session = self._session_id_from_ref(ref) or row_session
                if target_sessions and ref_session and ref_session not in target_sessions:
                    continue
                add_user(ref_session, ref.get("user_id"))

            for candidate in (
                namespace_scope.get("namespace_id")
                if isinstance(namespace_scope, dict)
                else None,
                row.get("user_id"),
                write_request_summary.get("namespace_id")
                if isinstance(write_request_summary, dict)
                else None,
            ):
                if target_sessions and row_session and row_session not in target_sessions:
                    continue
                add_user(row_session, candidate)

        add_user("", user_id)
        for session_id in session_ids:
            if not user_ids_by_session.get(session_id):
                user_ids_by_session[session_id] = list(fallback_user_ids)
        if not session_ids:
            user_ids_by_session[""] = list(fallback_user_ids)
        return user_ids_by_session

    @staticmethod
    def _memory_session_lookup_from_manifest(
        context: Optional[Dict[str, Any]],
        session_ids: List[str],
    ) -> Dict[str, str]:
        """Map Mem0 memory ids back to target QA sessions from manifest refs."""
        target_sessions = set(session_ids)
        context = context or {}
        context_conversation_id = str(context.get("conversation_id") or "").strip()
        lookup: Dict[str, str] = {}
        for row in context.get("import_manifest_rows", []) or []:
            if not isinstance(row, dict):
                continue
            row_conversation_id = str(row.get("conversation_id") or "").strip()
            if (
                context_conversation_id
                and row_conversation_id
                and row_conversation_id != context_conversation_id
            ):
                continue
            row_session = str(
                row.get("source_session_id")
                or row.get("session_id")
                or row.get("run_id")
                or ""
            ).strip()
            for ref in Mem0Adapter._iter_manifest_memory_refs(row):
                memory_id = str(
                    ref.get("memory_id") or ref.get("id") or ""
                ).strip()
                if not memory_id:
                    continue
                ref_session = (
                    Mem0Adapter._session_id_from_ref(ref) or row_session
                )
                if target_sessions and ref_session not in target_sessions:
                    continue
                lookup.setdefault(memory_id, ref_session)
        return lookup

    @staticmethod
    def _mem0_error_status(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        message = str(exc).lower()
        if status_code == 429 or "429" in message or "rate limit" in message:
            return "rate_limited"
        if isinstance(exc, TimeoutError) or "timeout" in message or "timed out" in message:
            return "timeout"
        if isinstance(status_code, int) and status_code >= 500:
            return "error"
        if "connection" in message or "transport" in message:
            return "transport_error"
        return "error"

    @staticmethod
    def _mem0_error_record(
        exc: Exception,
        *,
        user_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        error = {
            "stage": "get_storage_readback",
            "user_id": user_id,
            "session_id": session_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        if status_code is not None:
            error["status_code"] = status_code
        return error

    @staticmethod
    def _normalize_mem0_storage_object(
        memory: Dict[str, Any],
        *,
        user_id: str,
        session_id: str,
        memory_session_lookup: Optional[Dict[str, str]] = None,
    ) -> NormalizedStorageObject:
        metadata = memory.get("metadata", {}) or {}
        memory_id = str(memory.get("id") or memory.get("memory_id") or "").strip()
        mapped_session_id = (
            str((memory_session_lookup or {}).get(memory_id) or "").strip()
            if memory_id
            else ""
        )
        resolved_session_id = str(
            session_id
            or memory.get("run_id")
            or metadata.get("run_id")
            or metadata.get("source_session_id")
            or memory.get("session_id")
            or metadata.get("session_id")
            or mapped_session_id
            or "unknown"
        ).strip()
        return NormalizedStorageObject(
            session_id=resolved_session_id,
            kind="memory",
            id=memory_id,
            content=Mem0Adapter._get_memory_text(memory),
            metadata={
                "provider": "mem0",
                "user_id": memory.get("user_id") or user_id,
                "run_id": memory.get("run_id") or metadata.get("run_id") or session_id,
                "source_session_id": metadata.get("source_session_id") or session_id,
                "categories": memory.get("categories", []) or [],
                "created_at": memory.get("created_at"),
                "updated_at": memory.get("updated_at"),
                "score": memory.get("score"),
                "status": memory.get("status") or memory.get("state"),
            },
            raw=memory,
        )

    def _readback_interval(self) -> float:
        readback_config = self.config.get("readback", {}) or {}
        configured = readback_config.get("interval_seconds")
        if configured is None:
            configured = readback_config.get("readback_interval_seconds")
        if configured is None:
            configured = getattr(self, "search_interval", 0.0)
        try:
            return max(float(configured or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _readback_session_cache_key(user_id: str, session_id: str) -> str:
        return f"{user_id}\0{session_id}"

    async def _await_readback_slot(self, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            return
        async with self._readback_request_lock:
            now = perf_counter()
            delay = self._readback_next_allowed_at - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = perf_counter()
            self._readback_next_allowed_at = now + interval_seconds

    async def _fetch_mem0_readback_for_session(
        self,
        *,
        user_id: str,
        session_id: str,
        interval_seconds: float,
        memory_session_lookup: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        await self._await_readback_slot(interval_seconds)
        response = await self._list_memories_for_wait(user_id, session_id or None)
        results = self._extract_results_list(response)
        objects: List[NormalizedStorageObject] = []
        object_keys: set[tuple[str, str, str]] = set()
        for memory in results:
            if not isinstance(memory, dict):
                continue
            normalized = self._normalize_mem0_storage_object(
                memory,
                user_id=user_id,
                session_id=session_id,
                memory_session_lookup=memory_session_lookup,
            )
            object_key = (
                normalized.session_id,
                normalized.id,
                normalized.content,
            )
            if object_key in object_keys:
                continue
            object_keys.add(object_key)
            objects.append(normalized)
        return {
            "user_id": user_id,
            "session_id": session_id,
            "objects": objects,
            "raw_memory_count": len(results),
        }

    async def _get_cached_mem0_readback_for_session(
        self,
        *,
        user_id: str,
        session_id: str,
        interval_seconds: float,
        memory_session_lookup: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        cache_key = self._readback_session_cache_key(user_id, session_id)
        cached = self._readback_session_cache.get(cache_key)
        if cached is not None:
            return cached

        lock = self._readback_session_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._readback_session_cache.get(cache_key)
            if cached is not None:
                return cached

            cache_entry = await self._fetch_mem0_readback_for_session(
                user_id=user_id,
                session_id=session_id,
                interval_seconds=interval_seconds,
                memory_session_lookup=memory_session_lookup,
            )
            self._readback_session_cache[cache_key] = cache_entry
            return cache_entry

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
        """Read Mem0 memories for the current QA sessions via user_id + run_id."""
        del run_id, evidence_texts, kwargs
        target_session_ids = self._dedupe_nonempty(list(session_ids or []))
        user_ids_by_session = self._find_mem0_user_ids_for_readback(
            user_id=user_id,
            context=context,
            session_ids=target_session_ids,
        )
        all_user_ids = self._dedupe_nonempty(
            [
                uid
                for user_ids in user_ids_by_session.values()
                for uid in user_ids
            ]
        )
        memory_session_lookup = self._memory_session_lookup_from_manifest(
            context,
            target_session_ids,
        )

        if not all_user_ids:
            return StorageReadbackResult(
                status="unsupported",
                checked_session_ids=target_session_ids,
                missing_session_ids=[],
                objects=[],
                metadata={
                    "provider": "mem0",
                    "question_id": question_id,
                    "readback_scope": "session" if target_session_ids else "user",
                    "reason": "missing user_id/namespace",
                },
                errors=[
                    {
                        "stage": "get_storage_readback",
                        "error_type": "missing_user_id",
                        "error_message": "Cannot determine Mem0 user_id for storage readback.",
                    }
                ],
            )

        interval_seconds = self._readback_interval()
        objects: List[NormalizedStorageObject] = []
        errors: List[Dict[str, Any]] = []
        error_statuses: List[str] = []
        object_keys: set[tuple[str, str, str]] = set()
        checked_session_ids = list(target_session_ids)
        successful_session_ids: set[str] = set()
        raw_memory_count = 0

        if target_session_ids:
            readback_targets = [
                (session_id, user_id_for_session)
                for session_id in target_session_ids
                for user_id_for_session in user_ids_by_session.get(session_id, [])
            ]
        else:
            readback_targets = [("", mem0_user_id) for mem0_user_id in all_user_ids]

        for session_id, mem0_user_id in readback_targets:
            try:
                session_readback = await self._get_cached_mem0_readback_for_session(
                    user_id=mem0_user_id,
                    session_id=session_id,
                    interval_seconds=interval_seconds,
                    memory_session_lookup=memory_session_lookup,
                )
                raw_memory_count += int(session_readback.get("raw_memory_count", 0) or 0)
                if session_id:
                    successful_session_ids.add(session_id)
                for normalized in session_readback.get("objects", []) or []:
                    object_key = (
                        normalized.session_id,
                        normalized.id,
                        normalized.content,
                    )
                    if object_key in object_keys:
                        continue
                    object_keys.add(object_key)
                    objects.append(normalized)
            except Exception as exc:  # noqa: BLE001
                error_statuses.append(self._mem0_error_status(exc))
                errors.append(
                    self._mem0_error_record(
                        exc,
                        user_id=mem0_user_id,
                        session_id=session_id,
                    )
                )

        if target_session_ids and memory_session_lookup:
            target_memory_ids = set(memory_session_lookup)
            existing_memory_ids = {
                str(obj.id or "").strip() for obj in objects if str(obj.id or "").strip()
            }
            missing_memory_ids = target_memory_ids - existing_memory_ids
            if missing_memory_ids:
                for mem0_user_id in all_user_ids:
                    try:
                        user_readback = await self._get_cached_mem0_readback_for_session(
                            user_id=mem0_user_id,
                            session_id="",
                            interval_seconds=interval_seconds,
                            memory_session_lookup=memory_session_lookup,
                        )
                        raw_memory_count += int(
                            user_readback.get("raw_memory_count", 0) or 0
                        )
                        for normalized in user_readback.get("objects", []) or []:
                            memory_id = str(normalized.id or "").strip()
                            if memory_id not in missing_memory_ids:
                                continue
                            mapped_session_id = memory_session_lookup.get(memory_id)
                            if not mapped_session_id:
                                continue
                            normalized = NormalizedStorageObject(
                                session_id=mapped_session_id,
                                kind=normalized.kind,
                                id=normalized.id,
                                content=normalized.content,
                                metadata={
                                    **(normalized.metadata or {}),
                                    "run_id": mapped_session_id,
                                    "source_session_id": mapped_session_id,
                                },
                                raw=normalized.raw,
                            )
                            object_key = (
                                normalized.session_id,
                                normalized.id,
                                normalized.content,
                            )
                            if object_key in object_keys:
                                continue
                            object_keys.add(object_key)
                            objects.append(normalized)
                            successful_session_ids.add(mapped_session_id)
                            existing_memory_ids.add(memory_id)
                        missing_memory_ids = target_memory_ids - existing_memory_ids
                        if not missing_memory_ids:
                            break
                    except Exception as exc:  # noqa: BLE001
                        error_statuses.append(self._mem0_error_status(exc))
                        errors.append(
                            self._mem0_error_record(
                                exc,
                                user_id=mem0_user_id,
                                session_id="",
                            )
                        )

        object_session_ids = {
            obj.session_id for obj in objects if obj.session_id and obj.session_id != "unknown"
        }
        missing_session_ids = [
            session_id
            for session_id in target_session_ids
            if session_id in successful_session_ids and session_id not in object_session_ids
        ]
        if error_statuses:
            status = "rate_limited" if "rate_limited" in error_statuses else error_statuses[0]
        else:
            status = "ok"

        return StorageReadbackResult(
            status=status,
            checked_session_ids=checked_session_ids,
            missing_session_ids=missing_session_ids,
            objects=objects,
            metadata={
                "provider": "mem0",
                "question_id": question_id,
                "user_ids": all_user_ids,
                "readback_scope": "session" if target_session_ids else "user",
                "raw_memory_count": raw_memory_count,
                "object_count": len(objects),
                "readback_interval_seconds": interval_seconds,
            },
            errors=errors,
        )

    @staticmethod
    def _extract_readback_object_fields(
        obj: NormalizedStorageObject | Dict[str, Any],
    ) -> Dict[str, Any]:
        if isinstance(obj, NormalizedStorageObject):
            return {
                "session_id": str(obj.session_id or "").strip(),
                "kind": str(obj.kind or "").strip(),
                "memory_id": str(obj.id or "").strip(),
                "content": str(obj.content or "").strip(),
                "metadata": obj.metadata or {},
            }

        metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
        return {
            "session_id": str(obj.get("session_id") or "").strip(),
            "kind": str(obj.get("kind") or "").strip(),
            "memory_id": str(obj.get("id") or obj.get("memory_id") or "").strip(),
            "content": str(obj.get("content") or "").strip(),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    @staticmethod
    def _build_readback_result_metadata(fields: Dict[str, Any]) -> Dict[str, Any]:
        metadata = fields.get("metadata") or {}
        return {
            "memory_id": fields.get("memory_id", ""),
            "kind": fields.get("kind", ""),
            "session_id": fields.get("session_id", ""),
            "user_id": metadata.get("user_id"),
            "run_id": metadata.get("run_id"),
            "source_session_id": metadata.get("source_session_id"),
            "categories": metadata.get("categories") or [],
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "status": metadata.get("status"),
        }

    @staticmethod
    def _map_readback_search_status(
        readback_status: str,
        *,
        has_content: bool,
    ) -> str:
        if readback_status == "ok":
            return "ok" if has_content else "empty"
        if readback_status == "unsupported":
            return "unsupported"
        if readback_status in {
            "rate_limited",
            "timeout",
            "transport_error",
            "error",
        }:
            return "error"
        return "error"

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
        """Build a Mem0 search result from provider storage readback."""
        del index, conversation
        metadata = question_metadata or {}
        raw_session_ids = metadata.get("session_ids")
        if isinstance(raw_session_ids, list):
            raw_session_values = raw_session_ids
        elif raw_session_ids:
            raw_session_values = [raw_session_ids]
        else:
            raw_session_values = []
        session_ids = self._dedupe_nonempty(raw_session_values)

        if not session_ids:
            readback = StorageReadbackResult(
                status="unsupported",
                checked_session_ids=[],
                missing_session_ids=[],
                objects=[],
                metadata={
                    "provider": "mem0",
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
                    "system": "mem0",
                    "search_mode": "readback",
                    "session_ids": [],
                    "formatted_context": "",
                    "user_ids": [],
                    "readback": readback.to_dict(),
                },
                retrieval_status="unsupported",
            )

        readback_context = {
            "question_id": question_id,
            "conversation_id": conversation_id,
            "session_ids": session_ids,
            "import_manifest_rows": import_manifest_records or [],
            "qa_row": {"metadata": metadata},
        }
        readback = await self.get_storage_readback(
            session_ids=session_ids,
            question_id=question_id,
            context=readback_context,
        )
        if (
            bool(kwargs.get("allow_user_scope_fallback"))
            and readback.status == "ok"
            and not readback.objects
        ):
            fallback_readback = await self.get_storage_readback(
                session_ids=[],
                question_id=question_id,
                context=readback_context,
            )
            if fallback_readback.objects:
                fallback_metadata = dict(fallback_readback.metadata or {})
                fallback_metadata["user_scope_fallback_from_session_ids"] = list(
                    session_ids
                )
                fallback_metadata["session_scoped_readback"] = readback.to_dict()
                fallback_readback.metadata = fallback_metadata
                readback = fallback_readback

        requested_sessions = set(session_ids)
        ordered_fields_by_session: Dict[str, List[Dict[str, Any]]] = {
            session_id: [] for session_id in session_ids
        }
        extra_fields: List[Dict[str, Any]] = []
        for obj in readback.objects:
            fields = self._extract_readback_object_fields(obj)
            if not fields["content"]:
                continue
            if fields["session_id"] in requested_sessions:
                ordered_fields_by_session[fields["session_id"]].append(fields)
            else:
                extra_fields.append(fields)

        ordered_fields: List[Dict[str, Any]] = []
        for session_id in session_ids:
            ordered_fields.extend(ordered_fields_by_session[session_id])
        ordered_fields.extend(extra_fields)

        results = [
            {
                "content": fields["content"],
                "score": 1.0,
                "metadata": self._build_readback_result_metadata(fields),
            }
            for fields in ordered_fields
        ]
        formatted_context = "\n".join(
            f"{idx}. {fields['content']}"
            for idx, fields in enumerate(ordered_fields, start=1)
        )

        user_ids_raw = readback.metadata.get("user_ids") or []
        if not isinstance(user_ids_raw, list):
            user_ids_raw = [user_ids_raw]
        retrieval_status = self._map_readback_search_status(
            readback.status,
            has_content=bool(formatted_context or results),
        )

        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "mem0",
                "search_mode": "readback",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "user_ids": self._dedupe_nonempty(user_ids_raw),
                "readback": readback.to_dict(),
            },
            retrieval_status=retrieval_status,
        )

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (Mem0-specific timestamp handling).

        Calls Mem0 search API and converts results to standard format,
        preferring event timestamps stored in metadata and falling back to
        created_at for backwards compatibility.

        Args:
            query: Query text
            conversation_id: Conversation ID (not used by Mem0)
            user_id: User ID to search for
            top_k: Number of results to retrieve
            **kwargs: Additional parameters

        Returns:
            List of search results with event timestamps when available
        """
        # Add interval before search to avoid rate limiting (429 errors)
        if self.search_interval > 0:
            await asyncio.sleep(self.search_interval)

        try:
            # Use async client for search operation
            raw_results = await self.client.search(
                query=query,
                top_k=top_k,
                filters={"user_id": user_id},
            )

            # Debug: print raw search results
            self.console.print(f"\n[DEBUG] Mem0 Search Results:", style="yellow")
            self.console.print(f"  Query: {query}", style="dim")
            self.console.print(f"  User ID: {user_id}", style="dim")
            self.console.print(
                f"  Results: {json.dumps(raw_results, indent=2, ensure_ascii=False)}",
                style="dim",
            )

        except Exception as e:
            self.console.print(f"❌ Mem0 search error: {e}", style="red")
            raise

        # Convert to standard format, preferring the official metadata timestamp
        results = []
        for memory in raw_results.get("results", []):
            event_timestamp_raw = self._get_memory_timestamp(memory)
            created_at_original = memory.get("created_at", "")
            created_at_display = self._convert_timestamp_to_display_timezone(
                created_at_original
            )
            event_timestamp_display = (
                str(event_timestamp_raw).strip() or created_at_display
            )
            memory_text = self._get_memory_text(memory)

            results.append(
                {
                    "content": f"{event_timestamp_display}: {memory_text}",
                    "score": memory.get("score", 0.0),
                    "user_id": user_id,
                    "metadata": {
                        "id": memory.get("id", ""),
                        "event_timestamp_raw": event_timestamp_raw,
                        "event_timestamp_display": event_timestamp_display,
                        "created_at": created_at_original,
                        "created_at_display": created_at_display,
                        "memory": memory_text,
                        "user_id": memory.get("user_id", ""),
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
        """
        Build SearchResult for single perspective (Mem0: simple metadata).

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult (no formatted_context, uses fallback)
        """
        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "mem0",
                "top_k": top_k,
                "dual_perspective": False,
                "user_ids": [user_id],
            },
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
        """
        Build SearchResult for dual perspective (Mem0: use template).

        Formats memories using the default template for dual-speaker scenarios.

        Args:
            query: Query text
            conversation_id: Conversation ID
            all_results: Merged results (for fallback)
            results_a: Speaker A's search results
            results_b: Speaker B's search results
            speaker_a: Speaker A name
            speaker_b: Speaker B name
            speaker_a_user_id: Speaker A user ID
            speaker_b_user_id: Speaker B user ID
            top_k: Number of results per user
            **kwargs: Additional parameters

        Returns:
            SearchResult with formatted_context
        """
        # Extract content from results (already prefixed with event timestamps)
        speaker_a_memories_text = (
            "\n".join([r["content"] for r in results_a])
            if results_a
            else "(No memories found)"
        )
        speaker_b_memories_text = (
            "\n".join([r["content"] for r in results_b])
            if results_b
            else "(No memories found)"
        )

        # Use default template
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        formatted_context = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_memories_text,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_memories_text,
        )

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            retrieval_metadata={
                "system": "mem0",
                "top_k": top_k,
                "dual_perspective": True,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "formatted_context": formatted_context,
                "speaker_a_memories_count": len(results_a),
                "speaker_b_memories_count": len(results_b),
            },
        )

    def _get_answer_prompt(self) -> str:
        """
        Return answer prompt.

        Uses generic default prompt (loaded from YAML).
        """
        dataset_id = str((self.run_context or {}).get("dataset_id", "")).strip()
        if dataset_id.startswith("subtlememory"):
            return get_subtlememory_unified_answer_prompt(self._prompts, config=self.config)
        return self._prompts["online_api"]["default"]["answer_prompt_mem0"]

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Mem0",
            "type": "online_api",
            "description": "Mem0 - Personalized AI Memory Layer",
            "adapter": "Mem0Adapter",
        }
