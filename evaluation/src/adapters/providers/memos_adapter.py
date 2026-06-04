"""
Memos Adapter - adapt Memos online API for evaluation framework.
Reference: https://www.memos.so/
"""

import asyncio
import json
import os
from pathlib import Path
import re
import ssl
from typing import Any, Dict, List, Optional

import aiohttp
from aiolimiter import AsyncLimiter
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


@register_adapter("memos")
class MemosAdapter(OnlineAPIAdapter):
    """
    Memos online API adapter.

    Supports:
    - Memory ingestion (supports conversation context)
    - Memory retrieval

    Official API supported parameters:
    - user_id (required) - Format: {conv_id}_{speaker}, already contains session info
    - query (required)
    - memory_limit_number (optional, default 6)

    Note: Does not use conversation_id parameter, as user_id already contains session info

    Config example:
    ```yaml
    adapter: "memos"
    api_url: "${MEMOS_URL}"
    api_key: "${MEMOS_KEY}"
    ```
    """

    MEMOS_DETAIL_LIST_KINDS = {
        "memory_detail_list": "memory",
        "preference_detail_list": "preference",
        "tool_memory_detail_list": "tool",
        "skill_detail_list": "skill",
    }

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # Get API configuration
        self.api_url = config.get("api_url", "")
        if not self.api_url:
            raise ValueError("Memos API URL is required. Set 'api_url' in config.")

        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Memos API key is required. Set 'api_key' in config.")

        self.headers = {"Content-Type": "application/json", "Authorization": api_key}

        # Retrieval configuration (only keep batch_size and max_retries, other params not supported by official API)
        self.batch_size = config.get("batch_size", 9999)  # Memos supports large batches
        self.max_retries = config.get("max_retries", 5)
        self.trust_env = bool(config.get("trust_env", False))
        self.ssl_ca_file = config.get("ssl_ca_file") or os.environ.get("SSL_CERT_FILE")

        # Rate limiting configuration (default: 10 requests/second)
        requests_per_second = config.get("requests_per_second", 10)
        self.rate_limiter = AsyncLimiter(max_rate=requests_per_second, time_period=1.0)

        # Create aiohttp session (will be initialized on first use)
        self._session: Optional[aiohttp.ClientSession] = None
        self._readback_user_cache: Dict[str, Dict[str, Any]] = {}
        self._readback_user_locks: Dict[str, asyncio.Lock] = {}

        self.console = Console()

        print(f"   API URL: {self.api_url}")
        print(f"   Rate Limit: {requests_per_second} requests/second (async)")

    @staticmethod
    def _normalize_task_status(status: Any, default: str = "submitted") -> str:
        value = str(status or "").strip().lower()
        if not value:
            return default
        aliases = {
            "done": "completed",
            "complete": "completed",
            "completed": "completed",
            "ready": "completed",
            "success": "completed",
            "running": "processing",
            "processing": "processing",
            "pending": "pending",
            "queued": "pending",
            "submitted": "submitted",
        }
        return aliases.get(value, value)

    @staticmethod
    def _memos_envelope_ok(result: Any) -> bool:
        """OpenMem JSON envelopes use numeric ``code``; ``0`` means success."""
        return isinstance(result, dict) and result.get("code") == 0

    def _extract_memory_id_from_status_payload(self, payload: Dict[str, Any]) -> str:
        return self._extract_memory_id_from_payload(payload)

    @staticmethod
    def _dedupe_nonempty(values: List[Any]) -> List[str]:
        normalized: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        return normalized

    @staticmethod
    def _memos_wire_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Memos /add/message accepts only role/content message objects."""
        cleaned: List[Dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user").strip() or "user"
            content = str(message.get("content") or "").strip()
            if content:
                cleaned.append({"role": role, "content": content})
        return cleaned

    @staticmethod
    def _message_session_id(message: Any) -> str:
        metadata = getattr(message, "metadata", {}) or {}
        return str(
            metadata.get("source_session_id")
            or metadata.get("session_id")
            or metadata.get("source_session")
            or ""
        ).strip()

    def _should_use_session_scoped_add(self, conv: Conversation) -> bool:
        del conv
        dataset_id = str((self.run_context or {}).get("dataset_id") or "").strip()
        return dataset_id.startswith("subtlememory")

    def _session_add_info(
        self,
        *,
        conv: Conversation,
        source_messages: List[Any],
        session_id: str,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {"sessionId": session_id}
        dataset_id = str((self.run_context or {}).get("dataset_id") or "").strip()
        if dataset_id:
            info["dataset"] = dataset_id
        for message in source_messages:
            metadata = getattr(message, "metadata", {}) or {}
            if metadata.get("source_case_id") and "case_id" not in info:
                info["case_id"] = metadata.get("source_case_id")
            if metadata.get("source_session_source") and "session_source" not in info:
                info["session_source"] = metadata.get("source_session_source")
            if metadata.get("session_order") is not None and "session_order" not in info:
                info["session_order"] = metadata.get("session_order")
        if conv.metadata.get("persona_id"):
            info["persona"] = conv.metadata.get("persona_id")
        return info

    @classmethod
    def _iter_payload_candidates(cls, payload: Any):
        """Yield nested payload containers returned by Memos status/readback APIs."""
        if isinstance(payload, dict):
            yield payload
            for key in (
                "data",
                "result",
                "results",
                "items",
                "memories",
                "memory_list",
            ):
                value = payload.get(key)
                if isinstance(value, (dict, list)):
                    yield from cls._iter_payload_candidates(value)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, (dict, list)):
                    yield from cls._iter_payload_candidates(item)

    @staticmethod
    def _looks_like_memory_identifier(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        return bool(
            re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                text,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _looks_like_memory_record(cls, payload: Dict[str, Any]) -> bool:
        return any(
            key in payload
            for key in (
                "memory_id",
                "memory_ids",
                "memory",
                "content",
                "memory_type",
                "created_at",
                "tags",
                "confidence",
            )
        )

    @classmethod
    def _extract_memory_id_from_payload(
        cls, payload: Any, *, task_id: Optional[str] = None
    ) -> str:
        blocked_id = str(task_id or "").strip()

        def _normalize_candidate(value: Any) -> str:
            text = str(value or "").strip()
            if not text or text == blocked_id:
                return ""
            return text

        for candidate in cls._iter_payload_candidates(payload):
            if not isinstance(candidate, dict):
                continue

            direct_id = _normalize_candidate(candidate.get("memory_id"))
            if direct_id:
                return direct_id

            memory_ids = candidate.get("memory_ids")
            if isinstance(memory_ids, list):
                for memory_id in memory_ids:
                    normalized = _normalize_candidate(memory_id)
                    if normalized:
                        return normalized

            if cls._looks_like_memory_record(candidate):
                record_id = _normalize_candidate(candidate.get("id"))
                if record_id:
                    return record_id

        if isinstance(payload, list):
            for item in payload:
                normalized = _normalize_candidate(item)
                if cls._looks_like_memory_identifier(normalized):
                    return normalized

        return ""

    @classmethod
    def _extract_task_status_from_payload(
        cls,
        payload: Any,
        *,
        default: str = "submitted",
        resolved_memory_id: str = "",
    ) -> str:
        for candidate in cls._iter_payload_candidates(payload):
            if not isinstance(candidate, dict):
                continue
            for key in ("status", "state", "processing_status"):
                normalized = cls._normalize_task_status(candidate.get(key), default="")
                if normalized:
                    return normalized
        if resolved_memory_id:
            return "completed"
        return default

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session (lazy initialization).

        Returns:
            aiohttp.ClientSession instance
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            ssl_context = self._create_ssl_context()
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout,
                trust_env=self.trust_env,
                connector=aiohttp.TCPConnector(ssl=ssl_context),
            )
        return self._session

    def _create_ssl_context(self) -> ssl.SSLContext:
        if self.ssl_ca_file:
            return ssl.create_default_context(cafile=self.ssl_ca_file)

        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    async def close(self):
        """
        Close aiohttp session.

        Should be called when adapter is no longer needed.
        """
        if self._session and not self._session.closed:
            await self._session.close()

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user to Memos.

        Args:
            conv: Original conversation object
            messages: Formatted message list
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters

        Returns:
            None
        """
        # Extract user_id and conv_id
        user_id = self._extract_user_id(conv, speaker=speaker)
        conv_id = conv.conversation_id

        # Log info
        sender_name = conv.metadata.get(speaker, speaker)
        self.console.print(
            f"   📤 Adding for {sender_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )

        # Get session
        session = await self._get_session()

        # Send messages in batches with retry
        url = f"{self.api_url}/add/message"
        receipts = []

        if self._should_use_session_scoped_add(conv):
            grouped: Dict[str, List[tuple[Dict[str, Any], Any]]] = {}
            for payload_message, source_message in zip(messages, conv.messages):
                session_id = self._message_session_id(source_message) or conv_id
                grouped.setdefault(session_id, []).append(
                    (payload_message, source_message)
                )

            for session_id, pairs in grouped.items():
                session_payloads = [payload for payload, _ in pairs]
                session_source_messages = [source for _, source in pairs]
                session_info = self._session_add_info(
                    conv=conv,
                    source_messages=session_source_messages,
                    session_id=session_id,
                )
                for i in range(0, len(session_payloads), self.batch_size):
                    batch_pairs = pairs[i : i + self.batch_size]
                    batch_messages = [payload for payload, _ in batch_pairs]
                    batch_sources = [source for _, source in batch_pairs]
                    batch_receipts = await self._send_message_batch(
                        url=url,
                        batch_messages=batch_messages,
                        user_id=user_id,
                        conv_id=session_id,
                        sender_name=sender_name,
                        session=session,
                        info=session_info,
                    )
                    source_unit_ids = [
                        source.metadata.get("source_unit_id")
                        for source in batch_sources
                        if source.metadata.get("source_unit_id")
                    ]
                    namespace_scope = dict(kwargs.get("namespace_scope") or {})
                    namespace_scope["conversation_id"] = session_id
                    namespace_scope["session_id"] = session_id
                    namespace_scope["namespace_id"] = user_id
                    for receipt in batch_receipts:
                        receipt.setdefault("conversation_id", session_id)
                        receipt.setdefault("session_id", session_id)
                        receipt.setdefault("source_unit_ids", source_unit_ids)
                        receipt.setdefault(
                            "chunk_id",
                            f"{conv.conversation_id}:{speaker}:{session_id}:chunk{i // self.batch_size}",
                        )
                        receipt.setdefault("namespace_scope", namespace_scope)
                        receipt.setdefault("metadata", {})
                        receipt["metadata"].update(
                            {
                                "session_id": session_id,
                                "source_conversation_id": conv.conversation_id,
                            }
                        )
                        for memory_ref in receipt.get("memory_refs", []) or []:
                            if isinstance(memory_ref, dict):
                                memory_ref.setdefault("session_id", session_id)
                    receipts.extend(batch_receipts)
            return receipts

        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]

            # Try to send the batch with automatic batch size reduction on token limit error
            batch_receipts = await self._send_message_batch(
                url=url,
                batch_messages=batch_messages,
                user_id=user_id,
                conv_id=conv_id,
                sender_name=sender_name,
                session=session,
            )
            source_unit_ids = [
                msg.metadata.get("source_unit_id")
                for msg in conv.messages[i : i + len(batch_messages)]
                if msg.metadata.get("source_unit_id")
            ]
            for receipt in batch_receipts:
                receipt.setdefault("source_unit_ids", source_unit_ids)
            receipts.extend(batch_receipts)

        return receipts

    async def _send_message_batch(
        self,
        url: str,
        batch_messages: List[Dict[str, Any]],
        user_id: str,
        conv_id: str,
        sender_name: str,
        session: aiohttp.ClientSession,
        info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Send a batch of messages to Memos API.

        Handles token limit exceeded errors by automatically reducing batch size to 2.

        Args:
            url: API endpoint URL
            batch_messages: Messages to send in this batch
            user_id: User ID
            conv_id: Conversation ID
            sender_name: Sender name (for logging)
            session: aiohttp session
        """
        wire_messages = self._memos_wire_messages(batch_messages)
        if not wire_messages:
            return []

        payload_dict = {
            "messages": wire_messages,
            "user_id": user_id,
            "conversation_id": conv_id,
        }
        if info:
            payload_dict["info"] = dict(info)

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                async with self.rate_limiter:
                    async with session.post(url, json=payload_dict) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()

                        # Check for token limit exceeded error
                        if (
                            result.get("code") == 40302
                            and result.get("message") == "Input token limit exceeded"
                        ):
                            # If batch size > 1, try splitting into smaller batches
                            if len(batch_messages) > 1:
                                # Determine new batch size: if current > 2, use 2; otherwise use 1
                                new_batch_size = 2 if len(batch_messages) > 2 else 1
                                self.console.print(
                                    f"   ⚠️  [{sender_name}] Token limit exceeded, splitting batch of {len(batch_messages)} into smaller batches (size={new_batch_size})",
                                    style="yellow",
                                )
                                # Recursively send in smaller batches
                                receipts = []
                                for j in range(0, len(batch_messages), new_batch_size):
                                    sub_batch = batch_messages[j : j + new_batch_size]
                                    receipts.extend(
                                        await self._send_message_batch(
                                            url=url,
                                            batch_messages=sub_batch,
                                            user_id=user_id,
                                            conv_id=conv_id,
                                            sender_name=sender_name,
                                            session=session,
                                            info=info,
                                        )
                                    )
                                return receipts
                            else:
                                # Batch size is 1, cannot split further
                                # Try truncating the message content by removing last 1000 characters
                                message = batch_messages[0]
                                original_content = message.get("content", "")

                                if len(original_content) > 1000:
                                    self.console.print(
                                        f"   ⚠️  [{sender_name}] Single message token limit exceeded, truncating content (removing last 1000 chars)",
                                        style="yellow",
                                    )
                                    # Create a truncated version of the message
                                    truncated_message = message.copy()
                                    truncated_message["content"] = original_content[
                                        :-1000
                                    ]

                                    # Try sending the truncated message
                                    return await self._send_message_batch(
                                        url=url,
                                        batch_messages=[truncated_message],
                                        user_id=user_id,
                                        conv_id=conv_id,
                                        sender_name=sender_name,
                                        session=session,
                                        info=info,
                                    )
                                else:
                                    # Content is already short, cannot truncate further
                                    raise Exception(
                                        f"API error (token limit, single message too large, content length={len(original_content)}): {result}"
                                    )

                        if not self._memos_envelope_ok(result):
                            raise Exception(f"API error: {result}")

                data = result.get("data", {}) if isinstance(result, dict) else {}
                task_id = data.get("task_id") or result.get("task_id")
                status = self._normalize_task_status(
                    data.get("status") or result.get("status") or "submitted"
                )
                return [
                    {
                        "provider_receipt": result,
                        "provider_status": status,
                        "memory_refs": (
                            [
                                {
                                    "provider": "memos",
                                    "task_id": task_id,
                                    "user_id": user_id,
                                    "conversation_id": conv_id,
                                }
                            ]
                            if task_id
                            else [
                                {
                                    "provider": "memos",
                                    "user_id": user_id,
                                    "conversation_id": conv_id,
                                }
                            ]
                        ),
                    }
                ]

            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.console.print(
                        f"   ⚠️  [{sender_name}] Retry {attempt + 1}/{self.max_retries}: {e}",
                        style="yellow",
                    )
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.console.print(
                        f"   ❌ [{sender_name}] Failed after {self.max_retries} retries: {e}",
                        style="red",
                    )
                    raise e

        return []

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (Memos-specific with preference extraction).

        Calls Memos HTTP API and extracts preference information.

        Args:
            query: Query text
            conversation_id: Conversation ID (not used by Memos, user_id contains this info)
            user_id: User ID to search for (format: {conv_id}_{speaker})
            top_k: Number of results to retrieve
            **kwargs: Additional parameters

        Returns:
            List of search results with preference information in metadata

        Note:
            user_id already contains session info (format: {conv_id}_{speaker}).
            Example: user_id="locomo_0_Caroline" uniquely identifies the locomo_0 conversation.
        """
        # Get session
        session = await self._get_session()

        # Prepare HTTP request
        url = f"{self.api_url}/search/memory"
        payload_dict = {
            "query": query,
            "user_id": user_id,
            "memory_limit_number": top_k,
        }

        # Call API with retry mechanism
        text_mem_res = []
        pref_string = ""

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                async with self.rate_limiter:
                    async with session.post(url, json=payload_dict) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()
                        if not self._memos_envelope_ok(result):
                            raise Exception(f"API error: {result}")

                        data = result.get("data", {})
                        text_mem_res = data.get("memory_detail_list", [])
                        pref_mem_res = data.get("preference_detail_list", [])
                        preference_note = data.get("preference_note", "")

                        # Standardize field names: rename memory_value to memory
                        for i in text_mem_res:
                            i.update(
                                {"memory": i.pop("memory_value", i.get("memory", ""))}
                            )

                        # Format preference string
                        explicit_prefs = [
                            p["preference"]
                            for p in pref_mem_res
                            if p.get("preference_type", "") == "explicit_preference"
                        ]
                        implicit_prefs = [
                            p["preference"]
                            for p in pref_mem_res
                            if p.get("preference_type", "") == "implicit_preference"
                        ]

                        pref_parts = []
                        if explicit_prefs:
                            pref_parts.append(
                                "Explicit Preference:\n"
                                + "\n".join(
                                    f"{i + 1}. {p}"
                                    for i, p in enumerate(explicit_prefs)
                                )
                            )
                        if implicit_prefs:
                            pref_parts.append(
                                "Implicit Preference:\n"
                                + "\n".join(
                                    f"{i + 1}. {p}"
                                    for i, p in enumerate(implicit_prefs)
                                )
                            )

                        pref_string = "\n".join(pref_parts) + preference_note

                # Success - break retry loop
                break

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.console.print(f"❌ Memos search error: {e}", style="red")
                    return []

        # Convert to standard format
        results = []
        for item in text_mem_res:
            created_at = item.get("memory_time") or item.get("create_time", "")
            results.append(
                {
                    "content": item.get("memory", ""),
                    "score": item.get("relativity", item.get("score", 0.0)),
                    "user_id": user_id,
                    "metadata": {
                        "memory_id": item.get("id", ""),
                        "created_at": str(created_at) if created_at else "",
                        "memory_type": item.get("memory_type", ""),
                        "confidence": item.get("confidence", 0.0),
                        "tags": item.get("tags", []),
                        "pref_string": pref_string,  # Store preference for this user
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
        Build SearchResult for single perspective (Memos: include preference).

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult with preference metadata (no formatted_context, uses fallback)
        """
        # Extract pref_string from first result's metadata (all results share same pref_string)
        pref_string = results[0]["metadata"]["pref_string"] if results else ""

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "memos",
                "preferences": {"pref_string": pref_string},
                "top_k": top_k,
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
        Build SearchResult for dual perspective (Memos: use template + preference).

        Formats memories using the default template, including preference information
        for both speakers.

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
            SearchResult with formatted_context and preferences
        """
        # Extract preferences from results' metadata
        pref_a = results_a[0]["metadata"]["pref_string"] if results_a else ""
        pref_b = results_b[0]["metadata"]["pref_string"] if results_b else ""

        # Build context for each speaker (memories + preferences)
        speaker_a_memories = (
            "\n".join([r["content"] for r in results_a])
            if results_a
            else "(No memories found)"
        )
        speaker_b_memories = (
            "\n".join([r["content"] for r in results_b])
            if results_b
            else "(No memories found)"
        )

        speaker_a_context = speaker_a_memories + (f"\n{pref_a}" if pref_a else "")
        speaker_b_context = speaker_b_memories + (f"\n{pref_b}" if pref_b else "")

        # Use default template
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
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
                "system": "memos",
                "dual_perspective": True,
                "formatted_context": formatted_context,
                "top_k": top_k,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "preferences": {"speaker_a_pref": pref_a, "speaker_b_pref": pref_b},
            },
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
        """Build a SearchResult from Memos /get/memory readback."""
        del index, conversation, kwargs
        question_metadata = question_metadata or {}
        session_ids = self._dedupe_nonempty(
            list(question_metadata.get("session_ids") or [])
        )
        if not session_ids:
            return SearchResult(
                question_id=question_id or "",
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "system": "memos",
                    "search_mode": "readback",
                    "session_ids": [],
                    "formatted_context": "",
                    "readback": {
                        "status": "unsupported",
                        "checked_session_ids": [],
                        "missing_session_ids": [],
                        "objects": [],
                        "metadata": {
                            "provider": "memos",
                            "question_id": question_id,
                            "readback_scope": "question_sessions",
                            "reason": "missing question session_ids",
                        },
                        "errors": [
                            {
                                "error_type": "missing_session_ids",
                                "error_message": (
                                    "Readback search requires question_metadata.session_ids."
                                ),
                            }
                        ],
                    },
                },
                retrieval_status="unsupported",
            )
        context = {
            "question_id": question_id,
            "conversation_id": conversation_id,
            "session_ids": session_ids,
            "import_manifest_rows": import_manifest_records or [],
            "qa_row": {"metadata": question_metadata},
        }
        readback = await self.get_storage_readback(
            session_ids=session_ids,
            question_id=question_id,
            context=context,
        )
        results, formatted_context = self._build_readback_search_payload(
            readback,
            session_ids=session_ids,
        )
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
                "system": "memos",
                "search_mode": "readback",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "user_ids": readback.metadata.get("user_ids", []),
                "readback": readback.to_dict(),
            },
            retrieval_status=status,
        )

    def _get_answer_prompt(self) -> str:
        """
        Get answer prompt.

        Subclasses can override this method to return their own prompt.
        Defaults to generic default prompt.
        """
        dataset_id = str((self.run_context or {}).get("dataset_id", "")).strip()
        if dataset_id.startswith("subtlememory"):
            return get_subtlememory_unified_answer_prompt(self._prompts, config=self.config)
        return self._prompts["online_api"]["default"]["answer_prompt_memos"]

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Memos",
            "type": "online_api",
            "description": "Memos - Memory System with Preference Support",
            "adapter": "MemosAdapter",
        }

    async def _get_task_status(self, task_id: str) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self.api_url}/get/status"
        async with self.rate_limiter:
            async with session.post(url, json={"task_id": task_id}) as response:
                if response.status == 200:
                    return await response.json()
                post_status = response.status
                post_text = await response.text()

        # Best-effort compatibility for older deployments that still expose
        # the legacy path-shaped status endpoint.
        if post_status not in {404, 405}:
            raise RuntimeError(f"HTTP {post_status}: {post_text}")

        legacy_url = f"{self.api_url}/get/status/{task_id}"
        async with self.rate_limiter:
            async with session.get(legacy_url) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {text}")
                return await response.json()

    async def _get_memory_by_id(self, memory_id: str) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self.api_url}/get/memory"
        async with self.rate_limiter:
            async with session.post(url, json={"memory_id": memory_id}) as response:
                if response.status == 200:
                    return await response.json()
            async with session.get(url, params={"memory_id": memory_id}) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {text}")
                return await response.json()

    async def _get_memories_for_user(
        self,
        user_id: str,
        *,
        page: int = 1,
        size: int = 50,
        include_preference: bool = True,
        include_tool_memory: bool = True,
    ) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self.api_url}/get/memory"
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
        async with self.rate_limiter:
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
        normalized_size = max(1, min(int(size), 50))

        while True:
            payload = await self._get_memories_for_user(
                user_id,
                page=page,
                size=normalized_size,
                include_preference=True,
                include_tool_memory=True,
            )
            if not self._memos_envelope_ok(payload):
                raise RuntimeError(f"Memos /get/memory API error: {payload}")
            pages.append(payload)

            data = payload.get("data", {}) or {}
            provider_pages = data.get("pages")
            try:
                provider_page_count = int(provider_pages)
            except (TypeError, ValueError):
                provider_page_count = 0

            current_rows = sum(
                len(data.get(list_key, []) or [])
                for list_key in self.MEMOS_DETAIL_LIST_KINDS
            )
            if max_pages is not None and page >= max_pages:
                break
            if provider_page_count and page >= provider_page_count:
                break
            if not provider_page_count and current_rows < normalized_size:
                break
            if current_rows == 0:
                break
            page += 1

        return {"user_id": user_id, "pages": pages}

    async def _get_cached_memos_readback_for_user(
        self, user_id: str
    ) -> Dict[str, Any]:
        """Return full user readback from memory cache, fetching once per run."""
        cached = self._readback_user_cache.get(user_id)
        if cached is not None:
            return cached

        lock = self._readback_user_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            cached = self._readback_user_cache.get(user_id)
            if cached is not None:
                return cached

            pages_payload = await self._get_all_memories_for_user(user_id)
            objects = [
                self._normalize_memos_storage_object(row, kind=kind)
                for kind, row in self._iter_memos_detail_rows(pages_payload)
            ]
            raw_counts = self._memos_raw_counts([pages_payload])
            cache_entry = {
                "user_id": user_id,
                "objects": objects,
                "raw_counts": raw_counts,
                "page_count": len(pages_payload.get("pages", []) or []),
            }
            self._readback_user_cache[user_id] = cache_entry
            return cache_entry

    @classmethod
    def _iter_memos_detail_rows(
        cls, pages_payload: Dict[str, Any]
    ) -> List[tuple[str, Dict[str, Any]]]:
        rows: List[tuple[str, Dict[str, Any]]] = []
        for page_payload in pages_payload.get("pages", []) or []:
            if not isinstance(page_payload, dict):
                continue
            data = page_payload.get("data", {}) or {}
            if not isinstance(data, dict):
                continue
            for list_key, kind in cls.MEMOS_DETAIL_LIST_KINDS.items():
                for row in data.get(list_key, []) or []:
                    if isinstance(row, dict):
                        rows.append((kind, row))
        return rows

    @staticmethod
    def _resolve_memos_row_session_id(row: Dict[str, Any]) -> str:
        info = row.get("info", {}) or {}
        if not isinstance(info, dict):
            info = {}
        session_key = str(info.get("sessionKey") or "").strip()
        if session_key.startswith("agent:"):
            parts = session_key.split(":", 2)
            if len(parts) == 3 and parts[2].strip():
                return parts[2].strip()
        for value in (
            row.get("conversation_id"),
            info.get("sessionId"),
            info.get("session_id"),
            info.get("source_session_id"),
            row.get("session_id"),
            row.get("source_session_id"),
        ):
            text = str(value or "").strip()
            if text:
                return text
        if session_key:
            return session_key
        return "unknown"

    @staticmethod
    def _extract_memos_row_content(row: Dict[str, Any]) -> str:
        for key in ("memory_value", "memory", "preference", "reasoning"):
            text = str(row.get(key) or "").strip()
            if text:
                return text
        source_contents: List[str] = []
        for source in row.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            content = str(source.get("content") or "").strip()
            if content:
                source_contents.append(content)
        return "\n".join(source_contents)

    @staticmethod
    def _extract_memos_prompt_content(row: Dict[str, Any]) -> str:
        """Return content safe for readback search prompt injection.

        Readback search must not inject raw ``sources[].content`` into the
        answer prompt.
        """
        for key in ("memory_value", "memory", "preference", "reasoning"):
            text = str(row.get(key) or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _memos_object_sort_value(obj: NormalizedStorageObject) -> tuple[int, int, str]:
        raw = obj.raw if isinstance(obj.raw, dict) else {}
        for key in ("create_time", "update_time", "memory_time"):
            value = raw.get(key)
            try:
                return (0, int(value), obj.id)
            except (TypeError, ValueError):
                continue
        return (1, 0, obj.id)

    def _sort_memos_readback_objects(
        self,
        objects: List[NormalizedStorageObject],
        *,
        session_ids: List[str],
    ) -> List[NormalizedStorageObject]:
        return self.sort_memos_readback_objects_from_config(
            self.config, objects, session_ids=session_ids
        )

    def _format_memos_preferences_for_context(
        self, preference_objects: List[NormalizedStorageObject]
    ) -> str:
        return self.format_memos_preferences_for_context(preference_objects)

    def _build_readback_search_payload(
        self,
        readback: StorageReadbackResult,
        *,
        session_ids: List[str],
    ) -> tuple[List[Dict[str, Any]], str]:
        return self.build_readback_search_payload_from_config(
            self.config, readback, session_ids=session_ids
        )

    @classmethod
    def build_readback_search_payload_from_config(
        cls,
        config: Dict[str, Any],
        readback: StorageReadbackResult,
        *,
        session_ids: List[str],
    ) -> tuple[List[Dict[str, Any]], str]:
        include_preferences = bool(
            ((config.get("search") or {}).get("readback") or {}).get(
                "include_preferences", True
            )
        )
        normalized_objects = [
            obj for obj in readback.objects if isinstance(obj, NormalizedStorageObject)
        ]
        ordered_objects = cls.sort_memos_readback_objects_from_config(
            config, normalized_objects, session_ids=session_ids
        )

        content_items = []
        for obj in ordered_objects:
            if obj.kind == "preference":
                continue
            raw = obj.raw if isinstance(obj.raw, dict) else {}
            prompt_content = cls._extract_memos_prompt_content(raw)
            if not prompt_content and not raw:
                prompt_content = str(obj.content or "").strip()
            if prompt_content:
                content_items.append((obj, prompt_content))
        preference_objects = [
            obj
            for obj in ordered_objects
            if obj.kind == "preference" and str(obj.content or "").strip()
        ]

        results: List[Dict[str, Any]] = []
        for obj, prompt_content in content_items:
            results.append(
                {
                    "content": prompt_content,
                    "score": 1.0,
                    "metadata": {
                        "memory_id": obj.id,
                        "kind": obj.kind,
                        "session_id": obj.session_id,
                        **(obj.metadata or {}),
                    },
                }
            )

        context_parts = [
            f"{index + 1}. {content}" for index, (_obj, content) in enumerate(content_items)
        ]
        if include_preferences:
            pref_string = cls.format_memos_preferences_for_context(preference_objects)
            if pref_string:
                context_parts.append(pref_string)
        return results, "\n\n".join(context_parts)

    @classmethod
    def sort_memos_readback_objects_from_config(
        cls,
        config: Dict[str, Any],
        objects: List[NormalizedStorageObject],
        *,
        session_ids: List[str],
    ) -> List[NormalizedStorageObject]:
        order = str(
            ((config.get("search") or {}).get("readback") or {}).get(
                "order", "session_then_time"
            )
        )
        if order != "session_then_time":
            return sorted(objects, key=cls._memos_object_sort_value)
        session_order = {session_id: index for index, session_id in enumerate(session_ids)}
        return sorted(
            objects,
            key=lambda obj: (
                session_order.get(obj.session_id, len(session_order)),
                *cls._memos_object_sort_value(obj),
            ),
        )

    @classmethod
    def format_memos_preferences_for_context(
        cls, preference_objects: List[NormalizedStorageObject]
    ) -> str:
        explicit: List[str] = []
        implicit: List[str] = []
        other: List[str] = []
        for obj in preference_objects:
            raw = obj.raw if isinstance(obj.raw, dict) else {}
            content = cls._extract_memos_prompt_content(raw)
            if not content and not raw:
                content = str(obj.content or "").strip()
            if not content:
                continue
            pref_type = str(
                (obj.metadata or {}).get("preference_type")
                or raw.get("preference_type")
                or ""
            ).strip()
            if pref_type == "explicit_preference":
                explicit.append(content)
            elif pref_type == "implicit_preference":
                implicit.append(content)
            else:
                other.append(content)

        parts: List[str] = []
        if explicit:
            parts.append(
                "Explicit Preference:\n"
                + "\n".join(f"{index + 1}. {text}" for index, text in enumerate(explicit))
            )
        if implicit:
            parts.append(
                "Implicit Preference:\n"
                + "\n".join(f"{index + 1}. {text}" for index, text in enumerate(implicit))
            )
        if other:
            parts.append(
                "Preference:\n"
                + "\n".join(f"{index + 1}. {text}" for index, text in enumerate(other))
            )
        return "\n".join(parts)

    @classmethod
    def _normalize_memos_storage_object(
        cls,
        row: Dict[str, Any],
        *,
        kind: str,
    ) -> NormalizedStorageObject:
        session_id = cls._resolve_memos_row_session_id(row)
        source_snippets: List[str] = []
        for source in row.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            content = str(source.get("content") or "").strip()
            if content:
                source_snippets.append(content)
        return NormalizedStorageObject(
            session_id=session_id,
            kind=kind,
            id=str(row.get("id") or "").strip(),
            content=cls._extract_memos_row_content(row),
            metadata={
                "provider": "memos",
                "conversation_id": row.get("conversation_id"),
                "memory_key": row.get("memory_key"),
                "memory_type": row.get("memory_type"),
                "preference_type": row.get("preference_type"),
                "confidence": row.get("confidence"),
                "status": row.get("status"),
                "tags": row.get("tags", []) or [],
                "source_snippets": source_snippets,
            },
            raw=dict(row),
        )

    @staticmethod
    def _collect_memos_user_ids_from_manifest_rows(
        manifest_rows: List[Dict[str, Any]],
    ) -> List[str]:
        candidates: List[Any] = []

        def collect_from_receipt(receipt: Any) -> None:
            if not isinstance(receipt, dict):
                return
            namespace_scope = receipt.get("namespace_scope", {}) or {}
            if isinstance(namespace_scope, dict):
                candidates.append(namespace_scope.get("namespace_id"))
            for ref in receipt.get("memory_refs", []) or []:
                if isinstance(ref, dict):
                    candidates.append(ref.get("user_id"))

        for row in manifest_rows:
            if not isinstance(row, dict):
                continue
            namespace_scope = row.get("namespace_scope", {}) or {}
            if isinstance(namespace_scope, dict):
                candidates.append(namespace_scope.get("namespace_id"))
            for ref in row.get("memory_refs", []) or []:
                if isinstance(ref, dict):
                    candidates.append(ref.get("user_id"))
            collect_from_receipt(row.get("write_receipt"))

        return MemosAdapter._dedupe_nonempty(candidates)

    def _find_memos_user_ids_for_readback(
        self,
        *,
        user_id: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> List[str]:
        context = context or {}
        primary_candidates: List[Any] = [user_id]
        manifest_rows = context.get("import_manifest_rows") or []
        primary_candidates.extend(
            self._collect_memos_user_ids_from_manifest_rows(manifest_rows)
        )
        search_result = context.get("search_result", {}) or {}
        retrieval_metadata = search_result.get("retrieval_metadata", {}) or {}
        if isinstance(retrieval_metadata, dict):
            primary_candidates.extend(retrieval_metadata.get("user_ids") or [])
        primary_user_ids = self._dedupe_nonempty(primary_candidates)
        if primary_user_ids:
            return primary_user_ids

        fallback_candidates: List[Any] = []
        qa_row = context.get("qa_row", {}) or {}
        qa_metadata = qa_row.get("metadata", {}) or {}
        target_session_ids = self._dedupe_nonempty(
            list(context.get("session_ids") or [])
        )
        conversation_id = str(context.get("conversation_id") or "").strip()
        if conversation_id:
            for speaker in ("speaker_a", "speaker_b"):
                fallback_candidates.append(
                    self._namespace_cache.get((conversation_id, speaker))
                )
                for session_id in target_session_ids:
                    fallback_candidates.append(
                        self._apply_run_suffix(
                            self._build_legacy_entity_id(
                                conversation_id=session_id,
                                speaker=speaker,
                                sender_name=qa_metadata.get(speaker) or speaker,
                            )
                        )
                    )
        return self._dedupe_nonempty(fallback_candidates)

    @classmethod
    def _memos_raw_counts(cls, pages_payloads: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {list_key: 0 for list_key in cls.MEMOS_DETAIL_LIST_KINDS}
        for pages_payload in pages_payloads:
            for page_payload in pages_payload.get("pages", []) or []:
                data = page_payload.get("data", {}) or {}
                if not isinstance(data, dict):
                    continue
                for list_key in counts:
                    counts[list_key] += len(data.get(list_key, []) or [])
        return counts

    @classmethod
    def _merge_memos_raw_counts(cls, raw_counts: Any) -> Dict[str, int]:
        counts = {list_key: 0 for list_key in cls.MEMOS_DETAIL_LIST_KINDS}
        for item in raw_counts:
            if not isinstance(item, dict):
                continue
            for list_key in counts:
                counts[list_key] += int(item.get(list_key, 0) or 0)
        return counts

    @staticmethod
    def _filter_memos_memory_rows(
        rows: Any,
        *,
        conversation_id: str = "",
        memory_id: str = "",
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        target_conversation_id = str(conversation_id or "").strip()
        target_memory_id = str(memory_id or "").strip()

        for item in rows or []:
            if not isinstance(item, dict):
                continue
            item_conversation_id = str(item.get("conversation_id") or "").strip()
            if (
                target_conversation_id
                and item_conversation_id
                and item_conversation_id != target_conversation_id
            ):
                continue
            filtered.append(item)

        if target_memory_id:
            exact = [
                item
                for item in filtered
                if str(item.get("id") or "").strip() == target_memory_id
            ]
            if exact:
                return exact

        return filtered

    @staticmethod
    def _compact_memos_readback_row(item: Dict[str, Any]) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}

        for key in (
            "id",
            "conversation_id",
            "memory_key",
            "memory_type",
            "status",
            "confidence",
            "preference_type",
            "preference",
            "reasoning",
        ):
            value = item.get(key)
            if value not in (None, "", [], {}):
                compact[key] = value

        memory_value = item.get("memory")
        if memory_value in (None, ""):
            memory_value = item.get("memory_value")
        if memory_value not in (None, ""):
            compact["memory"] = memory_value

        tags = item.get("tags")
        if isinstance(tags, list) and tags:
            compact["tags"] = tags

        source_contents = []
        for source in item.get("sources", []) or []:
            if not isinstance(source, dict):
                continue
            content = str(source.get("content") or "").strip()
            if content:
                source_contents.append(content)
            if len(source_contents) >= 3:
                break
        if source_contents:
            compact["source_snippets"] = source_contents

        return compact

    @staticmethod
    def _extract_human_readable_text(row: Dict[str, Any]) -> str:
        """Extract human-readable text content from a memory/preference row.

        Priority:
        1. memory / memory_value (core factual content)
        2. preference (for preference rows)
        3. reasoning (as supplementary context)

        Returns empty string if no readable content found.
        """
        parts: List[str] = []

        memory_text = str(row.get("memory") or row.get("memory_value") or "").strip()
        if memory_text:
            parts.append(memory_text)

        preference_text = str(row.get("preference") or "").strip()
        if preference_text and preference_text not in parts:
            parts.append(preference_text)

        if not parts:
            reasoning_text = str(row.get("reasoning") or "").strip()
            if reasoning_text:
                parts.append(reasoning_text)

        return " ".join(parts)

    @staticmethod
    def _render_memos_readback_content(
        payload: Any,
        *,
        conversation_id: str = "",
        memory_id: str = "",
    ) -> tuple[str, Dict[str, Any]]:
        """Render Memos readback payload into human-readable content.

        Returns:
            (content, metadata) where content is human-readable text
            (not JSON), and metadata contains audit info.
        """
        if not isinstance(payload, dict):
            return "", {"raw_payload_type": type(payload).__name__}

        data = payload.get("data", {}) or {}
        memory_rows = MemosAdapter._filter_memos_memory_rows(
            data.get("memory_detail_list", []),
            conversation_id=conversation_id,
            memory_id=memory_id,
        )
        preference_rows = [
            item
            for item in (data.get("preference_detail_list", []) or [])
            if isinstance(item, dict)
        ]
        tool_rows = [
            item
            for item in (data.get("tool_memory_detail_list", []) or [])
            if isinstance(item, dict)
        ]

        selected_rows: List[Dict[str, Any]] = []
        selected_rows.extend(memory_rows)
        if not memory_id:
            selected_rows.extend(preference_rows)
            selected_rows.extend(tool_rows)

        base_metadata = {
            "matched_memory_ids": [],
            "memory_row_count": len(memory_rows),
            "preference_row_count": len(preference_rows),
            "tool_row_count": len(tool_rows),
        }

        if not selected_rows:
            return "", base_metadata

        rendered_texts: List[str] = []
        matched_memory_ids: List[str] = []
        compact_rows_for_audit: List[Dict[str, Any]] = []

        for item in selected_rows:
            compact_row = MemosAdapter._compact_memos_readback_row(dict(item))
            compact_rows_for_audit.append(compact_row)

            row_id = str(compact_row.get("id") or "").strip()
            if row_id:
                matched_memory_ids.append(row_id)

            readable_text = MemosAdapter._extract_human_readable_text(compact_row)
            if readable_text:
                rendered_texts.append(readable_text)

        content = "\n".join(rendered_texts)

        return content, {
            **base_metadata,
            "matched_memory_ids": matched_memory_ids,
            "compact_rows": compact_rows_for_audit,
            "extracted_text_count": len(rendered_texts),
        }

    async def _resolve_memory_ref(
        self, memory_ref: Dict[str, Any], *, status_payload: Optional[Any] = None
    ) -> tuple[Dict[str, Any], str]:
        """Resolve task-based references to a concrete memory_id when possible."""
        resolved_ref = dict(memory_ref)
        memory_id = resolved_ref.get("memory_id")
        if memory_id:
            return resolved_ref, "completed"

        task_id = resolved_ref.get("task_id")
        if not task_id:
            return resolved_ref, "missing_memory_id"

        payload = status_payload or await self._get_task_status(str(task_id))
        memory_id = self._extract_memory_id_from_payload(
            payload,
            task_id=str(task_id),
        )
        status = self._extract_task_status_from_payload(
            payload,
            default="submitted",
            resolved_memory_id=memory_id,
        )
        if memory_id:
            resolved_ref["memory_id"] = memory_id
        return resolved_ref, status

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
        """Collect Memos storage objects via paginated /get/memory."""
        del run_id, evidence_texts, kwargs
        target_session_ids = self._dedupe_nonempty(list(session_ids or []))
        target_session_set = set(target_session_ids)
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
                    "provider": "memos",
                    "question_id": question_id,
                    "readback_scope": "session" if target_session_ids else "user",
                    "reason": "missing user_id/namespace",
                },
                errors=[
                    {
                        "error_type": "missing_user_id",
                        "error_message": "Cannot determine Memos user_id for storage readback.",
                    }
                ],
            )

        all_objects: List[NormalizedStorageObject] = []
        fetched_payloads: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

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
            obj.session_id for obj in all_objects if obj.session_id and obj.session_id != "unknown"
        }
        missing_session_ids = [
            session_id
            for session_id in target_session_ids
            if session_id not in object_session_ids
        ]
        raw_counts = self._merge_memos_raw_counts(
            payload.get("raw_counts", {}) for payload in fetched_payloads
        )
        page_count = sum(int(payload.get("page_count", 0) or 0) for payload in fetched_payloads)
        status = "error" if errors and not fetched_payloads else "ok"

        return StorageReadbackResult(
            status=status,
            checked_session_ids=target_session_ids,
            missing_session_ids=missing_session_ids,
            objects=all_objects,
            metadata={
                "provider": "memos",
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

    async def probe_readiness(self, add_result: Any = None, **kwargs) -> Dict[str, Any]:
        """Use task-status polling when task ids are available."""
        manifest_rows = kwargs.get("import_manifest_records", []) or []
        task_ids = []
        for row in manifest_rows:
            for memory_ref in row.get("memory_refs", []):
                task_id = memory_ref.get("task_id")
                if task_id:
                    task_ids.append(task_id)

        if not task_ids:
            return {
                "supported": False,
                "ready": False,
                "status": "unsupported",
                "details": {},
            }

        statuses = {}
        ready = True
        for task_id in task_ids:
            try:
                result = await self._get_task_status(task_id)
                resolved_memory_id = self._extract_memory_id_from_payload(
                    result,
                    task_id=str(task_id),
                )
                status = self._extract_task_status_from_payload(
                    result,
                    default="submitted",
                    resolved_memory_id=resolved_memory_id,
                )
                statuses[task_id] = status
                if status.lower() != "completed":
                    ready = False
            except Exception as exc:  # noqa: BLE001
                statuses[task_id] = f"error:{exc}"
                ready = False

        return {
            "supported": True,
            "ready": ready,
            "status": "ready" if ready else "processing",
            "details": {"task_statuses": statuses},
        }

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
        """Finalize Memos writes as a manual-confirmation gate with best-effort enrichment."""
        del add_result, budget_seconds, poll_interval_seconds, dataset, kwargs

        provider_status_counts: Dict[str, int] = {}
        missing_memory_refs: List[str] = []
        warnings: List[str] = [
            "Memos finalize skips task-status readiness gating; continue only after manual provider confirmation that add has completed."
        ]
        updated_rows = 0
        task_status_cache: Dict[str, Any] = {}
        refreshed_rows: List[Dict[str, Any]] = []

        for row in import_manifest_rows:
            updated_row = dict(row)
            receipt = dict(updated_row.get("write_receipt", {}) or {})
            memory_refs = [
                dict(item) for item in (updated_row.get("memory_refs") or [])
            ]
            row_errors = list(updated_row.get("errors", []) or [])
            status = self._normalize_task_status(
                updated_row.get("write_status")
                or receipt.get("provider_status")
                or "submitted"
            )

            resolved_refs: List[Dict[str, Any]] = []
            for memory_ref in memory_refs:
                task_id = memory_ref.get("task_id")
                if not task_id:
                    resolved_refs.append(dict(memory_ref))
                    continue
                try:
                    if task_id not in task_status_cache:
                        task_status_cache[task_id] = await self._get_task_status(
                            str(task_id)
                        )
                    status_payload = task_status_cache[task_id]
                    resolved_ref, ref_status = await self._resolve_memory_ref(
                        memory_ref,
                        status_payload=status_payload,
                    )
                    resolved_refs.append(resolved_ref)
                    if ref_status == "completed":
                        status = "completed"
                except Exception as exc:  # noqa: BLE001
                    row_errors.append(
                        {
                            "stage": "finalize",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    warnings.append(
                        f"{updated_row.get('chunk_id', '')}: task-status enrichment failed ({type(exc).__name__}); using manual confirmation flow"
                    )
                    resolved_refs.append(dict(memory_ref))

            if not resolved_refs:
                missing_memory_refs.append(str(updated_row.get("chunk_id", "")))
                warnings.append(
                    f"{updated_row.get('chunk_id', '')}: missing task/memory refs during finalize"
                )
            elif not any(ref.get("memory_id") for ref in resolved_refs):
                missing_memory_refs.append(str(updated_row.get("chunk_id", "")))
                if status == "completed":
                    warnings.append(
                        f"{updated_row.get('chunk_id', '')}: completed task without stable memory_id"
                    )

            receipt["provider_status"] = status
            updated_row["write_receipt"] = receipt
            updated_row["memory_refs"] = resolved_refs
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

    @staticmethod
    def _extract_content_from_single_memory_payload(
        payload: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """Extract human-readable content from a single memory payload.

        Used for the _get_memory_by_id() fallback path.

        Returns:
            (content, metadata) where content is human-readable text.
        """
        if not isinstance(payload, dict):
            return "", {"raw_payload_type": type(payload).__name__}

        data = payload.get("data", payload)
        if isinstance(data, dict):
            readable = MemosAdapter._extract_human_readable_text(data)
            if readable:
                row_id = str(data.get("id") or "").strip()
                return readable, {
                    "matched_memory_ids": [row_id] if row_id else [],
                    "extraction_source": "single_memory",
                }

        if isinstance(data, list):
            texts: List[str] = []
            ids: List[str] = []
            for item in data:
                if isinstance(item, dict):
                    readable = MemosAdapter._extract_human_readable_text(item)
                    if readable:
                        texts.append(readable)
                    row_id = str(item.get("id") or "").strip()
                    if row_id:
                        ids.append(row_id)
            if texts:
                return "\n".join(texts), {
                    "matched_memory_ids": ids,
                    "extraction_source": "memory_list",
                }

        return "", {"extraction_source": "failed"}
