"""
EverMemOS HTTP Memory API adapter (evaluation side).

This adapter talks to EverMemOS server endpoints:
- POST   /api/v1/memories         (ingest personal messages)
- POST   /api/v1/memories/group   (ingest group messages)
- POST   /api/v1/memories/search  (retrieve memories)
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from evaluation.src.adapters.shared.online_base import OnlineAPIAdapter
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.shared.subtlememory_prompts import (
    get_subtlememory_unified_answer_prompt,
)
from evaluation.src.adapters.shared.evermemos_readback import (
    get_evermemos_storage_readback,
)
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.core.readback import (
    NormalizedStorageObject,
    StorageReadbackResult,
)


EVERMEMOS_API_ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references]
- Original reference: [e.g., "last year" from May 2022]
- Calculated actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 7: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the concise answer with ALL specific details preserved]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""


@register_adapter("evermemos_api")
class EverMemOSAPIAdapter(OnlineAPIAdapter):
    """
    Adapter for EverMemOS Memory API.

    Design:
    - Ingest each conversation once (do NOT duplicate per-speaker perspectives).
    - Retrieval is controlled by system config: search.scope = "personal" | "group".
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.api_key = str(config.get("api_key", "") or "")
        self.sync_mode = bool(config.get("sync_mode", False))
        self.max_retries = int(config.get("max_retries", 3))
        self.timeout_seconds = float(config.get("timeout_seconds", 60))
        self.request_interval = float(config.get("request_interval", 0.0))
        self.trust_env = bool(config.get("trust_env", False))
        self.ssl_verify = bool(config.get("ssl_verify", True))

        self._session: Optional[aiohttp.ClientSession] = None

        self._api_base_url = self._normalize_api_base_url(self.base_url)
        self._personal_memories_url = self._api_base_url.rstrip("/") + "/memories"
        self._group_memories_url = self._api_base_url.rstrip("/") + "/memories/group"
        self._personal_flush_url = self._api_base_url.rstrip("/") + "/memories/flush"
        self._group_flush_url = self._api_base_url.rstrip("/") + "/memories/group/flush"
        self._get_url = self._api_base_url.rstrip("/") + "/memories/get"
        self._search_url = self._api_base_url.rstrip("/") + "/memories/search"

        print(f"   Memory API: {self._api_base_url}")

    # --- override add() to support clean_groups ---
    async def add(
        self, conversations: List[Conversation], **kwargs: Any
    ) -> Dict[str, Any]:
        """Override to support clean_groups config before ingestion."""
        if self.config.get("clean_groups"):
            from evaluation.src.utils.cleaner import clear_group_data

            print("\n🧹 clean_groups enabled, clearing data for involved groups...")
            for conv in conversations:
                await clear_group_data(conv.conversation_id, verbose=True)
            print()
        return await super().add(conversations, **kwargs)

    # --- lifecycle ---
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = None
        if not self.ssl_verify:
            connector = aiohttp.TCPConnector(ssl=False)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            trust_env=self.trust_env,
            connector=connector,
        )
        return self._session

    # --- helpers ---
    @staticmethod
    def _normalize_api_base_url(base_url: str) -> str:
        url = (base_url or "").rstrip("/")
        if not url:
            return ""

        exact_suffixes = (
            "/api/v0/memories/group",
            "/api/v0/memories/search",
            "/api/v0/memories",
            "/api/v1/memories/group",
            "/api/v1/memories/search",
            "/api/v1/memories",
        )
        for suffix in exact_suffixes:
            if url.endswith(suffix):
                return url[: -len(suffix)] + "/api/v1"

        if url.endswith("/api/v0"):
            return url[: -len("/api/v0")] + "/api/v1"
        if url.endswith("/api/v1"):
            return url

        return url + "/api/v1"

    @staticmethod
    def _normalize_memory_types(memory_types: Any) -> List[str]:
        if not memory_types:
            return ["episodic_memory"]
        if isinstance(memory_types, str):
            return [memory_types]
        return [str(item) for item in memory_types]

    @staticmethod
    def _timestamp_to_unix_ms(timestamp: Any) -> int:
        return int(timestamp.timestamp() * 1000)

    @classmethod
    def _infer_message_role(
        cls, sender_id: Optional[str], sender_name: Optional[str]
    ) -> str:
        candidates = [str(value).lower() for value in (sender_id, sender_name) if value]
        for candidate in candidates:
            if "assistant" in candidate or "bot" in candidate:
                return "assistant"
            tokens = re.findall(r"[a-z0-9]+", candidate)
            if "ai" in tokens:
                return "assistant"
        return "user"

    @classmethod
    def _format_memory_content(cls, memory: Dict[str, Any]) -> str:
        timestamp = str(memory.get("timestamp") or "").strip()
        text = str(
            memory.get("episode")
            or memory.get("summary")
            or memory.get("subject")
            or ""
        ).strip()
        if timestamp and text:
            return f"{timestamp}: {text}"
        return timestamp or text

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _search_scope(config: Dict[str, Any]) -> str:
        search_cfg = config.get("search", {}) or {}
        return str(search_cfg.get("scope", "personal")).lower()

    @staticmethod
    def _provider_retrieve_method(config: Dict[str, Any]) -> str:
        """EverMemOS provider method is separate from evaluation search.mode."""
        search_cfg = config.get("search", {}) or {}
        method = str(search_cfg.get("retrieve_method") or "").strip()
        return method or "keyword"

    @staticmethod
    def _clean_id(value: Any) -> str:
        return str(value or "").strip()

    @classmethod
    def _session_filter_value(cls, session_ids: Any) -> Any:
        normalized: List[str] = []
        if session_ids is None:
            return None
        if isinstance(session_ids, (str, int, float)):
            raw_values = [session_ids]
        else:
            raw_values = list(session_ids or [])
        seen: set[str] = set()
        for value in raw_values:
            clean = cls._clean_id(value)
            if clean and clean not in seen:
                normalized.append(clean)
                seen.add(clean)
        if not normalized:
            return None
        if len(normalized) == 1:
            return normalized[0]
        return {"in": normalized}

    @classmethod
    def _extract_question_session_ids(cls, question_metadata: Any) -> List[str]:
        if not isinstance(question_metadata, dict):
            return []
        raw = (
            question_metadata.get("session_ids")
            or question_metadata.get("session_id")
            or question_metadata.get("source_session_ids")
            or question_metadata.get("source_session_id")
        )
        value = cls._session_filter_value(raw)
        if isinstance(value, dict):
            return list(value.get("in") or [])
        if value:
            return [str(value)]
        return []

    @classmethod
    def _message_session_id(cls, message: Dict[str, Any]) -> str:
        return cls._clean_id(
            message.get("source_session_id")
            or message.get("session_id")
            or message.get("run_id")
        )

    @staticmethod
    def _provider_message(
        message: Dict[str, Any], *, owner_user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        local_only_keys = {
            "source_unit_id",
            "source_session_id",
            "source_case_id",
            "source_session_source",
            "source_session_timestamp",
            "session_order",
            "session_id",
            "run_id",
        }
        return {
            key: value
            for key, value in message.items()
            if key not in local_only_keys and value is not None
        }

    @classmethod
    def _provider_personal_message(
        cls, message: Dict[str, Any], owner_user_id: str
    ) -> Dict[str, Any]:
        provider_message = cls._provider_message(message, owner_user_id=owner_user_id)
        role = str(provider_message.get("role") or "").strip()
        if role == "user":
            provider_message["sender_id"] = owner_user_id
            if not provider_message.get("sender_name"):
                provider_message["sender_name"] = owner_user_id
        elif role == "assistant" and provider_message.get("sender_id") == owner_user_id:
            provider_message.pop("sender_id", None)
        return provider_message

    def _build_v1_message(
        self, conversation_id: str, msg: Any, idx: int
    ) -> Dict[str, Any]:
        sender_id = msg.sender_id or self._speaker_to_user_id(
            conversation_id, msg.sender_name
        )
        sender_name = msg.sender_name or sender_id
        message_id = (
            msg.metadata.get("message_id")
            or msg.metadata.get("dia_id")
            or f"{conversation_id}_{idx}"
        )

        return {
            "message_id": str(message_id),
            "sender_id": sender_id,
            "sender_name": sender_name,
            "role": self._infer_message_role(sender_id, sender_name),
            "timestamp": self._timestamp_to_unix_ms(msg.timestamp),
            "content": msg.content,
            "source_unit_id": msg.metadata.get("source_unit_id", str(message_id)),
            "source_session_id": msg.metadata.get("source_session_id"),
            "source_case_id": msg.metadata.get("source_case_id"),
            "source_session_source": msg.metadata.get("source_session_source"),
            "source_session_timestamp": msg.metadata.get("source_session_timestamp"),
            "session_order": msg.metadata.get("session_order"),
        }

    def _build_ingest_payload(
        self, conversation: Conversation, message: Dict[str, Any]
    ) -> Dict[str, Any]:
        scope = self._search_scope(self.config)
        if scope == "group":
            namespace_scope = self._build_namespace_scope(
                conversation, speaker="speaker_a"
            )
            payload = {
                "group_id": namespace_scope["namespace_id"],
                "messages": [self._provider_message(message)],
            }
            if self.sync_mode:
                payload["async_mode"] = False
            return payload

        owner_user_id = self._extract_user_id(conversation, speaker="speaker_a")
        payload = {
            "user_id": owner_user_id,
            "messages": [
                self._provider_personal_message(message, owner_user_id)
            ],
        }
        session_id = self._message_session_id(message)
        if session_id:
            payload["session_id"] = session_id
        if self.sync_mode:
            payload["async_mode"] = False
        return payload

    def _build_search_payload(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        top_k: int,
        question_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Main answer retrieval searches the full persona/run namespace.
        # QA session ids are reserved for readback search, not provider search scope.
        del question_metadata
        search_cfg = self.config.get("search", {}) or {}
        scope = self._search_scope(self.config)
        namespace_cache = getattr(self, "_namespace_cache", {})
        group_id = namespace_cache.get((conversation_id, "speaker_a"), conversation_id)
        filters = {"group_id": group_id} if scope == "group" else {"user_id": user_id}

        return {
            "query": query,
            "method": self._provider_retrieve_method(self.config),
            "memory_types": self._normalize_memory_types(
                search_cfg.get("memory_types", [])
            ),
            "top_k": int(top_k),
            "filters": filters,
        }

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
            "queued": "queued",
            "queue": "queued",
            "pending": "pending",
            "processing": "processing",
            "running": "processing",
            "submitted": "submitted",
            "not_found": "not_found",
        }
        return aliases.get(value, value)

    def _extract_provider_status(
        self, payload: Dict[str, Any], default: str = "submitted"
    ) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        candidates = [
            payload.get("status") if isinstance(payload, dict) else None,
            data.get("status") if isinstance(data, dict) else None,
            payload.get("message") if isinstance(payload, dict) else None,
            data.get("message") if isinstance(data, dict) else None,
        ]
        for candidate in candidates:
            normalized = self._normalize_provider_status(candidate, default="")
            if normalized:
                return normalized
        return default

    def _memory_ref_filters(
        self,
        memory_ref: Dict[str, Any],
        namespace_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the actual provider filters used for readback/finalize probes."""
        group_id = str(
            memory_ref.get("group_id") or namespace_scope.get("namespace_id") or ""
        ).strip()
        user_id = str(memory_ref.get("user_id") or "").strip()
        session_id = str(
            memory_ref.get("session_id")
            or memory_ref.get("source_session_id")
            or memory_ref.get("run_id")
            or ""
        ).strip()
        if group_id:
            filters: Dict[str, Any] = {"group_id": group_id}
            if session_id and session_id != "-1":
                filters["session_id"] = session_id
            return filters
        if user_id:
            filters = {"user_id": user_id}
            if session_id and session_id != "-1":
                filters["session_id"] = session_id
            return filters
        return {}

    @staticmethod
    def _filter_cache_key(filters: Dict[str, Any]) -> str:
        return json.dumps(filters, sort_keys=True, ensure_ascii=True, default=str)

    def _parse_search_response(
        self, data: Dict[str, Any], user_id: str, conversation_id: str
    ) -> List[Dict[str, Any]]:
        payload = (data or {}).get("data") or {}
        episodes = payload.get("episodes") or []

        results_out: List[Dict[str, Any]] = []
        for episode in episodes:
            if not isinstance(episode, dict):
                continue

            results_out.append(
                {
                    "content": self._format_memory_content(episode),
                    "score": self._safe_float(episode.get("score")),
                    "user_id": episode.get("user_id") or user_id,
                    "metadata": {
                        "group_id": episode.get("group_id") or conversation_id,
                        "raw": episode,
                    },
                }
            )

        results_out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results_out

    @staticmethod
    def _build_probe_queries(memory_ref: Dict[str, Any]) -> List[str]:
        """Build a few best-effort query variants for finalize/readback probing."""
        candidates: List[str] = []
        seen: set[str] = set()

        def add_candidate(value: Any) -> None:
            text = str(value or "").strip()
            if len(text) < 3 or text in seen:
                return
            seen.add(text)
            candidates.append(text)

        content = str(memory_ref.get("content") or "").strip()
        add_candidate(content)
        if content:
            first_sentence = re.split(r"[.!?]\s+", content, maxsplit=1)[0].strip()
            add_candidate(first_sentence)
            if len(content) > 96:
                truncated = (
                    content[:96].rsplit(" ", 1)[0].strip() or content[:96].strip()
                )
                add_candidate(truncated)

        add_candidate(memory_ref.get("message_id"))
        add_candidate(memory_ref.get("sender_id"))
        return candidates[:4]

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request_json_with_retry(
        self, method: str, url: str, **kwargs: Any
    ) -> Dict[str, Any]:
        session = await self._get_session()
        req = getattr(session, method)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with req(url, **kwargs) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(
                            f"{method.upper()} {url} -> {resp.status}: {text[:800]}"
                        )
                    if not text:
                        return {}
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        # Some gateways may return wrong content-type; still parse as JSON.
                        return await resp.json(content_type=None)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise
        raise last_exc or RuntimeError("request failed")

    @staticmethod
    def _speaker_to_user_id(conversation_id: str, sender_name: str) -> str:
        # Align with evaluation loader sender_id style: "{speaker_lower}_{conv_id}"
        return f"{sender_name.lower().replace(' ', '_')}_{conversation_id}"

    # --- overrides to avoid per-speaker duplication on ingest/search ---
    def _need_dual_perspective(self, speaker_a: str, speaker_b: str) -> bool:
        # EverMemOS Memory API stores group chat stream; do not split perspectives.
        return False

    def _conversation_to_messages(
        self,
        conversation: Conversation,
        format_type: str = "basic",
        perspective: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del format_type, perspective
        conv_id = conversation.conversation_id

        out: List[Dict[str, Any]] = []
        for idx, msg in enumerate(conversation.messages):
            if not msg.timestamp:
                continue

            out.append(self._build_v1_message(conv_id, msg, idx))
        return out

    def _get_answer_prompt(self) -> str:
        """Use EverMemOS CoT answer prompt (same as evermemos adapter)."""
        dataset_id = str((self.run_context or {}).get("dataset_id", "")).strip()
        if dataset_id.startswith("subtlememory"):
            return get_subtlememory_unified_answer_prompt(config=self.config)
        return EVERMEMOS_API_ANSWER_PROMPT

    # --- required abstract methods (OnlineAPIAdapter hooks) ---
    def _uses_incremental_import_manifest(self) -> bool:
        return True

    async def _add_user_messages(
        self,
        conv: Conversation,
        messages: List[Dict[str, Any]],
        speaker: str,
        **kwargs: Any,
    ) -> Any:
        if not self._api_base_url:
            raise ValueError("base_url is empty; set system config 'base_url'")

        progress = kwargs.get("progress")
        task_id = kwargs.get("task_id")
        namespace_scope = kwargs.get("namespace_scope") or self._build_namespace_scope(
            conv, speaker=speaker
        )

        headers = self._headers()
        ingest_url = (
            self._group_memories_url
            if self._search_scope(self.config) == "group"
            else self._personal_memories_url
        )

        # Preserve ordering: send sequentially.
        receipts = []
        for index, payload in enumerate(messages):
            chunk_id = self._build_import_manifest_chunk_id(
                conv,
                namespace_scope,
                index,
            )
            existing_row = self._incremental_import_manifest_row(chunk_id)
            if existing_row:
                expected_source_unit_ids = (
                    [str(payload.get("source_unit_id"))]
                    if payload.get("source_unit_id")
                    else []
                )
                existing_source_unit_ids = [
                    str(value) for value in existing_row.get("source_unit_ids", [])
                ]
                existing_namespace = (
                    existing_row.get("write_receipt", {}).get("namespace_scope", {})
                    or existing_row.get("namespace_scope", {})
                    or {}
                )
                if existing_source_unit_ids != expected_source_unit_ids:
                    raise RuntimeError(
                        "Cannot resume add: checkpoint source_unit_ids differ for "
                        f"{chunk_id}. Reuse the original dataset/output directory or "
                        "start a fresh add output directory."
                    )
                if (
                    existing_namespace.get("namespace_id")
                    != namespace_scope.get("namespace_id")
                ):
                    raise RuntimeError(
                        "Cannot resume add: checkpoint namespace differs for "
                        f"{chunk_id}. Reuse the original run-name or start a fresh "
                        "add output directory."
                    )
                receipts.append(existing_row.get("write_receipt", {}))
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)
                continue

            response = await self._request_json_with_retry(
                "post",
                ingest_url,
                json=self._build_ingest_payload(conv, payload),
                headers=headers,
            )
            request_id = (
                response.get("request_id")
                or (response.get("data") or {}).get("request_id")
                or (response.get("data") or {}).get("id")
            )
            scope = self._search_scope(self.config)
            session_id = self._message_session_id(payload)
            memory_ref = {
                "provider": "evermemos_api",
                "request_id": request_id,
                "group_id": (
                    namespace_scope["namespace_id"]
                    if scope == "group"
                    else ""
                ),
                "user_id": self._extract_user_id(conv, speaker=speaker)
                if scope != "group"
                else "",
                "session_id": session_id,
                "source_session_id": session_id,
                "source_case_id": payload.get("source_case_id") or "",
                "source_unit_id": payload.get("source_unit_id") or "",
                "message_id": payload.get("message_id"),
                "sender_id": payload.get("sender_id"),
                "content": payload.get("content", ""),
            }
            if session_id:
                memory_ref["run_id"] = session_id
            receipt = {
                "system_id": self._system_id(),
                "namespace_scope": namespace_scope,
                "chunk_id": chunk_id,
                "source_unit_ids": [payload.get("source_unit_id")]
                if payload.get("source_unit_id")
                else [],
                "session_id": session_id,
                "source_session_id": session_id,
                "provider_receipt": response,
                "provider_status": self._extract_provider_status(
                    response,
                    default="submitted" if not self.sync_mode else "processing",
                ),
                "memory_refs": [memory_ref],
                "errors": [],
            }
            await self._record_incremental_import_manifest(
                conversation=conv,
                messages=[payload],
                speaker=speaker,
                namespace_scope=namespace_scope,
                raw_result=receipt,
            )
            receipts.append(receipt)
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
            if self.request_interval > 0:
                await asyncio.sleep(self.request_interval)

        return receipts

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        if not self._search_url:
            raise ValueError("base_url is empty; set system config 'base_url'")

        headers = self._headers()
        data = await self._request_json_with_retry(
            "post",
            self._search_url,
            json=self._build_search_payload(
                query,
                conversation_id,
                user_id,
                top_k,
                question_metadata=kwargs.get("question_metadata"),
            ),
            headers=headers,
        )

        return self._parse_search_response(data, user_id, conversation_id)[: int(top_k)]

    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs: Any,
    ) -> SearchResult:
        search_cfg = self.config.get("search", {}) or {}
        scope = self._search_scope(self.config)
        system_name = str(self.config.get("name") or "evermemos_api")
        memory_types = self._normalize_memory_types(search_cfg.get("memory_types", []))
        group_id = getattr(self, "_namespace_cache", {}).get(
            (conversation_id, "speaker_a"), conversation_id
        )
        provider_filters = (
            {"group_id": group_id} if scope == "group" else {"user_id": user_id}
        )

        retrieval_metadata = {
            "system": system_name,
            "top_k": int(top_k),
            "retrieve_method": self._provider_retrieve_method(self.config),
            "memory_types": memory_types,
            "user_id": user_id if scope != "group" else "",
            "group_id": group_id if scope == "group" else "",
            "provider_filters": provider_filters,
        }
        if scope != "group":
            session_ids = self._extract_question_session_ids(
                kwargs.get("question_metadata")
            )
            if session_ids:
                retrieval_metadata["qa_session_ids"] = session_ids
                retrieval_metadata["session_ids"] = session_ids

        return SearchResult(
            question_id=kwargs.get("question_id", ""),
            query=query,
            conversation_id=conversation_id,
            results=results[: int(top_k)],
            retrieval_metadata=retrieval_metadata,
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
        **kwargs: Any,
    ) -> SearchResult:
        # Not used (we force single perspective), but keep minimal implementation to satisfy ABC.
        del (
            all_results,
            results_a,
            results_b,
            speaker_a,
            speaker_b,
            speaker_b_user_id,
            kwargs,
        )
        return self._build_single_search_result(
            query=query,
            conversation_id=conversation_id,
            results=[],
            user_id=speaker_a_user_id,
            top_k=top_k,
        )

    def _unsupported_readback_search_result(
        self,
        *,
        query: str,
        conversation_id: str,
        question_id: Optional[str],
        session_ids: Optional[List[str]],
        reason: str,
        error_type: str,
    ) -> SearchResult:
        readback = StorageReadbackResult(
            status="unsupported",
            checked_session_ids=list(session_ids or []),
            missing_session_ids=list(session_ids or []),
            objects=[],
            metadata={
                "provider": "evermemos_api",
                "question_id": question_id,
                "readback_scope": "question_sessions",
                "reason": reason,
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
                "system": str(self.config.get("name") or "evermemos_api"),
                "search_mode": "readback",
                "session_ids": list(session_ids or []),
                "formatted_context": "",
                "readback": readback.to_dict(),
            },
            retrieval_status="unsupported",
        )

    @staticmethod
    def _readback_result_status(readback: StorageReadbackResult, has_context: bool) -> str:
        if readback.status in {"unsupported", "error", "no_user_id"}:
            return readback.status
        if not has_context:
            return "empty"
        return "ok"

    @staticmethod
    def _build_readback_search_payload(
        readback: StorageReadbackResult,
        *,
        session_ids: List[str],
    ) -> tuple[List[Dict[str, Any]], str]:
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
            ordered_objects.extend(buckets.get(session_id, []))

        results: List[Dict[str, Any]] = []
        context_parts: List[str] = []
        for obj in ordered_objects:
            content = str(obj.content or "").strip()
            if not content:
                continue
            results.append(
                {
                    "content": content,
                    "score": 1.0,
                    "metadata": {
                        "memory_id": obj.id,
                        "session_id": obj.session_id,
                        "kind": obj.kind,
                        "provider_metadata": obj.metadata or {},
                    },
                }
            )
            context_parts.append(f"{len(context_parts) + 1}. {content}")

        return results, "\n\n".join(context_parts)

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
        """Build a first-class SearchResult from EverMemOS storage readback."""
        del index, kwargs

        session_ids = self._extract_question_session_ids(question_metadata or {})
        if self._search_scope(self.config) == "group":
            return self._unsupported_readback_search_result(
                query=query,
                conversation_id=conversation_id,
                question_id=question_id,
                session_ids=session_ids,
                reason=(
                    "EverMemOS readback search only supports personal scope; "
                    "group scope would require broad group injection."
                ),
                error_type="unsupported_scope",
            )

        if not session_ids:
            return self._unsupported_readback_search_result(
                query=query,
                conversation_id=conversation_id,
                question_id=question_id,
                session_ids=[],
                reason="Readback search requires question_metadata.session_ids.",
                error_type="missing_session_ids",
            )

        user_id = None
        if conversation is not None:
            user_id = self._extract_user_id(conversation, speaker="speaker_a")

        context = {
            "question_id": question_id,
            "conversation_id": conversation_id,
            "session_ids": session_ids,
            "import_manifest_rows": import_manifest_records or [],
            "qa_row": {"metadata": question_metadata or {}},
        }
        readback = await self.get_storage_readback(
            user_id=user_id,
            session_ids=session_ids,
            question_id=question_id,
            context=context,
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
                "system": str(self.config.get("name") or "evermemos_api"),
                "search_mode": "readback",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "user_ids": readback.metadata.get("user_ids", []),
                "readback": readback.to_dict(),
            },
            retrieval_status=self._readback_result_status(
                readback,
                has_context=bool(formatted_context),
            ),
        )

    async def probe_readiness(self, add_result: Any = None, **kwargs) -> Dict[str, Any]:
        """EverMemOS sync mode is immediately ready; async mode falls back."""
        if self.sync_mode:
            return {
                "supported": True,
                "ready": True,
                "status": "ready",
                "details": {"sync_mode": True},
            }
        return {
            "supported": False,
            "ready": False,
            "status": "unsupported",
            "details": {"sync_mode": False},
        }

    async def _probe_memory_ref_searchable(
        self,
        memory_ref: Dict[str, Any],
        namespace_scope: Dict[str, Any],
    ) -> tuple[bool, str]:
        """Treat searchability/readback visibility as the readiness signal."""
        filters = self._memory_ref_filters(memory_ref, namespace_scope)
        if not filters:
            return False, "contract_mismatch"

        probe_queries = self._build_probe_queries(memory_ref)
        if not probe_queries:
            return False, "contract_mismatch"

        headers = self._headers()
        for query in probe_queries:
            payload = await self._request_json_with_retry(
                "post",
                self._search_url,
                json={
                    "query": str(query),
                    "method": "keyword",
                    "memory_types": ["episodic_memory"],
                    "top_k": 3,
                    "filters": filters,
                },
                headers=headers,
            )
            results = self._parse_search_response(
                payload,
                user_id=str(filters.get("user_id", "")),
                conversation_id=str(filters.get("group_id", "")),
            )
            if results:
                return True, "completed"
        return False, "processing"

    def _collect_personal_flush_scopes(
        self, import_manifest_rows: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        scopes: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in import_manifest_rows:
            for memory_ref in row.get("memory_refs", []) or []:
                if not isinstance(memory_ref, dict):
                    continue
                user_id = self._clean_id(memory_ref.get("user_id"))
                session_id = self._clean_id(
                    memory_ref.get("session_id")
                    or memory_ref.get("source_session_id")
                    or memory_ref.get("run_id")
                )
                if not user_id or not session_id:
                    continue
                key = (user_id, session_id)
                if key in seen:
                    continue
                seen.add(key)
                scopes.append({"user_id": user_id, "session_id": session_id})
        return scopes

    async def _flush_personal_sessions(
        self, import_manifest_rows: List[Dict[str, Any]]
    ) -> tuple[Dict[str, int], List[Dict[str, Any]], List[str]]:
        flush_status_counts: Dict[str, int] = {}
        errors: List[Dict[str, Any]] = []
        warnings: List[str] = []

        scopes = self._collect_personal_flush_scopes(import_manifest_rows)
        if not scopes:
            return flush_status_counts, errors, [
                "EverMemOS personal finalize found no (user_id, session_id) scopes to flush."
            ]

        headers = self._headers()
        for scope in scopes:
            try:
                payload = await self._request_json_with_retry(
                    "post",
                    self._personal_flush_url,
                    json=scope,
                    headers=headers,
                )
                status = self._extract_provider_status(payload, default="completed")
                flush_status_counts[status] = flush_status_counts.get(status, 0) + 1
            except Exception as exc:  # noqa: BLE001
                flush_status_counts["error"] = flush_status_counts.get("error", 0) + 1
                error = {
                    "stage": "finalize_flush",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "user_id": scope["user_id"],
                    "session_id": scope["session_id"],
                }
                errors.append(error)
                warnings.append(
                    f"{scope['user_id']} / {scope['session_id']}: personal session flush failed ({type(exc).__name__})"
                )
        return flush_status_counts, errors, warnings

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
        """Finalize Cloud API writes with best-effort enrichment under manual confirmation."""
        del add_result, poll_interval_seconds, dataset, kwargs

        provider_status_counts: Dict[str, int] = {}
        missing_memory_refs: List[str] = []
        updated_rows = 0
        refreshed_rows: List[Dict[str, Any]] = []
        warnings: List[str] = [
            "EverMemOS Cloud finalize skips automatic readiness gating; continue only after manual provider confirmation that add has completed."
        ]
        flush_status_counts: Dict[str, int] = {}
        flush_errors: List[Dict[str, Any]] = []
        if self._search_scope(self.config) != "group":
            (
                flush_status_counts,
                flush_errors,
                flush_warnings,
            ) = await self._flush_personal_sessions(import_manifest_rows)
            warnings.extend(flush_warnings)
            if flush_status_counts and not flush_errors and budget_seconds > 0:
                await asyncio.sleep(float(budget_seconds))

        for row in import_manifest_rows:
            updated_row = dict(row)
            receipt = dict(updated_row.get("write_receipt", {}) or {})
            namespace_scope = dict(receipt.get("namespace_scope", {}) or {})
            memory_refs = [
                dict(item) for item in (updated_row.get("memory_refs") or [])
            ]

            status = self._normalize_provider_status(
                updated_row.get("write_status")
                or receipt.get("provider_status")
                or self._extract_provider_status(
                    receipt.get("provider_receipt", {}) or {}
                ),
                default="submitted",
            )
            row_errors = list(updated_row.get("errors", []) or [])
            if flush_errors:
                row_filter_user_ids = {
                    str(ref.get("user_id"))
                    for ref in memory_refs
                    if isinstance(ref, dict) and ref.get("user_id")
                }
                row_filter_session_ids = {
                    str(
                        ref.get("session_id")
                        or ref.get("source_session_id")
                        or ref.get("run_id")
                    )
                    for ref in memory_refs
                    if isinstance(ref, dict)
                    and (ref.get("session_id") or ref.get("source_session_id") or ref.get("run_id"))
                }
                for flush_error in flush_errors:
                    if (
                        flush_error.get("user_id") in row_filter_user_ids
                        and flush_error.get("session_id") in row_filter_session_ids
                    ):
                        row_errors.append(flush_error)

            if not memory_refs:
                missing_memory_refs.append(str(updated_row.get("chunk_id", "")))
                warnings.append(
                    f"{updated_row.get('chunk_id', '')}: missing memory_refs during finalize"
                )
            else:
                search_confirmed = False
                for memory_ref in memory_refs:
                    if (
                        self._search_scope(self.config) == "group"
                        and "group_id" not in memory_ref
                        and namespace_scope.get("namespace_id")
                    ):
                        memory_ref["group_id"] = namespace_scope["namespace_id"]
                    if (
                        "user_id" not in memory_ref
                        and self._search_scope(self.config) != "group"
                    ):
                        memory_ref["user_id"] = memory_ref.get("user_id") or ""

                    try:
                        (
                            search_confirmed,
                            probed_status,
                        ) = await self._probe_memory_ref_searchable(
                            memory_ref,
                            namespace_scope,
                        )
                        status = (
                            "completed"
                            if search_confirmed
                            else self._normalize_provider_status(
                                probed_status,
                                default=status,
                            )
                        )
                        if search_confirmed:
                            break
                    except Exception as exc:  # noqa: BLE001
                        row_errors.append(
                            {
                                "stage": "finalize",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
                        warnings.append(
                            f"{updated_row.get('chunk_id', '')}: searchability probe failed ({type(exc).__name__}); using manual confirmation flow"
                        )
                        break

                if not search_confirmed and status in {
                    "submitted",
                    "queued",
                    "pending",
                    "processing",
                }:
                    warnings.append(
                        f"{updated_row.get('chunk_id', '')}: provider/search visibility not yet confirmed; finalize is proceeding under manual confirmation"
                    )

            receipt["provider_status"] = status
            updated_row["write_receipt"] = receipt
            updated_row["memory_refs"] = memory_refs
            updated_row["write_status"] = status
            updated_row["errors"] = row_errors
            provider_status_counts[status] = provider_status_counts.get(status, 0) + 1
            updated_rows += 1
            refreshed_rows.append(updated_row)

        ready = not flush_errors

        return {
            "import_manifest_records": refreshed_rows,
            "ready": ready,
            "status": "finalized" if ready else "finalize_errors",
            "provider_status_counts": provider_status_counts,
            "flush_status_counts": flush_status_counts,
            "updated_rows": updated_rows,
            "missing_memory_refs": missing_memory_refs,
            "warnings": warnings,
            "finalize_budget_exhausted": False,
        }

    @classmethod
    def _normalize_episode_object(
        cls, episode: Dict[str, Any]
    ) -> NormalizedStorageObject:
        session_id = cls._clean_id(episode.get("session_id"))
        content_parts = [
            cls._clean_id(episode.get("timestamp")),
            cls._clean_id(
                episode.get("episode")
                or episode.get("summary")
                or episode.get("subject")
            ),
        ]
        content = ": ".join(part for part in content_parts if part)
        return NormalizedStorageObject(
            session_id=session_id,
            kind="episodic_memory",
            id=cls._clean_id(episode.get("id")),
            content=content,
            metadata={
                "provider": "evermemos_api",
                "user_id": episode.get("user_id"),
                "group_id": episode.get("group_id"),
                "memory_type": episode.get("type"),
                "parent_type": episode.get("parent_type"),
                "parent_id": episode.get("parent_id"),
            },
            raw=episode,
        )

    async def _get_episodic_page(
        self,
        *,
        filters: Dict[str, Any],
        page: int,
        page_size: int,
    ) -> Dict[str, Any]:
        return await self._request_json_with_retry(
            "post",
            self._get_url,
            json={
                "memory_type": "episodic_memory",
                "filters": filters,
                "page": page,
                "page_size": page_size,
                "rank_by": "timestamp",
                "rank_order": "desc",
            },
            headers=self._headers(),
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
        del run_id, evidence_texts, kwargs
        helper_context = dict(context or {})
        if getattr(self, "run_context", None):
            helper_context.setdefault("user_id", self.run_context.get("user_id"))
        readback_config = (self.config.get("readback", {}) or {})

        async def request_json(
            method: str, path: str, payload: Dict[str, Any]
        ) -> Dict[str, Any]:
            del path
            return await self._request_json_with_retry(
                method.lower(),
                self._get_url,
                json=payload,
                headers=self._headers(),
            )

        return await get_evermemos_storage_readback(
            request_json=request_json,
            provider="evermemos_api",
            scope=self._search_scope(self.config),
            user_id=user_id,
            session_ids=session_ids,
            question_id=question_id,
            context=helper_context,
            page_size=int(readback_config.get("page_size", 100)),
            max_pages=int(readback_config.get("max_pages", 20)),
        )
