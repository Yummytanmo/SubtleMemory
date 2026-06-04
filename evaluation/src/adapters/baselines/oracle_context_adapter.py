"""
Oracle-context baseline adapter.

This adapter does not write to a memory system. It exposes oracle source
context to the normal answer/evaluate stages. For SubtleMemory, oracle context
is limited to the sessions attached to the current query.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.shared.subtlememory_prompts import (
    get_subtlememory_prompt_by_key,
    get_subtlememory_unified_answer_prompt,
)
from evaluation.src.run_artifacts.models import (
    ImportManifestRecord,
    dataclass_to_dict,
)
from evaluation.src.core.data_models import Conversation, Message, SearchResult
from evaluation.src.utils.answer_cleaner import clean_answer_text

from memory_layer.llm.llm_provider import LLMProvider


ORACLE_ANSWER_PROMPT = """You are a careful assistant answering questions from a complete conversation transcript.

# CONTEXT
The context below is the full available conversation history for the user pair or persona.

{context}

# INSTRUCTIONS
1. Answer using only the conversation transcript above.
2. Use timestamps, speaker names, and session order when they matter.
3. If the transcript contains conflicting information, resolve it according to the question wording; for time-anchored questions, prefer the relevant time period.
4. If the answer is not supported by the transcript, say you do not know.
5. Keep the answer concise and directly responsive.

Question: {question}

Answer:"""


@register_adapter("oracle_context")
class OracleContextAdapter(BaseAdapter):
    """Oracle source-context baseline for the evaluation framework."""

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self._conversations: Dict[str, Conversation] = {}
        self._import_manifest_records: List[Dict[str, Any]] = []
        self._answer_traces: Dict[str, Dict[str, Any]] = {}

        llm_config = config.get("llm", {})
        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 16384),
        )
        self.num_workers = int(config.get("num_workers", 20))

        print(f"OracleContextAdapter initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")
        print(f"   Num Workers: {self.num_workers}")

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """Cache source conversations and create a synthetic import manifest."""
        del kwargs
        self._cache_conversations(conversations)
        self._import_manifest_records = self._build_import_manifest_records(
            conversations
        )
        return {
            "type": "oracle_context_index",
            "conversation_ids": list(self._conversations.keys()),
            "conversation_count": len(self._conversations),
        }

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any]:
        """Rebuild the in-memory conversation map for resumed staged runs."""
        del output_dir
        self._cache_conversations(conversations)
        return {
            "type": "oracle_context_index",
            "conversation_ids": list(self._conversations.keys()),
            "conversation_count": len(self._conversations),
        }

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """Return oracle context as the formatted retrieval context."""
        del index
        started_at = time.perf_counter()
        conversation = kwargs.get("conversation") or self._conversations.get(
            conversation_id
        )
        question_id = str(kwargs.get("question_id") or "")
        question_metadata = kwargs.get("question_metadata") or {}

        if conversation is None:
            return SearchResult(
                question_id=question_id,
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "error": f"Conversation not found: {conversation_id}",
                    "retrieval_mode": "oracle_context",
                },
                retrieval_status="error",
                timing_ms=(time.perf_counter() - started_at) * 1000,
            )

        self._conversations[conversation.conversation_id] = conversation

        if self._is_subtlememory_run():
            scoped_conversation, scope_metadata, error = (
                self._scope_conversation_for_question(
                    conversation=conversation,
                    question_metadata=question_metadata,
                )
            )
            if error:
                return SearchResult(
                    question_id=question_id,
                    query=query,
                    conversation_id=conversation.conversation_id,
                    results=[],
                    retrieval_metadata={
                        **scope_metadata,
                        "error": error,
                        "retrieval_mode": "oracle_query_sessions",
                        "source": "query_scoped_sessions",
                    },
                    retrieval_status="error",
                    timing_ms=(time.perf_counter() - started_at) * 1000,
                )

            formatted_context = self._format_conversation(scoped_conversation)
            results = self._build_message_results(
                scoped_conversation, retrieval_mode="oracle_query_sessions"
            )
            retrieval_metadata = {
                **scope_metadata,
                "formatted_context": formatted_context,
                "retrieval_mode": "oracle_query_sessions",
                "top_k": len(results),
                "message_count": len(scoped_conversation.messages),
                "source": "query_scoped_sessions",
            }
        else:
            formatted_context = self._format_conversation(conversation)
            results = self._build_message_results(conversation)
            retrieval_metadata = {
                "formatted_context": formatted_context,
                "retrieval_mode": "oracle_context",
                "top_k": len(results),
                "message_count": len(conversation.messages),
                "source": "full_conversation",
            }

        return SearchResult(
            question_id=question_id,
            query=query,
            conversation_id=conversation.conversation_id,
            results=results,
            retrieval_metadata=retrieval_metadata,
            retrieval_status="ok",
            timing_ms=(time.perf_counter() - started_at) * 1000,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """Generate an answer using the same answer-stage LLM path as other adapters."""
        question_id = str(kwargs.get("question_id") or "")
        trace_key = self._answer_trace_key(question_id)
        self._answer_traces.pop(trace_key, None)

        if self._uses_two_step_fact_answer():
            return await self._answer_with_extracted_facts(
                query=query,
                context=context,
                question_id=question_id,
            )

        prompt = self._get_answer_prompt().format(context=context, question=query)
        max_retries = int(self.config.get("answer", {}).get("max_retries", 3))

        for attempt in range(max_retries):
            try:
                answer = await self.llm_provider.generate(prompt=prompt, temperature=0)
                answer = self._clean_answer(answer)
                if answer:
                    return answer
            except Exception:
                if attempt == max_retries - 1:
                    raise

        return ""

    def render_answer_prompt(
        self, query: str, context: str, **kwargs: Any
    ) -> Optional[str]:
        """Render the first answer prompt for audit artifacts when possible."""
        del kwargs
        if self._uses_two_step_fact_answer():
            return self._render_fact_extraction_prompt(query=query, context=context)
        return self._get_answer_prompt().format(context=context, question=query)

    def consume_answer_trace(
        self, question_id: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Return and clear two-step answer trace metadata."""
        del kwargs
        return self._answer_traces.pop(self._answer_trace_key(question_id), {})

    def _get_answer_prompt(self) -> str:
        dataset_id = str((self.run_context or {}).get("dataset_id", "")).strip()
        if dataset_id.startswith("subtlememory"):
            return get_subtlememory_unified_answer_prompt(config=self.config)
        return ORACLE_ANSWER_PROMPT

    async def _answer_with_extracted_facts(
        self, *, query: str, context: str, question_id: str
    ) -> str:
        answer_cfg = self.config.get("answer", {}) or {}
        max_retries = int(answer_cfg.get("max_retries", 3))
        max_facts = self._max_extracted_facts()
        trace_key = self._answer_trace_key(question_id)

        extraction_prompt = self._render_fact_extraction_prompt(
            query=query,
            context=context,
            max_facts=max_facts,
        )
        raw_facts = await self._generate_nonempty(
            prompt=extraction_prompt,
            max_retries=max_retries,
        )
        extracted_facts = self._parse_extracted_facts(
            raw_facts,
            max_facts=max_facts,
        )
        facts_text = self._format_extracted_facts(extracted_facts)
        fact_answer_prompt = self._render_fact_answer_prompt(
            query=query,
            facts_text=facts_text,
        )
        raw_answer = await self._generate_nonempty(
            prompt=fact_answer_prompt,
            max_retries=max_retries,
        )
        answer = self._clean_answer(raw_answer)

        self._answer_traces[trace_key] = {
            "answer_strategy": "two_step_fact_then_answer",
            "answer_prompt": fact_answer_prompt,
            "answer_fact_extraction_prompt": extraction_prompt,
            "answer_fact_answer_prompt": fact_answer_prompt,
            "answer_fact_extraction_raw": raw_facts,
            "answer_extracted_facts": extracted_facts,
            "answer_extracted_fact_count": len(extracted_facts),
            "answer_extracted_facts_context": facts_text,
        }
        return answer

    async def _generate_nonempty(self, *, prompt: str, max_retries: int) -> str:
        for attempt in range(max_retries):
            try:
                result = await self.llm_provider.generate(prompt=prompt, temperature=0)
                text = str(result or "").strip()
                if text:
                    return text
            except Exception:
                if attempt == max_retries - 1:
                    raise
        return ""

    def _render_fact_extraction_prompt(
        self, *, query: str, context: str, max_facts: Optional[int] = None
    ) -> str:
        prompt = get_subtlememory_prompt_by_key(self._fact_extraction_prompt_key())
        return prompt.format(
            context=context,
            question=query,
            max_facts=max_facts if max_facts is not None else self._max_extracted_facts(),
        )

    def _render_fact_answer_prompt(self, *, query: str, facts_text: str) -> str:
        prompt = get_subtlememory_prompt_by_key(self._fact_answer_prompt_key())
        return prompt.format(facts=facts_text, question=query)

    def _uses_two_step_fact_answer(self) -> bool:
        answer_cfg = self.config.get("answer", {}) or {}
        strategy = str(answer_cfg.get("strategy") or "").strip().lower()
        return strategy == "two_step_fact_then_answer"

    def _fact_extraction_prompt_key(self) -> str:
        answer_cfg = self.config.get("answer", {}) or {}
        return str(answer_cfg.get("fact_extraction_prompt_key") or "v5_fact_extractor")

    def _fact_answer_prompt_key(self) -> str:
        answer_cfg = self.config.get("answer", {}) or {}
        return str(answer_cfg.get("fact_answer_prompt_key") or "v5_fact_answer")

    def _max_extracted_facts(self) -> int:
        answer_cfg = self.config.get("answer", {}) or {}
        try:
            return max(1, int(answer_cfg.get("max_extracted_facts", 12)))
        except (TypeError, ValueError):
            return 12

    def _parse_extracted_facts(
        self, raw_facts: str, *, max_facts: int
    ) -> List[Dict[str, str]]:
        raw_text = str(raw_facts or "").strip()
        if not raw_text:
            return []

        try:
            data = json.loads(self._extract_json_payload(raw_text))
        except json.JSONDecodeError:
            return self._fallback_facts_from_text(raw_text, max_facts=max_facts)

        facts_value: Any
        if isinstance(data, dict):
            facts_value = data.get("facts", [])
        elif isinstance(data, list):
            facts_value = data
        else:
            return []

        if not isinstance(facts_value, list):
            return []

        facts: List[Dict[str, str]] = []
        for item in facts_value:
            fact: Dict[str, str] = {}
            if isinstance(item, str):
                fact_text = item.strip()
            elif isinstance(item, dict):
                fact_text = self._first_text_value(
                    item,
                    ("fact", "text", "content", "statement"),
                )
                source = self._first_text_value(
                    item,
                    ("source", "citation", "message", "session", "source_unit_id"),
                )
                role = self._first_text_value(item, ("role", "type", "use"))
                if source:
                    fact["source"] = source
                if role:
                    fact["role"] = role
            else:
                continue

            if not fact_text:
                continue
            fact["fact"] = fact_text
            facts.append(fact)
            if len(facts) >= max_facts:
                break

        return facts

    @staticmethod
    def _extract_json_payload(raw_text: str) -> str:
        code_block = re.search(
            r"```(?:json)?\s*(.*?)```",
            raw_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if code_block:
            return code_block.group(1).strip()

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end > start:
            return raw_text[start : end + 1]
        return raw_text

    @staticmethod
    def _first_text_value(item: Dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _fallback_facts_from_text(
        self, raw_text: str, *, max_facts: int
    ) -> List[Dict[str, str]]:
        facts: List[Dict[str, str]] = []
        for line in raw_text.splitlines():
            text = line.strip()
            if not text or text.startswith("```") or text in {"{", "}", "[", "]"}:
                continue
            text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
            if not text:
                continue
            facts.append({"fact": text, "role": "direct"})
            if len(facts) >= max_facts:
                break
        return facts

    @staticmethod
    def _format_extracted_facts(facts: List[Dict[str, str]]) -> str:
        if not facts:
            return "No extracted facts."

        lines: List[str] = []
        for idx, fact in enumerate(facts, 1):
            fact_text = str(fact.get("fact", "")).strip()
            if not fact_text:
                continue
            metadata_parts = []
            source = str(fact.get("source", "")).strip()
            role = str(fact.get("role", "")).strip()
            if source:
                metadata_parts.append(f"source: {source}")
            if role:
                metadata_parts.append(f"role: {role}")
            metadata = f" ({'; '.join(metadata_parts)})" if metadata_parts else ""
            lines.append(f"{idx}. {fact_text}{metadata}")
        return "\n".join(lines) if lines else "No extracted facts."

    @staticmethod
    def _answer_trace_key(question_id: str) -> str:
        return str(question_id or "").strip() or "__last__"

    def get_import_manifest_records(self) -> List[Dict[str, Any]]:
        """Return synthetic manifest rows for finalize gating."""
        return list(self._import_manifest_records)

    def get_system_info(self) -> Dict[str, Any]:
        """Return stable system info for run snapshots and CLI output."""
        return {"name": "oracle-context", "config": self.config}

    def _cache_conversations(self, conversations: List[Conversation]) -> None:
        self._conversations = {
            conversation.conversation_id: conversation for conversation in conversations
        }

    def _build_import_manifest_records(
        self, conversations: List[Conversation]
    ) -> List[Dict[str, Any]]:
        run_id = str(self.run_context.get("run_id", "run"))
        system_id = str(
            self.run_context.get("system_id")
            or self.config.get("name")
            or "oracle-context"
        )
        dataset_id = str(self.run_context.get("dataset_id", "dataset"))

        rows: List[Dict[str, Any]] = []
        for conversation in conversations:
            source_unit_ids = [
                self._source_unit_id_for_message(conversation.conversation_id, msg, idx)
                for idx, msg in enumerate(conversation.messages)
            ]
            chunk_id = f"{conversation.conversation_id}:oracle-context:full"
            memory_refs = [
                {
                    "provider": "oracle-context",
                    "conversation_id": conversation.conversation_id,
                    "storage_kind": "full_conversation_transcript",
                }
            ]
            manifest_row = ImportManifestRecord(
                run_id=run_id,
                system_id=system_id,
                conversation_id=conversation.conversation_id,
                view_id="shared",
                chunk_id=chunk_id,
                source_unit_ids=source_unit_ids,
                write_request_summary={
                    "message_count": len(conversation.messages),
                    "artifact_type": "oracle_context_full_conversation",
                    "storage": "in_memory_source_conversation",
                },
                write_receipt={
                    "system_id": system_id,
                    "namespace_scope": {
                        "namespace_id": conversation.conversation_id,
                        "view_id": "shared",
                        "dataset_id": dataset_id,
                    },
                    "chunk_id": chunk_id,
                    "source_unit_ids": source_unit_ids,
                    "provider_receipt": {
                        "provider": "oracle-context",
                        "message_count": len(conversation.messages),
                    },
                    "provider_status": "completed",
                    "memory_refs": memory_refs,
                    "errors": [],
                },
                memory_refs=memory_refs,
                write_status="completed",
                errors=[],
            )
            rows.append(dataclass_to_dict(manifest_row))
        return rows

    def _build_message_results(
        self, conversation: Conversation, retrieval_mode: str = "oracle_context"
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx, message in enumerate(conversation.messages):
            metadata = dict(message.metadata or {})
            metadata.update(
                {
                    "speaker": message.sender_name,
                    "sender_id": message.sender_id,
                    "message_index": idx,
                    "source_unit_id": self._source_unit_id_for_message(
                        conversation.conversation_id, message, idx
                    ),
                    "timestamp": self._format_timestamp(message),
                    "retrieval_mode": retrieval_mode,
                }
            )
            results.append(
                {"content": message.content, "score": 1.0, "metadata": metadata}
            )
        return results

    def _format_conversation(self, conversation: Conversation) -> str:
        speaker_a = (
            conversation.metadata.get("speaker_a") if conversation.metadata else None
        )
        speaker_b = (
            conversation.metadata.get("speaker_b") if conversation.metadata else None
        )
        header = [
            f"Conversation ID: {conversation.conversation_id}",
            f"Message count: {len(conversation.messages)}",
        ]
        scope_session_ids = (conversation.metadata or {}).get("scope_session_ids")
        if scope_session_ids:
            header.append(
                "Scoped session IDs: "
                + ", ".join(str(item) for item in scope_session_ids)
            )
        if speaker_a or speaker_b:
            header.append(
                "Speakers: "
                + ", ".join(
                    str(speaker) for speaker in (speaker_a, speaker_b) if speaker
                )
            )

        lines = ["# ORACLE CONTEXT TRANSCRIPT", *header, ""]
        for idx, message in enumerate(conversation.messages, 1):
            timestamp = self._format_timestamp(message) or "unknown time"
            metadata = message.metadata or {}
            session = metadata.get("session")
            source_session_id = metadata.get("source_session_id")
            source_case_id = metadata.get("source_case_id")
            dia_id = metadata.get("dia_id")
            source_unit_id = self._source_unit_id_for_message(
                conversation.conversation_id, message, idx - 1
            )
            metadata_parts = [
                f"timestamp={timestamp}",
                f"source_unit_id={source_unit_id}",
            ]
            if session:
                metadata_parts.append(f"session={session}")
            if source_session_id:
                metadata_parts.append(f"source_session_id={source_session_id}")
            if source_case_id:
                metadata_parts.append(f"source_case_id={source_case_id}")
            if dia_id:
                metadata_parts.append(f"dia_id={dia_id}")
            lines.append(f"[{idx}] {message.sender_name} ({'; '.join(metadata_parts)})")
            lines.append(message.content)
            lines.append("")

        return "\n".join(lines).strip()

    def _is_subtlememory_run(self) -> bool:
        dataset_id = str((self.run_context or {}).get("dataset_id", "")).strip()
        return dataset_id.startswith("subtlememory")

    def _scope_conversation_for_question(
        self, *, conversation: Conversation, question_metadata: Dict[str, Any]
    ) -> tuple[Conversation, Dict[str, Any], Optional[str]]:
        requested_session_ids = self._normalize_session_ids(
            question_metadata.get("session_ids")
        )
        scope_metadata = {"scope_session_ids": requested_session_ids}

        if not requested_session_ids:
            return (
                conversation,
                scope_metadata,
                "SubtleMemory oracle_context requires question_metadata.session_ids.",
            )

        requested_set = set(requested_session_ids)
        scoped_messages = [
            message
            for message in conversation.messages
            if str((message.metadata or {}).get("source_session_id") or "").strip()
            in requested_set
        ]
        scope_metadata.update(
            {
                "scope_message_count": len(scoped_messages),
                "conversation_message_count": len(conversation.messages),
            }
        )

        if not scoped_messages:
            return (
                conversation,
                scope_metadata,
                "No messages matched question_metadata.session_ids. "
                "Regenerate the SubtleMemory LoCoMo-style data if "
                "source_session_id is missing.",
            )

        scoped_metadata = dict(conversation.metadata or {})
        scoped_metadata["scope_session_ids"] = requested_session_ids
        return (
            Conversation(
                conversation_id=conversation.conversation_id,
                messages=scoped_messages,
                metadata=scoped_metadata,
            ),
            scope_metadata,
            None,
        )

    @staticmethod
    def _normalize_session_ids(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = [value]
        return [str(item).strip() for item in raw_items if str(item).strip()]

    @staticmethod
    def _source_unit_id_for_message(
        conversation_id: str, message: Message, idx: int
    ) -> str:
        metadata = message.metadata or {}
        for key in ("source_unit_id", "dia_id", "message_id"):
            value = metadata.get(key)
            if value:
                return str(value)
        session = metadata.get("session")
        if session:
            return f"{conversation_id}:{session}:{idx}"
        return f"{conversation_id}:msg:{idx}"

    @staticmethod
    def _format_timestamp(message: Message) -> Optional[str]:
        if message.timestamp is None:
            return None
        return message.timestamp.isoformat()

    @staticmethod
    def _clean_answer(answer: str) -> str:
        return clean_answer_text(answer)
