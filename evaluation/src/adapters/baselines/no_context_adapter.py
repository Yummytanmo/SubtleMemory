"""
No-context baseline adapter.

This adapter intentionally retrieves no memories and exposes no conversation
content to answer generation. It reuses the default answer prompts with an
empty context so the answer model only sees the prompt template and query.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.adapters.baselines.oracle_context_adapter import (
    ORACLE_ANSWER_PROMPT,
)
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.shared.subtlememory_prompts import (
    get_subtlememory_unified_answer_prompt,
)
from evaluation.src.run_artifacts.models import (
    ImportManifestRecord,
    dataclass_to_dict,
)
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.utils.answer_cleaner import clean_answer_text

from memory_layer.llm.llm_provider import LLMProvider


@register_adapter("no_context")
class NoContextAdapter(BaseAdapter):
    """Baseline that answers from the question only, with no retrieved context."""

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self._import_manifest_records: List[Dict[str, Any]] = []

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

        print("NoContextAdapter initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")
        print(f"   Num Workers: {self.num_workers}")

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """No-op import stage; no conversation content is indexed or cached."""
        del kwargs
        self._import_manifest_records = self._build_import_manifest_records(
            conversations
        )
        return {
            "type": "no_context_baseline",
            "conversation_count": len(conversations),
        }

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any]:
        """No-op lazy index rebuild."""
        del output_dir
        self._import_manifest_records = self._build_import_manifest_records(
            conversations
        )
        return {
            "type": "no_context_baseline",
            "conversation_count": len(conversations),
        }

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """Return an empty retrieval result with an empty formatted context."""
        del index
        started_at = time.perf_counter()
        question_id = str(kwargs.get("question_id") or "")

        return SearchResult(
            question_id=question_id,
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={
                "formatted_context": "",
                "retrieval_mode": "no_context",
                "top_k": 0,
                "message_count": 0,
                "source": "no_context",
            },
            retrieval_status="ok",
            timing_ms=(time.perf_counter() - started_at) * 1000,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """Generate an answer while ignoring any provided context."""
        del context, kwargs
        prompt = self._get_answer_prompt().format(context="", question=query)
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
        """Render the answer prompt with an empty context for audit artifacts."""
        del context, kwargs
        return self._get_answer_prompt().format(context="", question=query)

    def _get_answer_prompt(self) -> str:
        dataset_id = str((self.run_context or {}).get("dataset_id", "")).strip()
        if dataset_id.startswith("subtlememory"):
            return get_subtlememory_unified_answer_prompt(config=self.config)
        return ORACLE_ANSWER_PROMPT

    def get_import_manifest_records(self) -> List[Dict[str, Any]]:
        """Return synthetic no-op manifest rows for finalize/report contracts."""
        return list(self._import_manifest_records)

    def get_system_info(self) -> Dict[str, Any]:
        """Return stable system info for run snapshots and CLI output."""
        return {"name": "no-context", "config": self.config}

    def _build_import_manifest_records(
        self, conversations: List[Conversation]
    ) -> List[Dict[str, Any]]:
        run_id = str(self.run_context.get("run_id", "run"))
        system_id = str(
            self.run_context.get("system_id")
            or self.config.get("name")
            or "no-context"
        )
        dataset_id = str(self.run_context.get("dataset_id", "dataset"))

        rows: List[Dict[str, Any]] = []
        for conversation in conversations:
            source_unit_ids = [
                self._source_unit_id_for_message(conversation.conversation_id, msg, idx)
                for idx, msg in enumerate(conversation.messages)
            ]
            chunk_id = f"{conversation.conversation_id}:no-context:none"
            memory_refs = [
                {
                    "provider": "no-context",
                    "conversation_id": conversation.conversation_id,
                    "storage_kind": "no_context_baseline",
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
                    "message_count": 0,
                    "artifact_type": "no_context_baseline",
                    "storage": "none",
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
                        "provider": "no-context",
                        "stored_content": False,
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

    @staticmethod
    def _source_unit_id_for_message(
        conversation_id: str, message: Any, idx: int
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
    def _clean_answer(answer: str) -> str:
        return clean_answer_text(answer)
