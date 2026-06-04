"""OpenClaw session-memory adapter.

This adapter treats OpenClaw's session-memory markdown files as the
memory-system storage layer while keeping the SubtleMemory evaluation pipeline intact.
The add/finalize stages create OpenClaw transcripts, call OpenClaw's bundled
session-memory handler to write markdown, and build a memory-core config/index.
The search stage is deliberately a no-op:
question answering calls the real `openclaw agent --local` command directly and
lets OpenClaw decide whether to call memory_search.
"""

from __future__ import annotations

import hashlib
import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.adapters.baselines.oracle_context_adapter import ORACLE_ANSWER_PROMPT
from evaluation.src.adapters.core.registry import register_adapter
from evaluation.src.adapters.shared.subtlememory_prompts import (
    get_subtlememory_unified_answer_prompt,
)
from evaluation.src.run_artifacts.models import (
    ImportManifestRecord,
    dataclass_to_dict,
)
from evaluation.src.core.data_models import Conversation, Message, SearchResult
from evaluation.src.core.readback import NormalizedStorageObject, StorageReadbackResult
from memory_layer.llm.llm_provider import LLMProvider


CURRENT_SESSION_VERSION = 3
OPENCLAW_DEFAULT_QA_PROMPT_KEY = "answer_prompt_empty"
OPENCLAW_EMPTY_QA_PROMPT_KEYS = {
    "",
    "none",
    "null",
    "off",
    "disabled",
    "empty",
    "answer_prompt_empty",
}
OPENCLAW_BUILT_PLUGIN_LOAD_IDS = ("openai", "memory-core")
OPENCLAW_EVALUATION_AGENTS_MD = """# OpenClaw SubtleMemory Evaluation Workspace

- This is a SubtleMemory memory-system evaluation workspace.
- Answer the current benchmark question using only the current question and OpenClaw recall-injected memory context.
- Recalled memory/context may be auto-recalled or available through the configured OpenClaw memory backend or plugin.
- If memory_search is available and no recalled memory/context has already been injected, call memory_search with the current question before answering.
- This is a benchmark answer turn: produce a visible non-empty answer and never output NO_REPLY.
- Do not call session_status or other session-management/status tools for benchmark answers.
- Do not use outside knowledge, browser/web search, or unrelated workspace files.
- Do not modify this workspace or any evaluation files.
- Keep the final answer concise and grounded in the recalled memory context.

## SubtleMemory Answer Rules

1. Use only the recalled memory/context injected before the question. Do not use outside knowledge, common knowledge, or your own guess to resolve conflicts.
2. Your first priority is evidence fidelity: detect true unresolved conflicts, but do not over-detect conflicts from compatible evidence.
3. Identify the exact target needed by the question: preference, choice, attribute, state, factual answer, origin, date, name, or category.
4. A conflict requires mutually exclusive claims about the same target. Different interests, different sources, background facts, or multiple constraints are not conflicts unless they directly support incompatible answers to the question target.
5. Search the recalled memory/context for evidence that supports one answer and evidence that supports a different or opposing answer for the same target. Treat semantic opposites as conflict even when wording differs.
6. Treat these as unresolved conflicts unless the recalled memory/context explicitly resolves them with a time frame, context, correction, current-state update, condition, or exception:
   - one memory says the user likes/prefers/enjoys/appreciates something while another says the user dislikes/avoids/does not enjoy/prefers a different category
   - one memory supports choosing an option while another memory points away from that option or toward an incompatible option
   - one factual memory gives answer A while another factual memory gives answer B for the same question target
   - a user statement conflicts with an assistant summary or another session summary
7. Do not assume the newest statement overrides older evidence unless the recalled memory/context explicitly says it is an update, correction, or current state.
8. Do not invent a compromise, exception, or hierarchy to make conflicting evidence fit together. For example, do not infer "usually dislikes X but likes high-quality X" unless the transcript explicitly states that exact exception.
9. If unresolved conflict affects a recommendation, reservation, purchase, registration, list, ranking, yes/no answer, or any other decisive answer, do not choose a side and do not choose a safer alternative. Start with "Unclear — needs clarification first." Then briefly state both conflicting sides.
10. Apply this conflict gate before writing the final answer: if two recalled facts give different values for the same target and the question does not explicitly select one by time or context, the final answer must be an unresolved factual conflict. Do not answer with only one side of a factual conflict, even if one value appears later, more familiar, or more plausible.
11. If the question is explicitly time-anchored or context-anchored and that time/context clearly selects one side, answer that side directly.
12. For "latest before", "before [date/event]", "prior to [date/event]", or equivalent questions, choose the latest recalled answer before that excluded later anchor. Ignore records at or after the excluded later anchor; do not treat the excluded later record as a conflict.
13. Treat evidence as compatible, and answer directly, in these cases:
   - multiple memories can all be true at the same time
   - one memory is background while another directly answers the question
   - one memory adds a constraint that can be satisfied rather than rejected
   - different sources or phrasings support the same final answer
   - the question requires combining multiple facts from separate memories
   - several options or supporting paths are valid and the question only needs one or a fixed number of them
14. For multi-part questions, combine compatible facts across the recalled memory/context even if no single sentence contains the whole answer.
15. For choice questions, follow the option that best matches the question target; do not call unrelated background interests a conflict.
16. Keep the answer concise: one or two sentences. Do not show step-by-step reasoning.

Output patterns:
- Unresolved preference/choice conflict: "Unclear — needs clarification first. The context says both [side A] and [side B], so I can't safely [make the requested choice]."
- Unresolved factual conflict: "Unclear — needs clarification first. The context gives conflicting remembered answers: [A] and [B]."
- Resolved by explicit time/context: answer directly and mention the resolving time/context briefly.
"""
OPENCLAW_SESSION_MEMORY_API_FORCE_SEARCH_INSTRUCTION = (
    "- For API-mode benchmark answers, you MUST call the memory_search tool "
    "with the current question before giving the final answer; mentioning "
    "memory_search in text is not enough."
)


@dataclass
class ImportedMemory:
    conversation_id: str
    session_key: str
    source_session_id: str
    source_case_id: str
    session_order: Any
    session_source: str
    session_timestamp: str
    transcript_path: str
    memory_path: str
    content: str
    source_unit_ids: List[str]
    message_count: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class OpenClawRuntime:
    conversation_id: str
    run_root: Path
    state_dir: Path
    workspace_dir: Path
    config_path: Path
    sessions_dir: Path
    agent_dir: Path
    bridge_payload_dir: Path
    bridge_result_dir: Path
    bridge_workspace_dir: Path
    memory_dir: Path


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _safe_component(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text or fallback


def _utc_iso(value: Optional[datetime]) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat()
    return value.astimezone(timezone.utc).isoformat()


def _timestamp_ms(value: str) -> int:
    try:
        normalized = value.replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


def _short_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _absolute_path(path: Any) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _required_path_config(*values: Any, env_name: str, purpose: str) -> Any:
    for value in values:
        if str(value or "").strip():
            return value
    raise ValueError(
        f"{purpose} is not configured. Set {env_name} in .env or the shell."
    )


@register_adapter("openclaw_session_memory")
class OpenClawSessionMemoryAdapter(BaseAdapter):
    """OpenClaw session-memory markdown backend for the evaluation pipeline."""

    adapter_id = "openclaw_session_memory"

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = _absolute_path(output_dir or ".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        openclaw_cfg = config.get("openclaw", {})
        self.openclaw_root = _absolute_path(
            _required_path_config(
                openclaw_cfg.get("root"),
                os.environ.get("OPENCLAW_ROOT"),
                env_name="OPENCLAW_ROOT",
                purpose="OpenClaw root",
            )
        )
        self.openclaw_cli = str(
            _absolute_path(openclaw_cfg["cli"])
            if openclaw_cfg.get("cli")
            else self.openclaw_root / "openclaw.mjs"
        )
        self.session_memory_handler_path = _absolute_path(
            openclaw_cfg.get("session_memory_handler_path")
            or self.openclaw_root / "dist" / "bundled" / "session-memory" / "handler.js"
        )
        self.session_memory_bridge_path = _absolute_path(
            openclaw_cfg.get("session_memory_bridge_path")
            or Path(__file__).parent / "assets" / "session_memory_bridge.mjs"
        )
        self.base_config_path = _absolute_path(
            _required_path_config(
                openclaw_cfg.get("base_config_path"),
                os.environ.get("OPENCLAW_CONFIG_PATH"),
                env_name="OPENCLAW_CONFIG_PATH",
                purpose="OpenClaw base config path",
            )
        )
        self.run_root = _absolute_path(
            openclaw_cfg.get("run_root") or self.output_dir / "openclaw_runtime"
        )
        self.conversation_runtime_root = self.run_root / "conversations"
        self._runtime_by_conversation: Dict[str, OpenClawRuntime] = {}
        self.default_runtime = self._runtime_for("default")
        self.state_dir = self.default_runtime.state_dir
        self.workspace_dir = self.default_runtime.workspace_dir
        self.config_path = self.default_runtime.config_path
        self.sessions_dir = self.default_runtime.sessions_dir
        self.agent_dir = self.default_runtime.agent_dir
        self.bridge_payload_dir = self.default_runtime.bridge_payload_dir
        self.bridge_result_dir = self.default_runtime.bridge_result_dir
        self.bridge_workspace_dir = self.default_runtime.bridge_workspace_dir
        self.memory_dir = self.default_runtime.memory_dir
        self.artifact_transcript_dir = (
            self.output_dir / "openclaw_artifacts" / "transcripts"
        )
        self.artifact_memory_dir = self.output_dir / "openclaw_artifacts" / "memory"

        memory_cfg = config.get("memory", {})
        self.session_memory_messages = int(memory_cfg.get("messages", 50))
        self.search_top_k = int(
            config.get("search", {}).get("top_k", memory_cfg.get("search_top_k", 20))
        )
        self.min_score = float(memory_cfg.get("min_score", 0.0))
        answer_cfg = config.get("answer", {})
        self.use_openclaw_agent_answer = bool(
            answer_cfg.get("use_openclaw_agent", True)
        )
        self.allow_llm_fallback = bool(answer_cfg.get("allow_llm_fallback", False))

        llm_config = config.get("llm", {})
        self.llm_provider = None
        if (
            self.allow_llm_fallback
            or not self.use_openclaw_agent_answer
            or self._is_readback_search_mode()
        ):
            self.llm_provider = LLMProvider(
                provider_type=llm_config.get("provider", "openai"),
                model=llm_config.get("model", "gpt-4o-mini"),
                api_key=llm_config.get("api_key", ""),
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
                temperature=llm_config.get("temperature", 0.0),
                max_tokens=llm_config.get("max_tokens", 16384),
            )

        self.num_workers = int(config.get("num_workers", 1))
        self._conversations: Dict[str, Conversation] = {}
        self._import_manifest_records: List[Dict[str, Any]] = []
        self._memories_by_conversation: Dict[str, List[ImportedMemory]] = {}
        self._memories: List[ImportedMemory] = []
        self._runtime_recall_payloads: Dict[str, Dict[str, Any]] = {}
        self._readback_answer_cache: Dict[str, Dict[str, Any]] = {}

        print("OpenClawSessionMemoryAdapter initialized")
        print(f"   OpenClaw CLI: node {self.openclaw_cli}")
        print(f"   OpenClaw root: {self.openclaw_root}")
        print(f"   Output Dir: {self.output_dir}")
        print(f"   Runtime Dir: {self.run_root}")
        print(f"   Num Workers: {self.num_workers}")

    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        del kwargs
        self._cache_conversations(conversations)
        for conversation in conversations:
            self._prepare_runtime_dirs(self._runtime_for(conversation.conversation_id))

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        del kwargs
        self._cache_conversations(conversations)

        imported: List[ImportedMemory] = []
        import_rows: List[Dict[str, Any]] = []
        for conversation in conversations:
            runtime = self._runtime_for(conversation.conversation_id)
            self._prepare_runtime_dirs(runtime)
            for session_key, messages in self._group_messages_by_session(conversation):
                if not messages:
                    continue
                memory = self._write_session_memory(
                    conversation, session_key, messages, runtime
                )
                imported.append(memory)
                import_rows.append(self._build_manifest_record(memory))

        self._memories = imported
        self._memories_by_conversation = {}
        for memory in imported:
            self._memories_by_conversation.setdefault(
                memory.conversation_id, []
            ).append(memory)
        self._import_manifest_records = import_rows
        self._write_json(
            self.output_dir / "openclaw_import_summary.json",
            {
                "conversation_count": len(conversations),
                "memory_count": len(imported),
                "isolation": "conversation",
                "conversation_runtimes": self._conversation_runtime_summary(
                    conversation.conversation_id for conversation in conversations
                ),
                "session_memory_handler_path": str(self.session_memory_handler_path),
                "session_memory_bridge_path": str(self.session_memory_bridge_path),
                "session_memory_messages": self.session_memory_messages,
            },
        )
        return self._build_index_metadata()

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any]:
        del output_dir
        self._cache_conversations(conversations)
        self._load_imported_memories()
        if (
            not self._import_manifest_records
            and (self.output_dir / "import_manifest.jsonl").exists()
        ):
            self._import_manifest_records = self._read_jsonl(
                self.output_dir / "import_manifest.jsonl"
            )
        return self._build_index_metadata()

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
        del add_result, budget_seconds, poll_interval_seconds, dataset, kwargs
        if self._is_readback_search_mode():
            return self._finalize_readback_imports(import_manifest_rows)

        openclaw_cfg = self.config.get("openclaw", {}) or {}
        finalize_cfg = self.config.get("finalize", {}) or {}
        index_timeout = int(
            finalize_cfg.get(
                "openclaw_index_timeout_seconds",
                openclaw_cfg.get("index_timeout_seconds", 900),
            )
        )
        status_timeout = int(
            finalize_cfg.get(
                "openclaw_status_timeout_seconds",
                openclaw_cfg.get("status_timeout_seconds", 180),
            )
        )
        rows_by_conversation: Dict[str, List[Dict[str, Any]]] = {}
        for row in import_manifest_rows:
            rows_by_conversation.setdefault(str(row.get("conversation_id") or ""), []).append(row)

        index_results: Dict[str, Dict[str, Any]] = {}
        status_payloads: Dict[str, Any] = {}
        status_summaries: Dict[str, Dict[str, Any]] = {}
        total_status_summary = {"files": 0, "chunks": 0, "dirty": False, "db_paths": []}

        for conversation_id in sorted(rows_by_conversation):
            runtime = self._runtime_for(conversation_id)
            self._ensure_config(runtime)
            index_result = self._run_openclaw(
                ["memory", "index", "--force"],
                timeout=index_timeout,
                runtime=runtime,
            )
            status_result = self._run_openclaw(
                ["memory", "status", "--json"],
                timeout=status_timeout,
                runtime=runtime,
            )
            status_payload = self._parse_json_output(status_result)
            status_summary = self._summarize_openclaw_status(status_payload)
            index_results[conversation_id] = index_result
            status_payloads[conversation_id] = status_payload
            status_summaries[conversation_id] = status_summary
            total_status_summary["files"] += int(status_summary["files"])
            total_status_summary["chunks"] += int(status_summary["chunks"])
            total_status_summary["dirty"] = bool(
                total_status_summary["dirty"] or status_summary["dirty"]
            )
            total_status_summary["db_paths"].extend(status_summary["db_paths"])

        status_payload = {
            "isolation": "conversation",
            "conversations": status_payloads,
            "summaries": status_summaries,
            "summary": total_status_summary,
        }
        status_summary = total_status_summary
        self._write_json(self.output_dir / "memory_status.json", status_payload)

        ready = (
            bool(import_manifest_rows)
            and all(result["returncode"] == 0 for result in index_results.values())
            and status_summary["files"] >= len(import_manifest_rows)
            and status_summary["chunks"] > 0
        )
        warnings = []
        if status_summary["files"] < len(import_manifest_rows):
            warnings.append(
                "OpenClaw memory status reports fewer indexed files than imported "
                "session-memory markdown files."
            )
        if status_summary["chunks"] <= 0:
            warnings.append("OpenClaw memory status reports no indexed chunks.")
        if status_summary["dirty"]:
            warnings.append(
                "OpenClaw memory status reports a dirty index; continuing because "
                "the forced index command and status command both succeeded."
            )
        updated_rows = []
        for row in import_manifest_rows:
            next_row = dict(row)
            next_row["write_status"] = "ready" if ready else "index_error"
            updated_rows.append(next_row)
        self._import_manifest_records = updated_rows

        provider_status_counts: Dict[str, int] = {}
        for row in updated_rows:
            status = str(row.get("write_status", "unknown"))
            provider_status_counts[status] = provider_status_counts.get(status, 0) + 1

        return {
            "import_manifest_records": updated_rows,
            "ready": ready,
            "status": "indexed" if ready else "index_error",
            "provider_status_counts": provider_status_counts,
            "updated_rows": len(updated_rows),
            "missing_memory_refs": [
                str(row.get("chunk_id", ""))
                for row in updated_rows
                if not row.get("memory_refs")
            ],
            "warnings": warnings,
            "finalize_budget_exhausted": False,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "openclaw_index": index_results,
            "openclaw_status": status_payload,
            "openclaw_status_summary": status_summary,
            "isolation": "conversation",
            "conversation_runtimes": self._conversation_runtime_summary(
                rows_by_conversation
            ),
        }

    def _finalize_readback_imports(
        self, import_manifest_rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        rows_by_conversation: Dict[str, List[Dict[str, Any]]] = {}
        for row in import_manifest_rows:
            rows_by_conversation.setdefault(str(row.get("conversation_id") or ""), []).append(row)

        updated_rows = []
        missing_memory_refs = []
        existing_memory_files = 0
        for row in import_manifest_rows:
            next_row = dict(row)
            memory_refs = list(next_row.get("memory_refs") or [])
            has_existing_memory = any(
                self._memory_ref_path_exists(ref) for ref in memory_refs
            )
            if has_existing_memory:
                existing_memory_files += 1
                next_row["write_status"] = "ready"
            else:
                next_row["write_status"] = "missing_memory_ref"
                missing_memory_refs.append(str(next_row.get("chunk_id") or ""))
            updated_rows.append(next_row)

        self._import_manifest_records = updated_rows
        if not self._memories and import_manifest_rows:
            self._load_imported_memories()

        provider_status_counts: Dict[str, int] = {}
        for row in updated_rows:
            status = str(row.get("write_status", "unknown"))
            provider_status_counts[status] = provider_status_counts.get(status, 0) + 1

        ready = bool(import_manifest_rows) and not missing_memory_refs
        status_summary = {
            "files": existing_memory_files,
            "chunks": existing_memory_files,
            "dirty": False,
            "db_paths": [],
            "mode": "readback_local_session_memory",
        }
        status_payload = {
            "isolation": "conversation",
            "mode": "readback_local_session_memory",
            "summary": status_summary,
            "missing_memory_refs": missing_memory_refs,
        }
        self._write_json(self.output_dir / "memory_status.json", status_payload)

        warnings = []
        if missing_memory_refs:
            warnings.append(
                "OpenClaw readback finalize could not find all session-memory markdown files."
            )

        return {
            "import_manifest_records": updated_rows,
            "ready": ready,
            "status": "readback_ready" if ready else "missing_memory_refs",
            "provider_status_counts": provider_status_counts,
            "updated_rows": len(updated_rows),
            "missing_memory_refs": missing_memory_refs,
            "warnings": warnings,
            "finalize_budget_exhausted": False,
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "openclaw_index": {
                "skipped": True,
                "reason": "readback_mode_uses_local_session_memory_facade",
            },
            "openclaw_status": status_payload,
            "openclaw_status_summary": status_summary,
            "isolation": "conversation",
            "conversation_runtimes": self._conversation_runtime_summary(
                rows_by_conversation
            ),
        }

    @staticmethod
    def _memory_ref_path_exists(ref: Any) -> bool:
        if not isinstance(ref, dict):
            return False
        raw_path = ref.get("absolute_path") or ref.get("path")
        if not raw_path:
            return False
        path = Path(str(raw_path)).expanduser()
        try:
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        del index
        started_at = time.perf_counter()
        question_id = str(kwargs.get("question_id") or "")
        del kwargs
        retrieval_metadata = {
            "formatted_context": "",
            "retrieval_mode": "skipped_direct_openclaw_agent",
            "source": "openclaw_agent_answer_stage",
            "note": (
                "Search is intentionally skipped. The answer stage invokes "
                "the real OpenClaw agent command, which may call memory_search "
                "internally."
            ),
            "memory_status_path": str(self.output_dir / "memory_status.json"),
            "openclaw_isolation": "conversation",
            "openclaw_runtime": {
                "state_dir": str(self._runtime_for(conversation_id).state_dir),
                "workspace_dir": str(self._runtime_for(conversation_id).workspace_dir),
                "config_path": str(self._runtime_for(conversation_id).config_path),
                "memory_dir": str(self._runtime_for(conversation_id).memory_dir),
            },
        }
        return SearchResult(
            question_id=question_id,
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata=retrieval_metadata,
            retrieval_status="skipped",
            timing_ms=(time.perf_counter() - started_at) * 1000,
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
        """Build prompt-ready context from written OpenClaw memory markdown."""
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
            )

        records = self._readback_import_manifest_records(import_manifest_records)
        objects, errors = self._openclaw_readback_objects_from_records(
            conversation_id=conversation_id,
            session_ids=session_ids,
            import_manifest_records=records,
        )
        if not objects:
            objects, errors = self._openclaw_readback_objects_from_memory_cache(
                conversation_id=conversation_id,
                session_ids=session_ids,
                errors=errors,
            )

        objects_by_session: Dict[str, List[NormalizedStorageObject]] = {
            session_id: [] for session_id in session_ids
        }
        for obj in objects:
            if obj.session_id in objects_by_session:
                objects_by_session[obj.session_id].append(obj)

        ordered_objects: List[NormalizedStorageObject] = []
        for session_id in session_ids:
            ordered_objects.extend(objects_by_session[session_id])

        missing_session_ids = [
            session_id for session_id in session_ids if not objects_by_session[session_id]
        ]
        readback_status = "ok" if ordered_objects or not errors else "error"
        readback = StorageReadbackResult(
            status=readback_status,
            checked_session_ids=list(session_ids),
            missing_session_ids=missing_session_ids,
            objects=ordered_objects,
            metadata={
                "provider": "openclaw_session_memory",
                "question_id": question_id,
                "conversation_id": conversation_id,
                "readback_scope": "question_sessions",
                "source": "session_memory_markdown",
            },
            errors=errors,
        )
        results = [
            {
                "content": obj.content,
                "score": 1.0,
                "metadata": dict(obj.metadata),
            }
            for obj in ordered_objects
            if obj.content
        ]
        formatted_context = "\n\n".join(
            f"{index}. {item['content']}"
            for index, item in enumerate(results, start=1)
            if item.get("content")
        )
        if formatted_context:
            formatted_context = (
                "<recalled-memories>\n"
                f"{formatted_context}\n"
                "</recalled-memories>"
            )
        retrieval_status = "ok" if formatted_context else (
            "error" if readback_status == "error" else "empty"
        )
        if question_id:
            self._readback_answer_cache[str(question_id)] = {
                "conversation_id": conversation_id,
                "session_ids": list(session_ids),
                "formatted_context": formatted_context,
                "results": list(results),
            }
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": str(
                    self.config.get("name") or "openclaw-session-memory"
                ),
                "search_mode": "readback",
                "session_ids": session_ids,
                "formatted_context": formatted_context,
                "readback": readback.to_dict(),
            },
            retrieval_status=retrieval_status,
            timing_ms=(time.perf_counter() - started_at) * 1000,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        if self.use_openclaw_agent_answer:
            conversation_id = str(kwargs.get("conversation_id") or "").strip()
            runtime = self._runtime_for(conversation_id)
            self._ensure_answer_config(runtime)
            question_id = str(kwargs.get("question_id") or "").strip()
            self._ensure_readback_answer_payload_from_answer_kwargs(
                question_id=question_id,
                conversation_id=conversation_id,
                context=context,
                search_results=kwargs.get("search_results"),
                retrieval_metadata=kwargs.get("retrieval_metadata"),
            )
            runtime_message = self._get_openclaw_runtime_message(
                query,
                conversation_id=conversation_id,
                question_id=question_id,
            )
            answer_cfg = self.config.get("answer", {})
            timeout_seconds = int(answer_cfg.get("timeout_seconds", 120))
            openclaw_timeout_seconds = int(
                answer_cfg.get("openclaw_agent_timeout_seconds", timeout_seconds)
            )
            subprocess_timeout_seconds = int(
                answer_cfg.get(
                    "openclaw_subprocess_timeout_seconds",
                    openclaw_timeout_seconds + 30,
                )
            )
            max_retries = int(
                answer_cfg.get("openclaw_max_retries", 3)
            )
            retry_delay_seconds = float(
                answer_cfg.get("openclaw_retry_delay_seconds", 2.0)
            )
            session_stem = _safe_component(
                question_id or _short_hash(conversation_id + query), "question"
            )
            errors = []
            for attempt in range(max(1, max_retries)):
                session_id = (
                    f"answer-{session_stem}-{attempt + 1}-{uuid.uuid4().hex[:8]}"
                )
                if self._should_stage_answer_payload() and question_id:
                    payload = self._readback_answer_cache.get(question_id)
                    if payload:
                        self._write_readback_stage_manifest(
                            runtime=runtime,
                            answer_session_id=session_id,
                            question_id=question_id,
                            question_payload=payload,
                        )
                result = await asyncio.to_thread(
                    self._run_openclaw,
                    [
                        "agent",
                        "--local",
                        "--thinking",
                        "off",
                        "--timeout",
                        str(openclaw_timeout_seconds),
                        "--session-id",
                        session_id,
                        "--message",
                        runtime_message,
                        "--json",
                    ],
                    subprocess_timeout_seconds,
                    runtime,
                )
                payload = self._parse_json_output(result)
                agent_error = self._extract_openclaw_agent_error(payload)
                transcript_error = self._extract_session_error(session_id, runtime)
                command_error = ""
                if result.get("returncode") not in (0, None):
                    command_error = (
                        f"OpenClaw command exited with {result.get('returncode')}"
                    )
                if agent_error or transcript_error or command_error:
                    errors.append(
                        {
                            "attempt": attempt + 1,
                            "session_id": session_id,
                            "agent_error": agent_error,
                            "transcript_error": transcript_error,
                            "command_error": command_error,
                            "returncode": result.get("returncode"),
                            "stderr": str(result.get("stderr") or "")[:1000],
                        }
                    )
                    if attempt < max(1, max_retries) - 1 and retry_delay_seconds > 0:
                        await asyncio.sleep(retry_delay_seconds)
                    continue

                answer = self._extract_openclaw_response_text(payload)
                if answer and not self._is_openclaw_silent_reply(answer):
                    self._capture_openclaw_runtime_recall_payload(
                        question_id=question_id,
                        conversation_id=conversation_id,
                        session_id=session_id,
                        prompt=runtime_message,
                        recall_query=query,
                        runtime=runtime,
                        agent_payload=payload,
                    )
                    return self._clean_answer(answer)
                agent_error = (
                    "OpenClaw agent returned silent NO_REPLY."
                    if self._is_openclaw_silent_reply(answer)
                    else "OpenClaw agent returned no answer text."
                )
                errors.append(
                    {
                        "attempt": attempt + 1,
                        "session_id": session_id,
                        "agent_error": agent_error,
                        "returncode": result.get("returncode"),
                        "stdout": result.get("stdout", "")[:1000],
                        "stderr": result.get("stderr", "")[:1000],
                    }
                )
                if attempt < max(1, max_retries) - 1 and retry_delay_seconds > 0:
                    await asyncio.sleep(retry_delay_seconds)
            raise RuntimeError(
                "OpenClaw agent failed after retries: "
                f"{json.dumps(errors, ensure_ascii=False, default=_json_default)[:4000]}"
            )

        if self.llm_provider is None:
            raise RuntimeError(
                "OpenClaw agent answer mode is disabled without fallback."
            )
        return await self._answer_with_context_prompt(query, context)

    def render_answer_prompt(
        self, query: str, context: str, **kwargs: Any
    ) -> Optional[str]:
        if self.use_openclaw_agent_answer:
            conversation_id = str(kwargs.get("conversation_id") or "").strip()
            return self._get_openclaw_agent_prompt(
                query,
                conversation_id,
                str(kwargs.get("question_id") or "").strip(),
            )
        del kwargs
        return self._get_answer_prompt().format(context=context, question=query)

    def get_import_manifest_records(self) -> List[Dict[str, Any]]:
        return list(self._import_manifest_records)

    def pop_runtime_recall_payload(
        self, question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        key = str(question_id or "").strip()
        if key and key in self._runtime_recall_payloads:
            return self._runtime_recall_payloads.pop(key)
        if key:
            return {}
        if not self._runtime_recall_payloads:
            return {}
        first_key = next(iter(self._runtime_recall_payloads))
        return self._runtime_recall_payloads.pop(first_key)

    def pop_runtime_logger_payload(
        self, question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.pop_runtime_recall_payload(question_id)

    async def get_memory(self, memory_ref: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        del kwargs
        path = Path(
            str(memory_ref.get("path") or memory_ref.get("absolute_path") or "")
        )
        if not path.exists():
            return {
                "memory_ref": memory_ref,
                "storage_kind": "openclaw_session_memory_markdown",
                "content": "",
                "metadata": {},
                "status": "missing",
                "errors": [{"error": f"memory file not found: {path}"}],
            }
        return {
            "memory_ref": memory_ref,
            "storage_kind": "openclaw_session_memory_markdown",
            "content": path.read_text(encoding="utf-8"),
            "metadata": {"path": str(path)},
            "status": "ok",
            "errors": [],
        }

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "openclaw_session_memory",
            "config": self.config,
            "runtime": {
                "openclaw_cli": self.openclaw_cli,
                "openclaw_root": str(self.openclaw_root),
                "isolation": "conversation",
                "runtime_root": str(self.run_root),
                "conversation_runtime_root": str(self.conversation_runtime_root),
                "session_memory_handler_path": str(self.session_memory_handler_path),
                "session_memory_bridge_path": str(self.session_memory_bridge_path),
            },
        }

    def _runtime_for(self, conversation_id: str) -> OpenClawRuntime:
        key = _safe_component(conversation_id or "unknown")
        runtime = self._runtime_by_conversation.get(key)
        if runtime is not None:
            return runtime
        run_root = self.conversation_runtime_root / key
        state_dir = run_root / "state"
        workspace_dir = run_root / "workspace"
        runtime = OpenClawRuntime(
            conversation_id=key,
            run_root=run_root,
            state_dir=state_dir,
            workspace_dir=workspace_dir,
            config_path=state_dir / "openclaw.json",
            sessions_dir=state_dir / "agents" / "main" / "sessions",
            agent_dir=state_dir / "agents" / "main" / "agent",
            bridge_payload_dir=run_root / "session_memory_bridge" / "payloads",
            bridge_result_dir=run_root / "session_memory_bridge" / "results",
            bridge_workspace_dir=run_root
            / "session_memory_bridge"
            / "handler_workspaces",
            memory_dir=workspace_dir / "memory",
        )
        self._runtime_by_conversation[key] = runtime
        return runtime

    @staticmethod
    def _runtime_paths_payload(runtime: OpenClawRuntime) -> Dict[str, str]:
        return {
            "state_dir": str(runtime.state_dir),
            "workspace_dir": str(runtime.workspace_dir),
            "config_path": str(runtime.config_path),
            "memory_dir": str(runtime.memory_dir),
        }

    def _conversation_runtime_summary(
        self, conversation_ids: Iterable[str]
    ) -> Dict[str, Dict[str, str]]:
        summary: Dict[str, Dict[str, str]] = {}
        for conversation_id in sorted({str(value or "") for value in conversation_ids}):
            if not conversation_id:
                continue
            summary[conversation_id] = self._runtime_paths_payload(
                self._runtime_for(conversation_id)
            )
        return summary

    def _prepare_runtime_dirs(self, runtime: OpenClawRuntime) -> None:
        runtime.state_dir.mkdir(parents=True, exist_ok=True)
        runtime.workspace_dir.mkdir(parents=True, exist_ok=True)
        runtime.sessions_dir.mkdir(parents=True, exist_ok=True)
        runtime.agent_dir.mkdir(parents=True, exist_ok=True)
        runtime.bridge_payload_dir.mkdir(parents=True, exist_ok=True)
        runtime.bridge_result_dir.mkdir(parents=True, exist_ok=True)
        runtime.bridge_workspace_dir.mkdir(parents=True, exist_ok=True)
        runtime.memory_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_transcript_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_memory_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_qa_workspace(runtime)
        self._ensure_agent_profile_files(runtime)
        self._ensure_config(runtime)

    def _ensure_config(self, runtime: OpenClawRuntime) -> None:
        runtime.state_dir.mkdir(parents=True, exist_ok=True)
        if runtime.config_path.exists():
            cfg = json.loads(runtime.config_path.read_text(encoding="utf-8"))
        elif self.base_config_path.exists():
            cfg = json.loads(self.base_config_path.read_text(encoding="utf-8"))
        else:
            cfg = {}

        cfg = self._derive_openclaw_config(cfg, runtime)
        runtime.config_path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._ensure_agent_profile_files(runtime)

    def _ensure_answer_config(self, runtime: OpenClawRuntime) -> None:
        runtime.state_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_qa_workspace(runtime)
        if runtime.config_path.exists():
            cfg = json.loads(runtime.config_path.read_text(encoding="utf-8"))
        elif self.base_config_path.exists():
            cfg = json.loads(self.base_config_path.read_text(encoding="utf-8"))
        else:
            cfg = {}

        cfg = self._derive_openclaw_config(
            cfg, runtime, use_readback_plugin=self._is_readback_search_mode()
        )
        runtime.config_path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._ensure_agent_profile_files(runtime)

    def _initialize_qa_workspace(self, runtime: OpenClawRuntime) -> None:
        runtime.workspace_dir.mkdir(parents=True, exist_ok=True)
        runtime.memory_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "AGENTS.md": self._evaluation_agents_md(),
            "USER.md": "",
            "MEMORY.md": "",
            "IDENTITY.md": "",
            "SOUL.md": "",
            "TOOLS.md": "",
            "HEARTBEAT.md": "",
        }
        for filename, content in files.items():
            path = runtime.workspace_dir / filename
            path.write_text(content, encoding="utf-8")

    def _evaluation_agents_md(self) -> str:
        agents_md = OPENCLAW_EVALUATION_AGENTS_MD
        if (
            getattr(self, "adapter_id", "") == "openclaw_session_memory"
            and not self._is_readback_search_mode()
        ):
            anchor = (
                "- If memory_search is available and no recalled memory/context has "
                "already been injected, call memory_search with the current question "
                "before answering."
            )
            if (
                OPENCLAW_SESSION_MEMORY_API_FORCE_SEARCH_INSTRUCTION not in agents_md
                and anchor in agents_md
            ):
                agents_md = agents_md.replace(
                    anchor,
                    f"{anchor}\n{OPENCLAW_SESSION_MEMORY_API_FORCE_SEARCH_INSTRUCTION}",
                    1,
                )
        return agents_md

    def _ensure_agent_profile_files(self, runtime: OpenClawRuntime) -> None:
        runtime.agent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._write_json_private(
            runtime.agent_dir / "auth-profiles.json",
            self._build_qa_auth_profiles_payload(),
        )
        self._write_json_private(
            runtime.agent_dir / "models.json", self._build_qa_models_json_payload()
        )

    def _derive_openclaw_config(
        self,
        cfg: Dict[str, Any],
        runtime: OpenClawRuntime,
        *,
        use_readback_plugin: bool = False,
    ) -> Dict[str, Any]:
        next_cfg = json.loads(json.dumps(cfg))
        self._ensure_model_provider_config(next_cfg)
        is_readback = use_readback_plugin and self._is_readback_search_mode()

        model_id = self._qa_model_id()
        defaults = next_cfg.setdefault("agents", {}).setdefault("defaults", {})
        defaults["workspace"] = str(runtime.workspace_dir)
        defaults["model"] = {"primary": f"openai/{model_id}"}
        model_defaults: Dict[str, Any] = {}
        model_params = self._qa_model_extra_params()
        if model_params:
            model_defaults["params"] = model_params
        defaults["models"] = {f"openai/{model_id}": model_defaults}
        heartbeat = defaults.setdefault("heartbeat", {})
        heartbeat["every"] = "0m"
        heartbeat["includeSystemPromptSection"] = False
        context_limits = defaults.setdefault("contextLimits", {})
        context_limits.update(
            {
                "memoryGetDefaultLines": 80,
                "memoryGetMaxChars": 20000,
                "toolResultMaxChars": 24000,
            }
        )
        memory_search = defaults.setdefault("memorySearch", {})
        memory_cfg = self.config.get("memory", {}) or {}
        openclaw_cfg = self.config.get("openclaw", {}) or {}
        embedding_provider = (
            self._first_non_empty(
                os.environ.get("OPENCLAW_MEMORY_SEARCH_PROVIDER"),
                openclaw_cfg.get("memory_search_provider"),
                memory_cfg.get("provider"),
            )
            or "auto"
        )
        embedding_model = self._first_non_empty(
            os.environ.get("OPENCLAW_MEMORY_SEARCH_MODEL"),
            openclaw_cfg.get("memory_search_model"),
            memory_cfg.get("model"),
        )
        memory_search_payload: Dict[str, Any] = {
            "enabled": True,
            "sources": ["memory"],
            "fallback": "none",
            "extraPaths": [str(runtime.memory_dir)],
            "provider": embedding_provider,
            "sync": {
                "onSessionStart": False,
                "onSearch": False,
                "watch": False,
                "watchDebounceMs": 1500,
                "intervalMinutes": 0,
            },
        }
        if embedding_model is not None:
            memory_search_payload["model"] = embedding_model
        if embedding_provider != "local":
            memory_search_payload["remote"] = {
                "apiKey": self._env_secret_ref(self._memory_search_api_key_env()),
                "baseUrl": self._memory_search_base_url(next_cfg),
                "batch": {
                    "enabled": bool(
                        self.config.get("openclaw", {}).get(
                            "memory_search_batch_enabled", False
                        )
                    ),
                    "wait": True,
                    "concurrency": 1,
                    "pollIntervalMs": 2000,
                    "timeoutMinutes": 60,
                },
            }
        memory_search.update(memory_search_payload)
        query_cfg = memory_search.setdefault("query", {})
        query_cfg["maxResults"] = self.search_top_k
        query_cfg["minScore"] = self.min_score
        query_cfg.setdefault(
            "hybrid",
            {
                "enabled": True,
                "vectorWeight": 0.55,
                "textWeight": 0.45,
                "candidateMultiplier": 8,
                "mmr": {"enabled": True, "lambda": 0.7},
            },
        )
        compaction = defaults.setdefault("compaction", {})
        memory_flush = compaction.setdefault("memoryFlush", {})
        memory_flush["enabled"] = False

        next_cfg["agents"]["list"] = [{"id": "main"}]
        hooks = (
            next_cfg.setdefault("hooks", {})
            .setdefault("internal", {})
            .setdefault("entries", {})
        )
        session_memory_hook = hooks.setdefault("session-memory", {})
        session_memory_hook["enabled"] = False
        session_memory_hook["messages"] = self.session_memory_messages
        session_memory_hook["llmSlug"] = False

        next_cfg["gateway"] = {
            "mode": "local",
            "auth": {"mode": "none"},
        }
        plugin_entries: Dict[str, Any] = {
            "openai": {"enabled": True},
            "memory-core": {"enabled": True, "config": {}},
        }
        allow = ["openai", "memory-core"]
        if is_readback:
            plugin_id = "openclaw-session-memory-readback"
            plugin_entries[plugin_id] = {"enabled": True}
            if plugin_id not in allow:
                allow.append(plugin_id)
            load_paths = [
                str(
                    (
                        Path(__file__).parent
                        / "plugins"
                        / "session_memory_readback"
                    ).resolve()
                )
            ]
            slots = {"memory": "memory-core"}
        else:
            load_paths = []
            slots = {"memory": "memory-core"}
        load_paths = self._openclaw_builtin_plugin_load_paths() + load_paths
        next_cfg["plugins"] = {
            "enabled": True,
            "entries": plugin_entries,
            "allow": allow,
            "load": {"paths": load_paths},
            "slots": slots,
        }

        tools = next_cfg.setdefault("tools", {})
        tools["profile"] = "minimal"
        tools["alsoAllow"] = ["memory_search"]
        tools["deny"] = [
            "write",
            "edit",
            "apply_patch",
            "exec",
            "process",
            "cron",
            "web_search",
            "web_fetch",
            "image_generate",
            "browser",
            "message",
            "session_status",
        ]
        tools.setdefault("sessions", {}).setdefault("visibility", "agent")
        browser = next_cfg.setdefault("browser", {})
        browser["enabled"] = False
        return next_cfg

    def _openclaw_builtin_plugin_load_paths(self) -> List[str]:
        configured = (
            self.config.get("openclaw", {}).get("builtin_plugin_load_paths")
            or self.config.get("openclaw", {}).get("built_plugin_load_paths")
            or []
        )
        paths: List[str] = []
        if isinstance(configured, (str, os.PathLike)):
            configured = [configured]
        if isinstance(configured, list):
            for value in configured:
                if value:
                    paths.append(str(_absolute_path(value)))

        for plugin_id in OPENCLAW_BUILT_PLUGIN_LOAD_IDS:
            for candidate in (
                self.openclaw_root / "dist" / "extensions" / plugin_id,
                self.openclaw_root / "extensions" / plugin_id,
            ):
                if candidate.exists() and candidate.is_dir():
                    paths.append(str(candidate.resolve(strict=False)))
                    break

        deduped: List[str] = []
        for path in paths:
            if path not in deduped:
                deduped.append(path)
        return deduped

    @staticmethod
    def _env_secret_ref(env_name: str) -> Dict[str, str]:
        return {"source": "env", "provider": "default", "id": env_name}

    @staticmethod
    def _first_non_empty(*values: Optional[str]) -> Optional[str]:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _qa_model_id(self) -> str:
        return (
            self._first_non_empty(
                os.environ.get("OPENCLAW_NATIVE_OPENAI_MODEL"),
                self.config.get("answer", {}).get("model"),
                self.config.get("llm", {}).get("model"),
            )
            or "gpt-4o-mini"
        )

    def _qa_model_api(self) -> str:
        return (
            self._first_non_empty(
                os.environ.get("OPENCLAW_NATIVE_OPENAI_API"),
                self.config.get("answer", {}).get("model_api"),
                self.config.get("llm", {}).get("api"),
            )
            or "openai-completions"
        )

    def _qa_model_extra_params(self) -> Dict[str, Any]:
        llm_cfg = self.config.get("llm", {}) or {}
        answer_cfg = self.config.get("answer", {}) or {}
        if "parallel_tool_calls" in llm_cfg:
            return {"parallel_tool_calls": llm_cfg.get("parallel_tool_calls")}
        if "parallel_tool_calls" in answer_cfg:
            return {"parallel_tool_calls": answer_cfg.get("parallel_tool_calls")}
        return {}

    def _qa_base_url(self) -> str:
        return (
            self._first_non_empty(
                os.environ.get("OPENCLAW_NATIVE_OPENAI_BASE_URL"),
                self.config.get("llm", {}).get("base_url"),
                os.environ.get("ANSWER_LLM_BASE_URL"),
                os.environ.get("LLM_BASE_URL"),
                os.environ.get("OPENAI_BASE_URL"),
            )
            or "https://api.openai.com/v1"
        )

    def _qa_api_key(self) -> str:
        return (
            self._first_non_empty(
                os.environ.get("OPENCLAW_NATIVE_OPENAI_API_KEY"),
                self.config.get("llm", {}).get("api_key"),
                os.environ.get("ANSWER_LLM_API_KEY"),
                os.environ.get("LLM_API_KEY"),
                os.environ.get("OPENAI_API_KEY"),
            )
            or ""
        )

    def _build_qa_auth_profiles_payload(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "profiles": {
                "openai:default": {
                    "type": "api_key",
                    "provider": "openai",
                    "keyRef": self._env_secret_ref("OPENAI_API_KEY"),
                }
            },
            "lastGood": {"openai": "openai:default"},
        }

    def _build_qa_models_json_payload(self) -> Dict[str, Any]:
        model_id = self._qa_model_id()
        model_api = self._qa_model_api()
        return {
            "providers": {
                "openai": {
                    "baseUrl": self._qa_base_url(),
                    "apiKey": self._env_secret_ref("OPENAI_API_KEY"),
                    "auth": "api-key",
                    "api": model_api,
                    "authHeader": True,
                    "models": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "api": model_api,
                            "input": ["text"],
                            "reasoning": False,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                            "contextWindow": 200000,
                            "maxTokens": 8192,
                        }
                    ],
                }
            }
        }

    def _memory_search_api_key_env(self) -> str:
        return (
            self._first_non_empty(
                os.environ.get("OPENCLAW_MEMORY_SEARCH_API_KEY_ENV"),
                self.config.get("openclaw", {}).get("memory_search_api_key_env"),
            )
            or "OPENAI_API_KEY"
        )

    def _memory_search_base_url(self, cfg: Dict[str, Any]) -> str:
        return str(
            self._first_non_empty(
                os.environ.get("OPENCLAW_MEMORY_SEARCH_BASE_URL"),
                self.config.get("openclaw", {}).get("memory_search_base_url"),
            )
            or self._openai_base_url(cfg)
        )

    def _ensure_model_provider_config(self, cfg: Dict[str, Any]) -> None:
        model_id = self._qa_model_id()
        model_api = self._qa_model_api()
        model_cfg: Dict[str, Any] = {
            "id": model_id,
            "name": model_id,
            "api": model_api,
            "input": ["text"],
        }
        cfg["models"] = {
            "mode": "replace",
            "providers": {
                "openai": {
                    "baseUrl": self._qa_base_url(),
                    "apiKey": self._env_secret_ref("OPENAI_API_KEY"),
                    "auth": "api-key",
                    "api": model_api,
                    "authHeader": True,
                    "models": [model_cfg],
                }
            },
        }

    @staticmethod
    def _openai_base_url(cfg: Dict[str, Any]) -> str:
        provider_cfg = (
            cfg.get("models", {}).get("providers", {}).get("openai", {})
            if isinstance(cfg.get("models"), dict)
            else {}
        )
        return str(
            provider_cfg.get("baseUrl")
            or os.environ.get("ANSWER_LLM_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

    def _cache_conversations(self, conversations: List[Conversation]) -> None:
        self._conversations = {conv.conversation_id: conv for conv in conversations}

    def _group_messages_by_session(
        self, conversation: Conversation
    ) -> Iterable[Tuple[str, List[Message]]]:
        groups: Dict[str, List[Message]] = {}
        for message in conversation.messages:
            session_key = str(message.metadata.get("session") or "session_0")
            groups.setdefault(session_key, []).append(message)

        def sort_key(item: Tuple[str, List[Message]]) -> Tuple[Any, str]:
            messages = item[1]
            order = messages[0].metadata.get("session_order") if messages else None
            try:
                return (int(order), item[0])
            except Exception:
                return (999999, item[0])

        yield from sorted(groups.items(), key=sort_key)

    def _write_session_memory(
        self,
        conversation: Conversation,
        session_key: str,
        messages: List[Message],
        runtime: OpenClawRuntime,
    ) -> ImportedMemory:
        first = messages[0]
        source_session_id = str(
            first.metadata.get("source_session_id")
            or session_key
            or conversation.conversation_id
        )
        source_case_id = str(first.metadata.get("source_case_id") or "")
        session_order = first.metadata.get("session_order")
        session_source = str(first.metadata.get("source_session_source") or "")
        session_timestamp = str(
            first.metadata.get("source_session_timestamp") or _utc_iso(first.timestamp)
        )
        session_id = (
            f"{_safe_component(conversation.conversation_id)}--"
            f"{_safe_component(source_session_id or session_key)}"
        )
        transcript_path = runtime.sessions_dir / f"{session_id}.jsonl"
        artifact_transcript_path = self.artifact_transcript_dir / f"{session_id}.jsonl"
        memory_path = runtime.memory_dir / f"{session_id}.md"
        artifact_memory_path = self.artifact_memory_dir / f"{session_id}.md"

        self._write_transcript(transcript_path, session_id, messages, runtime)
        shutil.copy2(transcript_path, artifact_transcript_path)
        bridge_result = self._run_session_memory_handler(
            session_id=session_id,
            session_key=self._openclaw_session_key(session_id),
            transcript_path=transcript_path,
            memory_path=memory_path,
            timestamp=session_timestamp,
            runtime=runtime,
        )
        content = memory_path.read_text(encoding="utf-8")
        shutil.copy2(memory_path, artifact_memory_path)

        source_unit_ids = [
            str(
                message.metadata.get("source_unit_id")
                or message.metadata.get("dia_id")
                or ""
            )
            for message in messages
            if message.metadata.get("source_unit_id") or message.metadata.get("dia_id")
        ]
        metadata = {
            "conversation_id": conversation.conversation_id,
            "session_key": session_key,
            "source_session_id": source_session_id,
            "source_case_id": source_case_id,
            "session_order": session_order,
            "session_source": session_source,
            "session_timestamp": session_timestamp,
            "artifact_transcript_path": str(artifact_transcript_path),
            "artifact_memory_path": str(artifact_memory_path),
            "session_memory_handler_path": str(self.session_memory_handler_path),
            "session_memory_bridge_result": bridge_result,
            "openclaw_isolation": "conversation",
            "openclaw_runtime": {
                "state_dir": str(runtime.state_dir),
                "workspace_dir": str(runtime.workspace_dir),
                "config_path": str(runtime.config_path),
                "memory_dir": str(runtime.memory_dir),
            },
        }
        return ImportedMemory(
            conversation_id=conversation.conversation_id,
            session_key=session_key,
            source_session_id=source_session_id,
            source_case_id=source_case_id,
            session_order=session_order,
            session_source=session_source,
            session_timestamp=session_timestamp,
            transcript_path=str(transcript_path),
            memory_path=str(memory_path),
            content=content,
            source_unit_ids=source_unit_ids,
            message_count=len(messages),
            metadata=metadata,
        )

    def _write_transcript(
        self,
        path: Path,
        session_id: str,
        messages: List[Message],
        runtime: OpenClawRuntime,
    ) -> None:
        parent_id = None
        started = _utc_iso(messages[0].timestamp if messages else None)
        lines = [
            json.dumps(
                {
                    "type": "session",
                    "version": CURRENT_SESSION_VERSION,
                    "id": session_id,
                    "timestamp": started,
                    "cwd": str(runtime.workspace_dir),
                },
                ensure_ascii=False,
                default=_json_default,
            )
        ]
        for index, message in enumerate(messages):
            timestamp = _utc_iso(message.timestamp)
            role = self._openclaw_message_role(message)
            text = self._openclaw_message_text(message)
            msg_id = _short_hash(f"{session_id}:{index}:{role}:{message.content}", 12)
            lines.append(
                json.dumps(
                    {
                        "type": "message",
                        "id": msg_id,
                        "parentId": parent_id,
                        "timestamp": timestamp,
                        "message": {
                            "role": role,
                            "content": [{"type": "text", "text": text}],
                            "timestamp": _timestamp_ms(timestamp),
                        },
                    },
                    ensure_ascii=False,
                    default=_json_default,
                )
            )
            parent_id = msg_id
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _openclaw_message_role(message: Message) -> str:
        sender = str(message.sender_name or "").strip().lower()
        role = str(message.metadata.get("role") or "").strip().lower()
        if sender == "assistant" or role == "assistant":
            return "assistant"
        return "user"

    @staticmethod
    def _openclaw_message_text(message: Message) -> str:
        text = str(message.content or "")
        sender_name = str(message.sender_name or "").strip()
        if not sender_name or sender_name.lower() in {"user", "assistant"}:
            return text
        return f"{sender_name}: {text}"

    def _run_session_memory_handler(
        self,
        *,
        session_id: str,
        session_key: str,
        transcript_path: Path,
        memory_path: Path,
        timestamp: str,
        runtime: OpenClawRuntime,
    ) -> Dict[str, Any]:
        if not self.session_memory_handler_path.exists():
            raise FileNotFoundError(
                f"OpenClaw session-memory handler not found: "
                f"{self.session_memory_handler_path}"
            )
        if not self.session_memory_bridge_path.exists():
            raise FileNotFoundError(
                f"OpenClaw session-memory bridge not found: "
                f"{self.session_memory_bridge_path}"
            )

        bridge_payload_path = runtime.bridge_payload_dir / f"{session_id}.json"
        bridge_result_path = runtime.bridge_result_dir / f"{session_id}.json"
        handler_workspace = runtime.bridge_workspace_dir / session_id
        payload = {
            "handlerPath": str(self.session_memory_handler_path),
            "configPath": str(runtime.config_path),
            "memoryDir": str(runtime.memory_dir),
            "handlerWorkspaceDir": str(handler_workspace),
            "sessionKey": session_key,
            "sessionId": session_id,
            "sessionFile": str(transcript_path),
            "timestamp": timestamp,
            "targetFileName": memory_path.name,
            "resultPath": str(bridge_result_path),
            "commandSource": "evaluation-import",
        }
        self._write_json_private(bridge_payload_path, payload)
        result = self._run_node(
            [str(self.session_memory_bridge_path), str(bridge_payload_path)],
            cwd=self.openclaw_root,
            timeout=180,
            runtime=runtime,
        )
        if result["returncode"] != 0:
            raise RuntimeError(
                "OpenClaw session-memory handler failed: "
                f"{json.dumps(result, ensure_ascii=False, default=_json_default)[:4000]}"
            )
        if not memory_path.exists():
            raise RuntimeError(
                "OpenClaw session-memory handler completed but did not create "
                f"expected memory file: {memory_path}"
            )
        bridge_result = (
            json.loads(bridge_result_path.read_text(encoding="utf-8"))
            if bridge_result_path.exists()
            else {}
        )
        return {
            **bridge_result,
            "cmd": result["cmd"],
            "returncode": result["returncode"],
        }

    @staticmethod
    def _openclaw_session_key(session_id: str) -> str:
        return f"agent:main:{session_id}"

    def _readback_stage_root(self, runtime: OpenClawRuntime) -> Path:
        return runtime.workspace_dir / ".openclaw-readback"

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
        conversation_id = str(
            question_payload.get("conversation_id") or runtime.conversation_id or ""
        ).strip()
        files = []
        for item in question_payload.get("results", []):
            metadata = dict(item.get("metadata") or {})
            absolute_path = (
                str(metadata.get("path") or metadata.get("absolute_path") or "").strip()
            )
            file_record = {
                "sourceSessionId": metadata.get("source_session_id", ""),
                "relativeLabel": self._readback_relative_memory_label(
                    conversation_id, metadata
                ),
                "absolutePath": absolute_path,
            }
            if not absolute_path and item.get("content"):
                file_record["content"] = item.get("content", "")
            files.append(file_record)
        formatted_context = str(question_payload.get("formatted_context") or "")
        payload = {
            "sessionId": answer_session_id,
            "questionId": question_id,
            "conversationId": conversation_id,
            "sessionIds": list(question_payload.get("session_ids") or []),
            "formattedContext": formatted_context,
            "prependContext": formatted_context,
            "readback": question_payload.get("readback", {}),
            "files": files,
        }
        self._write_json_private(manifest_path, payload)
        return manifest_path

    @staticmethod
    def _ensure_markdown_name(name: str) -> str:
        text = str(name or "").strip()
        if not text:
            return ""
        return text if text.endswith(".md") else f"{text}.md"

    def _readback_relative_memory_label(
        self, conversation_id: str, metadata: Dict[str, Any]
    ) -> str:
        conversation_id = str(conversation_id or "").strip()
        source_session_id = str(metadata.get("source_session_id") or "").strip()
        if source_session_id:
            if conversation_id and source_session_id.startswith(f"{conversation_id}--"):
                basename = source_session_id
            elif conversation_id:
                basename = f"{conversation_id}--{source_session_id}"
            else:
                basename = source_session_id
            return f"memory/{self._ensure_markdown_name(basename)}"

        raw_path = str(metadata.get("path") or metadata.get("absolute_path") or "").strip()
        if raw_path:
            basename = Path(raw_path).name
            if basename:
                return f"memory/{self._ensure_markdown_name(basename)}"

        memory_id = str(metadata.get("memory_id") or metadata.get("id") or "").strip()
        if memory_id:
            return f"memory/{self._ensure_markdown_name(memory_id)}"

        fallback = f"{conversation_id}--memory" if conversation_id else "memory"
        return f"memory/{self._ensure_markdown_name(fallback)}"

    def _readback_question_payload(self, question_id: str) -> Dict[str, Any]:
        key = str(question_id or "").strip()
        if not key:
            return {}
        payload = self._readback_answer_cache.get(key)
        return dict(payload or {})

    def _ensure_readback_answer_payload_from_answer_kwargs(
        self,
        *,
        question_id: str,
        conversation_id: str,
        context: str,
        search_results: Any,
        retrieval_metadata: Any,
    ) -> None:
        if not (
            self._should_use_readback_answer_payload()
            and question_id
            and question_id not in self._readback_answer_cache
        ):
            return
        payload = self._readback_question_payload_from_answer_kwargs(
            question_id=question_id,
            conversation_id=conversation_id,
            context=context,
            search_results=search_results,
            retrieval_metadata=retrieval_metadata,
        )
        if payload:
            self._readback_answer_cache[question_id] = payload

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
        if not formatted_context:
            formatted_context = self._format_readback_results_for_context(results)
        return {
            "conversation_id": conversation_id,
            "session_ids": list(retrieval_metadata.get("session_ids") or []),
            "formatted_context": formatted_context,
            "results": list(results),
            "readback": retrieval_metadata.get("readback", {}),
        }

    @staticmethod
    def _format_readback_results_for_context(results: List[Dict[str, Any]]) -> str:
        context_parts = []
        for item in results:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if content:
                context_parts.append(f"{len(context_parts) + 1}. {content}")
        if not context_parts:
            return ""
        return (
            "<recalled-memories>\n"
            f"{chr(10).join(context_parts)}\n"
            "</recalled-memories>"
        )

    def _readback_target_memory_files(
        self, conversation_id: str, question_id: str = ""
    ) -> List[str]:
        payload = self._readback_question_payload(question_id)
        files: List[str] = []
        seen = set()
        for item in payload.get("results", []) or []:
            metadata = item.get("metadata") if isinstance(item, dict) else None
            if not isinstance(metadata, dict):
                metadata = {}
            rel_path = self._readback_relative_memory_label(conversation_id, metadata)
            if rel_path in seen:
                continue
            seen.add(rel_path)
            files.append(rel_path)
        return files

    def _build_manifest_record(self, memory: ImportedMemory) -> Dict[str, Any]:
        run_id = str((self.run_context or {}).get("run_id") or "")
        system_id = str(
            (self.run_context or {}).get("system_id") or "openclaw_session_memory"
        )
        chunk_id = f"{memory.conversation_id}:{memory.session_key}"
        record = ImportManifestRecord(
            run_id=run_id,
            system_id=system_id,
            conversation_id=memory.conversation_id,
            view_id=memory.session_key,
            chunk_id=chunk_id,
            source_unit_ids=memory.source_unit_ids,
            write_request_summary={
                "message_count": memory.message_count,
                "source_session_id": memory.source_session_id,
                "source_case_id": memory.source_case_id,
                "session_order": memory.session_order,
                "transport": "openclaw_session_memory_markdown",
            },
            write_receipt={
                "provider_status": "written",
                "transcript_path": memory.transcript_path,
                "memory_path": memory.memory_path,
            },
            memory_refs=[
                {
                    "type": "openclaw_session_memory_markdown",
                    "path": memory.memory_path,
                    "absolute_path": memory.memory_path,
                    "source_session_id": memory.source_session_id,
                    "source_case_id": memory.source_case_id,
                    "session_key": memory.session_key,
                    "openclaw_isolation": "conversation",
                    "openclaw_runtime": memory.metadata.get("openclaw_runtime", {}),
                }
            ],
            write_status="written",
            errors=[],
        )
        return dataclass_to_dict(record)

    def _build_index_metadata(self) -> Dict[str, Any]:
        return {
            "type": "openclaw_session_memory_markdown",
            "conversation_ids": list(self._conversations.keys()),
            "memory_count": len(self._memories),
            "isolation": "conversation",
            "runtime_root": str(self.run_root),
            "conversation_runtimes": self._conversation_runtime_summary(
                self._conversations.keys()
                or [memory.conversation_id for memory in self._memories]
            ),
        }

    def _get_answer_prompt(self) -> str:
        dataset_id = str((self.run_context or {}).get("dataset_id", "")).strip()
        if dataset_id.startswith("subtlememory"):
            return get_subtlememory_unified_answer_prompt(config=self.config)
        return ORACLE_ANSWER_PROMPT

    def _is_readback_search_mode(self) -> bool:
        return (
            str((self.config.get("search") or {}).get("mode") or "")
            .strip()
            .lower()
            == "readback"
        )

    def _should_use_readback_answer_payload(self) -> bool:
        return self._is_readback_search_mode()

    def _should_stage_answer_payload(self) -> bool:
        return self._should_use_readback_answer_payload()

    async def _answer_with_context_prompt(self, query: str, context: str) -> str:
        if self.llm_provider is None:
            llm_config = self.config.get("llm", {})
            self.llm_provider = LLMProvider(
                provider_type=llm_config.get("provider", "openai"),
                model=llm_config.get("model", "gpt-4o-mini"),
                api_key=llm_config.get("api_key", ""),
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
                temperature=llm_config.get("temperature", 0.0),
                max_tokens=llm_config.get("max_tokens", 16384),
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

    @staticmethod
    def _extract_question_session_ids(metadata: Dict[str, Any]) -> List[str]:
        raw_session_ids = metadata.get("session_ids")
        if isinstance(raw_session_ids, list):
            values = raw_session_ids
        elif raw_session_ids:
            values = [raw_session_ids]
        else:
            values = []
        return OpenClawSessionMemoryAdapter._dedupe_nonempty_strings(values)

    @staticmethod
    def _dedupe_nonempty_strings(values: Iterable[Any]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _unsupported_readback_search_result(
        self,
        *,
        query: str,
        conversation_id: str,
        question_id: Optional[str],
        session_ids: List[str],
        reason: str,
    ) -> SearchResult:
        readback = StorageReadbackResult(
            status="unsupported",
            checked_session_ids=list(session_ids),
            missing_session_ids=[],
            objects=[],
            metadata={
                "provider": "openclaw_session_memory",
                "question_id": question_id,
                "conversation_id": conversation_id,
                "reason": reason,
                "readback_scope": "question_sessions",
            },
            errors=[],
        )
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={
                "system": str(
                    self.config.get("name") or "openclaw-session-memory"
                ),
                "search_mode": "readback",
                "session_ids": list(session_ids),
                "formatted_context": "",
                "readback": readback.to_dict(),
            },
            retrieval_status="unsupported",
        )

    def _readback_import_manifest_records(
        self, import_manifest_records: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if import_manifest_records:
            return list(import_manifest_records)
        if self._import_manifest_records:
            return list(self._import_manifest_records)
        manifest_path = self.output_dir / "import_manifest.jsonl"
        if manifest_path.exists():
            return self._read_jsonl(manifest_path)
        return []

    def _openclaw_readback_objects_from_records(
        self,
        *,
        conversation_id: str,
        session_ids: List[str],
        import_manifest_records: List[Dict[str, Any]],
    ) -> Tuple[List[NormalizedStorageObject], List[Dict[str, Any]]]:
        requested = set(session_ids)
        objects: List[NormalizedStorageObject] = []
        errors: List[Dict[str, Any]] = []
        seen_refs = set()
        for row in import_manifest_records:
            row_conversation_id = str(row.get("conversation_id") or "")
            if row_conversation_id and row_conversation_id != conversation_id:
                continue
            row_summary = row.get("write_request_summary") or {}
            for memory_ref in row.get("memory_refs") or []:
                if not isinstance(memory_ref, dict):
                    continue
                source_session_id = str(
                    memory_ref.get("source_session_id")
                    or row_summary.get("source_session_id")
                    or memory_ref.get("session_key")
                    or row.get("view_id")
                    or ""
                ).strip()
                if source_session_id not in requested:
                    continue
                path = self._resolve_readback_memory_path(memory_ref)
                identity = (source_session_id, str(path))
                if identity in seen_refs:
                    continue
                seen_refs.add(identity)
                obj = self._openclaw_readback_object_from_path(
                    path=path,
                    source_session_id=source_session_id,
                    conversation_id=conversation_id,
                    session_key=str(
                        memory_ref.get("session_key") or row.get("view_id") or ""
                    ),
                    errors=errors,
                    session_timestamp=str(
                        memory_ref.get("session_timestamp")
                        or row_summary.get("session_timestamp")
                        or ""
                    ),
                )
                if obj is not None:
                    objects.append(obj)
        objects.sort(key=self._readback_object_sort_key)
        return objects, errors

    def _openclaw_readback_objects_from_memory_cache(
        self,
        *,
        conversation_id: str,
        session_ids: List[str],
        errors: List[Dict[str, Any]],
    ) -> Tuple[List[NormalizedStorageObject], List[Dict[str, Any]]]:
        if not self._memories:
            self._load_imported_memories()
        requested = set(session_ids)
        objects: List[NormalizedStorageObject] = []
        for memory in self._memories_by_conversation.get(conversation_id, []):
            if memory.source_session_id not in requested:
                continue
            content = self._render_openclaw_readback_content(memory.content)
            if not content:
                continue
            objects.append(
                NormalizedStorageObject(
                    session_id=memory.source_session_id,
                    kind="openclaw_session_memory_markdown",
                    id=Path(memory.memory_path).stem,
                    content=content,
                    metadata={
                        "provider": "openclaw_session_memory",
                        "memory_id": Path(memory.memory_path).stem,
                        "kind": "openclaw_session_memory_markdown",
                        "path": memory.memory_path,
                        "source_session_id": memory.source_session_id,
                        "session_key": memory.session_key,
                        "session_timestamp": memory.session_timestamp,
                        "conversation_id": memory.conversation_id,
                    },
                )
            )
        objects.sort(key=self._readback_object_sort_key)
        return objects, errors

    def _resolve_readback_memory_path(self, memory_ref: Dict[str, Any]) -> Path:
        raw_path = str(memory_ref.get("absolute_path") or memory_ref.get("path") or "")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.output_dir / path
        return path.expanduser().resolve(strict=False)

    def _openclaw_readback_object_from_path(
        self,
        *,
        path: Path,
        source_session_id: str,
        conversation_id: str,
        session_key: str,
        errors: List[Dict[str, Any]],
        session_timestamp: str = "",
    ) -> Optional[NormalizedStorageObject]:
        if not path.exists():
            errors.append(
                {
                    "stage": "search_from_readback",
                    "error_type": "missing_memory_file",
                    "source_session_id": source_session_id,
                    "path": str(path),
                }
            )
            return None
        markdown = path.read_text(encoding="utf-8")
        content = self._render_openclaw_readback_content(markdown)
        if not content:
            return None
        return NormalizedStorageObject(
            session_id=source_session_id,
            kind="openclaw_session_memory_markdown",
            id=path.stem,
            content=content,
            metadata={
                "provider": "openclaw_session_memory",
                "memory_id": path.stem,
                "kind": "openclaw_session_memory_markdown",
                "path": str(path),
                "source_session_id": source_session_id,
                "session_key": session_key,
                "session_timestamp": session_timestamp,
                "conversation_id": conversation_id,
            },
        )

    @staticmethod
    def _readback_object_sort_key(obj: NormalizedStorageObject) -> Tuple[float, str]:
        raw_timestamp = str((obj.metadata or {}).get("session_timestamp") or "")
        timestamp = float("inf")
        if raw_timestamp:
            try:
                timestamp = datetime.fromisoformat(
                    raw_timestamp.replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                timestamp = float("inf")
        return (timestamp, str(obj.id or ""))

    @staticmethod
    def _render_openclaw_readback_content(markdown: str) -> str:
        text = str(markdown or "").strip()
        if not text:
            return ""
        match = re.search(r"^##\s+Conversation Summary\s*$", text, flags=re.MULTILINE)
        if match:
            summary = text[match.end() :].strip()
            if summary:
                return summary

        lines = text.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and lines[0].lstrip().startswith("#"):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        return "\n".join(lines).strip()

    def _get_openclaw_agent_prompt(
        self,
        query: str,
        conversation_id: str = "",
        question_id: str = "",
    ) -> str:
        conversation_id = conversation_id or "unknown"
        template = self._get_openclaw_agent_prompt_template()
        if not template.strip():
            return query
        prompt_values = {
            "question": query,
            "conversation_id": conversation_id,
        }
        return template.format(**prompt_values)

    def _get_openclaw_runtime_message(
        self,
        query: str,
        *,
        conversation_id: str = "",
        question_id: str = "",
    ) -> str:
        return self._get_openclaw_agent_prompt(query, conversation_id, question_id)

    def _get_openclaw_agent_prompt_template(self) -> str:
        answer_cfg = self.config.get("answer", {})
        prompt_key = answer_cfg.get(
            "openclaw_agent_prompt_key", OPENCLAW_DEFAULT_QA_PROMPT_KEY
        )
        if prompt_key is None:
            return ""
        prompt_key = str(prompt_key).strip()
        if prompt_key.lower() in OPENCLAW_EMPTY_QA_PROMPT_KEYS:
            return ""
        raise KeyError(
            "OpenClaw agent answer prompts are no longer loaded from prompts.yaml; "
            "put answer rules in AGENTS.md and set answer.openclaw_agent_prompt_key "
            "to answer_prompt_empty so --message receives the raw question."
        )

    @staticmethod
    def _clean_answer(answer: str) -> str:
        answer = str(answer or "").strip()
        answer = re.sub(r"^Answer:\s*", "", answer, flags=re.IGNORECASE).strip()
        return answer

    @staticmethod
    def _is_openclaw_silent_reply(answer: Any) -> bool:
        text = str(answer or "").strip()
        if not text:
            return False
        if text.upper() == "NO_REPLY":
            return True
        try:
            payload = json.loads(text)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        for key in ("action", "type", "reply", "text", "answer"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip().upper() == "NO_REPLY":
                return True
        return False

    def _run_openclaw(
        self,
        args: List[str],
        timeout: int = 120,
        runtime: Optional[OpenClawRuntime] = None,
    ) -> Dict[str, Any]:
        preload = (
            Path(__file__).parent
            / "assets"
            / "preload_openclaw_local_embeddings.mjs"
        ).resolve()
        return self._run_node(
            ["--import", str(preload), self.openclaw_cli, *args],
            cwd=self.openclaw_root,
            timeout=timeout,
            runtime=runtime,
        )

    def _run_node(
        self,
        args: List[str],
        *,
        cwd: Path,
        timeout: int = 120,
        runtime: Optional[OpenClawRuntime] = None,
    ) -> Dict[str, Any]:
        runtime = runtime or self.default_runtime
        env = {
            **os.environ,
            "OPENCLAW_ROOT": str(self.openclaw_root),
            "OPENCLAW_STATE_DIR": str(runtime.state_dir),
            "OPENCLAW_WORKSPACE_DIR": str(runtime.workspace_dir),
            "OPENCLAW_CONFIG_PATH": str(runtime.config_path),
            "OPENCLAW_AGENT_DIR": str(runtime.agent_dir),
            "PI_CODING_AGENT_DIR": str(runtime.agent_dir),
        }
        env.setdefault("NPM_CONFIG_CACHE", str(runtime.state_dir / "npm-cache"))
        bundled_plugins_dir = self.openclaw_root / "dist" / "extensions"
        if bundled_plugins_dir.exists() and bundled_plugins_dir.is_dir():
            env.setdefault(
                "OPENCLAW_BUNDLED_PLUGINS_DIR",
                str(bundled_plugins_dir.resolve(strict=False)),
            )
        qa_api_key = self._qa_api_key()
        qa_base_url = self._qa_base_url()
        if qa_api_key:
            env["OPENAI_API_KEY"] = qa_api_key
        if qa_base_url:
            env["OPENAI_BASE_URL"] = qa_base_url
        hf_endpoint = self._first_non_empty(
            os.environ.get("OPENCLAW_MEMORY_SEARCH_HF_ENDPOINT"),
            self.config.get("openclaw", {}).get("memory_search_hf_endpoint"),
            os.environ.get("HF_ENDPOINT"),
            os.environ.get("MODEL_ENDPOINT"),
            "https://hf-mirror.com",
        )
        if hf_endpoint:
            env["HF_ENDPOINT"] = str(hf_endpoint)
        cmd = ["node", *args]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "cmd": cmd,
                "returncode": 124,
                "stdout": exc.stdout or "",
                "stderr": (
                    f"Command timed out after {timeout}s"
                    + (f"\n{exc.stderr}" if exc.stderr else "")
                ),
                "timed_out": True,
            }
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    @staticmethod
    def _parse_json_output(result: Dict[str, Any]) -> Any:
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        for text in (stdout, stderr):
            parsed = OpenClawSessionMemoryAdapter._last_json_from_text(text)
            if parsed is not None:
                return parsed
        raw = "\n".join(part for part in (stdout, stderr) if part).strip()
        if not raw:
            return {"raw": "", "returncode": result.get("returncode")}
        return {"raw": raw, "returncode": result.get("returncode")}

    @staticmethod
    def _last_json_from_text(text: str) -> Any:
        text = str(text or "")
        if not text.strip():
            return None
        decoder = json.JSONDecoder()
        parsed_values: List[Any] = []
        for match in re.finditer(r"[\{\[]", text):
            candidate = text[match.start() :].strip()
            try:
                parsed, end = decoder.raw_decode(candidate)
                tail = candidate[end:].strip()
                if tail.startswith(("}", "]", ",")):
                    continue
                parsed_values.append(parsed)
            except Exception:
                continue
        if not parsed_values:
            return None
        for parsed in reversed(parsed_values):
            if isinstance(parsed, dict) and "ok" in parsed:
                return parsed
        return parsed_values[-1]

    @staticmethod
    def _summarize_openclaw_status(payload: Any) -> Dict[str, Any]:
        records = payload if isinstance(payload, list) else [payload]
        files = 0
        chunks = 0
        dirty = False
        db_paths: List[str] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            status = record.get("status")
            if not isinstance(status, dict):
                status = record
            files += int(status.get("files") or 0)
            chunks += int(status.get("chunks") or 0)
            dirty = dirty or bool(status.get("dirty"))
            if status.get("dbPath"):
                db_paths.append(str(status["dbPath"]))
        return {"files": files, "chunks": chunks, "dirty": dirty, "db_paths": db_paths}

    @staticmethod
    def _extract_openclaw_response_text(payload: Any) -> str:
        if isinstance(payload, list):
            for item in reversed(payload):
                text = OpenClawSessionMemoryAdapter._extract_openclaw_response_text(
                    item
                )
                if text:
                    return text
            return ""

        if not isinstance(payload, dict):
            return ""

        payload_text = OpenClawSessionMemoryAdapter._extract_payload_text(payload)
        if payload_text:
            return payload_text

        candidates = [
            payload.get("response"),
            payload.get("answer"),
            payload.get("text"),
            payload.get("summary"),
        ]
        result = payload.get("result")
        if isinstance(result, dict):
            candidates.extend(
                [
                    result.get("response"),
                    result.get("answer"),
                    result.get("text"),
                    result.get("summary"),
                ]
            )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if isinstance(candidate, dict):
                nested_text = OpenClawSessionMemoryAdapter._extract_payload_text(
                    candidate
                )
                if nested_text:
                    return nested_text
                output_text = candidate.get("output_text")
                if isinstance(output_text, str) and output_text.strip():
                    return output_text.strip()
        return ""

    @staticmethod
    def _extract_openclaw_agent_error(payload: Any) -> str:
        errors: List[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                stop_reason = value.get("stopReason")
                error_message = value.get("errorMessage")
                if stop_reason == "error" or error_message:
                    errors.append(str(error_message or stop_reason))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        return "; ".join(error for error in errors if error).strip()

    def _extract_session_error(
        self, session_id: str, runtime: Optional[OpenClawRuntime] = None
    ) -> str:
        runtime = runtime or self.default_runtime
        transcript_path = runtime.sessions_dir / f"{session_id}.jsonl"
        if not transcript_path.exists():
            return ""
        errors = []
        for raw in transcript_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(raw)
            except Exception:
                continue
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            stop_reason = message.get("stopReason")
            error_message = message.get("errorMessage")
            if stop_reason == "error" or error_message:
                errors.append(str(error_message or stop_reason))
        return "; ".join(error for error in errors if error).strip()

    @staticmethod
    def _extract_payload_text(payload: Any) -> str:
        if isinstance(payload, list):
            texts = [
                OpenClawSessionMemoryAdapter._extract_payload_text(item)
                for item in payload
            ]
            return "\n".join(text for text in texts if text).strip()

        if not isinstance(payload, dict):
            return ""

        result_payload = payload.get("result")
        payloads = []
        if isinstance(result_payload, dict) and isinstance(
            result_payload.get("payloads"), list
        ):
            payloads = result_payload["payloads"]
        elif isinstance(payload.get("payloads"), list):
            payloads = payload["payloads"]
        texts = []
        for item in payloads:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text = item["text"].strip()
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

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
        del recall_query
        transcript_path = runtime.sessions_dir / f"{session_id}.jsonl"
        transcript_rows = self._read_transcript_rows(transcript_path)
        retrieved_items = self._extract_runtime_retrieved_items(
            agent_payload, transcript_rows
        )
        injected_context = self._extract_runtime_injected_context(
            agent_payload, transcript_rows, retrieved_items
        )
        readback_payload = self._readback_runtime_payload(question_id)
        recall_status = self._openclaw_runtime_recall_status(
            retrieved_items, injected_context
        )
        if readback_payload and not retrieved_items:
            retrieved_items = self._runtime_items_from_readback_payload(
                readback_payload
            )
        if readback_payload and not injected_context:
            injected_context = str(readback_payload.get("formatted_context") or "")
        if readback_payload and recall_status == "fallback_no_memory_retrieved":
            recall_status = "fallback_readback_search_cache"
        payload = {
            "runtime_answer_session_id": session_id,
            "runtime_prompt_messages": [{"role": "user", "content": prompt}],
            "runtime_injected_context": injected_context,
            "runtime_retrieved_items": retrieved_items,
            "runtime_logger_status": recall_status,
        }
        if question_id:
            self._runtime_recall_payloads[question_id] = payload
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

    def _readback_runtime_payload(self, question_id: str) -> Dict[str, Any]:
        if not self._should_use_readback_answer_payload():
            return {}
        key = str(question_id or "").strip()
        if not key:
            return {}
        payload = self._readback_answer_cache.get(key)
        return dict(payload or {})

    @staticmethod
    def _runtime_items_from_readback_payload(
        readback_payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for item in readback_payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            metadata = (
                dict(item.get("metadata") or {})
                if isinstance(item.get("metadata"), dict)
                else {}
            )
            metadata.setdefault("source", "openclaw_readback_search_cache")
            items.append(
                {
                    "content": content,
                    "score": item.get("score"),
                    "metadata": metadata,
                }
            )
        return items

    @staticmethod
    def _openclaw_runtime_recall_status(
        retrieved_items: List[Dict[str, Any]], injected_context: str = ""
    ) -> str:
        return (
            "fallback_agent_json_and_transcript"
            if retrieved_items or injected_context
            else "fallback_no_memory_retrieved"
        )

    @staticmethod
    def _openclaw_runtime_logger_status(
        retrieved_items: List[Dict[str, Any]], injected_context: str = ""
    ) -> str:
        return OpenClawSessionMemoryAdapter._openclaw_runtime_recall_status(
            retrieved_items, injected_context
        )

    @staticmethod
    def _read_transcript_rows(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except Exception:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    @classmethod
    def _extract_runtime_retrieved_items(
        cls, agent_payload: Any, transcript_rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        cls._collect_runtime_items(agent_payload, candidates)
        for row in transcript_rows:
            cls._collect_runtime_items(row, candidates)
        return cls._dedupe_runtime_items(candidates)

    @classmethod
    def _collect_runtime_items(
        cls, value: Any, candidates: List[Dict[str, Any]]
    ) -> None:
        if isinstance(value, dict):
            if cls._looks_like_memory_item(value):
                candidates.append(cls._normalize_runtime_item(value))

            for key in (
                "memory_search_results",
                "search_results",
                "retrieved_items",
                "results",
                "items",
                "memories",
            ):
                nested = value.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        cls._collect_runtime_items(item, candidates)

            for key in (
                "payload",
                "payloads",
                "rawAgentResult",
                "agentMeta",
                "result",
                "message",
                "details",
                "data",
                "response",
            ):
                nested = value.get(key)
                if nested is not None:
                    cls._collect_runtime_items(nested, candidates)

            content = value.get("content")
            if isinstance(content, list):
                for item in content:
                    cls._collect_runtime_items(item, candidates)
            elif isinstance(content, str):
                cls._collect_json_text_runtime_items(content, candidates)

            text = value.get("text")
            if isinstance(text, str):
                cls._collect_json_text_runtime_items(text, candidates)
            return

        if isinstance(value, list):
            for item in value:
                cls._collect_runtime_items(item, candidates)
            return

        if isinstance(value, str):
            cls._collect_json_text_runtime_items(value, candidates)

    @classmethod
    def _extract_runtime_injected_context(
        cls,
        agent_payload: Any,
        transcript_rows: List[Dict[str, Any]],
        retrieved_items: List[Dict[str, Any]],
    ) -> str:
        text_candidates: List[str] = []
        cls._collect_runtime_context_texts(agent_payload, text_candidates)
        for row in transcript_rows:
            cls._collect_runtime_context_texts(row, text_candidates)

        seen = set()
        for text in text_candidates:
            context = cls._extract_memory_context_block(text)
            if not context or context in seen:
                continue
            seen.add(context)
            return context
        return cls._format_runtime_injected_context(retrieved_items)

    @classmethod
    def _collect_runtime_context_texts(
        cls, value: Any, candidates: List[str], key_hint: str = ""
    ) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str):
                    if cls._is_runtime_context_text(child, key_hint=str(key)):
                        candidates.append(child)
                else:
                    cls._collect_runtime_context_texts(child, candidates, str(key))
            return
        if isinstance(value, list):
            for item in value:
                cls._collect_runtime_context_texts(item, candidates, key_hint)
            return
        if isinstance(value, str) and cls._is_runtime_context_text(value, key_hint=key_hint):
            candidates.append(value)

    @staticmethod
    def _is_runtime_context_text(text: str, *, key_hint: str = "") -> bool:
        if not text or len(text.strip()) < 12:
            return False
        lowered = text.lower()
        key_lowered = key_hint.lower()
        if any(
            marker in lowered
            for marker in (
                "<recalled-memories",
                "<relevant-memories",
                "<memories",
                "<memory",
                "user\u200boriginal\u200bquery",
            )
        ):
            return True
        return key_lowered in {
            "prependcontext",
            "appendcontext",
            "prependsystemcontext",
            "appendsystemcontext",
            "systempromptaddition",
            "formatted_context",
            "formattedcontext",
        }

    @classmethod
    def _extract_memory_context_block(cls, text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return ""

        for tag in ("recalled-memories", "relevant-memories", "memories", "memory"):
            pattern = re.compile(
                rf"<{tag}\b[^>]*>.*?</{tag}>",
                flags=re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(stripped)
            if match:
                return match.group(0).strip()

        marker_index = stripped.find("user\u200boriginal\u200bquery")
        if marker_index > 0:
            return stripped[:marker_index].strip()

        if len(stripped) > 120000:
            return stripped[:120000].rstrip()
        return stripped

    @classmethod
    def _collect_json_text_runtime_items(
        cls, text: str, candidates: List[Dict[str, Any]]
    ) -> None:
        stripped = text.strip()
        if not stripped or stripped[0] not in "[{":
            return
        try:
            parsed = json.loads(stripped)
        except Exception:
            return
        cls._collect_runtime_items(parsed, candidates)

    @staticmethod
    def _looks_like_memory_item(value: Dict[str, Any]) -> bool:
        content = value.get("content") or value.get("text") or value.get("memory")
        if not isinstance(content, str) or not content.strip():
            return False
        keys = set(value)
        return bool(
            keys
            & {
                "score",
                "path",
                "metadata",
                "memory_id",
                "source_session_id",
                "source",
                "id",
            }
        )

    @staticmethod
    def _normalize_runtime_item(value: Dict[str, Any]) -> Dict[str, Any]:
        content = str(
            value.get("content") or value.get("text") or value.get("memory") or ""
        ).strip()
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        normalized = {
            "content": content,
            "score": value.get("score"),
            "metadata": {
                **metadata,
                **{
                    key: value[key]
                    for key in (
                        "path",
                        "source",
                        "memory_id",
                        "id",
                        "source_session_id",
                    )
                    if key in value and key not in metadata
                },
            },
        }
        return normalized

    @staticmethod
    def _dedupe_runtime_items(
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for item in items:
            content = str(item.get("content") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            identity = (
                metadata.get("memory_id")
                or metadata.get("id")
                or metadata.get("path")
                or content
            )
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(item)
        return deduped

    @staticmethod
    def _format_runtime_injected_context(items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""
        lines = ["## Runtime Retrieved Memory"]
        for index, item in enumerate(items, start=1):
            content = str(item.get("content") or "").strip()
            if content:
                lines.append(f"{index}. {content}")
        return "\n".join(lines).strip()

    def _load_imported_memories(self) -> None:
        memories: List[ImportedMemory] = []
        paths = sorted(self.conversation_runtime_root.glob("*/workspace/memory/*.md"))
        if not paths:
            paths = sorted(self.memory_dir.glob("*.md"))
        for path in paths:
            content = path.read_text(encoding="utf-8")
            metadata = self._parse_memory_metadata(content)
            path_metadata = self._parse_stable_memory_path(path)
            conversation_id = str(
                metadata.get("conversation_id")
                or path_metadata.get("conversation_id")
                or self._conversation_id_from_memory_path(path)
                or ""
            )
            runtime = self._runtime_for(conversation_id or "unknown")
            memory = ImportedMemory(
                conversation_id=conversation_id,
                session_key=str(
                    metadata.get("session_key")
                    or path_metadata.get("session_key")
                    or path.stem
                ),
                source_session_id=str(
                    metadata.get("source_session_id")
                    or path_metadata.get("source_session_id")
                    or path.stem
                ),
                source_case_id=str(metadata.get("source_case_id") or ""),
                session_order=metadata.get("session_order"),
                session_source=str(metadata.get("session_source") or ""),
                session_timestamp=str(metadata.get("session_timestamp") or ""),
                transcript_path=str(runtime.sessions_dir / f"{path.stem}.jsonl"),
                memory_path=str(path),
                content=content,
                source_unit_ids=[],
                message_count=0,
                metadata={
                    **path_metadata,
                    **metadata,
                    "path": str(path),
                    "absolute_path": str(path),
                    "openclaw_isolation": "conversation",
                    "openclaw_runtime": {
                        "state_dir": str(runtime.state_dir),
                        "workspace_dir": str(runtime.workspace_dir),
                        "config_path": str(runtime.config_path),
                        "memory_dir": str(runtime.memory_dir),
                    },
                },
            )
            memories.append(memory)
        self._memories = memories
        self._memories_by_conversation = {}
        for memory in memories:
            self._memories_by_conversation.setdefault(
                memory.conversation_id, []
            ).append(memory)

    def _conversation_id_from_memory_path(self, path: Path) -> str:
        try:
            relative = path.resolve(strict=False).relative_to(
                self.conversation_runtime_root.resolve(strict=False)
            )
        except ValueError:
            return ""
        return relative.parts[0] if relative.parts else ""

    @staticmethod
    def _parse_memory_metadata(content: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for line in content.splitlines():
            match = re.match(r"^-\s+([A-Za-z0-9_]+):\s*(.*)$", line)
            if match:
                metadata[match.group(1)] = match.group(2).strip()
            rich_match = re.match(r"^-\s+\*\*([^*]+)\*\*:\s*(.*)$", line)
            if rich_match:
                key = re.sub(
                    r"[^a-z0-9]+", "_", rich_match.group(1).strip().lower()
                ).strip("_")
                if key:
                    metadata[key] = rich_match.group(2).strip()
        return metadata

    @staticmethod
    def _parse_stable_memory_path(path: Path) -> Dict[str, str]:
        stem = path.stem
        if "--" not in stem:
            return {}
        conversation_id, source_session_id = stem.split("--", 1)
        if not conversation_id or not source_session_id:
            return {}
        return {
            "conversation_id": conversation_id,
            "session_key": source_session_id,
            "source_session_id": source_session_id,
        }

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_json_private(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o600)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
        return rows
