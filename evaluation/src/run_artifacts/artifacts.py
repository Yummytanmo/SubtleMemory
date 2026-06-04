"""Evaluation run artifact builders."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from evaluation.src.core.data_models import Dataset
from evaluation.src.run_artifacts.models import AuditError, SourceUnit


_REDACTED_SECRET = "[REDACTED]"
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "secret",
    "token",
    "password",
)


def _redact_snapshot_secrets(value: Any) -> Any:
    """Redact resolved secrets before writing long-lived run config snapshots."""
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key).lower().replace("-", "_")
            if any(fragment in key_text for fragment in _SECRET_KEY_FRAGMENTS):
                redacted[key] = _REDACTED_SECRET if child else child
            else:
                redacted[key] = _redact_snapshot_secrets(child)
        return redacted
    if isinstance(value, list):
        return [_redact_snapshot_secrets(item) for item in value]
    return value


def _stringify_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def build_source_unit_id(conversation_id: str, message: Any, index: int) -> str:
    """Build a stable source-unit id for a message."""
    metadata = getattr(message, "metadata", {}) or {}
    dia_id = metadata.get("dia_id")
    if dia_id:
        return str(dia_id)

    session = metadata.get("session")
    if session:
        return f"{conversation_id}:{session}:{index}"
    return f"{conversation_id}:msg:{index}"


def build_source_units(dataset: Dataset) -> Tuple[List[SourceUnit], Dict[str, SourceUnit]]:
    """Build source units and persist source ids back onto message metadata."""
    source_units: List[SourceUnit] = []
    source_unit_map: Dict[str, SourceUnit] = {}

    for conversation in dataset.conversations:
        for idx, message in enumerate(conversation.messages):
            source_unit_id = build_source_unit_id(conversation.conversation_id, message, idx)
            message.metadata["source_unit_id"] = source_unit_id
            message.metadata["message_index"] = idx

            source_unit = SourceUnit(
                dataset_id=dataset.dataset_name,
                sample_id=conversation.conversation_id,
                conversation_id=conversation.conversation_id,
                source_unit_id=source_unit_id,
                speaker=message.sender_name,
                view=message.sender_name,
                timestamp=_stringify_timestamp(message.timestamp),
                raw_text=message.content,
                metadata={
                    **message.metadata,
                    "sender_id": message.sender_id,
                    "sender_name": message.sender_name,
                },
            )
            source_units.append(source_unit)
            source_unit_map[source_unit_id] = source_unit

    return source_units, source_unit_map


def resolve_qa_evidence(dataset: Dataset, source_unit_map: Dict[str, SourceUnit]) -> List[AuditError]:
    """Resolve QA evidence ids to canonical source-unit ids."""
    errors: List[AuditError] = []

    alias_map: Dict[str, str] = {}
    for source_unit in source_unit_map.values():
        alias_map[source_unit.source_unit_id] = source_unit.source_unit_id
        dia_id = source_unit.metadata.get("dia_id")
        if dia_id:
            alias_map[str(dia_id)] = source_unit.source_unit_id

    for qa in dataset.qa_pairs:
        resolved_ids: List[str] = []
        unresolved_ids: List[str] = []
        for raw_evidence in qa.evidence:
            canonical_id = alias_map.get(str(raw_evidence))
            if canonical_id:
                resolved_ids.append(canonical_id)
            else:
                unresolved_ids.append(str(raw_evidence))

        qa.evidence_source_unit_ids = resolved_ids
        if unresolved_ids:
            qa.metadata["unresolved_evidence"] = unresolved_ids
            errors.append(
                AuditError(
                    stage="dataset",
                    error_type="evidence_mapping_missing",
                    error_message=(
                        f"Could not map evidence ids for {qa.question_id}: {', '.join(unresolved_ids)}"
                    ),
                )
            )

    return errors


def build_run_config_snapshot(
    dataset: Dataset,
    adapter: Any,
    evaluator: Any,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a run config snapshot."""
    config = getattr(adapter, "config", {}) or {}
    search_cfg = config.get("search", {}) or {}
    answer_cfg = config.get("answer", {}) or {}
    readiness_cfg = config.get("readiness", {}) or {}

    return {
        "run_id": runtime.get("run_id"),
        "run_name": runtime.get("run_name"),
        "dataset_id": dataset.dataset_name,
        "system_id": runtime.get("system_id"),
        "benchmark_mode": runtime.get("benchmark_mode", "category1"),
        "requested_stages": runtime.get("requested_stages", []),
        "stage_order": runtime.get("stage_order", []),
        "requires_finalize_report": runtime.get("requires_finalize_report", False),
        "semantic_chunk_policy": config.get("semantic_chunk_policy", "native_system"),
        "transport_batch_policy": config.get("transport_batch_policy", "native_system"),
        "global_max_concurrency": runtime.get("global_max_concurrency", config.get("num_workers")),
        "stage_concurrency": {
            "add": config.get("num_workers"),
            "search": search_cfg.get("num_workers", getattr(adapter, "num_workers", None)),
            "answer": answer_cfg.get("num_workers", runtime.get("answer_num_workers")),
        },
        "qps_limit": config.get("requests_per_second"),
        "max_retries": config.get("max_retries"),
        "backoff_base_seconds": config.get("backoff_base_seconds", 1),
        "backoff_max_seconds": config.get("backoff_max_seconds", 8),
        "jitter": config.get("jitter", False),
        "timeout_seconds": config.get("timeout_seconds"),
        "readiness": {
            "budget_seconds": readiness_cfg.get(
                "budget_seconds", config.get("post_add_wait_seconds", 0)
            ),
            "poll_interval_seconds": readiness_cfg.get(
                "poll_interval_seconds",
                config.get("post_add_poll_interval_seconds"),
            ),
        },
        "dataset_metadata": dataset.metadata,
        "adapter_info": _redact_snapshot_secrets(adapter.get_system_info()),
        "evaluator": getattr(evaluator, "get_name", lambda: evaluator.__class__.__name__)(),
        "runtime": runtime,
    }


def source_units_to_rows(source_units: List[SourceUnit]) -> List[Dict[str, Any]]:
    """Convert source units to plain dict rows."""
    return [asdict(unit) for unit in source_units]
