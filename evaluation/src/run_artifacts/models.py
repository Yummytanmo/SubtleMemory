"""Evaluation run artifact data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Optional


def dataclass_to_dict(value: Any) -> Any:
    """Recursively convert dataclasses to plain JSON-serializable objects."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    return value


@dataclass
class SourceUnit:
    """Canonical source unit aligned with dataset evidence."""

    dataset_id: str
    sample_id: str
    conversation_id: str
    source_unit_id: str
    speaker: str
    view: str
    timestamp: Optional[str]
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditError:
    """Structured stage error."""

    stage: str
    error_type: str
    error_message: str
    attempt: int = 1
    retryable: Optional[bool] = None
    status_code: Optional[int] = None
    provider: Optional[str] = None
    raw_payload_ref: Optional[str] = None


@dataclass
class WriteReceipt:
    """Write-time receipt captured directly from provider responses."""

    system_id: str
    namespace_scope: Dict[str, Any]
    chunk_id: str
    source_unit_ids: List[str]
    provider_receipt: Dict[str, Any] = field(default_factory=dict)
    provider_status: str = "unknown"
    memory_refs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ImportManifestRecord:
    """Mapping from source units to provider writes."""

    run_id: str
    system_id: str
    conversation_id: str
    view_id: str
    chunk_id: str
    source_unit_ids: List[str]
    write_request_summary: Dict[str, Any] = field(default_factory=dict)
    write_receipt: Dict[str, Any] = field(default_factory=dict)
    memory_refs: List[Dict[str, Any]] = field(default_factory=list)
    write_status: str = "unknown"
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalArtifact:
    """Structured retrieval artifact for a single QA."""

    question_id: str
    query: str
    retrieved_items: List[Dict[str, Any]] = field(default_factory=list)
    formatted_context: str = ""
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_status: str = "ok"
    timing_ms: float = 0.0


@dataclass
class QAResult:
    """Single-source-of-truth QA row for a run."""

    question_id: str
    question: str
    golden_answer: str
    predicted_answer: str
    conversation_id: str
    category: Optional[str] = None
    evidence_source_unit_ids: List[str] = field(default_factory=list)
    retrieval_artifact: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
