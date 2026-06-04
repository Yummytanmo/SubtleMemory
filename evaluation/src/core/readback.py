"""Provider readback data contracts used by readback search mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List


def to_plain_dict(value: Any) -> Any:
    """Recursively convert dataclasses into JSON-serializable containers."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_dict(item) for key, item in value.items()}
    return value


@dataclass
class NormalizedStorageObject:
    """Provider-neutral storage object returned by readback hooks."""

    session_id: str
    kind: str
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageReadbackResult:
    """Provider-neutral storage readback returned by adapter hooks."""

    status: str = "unsupported"
    checked_session_ids: List[str] = field(default_factory=list)
    missing_session_ids: List[str] = field(default_factory=list)
    objects: List[NormalizedStorageObject | Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_plain_dict(self)
