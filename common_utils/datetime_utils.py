"""Minimal datetime helpers for standalone evaluation."""

from __future__ import annotations

from datetime import datetime, timezone


def get_now_with_timezone() -> datetime:
    """Return a timezone-aware current timestamp."""

    return datetime.now().astimezone()


def to_iso_format(value: datetime) -> str:
    """Serialize a datetime to ISO 8601, normalizing naive values to UTC."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def from_iso_format(value: str, strict: bool = False) -> datetime | None:
    """Parse ISO 8601 strings used by the evaluation datasets.

    Returns ``None`` on parse failure unless ``strict=True``.
    """

    normalized = str(value or "").strip()
    if not normalized:
        if strict:
            raise ValueError("Cannot parse empty datetime string")
        return None

    normalized = normalized.replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        if strict:
            raise
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
