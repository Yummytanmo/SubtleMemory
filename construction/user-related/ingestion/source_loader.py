from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.schemas import SanitizedPersonaProfile, SourcePersonaRecord


PERSONA_FIELD_EXCLUDED_NAMES = {"conversations", "matched_images", "preference_updates"}
PERSONA_STR_EXCLUDED_FIELD_NAMES = {
    "stereotypical_preferences",
    "anti_stereotypical_preferences",
    "neutral_preferences",
}


def discover_raw_files(
    input_dir: str | Path,
    limit: int | None = None,
    *,
    exclude_dirs: Iterable[str | Path] | None = None,
) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {root}")
    excluded = [path.resolve() for path in exclude_dirs or [] if Path(path).exists()]
    files = sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in {".json", ".jsonl"} and not is_under_excluded_dir(path, excluded)
    )
    if limit is not None:
        files = files[:limit]
    return files


def load_source_personas(
    input_dir: str | Path,
    limit: int | None = None,
    *,
    exclude_dirs: Iterable[str | Path] | None = None,
    persona_ids: Sequence[str] | None = None,
) -> list[SourcePersonaRecord]:
    if persona_ids is not None:
        return load_source_personas_by_id(input_dir, persona_ids, exclude_dirs=exclude_dirs)

    records: list[SourcePersonaRecord] = []
    for path in discover_raw_files(input_dir, exclude_dirs=exclude_dirs):
        for fallback_index, payload in enumerate(_iter_payloads(path)):
            persona_id = extract_persona_id(path, payload, fallback_index)
            persona_fields = build_persona_fields(payload)
            records.append(
                SourcePersonaRecord(
                    persona_id=persona_id,
                    source_file=str(path.resolve()),
                    raw_payload=payload,
                    persona_fields=persona_fields,
                    source_conversations=payload.get("conversations", {}),
                )
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def load_source_personas_by_id(
    input_dir: str | Path,
    persona_ids: Sequence[str],
    *,
    exclude_dirs: Iterable[str | Path] | None = None,
) -> list[SourcePersonaRecord]:
    requested_ids = list(persona_ids)
    requested_set = set(requested_ids)
    selected_records: dict[str, SourcePersonaRecord] = {}

    for path in discover_raw_files(input_dir, exclude_dirs=exclude_dirs):
        for fallback_index, payload in enumerate(_iter_payloads(path)):
            persona_id = extract_persona_id(path, payload, fallback_index)
            if persona_id not in requested_set:
                continue
            if persona_id in selected_records:
                raise ValueError(f"duplicate persona_id discovered in source data: {persona_id}")
            selected_records[persona_id] = SourcePersonaRecord(
                persona_id=persona_id,
                source_file=str(path.resolve()),
                raw_payload=payload,
                persona_fields=build_persona_fields(payload),
                source_conversations=payload.get("conversations", {}),
            )
            if len(selected_records) == len(requested_ids):
                return [selected_records[persona_id] for persona_id in requested_ids]

    missing_ids = [persona_id for persona_id in requested_ids if persona_id not in selected_records]
    raise ValueError(f"Requested persona_ids not found in source data: {', '.join(missing_ids)}")


def is_under_excluded_dir(path: Path, excluded_dirs: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == excluded or resolved.is_relative_to(excluded) for excluded in excluded_dirs)


def build_persona_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in PERSONA_FIELD_EXCLUDED_NAMES
    }


def sanitize_source_record(record: SourcePersonaRecord) -> SanitizedPersonaProfile:
    persona_str = build_persona_str(record.persona_fields)
    digest = build_persona_field_hash(persona_str)
    return SanitizedPersonaProfile(
        persona_id=record.persona_id,
        source_file=record.source_file,
        profile=record.persona_fields,
        persona_str=persona_str,
        field_hash=digest,
    )


def sanitize_source_records(records: Iterable[SourcePersonaRecord]) -> list[SanitizedPersonaProfile]:
    return [sanitize_source_record(record) for record in records]


def build_persona_prompt_fields(persona_fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in persona_fields.items()
        if key not in PERSONA_STR_EXCLUDED_FIELD_NAMES
    }


def build_persona_str(persona_fields: dict[str, Any]) -> str:
    persona_prompt_fields = build_persona_prompt_fields(persona_fields)
    return json.dumps(persona_prompt_fields, ensure_ascii=False)


def build_persona_field_hash(persona_str: str) -> str:
    return hashlib.sha256(persona_str.encode("utf-8")).hexdigest()


def extract_persona_id(path: Path, payload: dict[str, Any], fallback_index: int = 0) -> str:
    for key in ("persona_id", "id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    match = re.search(r"persona(\d+)", path.stem)
    if match:
        return match.group(1)
    return f"{path.stem}-{fallback_index}"


def _iter_payloads(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    unwrapped = _unwrap_persona_payload(payload)
                    if unwrapped is not None:
                        yield unwrapped
        return

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                unwrapped = _unwrap_persona_payload(item)
                if unwrapped is not None:
                    yield unwrapped
        return
    if isinstance(data, dict):
        if _looks_like_persona_payload(data):
            yield data
            return
        for key, value in data.items():
            if isinstance(value, dict):
                payload = dict(value)
                payload.setdefault("persona_id", str(key))
                if _looks_like_persona_payload(payload):
                    yield payload


def _unwrap_persona_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if _looks_like_persona_payload(payload):
        return payload
    if len(payload) == 1:
        key, value = next(iter(payload.items()))
        if isinstance(value, dict):
            unwrapped = dict(value)
            unwrapped.setdefault("persona_id", str(key))
            if _looks_like_persona_payload(unwrapped):
                return unwrapped
    return None


def _looks_like_persona_payload(data: dict[str, Any]) -> bool:
    persona_markers = {
        "short_persona",
        "conversations",
        "stereotypical_preferences",
        "anti_stereotypical_preferences",
        "neutral_preferences",
        "therapy_background",
        "health_and_medical_conditions",
    }
    return any(key in data for key in persona_markers)
