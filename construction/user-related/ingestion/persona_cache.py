from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from core.schemas import SanitizedPersonaProfile, SourcePersonaRecord, TopicPreferenceGroup
from ingestion.source_loader import build_persona_field_hash, build_persona_fields, build_persona_str, sanitize_source_record
from ingestion.topic_builder import SOURCE_PREFERENCE_FIELDS, categorize_topic_groups_for_record


CACHE_SCHEMA = "user_related_persona_cache_v1"


def persona_cache_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "persona"


def persona_cache_path(cache_dir: str | Path, persona_id: str) -> Path:
    safe_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in persona_id)
    return Path(cache_dir) / f"persona_{safe_id}.json"


def is_persona_cache_dir(input_dir: str | Path) -> bool:
    root = Path(input_dir)
    if not root.is_dir():
        return False
    return any(is_persona_cache_file(path) for path in root.glob("*.json"))


def is_persona_cache_file(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return isinstance(payload, dict) and payload.get("schema") == CACHE_SCHEMA
    except Exception:
        return False


def load_persona_caches(
    input_dir: str | Path,
    limit: int | None = None,
    *,
    persona_ids: Sequence[str] | None = None,
) -> tuple[list[SanitizedPersonaProfile], list[TopicPreferenceGroup]]:
    if persona_ids is not None:
        return load_persona_caches_by_id(input_dir, persona_ids)

    profiles: list[SanitizedPersonaProfile] = []
    groups: list[TopicPreferenceGroup] = []
    files = sorted(path for path in Path(input_dir).glob("*.json") if is_persona_cache_file(path))
    if limit is not None:
        files = files[:limit]
    for path in files:
        profile, topic_groups = load_persona_cache(path)
        profiles.append(profile)
        groups.extend(topic_groups)
    return profiles, groups


def load_persona_caches_by_id(
    input_dir: str | Path,
    persona_ids: Sequence[str],
) -> tuple[list[SanitizedPersonaProfile], list[TopicPreferenceGroup]]:
    root = Path(input_dir)
    profiles: list[SanitizedPersonaProfile] = []
    groups: list[TopicPreferenceGroup] = []
    missing_ids: list[str] = []
    for persona_id in persona_ids:
        path = persona_cache_path(root, persona_id)
        if not path.exists() or not is_persona_cache_file(path):
            missing_ids.append(persona_id)
            continue
        profile, topic_groups = load_persona_cache(path)
        profiles.append(profile)
        groups.extend(topic_groups)
    if missing_ids:
        raise ValueError(f"Requested persona_ids not found in persona cache: {', '.join(missing_ids)}")
    return profiles, groups


def load_persona_cache(path: str | Path) -> tuple[SanitizedPersonaProfile, list[TopicPreferenceGroup]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"not a user-related persona cache: {path}")
    profile = SanitizedPersonaProfile.model_validate(
        normalize_cached_sanitized_persona_payload(payload["sanitized_persona"])
    )
    groups = [TopicPreferenceGroup.model_validate(item) for item in payload.get("topic_preferences", [])]
    return profile, groups


def normalize_cached_sanitized_persona_payload(payload: dict[str, Any]) -> dict[str, Any]:
    profile_fields = build_persona_fields(payload.get("profile", {}))
    persona_str = build_persona_str(profile_fields)
    field_hash = build_persona_field_hash(persona_str)
    return {
        **payload,
        "profile": profile_fields,
        "persona_str": persona_str,
        "field_hash": field_hash,
    }


def load_or_build_persona_artifacts(
    records: list[SourcePersonaRecord],
    cache_dir: str | Path,
    llm_client: Any,
) -> tuple[list[SanitizedPersonaProfile], list[TopicPreferenceGroup], dict[str, int]]:
    profiles: list[SanitizedPersonaProfile] = []
    groups: list[TopicPreferenceGroup] = []
    counts = {"cache_hits": 0, "cache_misses": 0}
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    for record in records:
        path = persona_cache_path(root, record.persona_id)
        if path.exists() and is_persona_cache_file(path):
            profile, topic_groups = load_persona_cache(path)
            counts["cache_hits"] += 1
        else:
            profile = sanitize_source_record(record)
            topic_groups = categorize_topic_groups_for_record(record, llm_client)
            save_persona_cache(path, record, profile, topic_groups)
            counts["cache_misses"] += 1
        profiles.append(profile)
        groups.extend(topic_groups)
    return profiles, groups, counts


def save_persona_cache(
    path: str | Path,
    record: SourcePersonaRecord,
    profile: SanitizedPersonaProfile,
    topic_groups: list[TopicPreferenceGroup],
) -> None:
    payload = {
        "schema": CACHE_SCHEMA,
        "persona_id": profile.persona_id,
        "source_file": record.source_file,
        "sanitized_persona": profile.model_dump(mode="json"),
        "source_preference_fields": {
            key: record.persona_fields.get(key, [])
            for key in SOURCE_PREFERENCE_FIELDS
        },
        "topic_preferences_overview": build_topic_preferences_overview(topic_groups),
        "topic_preferences": [group.model_dump(mode="json") for group in topic_groups],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_topic_preferences_overview(topic_groups: list[TopicPreferenceGroup]) -> dict[str, Any]:
    return {
        group.topic_preference: len(group.preferences)
        for group in sorted(topic_groups, key=lambda item: item.topic_preference.lower())
    }
