from __future__ import annotations

import hashlib
import re
from collections import OrderedDict, defaultdict
from typing import Any, Iterable

from prompts import build_categorize_preference_topic_prompt, build_topic_merge_prompt
from core.schemas import PreferenceItem, SourceLocation, SourcePersonaRecord, TopicPreferenceGroup
from core.validators import parse_json_after_output


SOURCE_PREFERENCE_FIELDS = {
    "stereotypical_preferences": "stereotypical_pref",
    "anti_stereotypical_preferences": "anti_stereotypical_pref",
    "neutral_preferences": "neutral_preferences",
}
SOURCE_PREF_TYPES = set(SOURCE_PREFERENCE_FIELDS.values())


def build_topic_groups(records: Iterable[SourcePersonaRecord]) -> list[TopicPreferenceGroup]:
    groups: list[TopicPreferenceGroup] = []
    for record in records:
        groups.extend(build_topic_groups_for_record(record))
    return groups


def categorize_topic_groups(
    records: Iterable[SourcePersonaRecord],
    llm_client: Any,
) -> list[TopicPreferenceGroup]:
    groups: list[TopicPreferenceGroup] = []
    for record in records:
        groups.extend(categorize_topic_groups_for_record(record, llm_client))
    return groups


def categorize_topic_groups_for_record(
    record: SourcePersonaRecord,
    llm_client: Any,
) -> list[TopicPreferenceGroup]:
    by_topic: dict[str, OrderedDict[str, PreferenceItem]] = defaultdict(OrderedDict)
    source_counts: dict[str, int] = defaultdict(int)
    existing_topics: list[str] = []
    for raw_item in iter_profile_preferences(record):
        preference_text = normalize_preference_text(raw_item["preference"])
        if not preference_text:
            continue
        topic = categorize_preference_topic(preference_text, existing_topics, llm_client)
        if topic not in existing_topics:
            existing_topics.append(topic)
        source_counts[topic] += 1
        dedupe_key = preference_text.lower()
        location = SourceLocation(
            source_file=record.source_file,
            conversation_scenario=raw_item.get("conversation_scenario"),
            item_index=raw_item.get("item_index"),
        )
        if dedupe_key in by_topic[topic]:
            by_topic[topic][dedupe_key].source_location.duplicate_locations.append(location.model_dump(mode="json"))
            continue
        preference_id = make_preference_id(record.persona_id, topic, preference_text)
        by_topic[topic][dedupe_key] = PreferenceItem(
            preference_id=preference_id,
            persona_id=record.persona_id,
            topic_preference=topic,
            preference_text=preference_text,
            pref_type=raw_item.get("pref_type"),
            source_location=location,
        )
    return [
        TopicPreferenceGroup(
            topic_id=make_topic_id(record.persona_id, topic),
            persona_id=record.persona_id,
            topic_preference=topic,
            preferences=list(preferences.values()),
            source_count=source_counts[topic],
        )
        for topic, preferences in sorted(by_topic.items())
        if preferences
    ]


def categorize_preference_topic(preference: str, existing_topics: list[str], llm_client: Any) -> str:
    prompt = build_categorize_preference_topic_prompt(preference, existing_topics)
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="topic_categorization",
        temperature=0.0,
    )
    if not result.stream_completed:
        raise RuntimeError(f"topic categorization stream failed: {result.error}")
    topic = extract_topic_after_output(result.text)
    cleaned = clean_topic(topic)
    for existing in existing_topics:
        if cleaned.lower() == existing.lower():
            return existing
    return cleaned


def build_topic_groups_for_record(record: SourcePersonaRecord) -> list[TopicPreferenceGroup]:
    by_topic: dict[str, OrderedDict[str, PreferenceItem]] = defaultdict(OrderedDict)
    source_counts: dict[str, int] = defaultdict(int)
    for raw_item in iter_preference_sources(record):
        topic = normalize_topic(raw_item.get("topic_preference") or raw_item.get("topic_query") or "Uncategorized")
        preference_text = normalize_preference_text(raw_item.get("preference") or raw_item.get("sensitive_info") or "")
        if not preference_text:
            continue
        source_counts[topic] += 1
        dedupe_key = preference_text.lower()
        location = SourceLocation(
            source_file=record.source_file,
            conversation_scenario=raw_item.get("conversation_scenario"),
            item_index=raw_item.get("item_index"),
            duplicate_locations=[raw_item["topic_marker"]] if raw_item.get("topic_marker") else [],
        )
        if dedupe_key in by_topic[topic]:
            existing = by_topic[topic][dedupe_key]
            existing.source_location.duplicate_locations.append(location.model_dump(mode="json"))
            continue
        preference_id = make_preference_id(record.persona_id, topic, preference_text)
        by_topic[topic][dedupe_key] = PreferenceItem(
            preference_id=preference_id,
            persona_id=record.persona_id,
            topic_preference=topic,
            preference_text=preference_text,
            pref_type=raw_item.get("pref_type"),
            source_location=location,
        )

    return [
        TopicPreferenceGroup(
            topic_id=make_topic_id(record.persona_id, topic),
            persona_id=record.persona_id,
            topic_preference=topic,
            preferences=list(preferences.values()),
            source_count=source_counts[topic],
        )
        for topic, preferences in sorted(by_topic.items())
        if preferences
    ]


def merge_similar_topic_groups(
    groups: list[TopicPreferenceGroup],
    personas: dict[str, Any],
    llm_client: Any,
) -> list[TopicPreferenceGroup]:
    merged: list[TopicPreferenceGroup] = []
    by_persona: dict[str, list[TopicPreferenceGroup]] = defaultdict(list)
    for group in groups:
        by_persona[group.persona_id].append(group)

    for persona_id, persona_groups in by_persona.items():
        persona = personas[persona_id]
        if len(persona_groups) <= 1:
            merged.extend(persona_groups)
            continue
        prompt = build_topic_merge_prompt(persona.persona_str, topic_group_summaries(persona_groups))
        result = llm_client.stream_chat(
            [{"role": "user", "content": prompt}],
            phase="topic_merge",
            temperature=0.0,
        )
        if not result.stream_completed:
            merged.extend(merge_topic_groups_with_mapping(persona_groups, {}))
            continue
        payload = parse_json_after_output(result.text)
        mapping = parse_topic_mapping(payload)
        merged.extend(merge_topic_groups_with_mapping(persona_groups, mapping))
    return sorted(merged, key=lambda group: (group.persona_id, group.topic_preference.lower()))


def topic_group_summaries(groups: list[TopicPreferenceGroup]) -> list[dict[str, Any]]:
    return [
        {
            "topic_preference": group.topic_preference,
            "source_count": group.source_count,
            "preference_count": len(group.preferences),
            "example_preferences": [item.preference_text for item in group.preferences[:3]],
        }
        for group in groups
    ]


def parse_topic_mapping(payload: dict[str, Any]) -> dict[str, str]:
    raw_mapping = payload.get("topic_mapping", payload.get("mapping", {}))
    mapping: dict[str, str] = {}
    if isinstance(raw_mapping, dict):
        for source_topic, canonical_topic in raw_mapping.items():
            mapping[normalize_topic(source_topic)] = normalize_topic(canonical_topic)
    elif isinstance(raw_mapping, list):
        for item in raw_mapping:
            if not isinstance(item, dict):
                continue
            source = item.get("source_topic")
            canonical = item.get("canonical_topic")
            if source and canonical:
                mapping[normalize_topic(source)] = normalize_topic(canonical)
    return mapping


def merge_topic_groups_with_mapping(
    groups: list[TopicPreferenceGroup],
    mapping: dict[str, str],
) -> list[TopicPreferenceGroup]:
    by_topic: dict[str, OrderedDict[str, PreferenceItem]] = defaultdict(OrderedDict)
    source_counts: dict[str, int] = defaultdict(int)
    persona_id = groups[0].persona_id if groups else ""

    for group in groups:
        canonical = normalize_topic(mapping.get(group.topic_preference, group.topic_preference))
        canonical = canonical_topic_alias(canonical)
        source_counts[canonical] += group.source_count
        for preference in group.preferences:
            dedupe_key = preference.preference_text.lower()
            updated = preference.model_copy(update={"topic_preference": canonical})
            if dedupe_key in by_topic[canonical]:
                existing = by_topic[canonical][dedupe_key]
                existing.source_location.duplicate_locations.append(preference.source_location.model_dump(mode="json"))
                continue
            by_topic[canonical][dedupe_key] = updated

    return [
        TopicPreferenceGroup(
            topic_id=make_topic_id(persona_id, topic),
            persona_id=persona_id,
            topic_preference=topic,
            preferences=list(preferences.values()),
            source_count=source_counts[topic],
        )
        for topic, preferences in sorted(by_topic.items())
        if preferences
    ]


def iter_preference_sources(record: SourcePersonaRecord) -> Iterable[dict[str, Any]]:
    topic_index = build_conversation_topic_index(record)
    for key, pref_type in SOURCE_PREFERENCE_FIELDS.items():
        values = record.persona_fields.get(key, [])
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            preference_text = normalize_preference_text(str(item))
            if not preference_text:
                continue
            marker = topic_index.get(preference_text.lower())
            if not marker:
                continue
            yield {
                "preference": preference_text,
                "pref_type": pref_type,
                "topic_preference": marker["topic_preference"],
                "conversation_scenario": f"persona_profile.{key}",
                "item_index": index,
                "topic_marker": marker,
            }


def iter_profile_preferences(record: SourcePersonaRecord) -> Iterable[dict[str, Any]]:
    for key, pref_type in SOURCE_PREFERENCE_FIELDS.items():
        values = record.persona_fields.get(key, [])
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            preference_text = normalize_preference_text(str(item))
            if not preference_text:
                continue
            yield {
                "preference": preference_text,
                "pref_type": pref_type,
                "conversation_scenario": f"persona_profile.{key}",
                "item_index": index,
            }


def build_conversation_topic_index(record: SourcePersonaRecord) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    conversations = record.source_conversations
    for marker in iter_conversation_topic_markers(conversations):
        preference_text = normalize_preference_text(str(marker.get("preference", "")))
        if not preference_text:
            continue
        index.setdefault(preference_text.lower(), marker)
    return index


def iter_conversation_topic_markers(conversations: Any) -> Iterable[dict[str, Any]]:
    if isinstance(conversations, dict):
        for scenario, items in conversations.items():
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                marker = normalize_conversation_topic_marker(item, scenario, index)
                if marker:
                    yield marker
    elif isinstance(conversations, list):
        for index, item in enumerate(conversations):
            marker = normalize_conversation_topic_marker(item, "conversations", index)
            if marker:
                yield marker


def normalize_conversation_topic_marker(item: Any, scenario: str, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    pref_type = item.get("pref_type")
    if pref_type not in SOURCE_PREF_TYPES:
        return None
    if item.get("updated") is True:
        return None
    preference_text = normalize_preference_text(str(item.get("preference", "")))
    topic = normalize_topic(item.get("topic_preference") or item.get("topic_query") or "")
    if not preference_text or not topic:
        return None
    return {
        "preference": preference_text,
        "pref_type": pref_type,
        "topic_preference": topic,
        "conversation_scenario": scenario,
        "item_index": index,
    }


def pref_type_from_profile_key(key: str) -> str:
    mapping = {
        "stereotypical_preferences": "stereotypical_pref",
        "anti_stereotypical_preferences": "anti_stereotypical_pref",
    }
    return mapping.get(key, key)


def normalize_topic(value: str) -> str:
    topic = re.sub(r"\s+", " ", str(value).replace(":", " ")).strip()
    return topic or "Uncategorized"


def extract_topic_after_output(text: str) -> str:
    if "###Output" in text:
        return text.split("###Output", 1)[1].strip()
    if "### Output" in text:
        return text.split("### Output", 1)[1].strip()
    return text.strip()


def clean_topic(value: str) -> str:
    topic = value.strip()
    topic = re.sub(r"^```(?:text)?", "", topic).strip()
    topic = re.sub(r"```$", "", topic).strip()
    topic = topic.splitlines()[0].strip() if topic else ""
    topic = topic.strip("\"'` .,:;")
    topic = normalize_topic(topic)
    if topic.lower() in {"uncategorized", "unknown", "undefined", "none", "n/a"}:
        raise ValueError("topic categorization returned a fuzzy topic")
    return canonical_topic_alias(topic)


def canonical_topic_alias(value: str) -> str:
    aliases = {
        "cook": "Cooking",
        "cooks": "Cooking",
        "cuisine": "Food",
        "meal": "Food",
        "meals": "Food",
        "movie": "Film",
        "movies": "Film",
        "cinema": "Film",
        "films": "Film",
        "book": "Books",
        "reading": "Books",
        "relationship": "Relationships",
        "romance": "Relationships",
        "fitness": "Fitness",
        "exercise": "Fitness",
        "workout": "Fitness",
        "sports": "Sports",
    }
    key = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return aliases.get(key, value)


def normalize_preference_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def make_topic_id(persona_id: str, topic: str) -> str:
    digest = hashlib.sha1(f"{persona_id}|{topic.lower()}".encode("utf-8")).hexdigest()[:12]
    return f"topic-{digest}"


def make_preference_id(persona_id: str, topic: str, preference_text: str) -> str:
    digest = hashlib.sha1(f"{persona_id}|{topic.lower()}|{preference_text.lower()}".encode("utf-8")).hexdigest()[:12]
    return f"pref-{digest}"
