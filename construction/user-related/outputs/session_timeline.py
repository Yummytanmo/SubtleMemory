from __future__ import annotations

import random
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone

from core.schemas import ConversationSession, TimestampedSession


SESSION_START_DATE = date(2025, 4, 1)
SESSION_END_DATE = date(2026, 4, 1)
SESSION_TIMEZONE = timezone(timedelta(hours=8))
SESSION_START_HOUR = 8
SESSION_END_HOUR = 22


def build_timestamped_sessions(
    conversations: list[ConversationSession],
    *,
    seed: int,
) -> list[TimestampedSession]:
    ordered = sorted(conversations, key=session_sort_key)
    timestamps = assign_session_timestamps(len(ordered), seed=seed)
    return [
        build_timestamped_session(session, order=index, timestamp=timestamps[index])
        for index, session in enumerate(ordered)
    ]


def conversation_metric_totals(sessions: list[ConversationSession] | list[TimestampedSession]) -> dict[str, int]:
    return {
        "conversation_total_turns": sum(int(session.turn_count or 0) for session in sessions),
        "conversation_total_tokens": sum(int(session.token_count or 0) for session in sessions),
    }


def session_sort_key(session: ConversationSession) -> tuple[str, str, str, str, str]:
    return (
        session.persona_id,
        session.topic_preference or "",
        session.case_id,
        session.fact_id,
        session.conversation_id,
    )


def assign_session_timestamps(count: int, *, seed: int) -> list[str]:
    if count < 1:
        return []
    rng = random.Random(seed)
    dates = evenly_spaced_dates(count)
    index_by_date: dict[date, list[int]] = defaultdict(list)
    for index, item_date in enumerate(dates):
        index_by_date[item_date].append(index)

    timestamps = [""] * count
    for item_date, indices in index_by_date.items():
        seconds = sorted(random_reasonable_seconds(rng) for _ in indices)
        for index, second in zip(indices, seconds):
            timestamps[index] = make_timestamp(item_date, second)
    return timestamps


def evenly_spaced_dates(count: int) -> list[date]:
    if count < 1:
        return []
    if count == 1:
        return [SESSION_START_DATE]
    span_days = (SESSION_END_DATE - SESSION_START_DATE).days
    return [
        SESSION_START_DATE + timedelta(days=round(index * span_days / (count - 1)))
        for index in range(count)
    ]


def random_reasonable_seconds(rng: random.Random) -> int:
    start = SESSION_START_HOUR * 60 * 60
    end = (SESSION_END_HOUR * 60 * 60) + (59 * 60) + 59
    return rng.randint(start, end)


def make_timestamp(item_date: date, seconds: int) -> str:
    hour, remainder = divmod(seconds, 60 * 60)
    minute, second = divmod(remainder, 60)
    value = datetime.combine(item_date, time(hour=hour, minute=minute, second=second), tzinfo=SESSION_TIMEZONE)
    return value.isoformat(timespec="seconds")


def build_timestamped_session(session: ConversationSession, *, order: int, timestamp: str) -> TimestampedSession:
    return TimestampedSession(
        session_id=session.conversation_id,
        conversation_id=session.conversation_id,
        timestamp=timestamp,
        order=order,
        persona_id=session.persona_id,
        case_id=session.case_id,
        fact_id=session.fact_id,
        topic_preference=session.topic_preference or "",
        case_description=session.case_description or "",
        case_relation_type=session.case_relation_type,
        case_relation_subtype=session.case_relation_subtype,
        case_facts=session.case_facts,
        fact_text=session.fact_text or "",
        sampled_conversation_types=session.sampled_conversation_types,
        selected_conversation_type=session.selected_conversation_type,
        preferred_conversation_type=session.preferred_conversation_type,
        preferred_conversation_type_used=session.preferred_conversation_type_used,
        sampled_conversation_flows=session.sampled_conversation_flows,
        selected_conversation_flow=session.selected_conversation_flow,
        persona_signal_level=session.persona_signal_level,
        persona_signal_guidance=session.persona_signal_guidance,
        messages=session.messages,
        turn_count=session.turn_count or 0,
        token_count=session.token_count or 0,
        generation_model=session.generation_model,
        stream_completed=session.stream_completed,
        source_created_at=session.created_at,
    )
