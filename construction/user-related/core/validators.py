from __future__ import annotations

import json
import re
from ast import literal_eval
from typing import Any

from json_repair import loads as repair_json_loads

from core.schemas import (
    AcceptedCase,
    ConversationSession,
    FilterDecision,
    MemoryCase,
    Message,
    MIN_CONVERSATION_MESSAGES,
    ConversationType,
    RelationSubtype,
    RelationType,
    RejectedCase,
    RELATION_SUBTYPE_MAP,
)


class ValidationError(ValueError):
    pass


def parse_json_after_output(text: str) -> Any:
    body = extract_output_body(text)
    candidates = [body]
    extracted = extract_first_json(body)
    if extracted and extracted != body:
        candidates.append(extracted)

    errors: list[str] = []
    for candidate in candidates:
        try:
            return load_json_with_repair(candidate)
        except Exception as exc:
            errors.append(str(exc))
    raise ValidationError(f"No parseable JSON found after ###Output: {'; '.join(errors[-2:])}")


def extract_output_body(text: str) -> str:
    if "###Output" in text:
        body = text.split("###Output", 1)[1]
    elif "### Output" in text:
        body = text.split("### Output", 1)[1]
    else:
        body = text
    body = body.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json|JSON)?", "", body).strip()
        body = re.sub(r"```$", "", body).strip()
    return body


def load_json_with_repair(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        raise ValidationError("empty JSON candidate")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return repair_json_loads(candidate)
    except Exception:
        return fallback_load_relaxed_json(candidate)


def fallback_load_relaxed_json(text: str) -> Any:
    candidate = re.sub(r",(\s*[}\]])", r"\1", text.strip())
    candidate = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)", r'\1"\2"\3', candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pythonish = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
        pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
        pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)
        return literal_eval(pythonish)


def extract_first_json(text: str) -> str | None:
    starts = [idx for idx, char in enumerate(text) if char in "[{"]
    for start in starts:
        opener = text[start]
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return None


def relation_subtype_compatible(relation_type: str, relation_subtype: str) -> bool:
    relation = RelationType(relation_type)
    subtype = RelationSubtype(relation_subtype)
    return subtype in RELATION_SUBTYPE_MAP[relation]


def validate_relation_subtype(relation_type: str, relation_subtype: str) -> None:
    if not relation_subtype_compatible(relation_type, relation_subtype):
        raise ValidationError(f"{relation_subtype} is incompatible with {relation_type}")


def ensure_one_sentence(text: str) -> None:
    text = text.strip()
    if not text:
        raise ValidationError("fact is empty")
    if any(label in text.lower() for label in ("benchmark", "test fact", "memory case")):
        raise ValidationError("fact contains benchmark labels")
    if not re.search(r"[.!?]$", text):
        raise ValidationError("fact must end with terminal punctuation")


def normalize_messages(messages: list[dict[str, Any]]) -> list[Message]:
    normalized: list[Message] = []
    for item in messages:
        message = Message(role=item["role"], content=item["content"])
        if normalized and normalized[-1].role == message.role and isinstance(normalized[-1].content, str) and isinstance(message.content, str):
            normalized[-1] = Message(role=message.role, content=f"{normalized[-1].content}\n\n{message.content}")
        else:
            normalized.append(message)
    return normalized


def validate_openai_messages(messages: list[Message]) -> None:
    if len(messages) < MIN_CONVERSATION_MESSAGES:
        raise ValidationError("messages must contain at least three complete turns")
    if len(messages) % 2:
        raise ValidationError("messages must contain complete user/assistant turns")
    expected_roles = ["user", "assistant"]
    for index, message in enumerate(messages):
        if message.role not in {"user", "assistant"}:
            raise ValidationError(f"invalid role: {message.role}")
        if message.role != expected_roles[index % 2]:
            raise ValidationError("messages must start with user and alternate user/assistant")
        if isinstance(message.content, str) and not message.content.strip():
            raise ValidationError("empty message content")


def normalize_reject_categories(categories: Any) -> list[str]:
    if categories is None:
        return []
    if isinstance(categories, str):
        raw = re.split(r"[,;]", categories)
    elif isinstance(categories, list):
        raw = categories
    else:
        raw = [str(categories)]
    normalized = []
    for item in raw:
        value = str(item).strip().lower().replace(" ", "_")
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def validate_filter_partition(
    generated_cases: list[MemoryCase],
    accepted_cases: list[AcceptedCase],
    rejected_cases: list[RejectedCase],
) -> None:
    generated_ids = {case.case_id for case in generated_cases}
    accepted_ids = {item.case.case_id for item in accepted_cases}
    rejected_ids = {item.case.case_id for item in rejected_cases}
    if accepted_ids & rejected_ids:
        raise ValidationError("case cannot be accepted and rejected")
    if accepted_ids | rejected_ids != generated_ids:
        raise ValidationError("filter partition must cover generated cases exactly")


def calculate_pass_rate(accepted_count: int, generated_count: int) -> float:
    return accepted_count / generated_count if generated_count else 0.0


def validate_duplicate_accepted_cases(accepted_cases: list[AcceptedCase]) -> None:
    seen: set[tuple[str, str, str]] = set()
    for accepted in accepted_cases:
        key = (
            accepted.case.persona_id,
            accepted.case.topic_preference.lower(),
            accepted.case.description.lower(),
        )
        if key in seen:
            raise ValidationError(f"duplicate accepted case: {accepted.case.case_id}")
        seen.add(key)


def validate_rejected_cases_excluded(conversations: list[ConversationSession], rejected_cases: list[RejectedCase]) -> None:
    rejected_ids = {item.case.case_id for item in rejected_cases}
    leaked = [session.case_id for session in conversations if session.case_id in rejected_ids]
    if leaked:
        raise ValidationError(f"rejected cases used in conversations: {leaked}")


def validate_conversation_session(session: ConversationSession, fact_text: str) -> None:
    if len(session.sampled_conversation_types) != 4 or len(set(session.sampled_conversation_types)) != 4:
        raise ValidationError("sampled_conversation_types must contain four unique values")
    if ConversationType(session.selected_conversation_type) not in {
        ConversationType(item)
        for item in session.sampled_conversation_types
    }:
        raise ValidationError("selected_conversation_type not sampled")
    validate_openai_messages(session.messages)
    joined = "\n".join(str(message.content) for message in session.messages).lower()
    if "benchmark" in joined or "memory label" in joined:
        raise ValidationError("conversation exposes benchmark framing")
    if joined.strip() == fact_text.strip().lower():
        raise ValidationError("conversation only repeats the fact")


def assert_no_secrets(value: Any, secrets: list[str]) -> None:
    text = json.dumps(value, ensure_ascii=False, default=str)
    for secret in secrets:
        if secret and secret in text:
            raise ValidationError("secret leaked into output")


def filter_report_from_decisions(decisions: list[FilterDecision]) -> dict[str, Any]:
    total = len(decisions)
    accepted = sum(1 for decision in decisions if decision.accepted)
    rejected = total - accepted
    return {
        "total_generated_cases": total,
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "retention_rate": calculate_pass_rate(accepted, total),
        "rejected_case_details": [
            {
                "case_id": decision.case_id,
                "reason": decision.reason,
                "reject_categories": decision.reject_categories,
            }
            for decision in decisions
            if not decision.accepted
        ],
    }
