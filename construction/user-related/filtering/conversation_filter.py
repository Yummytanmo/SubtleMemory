from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from infra.config import FILTER_MODEL
from infra.logging_utils import utc_now_iso
from core.parallel_utils import map_ordered, normalize_concurrency
from prompts import (
    build_conversation_filter_prompt,
    build_current_case_conversation_guidance,
    build_current_case_relation_guidance,
)
from core.schemas import (
    AcceptedCase,
    ConversationFilterDecision,
    ConversationSession,
    RejectedConversation,
    SanitizedPersonaProfile,
)
from core.validators import normalize_reject_categories, parse_json_after_output


DEFAULT_CONVERSATION_FILTER_BATCH_SIZE = 16


@dataclass
class ConversationFilterBatch:
    batch_id: int
    conversations: list[ConversationSession]
    context: dict[str, Any]


@dataclass
class ConversationFilterBatchResult:
    batch_id: int
    decisions: list[ConversationFilterDecision]
    ignored_decisions: list[dict[str, Any]]
    stream_completed: bool
    error: str | None = None


def filter_conversations(
    conversations: list[ConversationSession],
    accepted_cases: list[AcceptedCase],
    personas: dict[str, SanitizedPersonaProfile],
    llm_client: Any,
    *,
    filter_model: str = FILTER_MODEL,
    concurrency: int = 1,
    batch_size: int = DEFAULT_CONVERSATION_FILTER_BATCH_SIZE,
) -> tuple[list[ConversationSession], list[RejectedConversation], dict[str, Any]]:
    batches = build_conversation_filter_batches(
        conversations,
        accepted_cases,
        personas,
        batch_size=batch_size,
    )
    batch_results = map_ordered(
        batches,
        lambda batch: filter_conversation_batch(batch, llm_client, filter_model=filter_model),
        max_workers=normalize_concurrency(concurrency),
    )
    decisions_by_conversation_id = {
        decision.conversation_id: decision
        for result in batch_results
        for decision in result.decisions
    }

    accepted: list[ConversationSession] = []
    rejected: list[RejectedConversation] = []
    decisions: list[ConversationFilterDecision] = []
    for conversation in conversations:
        decision = decisions_by_conversation_id.get(conversation.conversation_id)
        if decision is None:
            decision = build_conversation_filter_decision(
                conversation.conversation_id,
                {"accepted": True, "reason": "accepted", "reject_categories": []},
                filter_model=filter_model,
                stream_completed=True,
            )
        decisions.append(decision)
        if decision.accepted:
            accepted.append(conversation)
        else:
            rejected.append(RejectedConversation(conversation=conversation, filter_decision=decision))

    validate_conversation_filter_partition(conversations, accepted, rejected)
    report = build_conversation_filter_report(conversations, decisions, batch_results, filter_model, batch_size)
    return accepted, rejected, report


def build_conversation_filter_batches(
    conversations: list[ConversationSession],
    accepted_cases: list[AcceptedCase],
    personas: dict[str, SanitizedPersonaProfile],
    *,
    batch_size: int = DEFAULT_CONVERSATION_FILTER_BATCH_SIZE,
) -> list[ConversationFilterBatch]:
    if batch_size < 1:
        raise ValueError("conversation filter batch_size must be >= 1")
    case_by_id = {accepted.case.case_id: accepted.case for accepted in accepted_cases}
    fact_by_id = {
        fact.fact_id: fact
        for accepted in accepted_cases
        for fact in accepted.case.facts
    }
    conversations_by_case_id: OrderedDict[str, list[ConversationSession]] = OrderedDict()
    for conversation in conversations:
        conversations_by_case_id.setdefault(conversation.case_id, []).append(conversation)
    batches: list[ConversationFilterBatch] = []
    case_ids = list(conversations_by_case_id)
    for batch_id, start in enumerate(range(0, len(case_ids), batch_size)):
        batch_case_ids = case_ids[start : start + batch_size]
        batch_conversations = [
            conversation
            for case_id in batch_case_ids
            for conversation in conversations_by_case_id[case_id]
        ]
        persona_ids = sorted({conversation.persona_id for conversation in batch_conversations})
        context = {
            "batch_id": batch_id,
            "batch_unit": "case",
            "personas": [
                {
                    "persona_id": persona_id,
                    "persona": personas[persona_id].persona_str,
                }
                for persona_id in persona_ids
            ],
            "cases": [
                build_conversation_filter_case_item(
                    case_by_id[case_id],
                    conversations_by_case_id[case_id],
                    fact_by_id,
                )
                for case_id in batch_case_ids
            ],
        }
        batches.append(ConversationFilterBatch(batch_id=batch_id, conversations=batch_conversations, context=context))
    return batches


def build_conversation_filter_case_item(
    case: Any,
    conversations: list[ConversationSession],
    fact_by_id: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "persona_id": case.persona_id,
        "topic_preference": case.topic_preference,
        "relation_type": case.relation_type,
        "relation_subtype": case.relation_subtype,
        "description": case.description,
        "facts": [item.model_dump(mode="json") for item in case.facts],
        "case_relation_guidance": build_current_case_relation_guidance(
            {
                "relation_type": case.relation_type,
                "relation_subtype": case.relation_subtype,
            }
        ),
        "case_conversation_guidance": build_current_case_conversation_guidance(
            {
                "relation_type": case.relation_type,
                "relation_subtype": case.relation_subtype,
            }
        ),
        "conversations": [
            build_conversation_filter_session_item(conversation, fact_by_id[conversation.fact_id])
            for conversation in conversations
        ],
    }


def build_conversation_filter_session_item(conversation: ConversationSession, fact: Any) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "fact_id": conversation.fact_id,
        "fact": fact.model_dump(mode="json"),
        "fact_text": conversation.fact_text,
        "sampled_conversation_types": conversation.sampled_conversation_types,
        "selected_conversation_type": conversation.selected_conversation_type,
        "turn_count": conversation.turn_count,
        "token_count": conversation.token_count,
        "messages": [message.model_dump(mode="json") for message in conversation.messages],
    }


def build_conversation_filter_session_summary(conversation: ConversationSession) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "fact_id": conversation.fact_id,
        "fact_text": conversation.fact_text,
        "selected_conversation_type": conversation.selected_conversation_type,
        "messages": [message.model_dump(mode="json") for message in conversation.messages],
    }


def filter_conversation_batch(
    batch: ConversationFilterBatch,
    llm_client: Any,
    *,
    filter_model: str = FILTER_MODEL,
) -> ConversationFilterBatchResult:
    prompt = build_conversation_filter_prompt(batch.context)
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="conversation_filter",
        temperature=0.0,
    )
    if not result.stream_completed:
        return reject_conversation_batch(
            batch,
            reason=result.error or "Conversation filter stream failed.",
            reject_categories=["stream_error"],
            filter_model=filter_model,
            stream_completed=False,
        )
    try:
        payload = parse_json_after_output(result.text)
        decisions, ignored = build_conversation_filter_decisions(
            batch.conversations,
            payload,
            filter_model=filter_model,
            stream_completed=True,
        )
        return ConversationFilterBatchResult(
            batch_id=batch.batch_id,
            decisions=decisions,
            ignored_decisions=ignored,
            stream_completed=True,
        )
    except Exception as exc:
        return reject_conversation_batch(
            batch,
            reason=f"Conversation filter JSON parse failed: {exc}",
            reject_categories=["parse_error"],
            filter_model=filter_model,
            stream_completed=True,
        )


def build_conversation_filter_decisions(
    conversations: list[ConversationSession],
    payload: Any,
    *,
    filter_model: str = FILTER_MODEL,
    stream_completed: bool = True,
) -> tuple[list[ConversationFilterDecision], list[dict[str, Any]]]:
    conversations_by_case_id: dict[str, list[ConversationSession]] = {}
    conversation_ids_by_case_id: dict[str, set[str]] = {}
    for conversation in conversations:
        conversations_by_case_id.setdefault(conversation.case_id, []).append(conversation)
        conversation_ids_by_case_id.setdefault(conversation.case_id, set()).add(conversation.conversation_id)
    known_case_ids = set(conversations_by_case_id)
    items = extract_problem_case_items(payload)
    if items is None:
        raise ValueError("conversation filter payload does not contain problem_cases")

    decisions: list[ConversationFilterDecision] = []
    ignored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            case_id = item.strip()
            decision_payload = {
                "accepted": False,
                "reason": "Case rejected by conversation filter.",
                "reject_categories": ["quality_issue"],
            }
        elif isinstance(item, dict):
            case_id = str(item.get("case_id") or "").strip()
            decision_payload = {
                "accepted": False,
                "reason": str(item.get("reason") or "Case rejected by conversation filter."),
                "reject_categories": item.get("reject_categories") or item.get("categories") or ["quality_issue"],
            }
            problem_payloads = build_problem_conversation_payloads(item, case_id, conversation_ids_by_case_id, ignored)
        else:
            ignored.append({"reason": "non_object_problem_item", "item": str(item)})
            continue
        if case_id not in known_case_ids:
            ignored.append({"case_id": case_id, "reason": "unknown_case_id"})
            continue
        if case_id in seen:
            ignored.append({"case_id": case_id, "reason": "duplicate_decision"})
            continue
        seen.add(case_id)
        if not isinstance(item, dict):
            problem_payloads = {}
        case_reject_categories = normalize_reject_categories(decision_payload["reject_categories"])
        for conversation in conversations_by_case_id[case_id]:
            per_conversation_payload = problem_payloads.get(conversation.conversation_id)
            if per_conversation_payload is None:
                per_conversation_payload = {
                    **decision_payload,
                    "reason": decision_payload["reason"],
                    "reject_categories": [
                        *case_reject_categories,
                        "case_rejected_due_to_conversation",
                    ],
                }
            decisions.append(
                build_conversation_filter_decision(
                    conversation.conversation_id,
                    per_conversation_payload,
                    filter_model=filter_model,
                    stream_completed=stream_completed,
                )
            )
    return decisions, ignored


def build_problem_conversation_payloads(
    item: dict[str, Any],
    case_id: str,
    conversation_ids_by_case_id: dict[str, set[str]],
    ignored: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    known_conversation_ids = conversation_ids_by_case_id.get(case_id, set())
    problem_payloads: dict[str, dict[str, Any]] = {}
    problem_items = item.get("problem_conversations")
    if problem_items is None:
        problem_items = item.get("problem_conversation_ids") or item.get("conversation_ids") or []
    if not isinstance(problem_items, list):
        return problem_payloads

    for problem in problem_items:
        if isinstance(problem, str):
            conversation_id = problem.strip()
            reason = str(item.get("reason") or "Conversation caused case rejection.")
            reject_categories = item.get("reject_categories") or item.get("categories") or ["quality_issue"]
        elif isinstance(problem, dict):
            conversation_id = str(problem.get("conversation_id") or "").strip()
            reason = str(problem.get("reason") or item.get("reason") or "Conversation caused case rejection.")
            reject_categories = (
                problem.get("reject_categories")
                or problem.get("categories")
                or item.get("reject_categories")
                or item.get("categories")
                or ["quality_issue"]
            )
        else:
            ignored.append({"case_id": case_id, "reason": "non_object_problem_conversation", "item": str(problem)})
            continue
        if not conversation_id:
            ignored.append({"case_id": case_id, "reason": "missing_problem_conversation_id"})
            continue
        if conversation_id not in known_conversation_ids:
            ignored.append(
                {
                    "case_id": case_id,
                    "conversation_id": conversation_id,
                    "reason": "unknown_problem_conversation_id",
                }
            )
            continue
        if conversation_id in problem_payloads:
            ignored.append(
                {
                    "case_id": case_id,
                    "conversation_id": conversation_id,
                    "reason": "duplicate_problem_conversation",
                }
            )
            continue
        problem_payloads[conversation_id] = {
            "accepted": False,
            "reason": reason,
            "reject_categories": reject_categories,
        }
    return problem_payloads


def extract_problem_case_items(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ("problem_cases", "rejected_cases", "problems"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def reject_conversation_batch(
    batch: ConversationFilterBatch,
    *,
    reason: str,
    reject_categories: list[str],
    filter_model: str = FILTER_MODEL,
    stream_completed: bool,
) -> ConversationFilterBatchResult:
    return ConversationFilterBatchResult(
        batch_id=batch.batch_id,
        decisions=[
            build_conversation_filter_decision(
                conversation.conversation_id,
                {
                    "accepted": False,
                    "reason": reason,
                    "reject_categories": reject_categories,
                },
                filter_model=filter_model,
                stream_completed=stream_completed,
            )
            for conversation in batch.conversations
        ],
        ignored_decisions=[],
        stream_completed=stream_completed,
        error=reason,
    )


def build_conversation_filter_decision(
    conversation_id: str,
    payload: dict[str, Any],
    *,
    filter_model: str = FILTER_MODEL,
    stream_completed: bool = True,
) -> ConversationFilterDecision:
    accepted = coerce_bool(payload.get("accepted", False))
    reason = str(payload.get("reason") or ("accepted" if accepted else "rejected")).strip()
    reject_categories = normalize_reject_categories(payload.get("reject_categories"))
    if accepted:
        reject_categories = []
    return ConversationFilterDecision(
        decision_id=make_conversation_filter_decision_id(conversation_id, str(accepted), reason),
        conversation_id=conversation_id,
        accepted=accepted,
        reason=reason,
        reject_categories=reject_categories,
        filter_model=filter_model,
        stream_completed=stream_completed,
        created_at=utc_now_iso(),
    )


def build_conversation_filter_report(
    conversations: list[ConversationSession],
    decisions: list[ConversationFilterDecision],
    batch_results: list[ConversationFilterBatchResult],
    filter_model: str,
    batch_size: int,
) -> dict[str, Any]:
    total = len(conversations)
    accepted = sum(1 for decision in decisions if decision.accepted)
    rejected = total - accepted
    case_ids = sorted({conversation.case_id for conversation in conversations})
    rejected_case_ids = {
        conversation.case_id
        for conversation, decision in zip(conversations, decisions)
        if not decision.accepted
    }
    accepted_case_count = len(case_ids) - len(rejected_case_ids)
    return {
        "total_generated_conversations": total,
        "accepted_conversations": accepted,
        "rejected_conversations": rejected,
        "retention_rate": accepted / total if total else 0.0,
        "filter_unit": "case",
        "total_generated_conversation_cases": len(case_ids),
        "accepted_conversation_cases": accepted_case_count,
        "rejected_conversation_cases": len(rejected_case_ids),
        "case_retention_rate": accepted_case_count / len(case_ids) if case_ids else 0.0,
        "rejected_conversation_details": [
            {
                "conversation_id": decision.conversation_id,
                "case_id": conversation.case_id,
                "fact_id": conversation.fact_id,
                "reason": decision.reason,
                "reject_categories": decision.reject_categories,
            }
            for conversation, decision in zip(conversations, decisions)
            if not decision.accepted
        ],
        "rejected_case_details": build_rejected_case_details(conversations, decisions),
        "filter_model": filter_model,
        "stream_completed": all(result.stream_completed for result in batch_results),
        "batch_size": batch_size,
        "batch_count": len(batch_results),
        "ignored_decisions": [
            {"batch_id": result.batch_id, **ignored}
            for result in batch_results
            for ignored in result.ignored_decisions
        ],
        "batch_errors": [
            {"batch_id": result.batch_id, "error": result.error}
            for result in batch_results
            if result.error
        ],
    }


def build_rejected_case_details(
    conversations: list[ConversationSession],
    decisions: list[ConversationFilterDecision],
) -> list[dict[str, Any]]:
    rejected: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for conversation, decision in zip(conversations, decisions):
        if decision.accepted:
            continue
        detail = rejected.setdefault(
            conversation.case_id,
            {
                "case_id": conversation.case_id,
                "reason": decision.reason,
                "reject_categories": decision.reject_categories,
                "rejected_conversation_ids": [],
                "rejected_conversations": [],
            },
        )
        detail["rejected_conversation_ids"].append(conversation.conversation_id)
        detail["rejected_conversations"].append(
            {
                "conversation_id": conversation.conversation_id,
                "fact_id": conversation.fact_id,
                "reason": decision.reason,
                "reject_categories": decision.reject_categories,
            }
        )
    return list(rejected.values())


def validate_conversation_filter_partition(
    generated_conversations: list[ConversationSession],
    accepted_conversations: list[ConversationSession],
    rejected_conversations: list[RejectedConversation],
) -> None:
    generated_ids = {conversation.conversation_id for conversation in generated_conversations}
    accepted_ids = {conversation.conversation_id for conversation in accepted_conversations}
    rejected_ids = {item.conversation.conversation_id for item in rejected_conversations}
    if accepted_ids & rejected_ids:
        raise ValueError("conversation cannot be accepted and rejected")
    if accepted_ids | rejected_ids != generated_ids:
        raise ValueError("conversation filter partition must cover generated conversations exactly")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "accept", "accepted"}:
            return True
        if normalized in {"false", "no", "n", "0", "reject", "rejected"}:
            return False
    return bool(value)


def make_conversation_filter_decision_id(conversation_id: str, accepted: str, reason: str) -> str:
    digest = hashlib.sha1(f"{conversation_id}|{accepted}|{reason}".encode("utf-8")).hexdigest()[:12]
    return f"conv-decision-{digest}"
