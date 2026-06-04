from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any

from infra.config import FILTER_MODEL
from infra.logging_utils import utc_now_iso
from core.parallel_utils import map_ordered, normalize_concurrency
from prompts import (
    FILTER_ACCEPT_TOKEN,
    FILTER_REJECT_TOKEN,
    build_filter_prompt,
    build_persona_consistency_filter_prompt,
    build_topic_filter_prompt,
)
from core.schemas import AcceptedCase, FilterDecision, MemoryCase, RejectedCase, SanitizedPersonaProfile
from core.validators import (
    ValidationError,
    filter_report_from_decisions,
    normalize_reject_categories,
    parse_json_after_output,
    validate_filter_partition,
)


def filter_memory_cases(
    cases: list[MemoryCase],
    personas: dict[str, SanitizedPersonaProfile],
    llm_client: Any,
    *,
    filter_model: str = FILTER_MODEL,
    concurrency: int = 1,
) -> tuple[list[AcceptedCase], list[RejectedCase], dict[str, Any]]:
    topic_jobs = build_topic_filter_jobs(cases, personas)
    grouped_decisions: list[list[FilterDecision]] = map_ordered(
        topic_jobs,
        lambda job: filter_topic_cases(job, llm_client, filter_model=filter_model),
        max_workers=normalize_concurrency(concurrency),
    )
    decisions_by_case_id = {
        decision.case_id: decision
        for decisions in grouped_decisions
        for decision in decisions
    }
    topic_decisions = [
        decisions_by_case_id.get(case.case_id)
        or build_filter_decision(
            case.case_id,
            {
                "accepted": False,
                "reason": "Filter did not return a decision for this case.",
                "reject_categories": ["missing_decision"],
            },
            filter_model=filter_model,
        )
        for case in cases
    ]
    persona_jobs = build_persona_consistency_jobs(cases, topic_decisions, personas)
    persona_results = map_ordered(
        persona_jobs,
        lambda job: filter_persona_consistency(job, llm_client, filter_model=filter_model),
        max_workers=normalize_concurrency(concurrency),
    )
    persona_rejections = {
        decision.case_id: decision
        for result in persona_results
        for decision in result.decisions
    }
    final_decisions = [
        persona_rejections.get(case.case_id, topic_decision)
        for case, topic_decision in zip(cases, topic_decisions)
    ]

    accepted: list[AcceptedCase] = []
    rejected: list[RejectedCase] = []
    for case, decision in zip(cases, final_decisions):
        if decision.accepted:
            accepted.append(AcceptedCase(case=case, filter_decision=decision))
        else:
            rejected.append(RejectedCase(case=case, filter_decision=decision))
    validate_filter_partition(cases, accepted, rejected)
    report = filter_report_from_decisions(final_decisions)
    report["filter_model"] = filter_model
    report["stream_completed"] = all(decision.stream_completed for decision in final_decisions) and all(
        result.stream_completed for result in persona_results
    )
    report["topic_filter_units"] = len(topic_jobs)
    report["persona_filter_units"] = len(persona_jobs)
    report["persona_filter_errors"] = [
        {"persona_id": result.persona_id, "error": result.error}
        for result in persona_results
        if result.error
    ]
    return accepted, rejected, report


@dataclass
class TopicFilterJob:
    persona: SanitizedPersonaProfile
    persona_id: str
    topic_preference: str
    cases: list[MemoryCase]


@dataclass
class PersonaConsistencyJob:
    persona: SanitizedPersonaProfile
    persona_id: str
    cases: list[MemoryCase]


@dataclass
class PersonaConsistencyResult:
    persona_id: str
    decisions: list[FilterDecision]
    stream_completed: bool
    error: str | None = None


def build_topic_filter_jobs(
    cases: list[MemoryCase],
    personas: dict[str, SanitizedPersonaProfile],
) -> list[TopicFilterJob]:
    grouped: OrderedDict[tuple[str, str], list[MemoryCase]] = OrderedDict()
    for case in cases:
        grouped.setdefault((case.persona_id, case.topic_preference), []).append(case)
    return [
        TopicFilterJob(
            persona=personas[persona_id],
            persona_id=persona_id,
            topic_preference=topic_preference,
            cases=topic_cases,
        )
        for (persona_id, topic_preference), topic_cases in grouped.items()
    ]


def build_persona_consistency_jobs(
    cases: list[MemoryCase],
    decisions: list[FilterDecision],
    personas: dict[str, SanitizedPersonaProfile],
) -> list[PersonaConsistencyJob]:
    grouped: OrderedDict[str, list[MemoryCase]] = OrderedDict()
    for case, decision in zip(cases, decisions):
        if decision.accepted:
            grouped.setdefault(case.persona_id, []).append(case)
    return [
        PersonaConsistencyJob(persona=personas[persona_id], persona_id=persona_id, cases=persona_cases)
        for persona_id, persona_cases in grouped.items()
        if len(persona_cases) > 1
    ]


def filter_topic_cases(
    job: TopicFilterJob,
    llm_client: Any,
    *,
    filter_model: str = FILTER_MODEL,
) -> list[FilterDecision]:
    prompt = build_topic_filter_prompt(
        job.persona.persona_str,
        job.topic_preference,
        [case.model_dump(mode="json") for case in job.cases],
    )
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="case_filter",
        temperature=0.0,
    )
    if not result.stream_completed:
        return reject_topic_cases(
            job.cases,
            reason=result.error or "Filter stream failed.",
            reject_categories=["stream_error"],
            filter_model=filter_model,
            stream_completed=False,
        )
    try:
        payload = parse_json_after_output(result.text)
        return build_topic_filter_decisions(job.cases, payload, filter_model=filter_model, stream_completed=True)
    except Exception as exc:
        return reject_topic_cases(
            job.cases,
            reason=f"Filter JSON parse failed: {exc}",
            reject_categories=["parse_error"],
            filter_model=filter_model,
            stream_completed=True,
        )


def filter_persona_consistency(
    job: PersonaConsistencyJob,
    llm_client: Any,
    *,
    filter_model: str = FILTER_MODEL,
) -> PersonaConsistencyResult:
    prompt = build_persona_consistency_filter_prompt(
        job.persona.persona_str,
        [case.model_dump(mode="json") for case in job.cases],
    )
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="persona_filter",
        temperature=0.0,
    )
    if not result.stream_completed:
        return PersonaConsistencyResult(
            persona_id=job.persona_id,
            decisions=[],
            stream_completed=False,
            error=result.error or "Persona consistency filter stream failed.",
        )
    try:
        payload = parse_json_after_output(result.text)
        decisions = build_persona_consistency_decisions(job.cases, payload, filter_model=filter_model)
        return PersonaConsistencyResult(persona_id=job.persona_id, decisions=decisions, stream_completed=True)
    except Exception as exc:
        return PersonaConsistencyResult(
            persona_id=job.persona_id,
            decisions=[],
            stream_completed=True,
            error=f"Persona consistency JSON parse failed: {exc}",
        )


def filter_one_case(
    case: MemoryCase,
    persona: SanitizedPersonaProfile,
    llm_client: Any,
    *,
    filter_model: str = FILTER_MODEL,
) -> FilterDecision:
    prompt = build_filter_prompt(persona.persona_str, case.model_dump(mode="json"))
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="case_filter",
        temperature=0.0,
    )
    if not result.stream_completed:
        return build_filter_decision(
            case.case_id,
            {
                "accepted": False,
                "reason": result.error or "Filter stream failed.",
                "reject_categories": ["stream_error"],
            },
            filter_model=filter_model,
            stream_completed=False,
        )
    payload = parse_json_after_output(result.text)
    return build_filter_decision(case.case_id, payload, filter_model=filter_model, stream_completed=True)


def build_topic_filter_decisions(
    cases: list[MemoryCase],
    payload: Any,
    *,
    filter_model: str = FILTER_MODEL,
    stream_completed: bool = True,
) -> list[FilterDecision]:
    case_by_id = {case.case_id: case for case in cases}
    decision_items = extract_filter_decision_items(payload, cases)
    if decision_items is None:
        raise ValidationError("filter payload does not contain case decisions")

    decisions_by_case_id: dict[str, FilterDecision] = {}
    for index, item in enumerate(decision_items):
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or "").strip()
        if not case_id and len(cases) == 1:
            case_id = cases[0].case_id
        if not case_id and index < len(cases):
            case_id = cases[index].case_id
        if case_id not in case_by_id or case_id in decisions_by_case_id:
            continue
        decisions_by_case_id[case_id] = build_filter_decision(
            case_id,
            item,
            filter_model=filter_model,
            stream_completed=stream_completed,
        )

    decisions = []
    for case in cases:
        decisions.append(
            decisions_by_case_id.get(case.case_id)
            or build_filter_decision(
                case.case_id,
                {
                    "accepted": False,
                    "reason": "Filter did not return a decision for this case.",
                    "reject_categories": ["missing_decision"],
                },
                filter_model=filter_model,
                stream_completed=stream_completed,
            )
        )
    return decisions


def build_persona_consistency_decisions(
    cases: list[MemoryCase],
    payload: Any,
    *,
    filter_model: str = FILTER_MODEL,
) -> list[FilterDecision]:
    case_ids = {case.case_id for case in cases}
    items = extract_persona_problem_items(payload)
    if items is None:
        raise ValidationError("persona filter payload does not contain problem cases")

    decisions: list[FilterDecision] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            case_id = item.strip()
            payload_item = {
                "accepted": False,
                "reason": "Rejected by persona consistency filter.",
                "reject_categories": ["cross_topic_conflict"],
            }
        elif isinstance(item, dict):
            accepted = item.get("accepted")
            if accepted is not None and coerce_bool(accepted):
                continue
            case_id = str(item.get("case_id") or "").strip()
            payload_item = {
                "accepted": False,
                "reason": str(item.get("reason") or "Rejected by persona consistency filter."),
                "reject_categories": item.get("reject_categories") or item.get("categories") or ["cross_topic_conflict"],
            }
        else:
            continue
        if case_id not in case_ids or case_id in seen:
            continue
        seen.add(case_id)
        decisions.append(build_filter_decision(case_id, payload_item, filter_model=filter_model, stream_completed=True))
    return decisions


def extract_persona_problem_items(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ("problem_cases", "rejected_cases", "problems"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    for key in ("case_decisions", "decisions", "filter_decisions", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, dict)
                and not coerce_bool(item.get("accepted", str(item.get("final_decision") or item.get("decision") or "").upper() == FILTER_ACCEPT_TOKEN))
            ]
    return None


def extract_filter_decision_items(payload: Any, cases: list[MemoryCase]) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ("case_decisions", "decisions", "filter_decisions", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    for key in ("problem_cases", "rejected_cases"):
        value = payload.get(key)
        if isinstance(value, list):
            return decision_items_from_problem_cases(value, cases)
    if len(cases) == 1 and ("accepted" in payload or "final_decision" in payload or "decision" in payload):
        item = dict(payload)
        item.setdefault("case_id", cases[0].case_id)
        return [item]
    return None


def decision_items_from_problem_cases(problem_cases: list[Any], cases: list[MemoryCase]) -> list[dict[str, Any]]:
    rejected_by_id: dict[str, dict[str, Any]] = {}
    for item in problem_cases:
        if isinstance(item, str):
            rejected_by_id[item] = {
                "case_id": item,
                "accepted": False,
                "reason": "Rejected by topic filter.",
                "reject_categories": ["quality_issue"],
            }
        elif isinstance(item, dict) and item.get("case_id"):
            case_id = str(item["case_id"])
            rejected_by_id[case_id] = {
                "case_id": case_id,
                "accepted": False,
                "reason": str(item.get("reason") or "Rejected by topic filter."),
                "reject_categories": item.get("reject_categories") or item.get("categories") or ["quality_issue"],
            }
    return [
        rejected_by_id.get(case.case_id)
        or {
            "case_id": case.case_id,
            "accepted": True,
            "reason": "accepted",
            "reject_categories": [],
        }
        for case in cases
    ]


def reject_topic_cases(
    cases: list[MemoryCase],
    *,
    reason: str,
    reject_categories: list[str],
    filter_model: str = FILTER_MODEL,
    stream_completed: bool,
) -> list[FilterDecision]:
    return [
        build_filter_decision(
            case.case_id,
            {
                "accepted": False,
                "reason": reason,
                "reject_categories": reject_categories,
            },
            filter_model=filter_model,
            stream_completed=stream_completed,
        )
        for case in cases
    ]


def build_filter_decision(
    case_id: str,
    payload: dict[str, Any],
    *,
    filter_model: str = FILTER_MODEL,
    stream_completed: bool = True,
) -> FilterDecision:
    final_decision = str(payload.get("final_decision") or payload.get("decision") or "").strip().upper()
    accepted = payload.get("accepted")
    if accepted is None:
        accepted = final_decision == FILTER_ACCEPT_TOKEN
    else:
        accepted = coerce_bool(accepted)
    if final_decision == FILTER_ACCEPT_TOKEN:
        accepted = True
    elif final_decision == FILTER_REJECT_TOKEN:
        accepted = False
    reason = str(payload.get("reason") or ("accepted" if accepted else "rejected")).strip()
    reject_categories = normalize_reject_categories(payload.get("reject_categories"))
    if accepted:
        reject_categories = []
    return FilterDecision(
        decision_id=make_decision_id(case_id, final_decision or str(accepted)),
        case_id=case_id,
        accepted=accepted,
        reason=reason,
        reject_categories=reject_categories,
        filter_model=filter_model,
        stream_completed=stream_completed,
        created_at=utc_now_iso(),
    )


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


def make_decision_id(case_id: str, decision: str) -> str:
    digest = hashlib.sha1(f"{case_id}|{decision}".encode("utf-8")).hexdigest()[:12]
    return f"decision-{digest}"
