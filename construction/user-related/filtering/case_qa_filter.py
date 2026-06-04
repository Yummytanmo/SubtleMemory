from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from infra.config import FILTER_MODEL
from infra.logging_utils import utc_now_iso
from core.parallel_utils import map_ordered, normalize_concurrency
from prompts import (
    build_case_qa_filter_prompt,
    build_current_case_relation_guidance,
    build_task_qa_filter_prompt,
)
from core.schemas import (
    AcceptedCase,
    AcceptedCaseQA,
    CaseQA,
    ConversationSession,
    QAMode,
    QAFilterDecision,
    QAQuestion,
    RejectedCaseQA,
    SanitizedPersonaProfile,
)
from core.validators import normalize_reject_categories, parse_json_after_output


DEFAULT_CASE_QA_FILTER_BATCH_SIZE = 8
FATAL_CASE_QA_FILTER_REJECT_CATEGORIES = {"stream_error", "parse_error"}
QA_FILTER_SESSION_EXCLUDE_FIELDS = {
    "sampled_conversation_flows",
    "selected_conversation_flow",
    "persona_signal_level",
    "persona_signal_guidance",
}


@dataclass
class CaseQAFilterBatch:
    batch_id: int
    case_qa: list[CaseQA]
    context: dict[str, Any]


@dataclass
class CaseQAFilterBatchResult:
    batch_id: int
    decisions: list[QAFilterDecision]
    ignored_decisions: list[dict[str, Any]]
    stream_completed: bool
    error: str | None = None


def filter_case_qa(
    generated_case_qa: list[CaseQA],
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
    llm_client: Any,
    *,
    filter_model: str = FILTER_MODEL,
    concurrency: int = 1,
    batch_size: int = DEFAULT_CASE_QA_FILTER_BATCH_SIZE,
    qa_mode: QAMode | str = QAMode.QUESTION,
) -> tuple[list[AcceptedCaseQA], list[RejectedCaseQA], dict[str, Any]]:
    qa_mode = QAMode(qa_mode)
    batches = build_case_qa_filter_batches(
        generated_case_qa,
        accepted_cases,
        conversations,
        personas,
        batch_size=batch_size,
    )
    batch_results = map_ordered(
        batches,
        lambda batch: filter_case_qa_batch(batch, llm_client, filter_model=filter_model, qa_mode=qa_mode),
        max_workers=normalize_concurrency(concurrency),
    )
    raise_for_fatal_case_qa_filter_errors(batch_results)
    decisions_by_qa_id = {
        decision.qa_id: decision
        for result in batch_results
        for decision in result.decisions
    }

    accepted: list[AcceptedCaseQA] = []
    rejected: list[RejectedCaseQA] = []
    final_decisions: list[QAFilterDecision] = []
    for case_qa in generated_case_qa:
        decision = decisions_by_qa_id.get(case_qa.qa_id)
        if decision is None:
            filtered_qa, decision = build_qa_filter_decision(
                case_qa,
                rejected_questions=[],
                removed_answers=[],
                filter_model=filter_model,
                stream_completed=True,
            )
        else:
            filtered_qa, decision = apply_existing_qa_filter_decision(case_qa, decision)
        final_decisions.append(decision)
        if decision.accepted and filtered_qa is not None:
            accepted.append(AcceptedCaseQA(qa=filtered_qa, filter_decision=decision))
        if decision.rejected_questions or decision.removed_answers or not decision.accepted:
            rejected.append(RejectedCaseQA(original_qa=case_qa, filtered_qa=filtered_qa, filter_decision=decision))

    validate_case_qa_filter_partition(generated_case_qa, accepted)
    report = build_case_qa_filter_report(generated_case_qa, final_decisions, batch_results, filter_model, batch_size)
    return accepted, rejected, report


def raise_for_fatal_case_qa_filter_errors(batch_results: list[CaseQAFilterBatchResult]) -> None:
    failed_results = [
        result
        for result in batch_results
        if result.error and case_qa_filter_result_has_fatal_category(result)
    ]
    if not failed_results:
        return

    details = [
        {
            "batch_id": result.batch_id,
            "error": result.error,
            "qa_ids": [decision.qa_id for decision in result.decisions],
            "reject_categories": sorted(case_qa_filter_result_reject_categories(result)),
        }
        for result in failed_results
    ]
    first = details[0]
    raise RuntimeError(
        "Case QA filter failed with fatal batch errors: "
        f"{len(details)} batch(es); first batch_id={first['batch_id']}; "
        f"categories={','.join(first['reject_categories'])}; error={first['error']}"
    )


def case_qa_filter_result_has_fatal_category(result: CaseQAFilterBatchResult) -> bool:
    return bool(case_qa_filter_result_reject_categories(result) & FATAL_CASE_QA_FILTER_REJECT_CATEGORIES)


def case_qa_filter_result_reject_categories(result: CaseQAFilterBatchResult) -> set[str]:
    categories: set[str] = set()
    for decision in result.decisions:
        for rejected_question in decision.rejected_questions:
            categories.update(normalize_reject_categories(rejected_question.get("reject_categories", [])))
        for removed_answer in decision.removed_answers:
            categories.update(normalize_reject_categories(removed_answer.get("reject_categories", [])))
    return categories


def build_case_qa_filter_batches(
    generated_case_qa: list[CaseQA],
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
    *,
    batch_size: int = DEFAULT_CASE_QA_FILTER_BATCH_SIZE,
) -> list[CaseQAFilterBatch]:
    if batch_size < 1:
        raise ValueError("case QA filter batch_size must be >= 1")
    case_by_id = {accepted.case.case_id: accepted.case for accepted in accepted_cases}
    same_topic_siblings_by_case_id = build_same_topic_sibling_summaries_by_case_id(accepted_cases)
    conversation_by_id = {conversation.conversation_id: conversation for conversation in conversations}
    batches: list[CaseQAFilterBatch] = []
    for batch_id, start in enumerate(range(0, len(generated_case_qa), batch_size)):
        batch_items = generated_case_qa[start : start + batch_size]
        persona_ids = sorted({item.persona_id for item in batch_items})
        context = {
            "batch_id": batch_id,
            "personas": [
                {"persona_id": persona_id, "persona": personas[persona_id].persona_str}
                for persona_id in persona_ids
            ],
            "case_qa": [
                build_case_qa_filter_item(
                    item,
                    case_by_id,
                    conversation_by_id,
                    same_topic_siblings_by_case_id=same_topic_siblings_by_case_id,
                )
                for item in batch_items
            ],
        }
        batches.append(CaseQAFilterBatch(batch_id=batch_id, case_qa=batch_items, context=context))
    return batches


def build_case_qa_filter_item(
    case_qa: CaseQA,
    case_by_id: dict[str, Any],
    conversation_by_id: dict[str, ConversationSession],
    *,
    same_topic_siblings_by_case_id: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    case = case_by_id[case_qa.case_id]
    return {
        "qa": case_qa.model_dump(mode="json"),
        "case": {
            "case_id": case.case_id,
            "topic_preference": case.topic_preference,
            "relation_type": case.relation_type,
            "relation_subtype": case.relation_subtype,
            "description": case.description,
            "facts": [fact.model_dump(mode="json") for fact in case.facts],
        },
        "case_relation_guidance": build_current_case_relation_guidance(
            {
                "relation_type": case.relation_type,
                "relation_subtype": case.relation_subtype,
            }
        ),
        "same_topic_sibling_cases": (same_topic_siblings_by_case_id or {}).get(case_qa.case_id, []),
        "sessions": [
            conversation_by_id[session_id].model_dump(mode="json", exclude=QA_FILTER_SESSION_EXCLUDE_FIELDS)
            for session_id in case_qa.session_ids
            if session_id in conversation_by_id
        ],
    }


def build_same_topic_sibling_summaries_by_case_id(accepted_cases: list[AcceptedCase]) -> dict[str, list[dict[str, Any]]]:
    summaries_by_case_id: dict[str, list[dict[str, Any]]] = {}
    for accepted in accepted_cases:
        current = accepted.case
        summaries_by_case_id[current.case_id] = [
            compact_same_topic_case_summary(candidate.case)
            for candidate in accepted_cases
            if candidate.case.case_id != current.case_id
            and candidate.case.persona_id == current.persona_id
            and candidate.case.topic_preference == current.topic_preference
        ]
    return summaries_by_case_id


def compact_same_topic_case_summary(memory_case: Any) -> dict[str, Any]:
    return {
        "case_id": memory_case.case_id,
        "relation_type": memory_case.relation_type,
        "relation_subtype": memory_case.relation_subtype,
        "description": memory_case.description,
        "fact_texts": [fact.text for fact in memory_case.facts],
    }


def filter_case_qa_batch(
    batch: CaseQAFilterBatch,
    llm_client: Any,
    *,
    filter_model: str = FILTER_MODEL,
    qa_mode: QAMode | str = QAMode.QUESTION,
) -> CaseQAFilterBatchResult:
    qa_mode = QAMode(qa_mode)
    prompt = build_task_qa_filter_prompt(batch.context) if qa_mode == QAMode.TASK else build_case_qa_filter_prompt(batch.context)
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="case_qa_filter",
        temperature=0.0,
    )
    if not result.stream_completed:
        return reject_case_qa_batch(
            batch,
            reason=result.error or "Case QA filter stream failed.",
            reject_categories=["stream_error"],
            filter_model=filter_model,
            stream_completed=False,
        )
    try:
        payload = parse_json_after_output(result.text)
        decisions, ignored = build_case_qa_filter_decisions(
            batch.case_qa,
            payload,
            filter_model=filter_model,
            stream_completed=True,
        )
        return CaseQAFilterBatchResult(
            batch_id=batch.batch_id,
            decisions=decisions,
            ignored_decisions=ignored,
            stream_completed=True,
        )
    except Exception as exc:
        return reject_case_qa_batch(
            batch,
            reason=f"Case QA filter JSON parse failed: {exc}",
            reject_categories=["parse_error"],
            filter_model=filter_model,
            stream_completed=True,
        )


def build_case_qa_filter_decisions(
    batch_case_qa: list[CaseQA],
    payload: Any,
    *,
    filter_model: str = FILTER_MODEL,
    stream_completed: bool = True,
) -> tuple[list[QAFilterDecision], list[dict[str, Any]]]:
    rejected_items = extract_rejected_question_items(payload)
    removed_items = extract_removed_answer_items(payload)
    if rejected_items is None or removed_items is None:
        raise ValueError("case QA filter payload must contain rejected_questions and removed_answers")

    question_to_qa_id = {
        question.question_id: case_qa.qa_id
        for case_qa in batch_case_qa
        for question in case_qa.questions
    }
    answer_to_qa_id = {
        answer.answer_id: case_qa.qa_id
        for case_qa in batch_case_qa
        for question in case_qa.questions
        for answer in [*question.correct_answers, *question.incorrect_answers]
    }
    rejected_by_qa_id: dict[str, list[dict[str, Any]]] = {case_qa.qa_id: [] for case_qa in batch_case_qa}
    removed_by_qa_id: dict[str, list[dict[str, Any]]] = {case_qa.qa_id: [] for case_qa in batch_case_qa}
    ignored: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()

    for item in rejected_items:
        if not isinstance(item, dict):
            ignored.append({"reason": "non_object_rejected_question", "item": str(item)})
            continue
        question_id = str(item.get("question_id") or "").strip()
        qa_id = question_to_qa_id.get(question_id)
        if qa_id is None:
            ignored.append({"question_id": question_id, "reason": "unknown_question_id"})
            continue
        if question_id in seen_questions:
            ignored.append({"question_id": question_id, "reason": "duplicate_question_decision"})
            continue
        seen_questions.add(question_id)
        rejected_by_qa_id[qa_id].append(normalize_rejected_question_item(item))

    for item in removed_items:
        if not isinstance(item, dict):
            ignored.append({"reason": "non_object_removed_answer", "item": str(item)})
            continue
        answer_id = str(item.get("answer_id") or "").strip()
        qa_id = answer_to_qa_id.get(answer_id)
        if qa_id is None:
            ignored.append({"answer_id": answer_id, "reason": "unknown_answer_id"})
            continue
        if answer_id in seen_answers:
            ignored.append({"answer_id": answer_id, "reason": "duplicate_answer_decision"})
            continue
        seen_answers.add(answer_id)
        removed_by_qa_id[qa_id].append(normalize_removed_answer_item(item))

    decisions: list[QAFilterDecision] = []
    for case_qa in batch_case_qa:
        _, decision = build_qa_filter_decision(
            case_qa,
            rejected_questions=rejected_by_qa_id[case_qa.qa_id],
            removed_answers=removed_by_qa_id[case_qa.qa_id],
            filter_model=filter_model,
            stream_completed=stream_completed,
        )
        decisions.append(decision)
    return decisions, ignored


def build_qa_filter_decision(
    case_qa: CaseQA,
    *,
    rejected_questions: list[dict[str, Any]],
    removed_answers: list[dict[str, Any]],
    filter_model: str = FILTER_MODEL,
    stream_completed: bool = True,
) -> tuple[CaseQA | None, QAFilterDecision]:
    filtered_qa, final_rejected_questions, final_removed_answers = apply_qa_changes(
        case_qa,
        rejected_questions,
        removed_answers,
    )
    accepted = filtered_qa is not None and bool(filtered_qa.questions)
    reason = "" if accepted else "No valid questions remain after QA filtering."
    decision = QAFilterDecision(
        decision_id=make_case_qa_filter_decision_id(case_qa.qa_id, accepted, final_rejected_questions, final_removed_answers),
        qa_id=case_qa.qa_id,
        accepted=accepted,
        reason=reason,
        rejected_questions=final_rejected_questions,
        removed_answers=final_removed_answers,
        filter_model=filter_model,
        stream_completed=stream_completed,
        created_at=utc_now_iso(),
    )
    return filtered_qa, decision


def apply_existing_qa_filter_decision(case_qa: CaseQA, decision: QAFilterDecision) -> tuple[CaseQA | None, QAFilterDecision]:
    return build_qa_filter_decision(
        case_qa,
        rejected_questions=decision.rejected_questions,
        removed_answers=decision.removed_answers,
        filter_model=decision.filter_model,
        stream_completed=decision.stream_completed,
    )


def apply_qa_changes(
    case_qa: CaseQA,
    rejected_questions: list[dict[str, Any]],
    removed_answers: list[dict[str, Any]],
) -> tuple[CaseQA | None, list[dict[str, Any]], list[dict[str, Any]]]:
    rejected_by_question_id = {str(item.get("question_id") or ""): item for item in rejected_questions}
    removed_answer_ids = {str(item.get("answer_id") or "") for item in removed_answers}
    final_rejected = list(rejected_questions)
    final_removed = list(removed_answers)
    filtered_questions: list[QAQuestion] = []
    for question in case_qa.questions:
        if question.question_id in rejected_by_question_id:
            continue
        correct_answers = [answer for answer in question.correct_answers if answer.answer_id not in removed_answer_ids]
        incorrect_answers = [answer for answer in question.incorrect_answers if answer.answer_id not in removed_answer_ids]
        if not correct_answers:
            final_rejected.append(
                {
                    "question_id": question.question_id,
                    "reason": "No valid correct answers remain.",
                    "reject_categories": ["no_remaining_correct_answers"],
                }
            )
            continue
        if not incorrect_answers:
            final_rejected.append(
                {
                    "question_id": question.question_id,
                    "reason": "No valid incorrect answers remain.",
                    "reject_categories": ["no_remaining_incorrect_answers"],
                }
            )
            continue
        filtered_questions.append(
            QAQuestion(
                question_id=question.question_id,
                question=question.question,
                task_form=question.task_form,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
            )
        )
    if not filtered_questions:
        return None, final_rejected, final_removed
    filtered_qa = case_qa.model_copy(update={"questions": filtered_questions})
    return filtered_qa, final_rejected, final_removed


def reject_case_qa_batch(
    batch: CaseQAFilterBatch,
    *,
    reason: str,
    reject_categories: list[str],
    filter_model: str = FILTER_MODEL,
    stream_completed: bool,
) -> CaseQAFilterBatchResult:
    decisions = []
    for case_qa in batch.case_qa:
        _, decision = build_qa_filter_decision(
            case_qa,
            rejected_questions=[
                {
                    "question_id": question.question_id,
                    "reason": reason,
                    "reject_categories": reject_categories,
                }
                for question in case_qa.questions
            ],
            removed_answers=[],
            filter_model=filter_model,
            stream_completed=stream_completed,
        )
        decisions.append(decision)
    return CaseQAFilterBatchResult(
        batch_id=batch.batch_id,
        decisions=decisions,
        ignored_decisions=[],
        stream_completed=stream_completed,
        error=reason,
    )


def extract_rejected_question_items(payload: Any) -> list[Any] | None:
    if isinstance(payload, dict):
        for key in ("rejected_questions", "problem_questions"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return None


def extract_removed_answer_items(payload: Any) -> list[Any] | None:
    if isinstance(payload, dict):
        for key in ("removed_answers", "problem_answers"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return None


def normalize_rejected_question_item(item: dict[str, Any]) -> dict[str, Any]:
    output = {
        "question_id": str(item.get("question_id") or "").strip(),
        "reason": str(item.get("reason") or "Rejected by QA filter.").strip(),
        "reject_categories": normalize_reject_categories(item.get("reject_categories") or item.get("categories")),
    }
    duplicate_of = str(item.get("duplicate_of_question_id") or "").strip()
    if duplicate_of:
        output["duplicate_of_question_id"] = duplicate_of
    return output


def normalize_removed_answer_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_id": str(item.get("answer_id") or "").strip(),
        "answer_type": str(item.get("answer_type") or "").strip(),
        "reason": str(item.get("reason") or "Removed by QA filter.").strip(),
        "reject_categories": normalize_reject_categories(item.get("reject_categories") or item.get("categories")),
    }


def build_case_qa_filter_report(
    generated_case_qa: list[CaseQA],
    decisions: list[QAFilterDecision],
    batch_results: list[CaseQAFilterBatchResult],
    filter_model: str,
    batch_size: int,
) -> dict[str, Any]:
    total = len(generated_case_qa)
    accepted = sum(1 for decision in decisions if decision.accepted)
    rejected = total - accepted
    rejected_question_details = [
        {"qa_id": decision.qa_id, **item}
        for decision in decisions
        for item in decision.rejected_questions
    ]
    removed_answer_details = [
        {"qa_id": decision.qa_id, **item}
        for decision in decisions
        for item in decision.removed_answers
    ]
    qa_item_metrics = build_case_qa_item_filter_metrics(generated_case_qa, decisions)
    return {
        "total_generated_case_qa": total,
        "accepted_case_qa": accepted,
        "rejected_case_qa": rejected,
        "retention_rate": accepted / total if total else 0.0,
        "case_qa_with_filter_changes": sum(
            1
            for decision in decisions
            if decision.rejected_questions or decision.removed_answers or not decision.accepted
        ),
        "rejected_case_questions": len(rejected_question_details),
        "removed_case_qa_answers": len(removed_answer_details),
        **qa_item_metrics,
        "rejected_question_details": rejected_question_details,
        "removed_answer_details": removed_answer_details,
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


def build_case_qa_item_filter_metrics(
    generated_case_qa: list[CaseQA],
    decisions: list[QAFilterDecision],
) -> dict[str, Any]:
    decisions_by_qa_id = {decision.qa_id: decision for decision in decisions}
    accepted_case_qa: list[CaseQA] = []
    for case_qa in generated_case_qa:
        decision = decisions_by_qa_id.get(case_qa.qa_id)
        if decision is None:
            filtered_qa = case_qa
        else:
            filtered_qa, _, _ = apply_qa_changes(case_qa, decision.rejected_questions, decision.removed_answers)
        if filtered_qa is not None:
            accepted_case_qa.append(filtered_qa)

    generated_questions = count_case_qa_questions(generated_case_qa)
    accepted_questions = count_case_qa_questions(accepted_case_qa)
    return {
        "generated_case_qa_questions": generated_questions,
        "accepted_case_qa_questions": accepted_questions,
        "rejected_case_qa_questions": generated_questions - accepted_questions,
        "question_retention_rate": safe_pass_rate(accepted_questions, generated_questions),
    }


def count_case_qa_questions(case_qa: list[CaseQA]) -> int:
    return sum(len(item.questions) for item in case_qa)


def safe_pass_rate(accepted: int, total: int) -> float:
    return accepted / total if total else 0.0


def validate_case_qa_filter_partition(generated_case_qa: list[CaseQA], accepted_case_qa: list[AcceptedCaseQA]) -> None:
    generated_ids = {case_qa.qa_id for case_qa in generated_case_qa}
    accepted_ids = {item.qa.qa_id for item in accepted_case_qa}
    if not accepted_ids <= generated_ids:
        raise ValueError("accepted QA must be a subset of generated QA")


def make_case_qa_filter_decision_id(
    qa_id: str,
    accepted: bool,
    rejected_questions: list[dict[str, Any]],
    removed_answers: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha1(
        f"{qa_id}|{accepted}|{rejected_questions}|{removed_answers}".encode("utf-8")
    ).hexdigest()[:12]
    return f"qa-decision-{digest}"
