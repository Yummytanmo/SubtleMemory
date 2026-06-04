"""Evaluation run post-processing helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

from evaluation.src.core.data_models import AnswerResult, EvaluationResult


def _normalize_text(text: Any) -> str:
    if text is None:
        normalized = ""
    else:
        normalized = str(text)
    return re.sub(r"\s+", " ", normalized.strip().lower())


def flatten_evaluation_details(eval_result: EvaluationResult) -> List[Dict[str, Any]]:
    """Flatten grouped evaluation details into a list."""
    details = eval_result.detailed_results
    if isinstance(details, dict):
        rows: List[Dict[str, Any]] = []
        for grouped_rows in details.values():
            rows.extend(grouped_rows)
        return rows
    return list(details or [])


def project_qa_results_to_answer_results(
    qa_rows: List[Dict[str, Any]],
) -> List[AnswerResult]:
    """Project QAResult-like rows back to evaluator inputs."""
    answer_results: List[AnswerResult] = []
    for row in qa_rows:
        retrieval_artifact = row.get("retrieval_artifact", {}) or {}
        answer_results.append(
            AnswerResult(
                question_id=row["question_id"],
                question=row["question"],
                answer=row.get("predicted_answer", ""),
                golden_answer=row.get("golden_answer", ""),
                category=row.get("category"),
                conversation_id=row.get("conversation_id", ""),
                formatted_context=retrieval_artifact.get("formatted_context", ""),
                search_results=retrieval_artifact.get("retrieved_items", []),
                latency_ms=row.get("latency_ms", 0.0),
                errors=row.get("errors", []),
                metadata=row.get("metadata", {}),
            )
        )
    return answer_results


def build_evaluation_rows(
    qa_rows: List[Dict[str, Any]], eval_result: EvaluationResult
) -> Dict[str, Dict[str, Any]]:
    """Build per-question evaluation rows from existing evaluator output."""
    flattened_rows = flatten_evaluation_details(eval_result)
    raw_eval_map = {
        row["question_id"]: row for row in flattened_rows if "question_id" in row
    }

    evaluation_rows: Dict[str, Dict[str, Any]] = {}
    for qa_row in qa_rows:
        question_id = qa_row["question_id"]
        eval_row = raw_eval_map.get(question_id, {})
        predicted_answer = qa_row.get("predicted_answer", "")
        golden_answer = qa_row.get("golden_answer", "")
        metadata = qa_row.get("metadata", {}) or {}

        exact_match = _normalize_text(predicted_answer) == _normalize_text(
            golden_answer
        )
        judge_label = "UNKNOWN"
        judge_reason = ""
        final_score = False
        answer_judge_llm_backed = False
        answer_judge_source = "evaluator"

        if "is_correct" in eval_row:
            final_score = bool(eval_row.get("is_correct"))
            judge_label = "CORRECT" if final_score else "WRONG"
            judge_reason = str(eval_row.get("judge_reason") or "").strip()
            answer_judge_source = str(
                eval_row.get("answer_judge_source") or answer_judge_source
            )
        elif "llm_judgments" in eval_row:
            judgments = eval_row.get("llm_judgments", {}) or {}
            values = [bool(value) for value in judgments.values()]
            if values:
                final_score = Counter(values).most_common(1)[0][0]
                judge_label = "CORRECT" if final_score else "WRONG"
                majority_summary = f"majority_vote={sum(values)}/{len(values)}"
                llm_reason = str(eval_row.get("judge_reason") or "").strip()
                judge_reason = (
                    llm_reason
                    if len(values) == 1 and llm_reason
                    else (
                        f"{majority_summary}; {llm_reason}"
                        if llm_reason
                        else majority_summary
                    )
                )
                answer_judge_llm_backed = True
                answer_judge_source = "llm_judge"

        evaluation_rows[question_id] = {
            "question_id": question_id,
            "question": qa_row.get("question", ""),
            "conversation_id": qa_row.get("conversation_id", ""),
            "category": qa_row.get("category"),
            "relation_type": metadata.get("relation_type", ""),
            "relation_subtype": metadata.get("relation_subtype", ""),
            "topic": metadata.get("topic", ""),
            "source": metadata.get("source", ""),
            "persona_id": metadata.get("persona_id", ""),
            "exact_match": exact_match,
            "judge_label": judge_label,
            "judge_reason": judge_reason,
            "final_score": bool(final_score),
            "predicted_answer": predicted_answer,
            "golden_answer": golden_answer,
            "answer_judge_llm_backed": answer_judge_llm_backed,
            "answer_judge_source": answer_judge_source,
        }

    return evaluation_rows


def _breakdown_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _finalize_breakdown(
    buckets: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for key, counts in buckets.items():
        total = counts["total"]
        correct = counts["correct"]
        summary[key] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }
    return dict(sorted(summary.items(), key=lambda item: str(item[0])))


def _score_breakdown(
    evaluation_rows: List[Dict[str, Any]], field_name: str
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    for row in evaluation_rows:
        key = _breakdown_key(_breakdown_value(row, field_name))
        if not key:
            continue
        buckets[key]["total"] += 1
        if row.get("final_score"):
            buckets[key]["correct"] += 1
    return _finalize_breakdown(buckets)


def _score_cross_breakdown(
    evaluation_rows: List[Dict[str, Any]], outer_field: str, inner_field: str
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    buckets: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "correct": 0})
    )
    for row in evaluation_rows:
        outer_key = _breakdown_key(_breakdown_value(row, outer_field))
        inner_key = _breakdown_key(_breakdown_value(row, inner_field))
        if not outer_key or not inner_key:
            continue
        buckets[outer_key][inner_key]["total"] += 1
        if row.get("final_score"):
            buckets[outer_key][inner_key]["correct"] += 1

    return {
        outer_key: _finalize_breakdown(inner_buckets)
        for outer_key, inner_buckets in sorted(
            buckets.items(), key=lambda item: str(item[0])
        )
    }


def _breakdown_value(row: Dict[str, Any], field_name: str) -> Any:
    if (
        field_name == "relation_subtype"
        and _breakdown_key(row.get("relation_type")) == "contradictory"
    ):
        return "contradictory"
    return row.get(field_name)


def build_score_summary(evaluation_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build run-level score summary."""
    total_questions = len(evaluation_rows)
    correct = sum(1 for row in evaluation_rows if row.get("final_score"))

    category_breakdown: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    for row in evaluation_rows:
        category = str(row.get("category"))
        category_breakdown[category]["total"] += 1
        if row.get("final_score"):
            category_breakdown[category]["correct"] += 1

    category_summary = {}
    for category, counts in category_breakdown.items():
        total = counts["total"]
        category_summary[category] = {
            "total": total,
            "correct": counts["correct"],
            "accuracy": counts["correct"] / total if total else 0.0,
        }

    summary = {
        "total_questions": total_questions,
        "correct": correct,
        "accuracy": correct / total_questions if total_questions else 0.0,
        "category_breakdown": category_summary,
    }

    optional_breakdowns = {
        "source_breakdown": _score_breakdown(evaluation_rows, "source"),
        "relation_type_breakdown": _score_breakdown(
            evaluation_rows, "relation_type"
        ),
        "relation_subtype_breakdown": _score_breakdown(
            evaluation_rows, "relation_subtype"
        ),
        "topic_breakdown": _score_breakdown(evaluation_rows, "topic"),
        "source_relation_type_breakdown": _score_cross_breakdown(
            evaluation_rows, "source", "relation_type"
        ),
        "source_relation_subtype_breakdown": _score_cross_breakdown(
            evaluation_rows, "source", "relation_subtype"
        ),
    }
    summary.update(
        {
            key: breakdown
            for key, breakdown in optional_breakdowns.items()
            if breakdown
        }
    )
    return summary
