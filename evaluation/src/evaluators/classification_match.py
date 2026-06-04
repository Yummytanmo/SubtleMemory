"""
Classification match evaluator - deterministic relation-label matching.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional dependency
    repair_json = None


VALID_PAIRS = {
    ("complementary", "K=1"),
    ("complementary", "K>1"),
    ("complementary", "any_one"),
    ("nuanced", "Temporal"),
    ("nuanced", "Context"),
    ("contradictory", "contradictory"),
}

LEGACY_CONTRADICTORY_SUBTYPES = {
    "a_user_vs_user",
    "b_user_vs_non_user",
    "c_non_user_vs_non_user",
}

FINAL_CHANNEL_MARKER = "<|start|>assistant<|channel|>final<|message|>"
REQUIRED_LABEL_KEYS = {"relation_type", "relation_subtype"}


@register_evaluator("classification_match")
class ClassificationMatch(BaseEvaluator):
    """Score generated relation classifications by direct label matching."""

    def __init__(self, config: dict):
        super().__init__(config or {})
        self.canonicalize_contradictory_subtypes = bool(
            self.config.get("canonicalize_contradictory_subtypes", True)
        )

    async def evaluate(
        self, answer_results: List[AnswerResult]
    ) -> EvaluationResult:
        print(f"\n{'=' * 60}")
        print("Evaluation: Classification Match")
        print(
            "  - Canonicalize contradictory subtypes: "
            f"{self.canonicalize_contradictory_subtypes}"
        )
        print(f"{'=' * 60}")

        detailed_results = []
        total_correct = 0
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for answer_result in answer_results:
            detail = self._evaluate_single_answer(answer_result)
            detailed_results.append(detail)

            if detail["is_correct"]:
                total_correct += 1

            category = detail.get("category")
            if category is not None:
                category_stats[str(category)]["total"] += 1
                if detail["is_correct"]:
                    category_stats[str(category)]["correct"] += 1

        accuracy = total_correct / len(answer_results) if answer_results else 0.0
        category_accuracies = {
            category: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": (
                    stats["correct"] / stats["total"] if stats["total"] else 0.0
                ),
            }
            for category, stats in sorted(category_stats.items())
        }

        print("\n✅ Evaluation complete:")
        print(f"   - Total questions: {len(answer_results)}")
        print(f"   - Correct: {total_correct}")
        print(f"   - Accuracy: {accuracy:.2%}")

        return EvaluationResult(
            total_questions=len(answer_results),
            correct=total_correct,
            accuracy=accuracy,
            detailed_results=detailed_results,
            metadata={
                "evaluator": "classification_match",
                "valid_pairs": [list(pair) for pair in sorted(VALID_PAIRS)],
                "canonicalize_contradictory_subtypes": (
                    self.canonicalize_contradictory_subtypes
                ),
                "category_accuracies": category_accuracies,
            },
        )

    def _evaluate_single_answer(self, answer_result: AnswerResult) -> Dict[str, Any]:
        metadata = answer_result.metadata or {}
        expected_type = self._normalize_relation_type(
            metadata.get("relation_type") or answer_result.category
        )
        expected_subtype = self._normalize_relation_subtype(
            metadata.get("relation_subtype"),
            relation_type=expected_type,
        )
        expected_pair = (expected_type, expected_subtype)

        parsed, parse_error = self._parse_generated_answer(answer_result.answer)
        predicted_type = ""
        predicted_subtype = ""
        classification_reason = ""
        invalid_pair = False

        if parsed is not None:
            predicted_type = self._normalize_relation_type(parsed.get("relation_type"))
            predicted_subtype = self._normalize_relation_subtype(
                parsed.get("relation_subtype"),
                relation_type=predicted_type,
            )
            classification_reason = self._clean_reason(parsed.get("reason"))
            invalid_pair = (predicted_type, predicted_subtype) not in VALID_PAIRS

        expected_valid = expected_pair in VALID_PAIRS
        predicted_pair = (predicted_type, predicted_subtype)
        is_correct = (
            parse_error is None
            and expected_valid
            and not invalid_pair
            and predicted_pair == expected_pair
        )

        judge_reason = self._build_judge_reason(
            expected_pair=expected_pair,
            predicted_pair=predicted_pair,
            parse_error=parse_error,
            expected_valid=expected_valid,
            invalid_pair=invalid_pair,
            classification_reason=classification_reason,
        )

        return {
            "question_id": answer_result.question_id,
            "question": answer_result.question,
            "golden_answer": answer_result.golden_answer,
            "generated_answer": answer_result.answer,
            "category": answer_result.category,
            "is_correct": is_correct,
            "expected_relation_type": expected_type,
            "expected_relation_subtype": expected_subtype,
            "predicted_relation_type": predicted_type,
            "predicted_relation_subtype": predicted_subtype,
            "classification_reason": classification_reason,
            "parse_error": parse_error or "",
            "invalid_pair": invalid_pair,
            "judge_reason": judge_reason,
            "answer_judge_source": "classification_match",
        }

    def _parse_generated_answer(
        self, generated_answer: Any
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        text = str(generated_answer or "").strip()
        if not text:
            return None, "empty_answer"

        json_text = self._extract_json_object(text)
        if not json_text:
            return None, "json_not_found"

        parsed, parse_error = self._loads_json_object(json_text)
        if parse_error:
            return None, parse_error

        if not isinstance(parsed, dict):
            return None, "json_not_object"

        if "relation_type" not in parsed or "relation_subtype" not in parsed:
            return parsed, "missing_required_label"

        return parsed, None

    def _loads_json_object(
        self, json_text: str
    ) -> tuple[Optional[Any], Optional[str]]:
        try:
            return json.loads(json_text), None
        except json.JSONDecodeError:
            if repair_json is None:
                return None, "json_decode_error"

        try:
            repaired = repair_json(json_text)
            return json.loads(repaired), None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None, "json_decode_error"

    def _extract_json_object(self, text: str) -> str:
        final_text = self._extract_final_channel_text(text)
        search_texts = [final_text]
        if final_text != text:
            search_texts.append(text)

        for require_label_keys in (True, False):
            for search_text in search_texts:
                objects = self._find_json_objects(
                    search_text, require_label_keys=require_label_keys
                )
                if objects:
                    return objects[-1]

        return ""

    def _extract_final_channel_text(self, text: str) -> str:
        if FINAL_CHANNEL_MARKER not in text:
            return text
        return text.rsplit(FINAL_CHANNEL_MARKER, maxsplit=1)[-1]

    def _find_json_objects(
        self, text: str, *, require_label_keys: bool
    ) -> List[str]:
        objects: List[str] = []
        for candidate in self._iter_balanced_json_candidates(text):
            parsed, parse_error = self._loads_json_object(candidate)
            if parse_error or not isinstance(parsed, dict):
                continue
            if require_label_keys and not REQUIRED_LABEL_KEYS.issubset(parsed):
                continue
            objects.append(candidate)
        return objects

    def _iter_balanced_json_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        start: Optional[int] = None
        depth = 0
        in_string = False
        escape = False

        for index, char in enumerate(text):
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : index + 1].strip())
                    start = None

        return candidates

    def _normalize_relation_type(self, raw_value: Any) -> str:
        value = " ".join(str(raw_value or "").split()).strip().lower()
        if value in {"complementary", "nuanced", "contradictory"}:
            return value
        return value

    def _normalize_relation_subtype(
        self, raw_value: Any, *, relation_type: str = ""
    ) -> str:
        value = " ".join(str(raw_value or "").split()).strip()
        if (
            self.canonicalize_contradictory_subtypes
            and relation_type == "contradictory"
            and value in LEGACY_CONTRADICTORY_SUBTYPES
        ):
            return "contradictory"

        lookup = {
            "k=1": "K=1",
            "k>1": "K>1",
            "any_one": "any_one",
            "temporal": "Temporal",
            "context": "Context",
            "contradictory": "contradictory",
        }
        return lookup.get(value.lower(), value)

    def _clean_reason(self, reason: Any, limit: int = 240) -> str:
        text = " ".join(str(reason or "").split()).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "..."

    def _build_judge_reason(
        self,
        *,
        expected_pair: tuple[str, str],
        predicted_pair: tuple[str, str],
        parse_error: Optional[str],
        expected_valid: bool,
        invalid_pair: bool,
        classification_reason: str,
    ) -> str:
        expected_text = f"{expected_pair[0]}/{expected_pair[1]}"
        predicted_text = f"{predicted_pair[0]}/{predicted_pair[1]}"

        if parse_error:
            return f"Classification output could not be scored: {parse_error}."
        if not expected_valid:
            return f"Gold classification is not a valid configured pair: {expected_text}."
        if invalid_pair:
            return f"Predicted classification is not a valid pair: {predicted_text}."
        if predicted_pair == expected_pair:
            return f"Predicted classification matches gold: {expected_text}."
        if classification_reason:
            return (
                f"Predicted {predicted_text}, expected {expected_text}. "
                f"Model reason: {classification_reason}"
            )
        return f"Predicted {predicted_text}, expected {expected_text}."
