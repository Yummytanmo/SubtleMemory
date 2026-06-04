"""
LLM Judge evaluator - use LLM to judge answer correctness.

Aligned with evaluation_archive logic:
- Keep independent judgments for each run (judgment_1, judgment_2, judgment_3)
- Calculate accuracy for each run separately
- Output mean and std
"""

import asyncio
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from openai import AsyncOpenAI
from tqdm import tqdm

from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator
from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.utils.prompts import get_prompt, format_prompt, get_prompt_metadata


RELATION_TYPE_GUIDANCE = {
    "complementary": (
        "The memory items are jointly valid "
        "Judge whether the answer correctly integrates the compatible evidence "
        "do not penalize answers that comprehensively list facts under different conditions."
    ),
    "nuanced": (
        "The memory items are jointly valid only when target-affecting temporal "
        "or contextual conditions are preserved. Judge whether the answer "
        "selects the memory that matches the relevant condition."
    ),
    "contradictory": (
        "No condition supported by the memory content makes the memory items "
        "jointly valid. Judge whether the answer respects the unresolved "
        "inconsistency instead of merging incompatible memories into one "
        "consistent state."
    ),
    "default": (
        "No additional relation-type guidance is available. Judge only against "
        "the provided facts, case, accepted correct answers, known incorrect "
        "answers, generated answer, and any available relation metadata."
    ),
}

RELATION_SUBTYPE_GUIDANCE = {
    "K=1": (
        "One memory item is decisive for the target while other items provide "
        "compatible background. Do not mark an answer correct just because it "
        "mentions background if it misses the decisive point."
    ),
    "K>1": (
        "Multiple compatible memory items must be combined to answer the target. "
        "Do not treat condition, location, time, or scope differences as "
        "contradictions. Mark CORRECT only when all required target facts or "
        "condition-specific values are present. Mark WRONG when a required fact, "
        "constraint, or evidence role is omitted, even if the selected option is "
        "right."
    ),
    "any_one": (
        "Any one of multiple compatible memory items is sufficient to support "
        "the same target. Do not mark an answer incorrect only because it cites "
        "one valid supporting path instead of another."
    ),
    "Temporal": (
        "Time determines which memory applies. Judge against the answer that "
        "matches the relevant time rather than a timeless average or a "
        "conflicting time period."
    ),
    "Context": (
        "Context such as role, task, scope, location, version, definition, or "
        "attribute determines which memory applies. Judge against the answer "
        "that matches the relevant context."
    ),
    "contradictory": (
        "The contradictory subtype means the memories remain irreconcilable "
        "under supported conditions. Do not accept answers that smooth over the "
        "conflict unless the references explicitly support that."
    ),
    "non_persona_contradiction": (
        "This subtype describes a factual or non-persona contradiction pattern. "
        "Do not treat words such as user/user-vs-non-user inside the subtype "
        "label as persona evidence; judge whether the answer respects the "
        "unresolved factual inconsistency under the provided references."
    ),
    "default": (
        "No additional relation-subtype guidance is available. Judge only "
        "against the provided facts, case, accepted correct answers, known "
        "incorrect answers, generated answer, and any available relation "
        "metadata."
    ),
}

RELATION_SUBTYPE_ALIASES = {
    "K=1": "K=1",
    "K>1": "K>1",
    "any_one": "any_one",
    "Temporal": "Temporal",
    "Context": "Context",
    "contradictory": "contradictory",
    "a_user_vs_user": "non_persona_contradiction",
    "b_user_vs_non_user": "non_persona_contradiction",
    "c_non_user_vs_non_user": "non_persona_contradiction",
}

SOURCE_GUIDANCE = {
    "user-related": (
        "This is a user-related memory question. The case, facts, references, "
        "and provided persona context are evidence for judging user preferences, "
        "status, habits, identity, or contextual state. Do not invent persona "
        "facts beyond the provided case, facts, and references."
    ),
    "user-unrelated": (
        "This is not a persona/user-related grading case. Do not interpret the "
        "word user inside relation_subtype labels as persona evidence. Judge "
        "only by the facts, case, references, and relation semantics."
    ),
    "default": (
        "Source is missing or unknown. Judge neutrally using the provided "
        "facts, case, references, and relation semantics; do not infer persona "
        "context."
    ),
}

GUIDANCE_SCHEMA_VERSION = "relation_source_guidance_v1"


def _config_or_env(config: Dict[str, Any], key: str, env_key: str, default: Any) -> Any:
    if key in config and config[key] is not None:
        return config[key]
    env_value = os.environ.get(env_key)
    if env_value is not None:
        return env_value
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


@register_evaluator("llm_judge")
class LLMJudge(BaseEvaluator):
    """LLM judge evaluator."""

    PROMPT_SCHEMA_VERSION = "subtlememory_llm_judge_v3_relation_source_reason"
    GUIDANCE_SCHEMA_VERSION = "relation_source_guidance_v1"

    def __init__(self, config: dict):
        super().__init__(config)

        # Initialize OpenAI client
        llm_config = config.get("llm", {})
        self.client = AsyncOpenAI(
            api_key=(
                llm_config.get("api_key")
                or os.environ.get("JUDGE_LLM_API_KEY")
                or os.environ.get("LLM_API_KEY")
            ),
            base_url=(
                llm_config.get("base_url")
                or os.environ.get("JUDGE_LLM_BASE_URL")
                or os.environ.get("LLM_BASE_URL")
                or "https://api.openai.com/v1"
            ),
        )
        self.model = (
            llm_config.get("model")
            or os.environ.get("JUDGE_LLM_MODEL")
            or os.environ.get("LLM_MODEL")
            or "gpt-4o-mini"
        )
        self.num_runs = config.get("num_runs", 3)
        self.num_workers = int(config.get("num_workers", 32))
        self.checkpoint_interval = int(config.get("checkpoint_interval", 50))
        self.fail_on_error = _as_bool(config.get("fail_on_error", True))
        self.max_retries = max(
            1,
            int(
                config.get(
                    "max_retries",
                    _config_or_env(
                        llm_config,
                        "max_retries",
                        "JUDGE_LLM_MAX_RETRIES",
                        3,
                    ),
                )
            ),
        )
        self.retry_base_seconds = max(
            0.0,
            float(
                config.get(
                    "retry_base_seconds",
                    _config_or_env(
                        llm_config,
                        "retry_base_seconds",
                        "JUDGE_LLM_RETRY_BASE_SECONDS",
                        1,
                    ),
                )
            ),
        )
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(
                config.get(
                    "retry_max_seconds",
                    _config_or_env(
                        llm_config,
                        "retry_max_seconds",
                        "JUDGE_LLM_RETRY_MAX_SECONDS",
                        8,
                    ),
                )
            ),
        )
        self.checkpoint_manager = None

    def set_checkpoint_manager(self, checkpoint_manager: Any) -> None:
        """Attach the pipeline checkpoint manager for per-question resume."""
        self.checkpoint_manager = checkpoint_manager

    async def evaluate(self, answer_results: List[AnswerResult]) -> EvaluationResult:
        """
        Evaluate answers using LLM, return statistics from multiple runs.

        Args:
            answer_results: List of answer results

        Returns:
            Evaluation result with mean and std
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluation: LLM Judge (model={self.model}, runs={self.num_runs})")
        print(f"{'=' * 60}")

        completed_results: Dict[str, Dict[str, Any]] = {}
        if self.checkpoint_manager:
            completed_results = self.checkpoint_manager.load_evaluation_progress()
        current_question_ids = {answer_result.question_id for answer_result in answer_results}
        completed_results = {
            question_id: result
            for question_id, result in completed_results.items()
            if question_id in current_question_ids
        }

        pending_answer_results = [
            answer_result
            for answer_result in answer_results
            if answer_result.question_id not in completed_results
        ]
        if completed_results:
            print(f"Already evaluated: {len(completed_results)} questions")
            print(f"Remaining: {len(pending_answer_results)} questions")

        # Evaluate remaining answers concurrently
        semaphore = asyncio.Semaphore(self.num_workers)  # Limit concurrency
        checkpoint_lock = asyncio.Lock()
        completed_since_checkpoint = 0

        # Use tqdm progress bar
        pbar = tqdm(
            total=len(answer_results),
            initial=len(completed_results),
            desc="⚖️  Evaluate Progress",
            unit="qa",
        )

        async def evaluate_single(answer_result: AnswerResult):
            nonlocal completed_since_checkpoint
            async with semaphore:
                result = await self._evaluate_single_answer(answer_result)
                async with checkpoint_lock:
                    completed_results[answer_result.question_id] = result
                    completed_since_checkpoint += 1
                    should_checkpoint = (
                        self.checkpoint_manager
                        and self.checkpoint_interval > 0
                        and completed_since_checkpoint >= self.checkpoint_interval
                    )
                    if should_checkpoint:
                        self.checkpoint_manager.save_evaluation_progress(
                            completed_results
                        )
                        completed_since_checkpoint = 0
                    pbar.update(1)  # Update progress bar
                return result

        tasks = [evaluate_single(ar) for ar in pending_answer_results]
        if tasks:
            await asyncio.gather(*tasks)
        if self.checkpoint_manager and pending_answer_results:
            self.checkpoint_manager.save_evaluation_progress(completed_results)

        # Close progress bar
        pbar.close()

        detailed_results = [
            completed_results[answer_result.question_id]
            for answer_result in answer_results
            if answer_result.question_id in completed_results
        ]

        # Calculate accuracy for each run separately
        run_scores = []
        category_stats = defaultdict(
            lambda: {"correct": [0] * self.num_runs, "total": 0}
        )

        for i in range(self.num_runs):
            judgment_key = f"judgment_{i + 1}"
            correct_count = 0
            total_count = 0

            for result in detailed_results:
                llm_judgments = result.get("llm_judgments", {})
                category = result.get("category")

                if judgment_key in llm_judgments:
                    total_count += 1
                    if llm_judgments[judgment_key]:
                        correct_count += 1
                        if category is not None:
                            category_stats[category]["correct"][i] += 1

                    # Count category total (only need once)
                if i == 0 and category is not None:
                    category_stats[category]["total"] += 1

            if total_count > 0:
                run_accuracy = correct_count / total_count
                run_scores.append(run_accuracy)

        # Calculate statistics
        mean_accuracy = np.mean(run_scores) if run_scores else 0.0
        std_accuracy = np.std(run_scores) if run_scores else 0.0

        # Calculate accuracy for each category
        category_accuracies = {}
        for category, stats in category_stats.items():
            cat_accuracies = []
            for i in range(self.num_runs):
                if stats["total"] > 0:
                    cat_acc = stats["correct"][i] / stats["total"]
                    cat_accuracies.append(cat_acc)

            if cat_accuracies:
                category_accuracies[str(category)] = {
                    "mean": np.mean(cat_accuracies),
                    "std": np.std(cat_accuracies),
                    "individual_runs": cat_accuracies,
                    "total": stats["total"],
                }

        majority_vote_correct = sum(
            1 for result in detailed_results if self._is_majority_correct(result)
        )
        majority_vote_accuracy = (
            majority_vote_correct / len(answer_results) if answer_results else 0.0
        )

        print(f"\n✅ Evaluation complete:")
        print(f"   - Total questions: {len(answer_results)}")
        print(
            f"   - Majority-vote accuracy: {majority_vote_accuracy:.4f} "
            f"({majority_vote_accuracy * 100:.2f}%)"
        )
        print(
            f"   - Mean run accuracy: {mean_accuracy:.4f} ({mean_accuracy * 100:.2f}%)"
        )
        print(f"   - Std deviation: {std_accuracy:.4f}")
        print(f"   - Run accuracies: {[f'{s:.4f}' for s in run_scores]}")

        if category_accuracies:
            print(f"\n📊 Category statistics:")
            for cat, stats in sorted(category_accuracies.items()):
                print(
                    f"   Category {cat}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['total']})"
                )

        # Group by conversation
        grouped_results = self._group_by_conversation(detailed_results)

        return EvaluationResult(
            total_questions=len(answer_results),
            correct=majority_vote_correct,
            accuracy=majority_vote_accuracy,
            detailed_results=grouped_results,
            metadata={
                "model": self.model,
                "num_runs": self.num_runs,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "run_scores": run_scores,
                "majority_vote_correct": majority_vote_correct,
                "majority_vote_accuracy": majority_vote_accuracy,
                "category_accuracies": category_accuracies,
                **self._get_prompt_run_metadata(),
            },
        )

    def _is_majority_correct(self, result: Dict[str, Any]) -> bool:
        llm_judgments = result.get("llm_judgments", {})
        values = [bool(value) for value in llm_judgments.values()]
        if not values:
            return False
        return Counter(values).most_common(1)[0][0]

    def _group_by_conversation(
        self, detailed_results: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Group results by conversation (e.g., locomo_exp_user_0, locomo_exp_user_1, etc.).
        """
        grouped = defaultdict(list)

        for result in detailed_results:
            question_id = result.get("question_id", "")

            # Extract conversation info from question_id
            # Example: "locomo_0_qa0" -> "locomo_exp_user_0"
            # Example: "personamem_5_qa2" -> "personamem_exp_user_5"
            if "_qa" in question_id:
                parts = question_id.split("_qa")
                conv_id = parts[0]  # "locomo_0" or "personamem_5"

                # Convert to evaluation_archive format
                if "_" in conv_id:
                    dataset_name, conv_num = conv_id.rsplit("_", 1)
                    group_key = f"{dataset_name}_exp_user_{conv_num}"
                else:
                    group_key = f"{conv_id}_exp_user_0"
            else:
                # Use default group if format doesn't match
                group_key = "default_group"

            grouped[group_key].append(result)

        return dict(grouped)

    async def _evaluate_single_answer(self, answer_result: AnswerResult) -> dict:
        """
        Evaluate single answer, keep independent judgment for each run.
        """
        question = answer_result.question
        golden_answer = answer_result.golden_answer
        generated_answer = answer_result.answer
        metadata = answer_result.metadata or {}
        correct_answers = self._clean_reference_answers(
            metadata.get("correct_answers", [])
        )
        incorrect_answers = self._clean_reference_answers(
            metadata.get("incorrect_answers", [])
        )
        if not correct_answers and golden_answer:
            correct_answers = [str(golden_answer).strip()]

        # Multiple evaluations, keep independent judgments and brief reasons.
        deterministic_verdict = self._deterministic_reference_verdict(
            generated_answer=generated_answer,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers,
        )

        if deterministic_verdict is not None:
            deterministic_label, deterministic_reason = deterministic_verdict
            judgments = [deterministic_label] * self.num_runs
            judgment_reasons = [deterministic_reason] * self.num_runs
            judge_prompt_metadata = self._build_judge_prompt_metadata(
                metadata=metadata,
                prompt_used=False,
                judge_source="deterministic_reference_bypass",
            )
        else:
            judgments = []
            judgment_reasons = []
            for _ in range(self.num_runs):
                judge_result = await self._judge_answer(
                    question=question,
                    golden_answer=golden_answer,
                    generated_answer=generated_answer,
                    correct_answers=correct_answers,
                    incorrect_answers=incorrect_answers,
                    metadata=metadata,
                )
                is_correct, reason = self._coerce_judge_result(judge_result)
                judgments.append(is_correct)
                judgment_reasons.append(reason)
            judge_prompt_metadata = self._build_judge_prompt_metadata(
                metadata=metadata, prompt_used=True, judge_source="llm_judge"
            )

        # Use judgment_1, judgment_2, ... format
        llm_judgments = {
            f"judgment_{i + 1}": judgment for i, judgment in enumerate(judgments)
        }
        llm_judgment_reasons = {
            f"judgment_{i + 1}": reason
            for i, reason in enumerate(judgment_reasons)
            if reason
        }
        judge_reason = self._select_majority_reason(judgments, judgment_reasons)

        return {
            "question_id": answer_result.question_id,
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": generated_answer,
            "llm_judgments": llm_judgments,
            "llm_judgment_reasons": llm_judgment_reasons,
            "judge_reason": judge_reason,
            "category": answer_result.category,
            "judge_prompt_metadata": judge_prompt_metadata,
        }

    async def _judge_answer(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str,
        correct_answers: Optional[List[str]] = None,
        incorrect_answers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to judge if answer is correct.

        Returns:
            Dict with is_correct and a concise reason.
        """
        correct_refs = correct_answers or (
            [] if not golden_answer else [str(golden_answer)]
        )
        incorrect_refs = incorrect_answers or []

        # Use configured prompts
        system_prompt = get_prompt("llm_judge", "system_prompt")
        user_prompt = format_prompt(
            "llm_judge",
            "user_prompt",
            question=question,
            golden_answer=golden_answer,
            accepted_correct_answers_block=self._format_reference_block(correct_refs),
            known_incorrect_answers_block=self._format_reference_block(
                incorrect_refs,
                empty_placeholder="(no explicit incorrect references provided)",
            ),
            **self._build_case_context_blocks(metadata or {}),
            generated_answer=generated_answer,
        )

        try:
            response = await self._create_completion_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                question=question,
            )
        except RuntimeError as e:
            if self.fail_on_error:
                raise
            print(f"  ⚠️ LLM Judge soft-failed question: {e}")
            return {
                "is_correct": False,
                "reason": f"Judge failed after retries: {e}",
            }

        try:
            content = response.choices[0].message.content

            # Debug: check if content is empty or None
            if not content:
                print(f"  ⚠️ LLM Judge: Empty response from model {self.model}")
                return {
                    "is_correct": False,
                    "reason": "Judge returned an empty response.",
                }

            # Extract JSON from response (handle models that add explanation text)
            json_str = self._extract_json(content)
            if not json_str:
                print(f"  ⚠️ LLM Judge: No JSON found in response")
                print(f"     Raw response: {content[:200]}...")
                return {
                    "is_correct": False,
                    "reason": "Judge response did not contain valid JSON.",
                }

            result = json.loads(json_str)
            label = result.get("label", "")
            if not label:
                print(f"  ⚠️ LLM Judge: No label found in response")
                print(f"     Raw response: {content}...")
                return {
                    "is_correct": False,
                    "reason": "Judge response did not include a label.",
                }

            normalized_label = label.strip().upper()
            reason = self._clean_judge_reason(result.get("reason", ""))
            if not reason:
                reason = "The answer was graded from the provided references."

            return {"is_correct": normalized_label == "CORRECT", "reason": reason}

        except json.JSONDecodeError as e:
            print(f"  ⚠️ LLM Judge JSON parse failed: {e}")
            print(f"     Raw response: {content[:200] if content else 'None'}...")
            return {
                "is_correct": False,
                "reason": "Judge response JSON could not be parsed.",
            }

    async def _create_completion_with_retry(
        self, system_prompt: str, user_prompt: str, question: str = ""
    ):
        last_error = None
        max_retries = max(1, self.max_retries)
        for attempt in range(max_retries):
            try:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    print(
                        "  ❌ LLM Judge failed after "
                        f"{max_retries} attempts: {type(e).__name__}: {e}"
                    )
                    break

                print(
                    "  ⚠️ LLM Judge failed "
                    f"(attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                delay = min(
                    self.retry_max_seconds,
                    self.retry_base_seconds * (2**attempt),
                )
                if delay > 0:
                    await asyncio.sleep(delay)

        raise RuntimeError(
            "LLM Judge failed after "
            f"{max_retries} attempts for question: {question[:120]}"
        ) from last_error

    def _extract_json(self, content: str) -> str:
        """
        Extract JSON from LLM response that may contain explanation text.

        Handles:
        1. Pure JSON: {"label": "CORRECT", "reason": "..."}
        2. JSON with explanation: Some text... {"label": "CORRECT", "reason": "..."}
        3. Markdown code block: ```json {"label": "CORRECT", "reason": "..."} ```
        """
        import re

        # Try 1: Extract from markdown code block
        code_block_match = re.search(
            r"```(?:json)?\s*(\{[^`]*\})\s*```", content, re.DOTALL
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try 2: Find JSON object pattern
        json_match = re.search(r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}', content)
        if json_match:
            return json_match.group(0)

        # Try 3: Return original content (let json.loads handle it)
        return content.strip()

    def _normalize_reference_text(self, text: Any) -> str:
        if text is None:
            return ""
        return " ".join(str(text).split()).strip().lower()

    def _clean_reference_answers(self, answers: Any) -> List[str]:
        cleaned: List[str] = []
        for answer in answers or []:
            text = str(answer or "").strip()
            if text:
                cleaned.append(text)
        return cleaned

    def _deterministic_reference_verdict(
        self,
        *,
        generated_answer: Any,
        correct_answers: List[str],
        incorrect_answers: List[str],
    ) -> Optional[tuple[bool, str]]:
        normalized_generated = self._normalize_reference_text(generated_answer)
        if not normalized_generated:
            return None

        normalized_correct = {
            self._normalize_reference_text(answer)
            for answer in correct_answers
            if answer
        }
        if normalized_generated in normalized_correct:
            return (
                True,
                "Generated answer exactly matches an accepted correct reference.",
            )

        normalized_incorrect = {
            self._normalize_reference_text(answer)
            for answer in incorrect_answers
            if answer
        }
        if normalized_generated in normalized_incorrect:
            return (
                False,
                "Generated answer exactly matches a known incorrect reference.",
            )

        return None

    def _clean_judge_reason(self, reason: Any, limit: int = 240) -> str:
        text = " ".join(str(reason or "").split()).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    def _coerce_judge_result(self, judge_result: Any) -> tuple[bool, str]:
        """Normalize new dict judge results and legacy bool-only test doubles."""
        if isinstance(judge_result, dict):
            return (
                bool(judge_result.get("is_correct")),
                self._clean_judge_reason(judge_result.get("reason", "")),
            )
        return bool(judge_result), ""

    def _select_majority_reason(
        self, judgments: List[bool], judgment_reasons: List[str]
    ) -> str:
        if not judgments:
            return ""
        majority_label = Counter(bool(value) for value in judgments).most_common(1)[0][
            0
        ]
        for judgment, reason in zip(judgments, judgment_reasons):
            if bool(judgment) == majority_label and reason:
                return self._clean_judge_reason(reason)
        return ""

    def _format_reference_block(
        self, answers: List[str], empty_placeholder: str = "(none)"
    ) -> str:
        if not answers:
            return empty_placeholder
        return "\n".join(f"- {answer}" for answer in answers)

    def _format_optional_text(
        self, value: Any, empty_placeholder: str = "(none)"
    ) -> str:
        text = str(value or "").strip()
        return text if text else empty_placeholder

    def _metadata_text(self, metadata: Dict[str, Any], key: str) -> str:
        return str(metadata.get(key) or "").strip()

    def _has_relation_metadata(self, metadata: Dict[str, Any]) -> bool:
        return bool(
            self._metadata_text(metadata, "relation_type")
            or self._metadata_text(metadata, "relation_subtype")
        )

    def _select_relation_type_guidance_key(self, relation_type: Any) -> str:
        relation_type_text = str(relation_type or "").strip()
        if relation_type_text in RELATION_TYPE_GUIDANCE:
            return relation_type_text
        return "default"

    def _normalize_relation_subtype(self, raw_subtype: Any) -> str:
        relation_subtype = str(raw_subtype or "").strip()
        canonical_subtype = RELATION_SUBTYPE_ALIASES.get(
            relation_subtype, relation_subtype
        )
        if canonical_subtype in RELATION_SUBTYPE_GUIDANCE:
            return canonical_subtype
        return "default"

    def _select_source_guidance_key(self, source: Any) -> str:
        source_text = str(source or "").strip()
        if source_text in SOURCE_GUIDANCE:
            return source_text
        return "default"

    def _build_relation_guidance(self, metadata: Dict[str, Any]) -> str:
        if not self._has_relation_metadata(metadata):
            return ""

        relation_type_key = self._select_relation_type_guidance_key(
            metadata.get("relation_type")
        )
        relation_subtype_key = self._normalize_relation_subtype(
            metadata.get("relation_subtype")
        )

        return "\n".join(
            [
                "Relation semantics guidance:",
                (
                    f"- Relation type guidance ({relation_type_key}): "
                    f"{RELATION_TYPE_GUIDANCE[relation_type_key]}"
                ),
                (
                    f"- Relation subtype guidance ({relation_subtype_key}): "
                    f"{RELATION_SUBTYPE_GUIDANCE[relation_subtype_key]}"
                ),
            ]
        )

    def _build_source_guidance(
        self, metadata: Dict[str, Any], has_relation_metadata: bool
    ) -> str:
        source = self._metadata_text(metadata, "source")
        if not source and not has_relation_metadata:
            return ""

        source_key = self._select_source_guidance_key(source)
        return f"Source guidance ({source_key}): {SOURCE_GUIDANCE[source_key]}"

    def _build_persona_context(self, metadata: Dict[str, Any]) -> str:
        if self._metadata_text(metadata, "source") != "user-related":
            return ""

        persona = self._metadata_text(metadata, "persona_str")
        if not persona:
            return ""

        return (
            "Persona context:\n"
            f"{persona}\n"
            "Use only this provided persona context; do not use or infer any "
            "external persona profile."
        )

    def _build_judge_prompt_metadata(
        self, metadata: Dict[str, Any], prompt_used: bool, judge_source: str
    ) -> Dict[str, Any]:
        has_relation_metadata = self._has_relation_metadata(metadata)
        source = self._metadata_text(metadata, "source")
        has_source_guidance = bool(source or has_relation_metadata)

        return {
            "prompt_schema_version": self.PROMPT_SCHEMA_VERSION,
            "guidance_schema_version": self.GUIDANCE_SCHEMA_VERSION,
            "source": source,
            "relation_type": self._metadata_text(metadata, "relation_type"),
            "relation_subtype": self._metadata_text(metadata, "relation_subtype"),
            "relation_type_guidance_key": (
                self._select_relation_type_guidance_key(metadata.get("relation_type"))
                if has_relation_metadata
                else ""
            ),
            "relation_subtype_guidance_key": (
                self._normalize_relation_subtype(metadata.get("relation_subtype"))
                if has_relation_metadata
                else ""
            ),
            "source_guidance_key": (
                self._select_source_guidance_key(source) if has_source_guidance else ""
            ),
            "persona_included": bool(self._build_persona_context(metadata)),
            "prompt_used": prompt_used,
            "judge_source": judge_source,
        }

    def _build_case_context_blocks(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        has_relation_metadata = self._has_relation_metadata(metadata)
        return {
            "case_description_block": self._format_optional_text(metadata.get("case")),
            "facts_block": self._format_reference_block(
                self._clean_reference_answers(metadata.get("facts", [])),
                empty_placeholder="(no facts provided)",
            ),
            "relation_type": self._format_optional_text(metadata.get("relation_type")),
            "relation_subtype": self._format_optional_text(
                metadata.get("relation_subtype")
            ),
            "topic": self._format_optional_text(metadata.get("topic")),
            "persona_str": self._format_optional_text(metadata.get("persona_str")),
            "persona_id": self._format_optional_text(metadata.get("persona_id")),
            "case_id": self._format_optional_text(metadata.get("case_id")),
            "instance_id": self._format_optional_text(metadata.get("instance_id")),
            "source": self._format_optional_text(metadata.get("source")),
            "relation_guidance_block": self._build_relation_guidance(metadata),
            "source_guidance_block": self._build_source_guidance(
                metadata, has_relation_metadata
            ),
            "persona_context_block": self._build_persona_context(metadata),
        }

    def _get_prompt_run_metadata(self) -> Dict[str, Any]:
        metadata = get_prompt_metadata("llm_judge", ("system_prompt", "user_prompt"))
        metadata["prompt_schema_version"] = self.PROMPT_SCHEMA_VERSION
        metadata["guidance_schema_version"] = self.GUIDANCE_SCHEMA_VERSION
        return metadata
