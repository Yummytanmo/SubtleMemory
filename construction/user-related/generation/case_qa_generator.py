from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from infra.config import GENERATION_MODEL
from infra.logging_utils import utc_now_iso
from core.parallel_utils import map_ordered, normalize_concurrency
from prompts import (
    build_case_answer_prompt,
    build_case_question_prompt,
    build_case_task_answer_prompt,
    build_case_task_prompt,
)
from core.schemas import AcceptedCase, CaseQA, ConversationSession, QAAnswer, QAQuestion, QAMode, SanitizedPersonaProfile, TaskForm
from core.validators import parse_json_after_output


DEFAULT_CASE_QA_QUESTION_COUNT = 2
DEFAULT_CASE_QA_CORRECT_ANSWER_COUNT = 3
DEFAULT_CASE_QA_INCORRECT_ANSWER_COUNT = 3
DEFAULT_CASE_QA_QUESTION_VALIDATION_ATTEMPTS = 3
DEFAULT_CASE_QA_ANSWER_VALIDATION_ATTEMPTS = 3


@dataclass
class CaseQAGenerationJob:
    accepted_case: AcceptedCase
    persona: SanitizedPersonaProfile
    sessions: list[ConversationSession]
    qa_mode: QAMode
    same_topic_sibling_cases: list[dict[str, Any]] = field(default_factory=list)


def generate_case_qa(
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
    concurrency: int = 1,
    checkpoint_path: Path | None = None,
    logger: Any | None = None,
    question_count: int = DEFAULT_CASE_QA_QUESTION_COUNT,
    correct_answer_count: int = DEFAULT_CASE_QA_CORRECT_ANSWER_COUNT,
    incorrect_answer_count: int = DEFAULT_CASE_QA_INCORRECT_ANSWER_COUNT,
    qa_mode: QAMode | str = QAMode.QUESTION,
) -> list[CaseQA]:
    qa_mode = QAMode(qa_mode)
    sessions_by_case_id: dict[str, list[ConversationSession]] = {}
    for session in conversations:
        sessions_by_case_id.setdefault(session.case_id, []).append(session)

    jobs = [
        CaseQAGenerationJob(
            accepted_case=accepted,
            persona=personas[accepted.case.persona_id],
            sessions=sessions_by_case_id.get(accepted.case.case_id, []),
            qa_mode=qa_mode,
            same_topic_sibling_cases=build_same_topic_sibling_case_summaries(accepted, accepted_cases)
            if qa_mode == QAMode.TASK
            else [],
        )
        for accepted in accepted_cases
        if case_has_full_session_coverage(accepted, sessions_by_case_id.get(accepted.case.case_id, []))
    ]
    checkpoint = CaseQACheckpoint(checkpoint_path, jobs)
    cached_count = checkpoint.cached_count()
    if logger is not None and checkpoint_path is not None:
        logger.info(
            "case_qa_generation",
            "case_qa_checkpoint_loaded",
            checkpoint_path=str(checkpoint_path),
            counts={"cached_case_qa": cached_count, "total_jobs": len(jobs)},
        )

    pending_jobs = [job for job in jobs if not checkpoint.has(job)]
    if pending_jobs:
        map_ordered(
            pending_jobs,
            lambda job: generate_and_checkpoint_case_qa(
                job,
                llm_client,
                checkpoint,
                generation_model=generation_model,
                question_count=question_count,
                correct_answer_count=correct_answer_count,
                incorrect_answer_count=incorrect_answer_count,
            ),
            max_workers=normalize_concurrency(concurrency),
        )

    case_qa = checkpoint.ordered_case_qa()
    if len(case_qa) != len(jobs):
        raise RuntimeError(f"case QA generation checkpoint incomplete: {len(case_qa)}/{len(jobs)} records")
    if logger is not None and checkpoint_path is not None:
        logger.info(
            "case_qa_generation",
            "case_qa_checkpoint_complete",
            checkpoint_path=str(checkpoint_path),
            counts={"cached_case_qa": cached_count, "generated_case_qa": len(pending_jobs), "total_case_qa": len(case_qa)},
        )
    return case_qa


def case_has_full_session_coverage(accepted_case: AcceptedCase, sessions: list[ConversationSession]) -> bool:
    covered_fact_ids = {session.fact_id for session in sessions}
    expected_fact_ids = {fact.fact_id for fact in accepted_case.case.facts}
    return bool(expected_fact_ids) and expected_fact_ids <= covered_fact_ids


def build_same_topic_sibling_case_summaries(
    accepted_case: AcceptedCase,
    accepted_cases: list[AcceptedCase],
) -> list[dict[str, Any]]:
    current = accepted_case.case
    return [
        compact_same_topic_case_summary(candidate.case)
        for candidate in accepted_cases
        if candidate.case.case_id != current.case_id
        and candidate.case.persona_id == current.persona_id
        and candidate.case.topic_preference == current.topic_preference
    ]


def compact_same_topic_case_summary(memory_case: Any) -> dict[str, Any]:
    return {
        "case_id": memory_case.case_id,
        "relation_type": memory_case.relation_type,
        "relation_subtype": memory_case.relation_subtype,
        "description": memory_case.description,
        "fact_texts": [fact.text for fact in memory_case.facts],
    }


class CaseQACheckpoint:
    def __init__(self, path: Path | None, jobs: list[CaseQAGenerationJob]) -> None:
        self.path = path
        self.job_keys = [case_qa_job_key(job) for job in jobs]
        self.case_qa_by_key = load_case_qa_checkpoint(path)
        self._lock = Lock()

    def has(self, job: CaseQAGenerationJob) -> bool:
        case_qa = self.case_qa_by_key.get(case_qa_job_key(job))
        if case_qa is None:
            return False
        if QAMode(job.qa_mode) == QAMode.TASK and any(question.task_form is None for question in case_qa.questions):
            return False
        return True

    def cached_count(self) -> int:
        return sum(1 for key in self.job_keys if key in self.case_qa_by_key)

    def record(self, job: CaseQAGenerationJob, case_qa: CaseQA) -> None:
        key = case_qa_job_key(job)
        with self._lock:
            self.case_qa_by_key[key] = case_qa
            self.flush_locked()

    def ordered_case_qa(self) -> list[CaseQA]:
        return [
            self.case_qa_by_key[key]
            for key in self.job_keys
            if key in self.case_qa_by_key
        ]

    def flush_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [case_qa.model_dump(mode="json") for case_qa in self.ordered_case_qa()]
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(self.path)


def load_case_qa_checkpoint(path: Path | None) -> dict[str, CaseQA]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"case QA checkpoint must be a JSON array: {path}")
    case_qa_by_key: dict[str, CaseQA] = {}
    for item in payload:
        case_qa = CaseQA.model_validate(item)
        key = case_qa_case_key(case_qa)
        if key not in case_qa_by_key:
            case_qa_by_key[key] = case_qa
    return case_qa_by_key


def generate_and_checkpoint_case_qa(
    job: CaseQAGenerationJob,
    llm_client: Any,
    checkpoint: CaseQACheckpoint,
    *,
    generation_model: str,
    question_count: int,
    correct_answer_count: int,
    incorrect_answer_count: int,
) -> CaseQA:
    case_qa = generate_case_qa_for_job(
        job,
        llm_client,
        generation_model=generation_model,
        question_count=question_count,
        correct_answer_count=correct_answer_count,
        incorrect_answer_count=incorrect_answer_count,
    )
    checkpoint.record(job, case_qa)
    return case_qa


def generate_case_qa_for_job(
    job: CaseQAGenerationJob,
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
    question_count: int = DEFAULT_CASE_QA_QUESTION_COUNT,
    correct_answer_count: int = DEFAULT_CASE_QA_CORRECT_ANSWER_COUNT,
    incorrect_answer_count: int = DEFAULT_CASE_QA_INCORRECT_ANSWER_COUNT,
) -> CaseQA:
    qa_id = make_case_qa_id(job.accepted_case.case.case_id)
    case_payload = job.accepted_case.case.model_dump(mode="json")
    sessions_payload = [session.model_dump(mode="json") for session in job.sessions]
    if job.qa_mode == QAMode.TASK:
        question_prompt = build_case_task_prompt(
            job.persona.persona_str,
            case_payload,
            sessions_payload,
            question_count=question_count,
            same_topic_sibling_cases=job.same_topic_sibling_cases,
        )
    else:
        question_prompt = build_case_question_prompt(
            job.persona.persona_str,
            case_payload,
            sessions_payload,
            question_count=question_count,
        )
    questions, question_result = generate_qa_questions_with_validation_retry(
        qa_id,
        job.accepted_case.case.case_id,
        llm_client,
        question_prompt,
        question_count=question_count,
        require_task_form=job.qa_mode == QAMode.TASK,
    )
    if job.qa_mode == QAMode.TASK:
        answer_prompt = build_case_task_answer_prompt(
            job.persona.persona_str,
            case_payload,
            sessions_payload,
            [
                {"question_id": item.question_id, "question": item.question, "task_form": item.task_form}
                for item in questions
            ],
            correct_count=correct_answer_count,
            incorrect_count=incorrect_answer_count,
        )
    else:
        answer_prompt = build_case_answer_prompt(
            job.persona.persona_str,
            case_payload,
            sessions_payload,
            [{"question_id": item.question_id, "question": item.question} for item in questions],
            correct_count=correct_answer_count,
            incorrect_count=incorrect_answer_count,
        )
    questions_with_answers, answer_result = generate_qa_answers_with_validation_retry(
        job.accepted_case.case.case_id,
        llm_client,
        answer_prompt,
        questions,
        correct_answer_count=correct_answer_count,
        incorrect_answer_count=incorrect_answer_count,
    )
    return CaseQA(
        qa_id=qa_id,
        persona_id=job.accepted_case.case.persona_id,
        case_id=job.accepted_case.case.case_id,
        topic_preference=job.accepted_case.case.topic_preference,
        relation_type=job.accepted_case.case.relation_type,
        relation_subtype=job.accepted_case.case.relation_subtype,
        qa_mode=job.qa_mode,
        session_ids=[session.conversation_id for session in job.sessions],
        questions=questions_with_answers,
        generation_model=generation_model,
        stream_completed=question_result.stream_completed and answer_result.stream_completed,
        created_at=utc_now_iso(),
    )


def generate_qa_questions_with_validation_retry(
    qa_id: str,
    case_id: str,
    llm_client: Any,
    question_prompt: str,
    *,
    question_count: int,
    require_task_form: bool = False,
    max_attempts: int = DEFAULT_CASE_QA_QUESTION_VALIDATION_ATTEMPTS,
) -> tuple[list[QAQuestion], Any]:
    attempts = max(1, max_attempts)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        question_result = llm_client.stream_chat(
            [{"role": "user", "content": question_prompt}],
            phase="case_qa_question_generation",
            temperature=0.3,
        )
        if not question_result.stream_completed:
            raise RuntimeError(f"case QA question generation failed for {case_id}: {question_result.error}")
        try:
            question_payload = parse_json_after_output(question_result.text)
            questions = build_qa_questions_without_answers(
                qa_id,
                question_payload,
                question_count=question_count,
                require_task_form=require_task_form,
            )
            return questions, question_result
        except ValueError as exc:
            last_error = exc
            if attempt == attempts:
                break

    raise ValueError(f"case QA question generation failed validation after {attempts} attempts: {last_error}")


def generate_qa_answers_with_validation_retry(
    case_id: str,
    llm_client: Any,
    answer_prompt: str,
    questions: list[QAQuestion],
    *,
    correct_answer_count: int,
    incorrect_answer_count: int,
    max_attempts: int = DEFAULT_CASE_QA_ANSWER_VALIDATION_ATTEMPTS,
) -> tuple[list[QAQuestion], Any]:
    attempts = max(1, max_attempts)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        answer_result = llm_client.stream_chat(
            [{"role": "user", "content": answer_prompt}],
            phase="case_qa_answer_generation",
            temperature=0.2,
        )
        if not answer_result.stream_completed:
            raise RuntimeError(f"case QA answer generation failed for {case_id}: {answer_result.error}")
        try:
            answer_payload = parse_json_after_output(answer_result.text)
            questions_with_answers = attach_answers_to_questions(
                questions,
                answer_payload,
                correct_answer_count=correct_answer_count,
                incorrect_answer_count=incorrect_answer_count,
            )
            return questions_with_answers, answer_result
        except ValueError as exc:
            last_error = exc
            if attempt == attempts:
                break

    raise ValueError(f"case QA answer generation failed validation after {attempts} attempts: {last_error}")


def build_qa_questions_without_answers(
    qa_id: str,
    payload: Any,
    *,
    question_count: int,
    require_task_form: bool = False,
) -> list[QAQuestion]:
    items = extract_question_items(payload)
    if len(items) < question_count:
        raise ValueError(f"case QA question generation returned {len(items)} questions, expected {question_count}")
    questions: list[QAQuestion] = []
    for index, item in enumerate(items[:question_count]):
        question_text = extract_question_text(item)
        task_form = extract_task_form(item)
        if require_task_form and task_form is None:
            raise ValueError("task QA question generation returned a question without task_form")
        question_id = f"{qa_id}-q-{index}"
        questions.append(
            QAQuestion(
                question_id=question_id,
                question=question_text,
                task_form=task_form,
                correct_answers=[QAAnswer(answer_id=f"{question_id}-correct-placeholder", text="placeholder")],
                incorrect_answers=[QAAnswer(answer_id=f"{question_id}-incorrect-placeholder", text="placeholder")],
            )
        )
    return questions


def attach_answers_to_questions(
    questions: list[QAQuestion],
    payload: Any,
    *,
    correct_answer_count: int,
    incorrect_answer_count: int,
) -> list[QAQuestion]:
    items = extract_answer_items(payload)
    items_by_question_id = {
        str(item.get("question_id") or "").strip(): item
        for item in items
        if isinstance(item, dict)
    }
    output: list[QAQuestion] = []
    for index, question in enumerate(questions):
        item = items_by_question_id.get(question.question_id)
        if item is None and index < len(items) and isinstance(items[index], dict):
            item = items[index]
        if item is None:
            raise ValueError(f"case QA answer generation missing answers for {question.question_id}")
        correct_answers = build_answers(
            question.question_id,
            item.get("correct_answers") or item.get("correct") or [],
            answer_type="correct",
            expected_count=correct_answer_count,
        )
        incorrect_answers = build_answers(
            question.question_id,
            item.get("incorrect_answers") or item.get("incorrect") or [],
            answer_type="incorrect",
            expected_count=incorrect_answer_count,
        )
        output.append(
            QAQuestion(
                question_id=question.question_id,
                question=question.question,
                task_form=question.task_form,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
            )
        )
    return output


def extract_question_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("questions", "candidate_questions", "queries"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("case QA question payload does not contain questions")


def extract_answer_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("answers", "question_answers", "questions"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("case QA answer payload does not contain answers")


def extract_question_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("question") or item.get("query") or "").strip()
    return str(item).strip()


def extract_task_form(item: Any) -> TaskForm | None:
    if not isinstance(item, dict):
        return None
    raw = str(item.get("task_form") or item.get("form") or item.get("task_type") or "").strip()
    if not raw:
        return None
    try:
        return TaskForm(raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in TaskForm)
        raise ValueError(f"invalid task_form {raw!r}; expected one of: {allowed}") from exc


def build_answers(
    question_id: str,
    raw_answers: Any,
    *,
    answer_type: str,
    expected_count: int,
) -> list[QAAnswer]:
    if isinstance(raw_answers, str):
        raw_items = [raw_answers]
    elif isinstance(raw_answers, list):
        raw_items = raw_answers
    else:
        raw_items = []
    if len(raw_items) < expected_count:
        raise ValueError(f"{question_id} returned {len(raw_items)} {answer_type} answers, expected {expected_count}")
    answers: list[QAAnswer] = []
    for index, item in enumerate(raw_items[:expected_count]):
        text = str(item.get("text") or item.get("answer") if isinstance(item, dict) else item).strip()
        answers.append(QAAnswer(answer_id=f"{question_id}-{answer_type}-{index}", text=text))
    return answers


def make_case_qa_id(case_id: str) -> str:
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:12]
    return f"case-qa-{digest}"


def case_qa_job_key(job: CaseQAGenerationJob) -> str:
    return f"{QAMode(job.qa_mode).value}:{job.accepted_case.case.case_id}"


def case_qa_case_key(case_qa: CaseQA) -> str:
    return f"{QAMode(case_qa.qa_mode).value}:{case_qa.case_id}"
