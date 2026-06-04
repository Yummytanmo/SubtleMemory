from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from infra.config import GENERATION_MODEL
from core.conversation_metrics import count_conversation_tokens, count_conversation_turns
from infra.logging_utils import utc_now_iso
from core.parallel_utils import map_ordered, normalize_concurrency
from prompts import (
    CANONICAL_CONVERSATION_TYPES,
    CONVERSATION_TYPE_FLOW_DESCRIPTIONS,
    PERSONA_SIGNAL_LEVEL_GUIDANCE,
    PERSONA_SIGNAL_LEVELS,
    build_conversation_prompt,
)
from core.schemas import (
    AcceptedCase,
    ConversationFilterDecision,
    ConversationSession,
    ConversationType,
    Fact,
    RejectedConversation,
    SanitizedPersonaProfile,
)
from core.validators import normalize_messages, parse_json_after_output, validate_conversation_session


CONVERSATION_TYPE_CANDIDATE_COUNT = 4
TARGET_CONVERSATION_MESSAGE_COUNTS = (10, 12, 14)
DEFAULT_CONVERSATION_VALIDATION_RETRIES = 2


def sample_conversation_types(rng: random.Random, n: int = 4) -> list[str]:
    if n > len(CANONICAL_CONVERSATION_TYPES):
        raise ValueError("cannot sample more conversation types than available")
    return rng.sample(CANONICAL_CONVERSATION_TYPES, n)


def balanced_conversation_type_preferences(job_count: int, rng: random.Random) -> list[str]:
    if job_count <= 0:
        return []
    base_count, remainder = divmod(job_count, len(CANONICAL_CONVERSATION_TYPES))
    preferred_types = list(CANONICAL_CONVERSATION_TYPES) * base_count
    if remainder:
        preferred_types.extend(rng.sample(CANONICAL_CONVERSATION_TYPES, remainder))
    rng.shuffle(preferred_types)
    return preferred_types


def sample_conversation_type_candidates(
    rng: random.Random,
    *,
    preferred_conversation_type: str | None = None,
    n: int = CONVERSATION_TYPE_CANDIDATE_COUNT,
) -> list[str]:
    if n > len(CANONICAL_CONVERSATION_TYPES):
        raise ValueError("cannot sample more conversation types than available")
    if preferred_conversation_type is None:
        return sample_conversation_types(rng, n)
    if preferred_conversation_type not in CANONICAL_CONVERSATION_TYPES:
        raise ValueError(f"unknown preferred conversation type: {preferred_conversation_type}")
    fallback_types = [item for item in CANONICAL_CONVERSATION_TYPES if item != preferred_conversation_type]
    return [preferred_conversation_type, *rng.sample(fallback_types, n - 1)]


def sample_target_message_count(rng: random.Random) -> int:
    return rng.choice(TARGET_CONVERSATION_MESSAGE_COUNTS)


def sample_conversation_type_flows(rng: random.Random, sampled_conversation_types: list[str]) -> dict[str, str]:
    return {
        conversation_type: rng.choice(CONVERSATION_TYPE_FLOW_DESCRIPTIONS[conversation_type])
        for conversation_type in sampled_conversation_types
    }


def sample_persona_signal_level(rng: random.Random) -> str:
    return rng.choice(PERSONA_SIGNAL_LEVELS)


def persona_signal_guidance_for_level(persona_signal_level: str) -> str:
    return PERSONA_SIGNAL_LEVEL_GUIDANCE[persona_signal_level]


def generate_conversations(
    accepted_cases: list[AcceptedCase],
    personas: dict[str, SanitizedPersonaProfile],
    llm_client: Any,
    *,
    rng: random.Random | None = None,
    generation_model: str = GENERATION_MODEL,
    concurrency: int = 1,
) -> list[ConversationSession]:
    rng = rng or random.Random()
    case_jobs = build_conversation_generation_case_jobs(accepted_cases, personas, rng)
    case_results = map_ordered(
        case_jobs,
        lambda job: generate_conversations_for_case(job, llm_client, generation_model=generation_model),
        max_workers=normalize_concurrency(concurrency),
    )
    return [session for sessions in case_results for session in sessions]


def generate_conversations_with_rejections(
    accepted_cases: list[AcceptedCase],
    personas: dict[str, SanitizedPersonaProfile],
    llm_client: Any,
    *,
    rng: random.Random | None = None,
    generation_model: str = GENERATION_MODEL,
    concurrency: int = 1,
    validation_retries: int = 0,
    checkpoint_path: Path | None = None,
    logger: Any | None = None,
) -> tuple[list[ConversationSession], list[RejectedConversation]]:
    rng = rng or random.Random()
    case_jobs = build_conversation_generation_case_jobs(accepted_cases, personas, rng)
    checkpoint = ConversationGenerationCheckpoint(checkpoint_path, case_jobs)
    cached_count = checkpoint.cached_count()
    if logger is not None and checkpoint_path is not None:
        logger.info(
            "conversation_generation",
            "conversation_generation_checkpoint_loaded",
            checkpoint_path=str(checkpoint_path),
            counts={"cached_conversation_jobs": cached_count, "total_jobs": checkpoint.job_count()},
        )

    case_results = map_ordered(
        case_jobs,
        lambda job: generate_conversation_results_for_case(
            job,
            llm_client,
            checkpoint=checkpoint,
            generation_model=generation_model,
            validation_retries=validation_retries,
        ),
        max_workers=normalize_concurrency(concurrency),
    )
    conversations = [session for result in case_results for session in result.conversations]
    rejected = [item for result in case_results for item in result.rejected]
    if checkpoint.completed_count() != checkpoint.job_count():
        raise RuntimeError(f"conversation generation checkpoint incomplete: {checkpoint.completed_count()}/{checkpoint.job_count()} jobs")
    if logger is not None and checkpoint_path is not None:
        logger.info(
            "conversation_generation",
            "conversation_generation_checkpoint_complete",
            checkpoint_path=str(checkpoint_path),
            counts={
                "cached_conversation_jobs": cached_count,
                "generated_conversation_jobs": checkpoint.job_count() - cached_count,
                "total_conversation_jobs": checkpoint.job_count(),
            },
        )
    return conversations, rejected


@dataclass
class ConversationGenerationResult:
    conversation: ConversationSession | None = None
    rejected: RejectedConversation | None = None


@dataclass
class ConversationCaseGenerationResult:
    conversations: list[ConversationSession]
    rejected: list[RejectedConversation]


class ConversationGenerationJob:
    def __init__(
        self,
        *,
        accepted_case: AcceptedCase,
        persona: SanitizedPersonaProfile,
        fact: Fact,
        sampled_conversation_types: list[str],
        preferred_conversation_type: str,
        target_message_count: int,
        candidate_conversation_flows: dict[str, str],
        persona_signal_level: str,
        persona_signal_guidance: str,
    ) -> None:
        self.accepted_case = accepted_case
        self.persona = persona
        self.fact = fact
        self.sampled_conversation_types = sampled_conversation_types
        self.preferred_conversation_type = preferred_conversation_type
        self.target_message_count = target_message_count
        self.candidate_conversation_flows = candidate_conversation_flows
        self.persona_signal_level = persona_signal_level
        self.persona_signal_guidance = persona_signal_guidance


class ConversationCaseGenerationJob:
    def __init__(
        self,
        *,
        accepted_case: AcceptedCase,
        persona: SanitizedPersonaProfile,
        fact_jobs: list[ConversationGenerationJob],
    ) -> None:
        self.accepted_case = accepted_case
        self.persona = persona
        self.fact_jobs = fact_jobs


def build_conversation_generation_case_jobs(
    accepted_cases: list[AcceptedCase],
    personas: dict[str, SanitizedPersonaProfile],
    rng: random.Random,
) -> list[ConversationCaseGenerationJob]:
    fact_jobs = build_conversation_generation_jobs(accepted_cases, personas, rng)
    fact_jobs_by_case_id: dict[str, list[ConversationGenerationJob]] = defaultdict(list)
    for job in fact_jobs:
        fact_jobs_by_case_id[job.accepted_case.case.case_id].append(job)
    return [
        ConversationCaseGenerationJob(
            accepted_case=accepted,
            persona=personas[accepted.case.persona_id],
            fact_jobs=fact_jobs_by_case_id[accepted.case.case_id],
        )
        for accepted in accepted_cases
    ]


def build_conversation_generation_jobs(
    accepted_cases: list[AcceptedCase],
    personas: dict[str, SanitizedPersonaProfile],
    rng: random.Random,
) -> list[ConversationGenerationJob]:
    raw_jobs: list[tuple[AcceptedCase, SanitizedPersonaProfile, Fact]] = []
    indices_by_persona: dict[str, list[int]] = defaultdict(list)
    for accepted in accepted_cases:
        persona = personas[accepted.case.persona_id]
        for fact in accepted.case.facts:
            indices_by_persona[accepted.case.persona_id].append(len(raw_jobs))
            raw_jobs.append((accepted, persona, fact))

    jobs: list[ConversationGenerationJob | None] = [None] * len(raw_jobs)
    for persona_id, indices in indices_by_persona.items():
        preferred_types = balanced_conversation_type_preferences(len(indices), rng)
        used_types_by_case_id: dict[str, set[str]] = defaultdict(set)
        for index in indices:
            accepted, persona, fact = raw_jobs[index]
            if persona.persona_id != persona_id:
                raise ValueError("persona grouping mismatch while building conversation generation jobs")
            case_id = accepted.case.case_id
            used_types = used_types_by_case_id[case_id]
            preferred_index = next(
                (candidate_index for candidate_index, item in enumerate(preferred_types) if item not in used_types),
                0,
            )
            preferred_type = preferred_types.pop(preferred_index)
            used_types.add(preferred_type)
            sampled_conversation_types = sample_conversation_type_candidates(
                rng,
                preferred_conversation_type=preferred_type,
            )
            target_message_count = sample_target_message_count(rng)
            candidate_conversation_flows = sample_conversation_type_flows(rng, sampled_conversation_types)
            persona_signal_level = sample_persona_signal_level(rng)
            jobs[index] = ConversationGenerationJob(
                accepted_case=accepted,
                persona=persona,
                fact=fact,
                sampled_conversation_types=sampled_conversation_types,
                preferred_conversation_type=preferred_type,
                target_message_count=target_message_count,
                candidate_conversation_flows=candidate_conversation_flows,
                persona_signal_level=persona_signal_level,
                persona_signal_guidance=persona_signal_guidance_for_level(persona_signal_level),
            )
    return [job for job in jobs if job is not None]


class ConversationGenerationCheckpoint:
    def __init__(self, path: Path | None, case_jobs: list[ConversationCaseGenerationJob]) -> None:
        self.path = path
        self.fact_jobs = [job for case_job in case_jobs for job in case_job.fact_jobs]
        self.job_keys = [conversation_generation_job_key(job) for job in self.fact_jobs]
        self.fact_jobs_by_key = {
            conversation_generation_job_key(job): job
            for job in self.fact_jobs
        }
        self.results_by_key = load_conversation_generation_checkpoint(path)
        self._lock = Lock()

    def job_count(self) -> int:
        return len(self.job_keys)

    def completed_count(self) -> int:
        return sum(1 for key in self.job_keys if key in self.results_by_key)

    def cached_count(self) -> int:
        return self.completed_count()

    def get(self, job: ConversationGenerationJob) -> ConversationGenerationResult | None:
        return self.results_by_key.get(conversation_generation_job_key(job))

    def record(self, job: ConversationGenerationJob, result: ConversationGenerationResult) -> None:
        key = conversation_generation_job_key(job)
        with self._lock:
            self.results_by_key[key] = result
            self.flush_locked()

    def ordered_results(self) -> list[ConversationGenerationResult]:
        return [
            self.results_by_key[key]
            for key in self.job_keys
            if key in self.results_by_key
        ]

    def flush_locked(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            conversation_generation_checkpoint_item(self.fact_jobs_by_key[key], self.results_by_key[key])
            for key in self.job_keys
            if key in self.results_by_key
        ]
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(self.path)


def load_conversation_generation_checkpoint(path: Path | None) -> dict[str, ConversationGenerationResult]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"conversation generation checkpoint must be a JSON array: {path}")
    results: dict[str, ConversationGenerationResult] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"conversation generation checkpoint items must be objects: {path}")
        key = str(item.get("job_key") or "").strip()
        if not key:
            raise ValueError(f"conversation generation checkpoint item missing job_key: {path}")
        result = conversation_generation_result_from_checkpoint_item(item)
        if key not in results:
            results[key] = result
    return results


def conversation_generation_checkpoint_item(
    job: ConversationGenerationJob,
    result: ConversationGenerationResult,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_key": conversation_generation_job_key(job),
        "case_id": job.accepted_case.case.case_id,
        "fact_id": job.fact.fact_id,
    }
    if result.conversation is not None:
        payload["status"] = "accepted"
        payload["conversation"] = result.conversation.model_dump(mode="json")
        return payload
    if result.rejected is not None:
        payload["status"] = "rejected"
        payload["rejected_conversation"] = result.rejected.model_dump(mode="json")
        return payload
    raise ValueError("conversation generation checkpoint result must contain conversation or rejected")


def conversation_generation_result_from_checkpoint_item(item: dict[str, Any]) -> ConversationGenerationResult:
    status = str(item.get("status") or "").strip()
    if status == "accepted":
        return ConversationGenerationResult(
            conversation=ConversationSession.model_validate(item.get("conversation")),
        )
    if status == "rejected":
        return ConversationGenerationResult(
            rejected=RejectedConversation.model_validate(item.get("rejected_conversation")),
        )
    raise ValueError(f"unknown conversation generation checkpoint status: {status}")


def conversation_generation_job_key(job: ConversationGenerationJob) -> str:
    return f"{job.accepted_case.case.case_id}:{job.fact.fact_id}"


def generate_conversations_for_case(
    case_job: ConversationCaseGenerationJob,
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
) -> list[ConversationSession]:
    sessions: list[ConversationSession] = []
    for job in case_job.fact_jobs:
        session = generate_conversation_for_job(
            job,
            llm_client,
            generation_model=generation_model,
            prior_case_sessions=sessions,
        )
        sessions.append(session)
    return sessions


def generate_conversation_results_for_case(
    case_job: ConversationCaseGenerationJob,
    llm_client: Any,
    *,
    checkpoint: ConversationGenerationCheckpoint | None = None,
    generation_model: str = GENERATION_MODEL,
    validation_retries: int = 0,
) -> ConversationCaseGenerationResult:
    conversations: list[ConversationSession] = []
    rejected: list[RejectedConversation] = []
    for job in case_job.fact_jobs:
        result = checkpoint.get(job) if checkpoint is not None else None
        if result is None:
            result = generate_conversation_result_for_job(
                job,
                llm_client,
                generation_model=generation_model,
                validation_retries=validation_retries,
                prior_case_sessions=conversations,
            )
            if checkpoint is not None:
                checkpoint.record(job, result)
        if result.conversation is not None:
            conversations.append(result.conversation)
        if result.rejected is not None:
            rejected.append(result.rejected)
    return ConversationCaseGenerationResult(conversations=conversations, rejected=rejected)


def generate_conversation_result_for_job(
    job: ConversationGenerationJob,
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
    validation_retries: int = 0,
    prior_case_sessions: list[ConversationSession] | None = None,
) -> ConversationGenerationResult:
    attempts = max(0, validation_retries) + 1
    last_rejected: RejectedConversation | None = None
    for attempt in range(1, attempts + 1):
        session: ConversationSession | None = None
        try:
            session = generate_conversation_without_local_validation(
                job,
                llm_client,
                generation_model=generation_model,
                prior_case_sessions=prior_case_sessions,
            )
            validate_conversation_session(session, job.fact.text)
            validate_target_message_count(session, job.target_message_count)
            return ConversationGenerationResult(conversation=session)
        except RuntimeError:
            raise
        except Exception as exc:
            reason = (
                f"Local conversation validation failed: {exc}"
                if session is not None
                else f"Conversation payload validation failed before local validation: {describe_conversation_generation_exception(exc)}"
            )
            last_rejected = build_local_validation_rejection(
                session
                or build_failed_conversation_placeholder(
                    job,
                    reason=reason,
                    generation_model=generation_model,
                ),
                reason=reason,
                generation_model=generation_model,
                attempt=attempt,
                max_attempts=attempts,
            )
    if last_rejected is None:
        raise RuntimeError(f"conversation validation failed for {job.fact.fact_id}")
    return ConversationGenerationResult(rejected=last_rejected)


def generate_conversation_for_job(
    job: ConversationGenerationJob,
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
    prior_case_sessions: list[ConversationSession] | None = None,
) -> ConversationSession:
    session = generate_conversation_without_local_validation(
        job,
        llm_client,
        generation_model=generation_model,
        prior_case_sessions=prior_case_sessions,
    )
    validate_conversation_session(session, job.fact.text)
    validate_target_message_count(session, job.target_message_count)
    return session


def generate_conversation_without_local_validation(
    job: ConversationGenerationJob,
    llm_client: Any,
    *,
    generation_model: str = GENERATION_MODEL,
    prior_case_sessions: list[ConversationSession] | None = None,
) -> ConversationSession:
    prompt = build_conversation_prompt(
        job.persona.persona_str,
        job.fact.model_dump(mode="json"),
        job.accepted_case.case.model_dump(mode="json"),
        job.sampled_conversation_types,
        preferred_conversation_type=job.preferred_conversation_type,
        target_message_count=job.target_message_count,
        prior_case_sessions=build_prior_case_session_context(prior_case_sessions or []),
        candidate_conversation_flows=job.candidate_conversation_flows,
        persona_signal_level=job.persona_signal_level,
        persona_signal_guidance=job.persona_signal_guidance,
    )
    result = llm_client.stream_chat(
        [{"role": "user", "content": prompt}],
        phase="conversation_generation",
        temperature=0.4,
    )
    if not result.stream_completed:
        raise RuntimeError(f"conversation stream failed for {job.fact.fact_id}: {result.error}")
    payload = parse_json_after_output(result.text)
    return build_conversation_session(
        payload,
        accepted_case=job.accepted_case,
        fact_id=job.fact.fact_id,
        sampled_conversation_types=job.sampled_conversation_types,
        preferred_conversation_type=job.preferred_conversation_type,
        candidate_conversation_flows=job.candidate_conversation_flows,
        persona_signal_level=job.persona_signal_level,
        persona_signal_guidance=job.persona_signal_guidance,
        generation_model=generation_model,
        stream_completed=result.stream_completed,
    )


def build_conversation_session(
    payload: dict[str, Any],
    *,
    accepted_case: AcceptedCase,
    fact_id: str,
    sampled_conversation_types: list[str],
    preferred_conversation_type: str | None = None,
    candidate_conversation_flows: dict[str, str] | None = None,
    persona_signal_level: str | None = None,
    persona_signal_guidance: str | None = None,
    generation_model: str = GENERATION_MODEL,
    stream_completed: bool = True,
) -> ConversationSession:
    selected = str(payload.get("selected_conversation_type") or sampled_conversation_types[0])
    if selected not in sampled_conversation_types:
        selected = sampled_conversation_types[0]
    messages = normalize_messages(payload.get("messages") or [])
    fact_text = next(
        (fact.text for fact in accepted_case.case.facts if fact.fact_id == fact_id),
        "",
    )
    return ConversationSession(
        conversation_id=make_conversation_id(accepted_case.case.case_id, fact_id, selected),
        persona_id=accepted_case.case.persona_id,
        case_id=accepted_case.case.case_id,
        fact_id=fact_id,
        topic_preference=accepted_case.case.topic_preference,
        case_description=accepted_case.case.description,
        case_relation_type=accepted_case.case.relation_type,
        case_relation_subtype=accepted_case.case.relation_subtype,
        case_facts=[fact.text for fact in accepted_case.case.facts],
        fact_text=fact_text,
        sampled_conversation_types=[ConversationType(item) for item in sampled_conversation_types],
        selected_conversation_type=ConversationType(selected),
        preferred_conversation_type=ConversationType(preferred_conversation_type) if preferred_conversation_type else None,
        preferred_conversation_type_used=selected == preferred_conversation_type if preferred_conversation_type else None,
        sampled_conversation_flows=candidate_conversation_flows or {},
        selected_conversation_flow=(candidate_conversation_flows or {}).get(selected),
        persona_signal_level=persona_signal_level,
        persona_signal_guidance=persona_signal_guidance,
        messages=messages,
        turn_count=count_conversation_turns(messages),
        token_count=count_conversation_tokens(messages),
        generation_model=generation_model,
        stream_completed=stream_completed,
        created_at=utc_now_iso(),
    )


def build_prior_case_session_context(sessions: list[ConversationSession]) -> list[dict[str, Any]]:
    return [
        {
            "conversation_id": session.conversation_id,
            "fact_id": session.fact_id,
            "fact_text": session.fact_text,
            "selected_conversation_type": session.selected_conversation_type,
            "messages": [message.model_dump(mode="json") for message in session.messages],
        }
        for session in sessions
    ]


def make_conversation_id(case_id: str, fact_id: str, selected_conversation_type: str) -> str:
    digest = hashlib.sha1(f"{case_id}|{fact_id}|{selected_conversation_type}".encode("utf-8")).hexdigest()[:12]
    return f"conv-{digest}"


def validate_target_message_count(session: ConversationSession, target_message_count: int) -> None:
    message_count = len(session.messages)
    if message_count != target_message_count:
        raise ValueError(
            f"conversation has {message_count} messages ({session.turn_count} turns); "
            f"expected exactly {target_message_count} messages ({target_message_count // 2} turns)"
        )


def describe_conversation_generation_exception(exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return f"missing field {exc.args[0]!r}"
    detail = str(exc).strip()
    return detail or exc.__class__.__name__


def build_failed_conversation_placeholder(
    job: ConversationGenerationJob,
    *,
    reason: str,
    generation_model: str,
) -> ConversationSession:
    selected = job.preferred_conversation_type or job.sampled_conversation_types[0]
    summary = reason if len(reason) <= 280 else f"{reason[:277]}..."
    return build_conversation_session(
        {
            "selected_conversation_type": selected,
            "messages": [
                {
                    "role": "user",
                    "content": "I need a placeholder because the generated conversation payload could not be validated.",
                },
                {
                    "role": "assistant",
                    "content": "This record exists only to capture a failed generation attempt and is not a usable session.",
                },
                {
                    "role": "user",
                    "content": "What went wrong with the generated payload?",
                },
                {
                    "role": "assistant",
                    "content": summary,
                },
                {
                    "role": "user",
                    "content": "Should this session be used for downstream evaluation?",
                },
                {
                    "role": "assistant",
                    "content": "No. It should stay only in rejected_conversations as a local validation artifact.",
                },
            ],
        },
        accepted_case=job.accepted_case,
        fact_id=job.fact.fact_id,
        sampled_conversation_types=job.sampled_conversation_types,
        preferred_conversation_type=job.preferred_conversation_type,
        candidate_conversation_flows=job.candidate_conversation_flows,
        persona_signal_level=job.persona_signal_level,
        persona_signal_guidance=job.persona_signal_guidance,
        generation_model=generation_model,
        stream_completed=True,
    )


def build_local_validation_rejection(
    session: ConversationSession,
    *,
    reason: str,
    generation_model: str,
    attempt: int,
    max_attempts: int,
) -> RejectedConversation:
    decision = ConversationFilterDecision(
        decision_id=make_local_validation_decision_id(session.conversation_id, reason, attempt),
        conversation_id=session.conversation_id,
        accepted=False,
        reason=reason,
        reject_categories=["local_validation"],
        filter_model=f"{generation_model}:local_validation",
        stream_completed=session.stream_completed,
        created_at=utc_now_iso(),
    )
    return RejectedConversation(
        conversation=session,
        filter_decision=decision.model_copy(
            update={
                "reject_categories": [
                    *decision.reject_categories,
                    f"attempt_{attempt}_of_{max_attempts}",
                ]
            }
        ),
    )


def make_local_validation_decision_id(conversation_id: str, reason: str, attempt: int) -> str:
    digest = hashlib.sha1(f"{conversation_id}|{reason}|{attempt}".encode("utf-8")).hexdigest()[:12]
    return f"conv-local-validation-{digest}"
