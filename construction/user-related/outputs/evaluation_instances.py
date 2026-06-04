from __future__ import annotations

import hashlib

from infra.logging_utils import utc_now_iso
from core.schemas import AcceptedCase, AcceptedCaseQA, ConversationSession, EvaluationInstance, SanitizedPersonaProfile


def build_evaluation_instances(
    accepted_case_qa: list[AcceptedCaseQA],
    accepted_cases: list[AcceptedCase],
    conversations: list[ConversationSession],
    personas: dict[str, SanitizedPersonaProfile],
) -> list[EvaluationInstance]:
    cases_by_id = {item.case.case_id: item for item in accepted_cases}
    sessions_by_id = {session.conversation_id: session for session in conversations}
    instances: list[EvaluationInstance] = []

    for accepted_qa in accepted_case_qa:
        qa = accepted_qa.qa
        accepted_case = cases_by_id.get(qa.case_id)
        if accepted_case is None:
            raise ValueError(f"accepted QA references missing case: {qa.case_id}")
        persona = personas.get(qa.persona_id)
        if persona is None:
            raise ValueError(f"accepted QA references missing persona: {qa.persona_id}")
        sessions = [sessions_by_id[session_id] for session_id in qa.session_ids if session_id in sessions_by_id]
        missing_sessions = [session_id for session_id in qa.session_ids if session_id not in sessions_by_id]
        if missing_sessions:
            raise ValueError(f"accepted QA references missing sessions for {qa.qa_id}: {missing_sessions}")
        for question in qa.questions:
            instances.append(
                EvaluationInstance(
                    instance_id=make_evaluation_instance_id(qa.qa_id, question.question_id),
                    persona_id=qa.persona_id,
                    persona=persona,
                    topic_preference=qa.topic_preference,
                    relation_type=qa.relation_type,
                    relation_subtype=qa.relation_subtype,
                    qa_mode=qa.qa_mode,
                    case_id=qa.case_id,
                    case=accepted_case.case,
                    facts=accepted_case.case.facts,
                    sessions=sessions,
                    qa_id=qa.qa_id,
                    question_id=question.question_id,
                    query=question.question,
                    task_form=question.task_form,
                    correct_answers=question.correct_answers,
                    incorrect_answers=question.incorrect_answers,
                    case_filter_decision=accepted_case.filter_decision,
                    qa_filter_decision=accepted_qa.filter_decision,
                    created_at=utc_now_iso(),
                )
            )
    return instances


def make_evaluation_instance_id(qa_id: str, question_id: str) -> str:
    digest = hashlib.sha1(f"{qa_id}|{question_id}".encode("utf-8")).hexdigest()[:12]
    return f"eval-{digest}"
