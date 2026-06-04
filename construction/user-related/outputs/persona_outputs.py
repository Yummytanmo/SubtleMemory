from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from outputs.session_timeline import conversation_metric_totals
from core.schemas import QAMode
from core.schemas import (
    AcceptedCase,
    AcceptedCaseQA,
    ConversationSession,
    EvaluationInstance,
    MemoryCase,
    RejectedCase,
    RejectedCaseQA,
    RejectedConversation,
    SanitizedPersonaProfile,
    TimestampedSession,
    TopicPreferenceGroup,
)


LEGACY_PERSONA_ARTIFACT_PATHS = {
    "persona": Path("00_profile/persona.json"),
    "topic_preferences": Path("00_profile/topic_preferences.json"),
    "case_relation_plan": Path("01_cases/case_relation_plan.json"),
    "case_relation_plan_report": Path("01_cases/case_relation_plan_report.json"),
    "generated_cases": Path("01_cases/generated_cases.json"),
    "filter_report": Path("01_cases/filter_report.json"),
    "accepted_cases": Path("01_cases/accepted_cases.json"),
    "qa_eligible_cases": Path("01_cases/qa_eligible_cases.json"),
    "rejected_cases": Path("01_cases/rejected_cases.json"),
    "conversations": Path("02_sessions/conversations.json"),
    "qa_conversations": Path("02_sessions/qa_conversations.json"),
    "sessions": Path("02_sessions/sessions.json"),
    "rejected_conversations": Path("02_sessions/rejected_conversations.json"),
    "dropped_cases_after_conversation": Path("02_sessions/dropped_cases_after_conversation.json"),
    "conversation_filter_report": Path("02_sessions/conversation_filter_report.json"),
    "generated_case_qa": Path("03_qa/generated_case_qa.json"),
    "accepted_case_qa": Path("03_qa/accepted_case_qa.json"),
    "rejected_case_qa": Path("03_qa/rejected_case_qa.json"),
    "qa_filter_report": Path("03_qa/qa_filter_report.json"),
    "evaluation_instances": Path("04_evaluation/evaluation_instances.json"),
}

MODE_SCOPED_PERSONA_ARTIFACT_STEMS = {
    "generated_case_qa",
    "accepted_case_qa",
    "rejected_case_qa",
    "qa_filter_report",
    "evaluation_instances",
}


@dataclass
class PersonaOutputBundle:
    persona_key: str
    persona_id: str
    artifacts: dict[str, Any]
    export_payload: dict[str, Any]


def build_persona_output_bundles(
    *,
    personas: dict[str, SanitizedPersonaProfile],
    topic_groups: list[TopicPreferenceGroup],
    generated_cases: list[MemoryCase],
    accepted_cases: list[AcceptedCase],
    qa_eligible_cases: list[AcceptedCase] | None = None,
    rejected_cases: list[RejectedCase],
    conversations: list[ConversationSession],
    qa_conversations: list[ConversationSession] | None = None,
    sessions: list[TimestampedSession],
    rejected_conversations: list[RejectedConversation],
    dropped_cases_after_conversation: list[dict[str, Any]] | None = None,
    generated_case_qa: list[Any],
    accepted_case_qa: list[AcceptedCaseQA],
    rejected_case_qa: list[RejectedCaseQA],
    evaluation_instances: list[EvaluationInstance],
    qa_mode: str,
    extra_artifacts_by_persona: dict[str, dict[str, Any]] | None = None,
) -> list[PersonaOutputBundle]:
    qa_eligible_cases = qa_eligible_cases if qa_eligible_cases is not None else accepted_cases
    qa_conversations = qa_conversations if qa_conversations is not None else conversations
    dropped_cases_after_conversation = dropped_cases_after_conversation or []
    extra_artifacts_by_persona = extra_artifacts_by_persona or {}
    persona_items = sorted(personas.values(), key=lambda item: item.persona_id)
    bundles: list[PersonaOutputBundle] = []
    for index, persona in enumerate(persona_items):
        persona_key = f"persona_{index}"
        persona_id = persona.persona_id
        artifacts = {
            "persona": persona,
            "topic_preferences": filter_by_attr(topic_groups, "persona_id", persona_id),
            "generated_cases": filter_by_attr(generated_cases, "persona_id", persona_id),
            "accepted_cases": filter_by_case_persona(accepted_cases, persona_id),
            "qa_eligible_cases": filter_by_case_persona(qa_eligible_cases, persona_id),
            "rejected_cases": filter_by_case_persona(rejected_cases, persona_id),
            "conversations": filter_by_attr(conversations, "persona_id", persona_id),
            "qa_conversations": filter_by_attr(qa_conversations, "persona_id", persona_id),
            "sessions": filter_by_attr(sessions, "persona_id", persona_id),
            "rejected_conversations": filter_by_conversation_persona(rejected_conversations, persona_id),
            "dropped_cases_after_conversation": filter_by_dropped_case_persona(dropped_cases_after_conversation, persona_id),
            "generated_case_qa": filter_by_attr(generated_case_qa, "persona_id", persona_id),
            "accepted_case_qa": filter_by_qa_persona(accepted_case_qa, persona_id),
            "rejected_case_qa": filter_by_rejected_qa_persona(rejected_case_qa, persona_id),
            "evaluation_instances": filter_by_attr(evaluation_instances, "persona_id", persona_id),
        }
        artifacts.update(extra_artifacts_by_persona.get(persona_id, {}))
        artifacts["manifest"] = build_persona_manifest(persona_key, persona_id, artifacts, qa_mode=qa_mode)
        bundles.append(
            PersonaOutputBundle(
                persona_key=persona_key,
                persona_id=persona_id,
                artifacts=artifacts,
                export_payload=build_persona_export_payload(persona_key, persona_id, artifacts),
            )
        )
    return bundles


def build_persona_export_payload(persona_key: str, persona_id: str, artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "persona": build_export_persona(artifacts["persona"]),
        "cases": build_export_cases(artifacts["evaluation_instances"], artifacts["sessions"]),
    }


def build_export_persona(persona: SanitizedPersonaProfile) -> dict[str, Any]:
    return {
        "persona_str": persona.persona_str,
        "profile": persona.profile,
    }


def build_export_sessions(sessions: list[TimestampedSession]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": session.timestamp,
            "conversation_type": session.selected_conversation_type,
            "conversation_flow": session.selected_conversation_flow,
            "persona_signal_level": session.persona_signal_level,
            "history": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in session.messages
            ],
        }
        for session in sessions
    ]


def build_export_cases(
    evaluation_instances: list[EvaluationInstance],
    sessions: list[TimestampedSession],
) -> list[dict[str, Any]]:
    sessions_by_case_id: dict[str, list[TimestampedSession]] = {}
    for session in sessions:
        sessions_by_case_id.setdefault(session.case_id, []).append(session)

    cases_by_id: dict[str, dict[str, Any]] = {}
    for item in evaluation_instances:
        case_payload = cases_by_id.setdefault(
            item.case_id,
            {
                "topic": item.topic_preference,
                "relation_type": item.relation_type,
                "relation_subtype": item.relation_subtype,
                "case": item.case.description,
                "facts": [fact.text for fact in item.facts],
                "qa": [],
                "sessions": build_export_sessions(sessions_by_case_id.get(item.case_id, [])),
            },
        )
        case_payload["qa"].append(
            {
                "query": item.query,
                "qa_mode": item.qa_mode,
                "correct_answers": [answer.text for answer in item.correct_answers],
                "incorrect_answers": [answer.text for answer in item.incorrect_answers],
            }
        )
    return list(cases_by_id.values())


def build_persona_manifest(persona_key: str, persona_id: str, artifacts: dict[str, Any], *, qa_mode: str) -> dict[str, Any]:
    return {
        "schema": "user_related_persona_manifest_v1",
        "persona_key": persona_key,
        "persona_id": persona_id,
        "qa_mode": qa_mode_value(qa_mode),
        "artifact_files": persona_artifact_files(artifacts, qa_mode=qa_mode),
        "filter_pass_rates": filter_pass_rates(artifacts),
        "filter_metrics": filter_metrics(artifacts),
        "conversation_metrics": conversation_metric_totals(artifacts["sessions"]),
        "distribution_counts": distribution_counts(artifacts),
        "counts": artifact_counts(artifacts),
    }


def build_export_manifest(bundles: list[PersonaOutputBundle], *, qa_mode: str) -> dict[str, Any]:
    counts = aggregate_manifest_counts(bundles)
    return {
        "schema": "user_related_export_manifest_v1",
        "qa_mode": qa_mode_value(qa_mode),
        "summary": {
            "filter_pass_rates": filter_pass_rates_from_counts(counts),
            "filter_metrics": filter_metrics_from_counts(counts),
            "conversation_metrics": {
                "conversation_total_turns": counts.get("conversation_total_turns", 0),
                "conversation_total_tokens": counts.get("conversation_total_tokens", 0),
            },
            "distribution_counts": aggregate_distribution_counts(bundles),
            "counts": counts,
        },
        "personas": [
            {
                "persona_key": bundle.persona_key,
                "persona_id": bundle.persona_id,
                "file": f"{bundle.persona_key}.json",
                "filter_pass_rates": bundle.artifacts["manifest"]["filter_pass_rates"],
                "filter_metrics": bundle.artifacts["manifest"]["filter_metrics"],
                "conversation_metrics": bundle.artifacts["manifest"]["conversation_metrics"],
                "distribution_counts": bundle.artifacts["manifest"]["distribution_counts"],
                "counts": bundle.artifacts["manifest"]["counts"],
            }
            for bundle in bundles
        ],
    }


def artifact_counts(artifacts: dict[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for name, value in artifacts.items():
        if name == "manifest":
            continue
        if isinstance(value, list):
            output[name] = len(value)
    sessions = artifacts.get("sessions", [])
    if isinstance(sessions, list):
        output.update(conversation_metric_totals(sessions))
    output.update(qa_item_counts(artifacts))
    return output


def qa_mode_value(qa_mode: str | QAMode) -> str:
    return QAMode(qa_mode).value


def mode_scoped_persona_artifact_path(stem: str, *, qa_mode: str | QAMode) -> Path:
    mode = qa_mode_value(qa_mode)
    legacy = legacy_persona_artifact_path(stem)
    if legacy.parent.name == "03_qa":
        return Path("03_qa") / mode / legacy.name
    if legacy.parent.name == "04_evaluation":
        return Path("04_evaluation") / mode / legacy.name
    raise KeyError(f"{stem} is not a mode-scoped persona artifact")


def legacy_persona_artifact_path(stem: str) -> Path:
    if stem == "manifest":
        return Path("manifest.json")
    if stem not in LEGACY_PERSONA_ARTIFACT_PATHS:
        raise KeyError(f"unknown persona artifact stem: {stem}")
    return LEGACY_PERSONA_ARTIFACT_PATHS[stem]


def persona_artifact_path(stem: str, *, qa_mode: str | QAMode) -> Path:
    if stem == "manifest":
        return Path("manifest.json")
    if stem in MODE_SCOPED_PERSONA_ARTIFACT_STEMS:
        return mode_scoped_persona_artifact_path(stem, qa_mode=qa_mode)
    return legacy_persona_artifact_path(stem)


def persona_artifact_files(artifacts: dict[str, Any], *, qa_mode: str) -> dict[str, str]:
    return {
        stem: persona_artifact_path(stem, qa_mode=qa_mode).as_posix()
        for stem in LEGACY_PERSONA_ARTIFACT_PATHS
        if stem in artifacts
    }


def filter_pass_rates(artifacts: dict[str, Any]) -> dict[str, float]:
    counts = artifact_counts(artifacts)
    return filter_pass_rates_from_counts(counts)


def filter_pass_rates_from_counts(counts: dict[str, int]) -> dict[str, float]:
    generated_cases = counts.get("generated_cases", 0)
    accepted_cases = counts.get("accepted_cases", 0)
    accepted_conversations = counts.get("conversations", 0)
    rejected_conversations = counts.get("rejected_conversations", 0)
    generated_case_qa = counts.get("generated_case_qa", 0)
    accepted_case_qa = counts.get("accepted_case_qa", 0)
    return {
        "case_filter": safe_pass_rate(accepted_cases, generated_cases),
        "conversation_filter": safe_pass_rate(accepted_conversations, accepted_conversations + rejected_conversations),
        "qa_filter": safe_pass_rate(accepted_case_qa, generated_case_qa),
    }


def filter_metrics(artifacts: dict[str, Any]) -> dict[str, Any]:
    return filter_metrics_from_counts(artifact_counts(artifacts))


def filter_metrics_from_counts(counts: dict[str, int]) -> dict[str, Any]:
    generated_cases = counts.get("generated_cases", 0)
    accepted_cases = counts.get("accepted_cases", 0)
    rejected_cases = counts.get("rejected_cases", 0)
    accepted_conversations = counts.get("conversations", 0)
    rejected_conversations = counts.get("rejected_conversations", 0)
    total_conversations = accepted_conversations + rejected_conversations
    qa_eligible_cases = counts.get("qa_eligible_cases", 0)
    dropped_cases_after_conversation = counts.get("dropped_cases_after_conversation", 0)
    qa_conversations = counts.get("qa_conversations", counts.get("sessions", 0))
    generated_case_qa = counts.get("generated_case_qa", 0)
    accepted_case_qa = counts.get("accepted_case_qa", 0)
    final_rejected_case_qa = max(generated_case_qa - accepted_case_qa, 0)
    generated_questions = counts.get("generated_case_qa_questions", 0)
    accepted_questions = counts.get("accepted_case_qa_questions", 0)
    return {
        "case_filter": {
            "generated_cases": generated_cases,
            "accepted_cases": accepted_cases,
            "rejected_cases": rejected_cases,
            "retention_rate": safe_pass_rate(accepted_cases, generated_cases),
        },
        "conversation_filter": {
            "total_generated_conversations": total_conversations,
            "accepted_conversations": accepted_conversations,
            "rejected_conversations": rejected_conversations,
            "retention_rate": safe_pass_rate(accepted_conversations, total_conversations),
        },
        "conversation_case_pruning": {
            "accepted_cases_before_pruning": accepted_cases,
            "qa_eligible_cases": qa_eligible_cases,
            "dropped_cases_after_conversation": dropped_cases_after_conversation,
            "accepted_conversations_before_pruning": accepted_conversations,
            "qa_conversations": qa_conversations,
            "dropped_accepted_conversations_after_case_pruning": max(accepted_conversations - qa_conversations, 0),
            "case_retention_rate": safe_pass_rate(qa_eligible_cases, accepted_cases),
            "conversation_retention_rate": safe_pass_rate(qa_conversations, accepted_conversations),
        },
        "qa_filter": {
            "generated_case_qa": generated_case_qa,
            "accepted_case_qa": accepted_case_qa,
            "rejected_case_qa": final_rejected_case_qa,
            "retention_rate": safe_pass_rate(accepted_case_qa, generated_case_qa),
            "generated_questions": generated_questions,
            "accepted_questions": accepted_questions,
            "rejected_questions": max(generated_questions - accepted_questions, 0),
            "question_retention_rate": safe_pass_rate(accepted_questions, generated_questions),
        },
    }


def qa_item_counts(artifacts: dict[str, Any]) -> dict[str, int]:
    generated_case_qa = artifacts.get("generated_case_qa", [])
    accepted_case_qa = artifacts.get("accepted_case_qa", [])
    generated_questions = count_case_qa_questions(generated_case_qa)
    accepted_questions = count_accepted_case_qa_questions(accepted_case_qa)
    return {
        "generated_case_qa_questions": generated_questions,
        "accepted_case_qa_questions": accepted_questions,
        "rejected_case_qa_questions": max(generated_questions - accepted_questions, 0),
    }


def count_case_qa_questions(items: list[Any]) -> int:
    return sum(len(getattr(item, "questions", [])) for item in items)


def count_accepted_case_qa_questions(items: list[Any]) -> int:
    return sum(len(getattr(item.qa, "questions", [])) for item in items)


def safe_pass_rate(accepted: int, total: int) -> float:
    return accepted / total if total else 0.0


def aggregate_manifest_counts(bundles: list[PersonaOutputBundle]) -> dict[str, int]:
    output: dict[str, int] = {}
    for bundle in bundles:
        for key, value in bundle.artifacts["manifest"]["counts"].items():
            output[key] = output.get(key, 0) + value
    return output


def distribution_counts(artifacts: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    return {
        "accepted_cases": case_distribution_counts(artifacts.get("accepted_cases", [])),
        "evaluation_instances": item_distribution_counts(artifacts.get("evaluation_instances", [])),
    }


def case_distribution_counts(items: list[Any]) -> dict[str, dict[str, int]]:
    cases = [getattr(item, "case", item) for item in items]
    return item_distribution_counts(cases)


def item_distribution_counts(items: list[Any]) -> dict[str, dict[str, int]]:
    return {
        "relation_type": count_attr(items, "relation_type"),
        "relation_subtype": count_attr(items, "relation_subtype"),
        "topic": count_attr(items, "topic_preference"),
    }


def count_attr(items: list[Any], attr: str) -> dict[str, int]:
    counts = Counter(
        str(value)
        for item in items
        if (value := getattr(item, attr, None)) is not None and str(value).strip()
    )
    return dict(sorted(counts.items()))


def aggregate_distribution_counts(bundles: list[PersonaOutputBundle]) -> dict[str, dict[str, dict[str, int]]]:
    output: dict[str, dict[str, Counter[str]]] = {}
    for bundle in bundles:
        for stage, metrics in bundle.artifacts["manifest"]["distribution_counts"].items():
            output.setdefault(stage, {})
            for metric, counts in metrics.items():
                output[stage].setdefault(metric, Counter())
                output[stage][metric].update(counts)
    return {
        stage: {metric: dict(sorted(counts.items())) for metric, counts in metrics.items()}
        for stage, metrics in sorted(output.items())
    }


def filter_by_attr(items: list[Any], attr: str, value: str) -> list[Any]:
    return [item for item in items if getattr(item, attr) == value]


def filter_by_case_persona(items: list[AcceptedCase] | list[RejectedCase], persona_id: str) -> list[Any]:
    return [item for item in items if item.case.persona_id == persona_id]


def filter_by_conversation_persona(items: list[RejectedConversation], persona_id: str) -> list[RejectedConversation]:
    return [item for item in items if item.conversation.persona_id == persona_id]


def filter_by_dropped_case_persona(items: list[dict[str, Any]], persona_id: str) -> list[dict[str, Any]]:
    return [item for item in items if item.get("persona_id") == persona_id]


def filter_by_qa_persona(items: list[AcceptedCaseQA], persona_id: str) -> list[AcceptedCaseQA]:
    return [item for item in items if item.qa.persona_id == persona_id]


def filter_by_rejected_qa_persona(items: list[RejectedCaseQA], persona_id: str) -> list[RejectedCaseQA]:
    return [item for item in items if item.original_qa.persona_id == persona_id]
