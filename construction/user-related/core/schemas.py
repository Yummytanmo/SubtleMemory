from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from core.conversation_metrics import count_conversation_tokens, count_conversation_turns
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RelationType(str, Enum):
    COMPLEMENTARY = "complementary"
    NUANCED = "nuanced"
    CONTRADICTORY = "contradictory"


class RelationSubtype(str, Enum):
    COMPLEMENTARY_K1 = "K=1"
    COMPLEMENTARY_K_GT_1 = "K>1"
    COMPLEMENTARY_ANY_ONE = "any_one"
    NUANCED_TEMPORAL = "Temporal"
    NUANCED_CONTEXT = "Context"
    CONTRADICTORY = "contradictory"


class ConversationType(str, Enum):
    DECISION_SUPPORT = "decision_support"
    PLANNING_COORDINATION = "planning_coordination"
    TROUBLESHOOTING = "troubleshooting"
    LEARNING_EXPLANATION = "learning_explanation"
    RESOURCE_SELECTION = "resource_selection"
    WORKFLOW_SETUP = "workflow_setup"
    INFORMATION_ORGANIZATION = "information_organization"
    PERSONAL_REFLECTION = "personal_reflection"
    ARTIFACT_PRODUCTION = "artifact_production"
    ARTIFACT_REVIEW_OR_LOCALIZATION = "artifact_review_or_localization"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class QAMode(str, Enum):
    QUESTION = "question"
    TASK = "task"


class TaskForm(str, Enum):
    STRUCTURED_FORM = "structured_form"
    RESOURCE_ARRANGEMENT = "resource_arrangement"


RELATION_SUBTYPE_MAP = {
    RelationType.COMPLEMENTARY: {
        RelationSubtype.COMPLEMENTARY_K1,
        RelationSubtype.COMPLEMENTARY_K_GT_1,
        RelationSubtype.COMPLEMENTARY_ANY_ONE,
    },
    RelationType.NUANCED: {
        RelationSubtype.NUANCED_TEMPORAL,
        RelationSubtype.NUANCED_CONTEXT,
    },
    RelationType.CONTRADICTORY: {RelationSubtype.CONTRADICTORY},
}
MIN_CONVERSATION_MESSAGES = 6
MIN_CONVERSATION_TURNS = MIN_CONVERSATION_MESSAGES // 2
SANITIZED_PERSONA_PROFILE_EXCLUDED_FIELDS = {"conversations", "matched_images", "preference_updates"}


def datetime_from_iso(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc


class ArtifactModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")


class SourceLocation(ArtifactModel):
    source_file: str
    conversation_scenario: str | None = None
    item_index: int | None = None
    duplicate_locations: list[dict[str, Any]] = Field(default_factory=list)


class SourcePersonaRecord(ArtifactModel):
    persona_id: str
    source_file: str
    raw_payload: dict[str, Any]
    persona_fields: dict[str, Any]
    source_conversations: Any = Field(default_factory=dict)

    @field_validator("source_file")
    @classmethod
    def source_file_exists(cls, value: str) -> str:
        if not Path(value).exists():
            raise ValueError(f"source_file does not exist: {value}")
        return value

    @field_validator("persona_fields")
    @classmethod
    def no_excluded_fields_in_persona_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        excluded = sorted(SANITIZED_PERSONA_PROFILE_EXCLUDED_FIELDS & set(value))
        if excluded:
            raise ValueError(f"persona_fields must not contain excluded fields: {', '.join(excluded)}")
        return value


class SanitizedPersonaProfile(ArtifactModel):
    persona_id: str
    source_file: str
    profile: dict[str, Any]
    persona_str: str
    field_hash: str

    @field_validator("profile")
    @classmethod
    def no_excluded_fields_in_profile(cls, value: dict[str, Any]) -> dict[str, Any]:
        excluded = sorted(SANITIZED_PERSONA_PROFILE_EXCLUDED_FIELDS & set(value))
        if excluded:
            raise ValueError(f"profile must not contain excluded fields: {', '.join(excluded)}")
        return value


class PreferenceItem(ArtifactModel):
    preference_id: str
    persona_id: str
    topic_preference: str
    preference_text: str
    pref_type: str | None = None
    source_location: SourceLocation

    @field_validator("topic_preference", "preference_text")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value


class TopicPreferenceGroup(ArtifactModel):
    topic_id: str
    persona_id: str
    topic_preference: str
    preferences: list[PreferenceItem]
    source_count: int

    @field_validator("topic_preference")
    @classmethod
    def topic_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("topic_preference must be non-empty")
        return value

    @model_validator(mode="after")
    def source_count_matches(self) -> "TopicPreferenceGroup":
        if self.source_count < len(self.preferences):
            raise ValueError("source_count must be at least the number of preferences")
        return self


class CaseRelationPlanItem(ArtifactModel):
    persona_id: str
    preference_id: str
    topic_preference: str
    relation_type: RelationType
    relation_subtype: RelationSubtype
    planning_reason: str = ""
    planning_model: str
    stream_completed: bool
    created_at: str

    @field_validator("persona_id", "preference_id", "topic_preference")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @model_validator(mode="after")
    def relation_subtype_is_compatible(self) -> "CaseRelationPlanItem":
        relation = RelationType(self.relation_type)
        subtype = RelationSubtype(self.relation_subtype)
        if subtype not in RELATION_SUBTYPE_MAP[relation]:
            raise ValueError(f"{subtype.value} is incompatible with {relation.value}")
        return self


class Fact(ArtifactModel):
    fact_id: str
    case_id: str
    text: str
    order: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def fact_is_clean_sentence(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("fact text must be non-empty")
        lowered = text.lower()
        banned = ("complementary memory", "nuanced memory", "contradictory memory", "benchmark", "test fact")
        if any(token in lowered for token in banned):
            raise ValueError("fact text must not include benchmark labels")
        if not re.search(r"[.!?]$", text):
            raise ValueError("fact text must end as a sentence")
        return text


class MemoryCase(ArtifactModel):
    model_config = ConfigDict(use_enum_values=True, extra="allow")

    case_id: str
    persona_id: str
    topic_preference: str
    source_preference_id: str
    relation_type: RelationType
    relation_subtype: RelationSubtype
    description: str
    facts: list[Fact]
    generation_model: str
    prompt_version: str
    created_at: str

    @field_validator("description")
    @classmethod
    def description_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("description must be non-empty")
        return value

    @field_validator("facts")
    @classmethod
    def facts_non_empty(cls, value: list[Fact]) -> list[Fact]:
        if not value:
            raise ValueError("facts must be non-empty")
        return value

    @model_validator(mode="after")
    def relation_subtype_is_compatible(self) -> "MemoryCase":
        relation = RelationType(self.relation_type)
        subtype = RelationSubtype(self.relation_subtype)
        if subtype not in RELATION_SUBTYPE_MAP[relation]:
            raise ValueError(f"{subtype.value} is incompatible with {relation.value}")
        return self


class FilterDecision(ArtifactModel):
    decision_id: str
    case_id: str
    accepted: bool
    reason: str = ""
    reject_categories: list[str] = Field(default_factory=list)
    filter_model: str
    stream_completed: bool
    created_at: str

    @model_validator(mode="after")
    def rejected_cases_have_reason(self) -> "FilterDecision":
        if not self.accepted and not self.reason.strip():
            raise ValueError("rejected cases must have a reason")
        if self.accepted and any(cat.startswith("fatal") for cat in self.reject_categories):
            raise ValueError("accepted cases cannot have fatal reject categories")
        return self


class AcceptedCase(ArtifactModel):
    case: MemoryCase
    filter_decision: FilterDecision

    @model_validator(mode="after")
    def decision_accepts_case(self) -> "AcceptedCase":
        if self.case.case_id != self.filter_decision.case_id:
            raise ValueError("accepted case and decision IDs differ")
        if not self.filter_decision.accepted:
            raise ValueError("accepted case requires an accepting decision")
        return self


class RejectedCase(ArtifactModel):
    case: MemoryCase
    filter_decision: FilterDecision

    @model_validator(mode="after")
    def decision_rejects_case(self) -> "RejectedCase":
        if self.case.case_id != self.filter_decision.case_id:
            raise ValueError("rejected case and decision IDs differ")
        if self.filter_decision.accepted:
            raise ValueError("rejected case requires a rejecting decision")
        return self


class Message(ArtifactModel):
    role: MessageRole
    content: str | list[Any]

    @field_validator("content")
    @classmethod
    def content_non_empty(cls, value: str | list[Any]) -> str | list[Any]:
        if isinstance(value, str) and not value.strip():
            raise ValueError("message content must be non-empty")
        if isinstance(value, list) and not value:
            raise ValueError("message content list must be non-empty")
        return value


class ConversationSession(ArtifactModel):
    conversation_id: str
    persona_id: str
    case_id: str
    fact_id: str
    topic_preference: str | None = None
    case_description: str | None = None
    case_relation_type: RelationType | None = None
    case_relation_subtype: RelationSubtype | None = None
    case_facts: list[str] = Field(default_factory=list)
    fact_text: str | None = None
    sampled_conversation_types: list[ConversationType]
    selected_conversation_type: ConversationType
    preferred_conversation_type: ConversationType | None = None
    preferred_conversation_type_used: bool | None = None
    sampled_conversation_flows: dict[str, str] = Field(default_factory=dict)
    selected_conversation_flow: str | None = None
    persona_signal_level: str | None = None
    persona_signal_guidance: str | None = None
    messages: list[Message]
    turn_count: int | None = Field(default=None, ge=0)
    token_count: int | None = Field(default=None, ge=0)
    generation_model: str
    stream_completed: bool
    created_at: str

    @model_validator(mode="after")
    def conversation_is_valid(self) -> "ConversationSession":
        sampled = [ConversationType(item) for item in self.sampled_conversation_types]
        if len(sampled) != 4 or len(set(sampled)) != 4:
            raise ValueError("sampled_conversation_types must contain exactly four unique values")
        if ConversationType(self.selected_conversation_type) not in sampled:
            raise ValueError("selected_conversation_type must be one of sampled_conversation_types")
        selected = ConversationType(self.selected_conversation_type)
        if self.preferred_conversation_type is not None:
            preferred = ConversationType(self.preferred_conversation_type)
            if preferred not in sampled:
                raise ValueError("preferred_conversation_type must be one of sampled_conversation_types")
            expected_used = selected == preferred
            if self.preferred_conversation_type_used is None:
                self.preferred_conversation_type_used = expected_used
            elif self.preferred_conversation_type_used != expected_used:
                raise ValueError("preferred_conversation_type_used must match selected_conversation_type")
        if self.sampled_conversation_flows:
            sampled_values = {item.value for item in sampled}
            if set(self.sampled_conversation_flows) != sampled_values:
                raise ValueError("sampled_conversation_flows must match sampled_conversation_types")
            selected_flow = self.sampled_conversation_flows[selected.value]
            if self.selected_conversation_flow is None:
                self.selected_conversation_flow = selected_flow
            elif self.selected_conversation_flow != selected_flow:
                raise ValueError("selected_conversation_flow must match selected_conversation_type")
        if self.selected_conversation_flow is not None and not self.selected_conversation_flow.strip():
            raise ValueError("selected_conversation_flow must be non-empty when provided")
        if self.persona_signal_level is not None and self.persona_signal_level not in {"low", "medium", "high"}:
            raise ValueError("persona_signal_level must be low, medium, or high")
        if self.persona_signal_guidance is not None and not self.persona_signal_guidance.strip():
            raise ValueError("persona_signal_guidance must be non-empty when provided")
        if len(self.messages) < MIN_CONVERSATION_MESSAGES:
            raise ValueError("messages must contain at least three complete turns")
        if len(self.messages) % 2:
            raise ValueError("messages must contain complete user/assistant turns")
        expected_roles = ["user", "assistant"]
        for index, message in enumerate(self.messages):
            if message.role != expected_roles[index % 2]:
                raise ValueError("messages must start with user and alternate user/assistant")
        expected_turns = count_conversation_turns(self.messages)
        expected_tokens = count_conversation_tokens(self.messages)
        if self.turn_count is None:
            self.turn_count = expected_turns
        elif self.turn_count != expected_turns:
            raise ValueError("turn_count must equal the number of user/assistant turns")
        if self.token_count is None:
            self.token_count = expected_tokens
        elif self.token_count != expected_tokens:
            raise ValueError("token_count must equal the computed message content token count")
        return self


class ConversationFilterDecision(ArtifactModel):
    decision_id: str
    conversation_id: str
    accepted: bool
    reason: str = ""
    reject_categories: list[str] = Field(default_factory=list)
    filter_model: str
    stream_completed: bool
    created_at: str

    @model_validator(mode="after")
    def rejected_conversations_have_reason(self) -> "ConversationFilterDecision":
        if not self.accepted and not self.reason.strip():
            raise ValueError("rejected conversations must have a reason")
        if self.accepted and any(cat.startswith("fatal") for cat in self.reject_categories):
            raise ValueError("accepted conversations cannot have fatal reject categories")
        return self


class AcceptedConversation(ArtifactModel):
    conversation: ConversationSession
    filter_decision: ConversationFilterDecision

    @model_validator(mode="after")
    def decision_accepts_conversation(self) -> "AcceptedConversation":
        if self.conversation.conversation_id != self.filter_decision.conversation_id:
            raise ValueError("accepted conversation and decision IDs differ")
        if not self.filter_decision.accepted:
            raise ValueError("accepted conversation requires an accepting decision")
        return self


class RejectedConversation(ArtifactModel):
    conversation: ConversationSession
    filter_decision: ConversationFilterDecision

    @model_validator(mode="after")
    def decision_rejects_conversation(self) -> "RejectedConversation":
        if self.conversation.conversation_id != self.filter_decision.conversation_id:
            raise ValueError("rejected conversation and decision IDs differ")
        if self.filter_decision.accepted:
            raise ValueError("rejected conversation requires a rejecting decision")
        return self


class TimestampedSession(ArtifactModel):
    session_id: str
    conversation_id: str
    timestamp: str
    order: int = Field(ge=0)
    persona_id: str
    case_id: str
    fact_id: str
    topic_preference: str
    case_description: str
    case_relation_type: RelationType
    case_relation_subtype: RelationSubtype
    case_facts: list[str]
    fact_text: str
    sampled_conversation_types: list[ConversationType]
    selected_conversation_type: ConversationType
    preferred_conversation_type: ConversationType | None = None
    preferred_conversation_type_used: bool | None = None
    sampled_conversation_flows: dict[str, str] = Field(default_factory=dict)
    selected_conversation_flow: str | None = None
    persona_signal_level: str | None = None
    persona_signal_guidance: str | None = None
    messages: list[Message]
    turn_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    generation_model: str
    stream_completed: bool
    source_created_at: str

    @model_validator(mode="after")
    def timestamped_session_is_valid(self) -> "TimestampedSession":
        if self.session_id != self.conversation_id:
            raise ValueError("session_id must match conversation_id")
        datetime_from_iso(self.timestamp)
        sampled = [ConversationType(item) for item in self.sampled_conversation_types]
        if len(sampled) != 4 or len(set(sampled)) != 4:
            raise ValueError("sampled_conversation_types must contain exactly four unique values")
        if ConversationType(self.selected_conversation_type) not in sampled:
            raise ValueError("selected_conversation_type must be one of sampled_conversation_types")
        selected = ConversationType(self.selected_conversation_type)
        if self.preferred_conversation_type is not None:
            preferred = ConversationType(self.preferred_conversation_type)
            if preferred not in sampled:
                raise ValueError("preferred_conversation_type must be one of sampled_conversation_types")
            expected_used = selected == preferred
            if self.preferred_conversation_type_used is None:
                self.preferred_conversation_type_used = expected_used
            elif self.preferred_conversation_type_used != expected_used:
                raise ValueError("preferred_conversation_type_used must match selected_conversation_type")
        if self.sampled_conversation_flows:
            sampled_values = {item.value for item in sampled}
            if set(self.sampled_conversation_flows) != sampled_values:
                raise ValueError("sampled_conversation_flows must match sampled_conversation_types")
            selected_flow = self.sampled_conversation_flows[selected.value]
            if self.selected_conversation_flow is None:
                self.selected_conversation_flow = selected_flow
            elif self.selected_conversation_flow != selected_flow:
                raise ValueError("selected_conversation_flow must match selected_conversation_type")
        if self.selected_conversation_flow is not None and not self.selected_conversation_flow.strip():
            raise ValueError("selected_conversation_flow must be non-empty when provided")
        if self.persona_signal_level is not None and self.persona_signal_level not in {"low", "medium", "high"}:
            raise ValueError("persona_signal_level must be low, medium, or high")
        if self.persona_signal_guidance is not None and not self.persona_signal_guidance.strip():
            raise ValueError("persona_signal_guidance must be non-empty when provided")
        if len(self.messages) < MIN_CONVERSATION_MESSAGES:
            raise ValueError("messages must contain at least three complete turns")
        if len(self.messages) % 2:
            raise ValueError("messages must contain complete user/assistant turns")
        expected_roles = ["user", "assistant"]
        for index, message in enumerate(self.messages):
            if message.role != expected_roles[index % 2]:
                raise ValueError("messages must start with user and alternate user/assistant")
        if self.turn_count != count_conversation_turns(self.messages):
            raise ValueError("turn_count must equal the number of user/assistant turns")
        if self.token_count != count_conversation_tokens(self.messages):
            raise ValueError("token_count must equal the computed message content token count")
        return self


class QAAnswer(ArtifactModel):
    answer_id: str
    text: str

    @field_validator("text")
    @classmethod
    def answer_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer text must be non-empty")
        return value


class QAQuestion(ArtifactModel):
    question_id: str
    question: str
    task_form: TaskForm | None = None
    correct_answers: list[QAAnswer]
    incorrect_answers: list[QAAnswer]

    @field_validator("question")
    @classmethod
    def question_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must be non-empty")
        return value

    @model_validator(mode="after")
    def answers_non_empty(self) -> "QAQuestion":
        if not self.correct_answers:
            raise ValueError("question must have at least one correct answer")
        if not self.incorrect_answers:
            raise ValueError("question must have at least one incorrect answer")
        return self


class CaseQA(ArtifactModel):
    qa_id: str
    persona_id: str
    case_id: str
    topic_preference: str
    relation_type: RelationType
    relation_subtype: RelationSubtype
    qa_mode: QAMode = QAMode.QUESTION
    session_ids: list[str]
    questions: list[QAQuestion]
    generation_model: str
    stream_completed: bool
    created_at: str

    @field_validator("questions")
    @classmethod
    def questions_non_empty(cls, value: list[QAQuestion]) -> list[QAQuestion]:
        if not value:
            raise ValueError("case QA must contain at least one question")
        return value

    @model_validator(mode="after")
    def relation_subtype_is_compatible(self) -> "CaseQA":
        relation = RelationType(self.relation_type)
        subtype = RelationSubtype(self.relation_subtype)
        if subtype not in RELATION_SUBTYPE_MAP[relation]:
            raise ValueError(f"{subtype.value} is incompatible with {relation.value}")
        return self


class QAFilterDecision(ArtifactModel):
    decision_id: str
    qa_id: str
    accepted: bool
    reason: str = ""
    rejected_questions: list[dict[str, Any]] = Field(default_factory=list)
    removed_answers: list[dict[str, Any]] = Field(default_factory=list)
    filter_model: str
    stream_completed: bool
    created_at: str

    @model_validator(mode="after")
    def rejected_qa_has_reason(self) -> "QAFilterDecision":
        if not self.accepted and not self.reason.strip():
            raise ValueError("rejected QA must have a reason")
        return self


class AcceptedCaseQA(ArtifactModel):
    qa: CaseQA
    filter_decision: QAFilterDecision

    @model_validator(mode="after")
    def decision_accepts_case_qa(self) -> "AcceptedCaseQA":
        if self.qa.qa_id != self.filter_decision.qa_id:
            raise ValueError("accepted QA and decision IDs differ")
        if not self.filter_decision.accepted:
            raise ValueError("accepted QA requires an accepting decision")
        return self


class RejectedCaseQA(ArtifactModel):
    original_qa: CaseQA
    filtered_qa: CaseQA | None = None
    filter_decision: QAFilterDecision

    @model_validator(mode="after")
    def decision_matches_case_qa(self) -> "RejectedCaseQA":
        if self.original_qa.qa_id != self.filter_decision.qa_id:
            raise ValueError("rejected QA and decision IDs differ")
        if self.filtered_qa is not None and self.filtered_qa.qa_id != self.original_qa.qa_id:
            raise ValueError("filtered QA and original QA IDs differ")
        return self


class EvaluationInstance(ArtifactModel):
    instance_id: str
    persona_id: str
    persona: SanitizedPersonaProfile
    topic_preference: str
    relation_type: RelationType
    relation_subtype: RelationSubtype
    qa_mode: QAMode = QAMode.QUESTION
    case_id: str
    case: MemoryCase
    facts: list[Fact]
    sessions: list[ConversationSession]
    qa_id: str
    question_id: str
    query: str
    task_form: TaskForm | None = None
    correct_answers: list[QAAnswer]
    incorrect_answers: list[QAAnswer]
    case_filter_decision: FilterDecision
    qa_filter_decision: QAFilterDecision
    created_at: str

    @field_validator("query")
    @classmethod
    def query_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must be non-empty")
        return value

    @field_validator("facts")
    @classmethod
    def final_facts_non_empty(cls, value: list[Fact]) -> list[Fact]:
        if not value:
            raise ValueError("evaluation instance must include facts")
        return value

    @field_validator("sessions")
    @classmethod
    def final_sessions_non_empty(cls, value: list[ConversationSession]) -> list[ConversationSession]:
        if not value:
            raise ValueError("evaluation instance must include sessions")
        return value

    @model_validator(mode="after")
    def ids_are_consistent(self) -> "EvaluationInstance":
        if self.persona.persona_id != self.persona_id:
            raise ValueError("persona_id does not match embedded persona")
        if self.case.case_id != self.case_id:
            raise ValueError("case_id does not match embedded case")
        if self.case.persona_id != self.persona_id:
            raise ValueError("case persona_id does not match instance persona_id")
        if self.case.topic_preference != self.topic_preference:
            raise ValueError("topic_preference does not match embedded case")
        if self.case.relation_type != self.relation_type:
            raise ValueError("relation_type does not match embedded case")
        if self.case.relation_subtype != self.relation_subtype:
            raise ValueError("relation_subtype does not match embedded case")
        if self.case_filter_decision.case_id != self.case_id or not self.case_filter_decision.accepted:
            raise ValueError("case_filter_decision must accept the embedded case")
        if self.qa_filter_decision.qa_id != self.qa_id or not self.qa_filter_decision.accepted:
            raise ValueError("qa_filter_decision must accept the embedded QA")
        case_fact_ids = {fact.fact_id for fact in self.case.facts}
        if {fact.fact_id for fact in self.facts} != case_fact_ids:
            raise ValueError("facts must match embedded case facts")
        if any(session.case_id != self.case_id for session in self.sessions):
            raise ValueError("all sessions must belong to the embedded case")
        if not self.correct_answers:
            raise ValueError("evaluation instance must include correct answers")
        if not self.incorrect_answers:
            raise ValueError("evaluation instance must include incorrect answers")
        return self


class RunReport(ArtifactModel):
    run_id: str
    started_at: str
    completed_at: str | None = None
    status: RunStatus
    input_dir: str
    output_dir: str
    config_path: str | None = None
    generation_model: str
    filter_model: str
    qa_mode: QAMode = QAMode.QUESTION
    persona_ids: list[str] = Field(default_factory=list)
    branched_from: bool = False
    source_run: str | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    pass_rate: float = 0.0
    errors: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def pass_rate_matches_counts(self) -> "RunReport":
        generated = self.counts.get("generated_cases", 0)
        accepted = self.counts.get("accepted_cases", 0)
        expected = accepted / generated if generated else 0.0
        if abs(self.pass_rate - expected) > 1e-9:
            raise ValueError("pass_rate must match accepted_cases / generated_cases")
        return self
