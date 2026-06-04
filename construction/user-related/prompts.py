from __future__ import annotations

import json
from typing import Any

from core.schemas import MIN_CONVERSATION_MESSAGES, MIN_CONVERSATION_TURNS, ConversationType, RelationSubtype, RelationType


PROMPT_VERSION = "user-related-v2"
FILTER_ACCEPT_TOKEN = "ACCEPT"
FILTER_REJECT_TOKEN = "REJECT"

CANONICAL_CONVERSATION_TYPES = [item.value for item in ConversationType]

CONVERSATION_TYPE_DESCRIPTIONS = {
    "decision_support": "An agent session where the user compares options, weighs tradeoffs, clarifies criteria, and decides what to do.",
    "planning_coordination": "An agent session where the user organizes steps, timing, dependencies, preparation, responsibilities, or contingencies.",
    "troubleshooting": "An agent session where the user diagnoses a practical problem, provides symptoms or constraints, and works toward next steps.",
    "learning_explanation": "An agent session where the user asks for an explanation, mechanism, background, example, or applied understanding.",
    "resource_selection": "An agent session where the user chooses among tools, places, courses, materials, activities, routes, services, or other resources.",
    "workflow_setup": "An agent session where the user builds a repeatable routine, checklist, tracking approach, template, or operating pattern.",
    "information_organization": "An agent session where the user sorts notes, options, evidence, constraints, ideas, or action items into a clearer structure.",
    "personal_reflection": "An agent session where the user reasons about personal patterns, boundaries, preferences, energy, values, or decision principles.",
    "artifact_production": "An agent session where the main task is producing a new text or content artifact, such as an email, message, post, paragraph, or script.",
    "artifact_review_or_localization": "An agent session where the main task is revising, shortening, translating, localizing, or changing the tone of an existing artifact.",
}

CONVERSATION_TYPE_FLOW_DESCRIPTIONS = {
    "decision_support": (
        "Start with a practical choice, then let criteria, tradeoffs, and one final boundary emerge before the assistant gives a decision rule.",
        "Move from a rough dilemma to narrowed options, with the user adding one constraint at a time as the assistant tests the fit.",
        "Let the assistant compare options, the user correct the weighting, and the ending settle on a clear next action.",
    ),
    "planning_coordination": (
        "Start with an upcoming task, then build sequence, dependencies, and timing through short user clarifications.",
        "Let the assistant sketch a plan, the user reveal constraints or missing pieces, and the final turn tighten the plan.",
        "Move from scattered preparation to a compact order of operations, with responsibilities or contingencies clarified late.",
    ),
    "troubleshooting": (
        "Start with a symptom or blocked task, then narrow likely causes through observations and constraints before choosing next steps.",
        "Let the assistant propose a check, the user report what is or is not true, and the session end with a practical fix path.",
        "Move from confusion to diagnosis, keeping each user turn focused on one clue, failed attempt, or operating condition.",
    ),
    "learning_explanation": (
        "Start with a concept or situation the user wants to understand, then refine the explanation through examples and limits.",
        "Let the assistant explain simply first, the user ask for a more applied version, and the final answer connect it to use.",
        "Move from background explanation to a concrete example, with the user adding what kind of understanding would help.",
    ),
    "resource_selection": (
        "Start with a need for options, then reveal constraints, selection criteria, and a reason to rule choices in or out.",
        "Let the assistant suggest categories first, the user narrow the context, and the ending recommend a short ranked set.",
        "Move from broad resource search to a best-fit choice by adding budget, setting, time, skill, access, or taste constraints.",
    ),
    "workflow_setup": (
        "Start with a recurring friction point, then turn it into a repeatable routine, checklist, or tracking pattern.",
        "Let the assistant propose a structure, the user identify what would break it, and the final answer simplify the workflow.",
        "Move from a messy process to a usable template, with each user turn adding one real-world constraint.",
    ),
    "information_organization": (
        "Start with scattered notes, options, or evidence, then sort them into categories, priorities, or action items.",
        "Let the assistant impose an initial structure, the user add missing context, and the final answer reorganize the result.",
        "Move from an unstructured list to a clearer map, with the user clarifying what distinction matters most.",
    ),
    "personal_reflection": (
        "Start with a pattern the user has noticed, then use concrete recent examples to name a boundary or decision principle.",
        "Let the assistant offer an interpretation, the user accept part of it and correct part of it, then land on a usable framing.",
        "Move from a vague feeling to a grounded self-read, without turning the session into a polished essay or generic advice.",
    ),
    "artifact_production": (
        "Start with a simple artifact request, then reveal tone, audience, and constraints across later turns before producing the final text.",
        "Let the assistant draft a first version, the user add one or two practical constraints, and the final version become more specific.",
        "Move from purpose to usable artifact, with the user keeping revisions focused rather than repeatedly reworking the same text.",
    ),
    "artifact_review_or_localization": (
        "Start from an existing artifact, then revise purpose, audience, tone, or language through a small number of targeted changes.",
        "Let the assistant diagnose what is not working, the user clarify the desired effect, and the final version adjust accordingly.",
        "Move from review to localized or tightened output, with the user adding context gradually instead of all at once.",
    ),
}

PERSONA_SIGNAL_LEVEL_GUIDANCE = {
    "low": "Keep the user's wording plain, task-first, and practical. Use only the personal context needed for the immediate task, with minimal distinctive persona texture.",
    "medium": "Let the user's voice include a modest amount of natural personal context, phrasing, or priorities when useful for the task. Keep the session centered on the agent task, with persona texture emerging through ordinary needs, examples, and constraints.",
    "high": "Let the user's voice, background texture, and task framing reflect the broader persona clearly while staying natural and relevant to the agent task. Use richer personal context across turns, with persona texture expressed through concrete details and priorities.",
}

PERSONA_SIGNAL_LEVELS = tuple(PERSONA_SIGNAL_LEVEL_GUIDANCE)

CASE_SPECS = [
    (RelationType.COMPLEMENTARY.value, RelationSubtype.COMPLEMENTARY_K1.value),
    (RelationType.COMPLEMENTARY.value, RelationSubtype.COMPLEMENTARY_K_GT_1.value),
    (RelationType.COMPLEMENTARY.value, RelationSubtype.COMPLEMENTARY_ANY_ONE.value),
    (RelationType.NUANCED.value, RelationSubtype.NUANCED_TEMPORAL.value),
    (RelationType.NUANCED.value, RelationSubtype.NUANCED_CONTEXT.value),
    (RelationType.CONTRADICTORY.value, RelationSubtype.CONTRADICTORY.value),
]

ACTIVE_CASE_SPECS = [
    (RelationType.COMPLEMENTARY.value, RelationSubtype.COMPLEMENTARY_K_GT_1.value),
    (RelationType.COMPLEMENTARY.value, RelationSubtype.COMPLEMENTARY_ANY_ONE.value),
    (RelationType.NUANCED.value, RelationSubtype.NUANCED_TEMPORAL.value),
    (RelationType.NUANCED.value, RelationSubtype.NUANCED_CONTEXT.value),
    (RelationType.CONTRADICTORY.value, RelationSubtype.CONTRADICTORY.value),
]

CASE_RELATION_TYPE_ORDER = (
    RelationType.NUANCED.value,
    RelationType.CONTRADICTORY.value,
    RelationType.COMPLEMENTARY.value,
)

CASE_RELATION_SUBTYPE_ORDER = {
    RelationType.COMPLEMENTARY.value: (
        RelationSubtype.COMPLEMENTARY_K_GT_1.value,
        RelationSubtype.COMPLEMENTARY_ANY_ONE.value,
        RelationSubtype.COMPLEMENTARY_K1.value,
    ),
    RelationType.NUANCED.value: (
        RelationSubtype.NUANCED_CONTEXT.value,
        RelationSubtype.NUANCED_TEMPORAL.value,
    ),
    RelationType.CONTRADICTORY.value: (
        RelationSubtype.CONTRADICTORY.value,
    ),
}

ACTIVE_CASE_RELATION_SUBTYPE_ORDER = {
    RelationType.COMPLEMENTARY.value: (
        RelationSubtype.COMPLEMENTARY_K_GT_1.value,
        RelationSubtype.COMPLEMENTARY_ANY_ONE.value,
    ),
    RelationType.NUANCED.value: CASE_RELATION_SUBTYPE_ORDER[RelationType.NUANCED.value],
    RelationType.CONTRADICTORY.value: CASE_RELATION_SUBTYPE_ORDER[RelationType.CONTRADICTORY.value],
}

CASE_SUBTYPE_DISPLAY_ORDER = tuple(dict.fromkeys(relation_subtype for _, relation_subtype in CASE_SPECS))
ACTIVE_CASE_SUBTYPE_DISPLAY_ORDER = tuple(dict.fromkeys(relation_subtype for _, relation_subtype in ACTIVE_CASE_SPECS))

CASE_RELATION_TYPE_DEFINITIONS = {
    "complementary": "Multiple memory items are related to the same question or topic and do not conflict, but they may differ in whether they determine the answer. Some may be background, side attributes, supplementary details, or nearby information that can distract retrieval but does not require resolving multiple valid answers.",
    "nuanced": "Multiple memory items are relevant to the same question and can all be true, but only under different qualifying conditions such as time, role, location, task, scope, version, definition, or attribute. The answer should identify the decisive condition and, when the question specifies one, return the answer for that condition.",
    "contradictory": "Multiple memory items cannot all be true under the same interpretation and conditions, and cannot be reconciled by adding qualifiers. This includes genuinely incompatible states and inconsistencies caused by erroneous facts; answers should point out the conflict rather than answer from only one side.",
}

CASE_DEFINITIONS = {
    "K=1": "Complementary subtype: exactly one memory item is decisive for the question; the other same-topic items are related background, nearby details, or distractors and should not determine the answer.",
    "K>1": "Complementary subtype: the question must be addressed by combining K memory items, where K is greater than one and may be the full set; this tests multi-hop reasoning or summarization over compatible memories.",
    "any_one": "Complementary subtype: any one memory item in the set can address the same question or intent, often as different natural expressions of the same underlying fact.",
    "Temporal": "Nuanced subtype: memory items differ because facts changed over time, and the answer depends on the stated or inferable time condition.",
    "Context": "Nuanced subtype: memory items hold under different contexts such as role, location, task, scope, version, definition, or attribute, and the answer depends on that context.",
    "contradictory": "Contradictory subtype: memory items cannot all be true under the same interpretation and conditions, including user-vs-user, user-vs-non-user/tool, or non-user-vs-non-user conflicts when provenance is available; answers should identify the conflict instead of using only one side.",
}

CASE_SUBTYPE_NOTES = {
    "K=1": [
        "The target memory is the decisive signal; other related memories may add color but should not change the answer.",
        "Keep the answer-determining relationship clear: only the decisive memory should be needed.",
        "Do not introduce time, role, setting, scope, or exception conditions that make the answer conditional.",
    ],
    "K>1": [
        "The case memories must work together; the answer should require combining K memories instead of using only one.",
        "Each required memory should contribute a distinct necessary constraint, audience, setting, criterion, or scope; do not make K>1 cases where all facts merely repeat the same broad preference in nearby contexts.",
        "Do not let one generated session make the other required memories unnecessary.",
    ],
    "any_one": [
        "Each memory should be an independent path to the same stable user-facing answer or intent.",
        "Prefer meaningfully different expressions of the same underlying fact or intent.",
        "Do not introduce conditions that make only one path valid.",
    ],
    "Temporal": [
        "The time boundary is relation-critical; preserve it naturally in sessions and QA.",
        "The time boundary must be stated in the memory content or question; artifact order, session order, and timestamps alone are not a temporal condition.",
        "Do not flatten past/current/future differences into one global preference.",
    ],
    "Context": [
        "The role, location, task, scope, version, definition, or attribute boundary is relation-critical; preserve it naturally.",
        "The context split must change the concrete user-facing response; it is not enough for both facts to express the same broad value in different settings.",
        "If another qualifying condition is needed, make that condition explicit rather than treating the memories as one unconditional answer.",
        "Do not average context-specific memory items into one global preference.",
    ],
    "contradictory": [
        "The conflict must remain under the same interpretation and conditions.",
        "Each side of the conflict must still be recoverable from its own session; do not make a session merely topic-adjacent or ambiguous.",
        "Do not resolve the conflict by switching time, role, setting, task scope, formality, or responsibility type.",
        "Do not treat artifact order, session order, or timestamps as resolving the conflict; a later session is not automatically the current truth.",
        "A correct answer should point out the conflicting memory content, uncertainty, or need for clarification instead of choosing one side.",
    ],
}

CASE_RELATION_BOUNDARY_RULES = [
    "Temporal and Context are subtypes of nuanced only.",
    "Do not label simple additive evidence as nuanced; use complementary K>1 when multiple memory items support one stable answer without a time or context condition.",
    "Use contradictory only when memory items cannot all be true under the same interpretation and conditions.",
    "Artifact order, session order, and timestamps are provenance metadata, not time conditions by themselves; use Temporal only when the memory content or question makes time relation-critical.",
]

QA_CASE_FIELDS = ("topic_preference", "description", "facts")
QA_SESSION_FIELDS = ("conversation_id", "messages")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _relation_guidance() -> str:
    return "\n".join(f"- {rule}" for rule in CASE_RELATION_BOUNDARY_RULES)


def _case_relation_type_targets(target_distribution: list[dict[str, Any]]) -> dict[str, int]:
    counts = {relation_type: 0 for relation_type in CASE_RELATION_TYPE_ORDER}
    for item in target_distribution:
        relation_type = str(item.get("relation_type") or "").strip()
        target_count = int(item.get("target_count") or 0)
        if relation_type in counts and target_count > 0:
            counts[relation_type] += target_count
    return {relation_type: counts[relation_type] for relation_type in CASE_RELATION_TYPE_ORDER if counts[relation_type] > 0}


def _ordered_case_subtypes(
    subtypes: list[str] | tuple[str, ...] | None = None,
    *,
    active_only: bool = False,
) -> list[str]:
    if subtypes is None:
        default_order = ACTIVE_CASE_SUBTYPE_DISPLAY_ORDER if active_only else CASE_SUBTYPE_DISPLAY_ORDER
        return list(default_order)
    requested = {str(subtype).strip() for subtype in subtypes}
    return [subtype for subtype in CASE_SUBTYPE_DISPLAY_ORDER if subtype in requested]


def _case_subtype_definitions_view(
    subtypes: list[str] | tuple[str, ...] | None = None,
    *,
    active_only: bool = False,
) -> dict[str, str]:
    return {
        subtype: CASE_DEFINITIONS[subtype]
        for subtype in _ordered_case_subtypes(subtypes, active_only=active_only)
        if subtype in CASE_DEFINITIONS
    }


def _case_subtype_notes_view(
    subtypes: list[str] | tuple[str, ...] | None = None,
    *,
    active_only: bool = False,
) -> dict[str, list[str]]:
    return {
        subtype: CASE_SUBTYPE_NOTES[subtype]
        for subtype in _ordered_case_subtypes(subtypes, active_only=active_only)
        if subtype in CASE_SUBTYPE_NOTES
    }


def _batch_relation_subtypes(batch_context: dict[str, Any]) -> list[str]:
    batch_subtypes: list[str] = []
    for item in batch_context.get("cases", []):
        if not isinstance(item, dict):
            continue
        relation_subtype = str(item.get("relation_subtype") or "").strip()
        if relation_subtype in CASE_DEFINITIONS:
            batch_subtypes.append(relation_subtype)
    ordered = _ordered_case_subtypes(batch_subtypes)
    return ordered or list(ACTIVE_CASE_SUBTYPE_DISPLAY_ORDER)


def _case_relation_subtype_balancing_policy() -> list[dict[str, Any]]:
    return [
        {
            "relation_type": relation_type,
            "subtypes_in_priority_order": list(ACTIVE_CASE_RELATION_SUBTYPE_ORDER[relation_type]),
            "rule": "Split assignments as evenly as possible within this relation_type. If one extra assignment remains, give it to the earliest subtype in the listed order.",
        }
        for relation_type in CASE_RELATION_TYPE_ORDER
    ]


def _relation_candidate_with_meaning(candidate: dict[str, str]) -> dict[str, str]:
    relation_type = str(candidate["relation_type"])
    relation_subtype = str(candidate["relation_subtype"])
    return {
        **candidate,
        "relation_type_meaning": CASE_RELATION_TYPE_DEFINITIONS[relation_type],
        "relation_subtype_meaning": CASE_DEFINITIONS[relation_subtype],
    }


def build_current_case_relation_guidance(memory_case: dict[str, Any]) -> dict[str, Any]:
    relation_type = str(memory_case.get("relation_type") or "")
    relation_subtype = str(memory_case.get("relation_subtype") or "")
    guidance: dict[str, Any] = {}
    if relation_type:
        guidance["relation_type"] = relation_type
        if relation_type in CASE_RELATION_TYPE_DEFINITIONS:
            guidance["relation_type_meaning"] = CASE_RELATION_TYPE_DEFINITIONS[relation_type]
    if relation_subtype:
        guidance["relation_subtype"] = relation_subtype
        if relation_subtype in CASE_DEFINITIONS:
            guidance["relation_subtype_meaning"] = CASE_DEFINITIONS[relation_subtype]
        if relation_subtype in CASE_SUBTYPE_NOTES:
            guidance["relation_subtype_notes"] = CASE_SUBTYPE_NOTES[relation_subtype]
    return guidance


def build_current_case_conversation_guidance(memory_case: dict[str, Any]) -> str:
    return _relation_conversation_guidance(memory_case)


def _qa_case_view(memory_case: dict[str, Any]) -> dict[str, Any]:
    return {field: memory_case[field] for field in QA_CASE_FIELDS if field in memory_case}


def _qa_case_relation_view(memory_case: dict[str, Any]) -> dict[str, Any]:
    return build_current_case_relation_guidance(memory_case)


def _case_relation_type(memory_case: dict[str, Any]) -> str:
    return str(memory_case.get("relation_type") or "")


def _case_relation_subtype(memory_case: dict[str, Any]) -> str:
    return str(memory_case.get("relation_subtype") or "")


def _relation_conversation_guidance(memory_case: dict[str, Any]) -> str:
    relation_type = _case_relation_type(memory_case)
    relation_subtype = _case_relation_subtype(memory_case)
    if relation_type == RelationType.COMPLEMENTARY.value:
        subtype_guidance = {
            RelationSubtype.COMPLEMENTARY_K1.value: "- K=1: let the target memory be decisive; nearby memories may be related but should not change the answer.",
            RelationSubtype.COMPLEMENTARY_K_GT_1.value: "- K>1: keep this memory as evidence that must combine with the other required case memories.",
            RelationSubtype.COMPLEMENTARY_ANY_ONE.value: "- any_one: keep this memory as one independent path to the same stable answer as the other case memories.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific conversation guidance:",
            "- Keep the case memory items compatible, with no conflict or condition that requires resolution.",
            "- Do not add time, role, setting, scope, or exception context that would make the answer conditional.",
            "- Do not introduce a conflict that undermines another case memory.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.NUANCED.value:
        subtype_guidance = {
            RelationSubtype.NUANCED_TEMPORAL.value: "- Temporal: keep the relevant time boundary natural and clear.",
            RelationSubtype.NUANCED_CONTEXT.value: "- Context: keep the relevant role, location, task, scope, version, definition, or attribute boundary natural and clear.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific conversation guidance:",
            "- Preserve the time or context condition that changes the answer.",
            "- For Temporal cases, make the time boundary explicit in the content instead of relying on session order or timestamps.",
            "- Do not collapse the memory items into one global preference or average them into a generic answer.",
            "- The session should make the relevant condition natural without exposing labels or hidden facts.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.CONTRADICTORY.value:
        return "\n".join(
            [
                "Relation-specific conversation guidance:",
                "- Preserve the contradiction as a same-condition conflict across the case.",
                "- Make the target side of the conflict recoverable from this session through a concrete user choice, boundary, complaint, plan, example, correction, or repeated practical need; do not leave it as a vague mood or incidental topic mention.",
                "- Do not add time, role, setting, task-scope, formality, or responsibility-boundary context that would make the target memory compatible with the other case memories.",
                "- Do not frame one side as newer, current, latest, old, or superseded; session order must not resolve the contradiction.",
                "- The session may express the target memory naturally, but it must not reinterpret it as only true in a special context.",
            ]
        )
    return ""


def _relation_question_guidance(memory_case: dict[str, Any]) -> str:
    relation_type = _case_relation_type(memory_case)
    relation_subtype = _case_relation_subtype(memory_case)
    if relation_type == RelationType.COMPLEMENTARY.value:
        subtype_guidance = {
            RelationSubtype.COMPLEMENTARY_K1.value: "- K=1: the question should allow the decisive memory to support the answer without needing extra hidden evidence; ask a stable recommendation, choice, or explanation where related memories remain non-decisive.",
            RelationSubtype.COMPLEMENTARY_K_GT_1.value: "- K>1: the question should require combining the case memories, not selecting only one; ask for a plan, recommendation, summary, or explanation that requires synthesizing compatible memories.",
            RelationSubtype.COMPLEMENTARY_ANY_ONE.value: "- any_one: the question should allow any case memory to support the same intent through a different expression; ask one stable intent that does not require choosing among conditions.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific question guidance:",
            "- Questions should invite one stable personalized answer from compatible memory items.",
            "- Do not make the question depend on a time, role, setting, or scope split.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.NUANCED.value:
        subtype_guidance = {
            RelationSubtype.NUANCED_TEMPORAL.value: "- Temporal: include only the neutral time condition needed to select the right answer; ask about the past, current, future, or before/after condition only when that time boundary is supported by the memory content.",
            RelationSubtype.NUANCED_CONTEXT.value: "- Context: include only the neutral role, location, task, scope, version, definition, or attribute condition needed to select one right answer. The condition must genuinely decide between otherwise plausible case facts; do not use a framing where one fact is already obviously decisive and the others are only background. If a context is given, make it implicit or lightly embedded rather than obvious or label-like, but the question must still resolve to one context-specific fact.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific question guidance:",
            "- Each question must target exactly one applicable memory item by specifying one decisive temporal or contextual condition.",
            "- Across the generated questions for the same nuanced case, different questions may target different condition-specific memory items.",
            "- Questions should make the relevant condition clear enough for a personalized answer.",
            "- For Temporal cases, the question should use a neutral time condition from the memory content, not merely earlier/later session order.",
            "- For Temporal cases, do not ask for a then-and-now card, timeline, comparison, full recap, or entries for multiple time periods.",
            "- For Context cases, do not ask for multiple contexts, separate fields for several scenarios, or a complete list of context-conditioned answers.",
            "- Do not flatten condition-specific facts into a single global preference.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.CONTRADICTORY.value:
        return "\n".join(
            [
                "Relation-specific question guidance:",
                "- Contradictory directions: ask for advice, explanation, a decision, or a next step where acting on only one memory item could be wrong.",
                "- The question may ask what to do, what is safe to assume, or what should be clarified before proceeding, but it should not announce the hidden conflict directly.",
                "- They should not be answerable by choosing only one memory item.",
                "- They should not be answerable by choosing the latest timestamp or later session.",
                "- Correct answers should surface that the memory items conflict, cannot both be true as stated, or need clarification.",
            ]
        )
    return ""


def _relation_answer_guidance(memory_case: dict[str, Any]) -> str:
    relation_type = _case_relation_type(memory_case)
    relation_subtype = _case_relation_subtype(memory_case)
    if relation_type == RelationType.COMPLEMENTARY.value:
        subtype_guidance = {
            RelationSubtype.COMPLEMENTARY_K1.value: "- K=1: correct answers should follow the decisive memory; incorrect answers should contradict or miss it.",
            RelationSubtype.COMPLEMENTARY_K_GT_1.value: "- K>1: correct answers must combine the required memories; answers using only one required memory are incomplete.",
            RelationSubtype.COMPLEMENTARY_ANY_ONE.value: "- any_one: correct answers may use any valid memory path, but all paths should support the same user-facing answer.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific answer guidance:",
            "- Correct answers should give one stable personalized response from compatible memory items.",
            "- Incorrect answers should be plausible but misaligned with that stable response.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.NUANCED.value:
        subtype_guidance = {
            RelationSubtype.NUANCED_TEMPORAL.value: "- Temporal: correct answers must use the fact that matches the stated time condition.",
            RelationSubtype.NUANCED_CONTEXT.value: "- Context: correct answers must use the fact that matches the stated role, location, task, scope, version, definition, or attribute condition. Incorrect answers should preferably reflect a plausible case fact from the wrong context, not an unrelated distractor.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific answer guidance:",
            "- Correct answers must use only the memory item that matches the question's single decisive condition.",
            "- Correct answers must respect the condition in the question.",
            "- Correct answers must not provide a then-and-now answer, full timeline, all contexts, or multiple condition-specific answers.",
            "- Incorrect answers can apply the wrong condition or ignore the condition and give a generic answer.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.CONTRADICTORY.value:
        return "\n".join(
            [
                "Relation-specific answer guidance:",
                "- Correct answers must explicitly acknowledge that the memory items conflict, cannot both be true as stated, or require clarification.",
                "- A correct answer must not rely on only one memory item, choose the later timestamp/session, or resolve the conflict by inventing context.",
                "- Incorrect answers should choose one side, ignore the contradiction, or add unsupported context that makes both memory items fit.",
            ]
        )
    return ""


def _relation_task_guidance(memory_case: dict[str, Any]) -> str:
    relation_type = _case_relation_type(memory_case)
    relation_subtype = _case_relation_subtype(memory_case)
    if relation_type == RelationType.COMPLEMENTARY.value:
        subtype_guidance = {
            RelationSubtype.COMPLEMENTARY_K1.value: "- K=1: the request should be completable from the decisive memory without requiring extra hidden evidence; use one natural request where related memories remain background or distractors.",
            RelationSubtype.COMPLEMENTARY_K_GT_1.value: "- K>1: the request should require combining the case memories, not selecting only one; use a natural request where synthesis is necessary to complete it correctly.",
            RelationSubtype.COMPLEMENTARY_ANY_ONE.value: "- any_one: the request should permit any case memory to support the same completion through a different expression; use one stable request that does not require choosing among conditions.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific task guidance:",
            "- Task prompts should present one natural user request whose correct completion depends on recovering or using the right compatible memory items.",
            "- Do not make the request depend on a time, role, setting, or scope split.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.NUANCED.value:
        subtype_guidance = {
            RelationSubtype.NUANCED_TEMPORAL.value: "- Temporal: include only the neutral time condition needed to select the right completion; frame the task so the time boundary is supported by the memory content.",
            RelationSubtype.NUANCED_CONTEXT.value: "- Context: include only the neutral role, location, task, scope, version, definition, or attribute condition needed to select the right completion. The condition must genuinely decide between otherwise plausible case facts; do not use a framing where one fact is already obviously decisive and the others are only background. If a context is given, embed it lightly in the task rather than making it explicit or label-like.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific task guidance:",
            "- Each task prompt must target exactly one applicable memory item by specifying one decisive temporal or contextual condition.",
            "- Across the generated task prompts for the same nuanced case, different prompts may target different condition-specific memory items.",
            "- Task prompts should make the relevant condition clear enough for the agent to select the right fact and complete the task correctly.",
            "- For Temporal cases, the task should use a neutral time condition from the memory content, not merely earlier/later session order.",
            "- For Temporal cases, do not ask for then-and-now fields, a timeline card, a before/after comparison, or entries for multiple periods.",
            "- For Context cases, the task context must change what the response should recommend, include, exclude, or clarify; do not use a context label when the same answer would work in every context.",
            "- For Context cases, do not ask the agent to fill or arrange outputs for multiple contexts in the same task.",
            "- Do not flatten condition-specific facts into a single global preference.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.CONTRADICTORY.value:
        return "\n".join(
            [
                "Relation-specific task guidance:",
                "- Use a natural user request where safe completion depends on noticing that acting on only one memory item could be wrong.",
                "- The request should create a real need to handle uncertainty, but it should not announce the hidden conflict directly.",
                "- It should not be completable by choosing only one memory item.",
                "- It should not be completable by choosing the latest timestamp or later session.",
                "- Correct completions should surface that the memory items conflict, cannot both be true as stated, or need clarification.",
            ]
        )
    return ""


def _relation_task_answer_guidance(memory_case: dict[str, Any]) -> str:
    relation_type = _case_relation_type(memory_case)
    relation_subtype = _case_relation_subtype(memory_case)
    if relation_type == RelationType.COMPLEMENTARY.value:
        subtype_guidance = {
            RelationSubtype.COMPLEMENTARY_K1.value: "- K=1: correct answers should capture the decisive memory; incorrect answers should point to directions that contradict or miss it.",
            RelationSubtype.COMPLEMENTARY_K_GT_1.value: "- K>1: correct answers must reflect the required memories together; answers that can be satisfied with only one required memory are incomplete.",
            RelationSubtype.COMPLEMENTARY_ANY_ONE.value: "- any_one: correct answers may reflect any valid memory path, but all paths should support the same user-facing result.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific task-answer guidance:",
            "- Correct answers should capture the fact alignment a good response must reflect from compatible memory items.",
            "- Incorrect answers should be plausible but point to the wrong fact use or an incomplete fact basis.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.NUANCED.value:
        subtype_guidance = {
            RelationSubtype.NUANCED_TEMPORAL.value: "- Temporal: correct answers must match the fact that fits the stated time condition.",
            RelationSubtype.NUANCED_CONTEXT.value: "- Context: correct answers must match the fact that fits the stated role, location, task, scope, version, definition, or attribute condition. Incorrect answers should preferably reflect a plausible case fact from the wrong context, not an unrelated distractor.",
        }.get(relation_subtype, "")
        lines = [
            "Relation-specific task-answer guidance:",
            "- Correct answers must use only the memory item that matches the task prompt's single decisive condition.",
            "- Correct answers must respect the condition in the task prompt and match the fact that fits it.",
            "- Correct answers must not provide a then-and-now answer, full timeline, all contexts, or multiple condition-specific completions.",
            "- Incorrect answers can apply the wrong condition or point to a generic response that ignores the condition.",
        ]
        if subtype_guidance:
            lines.append(subtype_guidance)
        return "\n".join(lines)
    if relation_type == RelationType.CONTRADICTORY.value:
        return "\n".join(
            [
                "Relation-specific task-answer guidance:",
                "- Correct answers must reflect that the memory items conflict, cannot both be true as stated, or need clarification before safe completion.",
                "- A correct answer must not rely on only one memory item, choose the later timestamp/session, or resolve the conflict by inventing context.",
                "- Incorrect answers should point to choosing one side, ignoring the contradiction, or adding unsupported context that makes both memory items fit.",
            ]
        )
    return ""


def _qa_session_view(session: dict[str, Any]) -> dict[str, Any]:
    return {field: session[field] for field in QA_SESSION_FIELDS if field in session}


def _same_case_session_view(session: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {}
    for field in ("conversation_id", "fact_id", "fact_text", "selected_conversation_type"):
        if field in session:
            view[field] = session[field]
    if "messages" in session:
        view["messages"] = session["messages"]
    return view


def build_categorize_preference_topic_prompt(preference: str, existing_topics: list[str]) -> str:
    if existing_topics:
        existing_topics_str = ", ".join(existing_topics)
        existing_section = f"**Existing Topics:** {existing_topics_str}"
    else:
        existing_section = "**Existing Topics:** None (this is the first preference being categorized)"

    return f"""You are categorizing user preferences into simple topic categories. Existing topics:

    {existing_section}

    **Preference to categorize:** "{preference}"

    **Instructions:**
    1. Read the preference and identify its main topic/theme
    2. Either choose the most appropriate existing topic OR create a new simple topic name
    3. Keep topic names in one word, and at most two words in rare, necessary cases
    4. Use general topics like food, sports, technology, pets, study, work, travel, entertainment, health, etc, avoiding too specific ones
    5. Use specific word for the topic. Do NOT use uncategorized, unknown, undefined, or similar fuzzy words.

    Return the topic name after ###Output."""


def _case_prompt(
    persona_str: str,
    preference: dict[str, Any],
    relation_type: str,
    relation_subtype: str,
    topic_preference: str,
) -> str:
    definition = CASE_DEFINITIONS[relation_subtype]
    relation_definition = CASE_RELATION_TYPE_DEFINITIONS[relation_type]
    return f"""
Given the persona and one source preference, write one natural memory case for the same user according to the relation type and subtype.

Persona:
{persona_str}

Topic preference:
{topic_preference}

Source preference:
{_json(preference)}

Relation type definitions:
{_json(CASE_RELATION_TYPE_DEFINITIONS)}

Boundary rules:
{_relation_guidance()}

Relation type:
{relation_type}

Relation type meaning:
{relation_definition}

Relation subtype:
{relation_subtype}

Subtype meaning:
{definition}

Requirements:
- Write one concise natural description of the case.
- Write precise one-sentence facts.
- Facts must be about this user and must not mention labels, task framing, or this instruction.
- Keep the case faithful to the persona and source preference.

Think step by step.

###Output
Return only JSON:
{{
  "description": "...",
  "facts": ["...", "..."]
}}
""".strip()


def build_case_prompt(
    persona_str: str,
    preference: dict[str, Any],
    relation_type: str,
    relation_subtype: str,
    topic_preference: str,
) -> str:
    return _case_prompt(persona_str, preference, relation_type, relation_subtype, topic_preference)


def build_topic_merge_prompt(persona_str: str, topic_groups: list[dict[str, Any]]) -> str:
    return f"""
Merge topic labels that mean the same user-interest area.

Persona:
{persona_str}

Topic groups:
{_json(topic_groups)}

Requirements:
- Keep distinct topics separate when they imply different user interests.
- Merge labels that only differ by wording, plurality, spelling, or narrow phrasing.
- Use short natural canonical topic names.
- Every source topic must appear exactly once in the mapping.

Think step by step.

###Output
Return only JSON:
{{
  "topic_mapping": [
    {{"source_topic": "...", "canonical_topic": "..."}}
  ]
}}
""".strip()


def build_case_relation_plan_prompt(
    persona_str: str,
    persona_id: str,
    preferences: list[dict[str, Any]],
    target_distribution: list[dict[str, Any]],
) -> str:
    preference_view = [
        {
            "preference_id": item.get("preference_id"),
            "topic_preference": item.get("topic_preference"),
            "preference_text": item.get("preference_text"),
            "pref_type": item.get("pref_type"),
        }
        for item in preferences
    ]
    return f"""
Plan the relation type and subtype for each source preference before memory case generation.

Persona:
{persona_str}

Persona id:
{persona_id}

Relation type definitions:
{_json(CASE_RELATION_TYPE_DEFINITIONS)}

Relation subtype definitions:
{_json(_case_subtype_definitions_view(active_only=True))}

Boundary rules:
{_relation_guidance()}

Target relation type counts:
{_json(_case_relation_type_targets(target_distribution))}

Subtype balancing policy:
{_json(_case_relation_subtype_balancing_policy())}

Preferences:
{_json(preference_view)}

Requirements:
- Assign exactly one relation_type and relation_subtype to every preference_id.
- Match the target relation type counts exactly.
- Within each relation_type, split relation_subtype assignments as evenly as possible following the fixed subtype order above.
- Use the fixed subtype order to place any remainder within a relation_type.
- Within these quantity constraints, choose the most natural preference-subtype pairing.
- Prefer pairings that can produce faithful, natural cases with the least unsupported invention.
- Do not invent synthetic facts, topics, task labels, or unsupported conflicts.

Think step by step.

###Output
Return only JSON:
{{
  "planned_relations": [
    {{
      "preference_id": "...",
      "relation_type": "...",
      "relation_subtype": "...",
      "planning_reason": "brief reason"
    }}
  ]
}}
""".strip()


def build_case_selection_prompt(
    persona_str: str,
    preference: dict[str, Any],
    topic_preference: str,
    candidate_relations: list[dict[str, str]],
    *,
    preferred_relation: dict[str, str] | None = None,
) -> str:
    required_relation = preferred_relation or candidate_relations[0]
    required = _relation_candidate_with_meaning(required_relation)
    return f"""
Given the persona and one source preference, write one natural memory case using the required relation type and subtype.

Persona:
{persona_str}

Topic preference:
{topic_preference}

Source preference:
{_json(preference)}

Relation type definitions:
{_json(CASE_RELATION_TYPE_DEFINITIONS)}

Boundary rules:
{_relation_guidance()}

Required relation:
{_json(required)}

Requirements:
- Use exactly this relation type and subtype for the case.
- The case must still be natural, faithful to the persona/source preference, and correctly labeled.
- Label Temporal or Context only as nuanced.
- If multiple facts simply support the same stable answer, choose complementary K>1 instead of nuanced Context.
- For K>1 cases, write facts that a plausible single user task must combine. Each fact should add a different required constraint, criterion, audience, setting, or scope; avoid facts that are just repeated examples of the same preference.
- For Context cases, write facts so the same practical user task would require different concrete actions, priorities, or exclusions under each context. Avoid cases where both facts merely show the same general trait.
- Write one concise natural description of the case.
- Write precise one-sentence facts.
- Facts must be about this user and must not mention labels, task framing, or this instruction.
- Keep the case faithful to the persona and source preference.

Think step by step.

###Output
Return only JSON:
{{
  "relation_type": "required relation_type",
  "relation_subtype": "required relation_subtype",
  "relation_choice_reason": "brief reason for how the required relation is expressed naturally",
  "description": "...",
  "facts": ["...", "..."]
}}
""".strip()


def build_complementary_k1_case_prompt(persona_str: str, preference: dict[str, Any], topic_preference: str) -> str:
    return _case_prompt(persona_str, preference, "complementary", "K=1", topic_preference)


def build_complementary_k_gt_1_case_prompt(persona_str: str, preference: dict[str, Any], topic_preference: str) -> str:
    return _case_prompt(persona_str, preference, "complementary", "K>1", topic_preference)


def build_complementary_any_one_case_prompt(persona_str: str, preference: dict[str, Any], topic_preference: str) -> str:
    return _case_prompt(persona_str, preference, "complementary", "any_one", topic_preference)


def build_nuanced_temporal_case_prompt(persona_str: str, preference: dict[str, Any], topic_preference: str) -> str:
    return _case_prompt(persona_str, preference, "nuanced", "Temporal", topic_preference)


def build_nuanced_context_case_prompt(persona_str: str, preference: dict[str, Any], topic_preference: str) -> str:
    return _case_prompt(persona_str, preference, "nuanced", "Context", topic_preference)


def build_contradictory_case_prompt(persona_str: str, preference: dict[str, Any], topic_preference: str) -> str:
    return _case_prompt(persona_str, preference, "contradictory", "contradictory", topic_preference)


def build_filter_prompt(persona_str: str, memory_case: dict[str, Any]) -> str:
    return f"""
Review this user memory case for data quality.

Persona:
{persona_str}

Relation type definitions:
{_json(CASE_RELATION_TYPE_DEFINITIONS)}

Boundary rules:
{_relation_guidance()}

Case:
{_json(memory_case)}

Accept only if the case is natural, faithful to the persona/source preference, correctly labeled, non-duplicative, and internally valid.
Reject if it is unnatural, mislabeled, factually unsupported, duplicated, conflicting outside the contradictory subtype, or impossible to resolve under its stated relation.
Do not reject solely because the case shares a broad or background persona fact with another case when the answer-driving facts, user-facing intent, and relation behavior are otherwise distinct.
Reject Temporal or Context cases if the memory items are only additive evidence for one stable answer and no time/context condition changes the answer.
Reject K>1 cases if any required fact can be dropped without changing the concrete answer, or if the facts merely repeat the same broad preference in nearby contexts.
Reject Context cases if the listed contexts are different on the surface but do not produce meaningfully different user-facing answers, actions, or constraints.
Reject Context cases if both facts mainly express the same broad preference/value and do not force different concrete actions, priorities, or exclusions for a plausible user task.
Reject Temporal cases if the time boundary is not expressed in the case facts themselves.
For contradictory cases, reject if the memory items can all be true together, such as following both African and non-African teams.

Think step by step.

###Output
Return only JSON:
{{
  "final_decision": "{FILTER_ACCEPT_TOKEN} or {FILTER_REJECT_TOKEN}",
  "accepted": true,
  "reason": "short reason",
  "reject_categories": []
}}
""".strip()


def build_topic_filter_prompt(persona_str: str, topic_preference: str, memory_cases: list[dict[str, Any]]) -> str:
    return f"""
Review these user memory cases for one topic.

Persona:
{persona_str}

Topic preference:
{topic_preference}

Relation type definitions:
{_json(CASE_RELATION_TYPE_DEFINITIONS)}

Boundary rules:
{_relation_guidance()}

Cases:
{_json(memory_cases)}

Assess each case by itself and against the other cases in this topic.
Accept only cases that are natural, faithful to the persona/source preference, correctly labeled, non-duplicative, and internally valid.
Reject cases that are unnatural, mislabeled, factually unsupported, duplicated, conflicting with another accepted case outside the contradictory subtype, or impossible to resolve under the stated relation.
Do not reject solely because a case shares a broad or background persona fact with another case when the answer-driving facts, user-facing intent, and relation behavior are otherwise distinct.
Reject Temporal or Context cases if the memory items are only additive evidence for one stable answer and no time/context condition changes the answer.
Reject K>1 cases if any required fact can be dropped without changing the concrete answer, or if the facts merely repeat the same broad preference in nearby contexts.
Reject Context cases if the listed contexts are different on the surface but do not produce meaningfully different user-facing answers, actions, or constraints.
Reject Context cases if both facts mainly express the same broad preference/value and do not force different concrete actions, priorities, or exclusions for a plausible user task.
Reject Temporal cases if the time boundary is not expressed in the case facts themselves.
For contradictory cases, reject if the memory items can all be true together, such as following both African and non-African teams.
Return one decision for every input case_id exactly once.

Think step by step.

###Output
Return only JSON:
{{
  "case_decisions": [
    {{
      "case_id": "...",
      "final_decision": "{FILTER_ACCEPT_TOKEN} or {FILTER_REJECT_TOKEN}",
      "accepted": true,
      "reason": "short reason",
      "reject_categories": []
    }}
  ]
}}
""".strip()


def build_persona_consistency_filter_prompt(persona_str: str, memory_cases: list[dict[str, Any]]) -> str:
    return f"""
Review accepted user memory cases for one persona across all topics.

Persona:
{persona_str}

Accepted cases:
{_json(memory_cases)}

Find only cases that should be rejected because they duplicate another accepted case, contradict another accepted case under the same conditions, or introduce unsupported cross-topic facts.
Do not reject a nuanced case when its condition clearly resolves the difference.
Do not reject solely because a case reuses a broad or background persona fact; shared background is only a problem when it makes the cases substantially redundant, duplicates the answer-driving memory, or creates a same-condition contradiction.
If two cases conflict and one is more specific or better supported, mark the weaker case as problematic.
If there are no cross-topic problems, return an empty problem_cases list.

Think step by step.

###Output
Return only JSON:
{{
  "problem_cases": [
    {{
      "case_id": "...",
      "reason": "short reason",
      "reject_categories": ["cross_topic_conflict"]
    }}
  ]
}}
""".strip()


def build_conversation_prompt(
    persona_str: str,
    fact: dict[str, Any],
    memory_case: dict[str, Any],
    sampled_conversation_types: list[str],
    preferred_conversation_type: str | None = None,
    target_message_count: int = MIN_CONVERSATION_MESSAGES,
    prior_case_sessions: list[dict[str, Any]] | None = None,
    candidate_conversation_flows: dict[str, str] | None = None,
    persona_signal_level: str | None = None,
    persona_signal_guidance: str | None = None,
) -> str:
    target_message_count = _normalize_target_message_count(target_message_count)
    type_descriptions = _conversation_type_prompt_view(sampled_conversation_types, candidate_conversation_flows or {})
    preferred_section = ""
    if preferred_conversation_type:
        fallback_types = [item for item in sampled_conversation_types if item != preferred_conversation_type]
        preferred_section = f"""
Preferred conversation type:
{preferred_conversation_type}

Fallback conversation types:
{_json(fallback_types)}

Try the preferred conversation type first. If it would make the session feel forced, unnatural, too direct, or inconsistent with the persona/case/fact, choose the most natural fallback type from the candidate list instead.
"""
    prior_sessions = [_same_case_session_view(session) for session in prior_case_sessions or []]
    prior_section = ""
    if prior_sessions:
        prior_section = f"""
Existing sessions for this same case:
{_json(prior_sessions)}

Use existing sessions only as same-case relation and style context. Do not continue a prior chat, refer to a prior draft, or ask the assistant to recall it.
The new session and existing sessions must still match the case design, current relation type meaning, subtype meaning, and subtype notes together.
Vary only expression style, such as conversation type, opening shape, request task, assistant output format, and rhythm. Do not change relation-critical context.
Avoid reusing the exact concrete setup, wording, imagery, or assistant format from existing sessions unless it is needed for relation consistency.
"""
    persona_signal_section = ""
    if persona_signal_level:
        if persona_signal_level not in PERSONA_SIGNAL_LEVEL_GUIDANCE:
            raise ValueError(f"unknown persona_signal_level: {persona_signal_level}")
        guidance = persona_signal_guidance or PERSONA_SIGNAL_LEVEL_GUIDANCE[persona_signal_level]
        persona_signal_section = f"""
Persona signal level: {persona_signal_level}.

{guidance}
"""
    relation_section = _relation_conversation_guidance(memory_case)
    relation_guidance = build_current_case_relation_guidance(memory_case)
    return f"""
Generate one realistic user-assistant session for this user.

Persona:
{persona_str}

Fact to embed naturally:
{_json(fact)}

Case context:
{_json(memory_case)}

Candidate conversation types:
{_json(type_descriptions)}

Current case relation guidance:
{_json(relation_guidance)}

{preferred_section}
{prior_section}
{persona_signal_section}
{relation_section}
Choose exactly one candidate type and write the session in that style.
Treat the selected conversation type as the primary agent task, not merely as an output format.
If a candidate type includes a suggested_flow, use it as a loose rhythm for the selected type, not a rigid template.
Do not force the suggested_flow if it conflicts with the persona, case, fact, relation guidance, or natural user-agent interaction.
Unless the selected type is artifact_production or artifact_review_or_localization, the main task must not be drafting, rewriting, translating, caption writing, or repeated text revision.
The target fact is hidden supervision. Do not state it directly, but make it recoverable from the user's side of the full session.
Before writing, identify the target object/category and any qualifier/context in the target fact.
User turns should provide the target object/category plus at least one concrete relation signal, such as polarity, behavior, boundary, example, past action, routine, complaint, correction, artifact detail, or use context.
Hidden does not mean cryptic: acceptable hidden means the fact is not directly said, but a reader can recover the object/category and the user's polarity, behavior, or context from the full session.
Too hidden means the user side only reveals the broad topic or lacks the object/category plus a concrete relation signal.
Too explicit means copying, near-paraphrasing, or writing the target fact as a direct profile sentence.
Only embed the target fact for this session. Other facts in the same case are relation context only: use them to avoid breaking the case design, but do not express, imply, paraphrase, combine, or leak them in this session.
Do not copy the fact sentence or closely paraphrase it.
Do not state the target fact as a direct self-profile sentence such as "I like...", "I prefer...", "I enjoy...", "I am...", or "My preference is...".
User messages should be concise and conversational. In most user turns, express one immediate intent, constraint, correction, or small piece of context.
Do not front-load all background, constraints, materials, preferences, and desired output into the opening user message.
Do not reveal the target fact in the opening user message. Let it emerge during the agent task through later user constraints, added context, material details, choice criteria, corrections, or boundaries.
Avoid one long user paragraph that fully exposes the target fact; disclose relevant context across later turns as the assistant responds.
Apply persona_signal_level only to the user's broader voice, background texture, and task framing. The target fact must still follow the fact-embedding rules above and remain faithful to the current case relation.
Avoid overusing repeated persona metaphors, motifs, or imagery unless they are genuinely relevant to the selected task.
Assistant messages should respond naturally and should not restate the hidden fact as a conclusion about the user.
Vary the opening and interaction shape. Avoid repeatedly starting with "Can you help me" or "Could you help me" unless that phrasing is truly natural for the selected type.
Avoid ending most assistant messages with generic follow-up offers such as "If you want...".
Do not state that this is a memory fact, a label, or a task target.
Use OpenAI message format with exactly {target_message_count // 2} turns, where one turn means one user message followed by one assistant message, and exactly {target_message_count} messages total: {target_message_count // 2} user messages and {target_message_count // 2} assistant messages.
Messages must start with the user and alternate user, assistant, user, assistant until the session ends.

Think step by step.

###Output
Return only JSON:
{{
  "selected_conversation_type": "one sampled type",
  "messages": [
{_conversation_message_skeleton(target_message_count)}
  ]
}}
""".strip()


def _conversation_type_prompt_view(
    sampled_conversation_types: list[str],
    candidate_conversation_flows: dict[str, str],
) -> dict[str, dict[str, str]]:
    type_view: dict[str, dict[str, str]] = {}
    for key in sampled_conversation_types:
        item = {"description": CONVERSATION_TYPE_DESCRIPTIONS[key]}
        if key in candidate_conversation_flows:
            item["suggested_flow"] = candidate_conversation_flows[key]
        type_view[key] = item
    return type_view


def _normalize_target_message_count(target_message_count: int) -> int:
    if target_message_count < MIN_CONVERSATION_MESSAGES:
        raise ValueError("target_message_count must be at least the minimum conversation message count")
    if target_message_count % 2:
        raise ValueError("target_message_count must be even so user and assistant turns alternate")
    return target_message_count


def _conversation_message_skeleton(target_message_count: int) -> str:
    lines = []
    for index in range(target_message_count):
        role = "user" if index % 2 == 0 else "assistant"
        comma = "," if index < target_message_count - 1 else ""
        lines.append(f'    {{"role": "{role}", "content": "..."}}{comma}')
    return "\n".join(lines)


def build_conversation_filter_prompt(batch_context: dict[str, Any]) -> str:
    batch_subtypes = _batch_relation_subtypes(batch_context)
    return f"""
Review a batch of generated user-assistant conversation cases.

Relation type definitions:
{_json(CASE_RELATION_TYPE_DEFINITIONS)}

Relation subtype definitions:
{_json(_case_subtype_definitions_view(batch_subtypes))}

Subtype notes:
{_json(_case_subtype_notes_view(batch_subtypes))}

Boundary rules:
{_relation_guidance()}

Batch:
{_json(batch_context)}

Find only cases that should be rejected.
For each conversation, first inspect user turns only and internally extract the target object/category, any qualifier/context, and the recoverable signals that make the target fact inferable. Use this extraction only for judgment; do not add these fields to the output JSON.
Hiddenness calibration:
- Reject as too_hidden if user turns reveal only a broad topic/category, or if they lack the target object/category plus at least one concrete relation signal such as polarity, behavior, boundary, example, past action, routine, complaint, correction, artifact detail, or use context.
- Treat hiddenness as acceptable hidden when the conversation does not state the fact directly, but the target object/category and the user's polarity, behavior, boundary, or practical use are recoverable from user turns.
- Reject as too_explicit if the conversation directly repeats the fact, closely paraphrases it, or turns it into a profile sentence such as "I like...", "I prefer...", "I enjoy...", "I am...", or "My preference is...".
Judge each case as a whole. Reject a case if any conversation in that case is unnatural, has fewer than {MIN_CONVERSATION_TURNS} complete user/assistant turns ({MIN_CONVERSATION_MESSAGES} messages), too mechanical, not a realistic multi-turn session, does not start with the user, does not alternate user/assistant, does not naturally and implicitly express its target fact, expresses the target fact only as a vague topic-adjacent aside that a reader could not recover from the full session, copies or closely paraphrases the fact, states the fact as a direct self-profile sentence, mismatches the selected conversation type, conflicts with the persona/case/fact, or exposes benchmark, memory, label, or task framing.
For non-artifact conversation types, reject conversations whose main task is actually drafting, rewriting, translating, caption writing, or repeated text revision.
Each case item includes case_relation_guidance, case_conversation_guidance, and all generated conversations for that case. Use the current relation type definition, subtype definition, subtype notes, and case_conversation_guidance to judge whether the conversations still satisfy the case design together.
Reject a case if its conversations no longer satisfy the current relation type/subtype design, even if each single conversation looks locally fluent.
Artifact order, conversation list order, and timestamps are provenance metadata, not relation-critical time conditions by themselves. Reject a non-Temporal case if it becomes valid only by treating a later conversation as the current truth or an earlier conversation as superseded; reject a Temporal case if its time split is only implied by artifact/session order rather than content.
For every rejected case, list the specific problem conversations and their problems when identifiable. If the case-level relation is broken across conversations, explain that at the case level even if no single conversation is solely responsible.
Do not reject a case merely because one or more conversations are concise if they are natural and the facts are implied through realistic, recoverable context.
If there are no problem cases in this batch, return an empty problem_cases list.

Think step by step.

###Output
Return only JSON:
{{
  "problem_cases": [
    {{
      "case_id": "...",
      "reason": "short case-level reason",
      "reject_categories": ["too_hidden"],
      "problem_conversations": [
        {{
          "conversation_id": "...",
          "reason": "short conversation-level reason",
          "reject_categories": ["too_hidden"]
        }}
      ]
    }}
  ]
}}
""".strip()


def build_case_question_prompt(
    persona_str: str,
    memory_case: dict[str, Any],
    sessions: list[dict[str, Any]],
    *,
    question_count: int,
) -> str:
    qa_case = _qa_case_view(memory_case)
    qa_relation = _qa_case_relation_view(memory_case)
    qa_sessions = [_qa_session_view(session) for session in sessions]
    relation_section = _relation_question_guidance(memory_case)
    return f"""
Generate candidate user questions for this user memory case.

Persona:
{persona_str}

Case:
{_json(qa_case)}

Case relation:
{_json(qa_relation)}

Accepted sessions for this case:
{_json(qa_sessions)}

{relation_section}
Requirements:
- Generate exactly {question_count} natural first-person user questions.
- Each question should sound like this user naturally asking an agent for recommendations, ideas, advice, or explanations.
- Each question must be tightly anchored to this case and its facts; avoid broad topic-level questions that could be answered well without this case.
- A strong personalized answer should need the hidden case facts and session context.
- Each question must have a clear, case-verifiable answer direction: the correct answer should be distinguishable from plausible wrong answers using the current case facts and accepted sessions, without relying on hidden labels or external knowledge.
- Each question must be self-contained and understandable without pointing to the accepted sessions.
- Use sessions only as background signal; do not ask about a prior chat, earlier draft, previous message, or text above.
- The question itself must not mention, paraphrase, combine, or strongly allude to the hidden case facts or ground-truth preference.
- A question may include only neutral conditions needed to select the right fact, such as home vs. work, before vs. now, role, location, task, scope, version, or object.
- Leave the user's actual preference, habit, conflict, state, or personalized answer direction unstated for the answer to infer.
- The questions must match the current case relation definition.
- If time or context changes the answer, include only the neutral condition needed to disambiguate, not the hidden preference itself.
- Do not ask the agent to recall preferences, memory, labels, hidden facts, benchmark framing, or prior conversation.
- Make question intents diverse; avoid repeated intent.

Think step by step.

###Output
Return only JSON:
{{
  "questions": [
    {{"question": "..."}}
  ]
}}
""".strip()


def build_case_answer_prompt(
    persona_str: str,
    memory_case: dict[str, Any],
    sessions: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    correct_count: int,
    incorrect_count: int,
) -> str:
    qa_case = _qa_case_view(memory_case)
    qa_relation = _qa_case_relation_view(memory_case)
    qa_sessions = [_qa_session_view(session) for session in sessions]
    relation_section = _relation_answer_guidance(memory_case)
    return f"""
Generate answer candidates for these user questions.

Persona:
{persona_str}

Case:
{_json(qa_case)}

Case relation:
{_json(qa_relation)}

Accepted sessions for this case:
{_json(qa_sessions)}

Questions:
{_json(questions)}

{relation_section}
Requirements:
- For each question_id, generate exactly {correct_count} correct answers and exactly {incorrect_count} incorrect answers.
- Correct answers must give the personalized response enabled by the hidden case facts and persona.
- Incorrect answers must be plausible but wrong for this user's case.
- Answers must match the current case relation definition.
- Respect any time or context condition stated in the question.
- If the supplied memory items conflict, answers must follow the current case relation definition.
- Do not make answers depend on unseen prior text; answer the standalone question using the case facts and persona.
- Keep answers natural, concise, and similar in style and length.
- Do not mention answer labels, relation labels, memory, benchmark, hidden facts, hidden preferences, or task framing.

Think step by step.

###Output
Return only JSON:
{{
  "answers": [
    {{
      "question_id": "...",
      "correct_answers": ["...", "..."],
      "incorrect_answers": ["...", "..."]
    }}
  ]
}}
""".strip()


def build_case_task_prompt(
    persona_str: str,
    memory_case: dict[str, Any],
    sessions: list[dict[str, Any]],
    *,
    question_count: int,
    same_topic_sibling_cases: list[dict[str, Any]] | None = None,
) -> str:
    qa_case = _qa_case_view(memory_case)
    qa_relation = _qa_case_relation_view(memory_case)
    qa_sessions = [_qa_session_view(session) for session in sessions]
    relation_section = _relation_task_guidance(memory_case)
    sibling_cases = same_topic_sibling_cases or []
    return f"""
Generate candidate user task prompts for this user memory case.

Persona:
{persona_str}

Case:
{_json(qa_case)}

Case relation:
{_json(qa_relation)}

Accepted sessions for this case:
{_json(qa_sessions)}

Same-topic sibling cases for interference avoidance:
{_json(sibling_cases)}

{relation_section}
Requirements:
- Generate exactly {question_count} natural first-person user task requests.
- Write each request from the user's point of view as a natural user-to-agent help request, not as a direct question about the user's own preference, habit, usual choice, or best fit.
- Each request must be self-contained, realistic, have one primary decision/output target, and match the current case relation definition.
- A correct response must require the current case facts and relation; avoid broad or generic tasks that can be completed well without this case.
- Prefer task shapes whose correctness can be checked through concrete selection, exclusion, ranking, checklist items, clarification, or fact-based criteria.
- Use accepted sessions only as background signal; do not reference a prior chat, earlier draft, previous message, text above, or details that appear only in sessions.
- Do not mention, paraphrase, combine, or strongly allude to hidden facts, ground-truth preferences, memory, labels, benchmark framing, or answer direction.
- Include only neutral time/context/role/location/task/scope/version/object conditions needed by the relation; leave the selected preference, habit, conflict, or state unstated.
- Avoid tasks judged mainly by tone polish, decorative wording, generic criteria, unsupported persona comparisons/salience, or contrived artifacts unless the case directly motivates them.
- Make task intents diverse; avoid repeated intent.

Task form menu:
- For each generated item, choose exactly one task form that best fits the current case facts and relation.
- Output the chosen form in the task_form field. Do not write the task_form label inside the natural-language question text.
- Each query must include a visible user goal, agent-available material, and a concrete output contract.

1. structured_form
- Use when a fixed form, card, table, or slot template can make the answer easy to judge.
- Present the material as an agent-available template, such as "Available working form:" or "Available field card:".
- The user asks the assistant to fill the fields for a visible purpose. The query must include fixed field names and should prevent extra fields unless extra fields are part of the task.
- Use structured_form for open-ended generated field values; do not add option banks just to constrain the answer.
- Keep each field specific enough that a correct answer needs memory-grounded content, not arbitrary generic prose.
- Each filled field should require a concrete memory-grounded value, decision, exclusion, action, condition, or clarification; vague positive wording must not be enough to count as correct.
- Avoid fields where many generic answers would be acceptable, such as vague "reason", "note", or "preference" fields, unless the query gives a very specific output contract.
- Example: "I need a compact signup profile for tomorrow's workshop registration. Use the available working form and fill each field with a short phrase. Do not add fields.\n\nAvailable working form:\n- Session format:\n- Interaction level:\n- Pace:"

2. resource_arrangement
- Use when the assistant should operate on existing resources or candidate materials that are visible in the query.
- Present the material as an agent-available brief, such as "Available resource brief:".
- Treat the brief as already available to the agent; do not frame it as something the user found, collected, pasted, or wants the agent to search for.
- The assistant must operate on the existing resources, such as selected names, ranked order, excluded items, slot assignment, schedule, cart, booking list, signup list, or next-action label.
- Do not ask the assistant to invent, write, expand, search for, or fetch resources. The deliverable must not be a rewritten resource brief.
- Include self-contained resource descriptions. Use realistic named resources, not bare A/B/C labels, and make all resources plausible unless the relation requires rejecting or clarifying.
- Prefer counter-stereotypical or stereotype-neutral resources; do not make the answer obvious from demographics, job, culture, age, location, topic label, or broad persona stereotypes.
- Do not tailor resources around hidden case facts, copy distinctive case/session keywords into the correct resource, or make names/descriptions encode the hidden preference, conflict, or answer direction.
- Resource attributes may vary by cost, duration, location, format, intensity, materials, audience, constraints, or style, but those attributes should not directly reveal the answer without the case facts.
- Include 5-6 candidate resources when possible, not only 3-4.
- Add matched distractors, not filler distractors: at least two distractors should share the same broad visible category, culture cue, topic cue, or practical advantage as the correct option.
- If the correct option is culturally/stereotypically aligned, include other culturally/stereotypically aligned options that are plausible but fail on hidden case constraints.
- If the correct option is selected for logistics, include other logistically strong options that fail on hidden case constraints.
- The correct answer must not be the only option with the obvious surface cue.
- Keep the requested selection narrow and exact: the memory should determine the full selected/excluded set or order, not merely make many subsets acceptable.
- Avoid high-probability selection tasks where selecting most resources would make random or partial-overlap answers likely correct, such as any 4 of 6 resources.
- A model seeing only the query and resource brief should find multiple plausible candidates and need the memory facts to choose.
- Example: "I need to decide what to register for next month. Use the available resource brief to build a two-item signup list. Return exactly two activity names and one short reason for each.\n\nAvailable resource brief:\n- Paper Studio: 90-minute collage session, quiet table setup, materials included, $25.\n- Supper Forum: two-hour dinner discussion, assigned group seating, fixed menu, $35.\n- Tool Basics: beginner repair demo, standing format, shared tools, $20.\n- Window Herbs: 75-minute tabletop planting session, small group, supplies included, $30."

Hint and interference control:
- The query must be underdetermined without the current case facts: visible goals, fields, resources, and neutral context may make the task natural, but they must not reveal the correct answer by themselves.
- Neutral conditions are allowed only when needed by the relation; do not include option-specific hints such as praising one option, naming the hidden criterion, saying what the user usually likes, or describing the desired answer direction.
- For structured_form, keep field names and slot labels neutral; they must not name the hidden fact, hidden criterion, or answer direction.
- For resource_arrangement, make all resources plausible under the visible user goal; avoid a unique surface winner that can be selected from resource attributes alone.
- Resource order in a brief should not make the correct resource obvious.
- Treat same-topic sibling cases only as interference-avoidance context, not as answer evidence.
- Avoid prompts whose candidates, surface context, or neutral condition could plausibly be driven by another same-topic sibling case, and do not blend sibling facts into the current answer direction.
- Before returning each task, run a visible-only check: if a model without the case facts could confidently pick the same answer from the query, resources, form fields, demographic cues, or broad stereotypes, rewrite the task with stronger matched distractors or reject that task shape.

Task quality calibration examples:
- Examples show task shape only; do not copy their domains, names, facts, wording, or answer content.
- Weak material: reject a query that asks the agent to get resources later, search for resources, or freely create resources.
- Generic or external-knowledge task: bad "Give reusable criteria" or "Recommend famous books"; better "Use the available resource brief to produce the requested structured arrangement."
- Contrived/session-only task: bad "write a quirky reminder" or "use the earlier chat detail"; better "state the neutral case condition in a natural task and ask for a practical output."

Think step by step.

###Output
Return only JSON:
{{
  "questions": [
    {{"task_form": "structured_form or resource_arrangement", "question": "..."}}
  ]
}}
""".strip()


def build_case_task_answer_prompt(
    persona_str: str,
    memory_case: dict[str, Any],
    sessions: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    correct_count: int,
    incorrect_count: int,
) -> str:
    qa_case = _qa_case_view(memory_case)
    qa_relation = _qa_case_relation_view(memory_case)
    qa_sessions = [_qa_session_view(session) for session in sessions]
    relation_section = _relation_task_answer_guidance(memory_case)
    return f"""
Generate short fact-grounded answer candidates for these user tasks.

Persona:
{persona_str}

Case:
{_json(qa_case)}

Case relation:
{_json(qa_relation)}

Accepted sessions for this case:
{_json(qa_sessions)}

Task prompts:
{_json(questions)}

{relation_section}
Requirements:
- For each question_id, generate exactly {correct_count} correct answers and exactly {incorrect_count} incorrect answers.
- Treat each item as a concise, standalone key answer, not as the full task output.
- Answers must match the current case relation, respect stated time/context conditions, and be grounded in the case facts and persona rather than unseen prior text.
- Correct answers should capture the essential facts, fact combinations, conditions, directions, exclusions, or candidate decisions a good response must reflect.
- Use task_form when present: structured_form keys identify the required field values or slot content; resource_arrangement keys identify selected/excluded resources, ordering, final set, slot assignment, needed clarification, or abstention/no-selection when the relation requires it.
- Incorrect answers should be plausible but wrong: wrong facts, missing facts, wrong condition/direction, wrong form values, wrong candidate/order/final set, surface-clue-only choice, stereotype-driven choice, overconfident contradiction resolution, hedged non-selection, or unsupported criterion.
- If external named candidates appear, score the in-prompt, case-grounded candidate decision rather than trivia or outside-world facts.
- Keep answers natural, specific, similar in length, and limited to the key answer content.
- Do not write meta-evaluative statements, full deliverables/drafts/plans, unsupported persona comparisons, or mentions of labels, memory, benchmark framing, hidden facts, hidden preferences, or task framing.

Answer-key shape examples:
- Selection task: Correct key answer: "shortlist the named candidates that match the case-grounded criterion and exclude the candidates optimizing the wrong factor." Incorrect key answer: "pick candidates for speed or broad popularity when those are not supported by the case, or refuse to choose after candidates are supplied."
- Contradictory task: Correct key answer: "state that the relevant facts conflict and clarification is needed before choosing." Incorrect key answer: "pick one side as fixed truth or invent a context that resolves the conflict."

Think step by step.

###Output
Return only JSON:
{{
  "answers": [
    {{
      "question_id": "...",
      "correct_answers": ["...", "..."],
      "incorrect_answers": ["...", "..."]
    }}
  ]
}}
""".strip()


def build_case_qa_filter_prompt(batch_context: dict[str, Any]) -> str:
    return f"""
Review generated user questions and answer candidates.

Batch:
{_json(batch_context)}

Each case QA item includes case_relation_guidance. Use the current relation type meaning, subtype meaning, and subtype notes when judging relation quality.
Reject whole questions if they are unnatural, too broad to require this case, not self-contained, depend on a prior session/draft/message/text instead of a standalone user need, leak hidden facts, paraphrase or strongly allude to hidden facts, mention memory/benchmark/labels, do not match the case relation subtype, cannot be answered from the case and sessions, or repeat the same intent as another question for the same case.
Relation-specific QA filter rules:
- For complementary cases, reject questions or answers that invent time/context conditions or conflicts; for K>1, correct answers must combine the required memory items.
- For nuanced cases, reject questions or answers that omit the neutral time/context needed to disambiguate, expose the condition-specific answer in the question, apply the wrong condition, flatten condition-specific memory items into one global answer, ask for or answer multiple condition-specific facts in one question, compare both versions, request a then-and-now/timeline/all-contexts output, or make the item answerable from one memory fact without real time/context disambiguation.
- For nuanced cases, do not reject a QA set merely because different questions target different case facts; reject only when a single question or answer requires multiple condition-specific facts.
- Artifact order, session order, and timestamps are not time conditions by themselves; reject questions or answers that use latest/earlier session order as the only way to resolve a non-Temporal case.
- For contradictory cases, reject questions that reveal the conflict directly or are answerable by choosing one side; reject questions or answers that turn the conflict into a resolved time/context split or choose the latest timestamp/session; correct answers must acknowledge contradiction, uncertainty, or need for clarification; answers that use any single memory item as the only basis are wrong.
Remove individual correct answers if they are unsupported, wrong, too generic, ignore required temporal/context conditions, choose only one memory item for a contradictory case, choose the later timestamp/session for a contradictory case, resolve a contradiction with unsupported context, or fail to acknowledge contradiction when needed.
Remove individual incorrect answers if they are actually correct, correctly acknowledge the contradiction for a contradictory case, contain the target fact, are too similar to a correct answer, or are unnatural.
If a question has no valid correct answers or no valid incorrect answers, mark that question as rejected.

Think step by step.

###Output
Return only JSON:
{{
  "rejected_questions": [
    {{
      "question_id": "...",
      "reason": "short reason",
      "reject_categories": ["duplicate_intent"],
      "duplicate_of_question_id": "..."
    }}
  ],
  "removed_answers": [
    {{
      "answer_id": "...",
      "answer_type": "correct or incorrect",
      "reason": "short reason",
      "reject_categories": ["unsupported"]
    }}
  ]
}}
""".strip()


def build_task_qa_filter_prompt(batch_context: dict[str, Any]) -> str:
    return f"""
Review generated user task prompts and short answer candidates.

Batch:
{_json(batch_context)}

Each case QA item includes case_relation_guidance. Use the current relation type meaning, subtype meaning, and subtype notes when judging relation quality.
Reject a task prompt for any of these issue groups:

Core task quality:
- Unnatural user-to-agent request; no realistic completion need; no clear real-world purpose; multiple independent deliverables; too broad/open-ended; repeated intent.
- Not self-contained, depends on prior sessions/drafts/messages/text, or uses session background as the main answer-driving fact.
- Directly asks for the user's own preference, habit, usual choice, or best fit; mentions memory, benchmark framing, or labels.
- Leaks, paraphrases, or strongly alludes to hidden facts; does not match the relation subtype; cannot be judged cleanly from case facts and relation.
- Mainly rewards copywriting, shopping phrasing, formatting, polish, generic criteria, unsupported persona comparisons, or contrived artifacts instead of fact use.

Task form, material, and hint failures:
- qa.questions[].task_form is missing/invalid, does not match the query, or is not exactly structured_form or resource_arrangement.
- Does not fit structured_form or resource_arrangement, or lacks a visible user goal, agent-available material, or concrete output contract.
- structured_form tasks lack fixed fields/slots, allow uncontrolled extra fields, or cannot be judged field by field.
- structured_form field names or slot labels directly reveal the hidden fact, hidden criterion, or answer direction.
- resource_arrangement tasks omit the available resource brief, ask only for generic standards, ask the assistant to invent/write/expand/search/fetch resources, or make the deliverable a rewritten resource brief.
- resource_arrangement tasks fail to operate on existing resources through selected names, ranked order, excluded items, slot assignment, schedule, cart, booking list, signup list, next-action label, or another concrete arrangement.
- Resource or form material relies on detailed external knowledge instead of self-contained descriptions.
- Resources are too case-tailored, over-tuned, copied/paraphrased from case facts or accepted sessions, too thin to judge, or contain option-specific hints such as praise, the hidden criterion, usual preference, or desired answer direction.
- Candidate sets have weak_distractors: distractors are generic fillers and the correct option is the only detailed, culturally marked, logistically superior, or topic-matching option.
- The task has a broad_answer_space: it accepts vague free-text reasons, generic criteria, or many equally valid completions without requiring a specific memory-grounded decision.
- The task has high_guess_probability: it asks for a large subset of available candidates, accepts partial-overlap answers, or makes many candidate combinations valid instead of one memory-determined selected/excluded set.
- The task frames material as something the user found, collected, pasted, or wants the agent to search for, instead of as available working material in the query.
- The correct answer is recoverable from the query alone: visible goal, form fields, resource names/descriptions, resource attributes, or neutral context create an obvious answer without using the current case facts.
- The answer is obvious from stereotypes, demographics, broad topic priors, job, culture, age, location, or persona background.
- Another same-topic sibling case could plausibly drive the answer; sibling cases are interference checks only, not answer evidence.

Use specific reject_categories when relevant: missing_or_invalid_task_form, missing_external_candidate_framing, case_keyword_leakage, option_specific_hint, answerable_without_memory, stereotype_answerability, over_tuned_candidates, weak_distractors, broad_answer_space, high_guess_probability, sibling_topic_interference, external_knowledge_dependency.

Relation-specific QA filter rules:
- For complementary cases, reject task prompts or answers that invent time/context conditions or conflicts; for K>1, correct answers must require the required memory items to be combined.
- For nuanced cases, reject task prompts or answers that omit the neutral time/context needed to disambiguate, expose the condition-specific answer in the task prompt, apply the wrong condition, flatten condition-specific memory items into one global answer, or make the task solvable from one memory fact without real time/context disambiguation. For Context cases, the stated context must change the correct completion; for Temporal cases, the time condition must come from the case facts, not from accepted-session details.
- Artifact order, session order, and timestamps are not time conditions by themselves; reject task prompts or answers that use latest/earlier session order as the only way to resolve a non-Temporal case.
- For contradictory cases, reject task prompts that reveal the conflict directly or are judgeable by choosing one side; reject task prompts or answers that turn the conflict into a resolved time/context split or choose the latest timestamp/session; correct answers must require acknowledging contradiction, uncertainty, or need for clarification; answers that reward any single memory item as the only basis are wrong.

Rejection calibration examples:
- Reject missing_or_invalid_task_form: task_form is absent, invalid, or conflicts with the visible query shape.
- Reject missing_external_candidate_framing: resource_arrangement task has no available resource brief, promises resources later, asks the assistant to search, or only asks for criteria.
- Reject over_tuned_candidates: a resource mirrors distinctive case/session wording while other resources are fillers.
- Reject case_keyword_leakage: a form field, resource name, or resource description copies/paraphrases the hidden fact.
- Reject option_specific_hint: the request praises a candidate, names the hidden criterion, or states the desired direction instead of only a neutral condition.
- Reject answerable_without_memory: the visible query, form, resource brief, candidate attributes, or neutral condition already makes the correct answer the unique or obviously best choice.
- Reject stereotype_answerability: the correct answer is the only candidate matching a demographic, cultural, age, gender, job, nationality, religion, or topic stereotype.
- Reject weak_distractors: distractors are generic fillers and the correct option is the only detailed, culturally marked, logistically superior, or topic-matching option.
- Reject broad_answer_space: the task accepts vague free-text reasons, generic criteria, vague form values, or many equally valid completions without requiring a specific memory-grounded decision.
- Reject high_guess_probability: the task asks for a large subset of candidates, accepts partial-overlap answers, or would mark many candidate combinations as correct, such as any 4 of 6 resources.
- Reject sibling_topic_interference: another same-topic sibling case could determine the choice or answer direction.
- Reject external_knowledge_dependency: working form or resource brief is not described enough to judge from the prompt.
- Reject weak Context: a task names a context but the same completion would be correct under every case context.

Remove a correct answer if it:
- is unsupported, wrong, or too generic
- does not point to the facts, fact combinations, or condition distinctions the response must recover or use
- is long, explanatory, or written more like a rubric than a short key answer
- introduces unsupported persona-level comparisons or salience judgments not stated in the case
- ignores required temporal/context conditions
- chooses only one memory item for a contradictory case
- chooses the later timestamp/session for a contradictory case
- resolves a contradiction with unsupported context
- fails to acknowledge contradiction when needed
- describes a concrete task output instead of a short answer target

Remove an incorrect answer if it:
- is actually correct
- correctly acknowledges the contradiction for a contradictory case
- contains the target fact
- rewards the right facts or the right condition even if framed as a surface deliverable judgment
- relies on unsupported persona-level comparisons or salience judgments not stated in the case
- is long, explanatory, or written more like a rubric than a short key answer
- is too similar to a correct answer
- is unnatural
- still functions as a valid correct answer target

If a task prompt has no valid correct answers or no valid incorrect answers, reject that task prompt.

Think step by step.

###Output
Return only JSON:
{{
  "rejected_questions": [
    {{
      "question_id": "...",
      "reason": "short reason",
      "reject_categories": ["duplicate_intent"],
      "duplicate_of_question_id": "..."
    }}
  ],
  "removed_answers": [
    {{
      "answer_id": "...",
      "answer_type": "correct or incorrect",
      "reason": "short reason",
      "reject_categories": ["unsupported"]
    }}
  ]
}}
""".strip()
