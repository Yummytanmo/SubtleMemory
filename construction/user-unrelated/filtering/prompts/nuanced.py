from __future__ import annotations

import json
from typing import Any


CONVERSATION_FILTER_SYSTEM_PROMPT = """You are judging one nuanced-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and naturalness requirements.
Think step by step.
"""


QUESTION_FILTER_SYSTEM_PROMPT = """You are judging the question quality of one nuanced-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and question-design requirements.
Think step by step.
"""


ANSWER_FILTER_SYSTEM_PROMPT = """You are judging the answer quality of one nuanced-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and answer-design requirements.
Think step by step.
"""


QA_FILTER_SYSTEM_PROMPT = """You are judging the question-and-answer quality of one nuanced-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge the whole QA set together against the subtype requirements.
Think step by step.
"""


def _pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)


def _background() -> str:
    return (
        "Project background:\n"
        "- This benchmark tests fine-grained fact discrimination, correct retrieval, and reasoning over similar memories.\n"
        "- This sample belongs to nuanced memory, where multiple remembered facts can all be true, but under different temporal or contextual conditions."
    )


def _subtype_definition(sample: dict[str, Any]) -> str:
    subtype = sample["nuanced_subtype"]
    if subtype == "temporal":
        return (
            "Nuanced subtype: temporal.\n"
            "- Multiple memories answer the same underlying question, and all can be true because they come from different dated snapshots or as-of versions.\n"
            "- The answer should change across time for the same question.\n"
            "- The later question should ask for the answer under one specific temporal snapshot; the snapshot may be identified by a direct date, an event, a task, a version, or relative time, but it should not ask when an event happened."
        )
    if subtype == "context":
        return (
            "Nuanced subtype: context.\n"
            "- Multiple related facts can all be true, but under different contexts such as role, location, scope, version, definition, or attribute.\n"
            "- The later question should specify one decisive context and expect the single answer for that context."
        )
    raise ValueError(f"Unknown nuanced subtype: {subtype}")


def _selected_facts(sample: dict[str, Any]) -> list[dict[str, Any]]:
    if sample["nuanced_subtype"] == "temporal":
        return sample["selected_temporal_facts"]
    return sample["selected_context_facts"]


def _scaffold_question(sample: dict[str, Any]) -> str:
    if sample["nuanced_subtype"] == "temporal":
        return sample["temporal_question"]
    return sample["context_question"]


def _question_items(sample: dict[str, Any], *, include_answers: bool) -> list[dict[str, Any]]:
    qa_pairs = sample.get("qa_pairs")
    if isinstance(qa_pairs, list) and qa_pairs:
        items: list[dict[str, Any]] = []
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            item = {
                "qa_id": pair.get("qa_id"),
                "difficulty": pair.get("difficulty"),
                "temporal_anchor_type": pair.get("temporal_anchor_type"),
                "target_memory_id": pair.get("target_memory_id"),
                "context_anchor": pair.get("context_anchor"),
                "question": pair.get("question"),
                "question_design_notes": pair.get("question_design_notes"),
            }
            if include_answers:
                item["correct_answers"] = pair.get("correct_answers")
                item["incorrect_answers"] = pair.get("incorrect_answers")
            items.append(item)
        return items

    item = {"question": sample["question"]}
    if include_answers:
        item["correct_answers"] = sample["correct_answers"]
        item["incorrect_answers"] = sample["incorrect_answers"]
    return [item]


def build_conversation_filter_prompt(sample: dict[str, Any]) -> str:
    return f"""
{_background()}

{_subtype_definition(sample)}

Conversation-stage judging requirements:
- Judge only the sessions and how the nuanced facts are embedded.
- Do not judge the final question or answer candidates here.
- The sessions should feel like natural user-assistant interactions rather than explicit benchmark recaps.
- The facts should be distributed across multiple sessions and remain natural.
- For temporal samples, the sessions should embed dated answer snapshots for the same underlying question. The date should be an as-of/snapshot/version date for the answer, not just the occurrence date of separate events.
- This should not become contradiction: the facts should still all be potentially true under different snapshot dates or context conditions.
- Say "no" for temporal samples if the facts are actually about different entities, different contexts, or separate event dates rather than one question whose answer changes over time.
- Say "no" if the sessions are unnatural, too explicit, or if the sample stops behaving like temporal/contextual nuance.

Broad scaffold question:
{_scaffold_question(sample)}

Selected nuanced facts:
{_pretty_json(_selected_facts(sample))}

Session plans:
{_pretty_json(sample["session_plans"])}

Sessions:
{_pretty_json(sample["sessions"])}

Return JSON in this format:
```json
{{
  "decision": "yes or no",
  "reason": "short concrete reason"
}}
```
Think step by step.
""".strip()


def build_question_filter_prompt(sample: dict[str, Any]) -> str:
    subtype = sample["nuanced_subtype"]
    if subtype == "temporal":
        subtype_requirements = (
            "- For temporal: judge the whole question set, not just the first item.\n"
            "- For temporal: there should be exactly 3 questions with different angles and difficulty levels: one easy, one medium, and one hard.\n"
            "- For temporal: each question must resolve to exactly one dated/as-of/version snapshot for the same underlying question.\n"
            "- For temporal: easy questions may name the exact as-of date directly.\n"
            "- For temporal: medium and hard questions may use event, task, draft, version, session, before/after, or latest-before anchors instead of explicit dates; do not reject solely because the date is implicit if the sessions make the target snapshot recoverable.\n"
            "- For temporal: reject if all questions are simple direct date lookups, because the set does not sufficiently test temporal reasoning.\n"
            "- For temporal: reject if a question merely asks when an event happened, asks for a release/birth/episode date, or asks for a fact's occurrence time without a remembered snapshot/version condition.\n"
            "- For temporal: reject if a question asks for a timeline, all dates, all periods, chronological order, latest/current answer without a temporal anchor, or a complete recap.\n"
            "- For temporal: reject if a medium/hard question's event/version/relative-time anchor is too vague to identify one target snapshot from the sessions.\n"
            "- For temporal: reject if a question is too under-specified to locate the relevant memory after many unrelated sessions have been stored. Non-direct-date questions must include enough topic/entity and fact-property clues, not only phrases like 'the current snapshot', 'that version', 'the note format', or 'the compact notes-app format'.\n"
            "- For temporal: accept implicit-date questions only when they combine a concrete topic clue with a concrete task/version/event/relative-time clue, so the intended session and temporal slice are recoverable.\n"
        )
    else:
        subtype_requirements = (
            "- For context: the question should specify one decisive context such as role, location, scope, version, definition, unit, attribute, jurisdiction, or scenario.\n"
            "- For context: reject if the question is the broad ambiguous question, asks for all contexts, asks for a complete list, or would require enumerating multiple context-conditioned answers.\n"
            "- For context: if there are two QA items, reject if the questions are near-paraphrases, target the same context, or differ only by wording while testing the same fact.\n"
            "- For context: accept only if each question can be answered by exactly one selected context fact."
        )
    return f"""
{_background()}

{_subtype_definition(sample)}

Question-stage judging requirements:
- Judge only the final benchmark question, with the sessions visible for context.
- Do not judge the answer candidates here.
- The question should sound natural and subtype-appropriate.
{subtype_requirements}
- Say "no" if the question mismatches the subtype or becomes too explicit in the wrong way.

Broad scaffold question:
{_scaffold_question(sample)}

Selected nuanced facts:
{_pretty_json(_selected_facts(sample))}

Sessions:
{_pretty_json(sample["sessions"])}

Question items to judge:
{_pretty_json(_question_items(sample, include_answers=False))}

Return JSON in this format:
```json
{{
  "decision": "yes or no",
  "reason": "short concrete reason"
}}
```
Think step by step.
""".strip()


def build_answer_filter_prompt(sample: dict[str, Any]) -> str:
    subtype = sample["nuanced_subtype"]
    if subtype == "temporal":
        subtype_requirements = (
            "- For temporal: judge every QA item in the question set.\n"
            "- For temporal: for each QA item, all correct answers should match only that item's target temporal snapshot.\n"
            "- For temporal: if a question uses an event/version/relative anchor instead of an explicit date, use the sessions and target_memory_id metadata to judge whether the correct answers match the intended snapshot.\n"
            "- For temporal: do not require correct answers to restate the date if they clearly answer the asked temporal slice."
        )
    else:
        subtype_requirements = (
            "- For context: all correct answers should match only the context explicitly specified by the question.\n"
            "- For context: reject if a correct answer gives all context-conditioned answers, asks for clarification, or answers a different context.\n"
            "- For context: if there are two QA items, judge each QA item independently against its target context."
        )
    return f"""
{_background()}

{_subtype_definition(sample)}

Answer-stage judging requirements:
- Judge only the answer candidates.
- All three correct answers must be genuinely correct for the subtype and question.
- Incorrect answers are auxiliary distractor metadata, not the main benchmark target.
- Do not reject solely because incorrect answers are easy, weak, not very plausible, or not ideally matched to the likely confusion pattern.
- Still reject if an incorrect answer is actually correct, equivalent to a correct answer, or makes the intended answer ambiguous.
{subtype_requirements}
- Say "no" only if a correct answer is incomplete or wrong, if an incorrect answer is actually valid/equivalent, if the answer set is ambiguous, or if the correct answers mismatch the subtype requirements.

Selected nuanced facts:
{_pretty_json(_selected_facts(sample))}

QA items to judge:
{_pretty_json(_question_items(sample, include_answers=True))}

Return JSON in this format:
```json
{{
  "decision": "yes or no",
  "reason": "short concrete reason"
}}
```
Think step by step.
""".strip()


def build_qa_filter_prompt(sample: dict[str, Any]) -> str:
    subtype = sample["nuanced_subtype"]
    if subtype == "temporal":
        subtype_requirements = (
            "- The sample should contain exactly 3 QA items, not more.\n"
            "- The 3 QA items should include exactly one easy, one medium, and one hard question.\n"
            "- Each question must resolve to exactly one temporal snapshot for the same underlying question.\n"
            "- At least two questions should require more than direct date-string retrieval, using event, task, draft, version, session, before/after, or latest-before anchors.\n"
            "- Reject if all questions are simple \"as of YYYY-MM-DD\" lookups.\n"
            "- Reject if any question asks for a timeline, all periods, chronological order, current/latest answer without an anchor, or when an event happened rather than what answer was valid under a remembered temporal snapshot.\n"
            "- Reject if a question is too under-specified to locate the relevant memory after many unrelated sessions have been stored.\n"
            "- For non-direct-date questions, require enough context in the question itself: a concrete topic/entity or target property plus a concrete task/version/event/relative-time anchor.\n"
            "- Reject vague local references such as \"the current snapshot\", \"that version\", \"the note format\", \"the compact notes-app format\", or \"the thing we marked current\" when the question does not also name the topic and target property.\n"
            "- For each QA item, all correct answers must match only that item's target temporal snapshot.\n"
        )
        selected_label = "Selected temporal facts"
    elif subtype == "context":
        subtype_requirements = (
            "- The sample should contain one or two QA items, not more.\n"
            "- Each question must specify one decisive context, such as role, location, jurisdiction, scope, version, definition, unit, attribute, task, object range, or scenario.\n"
            "- Each question must resolve to exactly one context-conditioned fact for the same broad topic.\n"
            "- Reject if any question is still the broad ambiguous question, asks for all contexts, asks for a full list, or requires clarification/enumeration rather than a single context answer.\n"
            "- If there are two QA items, reject if the questions are too similar, target the same memory_id/context, or differ only by superficial wording.\n"
            "- For each QA item, all correct answers must match only that item's target context-specific answer.\n"
            "- Reject if a correct answer gives all context-conditioned answers, asks for clarification, or answers a different context.\n"
        )
        selected_label = "Selected context facts"
    else:
        raise ValueError(f"Unsupported nuanced subtype for combined QA filtering: {subtype}")
    return f"""
{_background()}

{_subtype_definition(sample)}

Combined QA-stage judging requirements:
- Judge the final benchmark question set and answer candidates together.
- Do not re-judge conversation naturalness unless the QA cannot be resolved from the sessions.
- Questions within the same sample must not be near-duplicates. Reject if multiple questions would test essentially the same retrieval with only minor wording changes.
{subtype_requirements}
- Incorrect answers are auxiliary distractor metadata; do not reject just because they are weak or easy.
- Still reject if an incorrect answer is actually equivalent to a correct answer or makes the intended answer ambiguous.
- Return "yes" only if the question set and all correct answers are suitable for nuanced {subtype} memory testing.

Broad scaffold question:
{_scaffold_question(sample)}

{selected_label}:
{_pretty_json(_selected_facts(sample))}

Sessions:
{_pretty_json(sample["sessions"])}

QA items to judge:
{_pretty_json(_question_items(sample, include_answers=True))}

Return JSON in this format:
```json
{{
  "decision": "yes or no",
  "reason": "short concrete reason"
}}
```
Think step by step.
""".strip()
