from __future__ import annotations

import json
from typing import Any


FACT_SELECTION_SYSTEM_PROMPT = """You convert dated answer snapshots into a cleaner benchmark-ready temporal fact bundle.
Return valid JSON only and follow the requested schema exactly.
Keep snapshot dates explicit and preserve the same underlying question across time.
Think step by step.
"""


SESSION_PLAN_SYSTEM_PROMPT = """You plan distinct user-assistant sessions for a memory benchmark.
Return valid JSON only and follow the requested schema exactly.
Optimize for diversity, realism, and schema compliance.
Think step by step.
"""


CONVERSATION_SYSTEM_PROMPT = """You generate realistic benchmark data.
Return valid JSON only and follow the requested schema exactly.
Preserve the requested conversation structure and keep the dialogue natural.
Think step by step.
"""


QUESTION_SYSTEM_PROMPT = """You write several realistic memory-testing questions.
Return valid JSON only and follow the requested schema exactly.
Think step by step.
"""


ANSWER_SYSTEM_PROMPT = """You generate answer candidates for a memory benchmark.
Return valid JSON only and follow the requested schema exactly.
Match the answer style to the question while preserving correctness.
Think step by step.
"""


def _pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)


def _quality_feedback_block(sample: dict[str, Any], stage: str) -> str:
    feedback = sample.get("quality_feedback")
    if not isinstance(feedback, dict):
        return ""
    parts = []
    general_feedback = feedback.get("general")
    stage_feedback = feedback.get(stage)
    if general_feedback:
        parts.append(f"General filter feedback: {general_feedback}")
    if stage_feedback:
        parts.append(f"{stage.capitalize()} filter reason: {stage_feedback}")
    if not parts:
        return ""
    return "\nPrevious failed-attempt feedback to address:\n" + "\n".join(f"- {part}" for part in parts) + "\n"


def build_fact_selection_prompt(sample: dict[str, Any]) -> str:
    return f"""
Rewrite the HoH temporal source data below into a cleaner temporal fact bundle.

Goal:
- Produce one brief umbrella question for the topic. It should be the same underlying question across all snapshots.
- Preserve every dated snapshot fact exactly once.
- For each source fact, preserve the snapshot date and pair it with the correct answer for that date.
- Keep the result clearly temporal: the answer to the same question changes across dated snapshots.

Hard rules:
- Keep every source memory item exactly once. Do not drop, merge, or invent facts.
- The time condition must include the exact source snapshot date, e.g. "as of 2024-07-01".
- Do not turn the snapshot date into the date when the underlying event happened.
- The provided memory items have already removed redundant same-answer snapshots; preserve every provided memory item exactly once.
- Do not collapse multiple provided snapshots into one summary fact.
- The umbrella question should be short and natural, and should not include a time condition.
- The fact_statement for each item should be a compact sentence saying what the answer to the same underlying question was as of that snapshot date.
{_quality_feedback_block(sample, "conversation")}

Underlying source question:
{sample["source_question"]}

Source document:
{_pretty_json(sample.get("document", {}))}

Dated source memory items:
{_pretty_json(sample["memory_items"])}

Return JSON in this format:
```json
{{
  "temporal_question": "brief umbrella question",
  "temporal_facts": [
    {{
      "memory_id": "source memory id",
      "time_condition": "clean time qualifier",
      "answer_text": "one grounded answer alias",
      "fact_statement": "compact natural-language sentence"
    }}
  ],
  "fact_selection_notes": "one short sentence describing the temporal structure"
}}
```
Think step by step.
""".strip()


def build_session_plan_prompt(sample: dict[str, Any], session_plan_context: list[dict[str, Any]]) -> str:
    session_count = len(session_plan_context)
    return f"""
Plan independent user-assistant sessions for a nuanced-memory temporal benchmark.

High-priority planning rules:
- Produce exactly {session_count} session plans.
- Because this sample has {len(sample["selected_temporal_facts"])} temporal facts, the facts must be distributed across {session_count} sessions.
- Each session must contain between 1 and 3 assigned facts.
- Assign every temporal fact exactly once across the full session plan bundle.
- Group nearby or naturally related time-conditioned facts together when possible.
- The sessions must differ in both scenario and concrete event. They cannot all be the same memo, same document clean-up, or same conversation viewed from slightly different angles.
- For each session, choose exactly one conversation-type candidate from its provided pool.
- Prefer the candidate marked as preferred unless a fallback is clearly more natural for that grouped fact set.
- The chosen conversation type and flow should shape how the later dialogue unfolds, not just the label.
- Each session should feel like an ordinary task where those time-conditioned facts would naturally appear.
- Do not design any session so that it becomes a perfect self-contained timeline dump for the whole topic.
{_quality_feedback_block(sample, "conversation")}

Nuanced temporal definition:
- Multiple memories answer the same underlying question, and all can be true because they belong to different dated snapshots.
- The later benchmark question should ask for the answer under one specific snapshot date, not for the full timeline.
- The sessions should preserve enough dated snapshots that retrieval must discriminate the correct date from nearby dates.

Umbrella temporal question:
{sample["temporal_question"]}

Temporal facts to distribute:
{_pretty_json(sample["selected_temporal_facts"])}

Per-session conversation candidates:
{_pretty_json(session_plan_context)}

Return JSON in this format:
```json
{{
  "session_plans": [
    {{
      "session_id": "s1",
      "chosen_conversation_type": "one of the provided candidate conversation_type values",
      "chosen_conversation_flow": "the exact flow paired with that chosen type in the provided candidates",
      "scenario_label": "short_snake_case_label",
      "event_signature": "short_snake_case_event_id",
      "event_summary": "one sentence describing the concrete event or task",
      "opening_situation": "one sentence describing how the session begins",
      "user_goal": "what the user is trying to get done in this session",
      "assistant_role": "what kind of help the assistant gives in this session",
      "assigned_memory_ids": ["memory-id-1", "memory-id-2"],
      "temporal_grouping_rationale": "why these facts belong together in one session",
      "fact_integration_plan": "how the assigned temporal facts will appear naturally without turning into a full timeline dump",
      "distinct_from_other_sessions": "why this event is clearly different from the other planned sessions"
    }}
  ]
}}
```
Think step by step.
""".strip()


def build_session_prompt(
    sample: dict[str, Any],
    session_plan: dict[str, Any],
    assigned_facts: list[dict[str, Any]],
    min_rounds: int,
    max_rounds: int,
) -> str:
    return f"""
Generate one standalone user-assistant session for a nuanced temporal benchmark.

High-level goal:
- This session is only one piece of a larger temporal-memory sample.
- On its own, this session should feel normal and useful, not like a timeline quiz.
- It must naturally embed only its assigned facts.

What to generate:
- A realistic, everyday dialogue between a user and an assistant.
- At least {min_rounds} rounds, where one round means one user turn followed by one assistant turn.
- Aim for about {min_rounds}-{max_rounds} rounds overall.
- If needed, add small practical side detail or task-related distraction so the session reaches the desired length.

Temporal-memory rules:
- Multiple answers can all be true because they apply to different dated snapshots of the same underlying question.
- Preserve the exact snapshot date for each assigned fact.
- Make clear that the date is an "as of" / snapshot / version date for the answer, not necessarily the date of the underlying event.
- Mention each exact snapshot date naturally; after the date has been established, later turns may refer to "that June snapshot", "the autumn version", or "the older note" instead of robotically repeating the full YYYY-MM-DD string.
- Do not mention or imply facts from other hidden sessions.
- Do not compress all assigned facts into one single assistant turn unless there is no other natural option.
- Do not end with a full neat recap that makes later retrieval trivial.
- Spread the assigned facts across the dialogue naturally.

Umbrella temporal question:
{sample["temporal_question"]}

Assigned temporal facts for this session:
{_pretty_json(assigned_facts)}

Approved session plan:
- Session id: {session_plan["session_id"]}
- Chosen conversation type: {session_plan["chosen_conversation_type"]}
- Type description: {session_plan["chosen_conversation_type_description"]}
- Chosen candidate tier: {session_plan["chosen_candidate_tier"]}
- Chosen conversation flow: {session_plan["chosen_conversation_flow"]}
- Scenario label: {session_plan["scenario_label"]}
- Event signature: {session_plan["event_signature"]}
- Event summary: {session_plan["event_summary"]}
- Opening situation: {session_plan["opening_situation"]}
- User goal: {session_plan["user_goal"]}
- Assistant role: {session_plan["assistant_role"]}
- Temporal grouping rationale: {session_plan["temporal_grouping_rationale"]}
- Fact integration plan: {session_plan["fact_integration_plan"]}
- Distinctness note: {session_plan["distinct_from_other_sessions"]}

Critical session rules:
- Follow the approved plan closely.
- Let the chosen conversation type shape the mode of help.
- Let the chosen flow shape progression across turns.
- Keep the concrete event clearly different from the other hidden sessions.
- Do not let the user ask the umbrella temporal question verbatim as the main task.
- Do not turn the dialogue into a clean chart, table, spreadsheet, or full chronology covering the entire topic.
- Keep the dated answer snapshots explicit enough to matter, but conversational enough to feel natural.
- Avoid making the task a request to find the event's occurrence date. The temporal condition is the remembered snapshot date.
- Use `chosen_scenario` equal to the planned `scenario_label`, unless a very close variant is clearly more natural.
{_quality_feedback_block(sample, "conversation")}

Return JSON in this format:
```json
{{
  "session_id": "{session_plan["session_id"]}",
  "chosen_scenario": "short_scenario_label",
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "coverage_notes": [
    {{"memory_id": "memory-id", "covered_in_turns": [1, 3]}}
  ]
}}
```
Think step by step.
""".strip()


def build_question_prompt(sample: dict[str, Any], session_payload: list[dict[str, Any]]) -> str:
    return f"""
Write exactly 3 new follow-up questions based on the sessions below.

Requirements:
- Produce exactly 3 question objects.
- Every question must ask for the answer to the same underlying question under exactly one temporal condition.
- Every question must resolve to exactly one target memory_id from the temporal facts.
- Cover different angles and difficulty levels: include exactly one easy, one medium, and one hard question.
- Easy questions may use a direct as-of/snapshot date.
- Medium questions should usually use a natural event, task, draft, note, version, or session anchor instead of a raw date.
- Hard questions should avoid direct target-date string matching; use relative temporal reasoning such as "before the final revision", "after the earlier update but before the later note", or "the latest answer we had before <event>".
- The benchmark difficulty should come from identifying the right temporal slice across sessions, not from asking for a full timeline.
- Questions should sound like natural later follow-ups from the same user.
- Each question must still be self-contained enough for a memory agent that has stored many unrelated sessions. It should include enough topic/entity and fact-property clues to locate the relevant memory, not only a local phrase from one session.
- For non-direct-date questions, include at least one concrete topic clue and one concrete temporal/task/version clue. Example topic clues: the entity name, dataset subject, team, person, organization, title, or the property being asked. Example temporal/task/version clues: the draft, tracker, briefing, checklist, update, proof, handoff, or before/after relation.
- Avoid under-specified references such as "the current snapshot", "that version", "the note format", "the compact notes-app format", or "the thing we marked current" unless the same question also names the concrete topic and target property.
- Bad example: "In the compact notes-app format you suggested, what match should go on the Reference match line for the snapshot I marked as current?" This lacks enough topic context. Better: "In the compact Cameroon caps/goals notes-app format, what match should go on the Reference match line for the snapshot I marked as current?"
- Do not ask for all dates, all periods, the full timeline, chronological order, or a complete recap.
- Do not ask "when did the underlying event happen?" unless the question also clearly asks what the remembered answer was under a snapshot/version condition.
- Do not make it a meta-question about "which session said what."
- Do not make every question a direct "as of YYYY-MM-DD" lookup.
- A sample must never contain more than 3 questions.
- It is acceptable for one question to keep the underlying question wording close to the source question, but the full set should include harder anchored questions.
{_quality_feedback_block(sample, "question")}

Nuanced temporal definition:
- The same underlying question has different answers under different dated snapshots.
- Each question should make one specific temporal snapshot decisive, so only one answer is correct.

Allowed temporal_anchor_type values:
- direct_date: the question names the exact as-of/snapshot date.
- event_anchor: the question identifies the relevant time through a concrete event in one session.
- session_task_anchor: the question identifies the relevant time through the task being done in one session.
- version_anchor: the question identifies the relevant time through an older/newer draft, note, version, or revision.
- relative_time: the question identifies the relevant time through before/after/earlier/later relationships.
- latest_before: the question asks for the latest remembered answer before a concrete event or revision.

Umbrella temporal question:
{sample["temporal_question"]}

Temporal facts:
{_pretty_json(sample["selected_temporal_facts"])}

Independent sessions:
{_pretty_json(session_payload)}

Return JSON in this format:
```json
{{
  "questions": [
    {{
      "difficulty": "easy",
      "temporal_anchor_type": "direct_date",
      "target_memory_id": "memory-id from temporal facts",
      "question": "...?",
      "question_design_notes": "why this question uniquely targets that temporal fact"
    }},
    {{
      "difficulty": "medium",
      "temporal_anchor_type": "event_anchor",
      "target_memory_id": "memory-id from temporal facts",
      "question": "...?",
      "question_design_notes": "why this question uniquely targets that temporal fact"
    }},
    {{
      "difficulty": "hard",
      "temporal_anchor_type": "latest_before",
      "target_memory_id": "memory-id from temporal facts",
      "question": "...?",
      "question_design_notes": "why this question uniquely targets that temporal fact"
    }}
  ]
}}
```
Think step by step.
""".strip()


def build_answer_prompt(
    sample: dict[str, Any],
    session_payload: list[dict[str, Any]],
    question_item: dict[str, Any],
) -> str:
    target_fact = next(
        fact
        for fact in sample["selected_temporal_facts"]
        if fact["memory_id"] == question_item["target_memory_id"]
    )
    return f"""
Generate answer candidates for the question below.

Requirements:
- Produce exactly three correct answers and exactly three incorrect answers.
- Each answer must be a dictionary with one key: "text".
- Every correct answer must mean the same thing as the target answer only.
- Every correct answer should preserve the key entities, numbers, or names from the target answer; do not replace the target answer with only a vague description.
- Correct answers may mention the target temporal anchor briefly, but they must not expand into the whole timeline.
- Keep the answers natural and concise.
- Incorrect answers should reflect realistic temporal confusions: choosing the answer from a different snapshot date for the same underlying question.
- Do not use random unrelated wrong answers.
{_quality_feedback_block(sample, "answer")}

Nuanced temporal definition:
- Several answers may all be true, but only under different dated snapshots.
- Correct answers must return the single answer for the snapshot date asked in the question.

Target temporal fact:
{_pretty_json(target_fact)}

Temporal facts:
{_pretty_json(sample["selected_temporal_facts"])}

Independent sessions:
{_pretty_json(session_payload)}

Question item:
{_pretty_json(question_item)}

Return JSON in this format:
```json
{{
  "correct_answers": [
    {{"text": "..."}},
    {{"text": "..."}},
    {{"text": "..."}}
  ],
  "incorrect_answers": [
    {{"text": "..."}},
    {{"text": "..."}},
    {{"text": "..."}}
  ]
}}
```
Think step by step.
""".strip()
