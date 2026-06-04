from __future__ import annotations

import json
from typing import Any


FACT_SELECTION_SYSTEM_PROMPT = """You convert source context QA pairs into a cleaner benchmark-ready fact bundle.
Return valid JSON only and follow the requested schema exactly.
Keep the question ambiguous and preserve all context-conditioned facts.
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


QUESTION_SYSTEM_PROMPT = """You write realistic context-specific memory-testing questions.
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
Rewrite the context source data below into a cleaner context fact bundle.

Goal:
- Produce one brief ambiguous context question for the topic.
- Preserve every source fact.
- For each source fact, rewrite the decisive context cleanly and pair it with the correct answer.
- Keep the result clearly contextual: the different answers should all remain true, but under different contexts.

Hard rules:
- Keep every source memory item exactly once. Do not drop, merge, or invent facts.
- Preserve the decisive context for each fact, such as role, location, task, scope, version, definition, unit, or attribute.
- Do not collapse multiple facts into one summary fact.
- The broad question must stay ambiguous and should not secretly name one decisive context.
- For each fact, make `context_anchor` short and concrete so later answers can map answers back to contexts clearly.
- The `fact_statement` should be a compact natural-language sentence that already binds the answer to its context.
{_quality_feedback_block(sample, "conversation")}

Ambiguous source question:
{sample["source_question"]}

Source memory items:
{_pretty_json(sample["memory_items"])}

Return JSON in this format:
```json
{{
  "context_question": "brief ambiguous question",
  "context_facts": [
    {{
      "memory_id": "source memory id",
      "context_condition": "clean decisive context",
      "context_anchor": "short concrete anchor label",
      "answer_text": "one grounded answer alias",
      "fact_statement": "compact natural-language sentence"
    }}
  ],
  "fact_selection_notes": "one short sentence describing the context structure"
}}
```
Think step by step.
""".strip()


def build_session_plan_prompt(sample: dict[str, Any], session_plan_context: list[dict[str, Any]]) -> str:
    session_count = len(session_plan_context)
    return f"""
Plan independent user-assistant sessions for a nuanced-memory context benchmark.

High-priority planning rules:
- Produce exactly {session_count} session plans.
- Because this sample has {len(sample["selected_context_facts"])} context facts, the facts must be distributed across {session_count} sessions.
- Each session must contain between 1 and 3 assigned facts.
- Assign every context fact exactly once across the full session plan bundle.
- Group naturally related contexts together when possible.
- The sessions must differ in both scenario and concrete event. They cannot all be the same note clean-up, same write-up, or same lookup framed three slightly different ways.
- For each session, choose exactly one conversation-type candidate from its provided pool.
- Prefer the candidate marked as preferred unless a fallback is clearly more natural for that grouped fact set.
- The chosen conversation type and flow should shape how the later dialogue unfolds, not just the label.
- Each session should feel like an ordinary task where those context-conditioned facts would naturally appear.
- Do not design any session so that it becomes a perfect self-contained dump of every context-answer pair for the whole topic.
{_quality_feedback_block(sample, "conversation")}

Nuanced context definition:
- Multiple memories about the same topic can all be true, but under different contexts such as role, location, task, scope, version, definition, unit, or attribute.
- The later benchmark question should specify one decisive context, so the system must retrieve the matching context-conditioned answer while ignoring nearby alternatives.

Broad ambiguous question:
{sample["context_question"]}

Context facts to distribute:
{_pretty_json(sample["selected_context_facts"])}

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
      "context_grouping_rationale": "why these facts belong together in one session",
      "fact_integration_plan": "how the assigned context facts will appear naturally without turning into a full inventory dump",
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
Generate one standalone user-assistant session for a nuanced context benchmark.

High-level goal:
- This session is only one piece of a larger context-memory sample.
- On its own, this session should feel normal and useful, not like a context quiz.
- It must naturally embed only its assigned facts.

What to generate:
- A realistic, everyday dialogue between a user and an assistant.
- At least {min_rounds} rounds, where one round means one user turn followed by one assistant turn.
- Aim for about {min_rounds}-{max_rounds} rounds overall.
- If needed, add small practical side detail or task-related distraction so the session reaches the desired length.

Context-memory rules:
- Multiple answers can all be true because they apply to different contexts.
- Preserve the decisive context qualifiers for the assigned facts.
- Do not mention or imply facts from other hidden sessions.
- Do not compress all assigned facts into one single assistant turn unless there is no other natural option.
- Do not end with a neat recap that lists every context-answer pair in the session all at once.
- Spread the assigned facts across the dialogue naturally.

Broad ambiguous question:
{sample["context_question"]}

Assigned context facts for this session:
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
- Context grouping rationale: {session_plan["context_grouping_rationale"]}
- Fact integration plan: {session_plan["fact_integration_plan"]}
- Distinctness note: {session_plan["distinct_from_other_sessions"]}

Critical session rules:
- Follow the approved plan closely.
- Let the chosen conversation type shape the mode of help.
- Let the chosen flow shape progression across turns.
- Keep the concrete event clearly different from the other hidden sessions.
- Do not let the user ask the broad ambiguous question verbatim as the main task.
- Do not turn the dialogue into a clean chart, table, spreadsheet, or exhaustive role/state/league/version list.
- Keep the context qualifiers explicit enough to matter, but conversational enough to feel natural.
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
Write one or two new follow-up questions based on the sessions below.

Requirements:
- Produce one or two question objects.
- Every question must specify the decisive context and target exactly one context fact.
- The answer to each question should be a single context-specific answer, not a full list of all contexts.
- If you produce two questions, they must target different memory_ids and ask from meaningfully different angles. Do not make near-paraphrases.
- It should sound like a natural later follow-up from the same user.
- The question must be understandable if read on its own after many unrelated sessions have been stored.
- Include enough topic/entity and context clues to locate the relevant memory.
- Good context clues include role, location, jurisdiction, task, scope, version, definition, unit, attribute, object range, or scenario.
- Do not ask the broad ambiguous question directly.
- Do not ask for all contexts, all names, all roles, all versions, or a complete list.
- Do not produce vague questions such as "what is it in that case?" or "which one should I use for that version?"
{_quality_feedback_block(sample, "question")}

Nuanced context definition:
- Multiple related answers can all be true, but under different contexts.
- In this subtype, the final benchmark question should include one decisive context, so the correct response is a single answer for that context.

Broad ambiguous question:
{sample["context_question"]}

Context facts:
{_pretty_json(sample["selected_context_facts"])}

Independent sessions:
{_pretty_json(session_payload)}

Return JSON in this format:
```json
{{
  "questions": [
    {{
      "target_memory_id": "memory-id from context facts",
      "context_anchor": "short decisive context named by the question",
      "question": "...?",
      "question_design_notes": "why this question uniquely targets that context fact"
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
        for fact in sample["selected_context_facts"]
        if fact["memory_id"] == question_item["target_memory_id"]
    )
    return f"""
Generate answer candidates for the question below.

Requirements:
- Produce exactly three correct answers and exactly three incorrect answers.
- Each answer must be a dictionary with one key: "text".
- Every correct answer must be the single answer for the target context only.
- Correct answers may briefly restate the context, but they must not expand into all context-conditioned answers.
- Incorrect answers should reflect realistic context confusions: choosing the answer from another context for the same broad topic.
- Do not use random unrelated wrong answers.
- Keep the style natural and close to how a user or assistant would answer.
{_quality_feedback_block(sample, "answer")}

Nuanced context definition:
- Several answers in the sessions may all be true, but only under different contexts.
- Correct answers must return the answer for the context specified in the question.

Target context fact:
{_pretty_json(target_fact)}

Context facts:
{_pretty_json(sample["selected_context_facts"])}

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
