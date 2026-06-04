from __future__ import annotations

import json
from typing import Any


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


QUESTION_SYSTEM_PROMPT = """You write one realistic memory-testing question.
Return valid JSON only and follow the requested schema exactly.
Think step by step.
"""


ANSWER_SYSTEM_PROMPT = """You generate answer candidates for a memory benchmark.
Return valid JSON only and follow the requested schema exactly.
Match the answer style to the question while preserving correctness.
Think step by step.
"""


FACT_SELECTION_SYSTEM_PROMPT = """You convert source QA pairs into a contradictory fact setup for a memory benchmark.
Return valid JSON only and follow the requested schema exactly.
Make the result genuinely contradictory rather than temporal or contextual.
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


def _public_conflicting_facts(selected_conflicting_facts: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "session_id": fact["session_id"],
            "answer_text": fact["answer_text"],
            "fact_source_role": fact["fact_source_role"],
        }
        for fact in selected_conflicting_facts
    ]


def _subtype_definition(subtype: str) -> str:
    if subtype == "a_user_vs_user":
        return (
            "Contradictory memory, subtype a: user provided fact vs. user provided fact.\n"
            "- Each contradictory claim should come from a different user-side memory session.\n"
            "- The assistant should not resolve or correct the user inside those sessions.\n"
            "- The contradiction should only become visible when multiple sessions are recalled together."
        )
    if subtype == "b_user_vs_non_user":
        return (
            "Contradictory memory, subtype b: user provided fact vs. non-user fact.\n"
            "- At least one claim should be introduced by the user in one session.\n"
            "- At least one conflicting claim should be introduced by the assistant in a different session, possibly as a note, lookup, reference, or remembered source.\n"
            "- The sessions should each feel locally normal and should not explicitly debate each other."
        )
    if subtype == "c_non_user_vs_non_user":
        return (
            "Contradictory memory, subtype c: non-user fact vs. non-user fact.\n"
            "- The conflicting claims should all come from the assistant side across different sessions.\n"
            "- The user should not be the source of the contradictory facts.\n"
            "- Each session should look ordinary on its own; the contradiction only emerges across sessions."
        )
    raise ValueError(f"Unknown contradictory subtype: {subtype}")


def _session_source_guidance(fact_source_role: str) -> str:
    if fact_source_role == "user":
        return (
            "Session fact-source requirement:\n"
            "- The user must be the source of the target claim in this session.\n"
            "- The assistant may react, organize, plan around it, or lightly help phrase surrounding material, but should not challenge it or offer competing answers.\n"
            "- Do not mention that there are other possible answers."
        )
    if fact_source_role == "assistant":
        return (
            "Session fact-source requirement:\n"
            "- The assistant must be the source of the target claim in this session.\n"
            "- The user may ask for help, ask a factual question, check a detail during some task, or refer to messy notes, and the assistant should supply the claim naturally.\n"
            "- Do not mention that there are other possible answers."
        )
    raise ValueError(f"Unknown fact_source_role: {fact_source_role}")


def build_fact_selection_prompt(sample: dict[str, Any]) -> str:
    return f"""
Select two source QA entries and rewrite them into one contradictory setup.

Goal:
- Create a single canonical conflict question that both selected answers appear to answer under the same apparent condition.
- The result must be contradictory memory, not nuanced memory.
- Remove source qualifiers that would reconcile the answers, such as explicit year, version, edition, role variant, location, scope, or scenario markers.
- The canonical conflict question is an internal benchmark scaffold for later dialogue and QA generation. It may later be reused directly or lightly paraphrased.

Hard rules:
- Select exactly two different source QA entries.
- Use only answer content grounded in the selected source entries. Do not invent a new answer.
- The canonical conflict question must not quietly preserve a qualifier that makes both answers true.
- Do not output a question like "as of 2016 vs 2017", "in the 1967 or 2016 version", "for the TV series", or anything else that turns the setup back into temporal/contextual nuance.
- Good outcome: two mutually incompatible answers to one plain question.
- Bad outcome: two answers that are still separated by explicit time/version/context qualifiers.
{_quality_feedback_block(sample, "conversation")}

Source question:
{sample["source_question"]}

Available source QA pairs:
{_pretty_json(sample["memory_items"])}

Return JSON in this format:
```json
{{
  "conflict_question": "one canonical contradictory question without the source qualifiers that would reconcile the answers",
  "selected_facts": [
    {{
      "memory_id": "memory id from the source list",
      "contradictory_answer": "answer text grounded in that source entry",
      "why_it_fits": "one short sentence"
    }},
    {{
      "memory_id": "memory id from the source list",
      "contradictory_answer": "answer text grounded in that source entry",
      "why_it_fits": "one short sentence"
    }}
  ],
  "conflict_rationale": "one short sentence saying why these two answers now conflict under the same apparent condition"
}}
```
Think step by step.
""".strip()


def build_session_plan_prompt(
    sample: dict[str, Any],
    session_plan_context: list[dict[str, Any]],
) -> str:
    return f"""
Plan one independent session for each conflicting fact below.

This is the session-planning step of a contradictory-memory benchmark.
- The facts are already selected.
- Your job here is only to design two different session blueprints before dialogue writing begins.

High-priority planning rules:
- Produce exactly one session plan per selected fact.
- The two plans must differ in both scenario and concrete event.
- They must not be the same outing, same watch party, same family dinner, same office event, same shared document, or the same practical task viewed from two angles.
- Bad example: both sessions are about the same family movie night, with one plan for snacks and one plan for trivia.
- Good example: one session is about choosing a movie with relatives, while the other is about buying tickets, updating a watchlist, preparing a kid activity card, or answering a separate casual question on another day.
- For each session, choose exactly one conversation-type candidate from its provided pool.
- Prefer the candidate marked as preferred unless it is clearly awkward for that fact and event. If a fallback fits much better, use it.
- The chosen conversation type and flow should shape the turn structure of the later dialogue, not just the topic label.
- Scenario labels may reuse ordinary task wording or you may invent concise snake_case labels that match the event naturally.
- Keep each plan ordinary, low-drama, and natural for daily assistant use.
- Each plan must make it easy to embed exactly one target claim while staying silent about any contradiction.
{_quality_feedback_block(sample, "conversation")}

Subtype reminder:
{_subtype_definition(sample["contradictory_subtype"])}

Canonical conflict question:
{sample["conflict_question"]}

Selected conflicting facts:
{_pretty_json(_public_conflicting_facts(sample["selected_conflicting_facts"]))}

Fact-specific planning context:
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
      "fact_integration_plan": "how the target fact will appear naturally without highlighting contradiction",
      "distinct_from_other_session": "why this event is clearly different from the other planned session"
    }},
    {{
      "session_id": "s2",
      "chosen_conversation_type": "one of the provided candidate conversation_type values",
      "chosen_conversation_flow": "the exact flow paired with that chosen type in the provided candidates",
      "scenario_label": "short_snake_case_label",
      "event_signature": "short_snake_case_event_id",
      "event_summary": "one sentence describing the concrete event or task",
      "opening_situation": "one sentence describing how the session begins",
      "user_goal": "what the user is trying to get done in this session",
      "assistant_role": "what kind of help the assistant gives in this session",
      "fact_integration_plan": "how the target fact will appear naturally without highlighting contradiction",
      "distinct_from_other_session": "why this event is clearly different from the other planned session"
    }}
  ]
}}
```
Think step by step.
""".strip()


def build_session_prompt(
    sample: dict[str, Any],
    target_fact: dict[str, Any],
    session_plan: dict[str, Any],
    min_rounds: int,
    max_rounds: int,
) -> str:
    return f"""
Generate one standalone user-assistant session for a contradictory-memory benchmark.

Important high-level goal:
- This session is only one piece of a larger contradictory-memory sample.
- On its own, this session should feel normal and should not mention contradiction, disagreement, or other sessions.
- It should naturally embed one target factual claim and then move on.

What to generate:
- A realistic, everyday dialogue between a user and an assistant.
- At least {min_rounds} rounds, where one round means one user turn followed by one assistant turn.
- Aim for about {min_rounds}-{max_rounds} rounds overall.
- If needed, you may add a small amount of natural side detail or task-related distraction so the session reaches the desired length.
- The target fact should appear naturally and briefly inside the task, then the conversation should continue with ordinary adjacent details.

Contradictory-memory setup:
{_subtype_definition(sample["contradictory_subtype"])}

This session's target claim:
- Canonical conflict question it implicitly answers: {sample["conflict_question"]}
- Claim to embed as if it were the answer: {target_fact["answer_text"]}

Approved session plan:
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
- Fact integration plan: {session_plan["fact_integration_plan"]}
- Distinctness note: {session_plan["distinct_from_other_session"]}

Critical session rules:
- Follow the approved session plan closely. Do not drift into a different event or collapse back into the other hidden session's event family.
- Let the chosen conversation type shape the overall mode of help. For example, a planning session should feel like planning, a troubleshooting session should feel like diagnosing, and an artifact-production session should feel like drafting or producing something.
- Let the chosen conversation flow shape progression across turns. The beginning, middle, and ending of the dialogue should visibly follow that flow.
- The concrete event in this session must stay clearly different from the other planned session.
- Let the opening naturally reflect the stated opening situation.
- Make the session feel like a normal practical exchange tied to the planned event, not a generic fact lookup.
- Do not mention contradiction, "debate," "different notes," "mixed memories," or that this fact may conflict with another session.
- Treat the claim like a plain answer to the canonical conflict question above.
- Do not introduce year/version/edition/remake/TV-series qualifiers or similar context that would explain away the contradiction, unless the canonical conflict question itself already contains that qualifier.
- Do not explicitly ask the assistant to resolve a contradiction.
- Keep the session natural and low-drama. The target fact can pass by in one sentence among other ordinary details.
- Add a little harmless distractor material when helpful, but do not introduce a second competing answer in this same session.
- Do not make every turn about the target fact. After the fact appears, let the conversation continue with adjacent logistics, choices, reminders, or organization.
- Use `chosen_scenario` equal to the planned `scenario_label`, unless a very close variant is clearly more natural.
{_quality_feedback_block(sample, "conversation")}

{_session_source_guidance(target_fact["fact_source_role"])}

Return JSON in this format:
```json
{{
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
Write one new question based on the independent sessions below.

Requirements:
- Produce exactly one question.
- The question must be answerable from the stored sessions alone.
- It should target the main disputed point in the most natural, benchmark-useful way.
- The question should be self-contained and natural.
- It should not mention "contradiction," "conflict," or "different sessions."
- It should not add qualifiers that would resolve the contradiction.
- Start from the canonical conflict question below. You may reuse it directly or make a very light paraphrase, but do not reintroduce qualifiers that resolve the contradiction.
- Do not turn it into a meta-question about notes, clarification strategy, or process unless that is clearly the best wording.
{_quality_feedback_block(sample, "question")}

Relationship definition:
- Contradictory memory: independent remembered sessions contain incompatible claims under the same apparent condition.
- The question should expose that disputed point without quietly choosing one side.

Canonical conflict question:
{sample["conflict_question"]}

Selected conflicting facts:
{_pretty_json(_public_conflicting_facts(sample["selected_conflicting_facts"]))}

Independent sessions:
{_pretty_json(session_payload)}

Return JSON in this format:
```json
{{
  "question": "..."
}}
```
Think step by step.
""".strip()


def build_answer_prompt(
    sample: dict[str, Any],
    session_payload: list[dict[str, Any]],
    question: str,
) -> str:
    return f"""
Generate answer candidates for the question below.

Requirements:
- Produce exactly three correct answers and exactly three incorrect answers.
- Each answer must be a dictionary with one key: "text".
- Every correct answer must explicitly state that the remembered sessions conflict or remain unresolved.
- Correct answers must not choose a side, must not pretend one claim is final, and must not quietly repair the contradiction.
- Good correct answers may name the competing claims and say they need clarification or verification.
- Do not mention hidden source qualifiers such as explicit year, edition, version marker, remake label, or other contextual distinctions unless they already appear in the question itself.
- Incorrect answers should reflect failure modes such as: picking one side, pretending one session overrules the other, inventing a clean resolution, or ignoring the practical framing of the question.
- Keep the answers concise but complete. Do not output giant explanations.
{_quality_feedback_block(sample, "answer")}

Subtype reminder:
{_subtype_definition(sample["contradictory_subtype"])}

Selected conflicting facts:
{_pretty_json(_public_conflicting_facts(sample["selected_conflicting_facts"]))}

Independent sessions:
{_pretty_json(session_payload)}

Question:
{question}

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
