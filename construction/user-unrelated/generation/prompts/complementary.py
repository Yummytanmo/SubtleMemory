from __future__ import annotations

import json
from typing import Any


FACT_SELECTION_SYSTEM_PROMPT = """You convert source complementary-memory data into a cleaner benchmark-ready fact bundle.
Return valid JSON only and follow the requested schema exactly.
Keep the result complementary rather than contradictory or contextual.
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


QUESTION_SYSTEM_PROMPT = """You write one realistic memory-testing question.
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


def _is_musique(sample: dict[str, Any]) -> bool:
    source_dataset = str(sample.get("source_dataset", "")).casefold()
    sample_id = str(sample.get("sample_id", "")).casefold()
    source_path = str(sample.get("source_path", "")).casefold()
    return "musique" in source_dataset or "musique" in sample_id or "musique" in source_path


def _quality_feedback_block(sample: dict[str, Any], stage: str) -> str:
    feedback = sample.get("quality_feedback")
    if not feedback:
        return ""
    stage_feedback = feedback.get(stage) if isinstance(feedback, dict) else None
    general_feedback = feedback.get("general") if isinstance(feedback, dict) else None
    parts = []
    if general_feedback:
        parts.append(f"General quality feedback from a previous filter attempt: {general_feedback}")
    if stage_feedback:
        parts.append(f"{stage.capitalize()} quality feedback from a previous filter attempt: {stage_feedback}")
    if not parts:
        return ""
    return "\nPrevious failed-attempt feedback to address:\n" + "\n".join(f"- {part}" for part in parts) + "\n"


def _public_selected_facts(selected_facts: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "memory_id": fact["memory_id"],
            "fact_role": fact["fact_role"],
            "fact_statement": fact["fact_statement"],
            "answer_contribution": fact["answer_contribution"],
        }
        for fact in selected_facts
    ]


def _fact_selection_definition(sample: dict[str, Any]) -> str:
    subtype = sample["complementary_subtype_label"]
    if subtype == "k_eq_1":
        return (
            "Complementary memory, type1, k=1:\n"
            "- Pick exactly one decisive required fact and at least one distractor fact.\n"
            "- The later benchmark question must be answerable from the decisive fact alone.\n"
            "- The distractors should stay related to the same topic but should not determine the answer."
        )
    if subtype == "k_gt_1":
        return (
            "Complementary memory, type1, k>1:\n"
            "- Build a benchmark-ready question that requires integrating multiple selected required facts.\n"
            "- If the source sample has too many facts, narrow to a coherent subset of 2-6 selected facts.\n"
            "- You may drop extra source facts when needed for manageability, but the new question must still require multi-hop reasoning or multi-fact synthesis.\n"
            "- Prefer compact reasoning chains with 2-4 truly necessary required facts instead of long bridge-heavy chains that would force an unnatural riddle-like question.\n"
            "- Avoid selecting redundant required facts when one fact already subsumes another hop.\n"
            "- For MuSiQue-like sources, choose a fact subset that can later be asked about in plain language; do not select chains that can only be tested by hiding names behind descriptions.\n"
            "- Optional distractors are allowed, but the key requirement is that effective_k must be at least 2."
        )
    if subtype == "any_one_of_n":
        return (
            "Complementary memory, type2, any-one-of-n:\n"
            "- Select 2-6 equivalent facts that all support the same answer.\n"
            "- The later benchmark question should be answerable from any one of those selected facts.\n"
            "- Preserve redundancy, not contradiction."
        )
    raise ValueError(f"Unknown complementary subtype label: {subtype}")


def _session_definition(sample: dict[str, Any]) -> str:
    subtype = sample["complementary_subtype_label"]
    if subtype == "k_eq_1":
        return (
            "Session-writing rule for complementary k=1:\n"
            "- Keep the one decisive fact embedded inside ordinary context.\n"
            "- Do not isolate it as a short naked answer line.\n"
            "- Let the distractor facts feel nearby and plausible, but still non-decisive."
        )
    if subtype == "k_gt_1":
        return (
            "Session-writing rule for complementary k>1:\n"
            "- The required facts must stay distributed.\n"
            "- Do not collapse all decisive facts into one neat recap or one final answer-bearing turn.\n"
            "- Each session should only carry its assigned part of the reasoning chain or composition."
        )
    if subtype == "any_one_of_n":
        return (
            "Session-writing rule for complementary any-one-of-n:\n"
            "- Keep the selected equivalent facts redundant but natural.\n"
            "- Different sessions may phrase the same answer differently, but they should all point to the same answer."
        )
    raise ValueError(f"Unknown complementary subtype label: {subtype}")


def _question_definition(sample: dict[str, Any]) -> str:
    subtype = sample["complementary_subtype_label"]
    if subtype == "k_eq_1":
        return (
            "Write a natural follow-up question whose answer depends on the one decisive fact, not on the distractors."
        )
    if subtype == "k_gt_1":
        return (
            f"Write a natural follow-up question that requires integrating at least {sample['effective_k']} required facts. "
            "It should not become answerable from one single utterance or one single session."
        )
    if subtype == "any_one_of_n":
        return (
            "Write a natural follow-up question targeting the shared fact that could be recovered from any one of the selected equivalent facts."
        )
    raise ValueError(f"Unknown complementary subtype label: {subtype}")


def _answer_definition(sample: dict[str, Any]) -> str:
    subtype = sample["complementary_subtype_label"]
    if subtype == "k_eq_1":
        return (
            "Incorrect answers should reflect nearby topic confusions or distractor contamination, not random unrelated mistakes. "
            "Prefer wrong answers that are visibly pulled toward distractor details in the sessions, rather than arbitrary same-format substitutions."
        )
    if subtype == "k_gt_1":
        return (
            "Incorrect answers should reflect realistic integration failures: partial composition, wrong pairings, dropped entities, or mixing in distractor material."
        )
    if subtype == "any_one_of_n":
        return (
            "Incorrect answers should be plausible same-domain alternatives, not contradictions or meta-comments."
        )
    raise ValueError(f"Unknown complementary subtype label: {subtype}")


def build_fact_selection_prompt(sample: dict[str, Any]) -> str:
    musique_fact_guidance = ""
    if _is_musique(sample):
        musique_fact_guidance = """
MuSiQue-specific fact-selection guidance:
- Do not preserve the full original MuSiQue chain if it would force a riddle-like user question.
- Prefer 2-3 required facts total. Use 4 only when the chain can still be asked in plain, natural language.
- The selected chain must support a natural one-sentence follow-up question using concrete names or short anchors from the sessions.
- Avoid selected chains whose only possible question would need phrases like "the city where...", "the river through the city where...", or "the system that the game named after...".
- Prefer a useful, compact subset over maximal hop count. The benchmark should still require multi-fact retrieval, but not at the cost of unnatural wording.
"""
    return f"""
Rewrite the complementary source data below into a cleaner benchmark-ready complementary fact bundle.

Goal:
- Produce one benchmark-ready complementary question.
- Produce one canonical answer for that question.
- Select a manageable fact subset for multi-session generation.

Hard rules:
- Keep the result complementary: related facts may distract, support, or redundantly restate the answer, but they must not create contradictory or context-resolved multi-answer behavior.
- Keep the selected fact set manageable: choose between 2 and 6 selected facts total.
- Each selected fact must reference one source memory_id.
- The selected fact roles must obey the subtype definition below.
- The canonical answer should be natural text, not a debug note.
{musique_fact_guidance}
{_quality_feedback_block(sample, "conversation")}

Subtype definition:
{_fact_selection_definition(sample)}

Source question:
{sample["source_question"]}

Source answer for hidden grounding:
{sample["source_answer_text"]}

Available source memory items:
{_pretty_json(sample["memory_prompt_items"])}

Return JSON in this format:
```json
{{
  "complementary_question": "benchmark-ready question",
  "canonical_answer": "natural-language canonical answer",
  "effective_k": 1,
  "selected_facts": [
    {{
      "memory_id": "source memory id",
      "fact_role": "required or distractor or equivalent",
      "fact_statement": "compact natural-language fact",
      "answer_contribution": "how this fact contributes to the answer or acts as distraction",
      "selection_rationale": "why this fact was selected"
    }}
  ],
  "fact_selection_notes": "one short sentence describing why the selected facts form a complementary set"
}}
```
Think step by step.
""".strip()


def build_session_plan_prompt(sample: dict[str, Any], session_plan_context: list[dict[str, Any]]) -> str:
    session_count = len(session_plan_context)
    return f"""
Plan independent user-assistant sessions for a complementary-memory benchmark.

High-priority planning rules:
- Produce exactly {session_count} session plans.
- Because this sample has {len(sample["selected_complementary_facts"])} selected facts, distribute them across {session_count} sessions.
- Each session must contain between 1 and 3 assigned facts.
- Assign every selected fact exactly once across the full session plan bundle.
- Group naturally related facts together when possible.
- The sessions must differ in both scenario and concrete event.
- For each session, choose exactly one conversation-type candidate from its provided pool.
- Prefer the candidate marked as preferred unless a fallback is clearly more natural.
- The chosen conversation type and flow should shape how the later dialogue unfolds, not just the label.
- Each session should feel like an ordinary task where those selected facts would naturally appear.
- Facts assigned to other sessions must stay unmentioned.
- If a session does not contain answer-bearing facts, do not design an event that naturally forces out the canonical answer anyway.
{_quality_feedback_block(sample, "conversation")}

Subtype reminder:
{_session_definition(sample)}

Planning-specific rule for this sample:
- Benchmark question: {sample["complementary_question"]}
- Canonical answer: {sample["canonical_answer"]}
- Effective k: {sample["effective_k"]}
- If this is k>1, distribute the required facts so the later question genuinely needs cross-session integration.
- If this is k=1, keep the decisive fact from becoming too exposed.

Selected complementary facts:
{_pretty_json(_public_selected_facts(sample["selected_complementary_facts"]))}

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
      "grouping_rationale": "why these facts belong together in one session",
      "fact_integration_plan": "how the assigned facts will appear naturally without turning into a dump",
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
Generate one standalone user-assistant session for a complementary benchmark.

High-level goal:
- This session is only one piece of a larger complementary-memory sample.
- On its own, this session should feel natural and useful, not like a fact quiz.
- It must naturally embed only its assigned facts.

What to generate:
- A realistic, everyday dialogue between a user and an assistant.
- At least {min_rounds} rounds, where one round means one user turn followed by one assistant turn.
- Aim for about {min_rounds}-{max_rounds} rounds overall.
- If needed, add small practical side detail or task-related distraction so the session reaches the desired length.

Complementary-memory rules:
{_session_definition(sample)}

Benchmark target:
- Complementary question: {sample["complementary_question"]}
- Canonical answer: {sample["canonical_answer"]}
- Effective k: {sample["effective_k"]}

Assigned facts for this session:
{_pretty_json(_public_selected_facts(assigned_facts))}

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
- Grouping rationale: {session_plan["grouping_rationale"]}
- Fact integration plan: {session_plan["fact_integration_plan"]}
- Distinctness note: {session_plan["distinct_from_other_sessions"]}

Critical session rules:
- Follow the approved plan closely.
- Let the chosen conversation type shape the mode of help.
- Let the chosen flow shape progression across turns.
- Keep the concrete event clearly different from the other hidden sessions.
- Facts assigned to other sessions must remain unmentioned.
- If this session has only distractor facts, do not mention the canonical answer or the decisive answer-bearing detail.
- If this is a k>1 sample and this session does not contain all required facts, do not state the full final benchmark answer.
- Do not let the user ask the benchmark question verbatim as the main task.
- Do not turn the dialogue into a clean chart, explicit inventory, or one-shot recap of all assigned facts.
- Spread the assigned facts naturally across the dialogue.
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
    musique_guidance = ""
    if _is_musique(sample):
        musique_guidance = """
MuSiQue-specific question guidance:
- MuSiQue source chains often tempt the model into unnatural riddle phrasing. Avoid that.
- Prefer a normal user follow-up that refers to earlier discussed anchors in plain language.
- It is acceptable to use concrete names or short labels that appeared naturally in the sessions.
- It is better to mention a natural anchor by name than to disguise it as "the city where...", "the building by...", "the three-letter system...", or another puzzle clue.
- Do not try to preserve every original MuSiQue hop in the surface wording. Preserve the benchmark difficulty through retrieved session context and the final reasoning relation.
- Do not write nested clauses like "the X that Y that Z" or "the three-letter thing connected to the game named after...".
- Do not hide clear references behind awkward phrases such as "the three-letter system", "the companion building", or "the river city" when a natural user would use the name or a simple previous-reference phrase.
- Bad style: "the city where the company is headquartered", "the system that the 1989 game named after that league came out on", "the river through that project's city".
- Better style: "Zhengzhou", "the NES", "Minneapolis", "Riverside Plaza", or "the river example from the geography note", if those anchors appeared naturally in sessions.
- A good MuSiQue question should usually be one sentence, with at most one dependency phrase before the actual ask.
- Preserve multi-hop reasoning through what the user must recall from sessions, not through an over-engineered riddle in the question wording.
"""
    return f"""
Write one new follow-up question based on the sessions below.

Requirements:
- Produce exactly one question.
- The question must be answerable from the stored sessions alone.
- It should sound like a natural later follow-up from the same user.
- It may stay close to the benchmark-ready complementary question below, or use a light paraphrase.
- Keep the question self-contained and natural.
- Do not turn it into a meta-question about memory, sessions, or process.
- For k>1 specifically:
  - preserve the hidden dependency structure instead of restating every hop in one sentence.
  - do not enumerate all intermediate facts, titles, locations, or aliases verbatim.
  - do not add appositive shortcut clauses such as "X, the Y that..." or pile up multiple explicit clue expansions.
  - do not let one single clue in the question uniquely identify the answer-bearing fact by itself.
  - use light anaphora or natural reference to earlier discussion when possible, but keep the question answerable.
  - before finalizing, check that the question still genuinely depends on all required facts implied by effective_k.
{musique_guidance}
{_quality_feedback_block(sample, "question")}

Subtype reminder:
{_question_definition(sample)}

Benchmark-ready complementary question:
{sample["complementary_question"]}

Canonical answer:
{sample["canonical_answer"]}

Selected complementary facts:
{_pretty_json(_public_selected_facts(sample["selected_complementary_facts"]))}

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
- Every correct answer must mean the same thing as the canonical answer.
- If the canonical answer is structured, preserve all entities, pairings, and key details completely.
- Keep all six answers plausible and reasonably similar in style and length.
- Do not repeat the same wording across candidates.
- If the canonical answer is short, vary the surface form naturally instead of repeating the bare token three times.
- For complementary k=1 specifically, avoid generic wrong alternatives like arbitrary other weekdays or arbitrary other names when the sessions provide more local distractors. Prefer distractor-grounded confusions.
{_quality_feedback_block(sample, "answer")}

Subtype reminder:
{_answer_definition(sample)}

Complementary benchmark bundle:
- Benchmark question: {sample["complementary_question"]}
- Canonical answer: {sample["canonical_answer"]}
- Effective k: {sample["effective_k"]}
- Selected facts:
{_pretty_json(_public_selected_facts(sample["selected_complementary_facts"]))}

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
