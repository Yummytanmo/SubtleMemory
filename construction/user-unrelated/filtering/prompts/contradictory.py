from __future__ import annotations

import json
from typing import Any


CONVERSATION_FILTER_SYSTEM_PROMPT = """You are judging one contradictory-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and naturalness requirements.
Think step by step.
"""


QUESTION_FILTER_SYSTEM_PROMPT = """You are judging the question quality of one contradictory-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and question-design requirements.
Think step by step.
"""


ANSWER_FILTER_SYSTEM_PROMPT = """You are judging the answer quality of one contradictory-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and answer-design requirements.
Think step by step.
"""


def _pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)


def _filter_visible_conflicting_facts(sample: dict[str, Any]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for fact in sample.get("selected_conflicting_facts", []):
        visible.append(
            {
                "session_id": fact.get("session_id"),
                "memory_id": fact.get("memory_id"),
                "answer_text": fact.get("answer_text"),
                "answer_aliases": fact.get("answer_aliases", []),
                "fact_source_role": fact.get("fact_source_role"),
            }
        )
    return visible


def _background() -> str:
    return (
        "Project background:\n"
        "- This benchmark tests fine-grained fact discrimination, correct retrieval, and reasoning over similar memories.\n"
        "- This sample belongs to contradictory memory, where remembered facts under the same apparent condition cannot all be true at once."
    )


def _subtype_definition(sample: dict[str, Any]) -> str:
    subtype = sample["contradictory_subtype"]
    if subtype == "a_user_vs_user":
        return (
            "Contradictory subtype: user fact vs user fact.\n"
            "- The conflicting facts should come from user-side memories across different sessions."
        )
    if subtype == "b_user_vs_non_user":
        return (
            "Contradictory subtype: user fact vs non-user fact.\n"
            "- One side should come from the user and another from the assistant or assistant-provided source."
        )
    if subtype == "c_non_user_vs_non_user":
        return (
            "Contradictory subtype: non-user fact vs non-user fact.\n"
            "- The conflicting facts should be on the assistant side across different sessions."
        )
    raise ValueError(f"Unknown contradictory subtype: {subtype}")


def build_conversation_filter_prompt(sample: dict[str, Any]) -> str:
    return f"""
{_background()}

{_subtype_definition(sample)}

Conversation-stage judging requirements:
- Judge only the sessions and how the contradictory facts are embedded.
- Do not judge the final question or answer candidates here.
- The sessions should feel independent and ordinary on their own.
- The contradiction should emerge across sessions, not through explicit debate inside one session.
- The sessions should not openly announce the contradiction, choose a side, or resolve it.
- The facts should remain under the same apparent condition; this should not secretly become temporal or contextual nuance.
- Say "no" if the sessions are unnatural, if they explicitly argue about the contradiction, or if the contradiction is actually just different times/versions/contexts.

Canonical conflict question:
{sample["canonical_conflict_question"]}

Filter-visible conflicting facts:
{_pretty_json(_filter_visible_conflicting_facts(sample))}

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
    return f"""
{_background()}

{_subtype_definition(sample)}

Question-stage judging requirements:
- Judge only the final benchmark question, with the sessions visible for context.
- Do not judge the answer candidates here.
- The question should directly ask about the disputed fact.
- It should not quietly add qualifiers that resolve the contradiction.
- It should not turn into a meta-question about memory, disagreement, or process.
- Say "no" if the question resolves the contradiction by adding time/version/context qualifiers, or if it fails to target the disputed point.

Canonical conflict question:
{sample["canonical_conflict_question"]}

Filter-visible conflicting facts:
{_pretty_json(_filter_visible_conflicting_facts(sample))}

Sessions:
{_pretty_json(sample["sessions"])}

Question:
{sample["question"]}

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
    return f"""
{_background()}

{_subtype_definition(sample)}

Answer-stage judging requirements:
- Judge only the answer candidates.
- All correct answers must explicitly acknowledge the contradiction, uncertainty, or need for clarification/verification.
- Correct answers must not choose one side as final.
- Incorrect answers are auxiliary distractor metadata, not the main benchmark target.
- Do not reject solely because incorrect answers are easy, weak, repetitive, or not the best possible failure modes.
- Still reject if an incorrect answer is actually a valid contradiction-aware response, or if it makes the intended correct behavior ambiguous.
- Say "no" only if any correct answer improperly picks a side, fails to acknowledge the contradiction/uncertainty, or if any incorrect answer is actually valid/equivalent to a correct answer.

Filter-visible conflicting facts:
{_pretty_json(_filter_visible_conflicting_facts(sample))}

Question:
{sample["question"]}

Correct answers:
{_pretty_json(sample["correct_answers"])}

Incorrect answers:
{_pretty_json(sample["incorrect_answers"])}

Return JSON in this format:
```json
{{
  "decision": "yes or no",
  "reason": "short concrete reason"
}}
```
Think step by step.
""".strip()
