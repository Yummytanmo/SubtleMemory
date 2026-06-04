from __future__ import annotations

import json
from typing import Any


CONVERSATION_FILTER_SYSTEM_PROMPT = """You are judging one complementary-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and naturalness requirements.
Think step by step.
"""


QUESTION_FILTER_SYSTEM_PROMPT = """You are judging the question quality of one complementary-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and question-design requirements.
Think step by step.
"""


ANSWER_FILTER_SYSTEM_PROMPT = """You are judging the answer quality of one complementary-memory benchmark sample.
Return valid JSON only and follow the requested schema exactly.
Judge strictly against the subtype requirements and answer-design requirements.
Think step by step.
"""


def _pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)


def _is_musique(sample: dict[str, Any]) -> bool:
    source_dataset = str(sample.get("source_dataset", "")).casefold()
    sample_id = str(sample.get("sample_id", "")).casefold()
    return "musique" in source_dataset or "musique" in sample_id


def _is_reclassified_ambig_context(sample: dict[str, Any]) -> bool:
    return sample.get("complementary_source_subtype") == "ambig_context_reclassified"


def _background() -> str:
    return (
        "Project background:\n"
        "- This benchmark is designed to test a memory system's core abilities: fine-grained fact discrimination, correct retrieval, and reasoning over similar memories.\n"
        "- This sample belongs to complementary memory, where multiple memories are related to the same topic or later question, but some of them are background, side details, or alternative phrasings rather than conflicting answers."
    )


def _subtype_definition(sample: dict[str, Any]) -> str:
    subtype = sample["complementary_subtype"]
    if subtype == "k_eq_1":
        return (
            "Complementary subtype: type1, k=1.\n"
            "- Exactly one decisive fact should determine the later answer.\n"
            "- Other selected facts should be related but non-decisive distractors.\n"
            "- The conversation should not expose the decisive fact too directly or as a naked standalone answer."
        )
    if subtype == "k_gt_1":
        return (
            "Complementary subtype: type1, k>1.\n"
            f"- The later question should require integrating multiple required facts, with effective_k={sample['effective_k']}.\n"
            "- The required facts should stay distributed; no single session or one utterance should already hand over the full final answer too neatly.\n"
            "- This should feel like multi-hop reasoning or structured synthesis, not contradiction or context disambiguation."
        )
    if subtype == "any_one_of_n":
        return (
            "Complementary subtype: type2, any-one-of-n.\n"
            "- Multiple facts should redundantly support the same answer.\n"
            "- Any one of those equivalent facts should be enough to answer the later question.\n"
            "- The sample should preserve redundancy naturally, not create contradictions or context-conditioned multiple truths."
        )
    raise ValueError(f"Unknown complementary subtype: {subtype}")


def build_conversation_filter_prompt(sample: dict[str, Any]) -> str:
    reclassified_adjustment = ""
    if _is_reclassified_ambig_context(sample):
        reclassified_adjustment = (
            "\nReclassified ambiguous-context judging adjustment:\n"
            "- This sample was intentionally moved from nuanced/context into complementary k>1 because the final broad question requires synthesizing multiple context-conditioned facts.\n"
            "- Do not reject merely because the memories are context-conditioned; for this reclassified subtype, the complementary task is to recover the complete set of relevant context-answer pairs.\n"
            "- The conversation passes if the context-specific facts are naturally distributed and no single turn gives the full final answer too neatly.\n"
        )
    return f"""
{_background()}

{_subtype_definition(sample)}

Conversation-stage judging requirements:
- Judge only the conversation quality. Do not judge the final benchmark question or answer candidates here.
- The multi-session conversations should feel like ordinary user-assistant interactions, not quiz reveals or benchmark templates.
- The source facts should be integrated implicitly and naturally.
- The sessions should not sound overly synthetic, overly repetitive, or overly explicit about the hidden target.
- It is fine if different sessions use different task scenarios; do not reject merely because you only see one sample and cannot compare against the whole dataset.
- Say "no" if the conversation clearly fails the subtype requirement, leaks the target too directly, sounds obviously unnatural, or turns into an explicit fact dump.
{reclassified_adjustment}

Internal benchmark scaffold:
- Complementary question: {sample["complementary_question"]}
- Canonical answer: {sample["canonical_answer"]}
- Effective k: {sample["effective_k"]}
- Selected complementary facts:
{_pretty_json(sample["selected_complementary_facts"])}

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
    musique_adjustment = ""
    if _is_musique(sample):
        musique_adjustment = (
            "\nMuSiQue-specific judging adjustment:\n"
            "- MuSiQue questions may naturally use concrete names, short labels, or earlier-discussion anchors from the sessions.\n"
            "- MuSiQue can legitimately use long multi-hop chains. Do not reject merely because the chain is long, preserves several hops, or requires following multiple memories.\n"
            "- Do not mechanically enforce the original effective_k count for MuSiQue. Source chains can contain redundant, subsumed, or bridge hops.\n"
            "- Do not reject solely because the question names an intermediate entity or river to avoid riddle phrasing.\n"
            "- Accept if the question is understandable and still requires a meaningful memory-grounded relation, comparison, or cross-session synthesis.\n"
            "- A MuSiQue question may pass even if it only tests the useful subset of the selected chain, as long as it is not a trivial standalone fact lookup disconnected from the remembered sessions.\n"
            "- Reject only when the wording becomes genuinely unreadable/artificial, uses avoidable awkward hidden references where a normal anchor is available, or is answerable from a single obvious fact disconnected from the memory setup.\n"
        )
    question_reject_rule = (
        "- Say \"no\" if the question is too direct, mismatched to the subtype, answerable in the wrong way, or not natural."
    )
    reclassified_adjustment = ""
    if _is_reclassified_ambig_context(sample):
        reclassified_adjustment = (
            "\nReclassified ambiguous-context judging adjustment:\n"
            "- This sample is complementary k>1 because the broad question requires the answerer to retrieve and synthesize multiple context-conditioned facts.\n"
            "- Do not reject because the question is ambiguous across contexts; that ambiguity is the reason it was reclassified as complementary rather than nuanced/context.\n"
            "- Accept if the question naturally asks the broad topic and the only high-quality answer must include all remembered context-answer pairs or a clarification request that enumerates them completely.\n"
            "- Reject only if the question clearly targets one single context, is answerable from one memory item, contains the answer, or is too vague to identify the broad topic.\n"
        )
        question_reject_rule = (
            "- Say \"no\" only if the question targets one single context, is answerable from one memory item, "
            "contains the answer, is too vague to identify the broad topic, or is genuinely unnatural."
        )
    return f"""
{_background()}

{_subtype_definition(sample)}

Question-stage judging requirements:
- Judge only the final benchmark question, with access to the sessions for context.
- Do not judge the answer candidates here.
- The question should sound like a natural later follow-up, not a benchmark annotation.
- It should fit the subtype:
  - k=1: answer should depend on the one decisive fact, not on distractors.
  - k>1: answer should genuinely require integrating multiple required facts.
  - any-one-of-n: answer should target the shared fact recoverable from any equivalent mention.
{question_reject_rule}
{musique_adjustment}
{reclassified_adjustment}

Complementary scaffold:
- Complementary question: {sample["complementary_question"]}
- Canonical answer: {sample["canonical_answer"]}
- Effective k: {sample["effective_k"]}
- Selected facts:
{_pretty_json(sample["selected_complementary_facts"])}

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
    answer_reject_rule = (
        "- Say \"no\" only if a correct answer is wrong or incomplete, an incorrect answer is actually correct "
        "or semantically equivalent to the canonical answer, the answer set makes the intended answer ambiguous, "
        "or the correct answers do not match the subtype."
    )
    reclassified_adjustment = ""
    if _is_reclassified_ambig_context(sample):
        reclassified_adjustment = (
            "\nReclassified ambiguous-context judging adjustment:\n"
            "- Correct answers should completely cover all remembered context-answer pairs for the broad question.\n"
            "- A correct answer may either give all context-conditioned answers directly, or ask which context the user means while explicitly enumerating the full set of available contexts and answers.\n"
            "- Do not reject merely because the answer uses 'it depends' or clarification wording, as long as it fully enumerates the contexts and answers.\n"
            "- Reject if a correct answer gives only one context, omits a context, mixes up context-answer pairings, or treats the broad question as having one unconditional answer.\n"
        )
    return f"""
{_background()}

{_subtype_definition(sample)}

Answer-stage judging requirements:
- Judge only the answer candidates.
- The three correct answers should all be genuinely correct for the question and semantically aligned with the intended canonical answer.
- Incorrect answers are auxiliary distractor metadata, not the main benchmark target.
- Do not reject solely because incorrect answers are easy, weak, not very plausible, not distractor-grounded, or grammatically mismatched.
- Still reject if an incorrect answer is actually correct, semantically equivalent to a correct answer, or makes the intended answer ambiguous.
- The answers should match the subtype:
  - k=1: correct answers should return the one decisive fact.
  - k>1: correct answers should preserve the integrated multi-fact result.
  - any-one-of-n: correct answers should target the shared answer.
{answer_reject_rule}
{reclassified_adjustment}

Canonical answer:
{sample["canonical_answer"]}

Selected facts:
{_pretty_json(sample["selected_complementary_facts"])}

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
