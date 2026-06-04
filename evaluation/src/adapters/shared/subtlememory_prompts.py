"""Shared SubtleMemory answer prompt helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.src.utils.config import load_yaml


SUBTLEMEMORY_CONCISE_ANSWER_PROMPT_KEY = "answer_prompt_subtlememory_v1_concise"
SUBTLEMEMORY_CONFLICT_AUDITOR_ANSWER_PROMPT_KEY = (
    "answer_prompt_subtlememory_v2_conflict_auditor"
)
SUBTLEMEMORY_EVIDENCE_CALIBRATED_ANSWER_PROMPT_KEY = (
    "answer_prompt_subtlememory_v3_evidence_calibrated"
)
SUBTLEMEMORY_CASE_CLASSIFIER_ANSWER_PROMPT_KEY = (
    "answer_prompt_subtlememory_v4_case_classifier"
)
SUBTLEMEMORY_FACT_EXTRACTOR_ANSWER_PROMPT_KEY = (
    "answer_prompt_subtlememory_v5_fact_extractor"
)
SUBTLEMEMORY_FACT_ANSWER_PROMPT_KEY = "answer_prompt_subtlememory_v5_fact_answer"
SUBTLEMEMORY_DEFAULT_ANSWER_PROMPT_KEY = SUBTLEMEMORY_CONCISE_ANSWER_PROMPT_KEY

SUBTLEMEMORY_ANSWER_PROMPT_ALIASES = {
    "v1": SUBTLEMEMORY_CONCISE_ANSWER_PROMPT_KEY,
    "v1_concise": SUBTLEMEMORY_CONCISE_ANSWER_PROMPT_KEY,
    "concise": SUBTLEMEMORY_CONCISE_ANSWER_PROMPT_KEY,
    "answer_prompt_subtlememory_concise": SUBTLEMEMORY_CONCISE_ANSWER_PROMPT_KEY,
    "v2": SUBTLEMEMORY_CONFLICT_AUDITOR_ANSWER_PROMPT_KEY,
    "v2_conflict_auditor": SUBTLEMEMORY_CONFLICT_AUDITOR_ANSWER_PROMPT_KEY,
    "unified": SUBTLEMEMORY_CONFLICT_AUDITOR_ANSWER_PROMPT_KEY,
    "balanced": SUBTLEMEMORY_CONFLICT_AUDITOR_ANSWER_PROMPT_KEY,
    "conflict_auditor": SUBTLEMEMORY_CONFLICT_AUDITOR_ANSWER_PROMPT_KEY,
    "answer_prompt_subtlememory_unified": (
        SUBTLEMEMORY_CONFLICT_AUDITOR_ANSWER_PROMPT_KEY
    ),
    "v3": SUBTLEMEMORY_EVIDENCE_CALIBRATED_ANSWER_PROMPT_KEY,
    "v3_evidence_calibrated": SUBTLEMEMORY_EVIDENCE_CALIBRATED_ANSWER_PROMPT_KEY,
    "evidence_calibrated": SUBTLEMEMORY_EVIDENCE_CALIBRATED_ANSWER_PROMPT_KEY,
    "v4": SUBTLEMEMORY_CASE_CLASSIFIER_ANSWER_PROMPT_KEY,
    "v4_case_classifier": SUBTLEMEMORY_CASE_CLASSIFIER_ANSWER_PROMPT_KEY,
    "case_classifier": SUBTLEMEMORY_CASE_CLASSIFIER_ANSWER_PROMPT_KEY,
    "v5_fact_extractor": SUBTLEMEMORY_FACT_EXTRACTOR_ANSWER_PROMPT_KEY,
    "fact_extractor": SUBTLEMEMORY_FACT_EXTRACTOR_ANSWER_PROMPT_KEY,
    "v5_fact_answer": SUBTLEMEMORY_FACT_ANSWER_PROMPT_KEY,
    "fact_answer": SUBTLEMEMORY_FACT_ANSWER_PROMPT_KEY,
}


@lru_cache(maxsize=1)
def _load_prompts() -> Dict[str, Any]:
    evaluation_root = Path(__file__).resolve().parents[3]
    prompts_path = evaluation_root / "config" / "prompts.yaml"
    return load_yaml(str(prompts_path))


def _select_subtlememory_answer_prompt_key(config: Optional[Dict[str, Any]]) -> str:
    if not config:
        return SUBTLEMEMORY_DEFAULT_ANSWER_PROMPT_KEY

    raw_key = (
        config.get("answer", {}).get("subtlememory_prompt_key")
        or config.get("subtlememory_prompt_key")
        or SUBTLEMEMORY_DEFAULT_ANSWER_PROMPT_KEY
    )
    key = str(raw_key).strip()
    return SUBTLEMEMORY_ANSWER_PROMPT_ALIASES.get(key, key)


def get_subtlememory_answer_prompt(
    prompts: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the shared answer prompt for SubtleMemory runs."""
    prompt_config = prompts if prompts is not None else _load_prompts()
    prompt_key = _select_subtlememory_answer_prompt_key(config)
    prompt_map = prompt_config["online_api"]["default"]
    if prompt_key not in prompt_map:
        valid_keys = ", ".join(
            sorted(key for key in prompt_map if key.startswith("answer_prompt_"))
        )
        raise KeyError(
            f"Unknown SubtleMemory answer prompt key '{prompt_key}'. "
            f"Valid prompt keys: {valid_keys}"
        )
    return prompt_map[prompt_key]


def get_subtlememory_prompt_by_key(
    prompt_key: str,
    prompts: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a shared SubtleMemory prompt by raw key or alias."""
    prompt_config = prompts if prompts is not None else _load_prompts()
    key = str(prompt_key or "").strip()
    resolved_key = SUBTLEMEMORY_ANSWER_PROMPT_ALIASES.get(key, key)
    prompt_map = prompt_config["online_api"]["default"]
    if resolved_key not in prompt_map:
        valid_keys = ", ".join(
            sorted(key for key in prompt_map if key.startswith("answer_prompt_"))
        )
        raise KeyError(
            f"Unknown SubtleMemory prompt key '{prompt_key}'. "
            f"Valid prompt keys: {valid_keys}"
        )
    return prompt_map[resolved_key]


def get_subtlememory_unified_answer_prompt(
    prompts: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Backward-compatible wrapper for SubtleMemory answer prompt selection."""
    return get_subtlememory_answer_prompt(prompts=prompts, config=config)
