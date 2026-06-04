from __future__ import annotations

from pathlib import Path
from typing import Any

from infra.json_utils import ensure_dir, load_json, save_json
from infra.path_registry import display_path, outputs_to_filter_root
from ingestion.source_registry import RECLASSIFIED_OUTPUT_FILENAME, archive_path


SOURCE_PASSED = archive_path("ambiguous_context_passed")
TARGET_TO_FILTER = outputs_to_filter_root() / "complementary" / RECLASSIFIED_OUTPUT_FILENAME


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _clean_runtime(record: dict[str, Any]) -> dict[str, Any]:
    clean = dict(record)
    clean.pop("filter_result", None)
    clean.pop("quality_feedback", None)
    return clean


def _context_fact_to_complementary(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": fact.get("memory_id"),
        "source_question": fact.get("context_specific_question"),
        "context_condition": fact.get("context_condition"),
        "context_anchor": fact.get("context_anchor"),
        "answer_aliases": fact.get("answer_aliases") or [],
        "answer_text": fact.get("answer_text"),
        "fact_statement": fact.get("fact_statement"),
    }


def reclassify(record: dict[str, Any]) -> dict[str, Any]:
    clean = _clean_runtime(record)
    facts = clean.get("selected_context_facts") or []
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"Context sample {record.get('sample_id')} has no selected_context_facts.")
    sample_id = str(clean.get("sample_id"))
    converted = {
        "sample_id": f"comp-from-{sample_id}",
        "relationship": "complementary",
        "task_family": "complementary",
        "complementary_major_type": "type1",
        "complementary_subtype": "k_gt_1",
        "complementary_source_subtype": "ambig_context_reclassified",
        "source_dataset": "AmbigContext-Light-Reclassified",
        "complementary_question": clean.get("context_question") or clean.get("question"),
        "canonical_answer": (clean.get("correct_answers") or [{"text": ""}])[0].get("text", ""),
        "effective_k": len(facts),
        "k": len(facts),
        "selected_complementary_facts": [_context_fact_to_complementary(fact) for fact in facts],
        "session_plans": clean.get("session_plans") or [],
        "sessions": clean.get("sessions") or [],
        "question": clean.get("question"),
        "correct_answers": clean.get("correct_answers") or [],
        "incorrect_answers": clean.get("incorrect_answers") or [],
        "metadata": {
            **(clean.get("metadata") or {}),
            "reclassified_from": {
                "category": "nuanced",
                "nuanced_subtype": "context",
                "sample_id": sample_id,
                "reason": "Ambiguous context question requires synthesizing multiple context-conditioned facts, so it is treated as complementary k>1.",
            },
            "selected_context_facts": facts,
            "context_question": clean.get("context_question"),
        },
    }
    return converted


def reclassify_archive_records(source_path: Path = SOURCE_PASSED, target_path: Path = TARGET_TO_FILTER) -> tuple[Path, int]:
    source_records = _load_records(source_path)
    converted = [reclassify(record) for record in source_records]
    ensure_dir(target_path.parent)
    save_json(target_path, converted, indent=2)
    return target_path, len(converted)


def main(source_path: Path = SOURCE_PASSED, target_path: Path = TARGET_TO_FILTER) -> None:
    written_path, count = reclassify_archive_records(source_path, target_path)
    print(f"Saved {count} reclassified samples to {display_path(written_path)}")


if __name__ == "__main__":
    main()
