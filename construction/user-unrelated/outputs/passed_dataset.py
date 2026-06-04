from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from infra.json_utils import ensure_dir, load_json, save_json
from infra.path_registry import display_path, outputs_selected_root


def _first_present(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def _subtype(record: dict[str, Any], category: str) -> str:
    if category == "complementary":
        return str(record.get("complementary_subtype") or record.get("complementary_major_type") or "")
    if category == "contradictory":
        return str(record.get("contradictory_subtype") or "")
    if category == "nuanced":
        return str(record.get("nuanced_subtype") or "")
    return ""


def _category_details(record: dict[str, Any], category: str) -> dict[str, Any]:
    if category == "complementary":
        return {
            "complementary_major_type": record.get("complementary_major_type"),
            "complementary_subtype": record.get("complementary_subtype"),
            "complementary_source_subtype": record.get("complementary_source_subtype"),
            "k": record.get("k"),
            "effective_k": record.get("effective_k"),
        }
    if category == "contradictory":
        return {
            "contradictory_subtype": record.get("contradictory_subtype"),
            "canonical_conflict_question": record.get("canonical_conflict_question"),
        }
    if category == "nuanced":
        return {
            "nuanced_subtype": record.get("nuanced_subtype"),
            "temporal_question": record.get("temporal_question"),
            "context_question": record.get("context_question"),
        }
    return {}


def _memory_facts(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = _first_present(
        record,
        [
            "selected_complementary_facts",
            "selected_conflicting_facts",
            "selected_temporal_facts",
            "selected_context_facts",
        ],
        [],
    )
    return value if isinstance(value, list) else []


def _qa_pairs(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_pairs = record.get("qa_pairs")
    if isinstance(raw_pairs, list) and raw_pairs:
        pairs = raw_pairs
    else:
        pairs = [
            {
                "qa_id": f"{record.get('sample_id', 'sample')}-q01",
                "difficulty": None,
                "temporal_anchor_type": None,
                "target_memory_id": None,
                "question": record.get("question"),
                "correct_answers": record.get("correct_answers"),
                "incorrect_answers": record.get("incorrect_answers"),
            }
        ]

    normalized: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            continue
        normalized.append(
            {
                "qa_id": pair.get("qa_id") or f"{record.get('sample_id', 'sample')}-q{idx:02d}",
                "difficulty": pair.get("difficulty"),
                "temporal_anchor_type": pair.get("temporal_anchor_type"),
                "context_anchor": pair.get("context_anchor"),
                "target_memory_id": pair.get("target_memory_id"),
                "question": pair.get("question"),
                "correct_answers": pair.get("correct_answers") or [],
                "incorrect_answers": pair.get("incorrect_answers") or [],
            }
        )
    return normalized


def normalize_record(record: dict[str, Any], category: str, source_file: Path) -> dict[str, Any]:
    original_sample_id = str(record.get("sample_id") or "")
    source_file_rel = display_path(source_file)
    subtype = _subtype(record, category)
    return {
        "global_sample_id": f"{category}:{source_file.name}:{original_sample_id}",
        "sample_id": original_sample_id,
        "category": category,
        "relationship": record.get("relationship"),
        "task_family": record.get("task_family"),
        "subtype": subtype,
        "source_dataset": record.get("source_dataset"),
        "source_file": source_file_rel,
        "category_details": _category_details(record, category),
        "session_plans": record.get("session_plans") or [],
        "sessions": record.get("sessions") or [],
        "memory_facts": _memory_facts(record),
        "qa_pairs": _qa_pairs(record),
        "metadata": record.get("metadata") or {},
        "filter_result": record.get("filter_result") or {},
    }


def build_dataset(passed_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_files: list[str] = []
    for category_dir in sorted(passed_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        category_passed_dir = category_dir / "passed"
        if not category_passed_dir.exists():
            continue
        for path in sorted(category_passed_dir.glob("*.json")):
            data = load_json(path)
            if not isinstance(data, list):
                continue
            source_files.append(display_path(path))
            for record in data:
                if isinstance(record, dict):
                    records.append(normalize_record(record, category, path))

    by_category = Counter(record["category"] for record in records)
    by_subtype = Counter(f"{record['category']}::{record['subtype']}" for record in records)
    summary = {
        "total_records": len(records),
        "by_category": dict(sorted(by_category.items())),
        "by_category_subtype": dict(sorted(by_subtype.items())),
        "source_files": source_files,
    }
    return records, summary


def save_dataset_files(output_root: Path | None = None) -> tuple[Path, Path]:
    passed_root = output_root or outputs_selected_root()
    output_path = passed_root / "all_passed_samples.json"
    summary_path = passed_root / "all_passed_samples_summary.json"
    ensure_dir(output_path.parent)
    records, summary = build_dataset(passed_root)
    save_json(output_path, records, indent=2)
    save_json(summary_path, summary, indent=2)
    return output_path, summary_path


def main(output_root: Path | None = None) -> None:
    output_path, summary_path = save_dataset_files(output_root)
    records = load_json(output_path)
    print(f"Saved {len(records)} normalized passed samples to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
