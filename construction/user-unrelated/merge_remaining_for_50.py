from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from infra.json_utils import ensure_dir, load_json, save_json


BASE_PASSED = Path("data/user-unrelated/all_passed_samples.json")
NEW_PASSED = Path("data/user-unrelated/runs/remaining-for-50/outputs_selected/all_passed_samples.json")
MERGED_PASSED = Path("data/user-unrelated/all_passed_samples_50_merged.json")
MERGED_SUMMARY = Path("data/user-unrelated/all_passed_samples_50_merged_summary.json")
RELATION_QUOTA_FOR_50 = {
    "nuanced": 17,
    "contradictory": 17,
    "complementary": 16,
}


def main() -> int:
    records = merge_records([BASE_PASSED, NEW_PASSED])
    summary = build_summary(records)
    ensure_dir(MERGED_PASSED.parent)
    save_json(MERGED_PASSED, records, indent=2)
    save_json(MERGED_SUMMARY, summary, indent=2)
    print(f"Saved merged passed pool to {MERGED_PASSED}")
    print(f"Saved merged summary to {MERGED_SUMMARY}")
    print(f"50-case merge capacity: {summary['capacity_for_50_case_related_data']}")
    return 0


def merge_records(paths: list[Path]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list in {path}")
        for item in payload:
            if not isinstance(item, dict):
                continue
            sample_id = str(item.get("sample_id") or item.get("global_sample_id") or "").strip()
            if not sample_id:
                continue
            by_id.setdefault(sample_id, item)
    return list(by_id.values())


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = Counter(str(record.get("category") or record.get("relationship") or "") for record in records)
    by_subtype = Counter(
        f"{record.get('category') or record.get('relationship') or ''}::{record.get('subtype') or ''}"
        for record in records
    )
    capacity_by_relation = {
        relation_type: by_category.get(relation_type, 0) // needed
        for relation_type, needed in RELATION_QUOTA_FOR_50.items()
    }
    capacity = min(capacity_by_relation.values()) if capacity_by_relation else 0
    return {
        "total_records": len(records),
        "by_category": dict(sorted(by_category.items())),
        "by_category_subtype": dict(sorted(by_subtype.items())),
        "quota_per_50_case_related_data": RELATION_QUOTA_FOR_50,
        "capacity_by_relation_for_50_case_related_data": capacity_by_relation,
        "capacity_for_50_case_related_data": capacity,
    }


if __name__ == "__main__":
    raise SystemExit(main())
