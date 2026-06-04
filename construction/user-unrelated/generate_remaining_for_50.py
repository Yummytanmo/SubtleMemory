from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from typing import Any, Callable

from generation.complementary_pipeline import ComplementaryPipeline
from generation.contradictory_pipeline import ContradictoryPipeline
from generation.nuanced_context_pipeline import NuancedContextPipeline
from generation.nuanced_temporal_pipeline import NuancedTemporalPipeline
from infra.json_utils import ensure_dir, load_config, load_json, save_json
from infra.path_registry import data_root, display_path
from ingestion.source_registry import (
    COMPLEMENTARY_BATCH_SPECS,
    CONTRADICTORY_BATCH_SPECS,
    config_path,
)


OUTPUT_ROOT = Path("data/user-unrelated/runs/remaining-for-50/outputs_to_filter")
OUTPUT_SELECTED_ROOT = Path("data/user-unrelated/runs/remaining-for-50/outputs_selected")
EXISTING_PASSED = Path("data/user-unrelated/all_passed_samples.json")
MANIFEST_PATH = Path("data/user-unrelated/runs/remaining-for-50/generation_manifest.json")
MAX_WORKERS = 32
TARGET_RELATED_DATA_COUNT = 10
CASES_PER_RELATED_DATA = 50
EXPECTED_PASS_RATE = 0.60
RELATION_TYPE_BALANCE_ORDER = ("nuanced", "contradictory", "complementary")


def main() -> int:
    existing_ids, existing_relation_counts = load_existing_state()
    target_passed, missing_passed, raw_targets = raw_generation_targets(existing_relation_counts)
    reports: list[dict[str, Any]] = []
    ensure_dir(OUTPUT_ROOT)

    reports.extend(generate_complementary(existing_ids, raw_targets["complementary"]))
    reports.extend(generate_contradictory(existing_ids, raw_targets["contradictory"]))
    reports.extend(generate_nuanced(existing_ids, raw_targets["nuanced"]))

    ensure_dir(MANIFEST_PATH.parent)
    save_json(
        MANIFEST_PATH,
        {
            "description": "Remaining user-unrelated candidates for 50-case merge planning.",
            "existing_passed_file": str(EXISTING_PASSED),
            "current_run_selected_root": str(OUTPUT_SELECTED_ROOT),
            "excluded_existing_or_generated_ids": len(existing_ids),
            "target_related_data_count": TARGET_RELATED_DATA_COUNT,
            "cases_per_related_data": CASES_PER_RELATED_DATA,
            "expected_pass_rate_for_generation_sizing": EXPECTED_PASS_RATE,
            "target_passed_by_relation": target_passed,
            "existing_passed_by_relation": existing_relation_counts,
            "missing_passed_by_relation": missing_passed,
            "raw_generation_targets_by_relation": raw_targets,
            "output_root": str(OUTPUT_ROOT),
            "reports": reports,
        },
        indent=2,
    )
    print(f"Saved generation manifest to {display_path(MANIFEST_PATH)}")
    return 0


def load_existing_state() -> tuple[set[str], dict[str, int]]:
    ids, relation_counts = load_existing_passed(EXISTING_PASSED)
    selected_ids, selected_counts = load_current_selected_state(OUTPUT_SELECTED_ROOT)
    generated_ids = load_generated_ids(OUTPUT_ROOT)
    ids.update(selected_ids)
    ids.update(generated_ids)
    for relation_type, count in selected_counts.items():
        relation_counts[relation_type] = relation_counts.get(relation_type, 0) + count
    return ids, relation_counts


def load_existing_passed(path: Path) -> tuple[set[str], dict[str, int]]:
    if not path.exists():
        return set(), {}
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    ids: set[str] = set()
    relation_counts: dict[str, int] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("sample_id") or "").strip()
        if sample_id:
            ids.add(sample_id)
        relation_type = str(item.get("category") or item.get("relationship") or "").strip()
        if relation_type:
            relation_counts[relation_type] = relation_counts.get(relation_type, 0) + 1
    return ids, relation_counts


def load_current_selected_state(output_selected_root: Path) -> tuple[set[str], dict[str, int]]:
    ids: set[str] = set()
    relation_counts: dict[str, int] = {}
    if not output_selected_root.exists():
        return ids, relation_counts
    for category_dir in sorted(path for path in output_selected_root.iterdir() if path.is_dir()):
        category = category_dir.name
        for split in ("passed", "failed"):
            split_dir = category_dir / split
            if not split_dir.exists():
                continue
            for path in sorted(split_dir.glob("*.json")):
                records = load_json_list(path)
                ids.update(record_sample_ids(records))
                if split == "passed":
                    relation_counts[category] = relation_counts.get(category, 0) + len(records)
    return ids, relation_counts


def load_generated_ids(output_root: Path) -> set[str]:
    ids: set[str] = set()
    if not output_root.exists():
        return ids
    for path in sorted(output_root.glob("*/*.json")):
        ids.update(record_sample_ids(load_json_list(path)))
    return ids


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in payload if isinstance(item, dict)]


def record_sample_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record.get("sample_id") or "").strip() for record in records if str(record.get("sample_id") or "").strip()}


def raw_generation_targets(existing_relation_counts: dict[str, int]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    per_related_data = relation_type_target_counts(CASES_PER_RELATED_DATA)
    target_passed = {
        relation_type: count * TARGET_RELATED_DATA_COUNT
        for relation_type, count in per_related_data.items()
    }
    missing_passed = {
        relation_type: max(0, target_passed[relation_type] - existing_relation_counts.get(relation_type, 0))
        for relation_type in RELATION_TYPE_BALANCE_ORDER
    }
    raw_targets = {
        relation_type: math.ceil(missing_passed[relation_type] / EXPECTED_PASS_RATE)
        for relation_type in RELATION_TYPE_BALANCE_ORDER
    }
    return target_passed, missing_passed, raw_targets


def relation_type_target_counts(total_cases: int) -> dict[str, int]:
    base, remainder = divmod(total_cases, len(RELATION_TYPE_BALANCE_ORDER))
    counts = {relation_type: base for relation_type in RELATION_TYPE_BALANCE_ORDER}
    for relation_type in RELATION_TYPE_BALANCE_ORDER[:remainder]:
        counts[relation_type] += 1
    return counts


def allocate_counts(total: int, group_sizes: list[int]) -> list[int]:
    allocations = [0 for _ in group_sizes]
    remaining = total
    while remaining > 0:
        progressed = False
        for index, group_size in enumerate(group_sizes):
            if allocations[index] >= group_size:
                continue
            allocations[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return allocations


def generate_complementary(existing_ids: set[str], raw_target: int) -> list[dict[str, Any]]:
    config = load_config(config_path("complementary_generation"))
    reports: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]], Callable[[], ComplementaryPipeline]]] = []
    for source, filename in COMPLEMENTARY_BATCH_SPECS:
        def make_pipeline() -> ComplementaryPipeline:
            return ComplementaryPipeline(data_root(), config)

        samples = make_pipeline().load_samples([source])
        candidates = [sample for sample in samples if sample["sample_id"] not in existing_ids]
        groups.append((source, filename, candidates, make_pipeline))

    allocations = allocate_counts(raw_target, [len(group[2]) for group in groups])
    for (source, filename, candidates, make_pipeline), allocation in zip(groups, allocations):
        output_path = OUTPUT_ROOT / "complementary" / filename
        reports.append(
            generate_file(
                category="complementary",
                label=f"complementary:{source}",
                output_path=output_path,
                candidates=candidates[:allocation],
                make_pipeline=make_pipeline,
            )
        )
    return reports


def generate_contradictory(existing_ids: set[str], raw_target: int) -> list[dict[str, Any]]:
    config = load_config(config_path("contradictory_generation"))
    reports: list[dict[str, Any]] = []
    base_pipeline = ContradictoryPipeline(data_root(), config)
    base_samples = base_pipeline.load_samples(["contradictory_source"])
    groups: list[tuple[str, str, list[dict[str, Any]], Callable[[], ContradictoryPipeline]]] = []

    for subtype, filename in CONTRADICTORY_BATCH_SPECS:
        def make_pipeline() -> ContradictoryPipeline:
            return ContradictoryPipeline(data_root(), config)

        candidates = [
            base_pipeline.materialize_subtype_sample(sample, subtype)
            for sample in base_samples
        ]
        candidates = [sample for sample in candidates if sample["sample_id"] not in existing_ids]
        groups.append((subtype, filename, candidates, make_pipeline))

    allocations = allocate_counts(raw_target, [len(group[2]) for group in groups])
    for (subtype, filename, candidates, make_pipeline), allocation in zip(groups, allocations):
        output_path = OUTPUT_ROOT / "contradictory" / filename
        reports.append(
            generate_file(
                category="contradictory",
                label=f"contradictory:{subtype}",
                output_path=output_path,
                candidates=candidates[:allocation],
                make_pipeline=make_pipeline,
            )
        )
    return reports


def generate_nuanced(existing_ids: set[str], raw_target: int) -> list[dict[str, Any]]:
    temporal_config = load_config(config_path("nuanced_temporal_generation"))
    context_config = load_config(config_path("nuanced_context_generation"))

    def make_temporal_pipeline() -> NuancedTemporalPipeline:
        return NuancedTemporalPipeline(data_root(), temporal_config)

    temporal_samples = make_temporal_pipeline().load_samples(["temporal_light"])
    temporal_candidates = [
        sample for sample in temporal_samples if sample["sample_id"] not in existing_ids
    ]

    def make_context_pipeline() -> NuancedContextPipeline:
        return NuancedContextPipeline(data_root(), context_config)

    context_samples = make_context_pipeline().load_samples(["context_light"])
    context_candidates = [
        sample for sample in context_samples if sample["sample_id"] not in existing_ids
    ]

    groups: list[tuple[str, Path, list[dict[str, Any]], Callable[[], Any]]] = [
        (
            "nuanced:temporal",
            OUTPUT_ROOT / "nuanced" / f"temporal_hard_{len(temporal_candidates)}_samples.json",
            temporal_candidates,
            make_temporal_pipeline,
        ),
        (
            "nuanced:context",
            OUTPUT_ROOT / "nuanced" / f"context_{len(context_candidates)}_samples.json",
            context_candidates,
            make_context_pipeline,
        ),
    ]
    allocations = allocate_counts(raw_target, [len(group[2]) for group in groups])
    reports: list[dict[str, Any]] = []
    for (label, output_path, candidates, make_pipeline), allocation in zip(groups, allocations):
        reports.append(
            generate_file(
                category="nuanced",
                label=label,
                output_path=output_path,
                candidates=candidates[:allocation],
                make_pipeline=make_pipeline,
            )
        )
    return reports


def generate_file(
    *,
    category: str,
    label: str,
    output_path: Path,
    candidates: list[dict[str, Any]],
    make_pipeline: Callable[[], Any],
) -> dict[str, Any]:
    ensure_dir(output_path.parent)
    existing_records = load_json_list(output_path)
    existing_ids = record_sample_ids(existing_records)
    candidates = [candidate for candidate in candidates if candidate["sample_id"] not in existing_ids]
    print(
        f"Generating {label}: existing={len(existing_records)}, new_candidates={len(candidates)} -> {display_path(output_path)}",
        flush=True,
    )

    ordered: list[dict[str, Any] | None] = [None] * len(candidates)
    errors: list[dict[str, Any]] = []
    jobs = list(enumerate(candidates))
    if jobs:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as executor:
            for position, record, error in executor.map(
                lambda job: generate_one(job, make_pipeline),
                jobs,
            ):
                if record is None:
                    errors.append({"position": position, "error": error})
                    print(f"Skipped {label} candidate {position}: {error}", flush=True)
                else:
                    ordered[position] = record

    new_records = [record for record in ordered if record is not None]
    records = [*existing_records, *new_records]
    save_json(output_path, records, indent=2)
    print(
        f"Saved {len(records)} {label} samples to {display_path(output_path)} "
        f"({len(new_records)} new)",
        flush=True,
    )
    return {
        "category": category,
        "label": label,
        "output_file": display_path(output_path),
        "existing_output_records": len(existing_records),
        "attempted": len(candidates),
        "generated_new": len(new_records),
        "output_records": len(records),
        "failed_generation": len(errors),
        "errors": errors[:20],
    }


def generate_one(
    job: tuple[int, dict[str, Any]],
    make_pipeline: Callable[[], Any],
) -> tuple[int, dict[str, Any] | None, str | None]:
    position, sample = job
    try:
        result = make_pipeline().generate_sample(sample)
        return position, result.record, None
    except Exception as exc:
        return position, None, str(exc)


if __name__ == "__main__":
    raise SystemExit(main())
