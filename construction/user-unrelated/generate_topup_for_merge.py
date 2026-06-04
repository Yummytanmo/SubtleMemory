from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any


WORKFLOW_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_ROOT.parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_remaining_for_50 as generator
from construction.merge_user_related_unrelated import load_related_run, relation_type_target_counts
from infra.json_utils import ensure_dir, save_json
from infra.path_registry import display_path


RELATED_RUN_DIR = Path("data/user-related/related-10-diverse-personas")
QA_MODE = "task"
EXPECTED_PASS_RATE = 0.60
PASSED_MARGIN_BY_RELATION = {
    "nuanced": 0,
    "contradictory": 0,
    "complementary": 20,
}
MANIFEST_PATH = Path("data/user-unrelated/runs/remaining-for-50/topup_for_merge_manifest.json")
TOPUP_COMPLEMENTARY_FILENAMES = {
    "qacc_any_one": "topup_for_merge_qacc_any_one.json",
    "fanoutqa_kgt1": "topup_for_merge_fanoutqa_kgt1.json",
    "musique_kgt1": "topup_for_merge_musique_kgt1.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append top-up user-unrelated candidates needed by the related/unrelated merge."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the top-up plan and write the manifest without calling generation APIs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing_ids, existing_relation_counts = generator.load_existing_state()
    target_passed = target_counts_for_related_run()
    target_with_margin = {
        relation_type: target_passed[relation_type] + PASSED_MARGIN_BY_RELATION.get(relation_type, 0)
        for relation_type in generator.RELATION_TYPE_BALANCE_ORDER
    }
    missing_passed = {
        relation_type: max(0, target_with_margin[relation_type] - existing_relation_counts.get(relation_type, 0))
        for relation_type in generator.RELATION_TYPE_BALANCE_ORDER
    }
    raw_targets = {
        relation_type: math.ceil(missing_passed[relation_type] / EXPECTED_PASS_RATE)
        for relation_type in generator.RELATION_TYPE_BALANCE_ORDER
    }

    reports: list[dict[str, Any]] = []
    if args.dry_run:
        print("Dry run: no generation APIs were called.")
    elif raw_targets["complementary"]:
        reports.extend(generate_complementary_topup(existing_ids, raw_targets["complementary"]))
    if not args.dry_run and raw_targets["contradictory"]:
        reports.extend(generator.generate_contradictory(existing_ids, raw_targets["contradictory"]))
    if not args.dry_run and raw_targets["nuanced"]:
        reports.extend(generator.generate_nuanced(existing_ids, raw_targets["nuanced"]))

    ensure_dir(MANIFEST_PATH.parent)
    save_json(
        MANIFEST_PATH,
        {
            "description": "Top-up user-unrelated candidates for the actual related merge run.",
            "related_run_dir": str(RELATED_RUN_DIR),
            "qa_mode": QA_MODE,
            "expected_pass_rate": EXPECTED_PASS_RATE,
            "passed_margin_by_relation": PASSED_MARGIN_BY_RELATION,
            "dry_run": args.dry_run,
            "existing_passed_by_relation": existing_relation_counts,
            "target_passed_by_relation": target_passed,
            "target_passed_with_margin_by_relation": target_with_margin,
            "missing_passed_by_relation": missing_passed,
            "raw_generation_targets_by_relation": raw_targets,
            "output_root": str(generator.OUTPUT_ROOT),
            "topup_input_files_by_relation": {
                "complementary": [
                    str(generator.OUTPUT_ROOT / "complementary" / filename)
                    for filename in TOPUP_COMPLEMENTARY_FILENAMES.values()
                ]
            },
            "reports": reports,
        },
        indent=2,
    )
    print(f"Saved top-up manifest to {display_path(MANIFEST_PATH)}")
    if not args.dry_run and sum(raw_targets.values()) > 0:
        generated_new = sum(int(report.get("generated_new") or 0) for report in reports)
        output_records = sum(int(report.get("output_records") or 0) for report in reports)
        if generated_new == 0 and output_records == 0:
            print("No top-up samples were generated; check the configured LLM endpoint before filtering.", file=sys.stderr)
            return 1
    return 0


def generate_complementary_topup(existing_ids: set[str], raw_target: int) -> list[dict[str, Any]]:
    config = generator.load_config(generator.config_path("complementary_generation"))
    reports: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for source, _ in generator.COMPLEMENTARY_BATCH_SPECS:
        pipeline = generator.ComplementaryPipeline(generator.data_root(), config)
        samples = pipeline.load_samples([source])
        candidates = [sample for sample in samples if sample["sample_id"] not in existing_ids]
        groups.append((source, TOPUP_COMPLEMENTARY_FILENAMES[source], candidates))

    allocations = generator.allocate_counts(raw_target, [len(group[2]) for group in groups])
    for (source, filename, candidates), allocation in zip(groups, allocations):
        output_path = generator.OUTPUT_ROOT / "complementary" / filename

        def make_pipeline() -> Any:
            return generator.ComplementaryPipeline(generator.data_root(), config)

        reports.append(
            generator.generate_file(
                category="complementary",
                label=f"complementary-topup:{source}",
                output_path=output_path,
                candidates=candidates[:allocation],
                make_pipeline=make_pipeline,
            )
        )
    return reports


def target_counts_for_related_run() -> dict[str, int]:
    personas = load_related_run(RELATED_RUN_DIR, qa_mode=QA_MODE)
    targets = {relation_type: 0 for relation_type in generator.RELATION_TYPE_BALANCE_ORDER}
    for persona in personas.values():
        per_persona = relation_type_target_counts(len(persona.related_bench))
        for relation_type, count in per_persona.items():
            targets[relation_type] += count
    return targets


if __name__ == "__main__":
    raise SystemExit(main())
