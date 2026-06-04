from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKFLOW_ROOT = Path(__file__).resolve().parent
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from infra.cli_commands import (
    run_build_passed_dataset,
    run_filter,
    run_generate_to_filter,
    run_rebuild_filter_summary,
    run_refilter_answer_failures,
    run_reclassify_ambiguous_context,
    run_repair,
)
from infra.path_registry import outputs_selected_root, outputs_to_filter_root
from ingestion.reclassification import SOURCE_PASSED, TARGET_TO_FILTER
from ingestion.source_registry import CATEGORY_CHOICES, config_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct user-unrelated SubtleMemory data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    categories_help = f"Comma-separated subset of categories to run: {', '.join(CATEGORY_CHOICES)}, or all."

    generate_parser = subparsers.add_parser(
        "generate-to-filter",
        help="Generate benchmark samples into outputs_to_filter.",
    )
    generate_parser.add_argument("--count", type=positive_int, default=10, help="Number of samples to generate per subtype.")
    generate_parser.add_argument("--max-workers", type=positive_int, default=4, help="Max worker threads per subtype batch.")
    generate_parser.add_argument("--categories", default="all", help=categories_help)
    generate_parser.add_argument(
        "--output-root",
        type=Path,
        default=outputs_to_filter_root(),
        help="Root directory for generated samples before filtering.",
    )

    filter_parser = subparsers.add_parser("filter", help="Run staged LLM-based filtering over generated samples.")
    filter_parser.add_argument(
        "--config",
        type=Path,
        default=config_path("filter_selection"),
        help="Path to filter config JSON.",
    )
    filter_parser.add_argument(
        "--input-root",
        type=Path,
        default=outputs_to_filter_root(),
        help="Root directory containing generated samples to filter.",
    )
    filter_parser.add_argument(
        "--output-root",
        type=Path,
        default=outputs_selected_root(),
        help="Root directory where passed and failed filtered samples will be written.",
    )
    filter_parser.add_argument("--categories", default="all", help=categories_help)
    filter_parser.add_argument(
        "--files",
        default="all",
        help="Comma-separated JSON filenames to filter within the selected categories, or all.",
    )
    filter_parser.add_argument("--max-workers", type=positive_int, default=4, help="Max worker threads per input file.")

    repair_parser = subparsers.add_parser(
        "repair",
        help="Conditionally regenerate failed records and rerun filtering.",
    )
    repair_parser.add_argument("--input-root", type=Path, default=outputs_to_filter_root())
    repair_parser.add_argument("--output-root", type=Path, default=outputs_selected_root())
    repair_parser.add_argument("--filter-config", type=Path, default=config_path("filter_selection"))
    repair_parser.add_argument("--categories", default="all", help=categories_help)
    repair_parser.add_argument("--target-rate", type=positive_float, default=0.75)
    repair_parser.add_argument("--max-rounds", type=positive_int, default=2)
    repair_parser.add_argument("--max-workers", type=positive_int, default=3)

    refilter_parser = subparsers.add_parser(
        "refilter-answer-failures",
        help="Re-run only the answer-stage filter for failed records.",
    )
    refilter_parser.add_argument("--config", type=Path, default=config_path("filter_selection"))
    refilter_parser.add_argument("--output-root", type=Path, default=outputs_selected_root())
    refilter_parser.add_argument("--categories", default="all", help=categories_help)
    refilter_parser.add_argument("--max-workers", type=positive_int, default=4)

    rebuild_parser = subparsers.add_parser(
        "rebuild-filter-summary",
        help="Recompute filter_summary.json from passed and failed split directories.",
    )
    rebuild_parser.add_argument("--output-root", type=Path, default=outputs_selected_root())

    passed_parser = subparsers.add_parser(
        "build-passed-dataset",
        help="Assemble normalized passed records and summary files.",
    )
    passed_parser.add_argument("--output-root", type=Path, default=outputs_selected_root())

    reclassify_parser = subparsers.add_parser(
        "reclassify-ambiguous-context",
        help="Move retained ambiguous-context records into the complementary benchmark path.",
    )
    reclassify_parser.add_argument(
        "--input-root",
        type=Path,
        default=SOURCE_PASSED,
        help="Path to the archived nuanced/context passed file to reclassify.",
    )
    reclassify_parser.add_argument(
        "--output-root",
        type=Path,
        default=TARGET_TO_FILTER,
        help="Path to the complementary outputs_to_filter file to write.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "generate-to-filter":
        run_generate_to_filter(
            count=args.count,
            max_workers=args.max_workers,
            categories=args.categories,
            output_root=args.output_root,
        )
        return 0

    if args.command == "filter":
        run_filter(
            config_file=args.config,
            input_root=args.input_root,
            output_root=args.output_root,
            categories=args.categories,
            files=args.files,
            max_workers=args.max_workers,
        )
        return 0

    if args.command == "repair":
        run_repair(
            input_root=args.input_root,
            output_root=args.output_root,
            filter_config_file=args.filter_config,
            categories=args.categories,
            target_rate=args.target_rate,
            max_rounds=args.max_rounds,
            max_workers=args.max_workers,
        )
        return 0

    if args.command == "refilter-answer-failures":
        run_refilter_answer_failures(
            config_file=args.config,
            output_root=args.output_root,
            categories=args.categories,
            max_workers=args.max_workers,
        )
        return 0

    if args.command == "rebuild-filter-summary":
        run_rebuild_filter_summary(output_root=args.output_root)
        return 0

    if args.command == "build-passed-dataset":
        run_build_passed_dataset(output_root=args.output_root)
        return 0

    if args.command == "reclassify-ambiguous-context":
        run_reclassify_ambiguous_context(input_root=args.input_root, output_root=args.output_root)
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
