"""CLI for validating real smoke and integration run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_PATH = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_PATH):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from evaluation.src.run_artifacts.validation import (  # noqa: E402
    render_validation_report,
    validate_run_artifacts,
)


def main() -> int:
    """Validate a run directory and exit non-zero on failure."""
    parser = argparse.ArgumentParser(
        description="Validate run artifact smoke/integration output artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Run output directory, e.g. evaluation/results/locomo-mem0-smoke-20260419",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["add", "finalize", "search", "answer", "evaluate"],
        choices=["add", "finalize", "search", "answer", "evaluate"],
        help="Highest stage set to validate. Dependencies are included automatically.",
    )
    parser.add_argument(
        "--expected-dataset",
        type=str,
        default=None,
        help="Optional expected dataset_id from run_config_snapshot.json",
    )
    parser.add_argument(
        "--expected-system",
        type=str,
        default=None,
        help="Optional expected system_id from run_config_snapshot.json",
    )
    parser.add_argument(
        "--allow-non-category1",
        action="store_true",
        help="Skip benchmark_mode=category1 enforcement",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format",
    )
    args = parser.parse_args()

    report = validate_run_artifacts(
        output_dir=args.output_dir,
        stages=args.stages,
        expected_dataset=args.expected_dataset,
        expected_system=args.expected_system,
        require_category1=not args.allow_non_category1,
    )

    if args.format == "json":
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_validation_report(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
