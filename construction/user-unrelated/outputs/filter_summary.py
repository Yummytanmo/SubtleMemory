from __future__ import annotations

from pathlib import Path

from infra.json_utils import ensure_dir, load_json, save_json
from infra.path_registry import display_path, outputs_selected_root, outputs_to_filter_root


def _stage_pass_count(records: list[dict], stage: str) -> int:
    return sum(1 for item in records if item.get("filter_result", {}).get(stage, {}).get("decision") == "yes")


def build_summary(output_root: Path) -> dict:
    summaries: list[dict] = []
    if not output_root.exists():
        return {
            "files": summaries,
            "total_samples": 0,
            "overall_passed": 0,
            "overall_pass_rate": 0.0,
        }

    for category_dir in sorted(output_root.iterdir()):
        if not category_dir.is_dir():
            continue
        passed_dir = category_dir / "passed"
        failed_dir = category_dir / "failed"
        if not passed_dir.exists() or not failed_dir.exists():
            continue

        all_names = sorted({path.name for path in passed_dir.glob("*.json")} | {path.name for path in failed_dir.glob("*.json")})
        for name in all_names:
            passed_path = passed_dir / name
            failed_path = failed_dir / name
            passed_records = load_json(passed_path) if passed_path.exists() else []
            failed_records = load_json(failed_path) if failed_path.exists() else []
            all_records = list(passed_records) + list(failed_records)
            total = len(all_records)
            summary = {
                "category": category_dir.name,
                "input_file": display_path(outputs_to_filter_root() / category_dir.name / name),
                "total": total,
                "conversation_pass": _stage_pass_count(all_records, "conversation"),
                "question_pass": _stage_pass_count(all_records, "question"),
                "answer_pass": _stage_pass_count(all_records, "answer"),
                "overall_pass": len(passed_records),
                "overall_pass_rate": (len(passed_records) / total) if total else 0.0,
                "passed_file": display_path(passed_path),
                "failed_file": display_path(failed_path),
            }
            summaries.append(summary)

    total = sum(item["total"] for item in summaries)
    overall_passed = sum(item["overall_pass"] for item in summaries)
    return {
        "files": summaries,
        "total_samples": total,
        "overall_passed": overall_passed,
        "overall_pass_rate": (overall_passed / total) if total else 0.0,
    }


def save_summary_file(output_root: Path | None = None) -> Path:
    output_root = output_root or outputs_selected_root()
    ensure_dir(output_root)
    summary = build_summary(output_root)
    summary_path = output_root / "filter_summary.json"
    save_json(summary_path, summary, indent=2)
    return summary_path


def main(output_root: Path | None = None) -> None:
    summary_path = save_summary_file(output_root)
    print(f"Saved merged filter summary to {summary_path}")


if __name__ == "__main__":
    main()
