"""Validation helpers for real smoke and integration run artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


_STAGE_ORDER = (
    "add",
    "finalize",
    "search",
    "answer",
    "evaluate",
)
_BASE_FILES = ("run_config_snapshot.json", "source_units.jsonl")
_STAGE_FILES = {
    "add": ("import_manifest.jsonl",),
    "finalize": (),
    "search": ("search_results.json",),
    "answer": ("answer_results.json", "qa_results.jsonl"),
    "evaluate": ("eval_results.json", "evaluation_results.jsonl", "score_summary.json"),
}
_DEFAULT_MAIN_STAGES = ("add", "finalize", "search", "answer", "evaluate")


@dataclass
class ValidationIssue:
    """Single validation issue."""

    level: str
    code: str
    message: str


@dataclass
class ValidationReport:
    """Structured validation report."""

    output_dir: str
    stages: List[str]
    counts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def warn(self, code: str, message: str) -> None:
        self.warnings.append(ValidationIssue(level="warning", code=code, message=message))

    def error(self, code: str, message: str) -> None:
        self.errors.append(ValidationIssue(level="error", code=code, message=message))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "output_dir": self.output_dir,
            "stages": self.stages,
            "counts": self.counts,
            "warnings": [
                {"level": issue.level, "code": issue.code, "message": issue.message}
                for issue in self.warnings
            ],
            "errors": [
                {"level": issue.level, "code": issue.code, "message": issue.message}
                for issue in self.errors
            ],
        }


def _normalize_stages(stages: Optional[Sequence[str]]) -> List[str]:
    requested = list(stages or _DEFAULT_MAIN_STAGES)
    if not requested:
        return list(_DEFAULT_MAIN_STAGES)

    indexes = []
    for stage in requested:
        if stage not in _STAGE_ORDER:
            raise ValueError(f"Unsupported stage: {stage}")
        indexes.append(_STAGE_ORDER.index(stage))

    highest = max(indexes)
    return list(_STAGE_ORDER[: highest + 1])


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _duplicates(values: Iterable[str]) -> List[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _sample(values: Iterable[str], limit: int = 5) -> str:
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
        if len(unique_values) >= limit:
            break
    if not unique_values:
        return ""
    return ", ".join(unique_values)


def _dataset_matches_expected(actual: Any, expected: str) -> bool:
    actual_text = str(actual or "")
    expected_text = str(expected or "")
    return actual_text == expected_text or actual_text == f"{expected_text}_smoke"


def _is_smoke_snapshot(snapshot: Dict[str, Any]) -> bool:
    dataset_id = str((snapshot or {}).get("dataset_id") or "")
    runtime = (snapshot or {}).get("runtime") or {}
    return dataset_id.endswith("_smoke") or bool(runtime.get("smoke_test"))


def _load_artifact(
    path: Path,
    loader: Any,
    report: ValidationReport,
    code_prefix: str,
    default: Any,
) -> Any:
    if not path.exists():
        return default
    try:
        return loader(path)
    except Exception as exc:  # noqa: BLE001
        report.error(
            f"{code_prefix}.load_failed",
            f"Failed to load {path.name}: {type(exc).__name__}: {exc}",
        )
        return default


def validate_run_artifacts(
    output_dir: Path | str,
    stages: Optional[Sequence[str]] = None,
    expected_dataset: Optional[str] = None,
    expected_system: Optional[str] = None,
    require_category1: bool = True,
) -> ValidationReport:
    """Validate a real run directory produced by the evaluation pipeline."""
    run_dir = Path(output_dir)
    requested_stages = list(stages or _DEFAULT_MAIN_STAGES)
    normalized_stages = _normalize_stages(stages)
    report = ValidationReport(output_dir=str(run_dir), stages=normalized_stages)

    if not run_dir.exists():
        report.error("output_dir.missing", f"Output directory does not exist: {run_dir}")
        return report

    required_files = set(_BASE_FILES)
    for stage in normalized_stages:
        required_files.update(_STAGE_FILES.get(stage, ()))

    for filename in sorted(required_files):
        if not (run_dir / filename).exists():
            report.error(
                "artifact.missing",
                f"Required artifact is missing for stages {normalized_stages}: {filename}",
            )

    snapshot = _load_artifact(
        run_dir / "run_config_snapshot.json",
        _load_json,
        report,
        "run_config_snapshot",
        default={},
    )
    is_smoke_run = _is_smoke_snapshot(snapshot if isinstance(snapshot, dict) else {})
    source_units = _load_artifact(
        run_dir / "source_units.jsonl",
        _load_jsonl,
        report,
        "source_units",
        default=[],
    )
    import_manifest = _load_artifact(
        run_dir / "import_manifest.jsonl",
        _load_jsonl,
        report,
        "import_manifest",
        default=[],
    )
    search_results = _load_artifact(
        run_dir / "search_results.json",
        _load_json,
        report,
        "search_results",
        default=[],
    )
    answer_results = _load_artifact(
        run_dir / "answer_results.json",
        _load_json,
        report,
        "answer_results",
        default=[],
    )
    qa_results = _load_artifact(
        run_dir / "qa_results.jsonl",
        _load_jsonl,
        report,
        "qa_results",
        default=[],
    )
    eval_results = _load_artifact(
        run_dir / "eval_results.json",
        _load_json,
        report,
        "eval_results",
        default={},
    )
    evaluation_results = _load_artifact(
        run_dir / "evaluation_results.jsonl",
        _load_jsonl,
        report,
        "evaluation_results",
        default=[],
    )
    score_summary = _load_artifact(
        run_dir / "score_summary.json",
        _load_json,
        report,
        "score_summary",
        default={},
    )
    finalize_report = _load_artifact(
        run_dir / "finalize_report.json",
        _load_json,
        report,
        "finalize_report",
        default={},
    )

    if snapshot:
        runtime = snapshot.get("runtime", {}) or {}
        if not snapshot.get("run_id"):
            report.error("run_config.run_id_missing", "run_config_snapshot.json is missing run_id")
        if require_category1 and snapshot.get("benchmark_mode") != "category1":
            report.error(
                "run_config.benchmark_mode",
                f"Expected benchmark_mode=category1, got {snapshot.get('benchmark_mode')!r}",
            )
        if expected_dataset and not _dataset_matches_expected(
            snapshot.get("dataset_id"), expected_dataset
        ):
            report.error(
                "run_config.dataset_mismatch",
                f"Expected dataset_id={expected_dataset!r}, got {snapshot.get('dataset_id')!r}",
            )
        if expected_system and snapshot.get("system_id") != expected_system:
            report.error(
                "run_config.system_mismatch",
                f"Expected system_id={expected_system!r}, got {snapshot.get('system_id')!r}",
            )
    else:
        runtime = {}

    requires_finalize_report = bool(
        runtime.get("requires_finalize_report")
        or snapshot.get("requires_finalize_report")
    )
    finalize_gate_requested = any(
        stage in requested_stages for stage in ("finalize", "search", "answer")
    )
    finalize_visibility_requested = finalize_gate_requested or "evaluate" in requested_stages
    if finalize_report:
        report.counts["finalize_ready"] = bool(finalize_report.get("ready"))
        report.counts["finalize_status"] = str(finalize_report.get("status", "unknown"))
        report.counts["finalize_updated_rows"] = int(finalize_report.get("updated_rows", 0))
        report.counts["finalize_provider_status_counts"] = (
            finalize_report.get("provider_status_counts", {}) or {}
        )
        report.counts["finalize_budget_exhausted"] = bool(
            finalize_report.get("finalize_budget_exhausted")
        )
        if finalize_gate_requested and not finalize_report.get("ready"):
            report.error(
                "finalize_report.not_ready",
                "finalize_report.json exists but ready=false; search/answer should remain blocked",
            )
    elif finalize_visibility_requested:
        if requires_finalize_report:
            report.error(
                "finalize_report.missing",
                "This run requires finalize_report.json before finalize/search/answer stages can proceed",
            )
        else:
            report.warn(
                "finalize_report.missing_legacy",
                "finalize_report.json is missing; treating this as a legacy run and allowing read-only validation",
            )

    source_unit_ids = [str(row.get("source_unit_id", "")) for row in source_units]
    source_unit_id_set = {value for value in source_unit_ids if value}
    report.counts["source_units"] = len(source_units)
    report.counts["source_unit_conversations"] = len(
        {str(row.get("conversation_id", "")) for row in source_units if row.get("conversation_id")}
    )
    if "add" in normalized_stages and not source_units:
        report.error("source_units.empty", "source_units.jsonl is empty")
    duplicate_source_ids = _duplicates(source_unit_ids)
    if duplicate_source_ids:
        report.error(
            "source_units.duplicate_id",
            f"Duplicate source_unit_id values found: {_sample(duplicate_source_ids)}",
        )

    manifest_chunk_ids = [str(row.get("chunk_id", "")) for row in import_manifest]
    manifest_source_ids = [
        str(source_unit_id)
        for row in import_manifest
        for source_unit_id in row.get("source_unit_ids", [])
    ]
    manifest_source_id_set = set(manifest_source_ids)
    manifest_memory_refs = sum(len(row.get("memory_refs", [])) for row in import_manifest)
    report.counts["import_manifest_rows"] = len(import_manifest)
    report.counts["manifest_memory_refs"] = manifest_memory_refs
    report.counts["manifest_write_status"] = dict(
        Counter(str(row.get("write_status", "unknown")) for row in import_manifest)
    )

    if "add" in normalized_stages and not import_manifest:
        report.error("import_manifest.empty", "import_manifest.jsonl is empty")
    duplicate_chunk_ids = _duplicates(manifest_chunk_ids)
    if duplicate_chunk_ids:
        report.error(
            "import_manifest.duplicate_chunk_id",
            f"Duplicate manifest chunk_id values found: {_sample(duplicate_chunk_ids)}",
        )
    if is_smoke_run:
        unknown_manifest_source_ids = []
    else:
        unknown_manifest_source_ids = sorted(
            source_unit_id
            for source_unit_id in manifest_source_id_set
            if source_unit_id not in source_unit_id_set
        )
    if unknown_manifest_source_ids:
        report.error(
            "import_manifest.unknown_source_unit",
            "Manifest/source_units source_unit_id mismatch: "
            f"{_sample(unknown_manifest_source_ids)}",
        )
    if import_manifest and manifest_memory_refs == 0:
        report.error(
            "import_manifest.memory_refs_missing",
            "All import manifest rows are missing memory_refs; readback search cannot work",
        )

    search_question_ids = [str(row.get("question_id", "")) for row in search_results]
    answer_question_ids = [str(row.get("question_id", "")) for row in answer_results]
    qa_question_ids = [str(row.get("question_id", "")) for row in qa_results]
    evaluation_question_ids = [str(row.get("question_id", "")) for row in evaluation_results]

    report.counts["search_results"] = len(search_results)
    report.counts["answer_results"] = len(answer_results)
    report.counts["qa_results"] = len(qa_results)
    report.counts["evaluation_results"] = len(evaluation_results)

    if "search" in normalized_stages and not search_results:
        report.error("search_results.empty", "search_results.json is empty")
    if "answer" in normalized_stages and not answer_results:
        report.error("answer_results.empty", "answer_results.json is empty")
    if "answer" in normalized_stages and not qa_results:
        report.error("qa_results.empty", "qa_results.jsonl is empty")
    if "evaluate" in normalized_stages and not evaluation_results:
        report.error("evaluation_results.empty", "evaluation_results.jsonl is empty")
    if "evaluate" in normalized_stages and not score_summary:
        report.error("score_summary.empty", "score_summary.json is empty")

    for row in search_results:
        if "retrieval_metadata" not in row:
            report.warn(
                "search_results.metadata_missing",
                f"Search result {row.get('question_id')} is missing retrieval_metadata",
            )

    for code, values in (
        ("search_results.duplicate_question_id", search_question_ids),
        ("answer_results.duplicate_question_id", answer_question_ids),
        ("qa_results.duplicate_question_id", qa_question_ids),
        ("evaluation_results.duplicate_question_id", evaluation_question_ids),
    ):
        duplicates = _duplicates(values)
        if duplicates:
            report.error(code, f"Duplicate question_id values found: {_sample(duplicates)}")

    qa_question_id_set = {value for value in qa_question_ids if value}
    search_question_id_set = {value for value in search_question_ids if value}
    answer_question_id_set = {value for value in answer_question_ids if value}
    evaluation_question_id_set = {value for value in evaluation_question_ids if value}

    if qa_results and answer_results and qa_question_id_set != answer_question_id_set:
        missing = sorted(qa_question_id_set - answer_question_id_set)
        extra = sorted(answer_question_id_set - qa_question_id_set)
        report.error(
            "qa_results.answer_alignment",
            "qa_results.jsonl and answer_results.json question_id sets differ; "
            f"missing_in_answer={_sample(missing)}, extra_in_answer={_sample(extra)}",
        )

    if qa_results and search_results and qa_question_id_set != search_question_id_set:
        missing = sorted(qa_question_id_set - search_question_id_set)
        extra = sorted(search_question_id_set - qa_question_id_set)
        report.error(
            "qa_results.search_alignment",
            "qa_results.jsonl and search_results.json question_id sets differ; "
            f"missing_in_search={_sample(missing)}, extra_in_search={_sample(extra)}",
        )

    if qa_results and evaluation_results and qa_question_id_set != evaluation_question_id_set:
        missing = sorted(qa_question_id_set - evaluation_question_id_set)
        extra = sorted(evaluation_question_id_set - qa_question_id_set)
        report.error(
            "evaluation_results.qa_alignment",
            "evaluation_results.jsonl and qa_results.jsonl question_id sets differ; "
            f"missing_in_evaluation={_sample(missing)}, extra_in_evaluation={_sample(extra)}",
        )

    if qa_results:
        unknown_evidence_ids = sorted(
            {
                str(source_unit_id)
                for row in qa_results
                for source_unit_id in row.get("evidence_source_unit_ids", [])
                if str(source_unit_id) not in source_unit_id_set
            }
        )
        if unknown_evidence_ids:
            report.error(
                "qa_results.unknown_evidence_source_unit",
                "qa_results.jsonl references unknown evidence_source_unit_ids: "
                f"{_sample(unknown_evidence_ids)}",
            )

        empty_evidence_questions = [
            str(row.get("question_id", ""))
            for row in qa_results
            if not row.get("evidence_source_unit_ids")
        ]
        if empty_evidence_questions:
            report.warn(
                "qa_results.empty_evidence",
                "Some QA rows do not have mapped evidence_source_unit_ids: "
                f"{_sample(empty_evidence_questions)}",
            )

        retrieval_mismatches = [
            str(row.get("question_id", ""))
            for row in qa_results
            if ((row.get("retrieval_artifact") or {}).get("question_id") or "") != row.get("question_id")
        ]
        if retrieval_mismatches:
            report.error(
                "qa_results.retrieval_artifact_question_id",
                "retrieval_artifact.question_id does not match QA row question_id for: "
                f"{_sample(retrieval_mismatches)}",
            )

        uncovered_evidence_ids = sorted(
            {
                str(source_unit_id)
                for row in qa_results
                for source_unit_id in row.get("evidence_source_unit_ids", [])
                if str(source_unit_id) not in manifest_source_id_set
            }
        )
        if "add" in normalized_stages and uncovered_evidence_ids:
            report.error(
                "qa_results.evidence_manifest_coverage",
                "No import manifest row covers some evidence_source_unit_ids: "
                f"{_sample(uncovered_evidence_ids)}",
            )

    if eval_results and "evaluate" in normalized_stages:
        eval_total = int(eval_results.get("total_questions", 0))
        eval_correct = int(eval_results.get("correct", 0))
        eval_accuracy = float(eval_results.get("accuracy", 0.0))
        details = eval_results.get("detailed_results", [])
        if isinstance(details, dict):
            detail_rows = []
            for rows in details.values():
                detail_rows.extend(rows or [])
        else:
            detail_rows = list(details or [])

        if evaluation_results and eval_total != len(evaluation_results):
            report.error(
                "eval_results.total_questions",
                f"eval_results total_questions={eval_total} but evaluation_results has {len(evaluation_results)} rows",
            )
        if detail_rows and eval_total != len(detail_rows):
            report.error(
                "eval_results.detail_count",
                f"eval_results total_questions={eval_total} but detailed_results has {len(detail_rows)} rows",
            )
        expected_accuracy = eval_correct / eval_total if eval_total else 0.0
        if abs(eval_accuracy - expected_accuracy) > 1e-9:
            report.error(
                "eval_results.accuracy",
                f"eval_results accuracy={eval_accuracy} but expected {expected_accuracy}",
            )

    if score_summary and evaluation_results:
        total_questions = len(evaluation_results)
        correct = sum(1 for row in evaluation_results if row.get("final_score"))
        if score_summary.get("total_questions") != total_questions:
            report.error(
                "score_summary.total_mismatch",
                f"score_summary total_questions={score_summary.get('total_questions')} "
                f"but evaluation_results has {total_questions} rows",
            )
        if score_summary.get("correct") != correct:
            report.error(
                "score_summary.correct_mismatch",
                f"score_summary correct={score_summary.get('correct')} "
                f"but evaluation_results has {correct} final_score=true rows",
            )
        expected_accuracy = correct / total_questions if total_questions else 0.0
        actual_accuracy = float(score_summary.get("accuracy", 0.0))
        if abs(actual_accuracy - expected_accuracy) > 1e-9:
            report.error(
                "score_summary.accuracy_mismatch",
                f"score_summary accuracy={actual_accuracy} but expected {expected_accuracy}",
            )
        report.counts["correct"] = correct

    return report


def render_validation_report(report: ValidationReport) -> str:
    """Render a validation report as human-readable text."""
    lines = [
        "=" * 60,
        "Run Artifact Validation",
        "=" * 60,
        f"Output Dir: {report.output_dir}",
        f"Stages: {', '.join(report.stages)}",
        f"Status: {'PASS' if report.ok else 'FAIL'}",
        "",
    ]

    if report.counts:
        lines.append("Counts:")
        for key in sorted(report.counts):
            lines.append(f"- {key}: {report.counts[key]}")
        lines.append("")

    if report.warnings:
        lines.append(f"Warnings ({len(report.warnings)}):")
        for issue in report.warnings:
            lines.append(f"- [{issue.code}] {issue.message}")
        lines.append("")

    if report.errors:
        lines.append(f"Errors ({len(report.errors)}):")
        for issue in report.errors:
            lines.append(f"- [{issue.code}] {issue.message}")
        lines.append("")

    return "\n".join(lines)
