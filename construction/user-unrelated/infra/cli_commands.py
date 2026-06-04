from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from infra.json_utils import ensure_dir, load_config, load_json, save_json
from infra.path_registry import (
    data_root,
    display_path,
    ensure_runtime_layout,
    outputs_selected_root,
    outputs_to_filter_root,
    workflow_root,
)
from ingestion.reclassification import reclassify_archive_records
from ingestion.source_registry import (
    CATEGORY_CHOICES,
    COMPLEMENTARY_BATCH_SPECS,
    COMPLEMENTARY_FILE_TO_SOURCE,
    CONTRADICTORY_BATCH_SPECS,
    CONTRADICTORY_FILE_TO_SUBTYPE,
    config_path,
)
from outputs.filter_summary import save_summary_file
from outputs.passed_dataset import save_dataset_files


def _load_generation_classes():
    from generation.complementary_pipeline import ComplementaryPipeline
    from generation.contradictory_pipeline import ContradictoryPipeline
    from generation.nuanced_context_pipeline import NuancedContextPipeline
    from generation.nuanced_temporal_pipeline import NuancedTemporalPipeline

    return (
        ComplementaryPipeline,
        ContradictoryPipeline,
        NuancedTemporalPipeline,
        NuancedContextPipeline,
    )


def _load_filter_engine_class():
    from filtering.filter_engine import FilterEngine

    return FilterEngine


def normalize_categories(raw: str) -> tuple[str, ...]:
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    if not requested or "all" in requested:
        return CATEGORY_CHOICES
    invalid = requested - set(CATEGORY_CHOICES)
    if invalid:
        raise ValueError(f"Unsupported categories: {sorted(invalid)}")
    return tuple(category for category in CATEGORY_CHOICES if category in requested)


def normalize_name_filter(raw: str) -> set[str]:
    names = {item.strip() for item in raw.split(",") if item.strip()}
    if not names or "all" in names:
        return set()
    return names


def discover_input_files(input_root: Path, categories: tuple[str, ...], filenames: set[str] | None = None) -> list[tuple[str, Path]]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    selected_categories = set(categories)
    selected_filenames = filenames or set()
    jobs: list[tuple[str, Path]] = []
    for category_dir in sorted(input_root.iterdir()):
        if not category_dir.is_dir() or category_dir.name not in selected_categories:
            continue
        for json_file in sorted(category_dir.glob("*.json")):
            if selected_filenames and json_file.name not in selected_filenames:
                continue
            jobs.append((category_dir.name, json_file))
    return jobs


def _collect_successes(
    candidates: list[object],
    worker_fn: Callable[[tuple[int, object]], tuple[int, dict[str, Any] | None, str | None]],
    target_count: int,
    max_workers: int,
    label: str,
) -> list[dict[str, Any]]:
    successes: list[dict[str, Any]] = []
    cursor = 0
    batch_size = max(target_count * 2, max_workers)
    while len(successes) < target_count:
        batch_candidates = candidates[cursor : cursor + batch_size]
        if not batch_candidates:
            raise RuntimeError(
                f"Unable to collect {target_count} successful samples for {label}; only {len(successes)} succeeded."
            )
        jobs = list(enumerate(batch_candidates, start=cursor))
        ordered: list[dict[str, Any] | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs)) or 1) as executor:
            for position, record, error in executor.map(worker_fn, jobs):
                local_index = position - cursor
                if record is not None:
                    ordered[local_index] = record
                else:
                    print(f"Skipped {label} candidate {position}: {error}", flush=True)
        for record in ordered:
            if record is None:
                continue
            successes.append(record)
            if len(successes) >= target_count:
                break
        cursor += batch_size
    return successes[:target_count]


def _generate_complementary(count: int, max_workers: int, output_root: Path) -> None:
    ComplementaryPipeline, _, _, _ = _load_generation_classes()
    config = load_config(config_path("complementary_generation"))
    for source, filename in COMPLEMENTARY_BATCH_SPECS:
        pipeline = ComplementaryPipeline(data_root(), config)
        samples = pipeline.load_samples([source])

        def worker(job: tuple[int, object]) -> tuple[int, dict[str, Any] | None, str | None]:
            position, sample = job
            try:
                worker_pipeline = ComplementaryPipeline(data_root(), config)
                result = worker_pipeline.generate_sample(sample)
                return position, result.record, None
            except Exception as exc:
                return position, None, str(exc)

        records = _collect_successes(samples, worker, count, max_workers, f"complementary:{source}")
        output_path = output_root / "complementary" / filename
        ensure_dir(output_path.parent)
        save_json(output_path, records, indent=2)
        print(f"Saved {len(records)} complementary samples to {display_path(output_path)}", flush=True)


def _generate_contradictory(count: int, max_workers: int, output_root: Path) -> None:
    _, ContradictoryPipeline, _, _ = _load_generation_classes()
    config = load_config(config_path("contradictory_generation"))
    base_pipeline = ContradictoryPipeline(data_root(), config)
    all_samples = base_pipeline.load_samples(["contradictory_source"])
    for subtype, filename in CONTRADICTORY_BATCH_SPECS:
        candidates = [base_pipeline.materialize_subtype_sample(all_samples[idx], subtype) for idx in range(len(all_samples))]

        def worker(job: tuple[int, object]) -> tuple[int, dict[str, Any] | None, str | None]:
            position, sample = job
            try:
                worker_pipeline = ContradictoryPipeline(data_root(), config)
                result = worker_pipeline.generate_sample(sample)
                return position, result.record, None
            except Exception as exc:
                return position, None, str(exc)

        records = _collect_successes(candidates, worker, count, max_workers, f"contradictory:{subtype}")
        output_path = output_root / "contradictory" / filename
        ensure_dir(output_path.parent)
        save_json(output_path, records, indent=2)
        print(f"Saved {len(records)} contradictory samples to {display_path(output_path)}", flush=True)


def _generate_nuanced(count: int, max_workers: int, output_root: Path) -> None:
    _, _, NuancedTemporalPipeline, NuancedContextPipeline = _load_generation_classes()

    temporal_config = load_config(config_path("nuanced_temporal_generation"))
    temporal_pipeline = NuancedTemporalPipeline(data_root(), temporal_config)
    temporal_samples = temporal_pipeline.load_samples(["temporal_light"])

    def temporal_worker(job: tuple[int, object]) -> tuple[int, dict[str, Any] | None, str | None]:
        position, sample = job
        try:
            worker_pipeline = NuancedTemporalPipeline(data_root(), temporal_config)
            result = worker_pipeline.generate_sample(sample)
            return position, result.record, None
        except Exception as exc:
            return position, None, str(exc)

    temporal_records = _collect_successes(temporal_samples, temporal_worker, count, max_workers, "nuanced:temporal")
    temporal_output = output_root / "nuanced" / f"temporal_hard_{count}_samples.json"
    ensure_dir(temporal_output.parent)
    save_json(temporal_output, temporal_records, indent=2)
    print(f"Saved {len(temporal_records)} nuanced temporal samples to {display_path(temporal_output)}", flush=True)

    context_config = load_config(config_path("nuanced_context_generation"))
    context_pipeline = NuancedContextPipeline(data_root(), context_config)
    context_samples = context_pipeline.load_samples(["context_light"])

    def context_worker(job: tuple[int, object]) -> tuple[int, dict[str, Any] | None, str | None]:
        position, sample = job
        try:
            worker_pipeline = NuancedContextPipeline(data_root(), context_config)
            result = worker_pipeline.generate_sample(sample)
            return position, result.record, None
        except Exception as exc:
            return position, None, str(exc)

    context_records = _collect_successes(context_samples, context_worker, count, max_workers, "nuanced:context")
    context_output = output_root / "nuanced" / f"context_{count}_samples.json"
    ensure_dir(context_output.parent)
    save_json(context_output, context_records, indent=2)
    print(f"Saved {len(context_records)} nuanced context samples to {display_path(context_output)}", flush=True)


def run_generate_to_filter(*, count: int, max_workers: int, categories: str, output_root: Path) -> None:
    ensure_runtime_layout()
    ensure_dir(output_root)
    selected_categories = normalize_categories(categories)
    if "complementary" in selected_categories:
        _generate_complementary(count, max_workers, output_root)
    if "contradictory" in selected_categories:
        _generate_contradictory(count, max_workers, output_root)
    if "nuanced" in selected_categories:
        _generate_nuanced(count, max_workers, output_root)


def filter_file(category: str, input_path: Path, engine_config: dict[str, Any], output_root: Path, max_workers: int) -> dict[str, Any]:
    FilterEngine = _load_filter_engine_class()
    samples = load_json(input_path)
    if not isinstance(samples, list):
        raise ValueError(f"Expected a list in {input_path}")

    jobs = list(enumerate(samples))

    def worker(job: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        position, sample = job
        worker_engine = FilterEngine(workflow_root(), engine_config)
        result = worker_engine.filter_sample(category, sample)
        enriched = dict(sample)
        enriched["filter_result"] = result
        return position, enriched

    ordered: list[dict[str, Any] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs)) or 1) as executor:
        for position, enriched in executor.map(worker, jobs):
            ordered[position] = enriched

    results = [item for item in ordered if item is not None]
    passed = [item for item in results if item["filter_result"]["overall_pass"]]
    failed = [item for item in results if not item["filter_result"]["overall_pass"]]
    passed_path = output_root / category / "passed" / input_path.name
    failed_path = output_root / category / "failed" / input_path.name
    ensure_dir(passed_path.parent)
    ensure_dir(failed_path.parent)
    save_json(passed_path, passed, indent=2)
    save_json(failed_path, failed, indent=2)

    summary = {
        "category": category,
        "input_file": display_path(input_path),
        "total": len(results),
        "conversation_pass": sum(1 for item in results if item["filter_result"]["conversation"]["decision"] == "yes"),
        "question_pass": sum(1 for item in results if item["filter_result"]["question"]["decision"] == "yes"),
        "answer_pass": sum(1 for item in results if item["filter_result"]["answer"]["decision"] == "yes"),
        "overall_pass": len(passed),
        "overall_pass_rate": (len(passed) / len(results)) if results else 0.0,
        "passed_file": display_path(passed_path),
        "failed_file": display_path(failed_path),
    }
    print(
        f"Filtered {input_path.name}: overall {summary['overall_pass']}/{summary['total']} "
        f"({summary['overall_pass_rate']:.2%})",
        flush=True,
    )
    return summary


def run_filter(
    *,
    config_file: Path,
    input_root: Path,
    output_root: Path,
    categories: str,
    files: str,
    max_workers: int,
) -> dict[str, Any]:
    config = load_config(config_file)
    ensure_dir(output_root)
    selected_categories = normalize_categories(categories)
    selected_files = normalize_name_filter(files)
    summaries = [
        filter_file(category, input_path, config, output_root, max_workers)
        for category, input_path in discover_input_files(input_root, selected_categories, selected_files)
    ]

    total = sum(item["total"] for item in summaries)
    overall_pass = sum(item["overall_pass"] for item in summaries)
    global_summary = {
        "files": summaries,
        "total_samples": total,
        "overall_passed": overall_pass,
        "overall_pass_rate": (overall_pass / total) if total else 0.0,
    }
    summary_path = output_root / "filter_summary.json"
    save_json(summary_path, global_summary, indent=2)
    print(f"Saved filter summary to {display_path(summary_path)}", flush=True)
    return global_summary


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _clean_runtime_fields(record: dict[str, Any]) -> dict[str, Any]:
    clean = dict(record)
    clean.pop("filter_result", None)
    clean.pop("quality_feedback", None)
    metadata = clean.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.pop("repair_history", None)
        metadata.pop("last_repair_error", None)
        clean["metadata"] = metadata
    return clean


def _record_hash(record: dict[str, Any]) -> str:
    clean = _clean_runtime_fields(record)
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_previous_selection(output_root: Path, category: str, filename: str) -> list[dict[str, Any]]:
    previous: list[dict[str, Any]] = []
    for split in ("passed", "failed"):
        previous.extend(_load_records(output_root / category / split / filename))
    return previous


def _filter_and_save(
    *,
    category: str,
    input_path: Path,
    records: list[dict[str, Any]],
    filter_config: dict[str, Any],
    output_root: Path,
    max_workers: int,
    previous_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    FilterEngine = _load_filter_engine_class()
    previous_by_id = {
        record.get("sample_id"): record
        for record in (previous_records or [])
        if isinstance(record, dict) and record.get("sample_id") and record.get("filter_result")
    }
    ordered: list[dict[str, Any] | None] = [None] * len(records)
    jobs: list[tuple[int, dict[str, Any]]] = []
    reused = 0
    for idx, record in enumerate(records):
        previous = previous_by_id.get(record.get("sample_id"))
        if previous is not None and _record_hash(previous) == _record_hash(record):
            ordered[idx] = dict(previous)
            reused += 1
        else:
            jobs.append((idx, record))

    if previous_records is not None:
        print(
            f"Reused {reused} previous filter judgments for {category}/{input_path.name}; "
            f"filtering {len(jobs)} changed/new records.",
            flush=True,
        )

    def worker(job: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        position, sample = job
        worker_engine = FilterEngine(workflow_root(), filter_config)
        result = worker_engine.filter_sample(category, sample)
        enriched = dict(sample)
        enriched["filter_result"] = result
        return position, enriched

    if jobs:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs)) or 1) as executor:
            for position, enriched in executor.map(worker, jobs):
                ordered[position] = enriched

    results = [item for item in ordered if item is not None]
    passed = [item for item in results if item["filter_result"]["overall_pass"]]
    failed = [item for item in results if not item["filter_result"]["overall_pass"]]
    passed_path = output_root / category / "passed" / input_path.name
    failed_path = output_root / category / "failed" / input_path.name
    ensure_dir(passed_path.parent)
    ensure_dir(failed_path.parent)
    save_json(passed_path, passed, indent=2)
    save_json(failed_path, failed, indent=2)

    return {
        "category": category,
        "input_file": display_path(input_path),
        "total": len(results),
        "conversation_pass": sum(1 for item in results if item["filter_result"]["conversation"]["decision"] == "yes"),
        "question_pass": sum(1 for item in results if item["filter_result"]["question"]["decision"] == "yes"),
        "answer_pass": sum(1 for item in results if item["filter_result"]["answer"]["decision"] == "yes"),
        "overall_pass": len(passed),
        "overall_pass_rate": (len(passed) / len(results)) if results else 0.0,
        "passed_file": display_path(passed_path),
        "failed_file": display_path(failed_path),
    }


def _feedback_from_filter_result(filter_result: dict[str, Any]) -> dict[str, str]:
    for stage in ("conversation", "question", "answer"):
        stage_result = filter_result.get(stage, {})
        if stage_result.get("decision") == "no":
            return {
                "general": (
                    f"The previous sample failed the {stage} filter. Regenerate the sample so it satisfies "
                    "the benchmark subtype while addressing the concrete rejection reason."
                ),
                stage: str(stage_result.get("reason", "")).strip(),
            }
    return {
        "general": "The previous sample failed filtering. Regenerate it to better satisfy the benchmark requirements."
    }


def _save_raw_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    save_json(path, [_clean_runtime_fields(record) for record in records], indent=2)


def _source_map_for_file(category: str, filename: str) -> tuple[Any, dict[str, dict[str, Any]]]:
    ComplementaryPipeline, ContradictoryPipeline, NuancedTemporalPipeline, NuancedContextPipeline = _load_generation_classes()

    if category == "complementary":
        source = COMPLEMENTARY_FILE_TO_SOURCE[filename]
        config = load_config(config_path("complementary_generation"))
        pipeline = ComplementaryPipeline(data_root(), config)
        return pipeline, {sample["sample_id"]: sample for sample in pipeline.load_samples([source])}

    if category == "contradictory":
        subtype = CONTRADICTORY_FILE_TO_SUBTYPE[filename]
        config = load_config(config_path("contradictory_generation"))
        pipeline = ContradictoryPipeline(data_root(), config)
        base_samples = pipeline.load_samples(["contradictory_source"])
        source_map: dict[str, dict[str, Any]] = {}
        for sample in base_samples:
            materialized = pipeline.materialize_subtype_sample(sample, subtype)
            source_map[materialized["sample_id"]] = materialized
        return pipeline, source_map

    if category == "nuanced" and re.fullmatch(r"temporal_hard_\d+_samples\.json", filename):
        config = load_config(config_path("nuanced_temporal_generation"))
        pipeline = NuancedTemporalPipeline(data_root(), config)
        return pipeline, {sample["sample_id"]: sample for sample in pipeline.load_samples(["temporal_light"])}

    if category == "nuanced" and re.fullmatch(r"context_\d+_samples\.json", filename):
        config = load_config(config_path("nuanced_context_generation"))
        pipeline = NuancedContextPipeline(data_root(), config)
        return pipeline, {sample["sample_id"]: sample for sample in pipeline.load_samples(["context_light"])}

    raise ValueError(f"Unsupported repair target: {category}/{filename}")


def _regenerate_failed_records(
    *,
    category: str,
    filename: str,
    records: list[dict[str, Any]],
    failed_records_by_id: dict[str, dict[str, Any]],
    max_workers: int,
) -> list[dict[str, Any]]:
    failed_sample_ids = set(failed_records_by_id)
    if not failed_sample_ids:
        return [_clean_runtime_fields(record) for record in records]

    _, source_by_id = _source_map_for_file(category, filename)
    jobs = [(idx, record) for idx, record in enumerate(records) if record.get("sample_id") in failed_sample_ids]

    def worker(job: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        idx, record = job
        sample_id = record.get("sample_id")
        source_sample = source_by_id.get(sample_id)
        if source_sample is None:
            return idx, _clean_runtime_fields(record)
        source_sample = dict(source_sample)
        source_sample["quality_feedback"] = _feedback_from_filter_result(
            failed_records_by_id[sample_id].get("filter_result", {})
        )
        try:
            pipeline, _ = _source_map_for_file(category, filename)
            regenerated = pipeline.generate_sample(source_sample).record
            return idx, _clean_runtime_fields(regenerated)
        except Exception:
            return idx, _clean_runtime_fields(record)

    replacements: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs)) or 1) as executor:
        for idx, regenerated in executor.map(worker, jobs):
            replacements[idx] = regenerated

    updated: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        updated.append(replacements.get(idx, _clean_runtime_fields(record)))
    return updated


def repair_file(
    *,
    category: str,
    input_path: Path,
    filter_config: dict[str, Any],
    output_root: Path,
    target_rate: float,
    max_rounds: int,
    max_workers: int,
) -> dict[str, Any]:
    current_records = _load_records(input_path)
    previous_records = _load_previous_selection(output_root, category, input_path.name)
    summary = _filter_and_save(
        category=category,
        input_path=input_path,
        records=current_records,
        filter_config=filter_config,
        output_root=output_root,
        max_workers=max_workers,
        previous_records=previous_records,
    )
    print(
        f"Round 0 {category}/{input_path.name}: "
        f"{summary['overall_pass']}/{summary['total']} ({summary['overall_pass_rate']:.2%})",
        flush=True,
    )
    if summary["overall_pass_rate"] >= target_rate:
        return summary

    current_filtered = _load_previous_selection(output_root, category, input_path.name)
    for repair_round in range(1, max_rounds + 1):
        failed_records = _load_records(output_root / category / "failed" / input_path.name)
        failed_records_by_id = {record["sample_id"]: record for record in failed_records}
        current_records = _regenerate_failed_records(
            category=category,
            filename=input_path.name,
            records=current_records,
            failed_records_by_id=failed_records_by_id,
            max_workers=max_workers,
        )
        _save_raw_records(input_path, current_records)
        summary = _filter_and_save(
            category=category,
            input_path=input_path,
            records=current_records,
            filter_config=filter_config,
            output_root=output_root,
            max_workers=max_workers,
            previous_records=current_filtered,
        )
        print(
            f"Round {repair_round} {category}/{input_path.name}: "
            f"{summary['overall_pass']}/{summary['total']} ({summary['overall_pass_rate']:.2%})",
            flush=True,
        )
        if summary["overall_pass_rate"] >= target_rate:
            return summary
        current_filtered = _load_previous_selection(output_root, category, input_path.name)

    print(
        f"Stopped {category}/{input_path.name}: pass rate {summary['overall_pass_rate']:.2%} "
        f"is still below target {target_rate:.2%} after {max_rounds} repair rounds.",
        flush=True,
    )
    return summary


def run_repair(
    *,
    input_root: Path,
    output_root: Path,
    filter_config_file: Path,
    categories: str,
    target_rate: float,
    max_rounds: int,
    max_workers: int,
) -> dict[str, Any]:
    filter_config = load_config(filter_config_file)
    ensure_dir(output_root)
    selected_categories = normalize_categories(categories)
    summaries = [
        repair_file(
            category=category,
            input_path=input_path,
            filter_config=filter_config,
            output_root=output_root,
            target_rate=target_rate,
            max_rounds=max_rounds,
            max_workers=max_workers,
        )
        for category, input_path in discover_input_files(input_root, selected_categories)
    ]

    total = sum(item["total"] for item in summaries)
    overall_pass = sum(item["overall_pass"] for item in summaries)
    global_summary = {
        "files": summaries,
        "total_samples": total,
        "overall_passed": overall_pass,
        "overall_pass_rate": (overall_pass / total) if total else 0.0,
    }
    summary_path = output_root / "filter_summary.json"
    save_json(summary_path, global_summary, indent=2)
    print(f"Saved filter summary to {display_path(summary_path)}", flush=True)
    return global_summary


def _answer_failed(record: dict[str, Any]) -> bool:
    return record.get("filter_result", {}).get("answer", {}).get("decision") == "no"


def _refilter_answer(category: str, record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    FilterEngine = _load_filter_engine_class()
    engine = FilterEngine(workflow_root(), config)
    prompt_module = engine.prompt_modules[category]
    answer_result = engine._run_stage(
        prompt_module.ANSWER_FILTER_SYSTEM_PROMPT,
        prompt_module.build_answer_filter_prompt(record),
    )
    updated = dict(record)
    filter_result = dict(updated.get("filter_result") or {})
    filter_result["answer"] = answer_result
    filter_result["overall_pass"] = (
        filter_result.get("conversation", {}).get("decision") == "yes"
        and filter_result.get("question", {}).get("decision") == "yes"
        and answer_result.get("decision") == "yes"
    )
    updated["filter_result"] = filter_result
    return updated


def _process_answer_failed_file(category: str, failed_path: Path, config: dict[str, Any], max_workers: int) -> dict[str, Any]:
    filename = failed_path.name
    passed_path = failed_path.parents[1] / "passed" / filename
    failed_records = _load_records(failed_path)
    passed_records = _load_records(passed_path)

    answer_failed_records = [record for record in failed_records if _answer_failed(record)]
    untouched_failed = [record for record in failed_records if not _answer_failed(record)]
    if not answer_failed_records:
        return {
            "file": display_path(failed_path),
            "answer_failed": 0,
            "moved_to_passed": 0,
            "still_failed": 0,
        }

    with ThreadPoolExecutor(max_workers=min(max_workers, len(answer_failed_records)) or 1) as executor:
        refiltered = list(executor.map(lambda record: _refilter_answer(category, record, config), answer_failed_records))

    newly_passed = [record for record in refiltered if record["filter_result"]["overall_pass"]]
    still_failed = [record for record in refiltered if not record["filter_result"]["overall_pass"]]
    passed_records.extend(newly_passed)
    failed_records = untouched_failed + still_failed

    ensure_dir(passed_path.parent)
    ensure_dir(failed_path.parent)
    save_json(passed_path, passed_records, indent=2)
    save_json(failed_path, failed_records, indent=2)

    return {
        "file": display_path(failed_path),
        "answer_failed": len(answer_failed_records),
        "moved_to_passed": len(newly_passed),
        "still_failed": len(still_failed),
    }


def run_refilter_answer_failures(*, config_file: Path, output_root: Path, categories: str, max_workers: int) -> list[dict[str, Any]]:
    config = load_config(config_file)
    selected_categories = set(normalize_categories(categories))

    reports: list[dict[str, Any]] = []
    if not output_root.exists():
        return reports

    for category_dir in sorted(output_root.iterdir()):
        if not category_dir.is_dir() or category_dir.name not in selected_categories:
            continue
        failed_dir = category_dir / "failed"
        if not failed_dir.exists():
            continue
        for failed_path in sorted(failed_dir.glob("*.json")):
            report = _process_answer_failed_file(category_dir.name, failed_path, config, max_workers)
            reports.append(report)
            if report["answer_failed"]:
                print(
                    f"{report['file']}: answer_failed={report['answer_failed']}, "
                    f"moved_to_passed={report['moved_to_passed']}, still_failed={report['still_failed']}",
                    flush=True,
                )

    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)
    return reports


def run_rebuild_filter_summary(*, output_root: Path) -> Path:
    summary_path = save_summary_file(output_root)
    print(f"Saved merged filter summary to {display_path(summary_path)}", flush=True)
    return summary_path


def run_build_passed_dataset(*, output_root: Path) -> tuple[Path, Path]:
    output_path, summary_path = save_dataset_files(output_root)
    records = load_json(output_path)
    print(f"Saved {len(records)} normalized passed samples to {display_path(output_path)}", flush=True)
    print(f"Saved summary to {display_path(summary_path)}", flush=True)
    return output_path, summary_path


def run_reclassify_ambiguous_context(*, input_root: Path, output_root: Path) -> Path:
    written_path, count = reclassify_archive_records(input_root, output_root)
    print(f"Saved {count} reclassified samples to {display_path(written_path)}", flush=True)
    return written_path
