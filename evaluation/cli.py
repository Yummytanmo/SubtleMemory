"""
CLI entry point for the evaluation framework.

Usage:
    python -m evaluation.cli --dataset locomo --system evermemos
    python -m evaluation.cli --dataset locomo --system evermemos --smoke 10
    python -m evaluation.cli --dataset locomo --system evermemos --stages add
    python -m evaluation.cli --dataset locomo --system evermemos --stages finalize search answer evaluate
"""

import asyncio
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Environment initialization - must be done before importing EverMemOS components
# Reference: src/bootstrap.py initialization logic

# Add project paths
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load environment variables
from common_utils.load_env import setup_environment

setup_environment(load_env_file_name=".env", check_env_var="MONGODB_HOST")

from evaluation.src.core.loaders import load_dataset
from evaluation.src.core.pipeline import Pipeline
from evaluation.src.adapters.core.registry import create_adapter
from evaluation.src.evaluators.registry import create_evaluator
from evaluation.src.utils.config import load_yaml
from evaluation.src.utils.logger import get_console

from memory_layer.llm.llm_provider import LLMProvider


REUSABLE_ADD_FINALIZE_ARTIFACTS = (
    "run_config_snapshot.json",
    "import_manifest.jsonl",
    "finalize_report.json",
)
REUSABLE_SEARCH_ARTIFACTS = (
    *REUSABLE_ADD_FINALIZE_ARTIFACTS,
    "search_results.json",
)

OPTIONAL_REUSABLE_RUNTIME_ARTIFACTS = (
    "memory_status.json",
    "openclaw_import_summary.json",
    "source_units.jsonl",
)


def deep_merge_config(base: dict, override: dict) -> dict:
    """
    Deep merge configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_config(result[key], value)
        else:
            # Direct override
            result[key] = value
    return result


def build_effective_evaluation_config(dataset_config: dict, system_config: dict) -> dict:
    """Merge optional system-level evaluation overrides onto dataset defaults."""
    evaluation_config = dataset_config.get("evaluation", {}) or {}
    system_evaluation_override = system_config.get("evaluation", {}) or {}
    if not system_evaluation_override:
        return dict(evaluation_config)
    return deep_merge_config(evaluation_config, system_evaluation_override)


def resolve_system_config_path(evaluation_root: Path, system: str) -> Path:
    """Resolve a system YAML under config/systems, including nested folders."""
    system_ref = str(system or "").strip()
    if not system_ref:
        raise ValueError("System config name cannot be empty")

    relative_path = Path(
        system_ref
        if system_ref.endswith((".yaml", ".yml"))
        else f"{system_ref}.yaml"
    )
    if relative_path.is_absolute() or any(
        part in {"..", "."} for part in relative_path.parts
    ):
        raise ValueError(f"Invalid system config path: {system_ref}")

    systems_root = (evaluation_root / "config" / "systems").resolve()
    candidate = (systems_root / relative_path).resolve()
    try:
        candidate.relative_to(systems_root)
    except ValueError as exc:
        raise ValueError(f"Invalid system config path: {system_ref}") from exc
    return candidate


def normalize_evaluation_search_mode(system_config: dict) -> str:
    """Return the evaluation-level search mode.

    Existing adapters may use search.mode for provider-internal modes. Only the
    explicit value "readback" selects the readback evaluation path.
    """
    mode = str((system_config.get("search") or {}).get("mode") or "api").strip().lower()
    return mode if mode in {"api", "readback"} else "api"


def apply_cli_search_mode_override(system_config: dict, search_mode: str | None) -> None:
    """Apply a CLI search-mode override to the loaded system config."""
    if search_mode is None:
        return
    system_config.setdefault("search", {})["mode"] = search_mode


def build_output_paths(
    *,
    evaluation_root: Path,
    dataset: str,
    system: str,
    run_name: str | None,
    output_dir_arg: str | None,
    search_mode: str,
) -> tuple[Path, Path, str]:
    """Build root/mode output paths using results/{base}/{base}-{mode}."""
    if output_dir_arg:
        root_dir = Path(output_dir_arg)
        base_name = root_dir.name
    else:
        system_component = str(system).replace("/", "__").replace("\\", "__")
        run_component = (
            str(run_name or "default").replace("/", "__").replace("\\", "__")
        )
        base_name = f"{dataset}-{system_component}-{run_component}"
        root_dir = evaluation_root / "results" / base_name
    mode_dir = root_dir / f"{base_name}-{search_mode}"
    return root_dir, mode_dir, base_name


def _required_reuse_artifacts_exist(
    source_dir: Path, filenames: tuple[str, ...] = REUSABLE_ADD_FINALIZE_ARTIFACTS
) -> bool:
    return all((source_dir / filename).exists() for filename in filenames)


def _find_sibling_reuse_source(root_dir: Path, current_mode_dir: Path) -> Path | None:
    if not root_dir.exists():
        return None
    candidates = [
        path
        for path in root_dir.iterdir()
        if path.is_dir() and path != current_mode_dir and _required_reuse_artifacts_exist(path)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[0]


def _load_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _write_json_file(path: Path, data: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _load_jsonl_file(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl_file(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_openclaw_component(value: object, fallback: str = "unknown") -> str:
    import re

    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text or fallback


def _openclaw_runtime_info(run_dir: Path) -> dict:
    state_dir = run_dir / "state"
    workspace_dir = run_dir / "workspace"
    return {
        "run_dir": run_dir,
        "state_dir": state_dir,
        "workspace_dir": workspace_dir,
        "memory_dir": workspace_dir / "memory",
        "config_path": state_dir / "openclaw.json",
        "sessions_dir": state_dir / "agents" / "main" / "sessions",
    }


def _discover_openclaw_runtimes(run_root: Path) -> dict[str, dict]:
    conversations_root = run_root / "conversations"
    runtimes: dict[str, dict] = {}
    if conversations_root.exists():
        for run_dir in sorted(conversations_root.iterdir(), key=lambda path: path.name):
            if run_dir.is_dir():
                runtimes[run_dir.name] = _openclaw_runtime_info(run_dir)
    if runtimes:
        return runtimes
    return {"": _openclaw_runtime_info(run_root)}


def _rewrite_openclaw_config(config_path: Path, runtime: dict) -> None:
    if not config_path.exists():
        return
    cfg = _load_json_file(config_path)
    defaults = cfg.setdefault("agents", {}).setdefault("defaults", {})
    defaults["workspace"] = str(runtime["workspace_dir"])
    memory_search = defaults.setdefault("memorySearch", {})
    memory_search["extraPaths"] = [str(runtime["memory_dir"])]
    _write_json_file(config_path, cfg)


def _runtime_payload(runtime: dict) -> dict[str, str]:
    return {
        "state_dir": str(runtime["state_dir"]),
        "workspace_dir": str(runtime["workspace_dir"]),
        "config_path": str(runtime["config_path"]),
        "memory_dir": str(runtime["memory_dir"]),
    }


def _runtime_for_conversation(runtimes: dict[str, dict], conversation_id: str) -> dict:
    key = _safe_openclaw_component(conversation_id)
    return runtimes.get(key) or runtimes.get(conversation_id) or runtimes.get("") or next(
        iter(runtimes.values())
    )


def _rewrite_reused_openclaw_import_manifest(mode_dir: Path, runtimes: dict[str, dict]) -> None:
    manifest_path = mode_dir / "import_manifest.jsonl"
    if not manifest_path.exists():
        return

    rows = _load_jsonl_file(manifest_path)
    for row in rows:
        conversation_id = str(row.get("conversation_id") or "")
        runtime = _runtime_for_conversation(runtimes, conversation_id)
        runtime_payload = _runtime_payload(runtime)
        receipt = row.get("write_receipt")
        if isinstance(receipt, dict):
            memory_name = Path(str(receipt.get("memory_path") or "")).name
            transcript_name = Path(str(receipt.get("transcript_path") or "")).name
            if memory_name:
                receipt["memory_path"] = str(runtime["memory_dir"] / memory_name)
            if transcript_name:
                receipt["transcript_path"] = str(
                    runtime["sessions_dir"] / transcript_name
                )

        refs = row.get("memory_refs")
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            memory_name = Path(
                str(ref.get("path") or ref.get("absolute_path") or "")
            ).name
            if memory_name:
                memory_path = runtime["memory_dir"] / memory_name
                ref["path"] = str(memory_path)
                ref["absolute_path"] = str(memory_path)
            ref["openclaw_isolation"] = "conversation"
            ref["openclaw_runtime"] = runtime_payload

    _write_jsonl_file(manifest_path, rows)


def _rewrite_reused_openclaw_import_summary(
    mode_dir: Path, runtimes: dict[str, dict]
) -> None:
    import_summary_path = mode_dir / "openclaw_import_summary.json"
    if not import_summary_path.exists():
        return
    summary = _load_json_file(import_summary_path)
    if "" not in runtimes:
        summary["isolation"] = "conversation"
        summary["conversation_runtimes"] = {
            conversation_id: _runtime_payload(runtime)
            for conversation_id, runtime in sorted(runtimes.items())
            if conversation_id
        }
    else:
        runtime = _runtime_for_conversation(runtimes, "")
        summary.update(_runtime_payload(runtime))
    _write_json_file(import_summary_path, summary)


def _rewrite_openclaw_status_record(value: object, runtime: dict) -> None:
    if isinstance(value, list):
        for item in value:
            _rewrite_openclaw_status_record(item, runtime)
        return
    if not isinstance(value, dict):
        return
    if "workspaceDir" in value:
        value["workspaceDir"] = str(runtime["workspace_dir"])
    if "dbPath" in value:
        value["dbPath"] = str(runtime["state_dir"] / "memory" / "main.sqlite")
    if "extraPaths" in value:
        value["extraPaths"] = [str(runtime["memory_dir"])]
    status = value.get("status")
    if isinstance(status, dict):
        _rewrite_openclaw_status_record(status, runtime)


def _rewrite_reused_openclaw_memory_status(
    mode_dir: Path, runtimes: dict[str, dict]
) -> None:
    memory_status_path = mode_dir / "memory_status.json"
    if not memory_status_path.exists():
        return
    try:
        status_payload = json.loads(memory_status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    if isinstance(status_payload, dict) and isinstance(
        status_payload.get("conversations"), dict
    ):
        for conversation_id, payload in status_payload["conversations"].items():
            runtime = _runtime_for_conversation(runtimes, str(conversation_id))
            _rewrite_openclaw_status_record(payload, runtime)
    else:
        runtime = _runtime_for_conversation(runtimes, "")
        _rewrite_openclaw_status_record(status_payload, runtime)
    _write_json_file(memory_status_path, status_payload)


def _drop_reused_openclaw_answer_sessions(runtimes: dict[str, dict]) -> None:
    for runtime in runtimes.values():
        sessions_dir = runtime["sessions_dir"]
        if not sessions_dir.exists():
            sessions_dir.mkdir(parents=True, exist_ok=True)
            continue
        for path in sessions_dir.glob("answer-*.jsonl"):
            path.unlink()


def _rewrite_reused_openclaw_runtime_paths(mode_dir: Path) -> None:
    """Point copied OpenClaw runtime configs at the new run directory."""
    mode_dir = mode_dir.resolve(strict=False)
    run_root = mode_dir / "openclaw_runtime"
    runtimes = _discover_openclaw_runtimes(run_root)

    for runtime in runtimes.values():
        _rewrite_openclaw_config(runtime["config_path"], runtime)

    _rewrite_reused_openclaw_import_manifest(mode_dir, runtimes)
    _rewrite_reused_openclaw_import_summary(mode_dir, runtimes)
    _rewrite_reused_openclaw_memory_status(mode_dir, runtimes)
    _drop_reused_openclaw_answer_sessions(runtimes)


def _validate_reuse_source(
    source_dir: Path,
    *,
    dataset_id: str,
    system_id: str,
    smoke_test: bool = False,
    required_artifacts: tuple[str, ...] = REUSABLE_ADD_FINALIZE_ARTIFACTS,
) -> None:
    missing = [
        filename
        for filename in required_artifacts
        if not (source_dir / filename).exists()
    ]
    if missing:
        raise RuntimeError(
            f"Cannot reuse add/finalize artifacts from {source_dir}: missing {', '.join(missing)}"
        )

    snapshot = _load_json_file(source_dir / "run_config_snapshot.json")
    runtime = snapshot.get("runtime", {}) or {}
    snapshot_dataset = snapshot.get("dataset_id") or runtime.get("dataset_id")
    snapshot_system = snapshot.get("system_id") or runtime.get("system_id")
    expected_dataset_ids = {str(dataset_id)}
    if smoke_test:
        expected_dataset_ids.add(f"{dataset_id}_smoke")
    if snapshot_dataset and str(snapshot_dataset) not in expected_dataset_ids:
        raise RuntimeError(
            "Reusable run_config_snapshot.json belongs to a different dataset: "
            f"expected one of {sorted(expected_dataset_ids)!r}, found {snapshot_dataset!r}"
        )
    if snapshot_system and str(snapshot_system) != str(system_id):
        raise RuntimeError(
            "Reusable run_config_snapshot.json belongs to a different system: "
            f"expected {system_id!r}, found {snapshot_system!r}"
        )

    finalize_report = _load_json_file(source_dir / "finalize_report.json")
    finalize_dataset = finalize_report.get("dataset_id")
    finalize_system = finalize_report.get("system_id")
    if finalize_dataset and str(finalize_dataset) not in expected_dataset_ids:
        raise RuntimeError(
            "Reusable finalize_report.json belongs to a different dataset: "
            f"expected one of {sorted(expected_dataset_ids)!r}, found {finalize_dataset!r}"
        )
    if finalize_system and str(finalize_system) != str(system_id):
        raise RuntimeError(
            "Reusable finalize_report.json belongs to a different system: "
            f"expected {system_id!r}, found {finalize_system!r}"
        )


def maybe_reuse_add_finalize_artifacts(
    *,
    root_dir: Path,
    mode_dir: Path,
    explicit_source: str | None,
    dataset_id: str,
    system_id: str,
    smoke_test: bool = False,
    console,
) -> bool:
    """Copy add/finalize artifacts into a new mode dir when a source is available."""
    if mode_dir.exists() and not explicit_source:
        return False

    source_dir = Path(explicit_source) if explicit_source else _find_sibling_reuse_source(root_dir, mode_dir)
    if not source_dir:
        return False

    _validate_reuse_source(
        source_dir,
        dataset_id=dataset_id,
        system_id=system_id,
        smoke_test=smoke_test,
    )
    mode_dir.mkdir(parents=True, exist_ok=True)
    for filename in REUSABLE_ADD_FINALIZE_ARTIFACTS:
        shutil.copy2(source_dir / filename, mode_dir / filename)
    for filename in OPTIONAL_REUSABLE_RUNTIME_ARTIFACTS:
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, mode_dir / filename)
    runtime_source = source_dir / "openclaw_runtime"
    if runtime_source.exists():
        runtime_target = mode_dir / "openclaw_runtime"
        if runtime_target.exists():
            shutil.rmtree(runtime_target)
        shutil.copytree(runtime_source, runtime_target, symlinks=True)
        _rewrite_reused_openclaw_runtime_paths(mode_dir)

    console.print(
        f"[cyan]♻️  Reused add/finalize artifacts from {source_dir}[/cyan]"
    )
    return True


def drop_reused_add_finalize_stages(stages: list[str] | None, reused: bool) -> list[str] | None:
    if not reused:
        return stages
    if stages is None:
        return ["search", "answer", "evaluate"]
    return [stage for stage in stages if stage not in {"add", "finalize"}]


def maybe_reuse_search_artifacts(
    *,
    mode_dir: Path,
    explicit_source: str | None,
    dataset_id: str,
    system_id: str,
    smoke_test: bool = False,
    console,
) -> bool:
    """Copy add/finalize/search artifacts into a mode dir for answer-only runs."""
    if not explicit_source:
        return False

    source_dir = Path(explicit_source)
    _validate_reuse_source(
        source_dir,
        dataset_id=dataset_id,
        system_id=system_id,
        smoke_test=smoke_test,
        required_artifacts=REUSABLE_SEARCH_ARTIFACTS,
    )
    mode_dir.mkdir(parents=True, exist_ok=True)
    for filename in REUSABLE_SEARCH_ARTIFACTS:
        shutil.copy2(source_dir / filename, mode_dir / filename)

    console.print(
        f"[cyan]♻️  Reused add/finalize/search artifacts from {source_dir}[/cyan]"
    )
    return True


def drop_reused_search_stages(stages: list[str] | None, reused: bool) -> list[str] | None:
    if not reused:
        return stages
    if stages is None:
        return ["answer", "evaluate"]
    return [stage for stage in stages if stage not in {"add", "finalize", "search"}]


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Memory System Evaluation Framework")

    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., locomo)"
    )
    parser.add_argument(
        "--system", type=str, required=True, help="System name (e.g., evermemos)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        choices=["add", "finalize", "search", "answer", "evaluate"],
        help="Stages to run. Default: all",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke test mode (process small dataset for quick validation)",
    )
    parser.add_argument(
        "--smoke-messages",
        type=int,
        default=10,
        help="Smoke test: number of messages to process (use 0 for all). Default: 10",
    )
    parser.add_argument(
        "--smoke-questions",
        type=int,
        default=3,
        help="Smoke test: number of questions to test (use 0 for all). Default: 3",
    )
    parser.add_argument(
        "--from-conv",
        type=int,
        default=0,
        help="Starting conversation index to process (inclusive, 0-based). Default: 0",
    )
    parser.add_argument(
        "--to-conv",
        type=int,
        default=None,
        help="Ending conversation index to process (exclusive). Default: None (process all remaining)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name/version for distinguishing multiple runs (e.g., 'v1', 'baseline', '20241104')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory. Actual mode output is {output_dir}/{output_dir.name}-{search_mode}.",
    )
    parser.add_argument(
        "--reuse-add-artifacts-from",
        type=str,
        default=None,
        help="Reuse run_config_snapshot.json, import_manifest.jsonl, and finalize_report.json from this directory.",
    )
    parser.add_argument(
        "--reuse-search-artifacts-from",
        type=str,
        default=None,
        help="Reuse run_config_snapshot.json, import_manifest.jsonl, finalize_report.json, and search_results.json from this directory.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["api", "readback"],
        default=None,
        help="Override system config search.mode for this run.",
    )
    parser.add_argument(
        "--clean-groups",
        action="store_true",
        help="Before Add stage, clear database data for the groups (group_id=conversation_id) involved in this run. "
             "Useful for debugging to avoid polluted data.",
    )

    args = parser.parse_args()
    if args.reuse_add_artifacts_from and args.reuse_search_artifacts_from:
        parser.error(
            "--reuse-add-artifacts-from and --reuse-search-artifacts-from cannot be used together"
        )

    console = get_console()

    # Load configurations
    console.print("\n[bold cyan]Loading configurations...[/bold cyan]")

    evaluation_root = Path(__file__).parent

    # Load dataset configuration
    dataset_config_path = (
        evaluation_root / "config" / "datasets" / f"{args.dataset}.yaml"
    )
    if not dataset_config_path.exists():
        console.print(f"[red]❌ Dataset config not found: {dataset_config_path}[/red]")
        return

    dataset_config = load_yaml(str(dataset_config_path))
    console.print(f"  ✅ Loaded dataset config: {args.dataset}")

    # Override MEMORY_LANGUAGE from dataset config if specified
    if "memory_language" in dataset_config:
        os.environ["MEMORY_LANGUAGE"] = dataset_config["memory_language"]
        console.print(
            f"  🌐 Memory language: {dataset_config['memory_language']} (from dataset config)"
        )

    # Load system configuration
    try:
        system_config_path = resolve_system_config_path(evaluation_root, args.system)
    except ValueError as exc:
        console.print(f"[red]❌ {exc}[/red]")
        return
    if not system_config_path.exists():
        console.print(f"[red]❌ System config not found: {system_config_path}[/red]")
        return

    system_config = load_yaml(str(system_config_path))
    console.print(f"  ✅ Loaded system config: {args.system}")

    # Apply dataset-specific configuration overrides
    if (
        "dataset_overrides" in system_config
        and args.dataset in system_config["dataset_overrides"]
    ):
        overrides = system_config["dataset_overrides"][args.dataset]
        # Deep merge override configurations (supports nested field overrides)
        system_config = deep_merge_config(system_config, overrides)
        console.print(
            f"  🔧 Applied dataset overrides for {args.dataset}: {list(overrides.keys())}"
        )

    apply_cli_search_mode_override(system_config, args.search_mode)
    if args.search_mode:
        console.print(f"  🔎 Overrode search mode from CLI: {args.search_mode}")

    evaluation_config = build_effective_evaluation_config(
        dataset_config, system_config
    )
    if system_config.get("evaluation"):
        console.print(
            f"  ⚖️  Applied system evaluation overrides: {list(system_config['evaluation'].keys())}"
        )

    search_mode = normalize_evaluation_search_mode(system_config)
    system_id = str(system_config.get("name") or args.system)

    # Load dataset
    console.print(f"\n[bold cyan]Loading dataset: {args.dataset}[/bold cyan]")

    data_path = dataset_config["data"]["path"]
    if not Path(data_path).is_absolute():
        # Priority: load from evaluation/data/, fall back to project root
        eval_data_path = evaluation_root / "data" / data_path
        root_data_path = evaluation_root.parent / data_path

        if eval_data_path.exists():
            data_path = eval_data_path
            console.print(f"  📂 Using evaluation/data/{data_path}")
        elif root_data_path.exists():
            data_path = root_data_path
            console.print(f"  📂 Using project root data/{data_path}")
        else:
            console.print(
                f"[red]❌ Data not found in evaluation/data/ or project root data/[/red]"
            )
            return

    # Get max_content_length from dataset config (if specified)
    max_content_length = dataset_config.get("data", {}).get("max_content_length", None)
    if max_content_length:
        console.print(f"  ⚠️  Max content length: {max_content_length} characters")

    # Smart load with auto conversion
    dataset = load_dataset(
        args.dataset, str(data_path), max_content_length=max_content_length
    )

    console.print(
        f"  ✅ Loaded {len(dataset.conversations)} conversations, {len(dataset.qa_pairs)} QA pairs"
    )

    # Determine output directory family and mode-specific directory.
    root_dir, output_dir, base_name = build_output_paths(
        evaluation_root=evaluation_root,
        dataset=args.dataset,
        system=args.system,
        run_name=args.run_name,
        output_dir_arg=args.output_dir,
        search_mode=search_mode,
    )
    requested_stages = list(args.stages) if args.stages else None
    reused_search = maybe_reuse_search_artifacts(
        mode_dir=output_dir,
        explicit_source=args.reuse_search_artifacts_from,
        dataset_id=args.dataset,
        system_id=system_id,
        smoke_test=args.smoke,
        console=console,
    )
    reused_add_finalize = False
    if not reused_search:
        reused_add_finalize = maybe_reuse_add_finalize_artifacts(
            root_dir=root_dir,
            mode_dir=output_dir,
            explicit_source=args.reuse_add_artifacts_from,
            dataset_id=args.dataset,
            system_id=system_id,
            smoke_test=args.smoke,
            console=console,
        )
    requested_stages = drop_reused_search_stages(requested_stages, reused_search)
    requested_stages = drop_reused_add_finalize_stages(
        requested_stages, reused_add_finalize
    )
    if reused_search or reused_add_finalize:
        console.print(
            f"[cyan]   Reuse target stages: {requested_stages or 'default'}[/cyan]"
        )

    # Create components
    console.print(f"\n[bold cyan]Initializing components...[/bold cyan]")

    # Add dataset_name to system_config for adapter initialization
    # (Used to determine num_workers based on adapter + dataset combination)
    system_config["dataset_name"] = args.dataset
    system_config["evaluation_search_mode"] = search_mode
    # Pass CLI switch down to adapter via config (adapters can opt-in)
    system_config["clean_groups"] = bool(args.clean_groups)

    # Create adapter (pass output_dir for persistence)
    adapter = create_adapter(
        system_config["adapter"], system_config, output_dir=output_dir
    )
    console.print(f"  ✅ Created adapter: {adapter.get_system_info()['name']}")

    # Create evaluator
    evaluator = create_evaluator(
        evaluation_config["type"], evaluation_config
    )
    console.print(f"  ✅ Created evaluator: {evaluator.get_name()}")

    # Create LLM Provider for answer generation
    llm_config = system_config.get("llm", {})
    llm_provider = LLMProvider(
        provider_type=llm_config.get("provider", "openai"),
        model=llm_config.get("model"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 32768),
    )
    console.print(f"  Created LLM provider: {llm_config.get('model')}")

    # Create pipeline
    # Read filter categories from dataset configuration
    filter_categories = evaluation_config.get("filter_category", [])

    pipeline = Pipeline(
        adapter=adapter,
        evaluator=evaluator,
        llm_provider=llm_provider,
        output_dir=output_dir,
        run_name=args.run_name or "default",
        filter_categories=filter_categories,
    )

    console.print(f"  ✅ Created pipeline, output: {output_dir}")
    console.print(f"  🔎 Search mode: {search_mode}")
    if filter_categories:
        console.print(f"  📋 Filter categories: {filter_categories}")

    # Run pipeline
    try:
        results = await pipeline.run(
            dataset=dataset,
            stages=requested_stages,
            smoke_test=args.smoke,
            smoke_messages=args.smoke_messages,
            smoke_questions=args.smoke_questions,
            from_conv=args.from_conv,
            to_conv=args.to_conv,
        )

        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")

    finally:
        # Cleanup resources
        # Clean up adapter session (e.g., aiohttp.ClientSession)
        if hasattr(adapter, 'close') and callable(getattr(adapter, 'close')):
            try:
                await adapter.close()
                console.print("[dim]🧹 Cleaned up adapter resources[/dim]")
            except Exception as e:
                # Cleanup failure doesn't affect main process
                console.print(f"[dim]⚠️  Failed to cleanup adapter resources: {e}[/dim]")

        # Only systems using rerank need cleanup
        systems_need_rerank = ["evermemos"]
        if args.system in systems_need_rerank:
            try:
                from agentic_layer import rerank_service

                reranker = rerank_service.get_rerank_service()
                if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
                    await reranker.close()
                    console.print("[dim]🧹 Cleaned up rerank service resources[/dim]")
            except Exception as e:
                # Cleanup failure doesn't affect main process
                console.print(f"[dim]⚠️  Failed to cleanup rerank resources: {e}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
