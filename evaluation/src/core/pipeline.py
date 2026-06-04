"""
Pipeline core module.

Orchestrates the default evaluation workflow:
Add → Finalize → Search → Answer → Evaluate.
"""

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from common_utils.datetime_utils import get_now_with_timezone

from evaluation.src.run_artifacts.artifacts import (
    build_run_config_snapshot,
    build_source_units,
    resolve_qa_evidence,
    source_units_to_rows,
)
from evaluation.src.run_artifacts.models import QAResult, RetrievalArtifact
from evaluation.src.run_artifacts.postprocess import (
    build_evaluation_rows,
    build_score_summary,
)
from evaluation.src.core.data_models import (
    Conversation,
    Dataset,
    SearchResult,
    AnswerResult,
    EvaluationResult,
)
from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.utils.logger import setup_run_logger
from evaluation.src.utils.saver import ResultSaver
from evaluation.src.utils.checkpoint import CheckpointManager

# Import components for answer generation
from memory_layer.llm.llm_provider import LLMProvider

# Import stage execution functions
from evaluation.src.core.stages.add_stage import run_add_stage
from evaluation.src.core.stages.finalize_stage import run_finalize_stage
from evaluation.src.core.stages.search_stage import run_search_stage
from evaluation.src.core.stages.answer_stage import run_answer_stage
from evaluation.src.core.stages.evaluate_stage import run_evaluate_stage


class Pipeline:
    """
    Evaluation Pipeline.

    Default workflow:
    1. Add: Ingest conversation data and build indices
    2. Finalize: Refresh provisional import manifest rows and confirm readiness
    3. Search: Retrieve relevant memories
    4. Answer: Generate answers
    5. Evaluate: Evaluate answer quality

    """

    def __init__(
        self,
        adapter: BaseAdapter,
        evaluator: BaseEvaluator,
        llm_provider: LLMProvider,
        output_dir: Path,
        run_name: str = "default",
        use_checkpoint: bool = True,
        filter_categories: Optional[List[int]] = None,
    ):
        """
        Initialize Pipeline.

        Args:
            adapter: System adapter
            evaluator: Evaluator
            llm_provider: LLM Provider for answer generation
            output_dir: Output directory
            run_name: Run name to distinguish different runs
            use_checkpoint: Enable checkpoint/resume functionality
            filter_categories: List of question categories to filter out (e.g., [5] filters Category 5)
        """
        self.adapter = adapter
        self.evaluator = evaluator
        self.llm_provider = llm_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.run_id = self._load_or_create_run_id()
        self.run_log = setup_run_logger(self.output_dir / "pipeline.log", self.run_id)
        self.logger = self.run_log.logger
        self.console = self.run_log.console
        self.saver = ResultSaver(self.output_dir)

        # Checkpoint/resume support
        self.use_checkpoint = use_checkpoint
        self.checkpoint = (
            CheckpointManager(output_dir=output_dir, run_name=run_name)
            if use_checkpoint
            else None
        )
        self.completed_stages: set = set()

        # Question category filter configuration (read from dataset config)
        self.filter_categories = filter_categories or []

    @staticmethod
    def _default_stages() -> List[str]:
        return [
            "add",
            "finalize",
            "search",
            "answer",
            "evaluate",
        ]

    def _load_or_create_run_id(self) -> str:
        """Load stable run id from snapshot when resuming, otherwise create one."""
        snapshot_path = self.output_dir / "run_config_snapshot.json"
        if snapshot_path.exists():
            try:
                with open(snapshot_path, "r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                existing_run_id = snapshot.get("run_id")
                if existing_run_id:
                    return str(existing_run_id)
            except Exception:
                pass
        return uuid.uuid4().hex

    async def run(
        self,
        dataset: Dataset,
        stages: Optional[List[str]] = None,
        smoke_test: bool = False,
        smoke_messages: int = 10,
        smoke_questions: int = 3,
        from_conv: int = 0,
        to_conv: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run complete Pipeline.

        Args:
            dataset: Standard format dataset
            stages: List of stages to execute, None means all
                   Options: ["add", "finalize", "search", "answer", "evaluate"]
            smoke_test: Enable smoke test mode
            smoke_messages: Number of messages in smoke test (default 10)
            smoke_questions: Number of questions in smoke test (default 3)
            from_conv: Starting conversation index to process (inclusive, 0-based)
            to_conv: Ending conversation index to process (exclusive), None means all

        Returns:
            Evaluation results dictionary
        """
        start_time = time.time()
        if stages is None:
            stages = self._default_stages()
        else:
            stages = list(stages)

        self.console.print(f"\n{'=' * 60}", style="bold cyan")
        self.console.print("🚀 Evaluation Pipeline", style="bold cyan")
        self.console.print(f"{'=' * 60}", style="bold cyan")
        self.console.print(f"Dataset: {dataset.dataset_name}")
        self.console.print(f"System: {self.adapter.get_system_info()['name']}")
        self.console.print(f"Stages: {stages or 'all'}")
        if smoke_test:
            self.console.print(
                f"[yellow]🧪 Smoke Test Mode: {smoke_messages} messages, {smoke_questions} questions[/yellow]"
            )
        self.console.print(f"{'=' * 60}\n", style="bold cyan")

        # Apply conversation range filter (before smoke test)
        # This allows processing a subset of conversations for incremental/distributed testing
        if from_conv > 0 or to_conv is not None:
            dataset = self._apply_conversation_range(dataset, from_conv, to_conv)
            self.console.print(f"[cyan]📌 Conversation Range Filter Applied:[/cyan]")
            self.console.print(
                f"[cyan]   Range: [{from_conv}:{to_conv or 'end'}][/cyan]"
            )
            self.console.print(
                f"[cyan]   Conversations: {len(dataset.conversations)}[/cyan]"
            )
            self.console.print(f"[cyan]   Questions: {len(dataset.qa_pairs)}[/cyan]\n")

        # Smoke test: trim messages and questions for quick validation
        if smoke_test:
            dataset = self._apply_smoke_test(dataset, smoke_messages, smoke_questions)
            self.console.print(f"[yellow]✂️  Smoke test applied:[/yellow]")
            self.console.print(
                f"[yellow]   - Conversations: {len(dataset.conversations)}[/yellow]"
            )
            if len(dataset.conversations) == 0:
                self.console.print(
                    f"[red]   ⚠️  No conversations selected! Check your filters.[/red]"
                )
            elif len(dataset.conversations) == 1:
                self.console.print(
                    f"[yellow]   - Conversation ID: {dataset.conversations[0].conversation_id}[/yellow]"
                )
            else:
                first_id = dataset.conversations[0].conversation_id
                last_id = dataset.conversations[-1].conversation_id
                self.console.print(
                    f"[yellow]   - Range: {first_id} to {last_id}[/yellow]"
                )
            total_messages = sum(len(conv.messages) for conv in dataset.conversations)
            msg_limit = (
                f"max {smoke_messages} per conv" if smoke_messages > 0 else "all"
            )
            qa_limit = (
                f"max {smoke_questions} per conv" if smoke_questions > 0 else "all"
            )
            self.console.print(
                f"[yellow]   - Messages: {total_messages} ({msg_limit})[/yellow]"
            )
            self.console.print(
                f"[yellow]   - Questions: {len(dataset.qa_pairs)} ({qa_limit})[/yellow]\n"
            )

        # Check if we have any conversations to process
        if len(dataset.conversations) == 0:
            self.console.print(
                f"[red]❌ No conversations to process! Check your --from-conv and --to-conv parameters.[/red]"
            )
            self.console.print(
                f"[yellow]💡 Tip: --to-conv should be greater than --from-conv (uses Python slice [from:to))[/yellow]"
            )
            return {
                "error": "No conversations selected",
                "stages_completed": [],
                "total_conversations": 0,
                "total_questions": 0,
            }

        # Filter question categories based on config (e.g., filter out Category 5 adversarial questions)
        original_qa_count = len(dataset.qa_pairs)

        if self.filter_categories:
            # Normalize categories to strings (support both int and str configs)
            filter_set = {str(cat) for cat in self.filter_categories}

            # Filter out specified categories
            dataset.qa_pairs = [
                qa for qa in dataset.qa_pairs if qa.category not in filter_set
            ]

            filtered_count = original_qa_count - len(dataset.qa_pairs)

            if filtered_count > 0:
                filtered_categories_str = ", ".join(sorted(filter_set))
                self.console.print(
                    f"[dim]🔍 Filtered out {filtered_count} questions from categories: {filtered_categories_str}[/dim]"
                )
                self.console.print(
                    f"[dim]   Remaining questions: {len(dataset.qa_pairs)}[/dim]\n"
                )

        system_id = str(
            self.adapter.config.get("name")
            or self.adapter.get_system_info().get("name")
            or self.adapter.__class__.__name__
        )
        run_context = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "vision": None if self.run_name == "default" else self.run_name,
            "dataset_id": dataset.dataset_name,
            "system_id": system_id,
            "benchmark_mode": self.adapter.config.get("benchmark_mode", "category1"),
            "requested_stages": stages,
            "stage_order": self._default_stages(),
            "requires_finalize_report": True,
        }
        self.adapter.set_run_context(run_context)

        source_units, source_unit_map = build_source_units(dataset)
        source_errors = resolve_qa_evidence(dataset, source_unit_map)
        source_unit_rows = source_units_to_rows(source_units)
        self.saver.save_jsonl(source_unit_rows, "source_units.jsonl")
        self.run_log.artifact_written("source_units.jsonl", rows=len(source_unit_rows))
        run_config_snapshot = build_run_config_snapshot(
            dataset=dataset,
            adapter=self.adapter,
            evaluator=self.evaluator,
            runtime={
                **run_context,
                "global_max_concurrency": self.adapter.config.get("num_workers"),
                "answer_num_workers": self.adapter.config.get("answer", {}).get(
                    "num_workers",
                    self.adapter.config.get("answer", {}).get("max_concurrent", 50),
                ),
                "smoke_test": smoke_test,
                "smoke_messages": smoke_messages,
                "smoke_questions": smoke_questions,
                "from_conv": from_conv,
                "to_conv": to_conv,
                "requested_stages": stages,
                "stage_order": self._default_stages(),
                "requires_finalize_report": True,
            },
        )
        self.saver.save_json(run_config_snapshot, "run_config_snapshot.json")
        self.run_log.artifact_written("run_config_snapshot.json")
        results = {
            "source_units": source_unit_rows,
            "run_config_snapshot": run_config_snapshot,
        }

        if source_errors:
            self.logger.warning(
                "Source evidence mapping incomplete for %d QA pairs", len(source_errors)
            )
        # Try loading checkpoint
        search_results_data = None
        answer_results_data = None

        if self.use_checkpoint and self.checkpoint:
            checkpoint_data = self.checkpoint.load_checkpoint()
            if checkpoint_data:
                self.completed_stages = set(checkpoint_data.get("completed_stages", []))
                # Load saved intermediate results
                if "search_results" in checkpoint_data:
                    search_results_data = checkpoint_data["search_results"]
                if "answer_results" in checkpoint_data:
                    answer_results_data = checkpoint_data["answer_results"]

        # Stage 1: Add
        if "add" in stages and "add" not in self.completed_stages:
            self.run_log.stage_start(1, "Add")

            stage_results = await run_add_stage(
                adapter=self.adapter,
                dataset=dataset,
                output_dir=self.output_dir,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
                console=self.console,
                completed_stages=self.completed_stages,
            )
            results.update(stage_results)
            if stage_results.get("import_manifest_records"):
                self.saver.save_jsonl(
                    stage_results["import_manifest_records"], "import_manifest.jsonl"
                )
                self.run_log.artifact_written(
                    "import_manifest.jsonl",
                    rows=len(stage_results["import_manifest_records"]),
                )
                self._hydrate_adapter_namespace_cache(
                    stage_results["import_manifest_records"]
                )
            self.run_log.stage_complete(1, "Add")

        elif "add" in self.completed_stages:
            self.run_log.stage_skip(1, "Add", "already completed")
            # Rebuild index metadata (handled by adapter, only needed for local systems)
            # For online APIs, returns None but still need to set results["index"]
            index = self.adapter.build_lazy_index(
                dataset.conversations, self.output_dir
            )
            results["index"] = index  # Set even if None
            if self.saver.file_exists("import_manifest.jsonl"):
                results["import_manifest_records"] = self._load_saved_import_manifest(
                    dataset=dataset,
                    source_unit_rows=source_unit_rows,
                    system_id=system_id,
                )
            else:
                rebuilt_manifest = self.adapter.get_import_manifest_records()
                if rebuilt_manifest:
                    results["import_manifest_records"] = rebuilt_manifest
                    self.saver.save_jsonl(rebuilt_manifest, "import_manifest.jsonl")
                    self.run_log.artifact_written(
                        "import_manifest.jsonl", rows=len(rebuilt_manifest)
                    )
                    self._hydrate_adapter_namespace_cache(rebuilt_manifest)
        else:
            # Rebuild index metadata (handled by adapter, only needed for local systems)
            # For online APIs, returns None but still need to set results["index"]
            index = self.adapter.build_lazy_index(
                dataset.conversations, self.output_dir
            )
            results["index"] = index  # Set even if None
            if index is not None:
                self.run_log.stage_skip(1, "Add", "using lazy loading")
            if self.saver.file_exists("import_manifest.jsonl"):
                results["import_manifest_records"] = self._load_saved_import_manifest(
                    dataset=dataset,
                    source_unit_rows=source_unit_rows,
                    system_id=system_id,
                )
            else:
                rebuilt_manifest = self.adapter.get_import_manifest_records()
                if rebuilt_manifest:
                    results["import_manifest_records"] = rebuilt_manifest
                    self.saver.save_jsonl(rebuilt_manifest, "import_manifest.jsonl")
                    self.run_log.artifact_written(
                        "import_manifest.jsonl", rows=len(rebuilt_manifest)
                    )
                    self._hydrate_adapter_namespace_cache(rebuilt_manifest)

        # Stage 2: Finalize
        if "import_manifest_records" not in results and self.saver.file_exists(
            "import_manifest.jsonl"
        ):
            results["import_manifest_records"] = self._load_saved_import_manifest(
                dataset=dataset, source_unit_rows=source_unit_rows, system_id=system_id
            )

        if "finalize" in stages and "finalize" not in self.completed_stages:
            self.run_log.stage_start(2, "Finalize")
            import_manifest_rows = results.get("import_manifest_records", [])
            if not import_manifest_rows:
                raise FileNotFoundError(
                    "Finalize requires import_manifest.jsonl. Run '--stages add' first."
                )

            readiness_cfg = self.adapter.config.get("readiness", {}) or {}
            budget_seconds = int(
                readiness_cfg.get(
                    "budget_seconds",
                    self.adapter.config.get("post_add_wait_seconds", 0),
                )
            )
            poll_interval_seconds = readiness_cfg.get(
                "poll_interval_seconds",
                self.adapter.config.get("post_add_poll_interval_seconds"),
            )

            finalize_report = await run_finalize_stage(
                adapter=self.adapter,
                dataset=dataset,
                import_manifest_rows=import_manifest_rows,
                add_result=results.get("add_result"),
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
                budget_seconds=budget_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            finalize_report.setdefault(
                "finalized_at", get_now_with_timezone().isoformat()
            )
            finalize_report.setdefault("run_id", self.run_id)
            finalize_report.setdefault("system_id", system_id)
            finalize_report.setdefault("dataset_id", dataset.dataset_name)
            refreshed_manifest = (
                finalize_report.get("import_manifest_records") or import_manifest_rows
            )
            results["import_manifest_records"] = refreshed_manifest
            results["finalize_report"] = finalize_report
            self.saver.save_jsonl(refreshed_manifest, "import_manifest.jsonl")
            self.saver.save_json(finalize_report, "finalize_report.json")
            self.run_log.artifact_written(
                "import_manifest.jsonl", rows=len(refreshed_manifest)
            )
            self.run_log.artifact_written("finalize_report.json")
            self._hydrate_adapter_namespace_cache(refreshed_manifest)

            if finalize_report.get("ready"):
                self.completed_stages.add("finalize")
                self.run_log.stage_complete(2, "Finalize")
                if self.checkpoint:
                    self.checkpoint.save_checkpoint(self.completed_stages)
        elif "finalize" in self.completed_stages:
            self.run_log.stage_skip(2, "Finalize", "already completed")
            if "finalize_report" not in results and self.saver.file_exists(
                "finalize_report.json"
            ):
                results["finalize_report"] = self._load_saved_finalize_report(
                    dataset=dataset, system_id=system_id
                )
        elif self.saver.file_exists("finalize_report.json"):
            results["finalize_report"] = self._load_saved_finalize_report(
                dataset=dataset, system_id=system_id
            )

        if self._requires_finalize_for_stages(stages):
            self._ensure_finalize_ready(stages=stages, results=results)

        # Stage 3: Search
        if "search" in stages and "search" not in self.completed_stages:
            self.run_log.stage_start(3, "Search")

            search_results = await run_search_stage(
                adapter=self.adapter,
                qa_pairs=dataset.qa_pairs,
                index=results["index"],
                conversations=dataset.conversations,  # Pass conversations for cache rebuilding
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
                import_manifest_records=results.get("import_manifest_records", []),
                run_log=self.run_log,
            )

            self.saver.save_json(
                [self._search_result_to_dict(sr) for sr in search_results],
                "search_results.json",
            )
            self.run_log.artifact_written(
                "search_results.json", rows=len(search_results)
            )
            self._save_readback_search_artifacts(search_results)
            results["search_results"] = search_results
            self.run_log.stage_complete(3, "Search")

            # Save checkpoint
            self.completed_stages.add("search")
            if self.checkpoint:
                search_results_data = [
                    self._search_result_to_dict(sr) for sr in search_results
                ]
                self.checkpoint.save_checkpoint(
                    self.completed_stages, search_results=search_results_data
                )
        elif "search" in self.completed_stages:
            self.run_log.stage_skip(3, "Search", "already completed")
            if search_results_data:
                # Load from checkpoint
                search_results = [
                    self._dict_to_search_result(d) for d in search_results_data
                ]
                results["search_results"] = search_results
            elif self.saver.file_exists("search_results.json"):
                # Load from file
                search_data = self.saver.load_json("search_results.json")
                search_results = [self._dict_to_search_result(d) for d in search_data]
                results["search_results"] = search_results
        elif "answer" in stages or "evaluate" in stages:
            # Only try loading when subsequent stages need search_results
            if self.saver.file_exists("search_results.json"):
                search_data = self.saver.load_json("search_results.json")
                search_results = [self._dict_to_search_result(d) for d in search_data]
                results["search_results"] = search_results
                self.run_log.stage_skip(3, "Search", "loaded existing results")
            else:
                raise FileNotFoundError(
                    "Search results not found. Please run 'search' stage first."
                )
        else:
            # Don't need search_results (e.g., only running add stage)
            search_results = None
        qa_rows: Optional[List[Dict[str, Any]]] = None

        # Stage 4: Answer
        if "answer" in stages and "answer" not in self.completed_stages:
            self.run_log.stage_start(4, "Answer")

            answer_results = await run_answer_stage(
                adapter=self.adapter,
                qa_pairs=dataset.qa_pairs,
                search_results=search_results,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
                run_log=self.run_log,
            )
            search_results = self._maybe_backfill_runtime_recall_search_results(
                answer_results=answer_results,
                search_results=search_results,
            )
            search_results_data = (
                [self._search_result_to_dict(sr) for sr in search_results]
                if search_results is not None
                else []
            )

            self.saver.save_json(
                [self._answer_result_to_dict(ar) for ar in answer_results],
                "answer_results.json",
            )
            self.run_log.artifact_written(
                "answer_results.json", rows=len(answer_results)
            )
            results["answer_results"] = answer_results
            qa_rows = self._build_qa_rows(
                dataset.qa_pairs, answer_results, search_results
            )
            self.saver.save_jsonl(qa_rows, "qa_results.jsonl")
            self.run_log.artifact_written("qa_results.jsonl", rows=len(qa_rows))
            results["qa_results"] = qa_rows
            self.run_log.stage_complete(4, "Answer")

            # Save checkpoint
            self.completed_stages.add("answer")
            if self.checkpoint:
                answer_results_dict = [
                    self._answer_result_to_dict(ar) for ar in answer_results
                ]
                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=search_results_data,
                    answer_results=answer_results_dict,
                )
                # Sync answer_results_data to ensure subsequent stages use correct data
                answer_results_data = answer_results_dict
        elif "answer" in self.completed_stages:
            self.run_log.stage_skip(4, "Answer", "already completed")
            if answer_results_data:
                # Load from checkpoint
                answer_results = [
                    self._dict_to_answer_result(d) for d in answer_results_data
                ]
                results["answer_results"] = answer_results
                qa_rows = self._load_or_build_qa_rows(
                    dataset.qa_pairs, answer_results, search_results
                )
                results["qa_results"] = qa_rows
            elif self.saver.file_exists("answer_results.json"):
                # Load from file
                answer_data = self.saver.load_json("answer_results.json")
                answer_results = [self._dict_to_answer_result(d) for d in answer_data]
                results["answer_results"] = answer_results
                qa_rows = self._load_or_build_qa_rows(
                    dataset.qa_pairs, answer_results, search_results
                )
                results["qa_results"] = qa_rows
        elif "evaluate" in stages:
            # Only try loading when evaluate stage needs answer_results
            if self.saver.file_exists("answer_results.json"):
                answer_data = self.saver.load_json("answer_results.json")
                answer_results = [self._dict_to_answer_result(d) for d in answer_data]
                results["answer_results"] = answer_results
                qa_rows = self._load_or_build_qa_rows(
                    dataset.qa_pairs, answer_results, search_results
                )
                results["qa_results"] = qa_rows
                self.run_log.stage_skip(4, "Answer", "loaded existing results")
            else:
                raise FileNotFoundError(
                    "Answer results not found. Please run 'answer' stage first."
                )
        else:
            # Don't need answer_results (e.g., only running add or search)
            answer_results = None

        # Stage 5: Evaluate
        if "evaluate" in stages and "evaluate" not in self.completed_stages:
            self.run_log.stage_start(5, "Evaluate")
            eval_result = await run_evaluate_stage(
                evaluator=self.evaluator,
                answer_results=answer_results,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
            )

            self.saver.save_json(
                self._eval_result_to_dict(eval_result), "eval_results.json"
            )
            self.run_log.artifact_written("eval_results.json")
            results["eval_result"] = eval_result

            if qa_rows is None and answer_results:
                qa_rows = self._load_or_build_qa_rows(
                    dataset.qa_pairs, answer_results, search_results
                )
                results["qa_results"] = qa_rows

            if qa_rows is not None:
                evaluation_row_map = build_evaluation_rows(
                    qa_rows, eval_result
                )
                evaluation_rows = list(evaluation_row_map.values())
                score_summary = build_score_summary(evaluation_rows)
                self.saver.save_jsonl(evaluation_rows, "evaluation_results.jsonl")
                self.run_log.artifact_written(
                    "evaluation_results.jsonl", rows=len(evaluation_rows)
                )
                self.saver.save_json(score_summary, "score_summary.json")
                self.run_log.artifact_written("score_summary.json")

                results["evaluation_row_map"] = evaluation_row_map
                results["evaluation_results"] = evaluation_rows
                results["score_summary"] = score_summary

            # Save checkpoint
            self.completed_stages.add("evaluate")
            self.run_log.stage_complete(5, "Evaluate")
            if self.checkpoint:
                # Handle None cases for search_results and answer_results
                if search_results:
                    sr_data = [self._search_result_to_dict(sr) for sr in search_results]
                elif search_results_data:
                    sr_data = search_results_data
                else:
                    sr_data = []

                if answer_results_data:
                    ar_data = answer_results_data
                elif answer_results:
                    ar_data = [self._answer_result_to_dict(ar) for ar in answer_results]
                else:
                    ar_data = []

                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=sr_data,
                    answer_results=ar_data,
                    eval_results=self._eval_result_to_dict(eval_result),
                )
        elif "evaluate" in self.completed_stages:
            self.run_log.stage_skip(5, "Evaluate", "already completed")
            score_summary = self._load_or_rebuild_score_summary()
            if score_summary is not None:
                results["score_summary"] = score_summary
            if self.saver.file_exists("eval_results.json"):
                eval_data = self.saver.load_json("eval_results.json")
                eval_result = self._dict_to_eval_result(eval_data)
                results["eval_result"] = eval_result
                if "evaluation_row_map" not in results:
                    if qa_rows is None and answer_results:
                        qa_rows = self._load_or_build_qa_rows(
                            dataset.qa_pairs, answer_results, search_results
                        )
                        results["qa_results"] = qa_rows
                    if qa_rows is not None:
                        evaluation_row_map = build_evaluation_rows(
                            qa_rows, eval_result
                        )
                        evaluation_rows = list(evaluation_row_map.values())
                        results["evaluation_row_map"] = evaluation_row_map
                        results["evaluation_results"] = evaluation_rows

        if "score_summary" not in results:
            score_summary = self._load_or_rebuild_score_summary()
            if score_summary is not None:
                results["score_summary"] = score_summary

        # Generate report
        elapsed_time = time.time() - start_time
        self._generate_report(results, elapsed_time)

        return results

    def _apply_smoke_test(
        self, dataset: Dataset, num_messages: int, num_questions: int
    ) -> Dataset:
        """
        Apply smoke test: trim messages and questions for quick validation.

        This allows quick validation of the complete workflow (Add → Search → Answer → Evaluate)
        using only a small subset of data to save time.

        Strategy:
        - If dataset has multiple conversations (e.g., from conversation range filter):
          Apply smoke limits to ALL conversations in the range
        - If dataset has only one conversation:
          Apply smoke limits to that conversation (legacy behavior)

        Args:
            dataset: Original dataset (may be pre-filtered by conversation range)
            num_messages: Number of messages to keep per conversation (for Add stage), 0 means all
            num_questions: Number of questions to keep per conversation (for Search/Answer/Evaluate stages), 0 means all

        Returns:
            Trimmed dataset
        """
        if not dataset.conversations:
            return dataset

        # Process all conversations (respecting conversation range filter if applied)
        trimmed_conversations = []
        trimmed_qa_pairs = []

        total_messages_before = 0
        total_messages_after = 0
        total_questions_before = 0
        total_questions_after = 0

        for conv in dataset.conversations:
            conv_id = conv.conversation_id

            # Trim questions for this conversation
            conv_qa_pairs = [
                qa
                for qa in dataset.qa_pairs
                if qa.metadata.get("conversation_id") == conv_id
            ]

            if num_questions > 0:
                total_questions_before += len(conv_qa_pairs)
                selected_qa_pairs = conv_qa_pairs[:num_questions]
                total_questions_after += len(selected_qa_pairs)
            else:
                selected_qa_pairs = conv_qa_pairs
                total_questions_after += len(selected_qa_pairs)
                total_questions_before += len(selected_qa_pairs)

            trimmed_qa_pairs.extend(selected_qa_pairs)

            # Trim messages for this conversation. Preserve messages from the
            # selected QA sessions so session-scoped readback smokes are not
            # accidentally emptied by a simple first-N message slice.
            total_messages_before += len(conv.messages)
            if num_messages > 0:
                selected_qa_session_ids = {
                    str(session_id)
                    for qa in selected_qa_pairs
                    for session_id in ((qa.metadata or {}).get("session_ids") or [])
                    if str(session_id)
                }
                selected_messages = []
                seen_message_ids = set()
                for message in list(conv.messages[:num_messages]) + [
                    message
                    for message in conv.messages
                    if str((message.metadata or {}).get("source_session_id") or "")
                    in selected_qa_session_ids
                ]:
                    identity = id(message)
                    if identity in seen_message_ids:
                        continue
                    seen_message_ids.add(identity)
                    selected_messages.append(message)
            else:
                selected_messages = conv.messages
            total_messages_after += len(selected_messages)

            trimmed_conversations.append(
                Conversation(
                    conversation_id=conv.conversation_id,
                    messages=list(selected_messages),
                    metadata=dict(conv.metadata or {}),
                )
            )

        # Log summary
        if len(trimmed_conversations) == 1:
            conv_desc = f"Conv {trimmed_conversations[0].conversation_id}"
        else:
            conv_desc = f"{len(trimmed_conversations)} conversations"

        msg_desc = (
            f"{total_messages_after}/{total_messages_before}"
            if num_messages > 0
            else f"{total_messages_after} (all)"
        )
        qa_desc = (
            f"{total_questions_after}/{total_questions_before}"
            if num_questions > 0
            else f"{total_questions_after} (all)"
        )

        self.logger.info(
            f"Smoke test: {conv_desc} - {msg_desc} messages, {qa_desc} questions"
        )

        return Dataset(
            dataset_name=dataset.dataset_name + "_smoke",
            conversations=trimmed_conversations,
            qa_pairs=trimmed_qa_pairs,
            metadata={
                **dataset.metadata,
                "smoke_test": True,
                "smoke_messages": num_messages,
                "smoke_questions": num_questions,
                "total_conversations": len(trimmed_conversations),
            },
        )

    def _apply_conversation_range(
        self, dataset: Dataset, from_conv: int, to_conv: Optional[int]
    ) -> Dataset:
        """
        Filter conversations by index range.

        This allows processing a subset of conversations for incremental testing
        or distributed processing. The conversation_id attribute of each Conversation
        object remains unchanged, ensuring consistent user_id generation for online APIs.

        Args:
            dataset: Original dataset
            from_conv: Starting conversation index (inclusive, 0-based)
            to_conv: Ending conversation index (exclusive), None means all

        Returns:
            Filtered dataset with selected conversations and their QA pairs

        Example:
            - Original: 100 conversations (locomo_0 to locomo_99)
            - from_conv=10, to_conv=20: select conversations[10:20]
            - Result: 10 conversations (locomo_10 to locomo_19)
            - conversation_id attributes remain: "locomo_10", "locomo_11", ..., "locomo_19"
        """
        if not dataset.conversations:
            return dataset

        # Apply range slicing
        total_convs = len(dataset.conversations)
        end_idx = to_conv if to_conv is not None else total_convs

        # Validation
        if from_conv < 0:
            self.logger.warning(f"from_conv < 0, resetting to 0")
            from_conv = 0
        if from_conv >= total_convs:
            self.logger.warning(
                f"from_conv ({from_conv}) >= total conversations ({total_convs}), no data to process"
            )
            return Dataset(
                dataset_name=dataset.dataset_name,
                conversations=[],
                qa_pairs=[],
                metadata={
                    **dataset.metadata,
                    "conversation_range": [from_conv, end_idx],
                    "original_conversation_count": total_convs,
                    "original_qa_count": len(dataset.qa_pairs),
                },
            )

        # Slice conversations (conversation_id attributes remain unchanged)
        selected_convs = dataset.conversations[from_conv:end_idx]
        selected_conv_ids = {conv.conversation_id for conv in selected_convs}

        # Filter QA pairs for selected conversations
        selected_qa_pairs = [
            qa
            for qa in dataset.qa_pairs
            if qa.metadata.get("conversation_id") in selected_conv_ids
        ]

        self.logger.info(
            f"Conversation range [{from_conv}:{end_idx}] - "
            f"selected {len(selected_convs)}/{total_convs} conversations, "
            f"{len(selected_qa_pairs)}/{len(dataset.qa_pairs)} questions"
        )

        return Dataset(
            dataset_name=dataset.dataset_name,
            conversations=selected_convs,
            qa_pairs=selected_qa_pairs,
            metadata={
                **dataset.metadata,
                "conversation_range": [from_conv, end_idx],
                "original_conversation_count": total_convs,
                "original_qa_count": len(dataset.qa_pairs),
            },
        )

    def _generate_report(self, results: Dict[str, Any], elapsed_time: float):
        """Generate evaluation report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📊 Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        # System information
        system_info = self.adapter.get_system_info()
        report_lines.append(f"System: {system_info['name']}")
        report_lines.append(f"Time Elapsed: {elapsed_time:.2f}s")
        report_lines.append("")

        # Evaluation results
        if "score_summary" in results:
            score_summary = results["score_summary"]
            report_lines.append(
                f"Total Questions: {score_summary.get('total_questions', 0)}"
            )
            report_lines.append(f"Correct: {score_summary.get('correct', 0)}")
            report_lines.append(f"Accuracy: {score_summary.get('accuracy', 0.0):.2%}")
            report_lines.append("")
            category_breakdown = score_summary.get("category_breakdown", {})
            if category_breakdown:
                self._append_accuracy_breakdown_section(
                    report_lines,
                    title="Category Breakdown",
                    breakdown=category_breakdown,
                    label_prefix="Category ",
                )
            self._append_accuracy_breakdown_section(
                report_lines,
                title="Source Breakdown",
                breakdown=score_summary.get("source_breakdown", {}),
            )
            self._append_accuracy_breakdown_section(
                report_lines,
                title="Relation Type Breakdown",
                breakdown=score_summary.get("relation_type_breakdown", {}),
            )
            self._append_accuracy_breakdown_section(
                report_lines,
                title="Relation Subtype Breakdown",
                breakdown=score_summary.get("relation_subtype_breakdown", {}),
            )
            self._append_accuracy_breakdown_section(
                report_lines,
                title="Topic Breakdown",
                breakdown=score_summary.get("topic_breakdown", {}),
            )
            self._append_nested_accuracy_breakdown_section(
                report_lines,
                title="Source x Relation Type Breakdown",
                breakdown=score_summary.get("source_relation_type_breakdown", {}),
            )
            self._append_nested_accuracy_breakdown_section(
                report_lines,
                title="Source x Relation Subtype Breakdown",
                breakdown=score_summary.get("source_relation_subtype_breakdown", {}),
            )
        elif "eval_result" in results:
            eval_result = results["eval_result"]
            report_lines.append(f"Total Questions: {eval_result.total_questions}")
            report_lines.append(f"Correct: {eval_result.correct}")
            report_lines.append(f"Accuracy: {eval_result.accuracy:.2%}")
            report_lines.append("")

        report_lines.append("=" * 60)

        report_text = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        # Print to console
        self.console.print("\n" + report_text, style="bold green")
        self.logger.info(f"Report saved to: {report_path}")

    def _load_or_rebuild_score_summary(self) -> Optional[Dict[str, Any]]:
        """Load score summary or rebuild it from merged evaluation rows."""
        if self.saver.file_exists("score_summary.json"):
            return self.saver.load_json("score_summary.json")

        if not self.saver.file_exists("evaluation_results.jsonl"):
            return None

        evaluation_rows = self.saver.load_jsonl("evaluation_results.jsonl")
        score_summary = build_score_summary(evaluation_rows)
        self.saver.save_json(score_summary, "score_summary.json")
        self.logger.info(
            "Rebuilt missing score_summary.json from evaluation_results.jsonl"
        )
        return score_summary

    @staticmethod
    def _format_accuracy_breakdown_line(
        label: str, counts: Dict[str, Any], *, indent: str = ""
    ) -> str:
        total = int(counts.get("total", 0))
        correct = int(counts.get("correct", 0))
        accuracy = float(counts.get("accuracy", 0.0))
        return f"{indent}- {label}: {correct}/{total} ({accuracy:.2%})"

    @classmethod
    def _append_accuracy_breakdown_section(
        cls,
        report_lines: List[str],
        *,
        title: str,
        breakdown: Dict[str, Any],
        label_prefix: str = "",
    ) -> None:
        """Append a human-readable accuracy breakdown to the report."""
        if not breakdown:
            return

        report_lines.append(title)
        for key, counts in sorted(
            breakdown.items(), key=lambda item: str(item[0])
        ):
            report_lines.append(
                cls._format_accuracy_breakdown_line(
                    f"{label_prefix}{key}", counts
                )
            )
        report_lines.append("")

    @classmethod
    def _append_nested_accuracy_breakdown_section(
        cls, report_lines: List[str], *, title: str, breakdown: Dict[str, Any]
    ) -> None:
        """Append a two-level accuracy breakdown to the report."""
        if not breakdown:
            return

        report_lines.append(title)
        for outer_key, inner_breakdown in sorted(
            breakdown.items(), key=lambda item: str(item[0])
        ):
            report_lines.append(f"- {outer_key}")
            for inner_key, counts in sorted(
                (inner_breakdown or {}).items(), key=lambda item: str(item[0])
            ):
                report_lines.append(
                    cls._format_accuracy_breakdown_line(
                        str(inner_key), counts, indent="  "
                    )
                )
        report_lines.append("")

    @staticmethod
    def _append_breakdown_section(
        report_lines: List[str], *, title: str, breakdown: Dict[str, Any]
    ) -> None:
        """Append a human-readable count breakdown to the report."""
        if not breakdown:
            return

        report_lines.append(title)
        for key, count in sorted(
            breakdown.items(), key=lambda item: (-int(item[1]), str(item[0]))
        ):
            report_lines.append(f"- {key}: {count}")
        report_lines.append("")

    def _requires_finalize_for_stages(self, stages: List[str]) -> bool:
        """Search and answer must only run after a successful finalize."""
        return any(stage in stages for stage in ("search", "answer"))

    def _save_readback_search_artifacts(
        self, search_results: List[SearchResult]
    ) -> None:
        """Persist readback-search audit rows when readback mode was used."""
        rows: List[Dict[str, Any]] = []
        for result in search_results:
            metadata = result.retrieval_metadata or {}
            if metadata.get("search_mode") != "readback":
                continue
            readback = metadata.get("readback", {}) or {}
            if not isinstance(readback, dict):
                readback = {}
            rows.append(
                {
                    "question_id": result.question_id,
                    "conversation_id": result.conversation_id,
                    "search_mode": metadata.get("search_mode", ""),
                    "formatted_context": metadata.get("formatted_context") or "",
                    "retrieved_item_count": len(result.results or []),
                    "session_ids": metadata.get("session_ids", []),
                    "status": readback.get("status", result.retrieval_status),
                    "objects": readback.get("objects", []),
                    "metadata": readback.get("metadata", {}),
                    "errors": readback.get("errors", []),
                }
            )
        if rows:
            self.saver.save_jsonl(rows, "readback_search_readback.jsonl")

    @staticmethod
    def _format_artifact_samples(values: List[str], limit: int = 5) -> str:
        """Format a short sample list for artifact mismatch errors."""
        items = sorted({str(value) for value in values if value})
        if not items:
            return ""
        if len(items) <= limit:
            return ", ".join(items)
        return f"{', '.join(items[:limit])} (+{len(items) - limit} more)"

    @staticmethod
    def _manifest_dataset_id(row: Dict[str, Any]) -> str:
        """Extract dataset id hints from import-manifest rows."""
        namespace_scope = (
            (row.get("write_receipt") or {}).get("namespace_scope")
        ) or {}
        return str(
            row.get("dataset_id") or namespace_scope.get("dataset_id") or ""
        ).strip()

    def _validate_import_manifest_compatibility(
        self,
        import_manifest_rows: List[Dict[str, Any]],
        *,
        dataset: Dataset,
        source_unit_rows: List[Dict[str, Any]],
        system_id: str,
    ) -> None:
        """Fail fast when a saved manifest belongs to another run context."""
        if not import_manifest_rows:
            return

        manifest_system_ids = sorted(
            {
                str(row.get("system_id")).strip()
                for row in import_manifest_rows
                if str(row.get("system_id", "")).strip()
            }
        )
        if manifest_system_ids and set(manifest_system_ids) != {str(system_id)}:
            raise RuntimeError(
                "Existing import_manifest.jsonl belongs to a different system. "
                f"Expected system_id={system_id!r}, found "
                f"{self._format_artifact_samples(manifest_system_ids)}. "
                "Use a clean output directory or rerun '--stages add'."
            )

        manifest_run_ids = sorted(
            {
                str(row.get("run_id")).strip()
                for row in import_manifest_rows
                if str(row.get("run_id", "")).strip()
            }
        )
        if manifest_run_ids and set(manifest_run_ids) != {self.run_id}:
            raise RuntimeError(
                "Existing import_manifest.jsonl belongs to a different run_id. "
                f"Expected run_id={self.run_id!r}, found "
                f"{self._format_artifact_samples(manifest_run_ids)}. "
                "Use a clean output directory or rerun '--stages add'."
            )

        manifest_dataset_ids = sorted(
            {
                self._manifest_dataset_id(row)
                for row in import_manifest_rows
                if self._manifest_dataset_id(row)
            }
        )
        expected_dataset_ids = {dataset.dataset_name}
        if dataset.metadata.get("smoke_test") and dataset.dataset_name.endswith("_smoke"):
            expected_dataset_ids.add(dataset.dataset_name[: -len("_smoke")])
        if manifest_dataset_ids and not set(manifest_dataset_ids).issubset(
            expected_dataset_ids
        ):
            raise RuntimeError(
                "Existing import_manifest.jsonl belongs to a different dataset. "
                f"Expected dataset_id in {sorted(expected_dataset_ids)!r}, found "
                f"{self._format_artifact_samples(manifest_dataset_ids)}. "
                "Use a clean output directory or rerun '--stages add'."
            )

        dataset_conversation_ids = {
            str(conversation.conversation_id)
            for conversation in dataset.conversations
            if conversation.conversation_id
        }
        manifest_conversation_ids = {
            str(row.get("conversation_id"))
            for row in import_manifest_rows
            if row.get("conversation_id")
        }
        unexpected_conversations = sorted(
            manifest_conversation_ids - dataset_conversation_ids
        )
        if unexpected_conversations:
            raise RuntimeError(
                "Existing import_manifest.jsonl does not match the current dataset conversations. "
                "Unexpected conversation_id values: "
                f"{self._format_artifact_samples(unexpected_conversations)}. "
                "Use a clean output directory or rerun '--stages add'."
            )

        current_source_unit_ids = {
            str(row.get("source_unit_id"))
            for row in source_unit_rows
            if row.get("source_unit_id")
        }
        manifest_source_unit_ids = {
            str(source_unit_id)
            for row in import_manifest_rows
            for source_unit_id in (row.get("source_unit_ids") or [])
            if source_unit_id
        }
        if dataset.metadata.get("smoke_test"):
            unexpected_source_unit_ids = sorted(
                current_source_unit_ids - manifest_source_unit_ids
            )
        else:
            unexpected_source_unit_ids = sorted(
                manifest_source_unit_ids - current_source_unit_ids
            )
        if unexpected_source_unit_ids:
            raise RuntimeError(
                "Existing import_manifest.jsonl does not match the current source units. "
                "Unexpected source_unit_ids: "
                f"{self._format_artifact_samples(unexpected_source_unit_ids)}. "
                "Use a clean output directory or rerun '--stages add'."
            )

    def _validate_finalize_report_compatibility(
        self, finalize_report: Dict[str, Any], *, dataset: Dataset, system_id: str
    ) -> None:
        """Fail fast when a saved finalize report belongs to another run context."""
        if not finalize_report:
            return

        mismatches = []
        expected_dataset_ids = {dataset.dataset_name}
        if dataset.metadata.get("smoke_test") and dataset.dataset_name.endswith("_smoke"):
            expected_dataset_ids.add(dataset.dataset_name[: -len("_smoke")])
        for field, expected in (
            ("run_id", self.run_id),
            ("system_id", system_id),
            ("dataset_id", dataset.dataset_name),
        ):
            actual = finalize_report.get(field)
            if actual is None:
                continue
            if field == "dataset_id" and str(actual) in expected_dataset_ids:
                continue
            if str(actual) != str(expected):
                mismatches.append(f"{field}={actual!r} (expected {expected!r})")

        if mismatches:
            raise RuntimeError(
                "Existing finalize_report.json does not match the current run context: "
                + ", ".join(mismatches)
                + ". Use a clean output directory or rerun '--stages finalize'."
            )

    def _load_saved_import_manifest(
        self,
        *,
        dataset: Dataset,
        source_unit_rows: List[Dict[str, Any]],
        system_id: str,
    ) -> List[Dict[str, Any]]:
        """Load import_manifest.jsonl and verify it matches the current run context."""
        rows = self.saver.load_jsonl("import_manifest.jsonl")
        self._validate_import_manifest_compatibility(
            rows,
            dataset=dataset,
            source_unit_rows=source_unit_rows,
            system_id=system_id,
        )
        self._hydrate_adapter_namespace_cache(rows)
        return rows

    def _load_saved_finalize_report(
        self, *, dataset: Dataset, system_id: str
    ) -> Dict[str, Any]:
        """Load finalize_report.json and verify it matches the current run context."""
        report = self.saver.load_json("finalize_report.json")
        self._validate_finalize_report_compatibility(
            report, dataset=dataset, system_id=system_id
        )
        return report

    def _ensure_finalize_ready(
        self, *, stages: List[str], results: Dict[str, Any]
    ) -> None:
        """Block search/answer when manifest or finalize report is not ready."""
        import_manifest_rows = results.get("import_manifest_records", [])
        if not import_manifest_rows:
            raise FileNotFoundError(
                "Search/answer requires import_manifest.jsonl from a prior add run. "
                "Run '--stages add' first."
            )

        finalize_report = results.get("finalize_report")
        if not finalize_report:
            raise FileNotFoundError(
                "Search/answer requires finalize_report.json with ready=true. "
                "Run '--stages finalize search answer evaluate' after add."
            )

        if not finalize_report.get("ready"):
            raise RuntimeError(
                "finalize_report.json exists but ready=false. "
                "Re-run '--stages finalize' until finalize_report.json shows ready=true "
                "before continuing to search/answer."
            )

    # Serialization helper methods
    def _search_result_to_dict(self, sr: SearchResult) -> dict:
        """Convert SearchResult object to dictionary."""
        return {
            "question_id": sr.question_id,
            "query": sr.query,
            "conversation_id": sr.conversation_id,
            "results": sr.results,
            "retrieval_metadata": sr.retrieval_metadata,
            "retrieval_status": sr.retrieval_status,
            "timing_ms": sr.timing_ms,
        }

    def _dict_to_search_result(self, d: dict) -> SearchResult:
        """Convert dictionary to SearchResult object."""
        return SearchResult(
            question_id=d.get("question_id", ""),
            query=d.get("query", ""),
            conversation_id=d.get("conversation_id", ""),
            results=d.get("results", []),
            retrieval_metadata=d.get("retrieval_metadata", {}),
            retrieval_status=d.get("retrieval_status", "ok"),
            timing_ms=d.get("timing_ms", 0.0),
        )

    def _answer_result_to_dict(self, ar: AnswerResult) -> dict:
        """Convert AnswerResult object to dictionary."""
        return {
            "question_id": ar.question_id,
            "question": ar.question,
            "answer": ar.answer,
            "golden_answer": ar.golden_answer,
            "category": ar.category,
            "conversation_id": ar.conversation_id,
            "formatted_context": ar.formatted_context,
            "search_results": ar.search_results,
            "latency_ms": ar.latency_ms,
            "errors": ar.errors,
            "metadata": ar.metadata,
        }

    def _dict_to_answer_result(self, d: dict) -> AnswerResult:
        """Convert dictionary to AnswerResult object."""
        return AnswerResult(
            question_id=d.get("question_id", ""),
            question=d.get("question", ""),
            answer=d.get("answer", ""),
            golden_answer=d.get("golden_answer", ""),
            category=d.get("category"),
            conversation_id=d.get("conversation_id", ""),
            formatted_context=d.get("formatted_context", ""),
            search_results=d.get("search_results", []),
            latency_ms=d.get("latency_ms", 0.0),
            errors=d.get("errors", []),
            metadata=d.get("metadata", {}),
        )

    def _maybe_backfill_runtime_recall_search_results(
        self,
        *,
        answer_results: Optional[List[AnswerResult]],
        search_results: Optional[List[SearchResult]],
    ) -> Optional[List[SearchResult]]:
        """Replace deferred search rows with answer-runtime recall rows if opted in."""
        if not answer_results:
            return search_results
        if not getattr(self.adapter, "uses_runtime_recall_artifacts", False):
            return search_results
        exporter = getattr(self.adapter, "export_runtime_search_results", None)
        if not callable(exporter):
            return search_results

        exported = exporter(answer_results, search_results or [])
        if not exported:
            return search_results
        if isinstance(exported, dict):
            exported_map = {
                str(question_id): result
                for question_id, result in exported.items()
                if isinstance(result, SearchResult)
            }
        else:
            exported_map = {
                result.question_id: result
                for result in exported
                if isinstance(result, SearchResult) and result.question_id
            }
        if not exported_map:
            return search_results

        existing = list(search_results or [])
        existing_ids = {result.question_id for result in existing}
        backfilled: List[SearchResult] = [
            exported_map.get(result.question_id, result) for result in existing
        ]
        for answer in answer_results:
            if answer.question_id in existing_ids:
                continue
            if answer.question_id in exported_map:
                backfilled.append(exported_map[answer.question_id])

        result_map = {result.question_id: result for result in backfilled}
        for answer in answer_results:
            runtime_result = result_map.get(answer.question_id)
            if runtime_result is None:
                continue
            answer.search_results = runtime_result.results
            formatted_context = runtime_result.retrieval_metadata.get(
                "formatted_context", ""
            )
            if formatted_context:
                answer.formatted_context = formatted_context
            metadata = answer.metadata if isinstance(answer.metadata, dict) else {}
            metadata["runtime_recall_backfilled"] = True
            metadata["retrieved_item_count"] = len(runtime_result.results)
            metadata["answer_context_char_count"] = len(answer.formatted_context or "")
            answer.metadata = metadata

        self.saver.save_json(
            [self._search_result_to_dict(sr) for sr in backfilled],
            "search_results.json",
        )
        self._save_readback_search_artifacts(backfilled)
        return backfilled

    def _dict_to_eval_result(self, d: dict) -> EvaluationResult:
        """Convert dictionary to EvaluationResult object."""
        return EvaluationResult(
            total_questions=d.get("total_questions", 0),
            correct=d.get("correct", 0),
            accuracy=d.get("accuracy", 0.0),
            detailed_results=d.get("detailed_results", []),
            metadata=d.get("metadata", {}),
        )

    def _eval_result_to_dict(self, er: EvaluationResult) -> dict:
        """Convert EvaluationResult object to dictionary."""
        return {
            "total_questions": er.total_questions,
            "correct": er.correct,
            "accuracy": er.accuracy,
            "detailed_results": er.detailed_results,
            "metadata": er.metadata,
        }

    def _build_qa_rows(
        self,
        qa_pairs: List[Any],
        answer_results: List[AnswerResult],
        search_results: Optional[List[SearchResult]],
    ) -> List[Dict[str, Any]]:
        """Project search and answer outputs into run QA rows."""
        search_result_map = {
            result.question_id: result for result in search_results or []
        }
        answer_result_map = {
            result.question_id: result for result in answer_results or []
        }

        qa_rows: List[Dict[str, Any]] = []
        for qa in qa_pairs:
            answer_result = answer_result_map.get(qa.question_id)
            if answer_result is None:
                continue
            search_result = search_result_map.get(qa.question_id)
            retrieval_artifact = RetrievalArtifact(
                question_id=qa.question_id,
                query=search_result.query if search_result else qa.question,
                retrieved_items=(search_result.results if search_result else []),
                formatted_context=answer_result.formatted_context,
                retrieval_metadata=(
                    search_result.retrieval_metadata if search_result else {}
                ),
                retrieval_status=(
                    search_result.retrieval_status if search_result else "missing"
                ),
                timing_ms=search_result.timing_ms if search_result else 0.0,
            )
            qa_row = QAResult(
                question_id=qa.question_id,
                question=qa.question,
                golden_answer=qa.answer,
                predicted_answer=answer_result.answer,
                conversation_id=answer_result.conversation_id,
                category=qa.category,
                evidence_source_unit_ids=list(
                    getattr(qa, "evidence_source_unit_ids", [])
                ),
                retrieval_artifact=asdict(retrieval_artifact),
                errors=answer_result.errors,
                latency_ms=answer_result.latency_ms,
                metadata={
                    **answer_result.metadata,
                    **(qa.metadata or {}),
                    "raw_evidence": list(qa.evidence),
                },
            )
            qa_rows.append(asdict(qa_row))
        return qa_rows

    def _load_or_build_qa_rows(
        self,
        qa_pairs: List[Any],
        answer_results: List[AnswerResult],
        search_results: Optional[List[SearchResult]],
    ) -> List[Dict[str, Any]]:
        """Load QA rows if present, otherwise rebuild and persist them."""
        if self.saver.file_exists("qa_results.jsonl"):
            return self.saver.load_jsonl("qa_results.jsonl")
        qa_rows = self._build_qa_rows(qa_pairs, answer_results, search_results)
        self.saver.save_jsonl(qa_rows, "qa_results.jsonl")
        return qa_rows

    def _hydrate_adapter_namespace_cache(
        self, import_manifest_rows: List[Dict[str, Any]]
    ) -> None:
        """Restore adapter namespace cache from saved import manifest rows."""
        namespace_cache = getattr(self.adapter, "_namespace_cache", None)
        if namespace_cache is None:
            return
        for row in import_manifest_rows:
            namespace_scope = (
                (row.get("write_receipt") or {}).get("namespace_scope")
            ) or {}
            namespace_id = namespace_scope.get("namespace_id")
            conversation_id = row.get("conversation_id")
            view_id = (
                namespace_scope.get("view_id") or row.get("view_id") or "speaker_a"
            )
            if conversation_id and namespace_id:
                namespace_cache[(conversation_id, str(view_id))] = namespace_id
                if str(view_id) == "shared":
                    namespace_cache[(conversation_id, "speaker_a")] = namespace_id
        self.adapter._namespace_cache = namespace_cache
