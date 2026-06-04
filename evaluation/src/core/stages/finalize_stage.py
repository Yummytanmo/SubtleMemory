"""
Finalize stage - refresh provisional import manifest rows into finalized state.
"""
from logging import Logger
from typing import Any, Dict, List, Optional

from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.core.data_models import Dataset
from evaluation.src.utils.checkpoint import CheckpointManager


async def run_finalize_stage(
    adapter: BaseAdapter,
    dataset: Dataset,
    import_manifest_rows: List[Dict[str, Any]],
    add_result: Any,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    budget_seconds: int,
    poll_interval_seconds: Optional[float],
) -> Dict[str, Any]:
    """
    Execute Finalize stage.

    Args:
        adapter: System adapter
        dataset: Dataset for adapter-side reconciliation if needed
        import_manifest_rows: Provisional import manifest rows
        add_result: Raw add result from current or previous run
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        budget_seconds: Finalize budget
        poll_interval_seconds: Finalize poll interval

    Returns:
        Finalize report dict
    """
    logger.info("Starting Stage 2: Finalize")

    finalize_report = await adapter.finalize_imports(
        import_manifest_rows,
        add_result=add_result,
        budget_seconds=budget_seconds,
        poll_interval_seconds=poll_interval_seconds,
        dataset=dataset,
        checkpoint_manager=checkpoint_manager,
    )

    if finalize_report.get("ready"):
        logger.info("✅ Stage 2 completed")
    else:
        logger.warning(
            "⚠️ Stage 2 incomplete - finalize ready=%s status=%s",
            finalize_report.get("ready"),
            finalize_report.get("status"),
        )

    return finalize_report
