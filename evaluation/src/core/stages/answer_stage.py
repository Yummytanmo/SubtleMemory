"""
Answer stage - generate answers.
"""

import asyncio
import time
from typing import List, Optional
from logging import Logger
from tqdm import tqdm

from evaluation.src.core.data_models import QAPair, SearchResult, AnswerResult
from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager


def build_context(search_result: SearchResult) -> str:
    """
    Build context from search results.

    Prefer pre-formatted context (dual-speaker scenarios), else use simple numbering (single-speaker scenarios).

    Args:
        search_result: Search result

    Returns:
        Context string
    """
    # Prefer pre-formatted context (provided by adapter)
    formatted_context = search_result.retrieval_metadata.get("formatted_context", "")
    if formatted_context:
        return formatted_context

    # Single speaker scenario: simple formatting
    context_parts = []

    # Get top_k from retrieval_metadata, default to len(results) if not specified
    top_k = search_result.retrieval_metadata.get("top_k", len(search_result.results))

    # Add memory content (use top_k instead of hardcoded 10)
    for idx, result in enumerate(search_result.results[:top_k], 1):
        content = result.get("content", "")
        context_parts.append(f"{idx}. {content}")

    context = "\n\n".join(context_parts)

    # For systems supporting preferences (e.g., Memos), add formatted pref_string
    preferences = search_result.retrieval_metadata.get("preferences", {})
    pref_string = preferences.get("pref_string", "")

    if pref_string:
        context += "\n\n" + pref_string

    return context


def _is_successful_answer_result(result: dict) -> bool:
    """Return True only for reusable answer-stage checkpoint rows."""
    answer = str(result.get("answer") or "").strip()
    if not answer or answer.startswith("Error:"):
        return False
    return not bool(result.get("errors"))


def _pop_runtime_recall_payload(
    adapter: BaseAdapter, question_id: str
) -> dict:
    popper = getattr(adapter, "pop_runtime_recall_payload", None)
    if not callable(popper):
        popper = getattr(adapter, "pop_runtime_logger_payload", None)
    if not callable(popper):
        return {}
    try:
        payload = popper(question_id)
    except TypeError:
        payload = popper()
    return payload if isinstance(payload, dict) else {}


_RELEASE_METADATA_DENYLIST = {
    "answer_prompt",
    "answer_prompt_available",
    "answer_context_char_count",
    "answer_query",
    "answer_fact_extraction_prompt",
    "answer_fact_answer_prompt",
    "runtime_llm_input",
    "runtime_llm_output",
    "runtime_agent_end_messages",
    "runtime_prompt_messages",
    "runtime_raw_artifact_refs",
}


def _release_answer_metadata(metadata: dict) -> dict:
    """Drop review-only fields before persisting answer artifacts."""
    return {
        key: value
        for key, value in (metadata or {}).items()
        if key not in _RELEASE_METADATA_DENYLIST
    }


def _log_info(logger: Optional[Logger], message: str, *args) -> None:
    if logger is not None:
        logger.info(message, *args)


def _log_warning(logger: Optional[Logger], message: str, *args) -> None:
    if logger is not None:
        logger.warning(message, *args)


def _log_error(logger: Optional[Logger], message: str, *args) -> None:
    if logger is not None:
        logger.error(message, *args)


async def run_answer_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    search_results: List[SearchResult],
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    run_log: Optional[object] = None,
) -> List[AnswerResult]:
    """
    Generate answers with fine-grained checkpointing.

    Save checkpoint every SAVE_INTERVAL questions.

    Args:
        adapter: System adapter
        qa_pairs: List of QA pairs
        search_results: List of search results
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger

    Returns:
        List of answer results
    """
    log_event = run_log.event if run_log is not None else (
        lambda message, *args: _log_info(logger, message, *args)
    )

    answer_cfg = adapter.config.get("answer", {})
    SAVE_INTERVAL = int(answer_cfg.get("checkpoint_interval", 400))
    MAX_CONCURRENT = int(
        answer_cfg.get("num_workers", answer_cfg.get("max_concurrent", 50))
    )

    # Load fine-grained checkpoint
    all_answer_results = {}
    if checkpoint_manager:
        loaded_results = checkpoint_manager.load_answer_progress()
        # Convert to {question_id: AnswerResult} format
        for result in loaded_results.values():
            if _is_successful_answer_result(result):
                all_answer_results[result["question_id"]] = result

    total_qa_count = len(qa_pairs)
    processed_count = len(all_answer_results)

    log_event(f"Answer total questions={total_qa_count}")
    if processed_count > 0:
        log_event(
            "Answer checkpoint already processed questions=%d",
            processed_count,
        )
        log_event("Answer remaining questions=%d", total_qa_count - processed_count)

    search_result_map = {result.question_id: result for result in search_results or []}

    # Prepare pending tasks
    pending_tasks = []
    for qa in qa_pairs:
        if qa.question_id not in all_answer_results:
            search_result = search_result_map.get(qa.question_id)
            if search_result is None:
                search_result = SearchResult(
                    question_id=qa.question_id,
                    query=qa.question,
                    conversation_id=qa.metadata.get("conversation_id", ""),
                    results=[],
                    retrieval_metadata={"error": "Missing search result"},
                    retrieval_status="error",
                    timing_ms=0.0,
                )
            pending_tasks.append((qa, search_result))

    if not pending_tasks:
        log_event("Answer all questions already processed")
        # Convert to AnswerResult object list (original order)
        results = []
        for qa in qa_pairs:
            if qa.question_id in all_answer_results:
                result_dict = all_answer_results[qa.question_id]
                results.append(
                    AnswerResult(
                        question_id=result_dict["question_id"],
                        question=result_dict["question"],
                        answer=result_dict["answer"],
                        golden_answer=result_dict["golden_answer"],
                        category=result_dict.get("category"),
                        conversation_id=result_dict.get("conversation_id", ""),
                        formatted_context=result_dict.get(
                            "formatted_context", ""
                        ),  # Load formatted_context
                        search_results=result_dict.get("search_results", []),
                        latency_ms=result_dict.get("latency_ms", 0.0),
                        errors=result_dict.get("errors", []),
                        metadata=result_dict.get("metadata", {}),
                    )
                )
        return results

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    completed = processed_count
    failed = 0
    start_time = time.time()

    # Use tqdm progress bar
    pbar = tqdm(
        total=total_qa_count,
        initial=processed_count,
        desc="💬 Answer Progress",
        unit="qa",
    )

    async def answer_single_with_tracking(qa, search_result):
        nonlocal completed, failed

        async with semaphore:
            started_at = time.perf_counter()
            errors = []
            try:
                # Build context
                context = build_context(search_result)

                # Detect multiple-choice and enhance question if needed
                query = qa.question
                if "all_options" in qa.metadata:
                    options = qa.metadata["all_options"]
                    options_text = "\n".join(
                        [f"{key} {value}" for key, value in options.items()]
                    )

                    # Integrate options and requirements into question
                    query = f"""{qa.question}

OPTIONS:
{options_text}

IMPORTANT: This is a multiple-choice question. You MUST analyze the context and select the BEST option. In your FINAL ANSWER, return ONLY the option letter like (a), (b), (c), or (d), nothing else."""

                # Call adapter's answer method with timeout and retry
                max_retries = int(answer_cfg.get("max_retries", 3))
                timeout_seconds = float(answer_cfg.get("timeout_seconds", 120.0))
                answer = None

                for attempt in range(max_retries):
                    try:
                        answer = await asyncio.wait_for(
                            adapter.answer(
                                query=query,
                                context=context,
                                conversation_id=search_result.conversation_id,
                                question_id=qa.question_id,
                                search_results=search_result.results,
                                retrieval_metadata=search_result.retrieval_metadata,
                            ),
                            timeout=timeout_seconds,
                        )
                        answer = answer.strip()
                        break  # Success, exit retry loop

                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            message = f"Answer timeout ({timeout_seconds:.0f}s) for {qa.question_id}, retry {attempt + 1}/{max_retries}"
                            tqdm.write(f"  {message}...")
                            _log_warning(logger, message)
                            await asyncio.sleep(2)  # Short delay before retry
                        else:
                            message = f"Answer timeout after {max_retries} attempts for {qa.question_id}: {qa.question[:50]}"
                            tqdm.write(f"  {message}...")
                            _log_error(logger, message)
                            answer = "Error: Answer generation timeout after retries"
                            errors.append(
                                {
                                    "stage": "answer",
                                    "error_type": "timeout",
                                    "error_message": "Answer generation timeout after retries",
                                    "attempt": attempt + 1,
                                }
                            )
                            failed += 1

            except Exception as e:
                message = f"Answer generation failed for {qa.question_id}: {e}"
                tqdm.write(f"  {message}")
                _log_error(logger, message)
                answer = "Error: Failed to generate answer"
                errors.append(
                    {
                        "stage": "answer",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "attempt": 1,
                    }
                )
                failed += 1

            runtime_recall_payload = _pop_runtime_recall_payload(
                adapter, qa.question_id
            )
            answer_trace = adapter.consume_answer_trace(
                question_id=qa.question_id,
                conversation_id=search_result.conversation_id,
            ) or {}

            result = AnswerResult(
                question_id=qa.question_id,
                question=qa.question,
                answer=answer,
                golden_answer=qa.answer,
                category=qa.category,
                conversation_id=search_result.conversation_id,
                formatted_context=context,  # Save actual context used
                search_results=search_result.results,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                errors=errors,
                metadata={
                    **_release_answer_metadata(
                        {
                            **qa.metadata,
                            "retrieved_item_count": len(search_result.results),
                            **runtime_recall_payload,
                            **answer_trace,
                        }
                    ),
                },
            )

            # Save result
            all_answer_results[qa.question_id] = {
                "question_id": result.question_id,
                "question": result.question,
                "answer": result.answer,
                "golden_answer": result.golden_answer,
                "category": result.category,
                "conversation_id": result.conversation_id,
                "formatted_context": result.formatted_context,  # Save formatted_context
                "search_results": result.search_results,
                "latency_ms": result.latency_ms,
                "errors": result.errors,
                "metadata": result.metadata,  # Save metadata (contains all_options)
            }

            completed += 1
            pbar.update(1)  # Update progress bar

            # Save checkpoint periodically
            if checkpoint_manager and (
                completed % SAVE_INTERVAL == 0 or completed == total_qa_count
            ):
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                eta = (total_qa_count - completed) / speed if speed > 0 else 0

                progress_message = (
                    f"Progress: {completed}/{total_qa_count} ({completed/total_qa_count*100:.1f}%) | "
                    f"Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min"
                )
                tqdm.write(progress_message)
                _log_info(logger, progress_message)

                checkpoint_manager.save_answer_progress(
                    all_answer_results, completed, total_qa_count
                )

            return result

    # Create all pending tasks
    tasks = [answer_single_with_tracking(qa, sr) for qa, sr in pending_tasks]

    # Execute concurrently
    await asyncio.gather(*tasks)

    # Close progress bar
    pbar.close()

    # Statistics
    elapsed_time = time.time() - start_time
    success_rate = (completed - failed) / completed * 100 if completed > 0 else 0
    avg_speed = total_qa_count / elapsed_time if elapsed_time > 0 else 0.0

    log_event(
        "Answer completed total=%d successful=%d failed=%d success_rate=%.1f%% elapsed_seconds=%.0f avg_speed=%.1f qa/s",
        total_qa_count,
        completed - failed,
        failed,
        success_rate,
        elapsed_time,
        avg_speed,
    )

    # Delete fine-grained checkpoints after completion
    if checkpoint_manager:
        checkpoint_manager.delete_answer_checkpoints()

    # Convert to AnswerResult object list (original order)
    results = []
    for qa in qa_pairs:
        if qa.question_id in all_answer_results:
            result_dict = all_answer_results[qa.question_id]
            results.append(
                AnswerResult(
                    question_id=result_dict["question_id"],
                    question=result_dict["question"],
                    answer=result_dict["answer"],
                    golden_answer=result_dict["golden_answer"],
                    category=result_dict.get("category"),
                    conversation_id=result_dict.get("conversation_id", ""),
                    formatted_context=result_dict.get("formatted_context", ""),
                    search_results=result_dict.get("search_results", []),
                    latency_ms=result_dict.get("latency_ms", 0.0),
                    errors=result_dict.get("errors", []),
                    metadata=result_dict.get("metadata", {}),  # Restore metadata
                )
            )

    return results
