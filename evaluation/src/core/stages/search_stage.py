"""
Search stage - retrieve relevant memories.
"""
import asyncio
import time
from typing import List, Any, Optional, Dict
from logging import Logger
from tqdm import tqdm

from evaluation.src.core.data_models import QAPair, SearchResult
from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager


EVALUATION_SEARCH_MODES = {"api", "readback"}


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _search_error_message(result: SearchResult) -> Optional[str]:
    metadata = result.retrieval_metadata if isinstance(result.retrieval_metadata, dict) else {}
    error = metadata.get("error")
    status = str(result.retrieval_status or "").strip().lower()
    if status == "error" or error:
        detail = error or result.retrieval_status
        return (
            f"Search failed for question_id={result.question_id!r} "
            f"conversation_id={result.conversation_id!r}: {detail}"
        )
    return None


def normalize_evaluation_search_mode(search_cfg: Dict[str, Any]) -> str:
    """Return the evaluation-level search mode.

    Existing adapters may already use ``search.mode`` for provider-internal
    choices such as "agentic". Only the explicit value "readback" switches the
    pipeline away from the adapter's normal API search path.
    """
    mode = str((search_cfg or {}).get("mode") or "api").strip().lower()
    return mode if mode in EVALUATION_SEARCH_MODES else "api"


async def run_search_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    index: Any,
    conversations: List,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    import_manifest_records: Optional[List[Dict[str, Any]]] = None,
    run_log: Optional[Any] = None,
) -> List[SearchResult]:
    """
    Execute concurrent search with fine-grained checkpointing.
    
    Process by conversation groups, save checkpoint after each conversation.
    
    Args:
        adapter: System adapter
        qa_pairs: List of QA pairs
        index: Index
        conversations: Conversation list (for online API cache rebuild)
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        
    Returns:
        List of search results
    """
    log_event = run_log.event if run_log is not None else logger.info
    
    # Load fine-grained checkpoint
    all_search_results_dict = {}
    if checkpoint_manager:
        all_search_results_dict = checkpoint_manager.load_search_progress()
    
    # Group QA pairs by conversation
    conv_to_qa = {}
    for qa in qa_pairs:
        conv_id = qa.metadata.get("conversation_id", "unknown")
        if conv_id not in conv_to_qa:
            conv_to_qa[conv_id] = []
        conv_to_qa[conv_id].append(qa)
    
    total_convs = len(conv_to_qa)
    processed_convs = set(all_search_results_dict.keys())
    remaining_convs = set(conv_to_qa.keys()) - processed_convs
    
    log_event(f"Search total conversations={total_convs}")
    log_event(f"Search total questions={len(qa_pairs)}")
    if processed_convs:
        log_event(
            "Search checkpoint already processed conversations=%d",
            len(processed_convs),
        )
        log_event("Search remaining conversations=%d", len(remaining_convs))
    
    # Build conversation_id to conversation mapping (for online API cache rebuild)
    conv_id_to_conv = {conv.conversation_id: conv for conv in conversations}
    
    # Search-stage concurrency can be configured separately via system config:
    #   search.num_workers (fallback to adapter.num_workers, then 20)
    search_cfg = adapter.config.get("search", {})
    search_mode = normalize_evaluation_search_mode(search_cfg)
    fail_on_error = _config_bool(search_cfg.get("fail_on_error", False))
    num_workers = int(search_cfg.get("num_workers", getattr(adapter, "num_workers", 20)))
    semaphore = asyncio.Semaphore(num_workers)
    log_event(f"Search concurrency={num_workers} workers")
    log_event(f"Search mode={search_mode}")
    if fail_on_error:
        log_event("Search error policy=fail_on_error")
    
    # Create fine-grained progress bar (track by questions)
    total_questions = len(qa_pairs)
    processed_questions = sum(len(all_search_results_dict.get(conv_id, [])) for conv_id in processed_convs)
    
    pbar = tqdm(
        total=total_questions,
        initial=processed_questions,
        desc="🔍 Search Progress",
        unit="qa"
    )
    
    async def search_single_with_tracking(qa):
        async with semaphore:
            conv_id = qa.metadata.get("conversation_id", "0")
            conversation = conv_id_to_conv.get(conv_id)
            started_at = time.perf_counter()
            
            # Search with timeout and retry (similar to answer_stage.py)
            max_retries = 3
            timeout_seconds = 300.0  # Increased from 120s for complex agentic retrieval
            result = None
            
            for attempt in range(max_retries):
                try:
                    if search_mode == "readback":
                        search_coro = adapter.search_from_readback(
                            qa.question,
                            conv_id,
                            index,
                            conversation=conversation,
                            question_id=qa.question_id,
                            question_metadata=qa.metadata,
                            import_manifest_records=import_manifest_records or [],
                        )
                    else:
                        search_coro = adapter.search(
                            qa.question,
                            conv_id,
                            index,
                            conversation=conversation,
                            question_id=qa.question_id,
                            question_metadata=qa.metadata,
                            import_manifest_records=import_manifest_records or [],
                        )
                    result = await asyncio.wait_for(search_coro, timeout=timeout_seconds)
                    if not result.question_id:
                        result.question_id = qa.question_id
                    if not isinstance(result.retrieval_metadata, dict):
                        result.retrieval_metadata = {}
                    result.retrieval_metadata.setdefault("search_mode", search_mode)
                    break  # Success, exit retry loop
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        message = f"Search timeout ({timeout_seconds}s) for question in {conv_id}, retry {attempt + 1}/{max_retries}"
                        tqdm.write(f"  {message}...")
                        logger.warning(message)
                        await asyncio.sleep(2)  # Short delay before retry
                    else:
                        message = f"Search timeout after {max_retries} attempts for question in {conv_id}: {qa.question[:60]}"
                        tqdm.write(f"  {message}...")
                        logger.error(message)
                        # Return empty search result on timeout
                        from evaluation.src.core.data_models import SearchResult
                        result = SearchResult(
                            question_id=qa.question_id,
                            query=qa.question,
                            conversation_id=conv_id,
                            results=[],
                            retrieval_metadata={
                                "error": "Search timeout after retries",
                                "search_mode": search_mode,
                            },
                            retrieval_status="error",
                            timing_ms=(time.perf_counter() - started_at) * 1000,
                        )
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        message = f"Search failed for question in {conv_id}: {str(e)}, retry {attempt + 1}/{max_retries}"
                        tqdm.write(f"  {message}...")
                        logger.warning(message)
                        await asyncio.sleep(2)
                    else:
                        message = f"Search failed after {max_retries} attempts for question in {conv_id}: {str(e)}"
                        tqdm.write(f"  {message}")
                        logger.error(message)
                        # Return empty search result on error
                        from evaluation.src.core.data_models import SearchResult
                        result = SearchResult(
                            question_id=qa.question_id,
                            query=qa.question,
                            conversation_id=conv_id,
                            results=[],
                            retrieval_metadata={
                                "error": f"Search error: {str(e)}",
                                "search_mode": search_mode,
                            },
                            retrieval_status="error",
                            timing_ms=(time.perf_counter() - started_at) * 1000,
                        )

            if result and not result.timing_ms:
                result.timing_ms = (time.perf_counter() - started_at) * 1000
            if result and not result.question_id:
                result.question_id = qa.question_id
            if result and fail_on_error:
                error_message = _search_error_message(result)
                if error_message:
                    raise RuntimeError(error_message)
            
            pbar.update(1)  # Update progress bar after each question
            return result
    
    # Process by conversation (use numeric sort for conversation IDs like "longmemeval_10")
    def sort_key(item):
        """Sort by numeric part of conversation_id if possible, else alphabetically."""
        conv_id = item[0]
        # Try to extract numeric suffix (e.g., "longmemeval_10" -> 10)
        parts = conv_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (conv_id, 0)
    
    for idx, (conv_id, qa_list) in enumerate(sorted(conv_to_qa.items(), key=sort_key)):
        # Skip already processed conversations
        if conv_id in processed_convs:
            message = f"Skipping Conversation ID: {conv_id} (already processed)"
            tqdm.write(message)
            logger.info(message)
            continue
        
        message = f"Processing Conversation ID: {conv_id} ({idx+1}/{total_convs}) - {len(qa_list)} questions"
        tqdm.write(message)
        logger.info(message)
        
        # Process all questions for this conversation concurrently
        tasks = [search_single_with_tracking(qa) for qa in qa_list]
        results_for_conv = await asyncio.gather(*tasks)
        
        # Save results in dict format
        results_for_conv_dict = [
            {
                "question_id": qa.question_id,
                "query": qa.question,
                "conversation_id": conv_id,
                "results": result.results,
                "retrieval_metadata": result.retrieval_metadata,
                "retrieval_status": result.retrieval_status,
                "timing_ms": result.timing_ms,
            }
            for qa, result in zip(qa_list, results_for_conv)
        ]
        
        all_search_results_dict[conv_id] = results_for_conv_dict
        
        # Save checkpoint after each conversation
        if checkpoint_manager:
            checkpoint_manager.save_search_progress(all_search_results_dict)
    
    # Close progress bar
    pbar.close()
    
    # Delete fine-grained checkpoint after completion
    if checkpoint_manager:
        checkpoint_manager.delete_search_checkpoint()
    
    # Convert dict format to SearchResult object list (maintain original return format)
    # Use same numeric sort as above to ensure consistent ordering
    def sort_key_conv_id(conv_id):
        """Sort by numeric part of conversation_id if possible, else alphabetically."""
        parts = conv_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (conv_id, 0)
    
    search_result_map = {}
    for conv_id in sorted(conv_to_qa.keys(), key=sort_key_conv_id):
        if conv_id in all_search_results_dict:
            for result_dict in all_search_results_dict[conv_id]:
                search_result_map[result_dict["question_id"]] = SearchResult(
                    question_id=result_dict["question_id"],
                    query=result_dict["query"],
                    conversation_id=result_dict["conversation_id"],
                    results=result_dict["results"],
                    retrieval_metadata=result_dict.get("retrieval_metadata", {}),
                    retrieval_status=result_dict.get("retrieval_status", "ok"),
                    timing_ms=result_dict.get("timing_ms", 0.0),
                )

    all_results = [
        search_result_map[qa.question_id]
        for qa in qa_pairs
        if qa.question_id in search_result_map
    ]
    
    log_event(f"Search completed results={len(all_results)}")
    return all_results
