"""
Online API Adapter base class.

Provides common functionality for all online memory system APIs (Mem0, Memos, Memu, etc.).
All online API adapters can inherit from this class.

Design principles:
- Provide default answer() implementation (using generic prompt)
- Subclasses can override answer() to use their own specific prompts
- Provide helper methods for data format conversion
"""

from __future__ import annotations

import json
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Dict, Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from evaluation.src.adapters.core.base import BaseAdapter
from evaluation.src.run_artifacts.models import (
    ImportManifestRecord,
    WriteReceipt,
    dataclass_to_dict,
)
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.utils.answer_cleaner import clean_answer_text
from evaluation.src.utils.config import load_yaml

# Import Memory Layer components
from memory_layer.llm.llm_provider import LLMProvider


class OnlineAPIAdapter(BaseAdapter):
    """
    Online API Adapter base class.

    Provides common functionality:
    1. LLM Provider initialization
    2. Answer generation (reuses EverMemOS implementation)
    3. Standard format conversion helper methods

    Subclasses only need to implement:
    - add(): Call online API to ingest data
    - search(): Call online API for retrieval
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self._import_manifest_records: List[Dict[str, Any]] = []
        self._incremental_manifest_by_chunk_id: Dict[str, Dict[str, Any]] = {}
        self._namespace_cache: Dict[tuple[str, str], str] = {}
        self._last_add_result: Dict[str, Any] = {}

        # Initialize LLM Provider (for answer generation)
        llm_config = config.get("llm", {})

        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 16384),
        )

        # Load prompts (from YAML file)
        evaluation_root = Path(__file__).resolve().parents[3]
        prompts_path = evaluation_root / "config" / "prompts.yaml"
        self._prompts = load_yaml(str(prompts_path))

        # Set num_workers (conversation-level concurrency)
        # Can be overridden by subclass or config
        self.num_workers = self._get_num_workers(config)

        print(f"✅ {self.__class__.__name__} initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")
        print(f"   Num Workers: {self.num_workers}")

    def _get_num_workers(self, config: dict) -> int:
        """
        Get num_workers from config.

        Args:
            config: Configuration dict (should contain num_workers)

        Returns:
            Number of workers for conversation-level concurrency
        """
        return config.get("num_workers", 10)  # Default to 10 if not specified

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """
        Ingest conversation data (call online API) with concurrency control.

        Template method that implements the common add flow:
        1. Determine perspective (single or dual)
        2. Organize messages for each user
        3. Call subclass _add_user_messages for each user (with concurrency control)
        4. Post-processing (e.g., wait for tasks)

        Concurrency is controlled by self.num_workers (conversation-level).

        Subclasses can override this method for custom behavior,
        or implement _add_user_messages for standard flow.
        """
        import asyncio

        conversation_ids = []
        add_results = []
        self._import_manifest_records = []
        self._incremental_manifest_by_chunk_id = {}
        if self._uses_incremental_import_manifest():
            self._incremental_manifest_lock = asyncio.Lock()
            self._load_incremental_import_manifest(conversations)

        console = Console()
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print("Stage 1: Add", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        def _conv_label(conv_id: str) -> str:
            parts = conv_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[1]
            return conv_id

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.num_workers)

        async def process_single_conversation(conv, progress, main_task):
            """Process a single conversation with concurrency control."""
            async with semaphore:
                conv_id = conv.conversation_id

                # Extract conversation info (speaker names, user_ids, perspective mode)
                conv_info = self._extract_conversation_info(
                    conversation=conv, conversation_id=conv_id
                )

                # Get format type (subclass can override)
                format_type = self._get_format_type()

                # Organize messages based on perspective
                if conv_info["need_dual_perspective"]:
                    # Dual perspective: prepare messages for both speakers
                    speaker_a_messages = self._conversation_to_messages(
                        conv, format_type=format_type, perspective="speaker_a"
                    )
                    speaker_b_messages = self._conversation_to_messages(
                        conv, format_type=format_type, perspective="speaker_b"
                    )
                    total_messages = len(speaker_a_messages) + len(speaker_b_messages)
                    conv_task_id = progress.add_task(
                        f"[yellow]Conv-{_conv_label(conv_id)}",
                        total=total_messages,
                        completed=0,
                        status="Processing",
                    )

                    # Add messages for both users
                    namespace_scope_a = self._build_namespace_scope(
                        conv, speaker="speaker_a"
                    )
                    namespace_scope_b = self._build_namespace_scope(
                        conv, speaker="speaker_b"
                    )
                    result_a = await self._add_user_messages(
                        conv,
                        speaker_a_messages,
                        speaker="speaker_a",
                        namespace_scope=namespace_scope_a,
                        progress=progress,
                        task_id=conv_task_id,
                        **kwargs,
                    )
                    result_b = await self._add_user_messages(
                        conv,
                        speaker_b_messages,
                        speaker="speaker_b",
                        namespace_scope=namespace_scope_b,
                        progress=progress,
                        task_id=conv_task_id,
                        **kwargs,
                    )
                    if not self._uses_incremental_import_manifest():
                        self._record_import_manifest(
                            conversation=conv,
                            messages=speaker_a_messages,
                            speaker="speaker_a",
                            namespace_scope=namespace_scope_a,
                            raw_result=result_a,
                        )
                        self._record_import_manifest(
                            conversation=conv,
                            messages=speaker_b_messages,
                            speaker="speaker_b",
                            namespace_scope=namespace_scope_b,
                            raw_result=result_b,
                        )

                    # Wait for tasks to complete (per-conversation, before releasing semaphore)
                    # This is important for systems like Memu that need to limit concurrent tasks
                    await self._wait_for_conversation_tasks(
                        [result_a, result_b], conversation_id=conv_id, **kwargs
                    )

                    progress.update(conv_task_id, completed=total_messages, status="✅")
                    progress.update(main_task, advance=1)
                    return conv_id, [result_a, result_b]
                else:
                    # Single perspective: prepare messages for speaker_a only
                    messages = self._conversation_to_messages(
                        conv, format_type=format_type, perspective=None
                    )
                    total_messages = len(messages)
                    conv_task_id = progress.add_task(
                        f"[yellow]Conv-{_conv_label(conv_id)}",
                        total=total_messages,
                        completed=0,
                        status="Processing",
                    )

                    # Add messages for single user
                    namespace_scope = self._build_namespace_scope(
                        conv, speaker="speaker_a"
                    )
                    result = await self._add_user_messages(
                        conv,
                        messages,
                        speaker="speaker_a",
                        namespace_scope=namespace_scope,
                        progress=progress,
                        task_id=conv_task_id,
                        **kwargs,
                    )
                    if not self._uses_incremental_import_manifest():
                        self._record_import_manifest(
                            conversation=conv,
                            messages=messages,
                            speaker="speaker_a",
                            namespace_scope=namespace_scope,
                            raw_result=result,
                        )

                    # Wait for tasks to complete (per-conversation, before releasing semaphore)
                    await self._wait_for_conversation_tasks(
                        [result], conversation_id=conv_id, **kwargs
                    )

                    progress.update(conv_task_id, completed=total_messages, status="✅")
                    progress.update(main_task, advance=1)
                    return conv_id, [result]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[bold blue]{task.fields[status]}"),
            console=console,
            transient=False,
            refresh_per_second=1,
        ) as progress:
            main_task = progress.add_task(
                "[bold cyan]🎯 Overall Progress",
                total=len(conversations),
                completed=0,
                status="Processing",
            )

            # Process all conversations concurrently (with semaphore control)
            tasks = [
                process_single_conversation(conv, progress, main_task)
                for conv in conversations
            ]
            results = await asyncio.gather(*tasks)
            progress.update(main_task, status="✅ Complete")

        # Collect results
        for conv_id, conv_results in results:
            conversation_ids.append(conv_id)
            add_results.extend(conv_results)

        # Post-processing (e.g., wait for async tasks)
        await self._post_add_process(add_results, **kwargs)

        # Build and return result
        result = self._build_add_result(conversation_ids, add_results, **kwargs)
        if isinstance(result, dict):
            result["import_manifest_records"] = self.get_import_manifest_records()
        self._last_add_result = result if isinstance(result, dict) else {}
        return result

    @abstractmethod
    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user (subclass implementation).

        Args:
            conv: Original conversation object (for extracting extra info)
            messages: Formatted message list (ready to send)
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters (may include user_id, timestamp, etc.)

        Returns:
            Subclass-specific result (e.g., task_id for Memu, None for others)
        """
        pass

    async def _wait_for_conversation_tasks(
        self, task_results: List[Any], **kwargs
    ) -> None:
        """
        Wait for tasks from a single conversation to complete (per-conversation hook).

        This is called BEFORE releasing the semaphore, ensuring that systems like Memu
        which create async tasks don't exceed their concurrency limits.

        For systems that complete work immediately (Mem0, Memos), this is a no-op.
        For systems with async tasks (Memu), override this to wait for task completion.

        Args:
            task_results: Results from _add_user_messages for this conversation
            **kwargs: Extra parameters
        """
        # Default: no-op (most systems don't need per-conversation waiting)
        pass

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """
        Retrieve relevant memories (call online API).

        Template method that orchestrates the search process:
        1. Extract conversation info (determine perspective)
        2. Call single or dual perspective search
        3. Subclasses implement actual API calls and result building

        Args:
            query: Query text
            conversation_id: Conversation ID
            index: Index metadata (contains conversation_ids)
            **kwargs: Optional parameters (top_k, conversation, etc.)

        Returns:
            SearchResult with standard format
        """
        # Extract conversation information (speakers, user_ids, dual perspective)
        conv_info = self._extract_conversation_info(
            conversation_id=conversation_id, **kwargs
        )

        # Get top_k from kwargs, or fallback to config, or default to 10
        default_top_k = self.config.get("search", {}).get("top_k", 10)
        top_k = kwargs.get("top_k", default_top_k)

        if conv_info["need_dual_perspective"]:
            # Dual perspective: search from both speakers' perspectives
            return await self._search_dual_perspective(
                query=query,
                conversation_id=conversation_id,
                speaker_a=conv_info["speaker_a"],
                speaker_b=conv_info["speaker_b"],
                speaker_a_user_id=conv_info["speaker_a_user_id"],
                speaker_b_user_id=conv_info["speaker_b_user_id"],
                top_k=top_k,
                **kwargs,
            )
        else:
            # Single perspective: search from one user's perspective
            return await self._search_single_perspective(
                query=query,
                conversation_id=conversation_id,
                user_id=conv_info["speaker_a_user_id"],
                top_k=top_k,
                **kwargs,
            )

    async def _search_single_perspective(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> SearchResult:
        """
        Single perspective search flow (base class implementation).

        Subclasses should NOT override this unless necessary.
        Instead, implement _search_single_user and _build_single_search_result.

        Args:
            query: Query text
            conversation_id: Conversation ID
            user_id: User ID to search for
            top_k: Number of results to retrieve
            **kwargs: Additional parameters

        Returns:
            SearchResult
        """
        # Call subclass to perform search (API call + conversion + special processing)
        results = await self._search_single_user(
            query, conversation_id, user_id, top_k, **kwargs
        )

        # Call subclass to build SearchResult (including formatted_context)
        return self._build_single_search_result(
            query=query,
            conversation_id=conversation_id,
            results=results,
            user_id=user_id,
            top_k=top_k,
            **kwargs,
        )

    async def _search_dual_perspective(
        self,
        query: str,
        conversation_id: str,
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Dual perspective search flow (base class implementation).

        Subclasses should NOT override this unless necessary.
        Instead, implement _search_single_user and _build_dual_search_result.

        Args:
            query: Query text
            conversation_id: Conversation ID
            speaker_a: Speaker A name
            speaker_b: Speaker B name
            speaker_a_user_id: Speaker A user ID
            speaker_b_user_id: Speaker B user ID
            top_k: Number of results per user
            **kwargs: Additional parameters

        Returns:
            SearchResult
        """
        # Search both users separately
        results_a = await self._search_single_user(
            query, conversation_id, speaker_a_user_id, top_k, **kwargs
        )
        results_b = await self._search_single_user(
            query, conversation_id, speaker_b_user_id, top_k, **kwargs
        )

        # Merge results (for fallback, not re-sorted)
        all_results = results_a + results_b

        # Call subclass to build SearchResult (including formatted_context)
        return self._build_dual_search_result(
            query=query,
            conversation_id=conversation_id,
            all_results=all_results,
            results_a=results_a,
            results_b=results_b,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            speaker_a_user_id=speaker_a_user_id,
            speaker_b_user_id=speaker_b_user_id,
            top_k=top_k,
            **kwargs,
        )

    @abstractmethod
    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (subclass must implement).

        This method should:
        1. Call the system's search API
        2. Convert raw results to standard format
        3. Apply system-specific processing (e.g., timezone, preference, summary)

        Standard result format:
        [
            {
                "content": str,      # Display content (may include timestamp, etc.)
                "score": float,      # Relevance score
                "user_id": str,      # User ID
                "metadata": dict     # System-specific metadata
            },
            ...
        ]

        System-specific processing:
        - Mem0: Apply timezone conversion to timestamps
        - Memos: Extract and include preference information
        - Memu: Fetch and include categories summary

        Args:
            query: Query text
            conversation_id: Conversation ID (some systems may need it for context)
            user_id: User ID to search for
            top_k: Number of results to retrieve
            **kwargs: System-specific parameters (e.g., min_similarity)

        Returns:
            List of search results in standard format
        """
        pass

    @abstractmethod
    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Build SearchResult for single perspective (subclass must implement).

        This method should:
        1. Construct retrieval_metadata (system name, parameters, etc.)
        2. Build formatted_context (using template or custom logic)

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult with formatted_context
        """
        pass

    @abstractmethod
    def _build_dual_search_result(
        self,
        query: str,
        conversation_id: str,
        all_results: List[Dict[str, Any]],
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Build SearchResult for dual perspective (subclass must implement).

        This method should:
        1. Construct retrieval_metadata (system name, parameters, dual flag, etc.)
        2. Build formatted_context using both speakers' results
           - Use template or custom logic
           - Include system-specific information (preferences, summaries, etc.)

        Args:
            query: Query text
            conversation_id: Conversation ID
            all_results: Merged results (for fallback)
            results_a: Speaker A's search results
            results_b: Speaker B's search results
            speaker_a: Speaker A name
            speaker_b: Speaker B name
            speaker_a_user_id: Speaker A user ID
            speaker_b_user_id: Speaker B user ID
            top_k: Number of results per user
            **kwargs: Additional parameters

        Returns:
            SearchResult with formatted_context
        """
        pass

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """
        Generate answer (using generic MEMOS prompt).

        Subclasses can override this method to use their own specific prompt.
        Defaults to ANSWER_PROMPT_MEMOS (suitable for most systems).
        """
        # Get answer prompt (subclasses can override _get_answer_prompt)
        prompt = self._get_answer_prompt().format(context=context, question=query)

        # Get retry count
        max_retries = self.config.get("answer", {}).get("max_retries", 3)

        # Generate answer
        for i in range(max_retries):
            try:
                result = await self.llm_provider.generate(prompt=prompt, temperature=0)
                result = clean_answer_text(result)

                if result == "":
                    continue

                return result
            except Exception as e:
                print(f"⚠️  Answer generation error (attempt {i+1}/{max_retries}): {e}")
                if i == max_retries - 1:
                    raise
                continue

        return ""

    def _get_answer_prompt(self) -> str:
        """
        Get answer prompt.

        Subclasses can override this method to return their own prompt.
        Defaults to generic default prompt.
        """
        return self._prompts["online_api"]["default"]["answer_prompt_memos"]

    # ===== Helper methods: format conversion =====

    def _conversation_to_messages(
        self,
        conversation: Conversation,
        format_type: str = "basic",
        perspective: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert standard Conversation to message list.

        Args:
            conversation: Standard conversation object
            format_type: Format type (basic, mem0, memos, memu)
            perspective: Perspective (speaker_a or speaker_b), used for dual-perspective systems like Memos

        Returns:
            Message list
        """
        messages = []
        speaker_a = conversation.metadata.get("speaker_a", "")
        speaker_b = conversation.metadata.get("speaker_b", "")

        for msg in conversation.messages:
            # Intelligently determine role and content
            role, content = self._determine_role_and_content(
                msg.sender_name, msg.content, speaker_a, speaker_b, perspective
            )

            # Base message
            message = {"role": role, "content": content}

            # Add extra fields based on different system requirements
            if format_type == "memos":
                # Memos format: needs chat_time
                # Note: Memos directly sends messages to API, so this field is used
                if msg.timestamp:
                    from common_utils.datetime_utils import to_iso_format

                    message["chat_time"] = to_iso_format(msg.timestamp)

            elif format_type == "memu":
                # Memu format: needs name and time
                message["name"] = msg.sender_name
                message["time"] = (
                    msg.timestamp.isoformat() + "Z" if msg.timestamp else None
                )

            # Note: Mem0 extracts timestamps directly from conv.messages in _add_user_messages

            messages.append(message)

        return messages

    def _determine_role_and_content(
        self,
        sender_name: str,
        content: str,
        speaker_a: str,
        speaker_b: str,
        perspective: Optional[str] = None,
    ) -> tuple:
        """
        Intelligently determine message role and content.

        For systems that only support user/assistant (e.g., Memos), special handling is needed:
        1. If speaker is standard role (user/assistant and variants), use directly
        2. If custom name, convert based on perspective:
           - From speaker_a perspective: speaker_a messages are "user", speaker_b are "assistant"
           - From speaker_b perspective: speaker_b messages are "user", speaker_a are "assistant"
        3. Content for custom speakers needs "speaker: " prefix

        Args:
            sender_name: Sender name
            content: Message content
            speaker_a: speaker_a in conversation
            speaker_b: speaker_b in conversation
            perspective: Perspective (for dual-perspective systems)

        Returns:
            (role, content) tuple
        """
        # Case 1: Standard roles (user/assistant and variants)
        speaker_lower = sender_name.lower()

        # Check if standard role or variant
        if speaker_lower in ["user", "assistant"]:
            # Exact match: "user", "User", "assistant", "Assistant"
            return speaker_lower, content
        elif speaker_lower.startswith("user"):
            # Variants: "user_123", "User_456", etc.
            return "user", content
        elif speaker_lower.startswith("assistant"):
            # Variants: "assistant_123", "Assistant_456", etc.
            return "assistant", content

        # Case 2: Custom speaker, needs conversion
        # Default behavior: speaker_a is user, speaker_b is assistant
        if perspective == "speaker_b":
            # From speaker_b's perspective
            if sender_name == speaker_b:
                role = "user"
            elif sender_name == speaker_a:
                role = "assistant"
            else:
                # Unknown speaker, default to assistant
                role = "assistant"
        else:
            # From speaker_a's perspective (default)
            if sender_name == speaker_a:
                role = "user"
            elif sender_name == speaker_b:
                role = "assistant"
            else:
                # Unknown speaker, default to user
                role = "user"

        # For custom speakers, content needs prefix
        formatted_content = f"{sender_name}: {content}"

        return role, formatted_content

    def _extract_user_id(
        self, conversation: Conversation, speaker: str = "speaker_a"
    ) -> str:
        """
        Extract user_id from Conversation (for online API).

        Logic: Use the legacy {conv_id}_{speaker} style id, and add a per-run
        vision suffix when the user explicitly provides one, so add/search stay aligned.

        Args:
            conversation: Standard conversation object
            speaker: Speaker identifier (speaker_a or speaker_b)

        Returns:
            user_id string

        Examples:
            - Legacy: speaker_a="Caroline" → user_id="locomo_0_Caroline"
            - With vision: → user_id="locomo_0_Caroline__benchmark_v1"
            - No speaker: → user_id="locomo_0_speaker_a"

        Design rationale:
            - Include conv_id: Ensure memory isolation between conversations (evaluation accuracy)
            - Include speaker name: More intuitive for backend viewing (e.g., Caroline vs speaker_a)
            - Replace spaces with underscores: Avoid spaces in user_id
        """
        cached_namespace_id = self._namespace_cache.get(
            (conversation.conversation_id, speaker)
        )
        if cached_namespace_id:
            return cached_namespace_id
        namespace_scope = self._build_namespace_scope(conversation, speaker=speaker)
        return namespace_scope["namespace_id"]

    def _get_user_id_from_conversation_id(self, conversation_id: str) -> str:
        """
        Derive user_id from conversation_id (simplified version).

        Args:
            conversation_id: Conversation ID

        Returns:
            user_id string
        """
        namespace_id = self._namespace_cache.get((conversation_id, "speaker_a"))
        if namespace_id:
            return namespace_id
        base_id = self._build_legacy_entity_id(
            conversation_id=conversation_id,
            speaker="speaker_a",
            sender_name=None,
        )
        return self._apply_run_suffix(base_id)

    def _get_format_type(self) -> str:
        """
        Get format type for _conversation_to_messages.

        Subclasses can override this method to specify their format type.
        Default implementation infers from class name.

        Returns:
            Format type string (e.g., "mem0", "memos", "memu", "basic")
        """
        class_name = self.__class__.__name__.lower()

        # Infer format type from class name
        if "mem0" in class_name:
            return "mem0"
        elif "memos" in class_name:
            return "memos"
        elif "memu" in class_name:
            return "memu"
        else:
            return "basic"

    async def _post_add_process(self, add_results: List[Any], **kwargs) -> None:
        """
        Post-processing after adding all conversations.

        Subclasses can override this method for custom post-processing
        (e.g., Memu waiting for async tasks to complete).

        Args:
            add_results: List of results from _add_user_messages calls
            **kwargs: Extra parameters
        """
        # Default: no post-processing
        pass

    def set_run_context(self, run_context: Dict[str, Any]) -> None:
        """Attach run context and reset per-run adapter state."""
        super().set_run_context(run_context)
        self._import_manifest_records = []
        self._namespace_cache = {}
        self._last_add_result = {}
        self._incremental_manifest_by_chunk_id = {}

    def get_import_manifest_records(self) -> List[Dict[str, Any]]:
        """Return captured import manifest rows."""
        return list(self._import_manifest_records)

    def _uses_incremental_import_manifest(self) -> bool:
        """Whether add should persist import manifest rows as writes succeed."""
        return False

    def _build_add_result(
        self, conversation_ids: List[str], add_results: List[Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Build the final result dict for add method.

        Subclasses can override this method to customize the return structure.

        Args:
            conversation_ids: List of conversation IDs that were added
            add_results: List of results from _add_user_messages calls
            **kwargs: Extra parameters

        Returns:
            Result dictionary
        """
        system_name = self.__class__.__name__.replace("Adapter", "").lower()

        result = {
            "type": "online_api",
            "system": system_name,
            "conversation_ids": conversation_ids,
        }

        # If add_results contains non-None values, include them
        # (e.g., Memu's task_ids)
        non_none_results = [r for r in add_results if r is not None]
        if non_none_results:
            result["add_results"] = non_none_results

        return result

    def _normalize_namespace_component(self, value: Any) -> str:
        """Normalize namespace components for provider ids."""
        text = str(value or "unknown").strip()
        if not text:
            text = "unknown"
        return text.replace(" ", "_").replace("/", "_")

    def _system_id(self) -> str:
        """Normalized system id for manifests and namespaces."""
        run_context = getattr(self, "run_context", {}) or {}
        return self._normalize_namespace_component(
            run_context.get("system_id")
            or self.config.get("name")
            or self.__class__.__name__.replace("Adapter", "").lower()
        )

    def _build_legacy_entity_id(
        self,
        conversation_id: str,
        speaker: str,
        sender_name: Any,
    ) -> str:
        """Build the pre-Category-1 style provider id: {conv_id}_{speaker-or-name}."""
        normalized_conv_id = self._normalize_namespace_component(conversation_id)
        normalized_sender = self._normalize_namespace_component(sender_name or speaker)
        return f"{normalized_conv_id}_{normalized_sender}"

    def _get_namespace_vision(self) -> str | None:
        """Return the user-defined vision token for provider ids, if any."""
        run_context = getattr(self, "run_context", {}) or {}
        vision = (
            run_context.get("vision")
            or self.config.get("vision")
        )
        if vision in (None, "", "default"):
            return None
        return self._normalize_namespace_component(vision)

    def _apply_run_suffix(self, base_id: str) -> str:
        """Add the user-defined vision suffix to a provider id when available."""
        vision = self._get_namespace_vision()
        if not vision:
            return base_id
        return f"{base_id}__{vision}"

    def _build_namespace_scope(
        self, conversation: Conversation, speaker: str = "speaker_a"
    ) -> Dict[str, Any]:
        """Build provider namespace metadata with a lightweight per-run suffix."""
        run_context = getattr(self, "run_context", {}) or {}
        namespace_cache = getattr(self, "_namespace_cache", {})
        sender_name = conversation.metadata.get(speaker) or speaker
        need_dual = self._need_dual_perspective(
            conversation.metadata.get("speaker_a", ""),
            conversation.metadata.get("speaker_b", ""),
        )
        base_namespace_id = self._build_legacy_entity_id(
            conversation_id=conversation.conversation_id,
            speaker=speaker,
            sender_name=sender_name,
        )
        namespace_id = self._apply_run_suffix(base_namespace_id)
        view_id = speaker if need_dual else "shared"

        if not run_context:
            namespace_cache[(conversation.conversation_id, speaker)] = namespace_id
            self._namespace_cache = namespace_cache
            return {
                "run_id": "legacy",
                "system_id": self._system_id(),
                "dataset_id": conversation.conversation_id.split("_", 1)[0],
                "conversation_id": conversation.conversation_id,
                "view_id": view_id,
                "speaker": sender_name,
                "namespace_id": namespace_id,
            }

        run_id = self._normalize_namespace_component(run_context.get("run_id", "run"))
        dataset_id = self._normalize_namespace_component(
            run_context.get("dataset_id", conversation.conversation_id.split("_", 1)[0])
        )
        namespace_cache[(conversation.conversation_id, speaker)] = namespace_id
        self._namespace_cache = namespace_cache
        return {
            "run_id": run_id,
            "system_id": self._system_id(),
            "dataset_id": dataset_id,
            "conversation_id": conversation.conversation_id,
            "view_id": view_id,
            "speaker": sender_name,
            "namespace_id": namespace_id,
        }

    def _extract_source_unit_ids(
        self, conversation: Conversation, messages: List[Dict[str, Any]]
    ) -> List[str]:
        """Best-effort extraction of source-unit ids for a transport chunk."""
        if not messages:
            return []

        source_unit_ids: List[str] = []
        message_id_to_source: Dict[str, str] = {}
        for idx, message in enumerate(conversation.messages):
            source_unit_id = message.metadata.get("source_unit_id")
            if source_unit_id:
                source_unit_ids.append(source_unit_id)
            for alias in (
                message.metadata.get("source_unit_id"),
                message.metadata.get("dia_id"),
                message.metadata.get("message_id"),
            ):
                if alias:
                    message_id_to_source[str(alias)] = message.metadata.get(
                        "source_unit_id", f"{conversation.conversation_id}:msg:{idx}"
                    )

        resolved_ids: List[str] = []
        for idx, payload in enumerate(messages):
            if payload.get("source_unit_id"):
                resolved_ids.append(str(payload["source_unit_id"]))
                continue
            for alias in (payload.get("message_id"), payload.get("dia_id")):
                if alias and str(alias) in message_id_to_source:
                    resolved_ids.append(message_id_to_source[str(alias)])
                    break
            else:
                if idx < len(conversation.messages):
                    fallback_id = conversation.messages[idx].metadata.get(
                        "source_unit_id"
                    )
                    if fallback_id:
                        resolved_ids.append(str(fallback_id))

        return resolved_ids

    def _build_write_request_summary(
        self,
        messages: List[Dict[str, Any]],
        speaker: str,
        namespace_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compact summary of a provider write request."""
        return {
            "speaker": speaker,
            "message_count": len(messages),
            "namespace_id": namespace_scope.get("namespace_id"),
            "message_keys": sorted(messages[0].keys()) if messages else [],
        }

    def _coerce_receipt_rows(self, raw_result: Any) -> List[Dict[str, Any]]:
        """Normalize adapter return values into receipt-like dictionaries."""
        if raw_result is None:
            return []
        if isinstance(raw_result, WriteReceipt):
            return [dataclass_to_dict(raw_result)]
        if isinstance(raw_result, list):
            rows: List[Dict[str, Any]] = []
            for item in raw_result:
                rows.extend(self._coerce_receipt_rows(item))
            return rows
        if isinstance(raw_result, dict):
            return [dict(raw_result)]
        return [{"provider_receipt": {"result": raw_result}}]

    def _record_import_manifest(
        self,
        conversation: Conversation,
        messages: List[Dict[str, Any]],
        speaker: str,
        namespace_scope: Dict[str, Any],
        raw_result: Any,
    ) -> None:
        """Record import manifest rows for one adapter add call."""
        source_unit_ids = self._extract_source_unit_ids(conversation, messages)
        receipts = self._coerce_receipt_rows(raw_result)
        if not receipts:
            receipts = [
                dataclass_to_dict(
                    WriteReceipt(
                        system_id=self._system_id(),
                        namespace_scope=namespace_scope,
                        chunk_id=f"{conversation.conversation_id}:{namespace_scope['view_id']}:chunk0",
                        source_unit_ids=source_unit_ids,
                        provider_receipt={},
                        provider_status="submitted",
                        memory_refs=[],
                        errors=[],
                    )
                )
            ]

        for index, receipt in enumerate(receipts):
            chunk_id = receipt.get(
                "chunk_id",
                f"{conversation.conversation_id}:{namespace_scope['view_id']}:chunk{index}",
            )
            receipt_row = {
                "system_id": receipt.get("system_id", self._system_id()),
                "namespace_scope": receipt.get("namespace_scope", namespace_scope),
                "chunk_id": chunk_id,
                "source_unit_ids": receipt.get("source_unit_ids", source_unit_ids),
                "provider_receipt": receipt.get("provider_receipt", receipt),
                "provider_status": receipt.get("provider_status", "submitted"),
                "memory_refs": receipt.get("memory_refs", []),
                "errors": receipt.get("errors", []),
            }
            manifest_row = ImportManifestRecord(
                run_id=str(self.run_context.get("run_id", "run")),
                system_id=self._system_id(),
                conversation_id=conversation.conversation_id,
                view_id=str(namespace_scope.get("view_id", speaker)),
                chunk_id=chunk_id,
                source_unit_ids=receipt_row["source_unit_ids"],
                write_request_summary=self._build_write_request_summary(
                    messages=messages,
                    speaker=speaker,
                    namespace_scope=namespace_scope,
                ),
                write_receipt=receipt_row,
                memory_refs=receipt_row["memory_refs"],
                write_status=receipt_row["provider_status"],
                errors=receipt_row["errors"],
            )
            self._import_manifest_records.append(dataclass_to_dict(manifest_row))

    def _manifest_checkpoint_path(self) -> Path:
        return Path(getattr(self, "output_dir", ".")) / "import_manifest.jsonl"

    def _load_incremental_import_manifest(
        self, conversations: List[Conversation]
    ) -> None:
        """Load compatible partial add checkpoint rows for the selected run."""
        manifest_path = self._manifest_checkpoint_path()
        self._import_manifest_records = []
        self._incremental_manifest_by_chunk_id = {}
        if not manifest_path.exists():
            return

        current_run_id = str(
            (getattr(self, "run_context", {}) or {}).get("run_id", "run")
        )
        current_system_id = self._system_id()
        selected_conversation_ids = {
            str(conversation.conversation_id) for conversation in conversations
        }
        seen_chunk_ids: set[str] = set()

        with manifest_path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Cannot resume add: malformed import_manifest.jsonl at "
                        f"{manifest_path}:{line_number}"
                    ) from exc

                conversation_id = str(row.get("conversation_id", ""))
                if conversation_id not in selected_conversation_ids:
                    continue

                row_run_id = str(row.get("run_id", ""))
                row_system_id = str(row.get("system_id", ""))
                if row_run_id != current_run_id:
                    raise RuntimeError(
                        "Cannot resume add: existing import_manifest.jsonl belongs "
                        f"to run_id={row_run_id}, current run_id={current_run_id}."
                    )
                if row_system_id != current_system_id:
                    raise RuntimeError(
                        "Cannot resume add: existing import_manifest.jsonl belongs "
                        f"to system_id={row_system_id}, current system_id={current_system_id}."
                    )

                chunk_id = str(row.get("chunk_id", ""))
                if not chunk_id or chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                self._import_manifest_records.append(row)
                if not row.get("errors") and str(row.get("write_status", "")) not in {
                    "failed",
                    "error",
                }:
                    self._incremental_manifest_by_chunk_id[chunk_id] = row

    def _save_incremental_import_manifest(self) -> None:
        """Persist current manifest rows atomically for add resume."""
        manifest_path = self._manifest_checkpoint_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file_obj:
            for row in self._import_manifest_records:
                file_obj.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        tmp_path.replace(manifest_path)

    async def _record_incremental_import_manifest(
        self,
        conversation: Conversation,
        messages: List[Dict[str, Any]],
        speaker: str,
        namespace_scope: Dict[str, Any],
        raw_result: Any,
    ) -> None:
        """Record one successful write and flush the add checkpoint."""

        def record_and_save() -> None:
            before_count = len(self._import_manifest_records)
            self._record_import_manifest(
                conversation=conversation,
                messages=messages,
                speaker=speaker,
                namespace_scope=namespace_scope,
                raw_result=raw_result,
            )
            for row in self._import_manifest_records[before_count:]:
                chunk_id = str(row.get("chunk_id", ""))
                if chunk_id:
                    self._incremental_manifest_by_chunk_id[chunk_id] = row
            self._save_incremental_import_manifest()

        lock = getattr(self, "_incremental_manifest_lock", None)
        if lock is None:
            record_and_save()
            return
        async with lock:
            record_and_save()

    def _incremental_import_manifest_row(
        self, chunk_id: str
    ) -> Optional[Dict[str, Any]]:
        return getattr(self, "_incremental_manifest_by_chunk_id", {}).get(chunk_id)

    def _build_import_manifest_chunk_id(
        self,
        conversation: Conversation,
        namespace_scope: Dict[str, Any],
        index: int,
    ) -> str:
        return (
            f"{conversation.conversation_id}:"
            f"{namespace_scope.get('view_id', 'shared')}:chunk{index}"
        )

    def _batch_messages_with_retry(
        self,
        messages: List[Dict[str, Any]],
        batch_size: int,
        add_func: callable,
        max_retries: int = None,
        description: str = "Batch",
    ) -> None:
        """
        Helper method for batching messages with retry logic.

        Subclasses can use this method to simplify batch processing.

        Args:
            messages: Message list to batch
            batch_size: Batch size
            add_func: Function to call for each batch (should accept List[Dict])
            max_retries: Max retry attempts (defaults to self.max_retries)
            description: Description for logging
        """
        if max_retries is None:
            max_retries = getattr(self, 'max_retries', 3)

        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]

            # Retry mechanism
            for attempt in range(max_retries):
                try:
                    add_func(batch_messages)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(
                            f"   ⚠️  [{description}] Retry {attempt + 1}/{max_retries}: {e}"
                        )
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        print(
                            f"   ❌ [{description}] Failed after {max_retries} retries: {e}"
                        )
                        raise e

    def _need_dual_perspective(self, speaker_a: str, speaker_b: str) -> bool:
        """
        Determine if dual-perspective handling is needed.

        Single perspective (no dual-perspective needed):
        - Standard roles: "user"/"assistant"
        - Case variants: "User"/"Assistant"
        - With suffix: "user_123"/"assistant_456"

        Dual perspective (dual-perspective needed):
        - Custom names: "Elena Rodriguez"/"Alex"

        Args:
            speaker_a: Speaker A name
            speaker_b: Speaker B name

        Returns:
            True if dual perspective is needed, False otherwise
        """

        def is_standard_role(speaker: str) -> bool:
            speaker = speaker.lower()
            # Exact match
            if speaker in ["user", "assistant"]:
                return True
            # Starts with user or assistant
            if speaker.startswith("user") or speaker.startswith("assistant"):
                return True
            return False

        # Only need dual perspective when both speakers are not standard roles
        return not (is_standard_role(speaker_a) or is_standard_role(speaker_b))

    def _extract_conversation_info(
        self,
        conversation: Optional[Conversation] = None,
        conversation_id: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Extract conversation information.

        This helper method extracts speaker information and determines if dual
        perspective handling is needed. Used by both add and search methods.

        Args:
            conversation: Conversation object (if directly available)
            conversation_id: Conversation ID (for fallback)
            **kwargs: May contain 'conversation' key if not passed directly

        Returns:
            Dictionary with keys:
            - speaker_a: Speaker A name
            - speaker_b: Speaker B name
            - speaker_a_user_id: User ID for speaker A
            - speaker_b_user_id: User ID for speaker B
            - need_dual_perspective: Whether dual perspective is needed
        """
        # Get conversation from parameter or kwargs
        if conversation is None:
            conversation = kwargs.get("conversation")

        if conversation:
            speaker_a = conversation.metadata.get("speaker_a", "")
            speaker_b = conversation.metadata.get("speaker_b", "")
            speaker_a_user_id = self._extract_user_id(conversation, speaker="speaker_a")
            speaker_b_user_id = self._extract_user_id(conversation, speaker="speaker_b")
            need_dual_perspective = self._need_dual_perspective(speaker_a, speaker_b)
        else:
            # Fallback: use default values (for search when conversation not available)
            if conversation_id is None:
                conversation_id = "unknown"
            speaker_a_user_id = self._apply_run_suffix(
                self._build_legacy_entity_id(
                    conversation_id=conversation_id,
                    speaker="speaker_a",
                    sender_name=None,
                )
            )
            speaker_b_user_id = self._apply_run_suffix(
                self._build_legacy_entity_id(
                    conversation_id=conversation_id,
                    speaker="speaker_b",
                    sender_name=None,
                )
            )
            speaker_a = "speaker_a"
            speaker_b = "speaker_b"
            need_dual_perspective = False

        return {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "speaker_a_user_id": speaker_a_user_id,
            "speaker_b_user_id": speaker_b_user_id,
            "need_dual_perspective": need_dual_perspective,
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": self.__class__.__name__,
            "type": "online_api",
            "description": f"{self.__class__.__name__} adapter for online memory API",
        }
