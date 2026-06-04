"""
Adapter base class - define unified memory system adapter interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.core.readback import StorageReadbackResult


class BaseAdapter(ABC):
    """Memory system adapter base class."""
    
    def __init__(self, config: dict):
        """
        Initialize adapter.
        
        Args:
            config: System config dict
        """
        self.config = config
        self.run_context: Dict[str, Any] = {}
    
    @abstractmethod
    async def add(self, conversations: List[Conversation], **kwargs) -> Any:
        """
        Ingest conversation data and build index (Add stage).

        This method encapsulates system-specific data ingestion and index building:
        - For EverMemOS: MemCell extraction + BM25/Embedding index building
        - For Mem0: Direct storage to vector database
        - For other systems: Their respective implementations

        Args:
            conversations: Standard format conversation list
            **kwargs: Extra parameters

        Returns:
            Index object (system internal format, different systems return different types)
        """
        pass
    
    @abstractmethod
    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """
        Retrieve relevant memories (Search stage).

        Args:
            query: Query text
            conversation_id: Conversation ID
            index: Index object (returned by add())
            **kwargs: Extra parameters (e.g., top_k)
            
        Returns:
            Standard format search result
        """
        pass

    async def search_from_readback(
        self,
        query: str,
        conversation_id: str,
        index: Any = None,
        *,
        conversation: Optional[Conversation] = None,
        question_id: Optional[str] = None,
        question_metadata: Optional[Dict[str, Any]] = None,
        import_manifest_records: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> SearchResult:
        """Build a search result from provider storage readback.

        Adapters that cannot read back provider storage should return an
        unsupported result rather than raising.
        """
        del index, conversation, import_manifest_records, kwargs
        metadata = question_metadata or {}
        session_ids = metadata.get("session_ids") or []
        if not isinstance(session_ids, list):
            session_ids = [session_ids]
        return SearchResult(
            question_id=question_id or "",
            query=query,
            conversation_id=conversation_id,
            results=[],
            retrieval_metadata={
                "search_mode": "readback",
                "provider": self.__class__.__name__,
                "session_ids": [str(value) for value in session_ids if value],
                "formatted_context": "",
                "error": "Adapter has not implemented search_from_readback().",
            },
            retrieval_status="unsupported",
        )

    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        """
        Preparation stage: operations executed before add.

        Optional preparation operations, e.g.:
        - Update project config (e.g., Mem0's custom_instructions)
        - Clean existing data (if clean_before_add configured)
        - Other system-specific initialization

        Args:
            conversations: Standard format conversation list (for extracting user_id etc.)
            **kwargs: Extra parameters
        
        Returns:
            None
        """
        pass  # Default: no operation

    def render_answer_prompt(
        self, query: str, context: str, **kwargs: Any
    ) -> Optional[str]:
        """Render the prompt used for answer generation when available."""
        del kwargs
        get_prompt = getattr(self, "_get_answer_prompt", None)
        if not callable(get_prompt):
            return None
        return str(get_prompt()).format(context=context, question=query)

    def consume_answer_trace(
        self, question_id: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Return and clear adapter-specific answer metadata when available."""
        del question_id, kwargs
        return {}

    def set_run_context(self, run_context: Dict[str, Any]) -> None:
        """Attach stable per-run context for namespace isolation and artifacts."""
        self.run_context = run_context

    async def probe_readiness(self, add_result: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Best-effort readiness probe for asynchronous providers.

        Returns a normalized result dict:
        {
            "supported": bool,
            "ready": bool,
            "status": str,
            "details": dict,
        }
        """
        return {
            "supported": False,
            "ready": False,
            "status": "unsupported",
            "details": {},
        }

    async def finalize_imports(
        self,
        import_manifest_rows: List[Dict[str, Any]],
        *,
        add_result: Any = None,
        budget_seconds: int = 0,
        poll_interval_seconds: float | None = None,
        dataset: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Refresh provisional import manifest rows into a finalized state.

        Default behavior is a no-op finalize that marks the current manifest as ready.
        Adapters with asynchronous or eventually-consistent writes should override this.
        """
        del add_result, poll_interval_seconds, dataset, kwargs

        provider_status_counts: Dict[str, int] = {}
        missing_memory_refs: List[str] = []
        for row in import_manifest_rows:
            status = str(row.get("write_status", "unknown"))
            provider_status_counts[status] = provider_status_counts.get(status, 0) + 1
            if not row.get("memory_refs"):
                missing_memory_refs.append(str(row.get("chunk_id", "")))

        return {
            "import_manifest_records": list(import_manifest_rows),
            "ready": True,
            "status": "finalized",
            "provider_status_counts": provider_status_counts,
            "updated_rows": len(import_manifest_rows),
            "missing_memory_refs": missing_memory_refs,
            "warnings": [],
            # Default finalize is a no-op and does not consume polling budget.
            "finalize_budget_exhausted": False,
            "finalized_at": None,
        }

    async def get_storage_readback(
        self,
        *,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        question_id: Optional[str] = None,
        evidence_texts: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> StorageReadbackResult:
        """Collect provider-specific storage objects for readback search.

        This is intentionally a virtual hook. The pipeline cannot assume
        whether a provider reads by memory id, user id, conversation id,
        namespace, filter, page cursor, or some other API. Adapters must
        implement provider pagination/filtering and normalize objects to
        ``NormalizedStorageObject``.
        """
        del user_id, run_id, evidence_texts, context, kwargs
        return StorageReadbackResult(
            status="unsupported",
            checked_session_ids=list(session_ids or []),
            missing_session_ids=[],
            objects=[],
            metadata={
                "provider": self.__class__.__name__,
                "question_id": question_id,
                "reason": "Adapter has not implemented get_storage_readback().",
            },
            errors=[],
        )

    def get_import_manifest_records(self) -> List[Dict[str, Any]]:
        """Return current run import-manifest rows if the adapter captured them."""
        return []
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Return system info (for result recording).
        Returns:
            System info dict
        """
        return {"name": self.__class__.__name__, "config": self.config}
    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Any:
        """
        Build lazy-loaded index metadata.
        
        Default: return None (online API systems don't need index)
        Local systems (e.g., EverMemOS) should override this method

        Args:
            conversations: Conversation list
            output_dir: Output directory

        Returns:
            Index object or metadata (local systems return index metadata, online systems return None)
        """
        return None
