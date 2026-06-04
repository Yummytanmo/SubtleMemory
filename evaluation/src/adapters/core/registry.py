"""
Adapter registry - provide adapter registration and creation.
Uses lazy loading strategy, keeps __init__.py empty.
"""
import importlib
from typing import Dict, Type, List
from evaluation.src.adapters.core.base import BaseAdapter


_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}

# Adapter module mapping (for lazy loading)
_ADAPTER_MODULES = {
    # Baseline systems
    "oracle_context": "evaluation.src.adapters.baselines.oracle_context_adapter",
    "no_context": "evaluation.src.adapters.baselines.no_context_adapter",
    
    # Online API systems
    "mem0": "evaluation.src.adapters.providers.mem0_adapter",
    "memos": "evaluation.src.adapters.providers.memos_adapter",
    "memu": "evaluation.src.adapters.providers.memu_adapter",
    "zep": "evaluation.src.adapters.providers.zep_adapter",
    "evermemos_api": "evaluation.src.adapters.providers.evermemos_api_adapter",
    "memobase": "evaluation.src.adapters.providers.memobase_adapter",
    "amem": "evaluation.src.adapters.providers.amem_adapter",
    "openclaw_session_memory": "evaluation.src.adapters.openclaw.session_memory_adapter",
    "openclaw_mem0_plugin": "evaluation.src.adapters.openclaw.mem0_plugin_adapter",
    "openclaw_memos_plugin": "evaluation.src.adapters.openclaw.memos_plugin_adapter",
    "openclaw_evermemos_plugin": "evaluation.src.adapters.openclaw.evermemos_plugin_adapter",
    "metaclaw": "evaluation.src.adapters.metaclaw.metaclaw_adapter",
    "mirix": "evaluation.src.adapters.mirix.mirix_adapter",
}


def register_adapter(name: str):
    """
    Decorator for registering adapters.
    
    Usage:
        @register_adapter("evermemos")
        class EverMemOSAdapter(BaseAdapter):
            ...
    """
    def decorator(cls: Type[BaseAdapter]):
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_adapter_loaded(name: str):
    """
    Ensure specified adapter is loaded (lazy loading strategy).
    
    Trigger @register_adapter decorator execution via dynamic import.
    This keeps __init__.py empty per project convention.
    
    Args:
        name: Adapter name
        
    Raises:
        ValueError: If adapter doesn't exist
        RuntimeError: If module loaded but not registered
    """
    if name in _ADAPTER_REGISTRY:
        return  # Already loaded
    
    if name not in _ADAPTER_MODULES:
        raise ValueError(
            f"Unknown adapter: {name}. "
            f"Available adapters: {list(_ADAPTER_MODULES.keys())}"
        )
    
    # Dynamically import module, trigger @register_adapter execution
    module_path = _ADAPTER_MODULES[name]
    importlib.import_module(module_path)
    
    # Verify registration success
    if name not in _ADAPTER_REGISTRY:
        raise RuntimeError(
            f"Adapter '{name}' module loaded but not registered. "
            f"Check if @register_adapter('{name}') decorator is present."
        )


def create_adapter(name: str, config: dict, output_dir = None) -> BaseAdapter:
    """
    Create adapter instance.
    
    Args:
        name: Adapter name
        config: Config dict
        output_dir: Output directory (for persistence, optional)
        
    Returns:
        Adapter instance
        
    Raises:
        ValueError: If adapter not registered
    """
    # Lazy loading: ensure adapter loaded
    _ensure_adapter_loaded(name)
    
    # Try passing output_dir, fallback if adapter doesn't support it
    try:
        return _ADAPTER_REGISTRY[name](config, output_dir=output_dir)
    except TypeError:
        # Adapter doesn't accept output_dir parameter, use default creation
        return _ADAPTER_REGISTRY[name](config)


def list_adapters() -> List[str]:
    """List all available adapters."""
    return list(_ADAPTER_MODULES.keys())
