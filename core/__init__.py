"""Core execution and multi-GPU routing engine."""

from .coordinator import CudaCoordinator
from .laya import LayaDecisionEngine
from .sudo_manager import SudoManager
from .llamashift import (
    detect_port_model,
    query_endpoint_model,
    trigger_llamashift_switch,
    get_llamashift_active,
    get_llamashift_models,
)


def __getattr__(name):
    # Lazy so leaf modules import without pulling in the SDK or a circular agents<->core import
    if name == "MultiGpuHybridSession":
        from .session import MultiGpuHybridSession
        return MultiGpuHybridSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MultiGpuHybridSession",
    "CudaCoordinator",
    "LayaDecisionEngine",
    "SudoManager",
    "detect_port_model",
    "query_endpoint_model",
    "trigger_llamashift_switch",
    "get_llamashift_active",
    "get_llamashift_models",
]
