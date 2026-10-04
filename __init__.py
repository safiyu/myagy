"""
myagy - Antigravity Multi-GPU Hybrid Agent Package
Dual-GPU executive pipeline (AMD ROCm + NVIDIA CUDA) with continuous background hippocampus curation,
Laya System 1 decision engine (~33ms ModernBERT), and Cloud Gemini fallback.
"""

from .config import (
    PORT_ROCM,
    PORT_CUDA,
    PORT_LLAMASHIFT,
    PORT_LAYA,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LAYA_ENDPOINT,
    DEFAULT_LLAMASHIFT_URL,
    SESSIONS_DIR,
    MCP_CONFIG_PATH,
    OAUTH_TOKEN_PATH,
)
from .ui import UI, RICH_AVAILABLE, console
from .llamashift import (
    detect_port_model,
    query_endpoint_model,
    trigger_llamashift_switch,
    get_llamashift_active,
    get_llamashift_models,
)
from .mcp_loader import load_mcp_servers, check_oauth_available
from .laya import LayaDecisionEngine
from .compactor import ContextCurator
from .coordinator import CudaCoordinator
from .subagent import SubagentManager, SubagentTask
from .repomap import RepoMap
from .instructions import ProjectInstructions
from .hooks_loader import ExternalHooksManager
from .brainstorm import BrainstormWorkflow
from .session import MultiGpuHybridSession
from .cli import read_input_prompt, interactive_loop, main

__all__ = [
    "MultiGpuHybridSession",
    "LayaDecisionEngine",
    "ContextCurator",
    "CudaCoordinator",
    "SubagentManager",
    "SubagentTask",
    "RepoMap",
    "ProjectInstructions",
    "ExternalHooksManager",
    "BrainstormWorkflow",
    "UI",
    "main",
    "read_input_prompt",
    "interactive_loop",
    "load_mcp_servers",
    "check_oauth_available",
    "detect_port_model",
    "query_endpoint_model",
    "trigger_llamashift_switch",
    "get_llamashift_active",
    "get_llamashift_models",
    "RICH_AVAILABLE",
    "console",
    "PORT_ROCM",
    "PORT_CUDA",
    "PORT_LLAMASHIFT",
    "PORT_LAYA",
    "DEFAULT_CLOUD_MODEL",
    "DEFAULT_LAYA_ENDPOINT",
    "DEFAULT_LLAMASHIFT_URL",
    "SESSIONS_DIR",
    "MCP_CONFIG_PATH",
    "OAUTH_TOKEN_PATH",
]
