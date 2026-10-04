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

from .terminal import (
    UI,
    RICH_AVAILABLE,
    console,
    EscListener,
    read_input_prompt,
    interactive_loop,
    main,
)

from .core import (
    MultiGpuHybridSession,
    CudaCoordinator,
    LayaDecisionEngine,
    detect_port_model,
    query_endpoint_model,
    trigger_llamashift_switch,
    get_llamashift_active,
    get_llamashift_models,
)

from .agents import (
    SubagentManager,
    SubagentTask,
    ContextCurator,
    BrainstormWorkflow,
)

from .context import (
    RepoMap,
    ProjectInstructions,
    ExternalHooksManager,
    load_mcp_servers,
    check_oauth_available,
    McpStdioServer,
)

from .tools import (
    WebSearchEngine,
)

__all__ = [
    # Core
    "MultiGpuHybridSession",
    "CudaCoordinator",
    "LayaDecisionEngine",
    "detect_port_model",
    "query_endpoint_model",
    "trigger_llamashift_switch",
    "get_llamashift_active",
    "get_llamashift_models",
    # Agents
    "SubagentManager",
    "SubagentTask",
    "ContextCurator",
    "BrainstormWorkflow",
    # Context
    "RepoMap",
    "ProjectInstructions",
    "ExternalHooksManager",
    "load_mcp_servers",
    "check_oauth_available",
    "McpStdioServer",
    # Tools
    "WebSearchEngine",
    # Terminal & UI
    "UI",
    "RICH_AVAILABLE",
    "console",
    "EscListener",
    "read_input_prompt",
    "interactive_loop",
    "main",
    # Config
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
