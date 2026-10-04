#!/usr/bin/env python3
"""
Antigravity Multi-GPU Hybrid Agent Runner
Modular entry point delegating to the `myagy` architecture:
  - config.py: Central paths, ports, endpoint definitions
  - ui.py: Terminal styling, colors, rich console integration
  - llamashift.py: Port detection, model querying, and LlamaShift hot-swapping
  - mcp_loader.py: Model Context Protocol discovery (Antigravity native, VS Code, Kontexta)
  - laya.py: Laya System 1 Decision Engine (~33ms ModernBERT Non-Autoregressive)
  - compactor.py: Context telemetry, Working Memory CUDA synthesis, continuous hippocampus
  - session.py: MultiGpuHybridSession cross-GPU orchestrator
  - cli.py: Multi-line prompt reader, command palette, interactive REPL loop
"""

import sys
import os
import asyncio

# Ensure parent and package directories are in sys.path
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)

for _p in (_PARENT_DIR, _PKG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Support running directly as a script or importing as a module
if __package__ is None or __package__ == "":
    from myagy.config import (
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
    from myagy.ui import UI, RICH_AVAILABLE, console
    from myagy.llamashift import (
        detect_port_model,
        query_endpoint_model,
        trigger_llamashift_switch,
        get_llamashift_active,
        get_llamashift_models,
    )
    from myagy.mcp_loader import load_mcp_servers, check_oauth_available
    from myagy.laya import LayaDecisionEngine
    from myagy.compactor import ContextCurator
    from myagy.coordinator import CudaCoordinator
    from myagy.subagent import SubagentManager, SubagentTask
    from myagy.repomap import RepoMap
    from myagy.session import MultiGpuHybridSession
    from myagy.cli import read_input_prompt, interactive_loop, main
else:
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

if __name__ == "__main__":
    asyncio.run(main())
