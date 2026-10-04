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
    from myagy import *
    from myagy.terminal.cli import main
else:
    from . import *
    from .terminal.cli import main

if __name__ == "__main__":
    asyncio.run(main())
