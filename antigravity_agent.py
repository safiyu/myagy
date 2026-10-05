#!/usr/bin/env python3
"""
Antigravity Multi-GPU Hybrid Agent Runner
Thin entry point delegating to the `myagy` package:
  - config.py: Central paths, ports, endpoint definitions
  - core/: session orchestrator, Laya engine, CUDA coordinator, llama-shift client
  - agents/: subagent queue, context compactor, /brainstorm workflow
  - context/: repo map, project instructions, hooks.json, MCP discovery
  - tools/: web search and page fetch
  - terminal/: REPL (cli.py), UI styling, ESC listener
"""

import sys
import os
import asyncio

# Ensure the package parent directory is in sys.path
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only the parent dir goes on the path so subpackages stay namespaced under myagy
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Support running directly as a script or importing as a module
if __package__ is None or __package__ == "":
    from myagy import *
    from myagy.terminal.cli import main
else:
    from . import *
    from .terminal.cli import main

if __name__ == "__main__":
    asyncio.run(main())
