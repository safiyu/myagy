# myagy: Antigravity Multi-GPU Hybrid Agent

An asynchronous, dual-accelerator pairing framework and terminal copilot built on top of the `google-antigravity` Python SDK.

---

## ⚡ Key Architectural Highlights

* **Dual-GPU Asymmetric Compute Pipeline**:
  * **Primary Executive (`ROCm:9000`)**: AMD ROCm GPU running deep reasoning models (e.g. Qwen 2.5 35B / Muse Glimmer) with full tool execution privileges.
  * **Dedicated Compactor & Subagent Engine (`CUDA:9001`)**: NVIDIA RTX 4060 running IBM Granite 4.2 8B for rolling background Working Memory synthesis and asynchronous task delegation.
  * **LlamaShift Hot-Swapping (`Port 8002`)**: Zero-downtime model lifecycle management and VRAM hot-swapping.
* **Laya System 1 Decision Engine (`Port 8003`)**:
  * ~33ms Non-Autoregressive ModernBERT (~421M params) for dynamic prompt auto-routing, escalation scoring, and Bayesian safe-action permission gating (`/noul`).
* **Hardware Slot Arbitrator (`CudaCoordinator`)**:
  * On-demand subagents have absolute priority over background housekeeping.
  * Background continuous hippocampus compaction automatically yields and preempts when subagents are active.
* **Zero-Shot Automatic Codebase AST Outline (`RepoMap`)**:
  * Auto-discovers and caches classes, methods, and functions across the repository in <1.5ms, automatically embedding symbol structure into the model's system prompt.
* **Automatic Project Instructions & Rules Injection (`instructions.py`)**:
  * Scans repository root for `GEMINI.md`, `antigravity.md`, `AGENTS.md` (or `.agents/` / `.gemini/` rules) and feeds project context into system memory on startup (<0.1ms mtime cached).
* **Lifecycle Hooks Engine (`hooks_loader.py`)**:
  * Discovers `.agents/hooks.json` and `~/.gemini/config/hooks.json`, executing `PreToolUse` security gates, parameter overwrites, and `PostToolUse` linters.
* **Universal MCP Server Support**:
  * Discovers Antigravity native, VS Code-compatible, and local Kontexta MCP tools.
* **Google Cloud Gemini Fallback**:
  * Native streaming through active Google Antigravity OAuth session.

---

## 📂 Project Structure

```text
Projects/myagy/
├── config.py             # Ports, endpoints, directory constants
├── ui.py                 # ANSI palette, styling, Rich console integration
├── coordinator.py        # NVIDIA CUDA:9001 resource arbitrator
├── subagent.py           # Background task delegation & context injection
├── compactor.py          # Slot metrics & continuous rolling hippocampus curation
├── repomap.py            # AST symbol mapping with fast mtime caching
├── instructions.py       # GEMINI.md / antigravity.md context discovery & injection
├── hooks_loader.py       # Lifecycle hooks loader (hooks.json Pre/Post tool gates)
├── laya.py               # Laya System 1 ModernBERT decision protocol
├── llamashift.py         # Port telemetry, model catalog & hot-swapping
├── mcp_loader.py         # Native, VS Code, and Kontexta MCP discovery
├── session.py            # MultiGpuHybridSession cross-GPU orchestrator
├── cli.py                # Multi-line prompt reader, slash commands & REPL
├── __init__.py           # Package exports
└── antigravity_agent.py  # Top-level executable runner & backward-compat wrapper
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.10+ and the required packages are installed:
```bash
pip install -r requirements.txt  # Or install google-antigravity, rich, httpx, etc.
```

### 2. Configure Bash Alias
To launch `myagy` from anywhere in your terminal, add the alias to your `~/.bashrc`:

```bash
# Append alias to ~/.bashrc
echo "alias myagy='python3 /home/safiyu/Projects/myagy/antigravity_agent.py --dangerously-skip-permissions'" >> ~/.bashrc

# Reload your shell environment
source ~/.bashrc
```

---

## 🚀 Quickstart & Usage

### Running the Agent
```bash
# Using the bash alias
myagy

# Or directly from the project directory
python3 /home/safiyu/Projects/myagy/antigravity_agent.py --dangerously-skip-permissions
```

### Slash Command Palette

| Command | Action |
| :--- | :--- |
| `/auto` | Enable Laya System 1 Non-Autoregressive routing (~33ms) |
| `/9000` or `/rocm` | Set active model to AMD ROCm GPU (Port 9000) |
| `/9001` or `/cuda` | Set active model to NVIDIA CUDA GPU (Port 9001) |
| `/cloud` or `/gemini` | Route queries through Google Cloud Gemini via OAuth |
| `/models` | List all local and cloud models in catalog |
| `/model <# or id>` | Hot-swap active model on the fly |
| `/spawn <task>` | Launch an autonomous background subagent on RTX 4060 |
| `/tasks` | List all active and completed background subagents |
| `/subagent view <#>` | Inspect the complete result and tool calls of a subagent |
| `/subagent inject <#>`| Inject subagent findings into current chat context |
| `/subagent cancel <#>`| Cancel a running background subagent task |
| `/cuda` | Show NVIDIA CUDA:9001 resource coordinator status |
| `/repomap [path]` | Render the rich AST symbol tree of codebase classes & functions |
| `/repomap on\|off` | Toggle automatic codebase outline injection into model memory |
| `/repomap refresh` | Force an immediate cache invalidation and re-index |
| `/instructions [view]` | Inspect or toggle auto-injected GEMINI.md / antigravity.md context |
| `/hooks [view\|refresh]` | Inspect, reload, or toggle lifecycle hooks (`hooks.json`) |
| `/compact` | Condense older conversation turns into dense Working Memory |
| `/curation on\|off` | Toggle continuous rolling background hippocampus on RTX 4060 |
| `/context` or `/ctx` | Display live context window telemetry and headroom progress bar |
| `/save [name]` | Save conversation session state to disk |
| `/load <name>` | Restore a previous conversation session |
| `/permissions auto` | Enable Laya dynamic safety gating (`/noul` P(Safe) ≥ 0.90) |
| `/reload` | Hot-reload code in-place preserving context |
| `/help` | Display full command reference |
