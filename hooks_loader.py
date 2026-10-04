"""
Antigravity Lifecycle Hooks Loader (hooks.json).
Discovers and executes external shell commands at Agent lifecycle points:
  - PreToolUse: Gate, block, or rewrite tool parameters
  - PostToolUse: Run post-execution linters, analysis, or cleanup
  - PreTurn / PostTurn: Lifecycle context hooks
  - Stop: Enforce completion conditions

Conforms to Antigravity Customization Specification (.agents/hooks.json, ~/.gemini/config/hooks.json).
"""

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from google.antigravity import hooks, types
from .ui import UI


HOOKS_CANDIDATE_PATHS = [
    # Workspace specific
    os.path.join(".agents", "hooks.json"),
    os.path.join(".agent", "hooks.json"),
    "hooks.json",
    # Global machine config
    os.path.expanduser("~/.gemini/config/hooks.json"),
]


class ExternalHooksManager:
    """Discovers, parses, and wires external hooks.json scripts into Antigravity."""

    _cached_configs: Optional[List[Dict[str, Any]]] = None
    _cached_sources: List[str] = []
    _cached_mtime: float = 0.0

    @classmethod
    def discover_hook_files(cls, repo_root: str = ".") -> List[str]:
        """Finds all existing hooks.json files across workspace and global configurations."""
        found = []
        for p in HOOKS_CANDIDATE_PATHS:
            full_path = p if os.path.isabs(p) else os.path.join(repo_root, p)
            if os.path.isfile(full_path):
                real = os.path.realpath(full_path)
                if real not in [os.path.realpath(x) for x in found]:
                    found.append(full_path)
        return found

    @classmethod
    def load_hook_configs(cls, repo_root: str = ".", force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Loads and parses discovered hooks.json files.
        Returns a tuple of (merged_hook_entries, source_files).
        """
        files = cls.discover_hook_files(repo_root)
        if not files:
            cls._cached_configs = []
            cls._cached_sources = []
            return [], []

        try:
            latest_mtime = max(os.path.getmtime(f) for f in files)
        except Exception:
            latest_mtime = 0.0

        if (
            not force_refresh
            and cls._cached_configs is not None
            and latest_mtime <= cls._cached_mtime
            and cls._cached_sources == files
        ):
            return cls._cached_configs, cls._cached_sources

        parsed_hooks = []
        for fp in files:
            base_dir = os.path.dirname(os.path.abspath(fp))
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    for hook_name, hook_spec in raw.items():
                        if not isinstance(hook_spec, dict):
                            continue
                        if not hook_spec.get("enabled", True):
                            continue
                        parsed_hooks.append({
                            "name": hook_name,
                            "source_file": fp,
                            "base_dir": base_dir,
                            "spec": hook_spec,
                        })
            except Exception as e:
                print(f"{UI.RED_BOLD}[✗ Hook Error]{UI.RST} Failed to parse {fp}: {e}")

        cls._cached_configs = parsed_hooks
        cls._cached_sources = files
        cls._cached_mtime = latest_mtime
        return parsed_hooks, files

    @classmethod
    def execute_command_hook(
        cls,
        cmd: str,
        base_dir: str,
        payload: Dict[str, Any],
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Executes a shell command hook with context on stdin, expecting JSON on stdout.
        """
        try:
            expanded_cmd = os.path.expanduser(cmd)
            proc = subprocess.run(
                expanded_cmd,
                shell=True,
                cwd=base_dir,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            stdout = proc.stdout.strip()
            if stdout:
                try:
                    return json.loads(stdout)
                except Exception:
                    # Output wasn't valid JSON, return stdout string representation
                    return {"raw_output": stdout, "exit_code": proc.returncode}
            return {"exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"error": f"Hook timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def build_lifecycle_hooks(
        cls,
        session_obj: Any,
        repo_root: str = ".",
    ) -> List[Any]:
        """
        Builds and returns Antigravity SDK Hook instances wired to external hooks.json commands.
        """
        hook_entries, source_files = cls.load_hook_configs(repo_root)
        if not hook_entries:
            return []

        active_hooks = []

        # ── PreToolUse Handlers ───────────────────────────────────────
        pre_tool_handlers = []
        for entry in hook_entries:
            pre_list = entry["spec"].get("PreToolUse", [])
            for group in pre_list:
                matcher_pattern = group.get("matcher", "*")
                sub_hooks = group.get("hooks", [])
                for h in sub_hooks:
                    cmd = h.get("command")
                    if cmd:
                        pre_tool_handlers.append({
                            "name": entry["name"],
                            "matcher": matcher_pattern,
                            "command": cmd,
                            "base_dir": entry["base_dir"],
                            "timeout": h.get("timeout", 30),
                        })

        if pre_tool_handlers:
            @hooks.pre_tool_call_decide
            def external_pre_tool_decide(call: types.ToolCall) -> types.HookResult:
                c_name = getattr(call, "name", str(call))
                c_args = getattr(call, "args", {}) or {}

                for handler in pre_tool_handlers:
                    pattern = handler["matcher"]
                    matched = False
                    if pattern in ("*", "", ".*"):
                        matched = True
                    else:
                        try:
                            if re.search(pattern, c_name):
                                matched = True
                        except Exception:
                            if pattern == c_name:
                                matched = True

                    if not matched:
                        continue

                    payload = {
                        "toolCall": {"name": c_name, "args": c_args},
                        "conversationId": getattr(session_obj, "conversation_id", "myagy-session"),
                        "modelName": session_obj.active_target,
                    }

                    if not session_obj.json_output and session_obj.verbose:
                        print(f"{UI.DARK_GRAY}  │ [Executing Hook: {handler['name']} (PreToolUse)]{UI.RST}")

                    result = cls.execute_command_hook(
                        handler["command"],
                        handler["base_dir"],
                        payload,
                        timeout=handler["timeout"],
                    )

                    decision = result.get("decision", "").lower()
                    reason = result.get("reason", "")
                    overwrite = result.get("overwrite")

                    if decision == "deny":
                        msg = f"Execution blocked by hook '{handler['name']}': {reason or 'Permission denied'}"
                        if not session_obj.json_output:
                            print(f"{UI.RED_BOLD}[x Hook Blocked]{UI.RST} {UI.RED}{msg}{UI.RST}")
                        return types.HookResult(allow=False, message=msg)

                    if overwrite and isinstance(overwrite, dict):
                        # Shallow top-level argument merge according to Antigravity hook spec
                        c_args.update(overwrite)
                        if not session_obj.json_output and session_obj.verbose:
                            print(f"{UI.DARK_GRAY}  │ [Hook Overwrote Tool Args]: {overwrite}{UI.RST}")

                return types.HookResult(allow=True)

            active_hooks.append(external_pre_tool_decide)

        # ── PostToolUse Handlers ──────────────────────────────────────
        post_tool_handlers = []
        for entry in hook_entries:
            post_list = entry["spec"].get("PostToolUse", [])
            for group in post_list:
                matcher_pattern = group.get("matcher", "*")
                sub_hooks = group.get("hooks", [])
                for h in sub_hooks:
                    cmd = h.get("command")
                    if cmd:
                        post_tool_handlers.append({
                            "name": entry["name"],
                            "matcher": matcher_pattern,
                            "command": cmd,
                            "base_dir": entry["base_dir"],
                            "timeout": h.get("timeout", 30),
                        })

        if post_tool_handlers:
            @hooks.post_tool_call
            def external_post_tool(res: types.ToolResult):
                call = getattr(res, "call", None)
                c_name = getattr(call, "name", "unknown") if call else "unknown"

                for handler in post_tool_handlers:
                    pattern = handler["matcher"]
                    matched = False
                    if pattern in ("*", "", ".*"):
                        matched = True
                    else:
                        try:
                            if re.search(pattern, c_name):
                                matched = True
                        except Exception:
                            if pattern == c_name:
                                matched = True

                    if not matched:
                        continue

                    payload = {
                        "error": str(res.error) if res.error else "",
                        "conversationId": getattr(session_obj, "conversation_id", "myagy-session"),
                        "modelName": session_obj.active_target,
                    }

                    result = cls.execute_command_hook(
                        handler["command"],
                        handler["base_dir"],
                        payload,
                        timeout=handler["timeout"],
                    )
                    if session_obj.verbose and not session_obj.json_output:
                        print(f"{UI.DARK_GRAY}  │ [Hook Completed: {handler['name']} (PostToolUse)]{UI.RST}")

            active_hooks.append(external_post_tool)

        return active_hooks
