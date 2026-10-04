"""Subagent delegation engine: Asynchronously executes on-demand background engineering tasks on NVIDIA RTX 4060."""

import time
import json
import asyncio
import urllib.request
from typing import Optional, List, Dict, Any

from google.antigravity import (
    Agent,
    LocalOpenAIAgentConfig,
    CapabilitiesConfig,
    hooks,
    types,
)
from google.antigravity.hooks import policy

try:
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    pass

from .ui import UI, RICH_AVAILABLE, console
from .coordinator import CudaCoordinator
from .llamashift import trigger_llamashift_switch


class SubagentTask:
    """Represents an asynchronous background engineering task delegated to the NVIDIA accelerator."""

    def __init__(self, task_id: int, description: str):
        self.id = task_id
        self.description = description
        self.status = "pending"  # pending | running | completed | failed | cancelled
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.duration: Optional[float] = None
        self.result: Optional[str] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self.error: Optional[str] = None
        self.injected: bool = False
        self._async_task: Optional[asyncio.Task] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "duration": round(self.duration, 2) if self.duration is not None else None,
            "injected": self.injected,
            "result_preview": (self.result[:120] + "...") if self.result and len(self.result) > 120 else (self.result or ""),
        }


class SubagentManager:
    """Manages the lifecycle, execution queue, and context injection of background subagents."""

    _tasks: List[SubagentTask] = []
    _next_id: int = 1

    @classmethod
    def list_tasks(cls) -> List[SubagentTask]:
        return cls._tasks

    @classmethod
    def get(cls, task_id: int) -> Optional[SubagentTask]:
        for t in cls._tasks:
            if t.id == task_id:
                return t
        return None

    @classmethod
    def spawn(cls, description: str, session: Any) -> SubagentTask:
        """
        Spawns a new subagent task in the background on NVIDIA CUDA:9001.
        Preempts any active background compaction so the subagent gets immediate execution.
        """
        task = SubagentTask(task_id=cls._next_id, description=description)
        cls._next_id += 1
        cls._tasks.append(task)

        # Synchronously reserve CUDA 9001 lock immediately to pause background compaction
        CudaCoordinator.reserve_for_subagent()

        # Launch worker coroutine in background without blocking the interactive CLI
        task._async_task = asyncio.create_task(cls._run_worker(task, session))

        if not session.json_output:
            print(f"\n{UI.CUDA_BOLD}⚡ [Subagent #{task.id} Launched]{UI.RST} {UI.WHITE}{description}{UI.RST}")
            print(f"{UI.DARK_GRAY}   Target: NVIDIA RTX 4060 (:9001) │ Background Compactor paused. Type {UI.CYAN}/tasks{UI.DARK_GRAY} to check status.{UI.RST}\n", flush=True)

        return task

    @classmethod
    def cancel(cls, task_id: int) -> bool:
        task = cls.get(task_id)
        if not task:
            return False
        if task.status in ("pending", "running") and task._async_task and not task._async_task.done():
            task._async_task.cancel()
            task.status = "cancelled"
            task.finished_at = time.time()
            if task.started_at:
                task.duration = task.finished_at - task.started_at
            return True
        return False

    @classmethod
    def inject(cls, task_id: int, session: Any) -> bool:
        """Injects completed subagent findings directly into the main conversation history."""
        task = cls.get(task_id)
        if not task or not task.result or task.status != "completed":
            return False

        summary_entry = {
            "role": "user",
            "target": session.active_target,
            "content": f"[SYSTEM: Subagent #{task.id} Autonomous Findings on \"{task.description}\"]\n{task.result}",
        }
        companion_entry = {
            "role": "assistant",
            "target": session.active_target,
            "content": f"[SYSTEM: Acknowledged Subagent #{task.id} findings. Ready to apply insights.]",
        }

        session.history.extend([summary_entry, companion_entry])
        task.injected = True
        session._active_local_agent = None  # refresh agent context
        return True

    @classmethod
    async def _run_worker(cls, task: SubagentTask, session: Any):
        """Worker executing autonomously on NVIDIA RTX 4060 with tool access."""
        task.started_at = time.time()
        task.status = "running"

        try:
            # 2. Ensure Port 9001 is online (auto-wake via llama-launcher if needed)
            ep9001 = session.endpoints.get("9001", {})
            if ep9001.get("status") != "online":
                try:
                    start_res = trigger_llamashift_switch("granite42_8b")
                    if start_res.get("success"):
                        for _ in range(8):
                            await asyncio.sleep(0.5)
                            session.refresh_endpoints()
                            if session.endpoints.get("9001", {}).get("status") == "online":
                                ep9001 = session.endpoints["9001"]
                                break
                except Exception:
                    pass

            if ep9001.get("status") != "online":
                raise RuntimeError("NVIDIA RTX 4060 (Port 9001) is offline. Start it via llama-launcher or /compactor start.")

            # 3. Create tool hook to capture tool executions
            @hooks.pre_tool_call_decide
            def log_tool(call: types.ToolCall) -> types.HookResult:
                c_name = getattr(call, "name", str(call))
                c_args = getattr(call, "args", {}) or {}
                task.tool_calls.append({"name": c_name, "args": c_args, "timestamp": time.time()})
                return types.HookResult(allow=True)

            subagent_instructions = (
                "You are an expert autonomous sub-agent running on the NVIDIA RTX 4060 accelerator.\n"
                "Your objective is to accomplish this specific background assignment:\n\n"
                f"TASK: {task.description}\n\n"
                "OPERATIONAL GUIDELINES:\n"
                "1. Direct execution: Use tools (inspect files, search, check shell) as needed to resolve the task.\n"
                "2. Conclude with a dense, structured, factual summary of your findings and completed changes.\n"
                "3. Preserve file paths, function names, and exact numbers."
            )

            # 4. Execute on NVIDIA RTX 4060 (:9001) via high-speed chat completion
            payload = {
                "model": ep9001.get("filename") or "granite-4.2-8b",
                "messages": [
                    {"role": "system", "content": subagent_instructions},
                    {"role": "user", "content": task.description},
                ],
                "max_tokens": 1500,
                "temperature": 0.2,
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:9001/v1/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json"},
            )

            # Run in thread so asyncio event loop remains non-blocking for user
            def _call_http():
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            data = await asyncio.to_thread(_call_http)
            choice = data["choices"][0]["message"]
            res_text = (choice.get("content") or choice.get("reasoning_content") or "").strip()
            if not res_text and choice.get("reasoning_content"):
                res_text = choice.get("reasoning_content").strip()

            if res_text:
                task.result = res_text
            else:
                raise RuntimeError("Port 9001 returned empty response")


        except asyncio.CancelledError:
            task.status = "cancelled"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
        else:
            task.status = "completed"
        finally:
            task.finished_at = time.time()
            task.duration = round((task.finished_at - task.started_at), 2) if task.started_at else 0.0
            # 5. Release CUDA lock back to idle
            await CudaCoordinator.release_from_subagent()

            # 6. Notify user in terminal
            if not session.json_output:
                if task.status == "completed":
                    print(
                        f"\n{UI.GREEN_BOLD}✓ [Subagent #{task.id} Completed]{UI.RST} "
                        f"{UI.WHITE}{task.description[:55]}{UI.RST} "
                        f"{UI.GRAY}(in {task.duration:.1f}s via RTX 4060){UI.RST}\n"
                        f"{UI.DARK_GRAY}   ➔ Type {UI.CYAN}/subagent view {task.id}{UI.DARK_GRAY} to inspect result or {UI.CYAN}/subagent inject {task.id}{UI.DARK_GRAY} to add to session history.{UI.RST}\n",
                        flush=True,
                    )
                elif task.status == "failed":
                    print(
                        f"\n{UI.RED_BOLD}✗ [Subagent #{task.id} Failed]{UI.RST} "
                        f"{UI.WHITE}{task.description[:55]}{UI.RST}: {UI.RED}{task.error}{UI.RST}\n",
                        flush=True,
                    )

    @classmethod
    def print_tasks(cls, session: Any):
        """Renders an interactive status table of all background subagents."""
        if not cls._tasks:
            print(f"\n{UI.GRAY}(no background subagents spawned. Use /spawn <task> to delegate){UI.RST}\n")
            return

        coord_status = CudaCoordinator.status_summary()
        c_state = coord_status["state"].upper()
        if c_state == "IDLE":
            c_badge = f"{UI.GREEN}IDLE{UI.RST}"
        elif c_state == "SUBAGENT":
            c_badge = f"{UI.CUDA_BOLD}SUBAGENT RUNNING{UI.RST}"
        else:
            c_badge = f"{UI.AMBER}COMPACTING{UI.RST}"

        if RICH_AVAILABLE and not session.json_output:
            table = Table(
                title=Text.from_ansi(f"{UI.WHITE}BACKGROUND SUBAGENTS ON NVIDIA RTX 4060{UI.RST} │ Port 9001: {c_badge}"),
                box=box.ROUNDED,
                header_style="bold cyan",
                border_style="bright_black",
            )
            table.add_column("#", justify="center", style="bold dim", width=5)
            table.add_column("Status", style="bold", width=12)
            table.add_column("Duration", justify="right", width=9)
            table.add_column("Task Description", style="white")
            table.add_column("Result / Preview", style="dim")
            table.add_column("Injected", justify="center", width=8)

            for t in cls._tasks:
                if t.status == "completed":
                    st = f"{UI.GREEN}● DONE{UI.RST}"
                elif t.status == "running":
                    st = f"{UI.CUDA_BOLD}⚡ RUNNING{UI.RST}"
                elif t.status == "cancelled":
                    st = f"{UI.GRAY}○ CANCELLED{UI.RST}"
                elif t.status == "failed":
                    st = f"{UI.RED}✗ FAILED{UI.RST}"
                else:
                    st = f"{UI.AMBER}… PENDING{UI.RST}"

                dur_str = f"{t.duration:.1f}s" if t.duration is not None else (f"{time.time() - t.started_at:.1f}s" if t.started_at else "-")
                preview = (t.result or t.error or "").replace("\n", " ").strip()
                if len(preview) > 75:
                    preview = preview[:75] + "..."

                inj_tag = f"{UI.GREEN}YES{UI.RST}" if t.injected else f"{UI.GRAY}NO{UI.RST}"

                table.add_row(
                    str(t.id),
                    Text.from_ansi(st),
                    dur_str,
                    t.description[:60],
                    Text.from_ansi(preview),
                    Text.from_ansi(inj_tag),
                )

            console.print()
            console.print(table)
            console.print(f"{UI.GRAY}Commands: {UI.CYAN}/subagent view <#>{UI.GRAY} │ {UI.CYAN}/subagent inject <#>{UI.GRAY} │ {UI.CYAN}/subagent cancel <#>{UI.RST}\n")
        else:
            print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}BACKGROUND SUBAGENTS (NVIDIA CUDA:9001 [{c_badge}]){UI.RST}{UI.DARK_GRAY} ─────────────╮{UI.RST}")
            for t in cls._tasks:
                dur_str = f"{t.duration:.1f}s" if t.duration is not None else "-"
                inj_str = " [Injected]" if t.injected else ""
                print(f"{UI.DARK_GRAY}│{UI.RST}  #{t.id:2d} [{t.status.upper():9s}] ({dur_str:5s}) {UI.WHITE}{t.description[:45]}{UI.RST}{inj_str}")
            print(f"{UI.DARK_GRAY}╰──────────────────────────────────────────────────────────────────╯{UI.RST}\n")

    @classmethod
    def print_detail(cls, task_id: int):
        task = cls.get(task_id)
        if not task:
            print(UI.warn(f"Subagent #{task_id} not found. Type /tasks to view active tasks."))
            return

        dur_str = f"{task.duration:.1f}s" if task.duration is not None else (f"{time.time() - task.started_at:.1f}s" if task.started_at else "-")
        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}SUBAGENT #{task.id} DETAILS{UI.RST}{UI.DARK_GRAY} ───────────────────────────────────────────────╮{UI.RST}")
        print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Task{UI.RST}       : {task.description}")
        print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Status{UI.RST}     : {task.status.upper()} ({dur_str})")
        print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Engine{UI.RST}     : NVIDIA RTX 4060 (Port 9001 - IBM Granite 8B)")
        print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Injected{UI.RST}   : {'YES' if task.injected else 'NO'}")
        if task.tool_calls:
            print(f"{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Tool Calls{UI.RST} : {len(task.tool_calls)} executed")
            for c in task.tool_calls:
                print(f"{UI.DARK_GRAY}│{UI.RST}    • {UI.AMBER}{c['name']}{UI.RST}({json.dumps(c.get('args', {}))[:80]})")
        print(f"{UI.DARK_GRAY}│{UI.RST}\n{UI.DARK_GRAY}│{UI.RST}  {UI.WHITE}Result Content:{UI.RST}")
        res_text = task.result or task.error or "(no output yet)"
        for line in res_text.splitlines():
            print(f"{UI.DARK_GRAY}│{UI.RST}    {line}")
        print(f"{UI.DARK_GRAY}╰────────────────────────────────────────────────────────────────────────╯")
        if task.status == "completed" and not task.injected:
            print(f"{UI.GRAY}Tip: Use '/subagent inject {task.id}' to add this result to conversation context.{UI.RST}\n")
        else:
            print()
