"""CUDA Resource Coordinator: Arbitrates Port 9001 between On-Demand Subagents and Background Compaction."""

import time
import asyncio
from typing import Optional, Any


class CudaCoordinator:
    """
    Arbitrates usage of NVIDIA CUDA:9001 (RTX 4060) between:
      1. On-Demand Subagents (High Priority): Executes user-requested background tasks.
      2. Continuous Context Compactor (Low Priority): Distills older turns into working memory only when idle.
    """
    _state: str = "idle"  # "idle" | "compacting" | "subagent"
    _active_subagents: int = 0
    _compactor_task: Optional[asyncio.Task] = None
    _state_lock = asyncio.Lock()
    _last_state_change: float = time.time()

    @classmethod
    def state(cls) -> str:
        return cls._state

    @classmethod
    def is_idle(cls) -> bool:
        return cls._state == "idle" and cls._active_subagents == 0

    @classmethod
    def can_compact(cls) -> bool:
        """Compactor is only allowed to run when CUDA 9001 is completely idle and no subagent is running/queued."""
        return cls._state == "idle" and cls._active_subagents == 0

    @classmethod
    def start_compaction(cls, task: Optional[asyncio.Task] = None) -> bool:
        """Attempts to lock Port 9001 for background compaction. Fails if a subagent is active."""
        if not cls.can_compact():
            return False
        cls._state = "compacting"
        cls._compactor_task = task
        cls._last_state_change = time.time()
        return True

    @classmethod
    def finish_compaction(cls):
        """Releases Port 9001 back to idle after compaction completes."""
        if cls._state == "compacting":
            cls._state = "idle"
            cls._compactor_task = None
            cls._last_state_change = time.time()

    @classmethod
    def reserve_for_subagent(cls):
        """
        Synchronously reserves CUDA 9001 immediately upon subagent spawn,
        preempting any active background compaction task and locking out new compaction.
        """
        if cls._compactor_task and not cls._compactor_task.done():
            cls._compactor_task.cancel()
            cls._compactor_task = None
        cls._active_subagents += 1
        cls._state = "subagent"
        cls._last_state_change = time.time()

    @classmethod
    async def acquire_for_subagent(cls) -> bool:
        """
        Sub-agents have highest priority.
        If a background compaction task is currently running, it is preempted/cancelled immediately
        to dedicate 100% of NVIDIA RTX 4060 compute to the subagent task.
        """
        async with cls._state_lock:
            if cls._state == "compacting" and cls._compactor_task and not cls._compactor_task.done():
                # Preempt low-priority background compactor
                cls._compactor_task.cancel()
                try:
                    await asyncio.sleep(0.05)
                except Exception:
                    pass
                cls._compactor_task = None

            cls._active_subagents += 1
            cls._state = "subagent"
            cls._last_state_change = time.time()
            return True

    @classmethod
    async def release_from_subagent(cls):
        """Releases Port 9001 from a subagent task. Returns to idle once all subagents finish."""
        async with cls._state_lock:
            cls._active_subagents = max(0, cls._active_subagents - 1)
            if cls._active_subagents == 0:
                cls._state = "idle"
            cls._last_state_change = time.time()

    @classmethod
    def status_summary(cls) -> dict:
        return {
            "state": cls._state,
            "active_subagents": cls._active_subagents,
            "compactor_running": (cls._state == "compacting"),
            "can_compact": cls.can_compact(),
            "uptime_in_state": round(time.time() - cls._last_state_change, 1),
        }
