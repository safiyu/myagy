"""Context window telemetry, NVIDIA RTX 4060 Working Memory synthesis, and continuous curation."""

import json
import time
import asyncio
import urllib.request
from typing import Optional, List, Dict, Any

from ..terminal.ui import UI, RICH_AVAILABLE, console
from ..core.llamashift import trigger_llamashift_switch
from ..core.coordinator import CudaCoordinator


class ContextCurator:
    """Manages context telemetry, slot tracking, and continuous working memory synthesis on NVIDIA CUDA."""

    @staticmethod
    def get_context_metrics(session: Any, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves exact or estimated context window size, used tokens, and percentage.
        Queries llama-server /slots if local GPU, or estimates via history + instructions.
        """
        tgt = session.resolve_target_alias(target or session.active_target)
        n_tokens = None
        max_ctx = None

        if tgt in ("9000", "9001"):
            port = session.endpoints[tgt]["port"]
            try:
                req = urllib.request.Request(f"http://localhost:{port}/slots", headers={"User-Agent": "Antigravity/1.0"})
                with urllib.request.urlopen(req, timeout=0.3) as resp:
                    slots_data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(slots_data, list) and len(slots_data) > 0:
                        slot0 = slots_data[0]
                        prompt_tok = slot0.get("n_prompt_tokens", 0)
                        cache_tok = slot0.get("n_prompt_tokens_cache", 0)
                        n_tokens = max(prompt_tok, cache_tok)
                        max_ctx = slot0.get("n_ctx")
            except Exception:
                pass

        if n_tokens is None or n_tokens == 0:
            sys_len = len(session.build_system_context(tgt, include_history=False))
            hist_len = sum(len(h.get("content", "")) for h in session.history)
            n_tokens = int((sys_len + hist_len) / 3.6)
            if tgt in ("9000", "9001"):
                max_ctx = 131072
            else:
                max_ctx = 1048576  # Gemini 1M context

        pct = (n_tokens / max_ctx * 100.0) if (max_ctx and max_ctx > 0) else 0.0
        return {
            "tokens": n_tokens,
            "max_ctx": max_ctx,
            "percentage": pct,
            "headroom": max(0, max_ctx - n_tokens) if max_ctx else 0,
            "is_exact": (tgt in ("9000", "9001") and max_ctx is not None),
        }

    @classmethod
    def format_context_badge(cls, session: Any, target: Optional[str] = None, reset_col: str = UI.RST) -> str:
        ctx = cls.get_context_metrics(session, target)
        tokens = ctx["tokens"]
        max_ctx = ctx["max_ctx"]
        pct = ctx["percentage"]

        t_str = f"{tokens/1000:.1f}k" if tokens >= 1000 else str(tokens)
        m_str = f"{max_ctx/1000:.1f}k" if max_ctx >= 1000 else str(max_ctx)

        if pct >= 85:
            col = UI.RED_BOLD
        elif pct >= 65:
            col = UI.AMBER_BOLD
        else:
            col = UI.CYAN

        return f"{col}Context: {t_str}/{m_str} ({pct:.0f}%){reset_col}"

    @classmethod
    def format_context_pill(cls, session: Any, target: Optional[str] = None) -> str:
        ctx = cls.get_context_metrics(session, target)
        tokens = ctx["tokens"]
        pct = ctx["percentage"]
        t_str = f"{tokens/1000:.1f}k" if tokens >= 1000 else str(tokens)
        if pct >= 85:
            col = UI.RED_BOLD
        elif pct >= 65:
            col = UI.AMBER
        else:
            col = UI.CYAN
        return f"{col}{t_str} ctx{UI.RST}"

    @classmethod
    def print_context_summary(cls, session: Any):
        """Displays a visual breakdown of active context size, headroom, and compactor status."""
        ctx = cls.get_context_metrics(session, session.active_target)
        tokens = ctx["tokens"]
        max_ctx = ctx["max_ctx"]
        pct = ctx["percentage"]
        headroom = ctx["headroom"]

        width = 24
        filled = int(round(width * (pct / 100.0)))
        filled = min(width, max(0, filled))
        bar = "█" * filled + "░" * (width - filled)

        if pct >= 85:
            bar_col = UI.RED_BOLD
        elif pct >= 65:
            bar_col = UI.AMBER_BOLD
        else:
            bar_col = UI.CYAN_BOLD

        target_lbl = session.target_name(session.active_target)
        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}CONTEXT WINDOW TELEMETRY ({target_lbl}){UI.RST}{UI.DARK_GRAY} ────────────────────╮{UI.RST}")
        print(f"{UI.DARK_GRAY}│{UI.RST}  Usage      : {bar_col}[{bar}]{UI.RST} {pct:.1f}%")
        print(f"{UI.DARK_GRAY}│{UI.RST}  Active     : {UI.WHITE}{tokens:,}{UI.RST} tokens  /  {UI.GRAY}{max_ctx:,}{UI.RST} limit")
        print(f"{UI.DARK_GRAY}│{UI.RST}  Headroom   : {UI.GREEN}{headroom:,}{UI.RST} tokens remaining")
        print(f"{UI.DARK_GRAY}│{UI.RST}  History    : {len(session.history)} recorded turns ({len(session.history)//2} exchanges)")
        ac_st = f"{UI.GREEN}ENABLED (every {session.autocompact_threshold} turns){UI.RST}" if session.autocompact_threshold > 0 else f"{UI.GRAY}DISABLED{UI.RST}"
        print(f"{UI.DARK_GRAY}│{UI.RST}  AutoCompact: {ac_st}")
        cuda_syn_st = (
            f"{UI.GREEN}ONLINE (NVIDIA RTX 4060 :9001 Granite-8B){UI.RST}"
            if session.endpoints.get("9001", {}).get("status") == "online"
            else f"{UI.GRAY}Standby / Offline (Launch via /home/safiyu/myagy/launch_cuda_compactor.sh or /compactor start){UI.RST}"
        )
        print(f"{UI.DARK_GRAY}│{UI.RST}  Compactor  : {cuda_syn_st}")
        curation_st = f"{UI.GREEN}ACTIVE (Continuous Rolling Hippocampus){UI.RST}" if session.continuous_curation else f"{UI.GRAY}OFF{UI.RST}"
        print(f"{UI.DARK_GRAY}│{UI.RST}  Curation   : {curation_st}")
        print(f"{UI.DARK_GRAY}╰─────────────────────────────────────────────────────────────╯{UI.RST}")
        print(f"{UI.GRAY}Tip: Use /compact to condense history, or /curation [on|off] to toggle rolling synthesis.{UI.RST}\n")

    @staticmethod
    def synthesize_context_cuda_sync(session: Any, turns: List[Dict[str, str]]) -> Optional[str]:
        """
        Sends older turns to Port 9001 (NVIDIA CUDA GPU) to generate high-density Working Memory.
        Auto-wakes via Llama-Launcher if offline.
        """
        if not turns:
            return None
        if CudaCoordinator.state() == "subagent":
            # Defer to running subagent - do not compete for CUDA slots
            return None
        ep9001 = session.endpoints.get("9001", {})
        if ep9001.get("status") != "online":
            try:
                start_res = trigger_llamashift_switch("granite42_8b")
                if start_res.get("success"):
                    for _ in range(8):
                        time.sleep(0.5)
                        session.refresh_endpoints()
                        if session.endpoints.get("9001", {}).get("status") == "online":
                            ep9001 = session.endpoints["9001"]
                            break
            except Exception:
                pass
        if ep9001.get("status") != "online":
            return None

        log_parts = []
        for t in turns:
            r = t.get("role", "unknown").upper()
            tgt = t.get("target", "").upper()
            c = t.get("content", "").strip()
            if "[Working Memory distilled" in c or "[SYSTEM: Context compacted" in c:
                log_parts.append(f"[PRIOR WORKING MEMORY BASELINE]:\n{c}")
            else:
                if len(c) > 1500:
                    c = c[:1400] + "... [truncated]"
                log_parts.append(f"[{r} on {tgt}]:\n{c}")

        conversation_log = "\n---\n".join(log_parts)
        system_instruction = (
            "You are a Software Engineering Context Synthesizer running on the NVIDIA accelerator.\n"
            "Distill the prior baseline memory and new conversation turns into an updated, dense Working Memory block.\n"
            "Structure strictly with:\n"
            "### Goals & Requirements\n"
            "### Architectural Decisions & Changes\n"
            "### Files Touched & Current Status\n"
            "### Next Immediate Steps\n"
            "Be factual, concise, and preserve filenames, functions, and error messages."
        )

        payload = {
            "model": ep9001.get("filename") or "granite-4.2-8b",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Synthesize this session history:\n\n{conversation_log}"},
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:9001/v1/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]["message"]
                res_text = choice.get("content") or choice.get("reasoning_content") or ""
                return res_text.strip() if res_text.strip() else None
        except Exception:
            return None

    @classmethod
    async def async_synthesize_cuda(cls, session: Any):
        """Background coroutine that pre-computes Working Memory synthesis on CUDA:9001 only when idle."""
        if not CudaCoordinator.start_compaction(asyncio.current_task()):
            return
        try:
            if len(session.history) <= session.max_recent_turns:
                return
            older = session.history[:-session.max_recent_turns]
            syn = await asyncio.to_thread(cls.synthesize_context_cuda_sync, session, older)
            if syn:
                session._cached_curated_memory = {
                    "summary": syn,
                    "covered_turns": len(older),
                    "timestamp": time.time(),
                }
                if session.continuous_curation and len(session.history) >= (session.max_recent_turns + 2):
                    removed = cls.compact_history(session, keep_recent=session.max_recent_turns, silent=True)
                    if removed > 0 and not session.json_output:
                        print(f"\n{UI.DARK_GRAY}⚡ [Hippocampus: {removed} older turns distilled into Working Memory via RTX 4060]{UI.RST}\n", flush=True)
        except asyncio.CancelledError:
            # Preempted by on-demand subagent task
            pass
        except Exception:
            pass
        finally:
            CudaCoordinator.finish_compaction()

    @classmethod
    def trigger_background_synthesis(cls, session: Any):
        """Fires non-blocking background synthesis task on CUDA:9001 only when idle and not busy with subagents."""
        if not session.continuous_curation:
            return
        if not CudaCoordinator.can_compact():
            return
        if session._synthesis_task and not session._synthesis_task.done():
            return
        if len(session.history) > session.max_recent_turns:
            # The coroutine takes the coordinator slot itself once it starts
            session._synthesis_task = asyncio.create_task(cls.async_synthesize_cuda(session))

    @classmethod
    def summarize_history_sync(cls, session: Any, turns: List[Dict[str, str]]) -> str:
        """Extracts salient points from older turns. Tries CUDA synthesis first, then falls back."""
        if not turns:
            return ""
        syn = cls.synthesize_context_cuda_sync(session, turns)
        if syn:
            return syn

        summary_points = []
        for t in turns:
            role = t.get("role", "unknown").upper()
            target = t.get("target", "").upper()
            content = t.get("content", "").strip()
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if lines:
                preview = lines[0][:140]
                summary_points.append(f"• [{role} on {target}]: {preview}")
        return "\n".join(summary_points)

    @classmethod
    def compact_history(cls, session: Any, keep_recent: Optional[int] = None, silent: bool = False) -> int:
        """
        Compact conversation history into structured Working Memory using NVIDIA RTX 4060 (:9001).
        Preserves keep_recent turns verbatim.
        """
        keep = keep_recent if keep_recent is not None else session.max_recent_turns
        if len(session.history) <= keep:
            if not silent:
                print(UI.warn(f"Only {len(session.history)} turn(s) in history — nothing to compact."))
            return 0

        older = session.history[:-keep] if keep > 0 else session.history
        recent = session.history[-keep:] if keep > 0 else []
        removed = len(older)

        synthesized_text = None
        synthesized_by = None
        if session._cached_curated_memory and session._cached_curated_memory.get("covered_turns") == removed:
            synthesized_text = session._cached_curated_memory["summary"]
            synthesized_by = "NVIDIA RTX 4060 (pre-computed cache)"
        elif session.endpoints.get("9001", {}).get("status") == "online":
            if not silent and not session.json_output:
                print(f"{UI.CUDA_BOLD}[⚡ CUDA:9001]{UI.RST} {UI.GRAY}Synthesizing working memory on NVIDIA RTX 4060...{UI.RST}")
            synthesized_text = cls.synthesize_context_cuda_sync(session, older)
            if synthesized_text:
                synthesized_by = "NVIDIA RTX 4060 (:9001)"

        if synthesized_text:
            summary_content = f"[Working Memory distilled by {synthesized_by}]\n{synthesized_text}"
        else:
            summary_lines = ["[Compact Context Summary — earlier turns condensed]"]
            for t in older:
                role = t.get("role", "?").upper()
                tgt = t.get("target", "?").upper()
                content = t.get("content", "").strip()
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                preview = lines[0][:180] if lines else "(empty)"
                summary_lines.append(f"• [{role} on {tgt}]: {preview}")
            summary_content = "\n".join(summary_lines)

        summary_entry = {
            "role": "user",
            "target": recent[0]["target"] if recent else session.active_target,
            "content": f"[SYSTEM: Context compacted. {removed} prior turn(s) distilled below]\n{summary_content}",
        }
        companion_entry = {
            "role": "assistant",
            "target": summary_entry["target"],
            "content": "[SYSTEM: Acknowledged. I have the distilled prior context and will continue from here.]",
        }

        session.history = [summary_entry, companion_entry] + recent
        session._active_local_agent = None
        session._cached_curated_memory = None

        if not silent:
            label_by = f" via {synthesized_by}" if synthesized_by else ""
            if RICH_AVAILABLE and not session.json_output:
                console.print(f"\n[bold cyan]✓ History compacted{label_by}:[/bold cyan] [dim]{removed} old turn(s) → working memory. Kept {len(recent)} recent verbatim.[/dim]\n")
            else:
                print(UI.ok(f"History compacted{label_by}: {removed} old turns → working memory. {len(recent)} recent turns kept verbatim."))

        return removed
