"""
Spec-Driven Autonomous Lifecycle Workflow (/brainstorm):
  1. Cloud (Gemini OAuth): Gathers inputs, codebase AST & guidelines -> drafts implementation plan.
  2. Gate 1: Interactive user approval/refinement of the plan.
  3. Local (AMD ROCm :9000): Executes implementation edits milestone by milestone.
  4. Build & Test: Automatically runs project tests/build commands and self-heals errors.
  5. Local Code Review: Analyzes git diff on local models.
  6. Cloud Adversarial Review (Gemini): Scrutinizes edge cases, security vulnerabilities, and bugs.
  7. Gate 2: User approval of reviewed changes -> automated git commit.
"""

import os
import re
import time
import subprocess
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

try:
    from rich.panel import Panel
    from rich.markdown import Markdown
except ImportError:
    pass

from ..terminal.ui import UI, RICH_AVAILABLE, console
from ..terminal.esc_listener import prompt_input
from ..config import PLANS_DIR
from ..context.repomap import RepoMap
from ..context.instructions import ProjectInstructions


class BrainstormWorkflow:
    """State machine governing end-to-end spec brainstorming, implementation, dual review, and commit."""

    @classmethod
    def _slugify(cls, text: str) -> str:
        s = re.sub(r"[^\w\s-]", "", text).strip().lower()
        return re.sub(r"[-\s]+", "_", s)[:40] or "task_plan"

    @classmethod
    def _detect_test_command(cls, repo_root: str = ".") -> Optional[str]:
        """Detects standard build/test runners present in the active repository."""
        if os.path.isfile(os.path.join(repo_root, "package.json")):
            try:
                import json
                with open(os.path.join(repo_root, "package.json"), "r") as f:
                    pkg = json.load(f)
                scripts = pkg.get("scripts", {})
                if "test" in scripts:
                    return "pnpm test" if os.path.isfile(os.path.join(repo_root, "pnpm-lock.yaml")) else "npm test"
                if "build" in scripts:
                    return "pnpm build" if os.path.isfile(os.path.join(repo_root, "pnpm-lock.yaml")) else "npm run build"
            except Exception:
                pass

        if os.path.isfile(os.path.join(repo_root, "pytest.ini")) or os.path.isfile(os.path.join(repo_root, "pyproject.toml")) or os.path.isdir(os.path.join(repo_root, "tests")):
            return "pytest"

        if os.path.isfile(os.path.join(repo_root, "Cargo.toml")):
            return "cargo test"

        if os.path.isfile(os.path.join(repo_root, "Makefile")):
            return "make test"

        return None

    @classmethod
    def _get_git_diff(cls, repo_root: str = ".") -> str:
        """Returns the staged/unstaged git diff plus synthetic diffs for untracked files."""
        try:
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = [res.stdout.strip()] if res.stdout.strip() else []
            for rel in sorted(cls._git_paths(repo_root, untracked_only=True)):
                try:
                    with open(os.path.join(repo_root, rel), "r", encoding="utf-8") as f:
                        body = f.read(20000)
                except Exception:
                    continue
                added = "\n".join("+" + ln for ln in body.splitlines())
                parts.append(f"diff --git a/{rel} b/{rel}\nnew file\n--- /dev/null\n+++ b/{rel}\n{added}")
            return "\n".join(parts).strip()
        except Exception:
            return ""

    @classmethod
    def _git_paths(cls, repo_root: str = ".", untracked_only: bool = False) -> set:
        """Lists modified, deleted and untracked paths (ignored files excluded)."""
        flags = ["-o"] if untracked_only else ["-m", "-o", "-d"]
        try:
            res = subprocess.run(
                ["git", "ls-files", *flags, "--exclude-standard"],
                cwd=repo_root, capture_output=True, text=True, timeout=10,
            )
            return {ln for ln in res.stdout.splitlines() if ln.strip()}
        except Exception:
            return set()

    @classmethod
    async def execute_workflow(cls, idea: str, session: Any, repo_root: str = "."):
        """Executes the full spec-driven brainstorm-to-commit lifecycle."""
        if not idea.strip():
            print(UI.warn("Usage: /brainstorm <feature, refactor, or bugfix idea>"))
            return

        slug = cls._slugify(idea)
        os.makedirs(PLANS_DIR, exist_ok=True)
        plan_path = os.path.join(PLANS_DIR, f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

        # Files already dirty before the run are excluded from the final commit
        dirty_before = cls._git_paths(repo_root)
        if dirty_before:
            print(UI.warn(f"{len(dirty_before)} file(s) already modified before this run will be left out of the commit."))

        print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}SPEC-DRIVEN WORKFLOW INITIALIZED{UI.RST}{UI.DARK_GRAY} ─────────────────────────────╮{UI.RST}")
        print(f"{UI.DARK_GRAY}│{UI.RST}  Idea   : {UI.CYAN}{idea}{UI.RST}")
        print(f"{UI.DARK_GRAY}│{UI.RST}  Phases : Cloud Plan ➔ Approval ➔ ROCm Build ➔ Dual Review ➔ Commit")
        print(f"{UI.DARK_GRAY}╰────────────────────────────────────────────────────────────────────────╯\n")

        # ── PHASE 1: Cloud Brainstorming & Spec Drafting ──────────────────
        print(f"{UI.CLOUD_BOLD}[Phase 1/5: Cloud Brainstorming & Plan Drafting via Gemini]{UI.RST}")
        ast_summary = RepoMap.get_cached_map(repo_root) or "No AST map available."
        guidelines, loaded_guides = ProjectInstructions.load_instructions(repo_root)
        guide_context = f"\n\nProject Guidelines ({', '.join(loaded_guides)}):\n{guidelines}" if guidelines else ""

        brainstorm_prompt = (
            f"You are the Lead Architect. We need to create a complete implementation spec and milestone plan for:\n"
            f"OBJECTIVE: {idea}\n\n"
            f"Codebase AST Map:\n{ast_summary}\n"
            f"{guide_context}\n\n"
            "Format your response as a rigorous, structured Markdown Spec with:\n"
            "1. Architecture & Approach (Trade-offs & Decisions)\n"
            "2. Files to Modify & Files to Create\n"
            "3. Step-by-Step Milestones (concrete, executable)\n"
            "4. Risk Mitigation & Verification Strategy\n"
            "Begin directly with the Markdown plan. Do not include conversational filler."
        )

        plan_content = await session.chat_cloud_oauth(brainstorm_prompt, with_project_context=False)
        if not plan_content.strip():
            print(UI.err("Failed to generate plan from Cloud Gemini."))
            return

        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(plan_content)

        print(f"\n{UI.GREEN_BOLD}✓ Plan generated and saved to {plan_path}{UI.RST}\n")

        # ── GATE 1: Human Approval of Plan ───────────────────────────────
        if RICH_AVAILABLE and not session.json_output:
            console.print(Panel(Markdown(plan_content), title="[bold cyan]Implementation Spec & Milestone Plan[/bold cyan]", border_style="cyan"))
        else:
            print(f"\n{plan_content}\n")

        print(f"{UI.AMBER_BOLD}╭─── GATE 1: SPEC APPROVAL ──────────────────────────────────────────╮{UI.RST}")
        print(f"{UI.AMBER_BOLD}│{UI.RST}  [y] Proceed to local implementation on AMD ROCm (:9000)")
        print(f"{UI.AMBER_BOLD}│{UI.RST}  [n] Abort workflow")
        print(f"{UI.AMBER_BOLD}╰────────────────────────────────────────────────────────────────────╯{UI.RST}")

        try:
            choice = (await asyncio.to_thread(prompt_input, f"{UI.AMBER_BOLD}Approve plan for implementation? [y/N]: {UI.RST}")).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(UI.warn("\nWorkflow aborted by user."))
            return

        if choice not in ("y", "yes"):
            print(UI.warn("Plan not approved. Workflow cancelled. You can review or edit the plan in: " + plan_path))
            return

        # ── PHASE 2: Local Implementation on AMD ROCm (:9000) ─────────────
        print(f"\n{UI.ROCM_BOLD}[Phase 2/5: Local Implementation on AMD ROCm (:9000)]{UI.RST}")
        session.active_target = "9000"
        implement_prompt = (
            f"Here is the approved implementation plan:\n\n{plan_content}\n\n"
            f"TASK: Implement the required changes in the codebase right now. Use tools (view_file, edit_file, write_to_file, run_command) "
            f"directly to write and modify all necessary files. Do not stop until all code is implemented."
        )

        await session.chat_turn(implement_prompt, target="9000")

        # ── PHASE 3: Automatic Build & Test Verification ─────────────────
        print(f"\n{UI.CUDA_BOLD}[Phase 3/5: Verification & Automated Self-Healing Tests]{UI.RST}")
        test_cmd = cls._detect_test_command(repo_root)
        if test_cmd:
            print(f"{UI.GRAY}Detected test suite command: {UI.WHITE}{test_cmd}{UI.RST}")
            try:
                proc = await asyncio.create_subprocess_shell(
                    test_cmd,
                    cwd=repo_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                out_text = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")

                if proc.returncode == 0:
                    print(f"{UI.GREEN_BOLD}✓ Automated tests PASSED cleanly.{UI.RST}")
                else:
                    print(f"{UI.AMBER_BOLD}⚠ Test failures detected. Engaging ROCm to fix issues...{UI.RST}")
                    fix_prompt = (
                        f"The automated test command '{test_cmd}' failed with return code {proc.returncode}:\n\n"
                        f"--- [TEST OUTPUT] ---\n{out_text[-2500:]}\n--- [END OUTPUT] ---\n\n"
                        f"Inspect the failure, edit the code using your tools, and resolve the issue."
                    )
                    await session.chat_turn(fix_prompt, target="9000")
            except Exception as test_err:
                print(f"{UI.GRAY}[Test runner skipped: {test_err}]{UI.RST}")
        else:
            print(f"{UI.GRAY}No automated test script detected. Skipping Phase 3 test run.{UI.RST}")

        # ── PHASE 4: Local Code Review ────────────────────────────────────
        diff_text = cls._get_git_diff(repo_root)
        if not diff_text:
            print(UI.warn("No git changes detected after implementation phase."))
            return

        print(f"\n{UI.CUDA_BOLD}[Phase 4/5: Local Functional Code Review on AMD ROCm (:9000)]{UI.RST}")
        local_review_prompt = (
            f"Perform a concise functional code review of this git diff:\n\n```diff\n{diff_text[:6000]}\n```\n\n"
            f"Check for:\n"
            f"1. Logic bugs or syntax regressions\n"
            f"2. Edge case handling\n"
            f"3. Style or formatting issues\n"
            f"Provide a short bulleted list of issues or verify that the changes are sound."
        )
        await session.chat_turn(local_review_prompt, target="9000")

        # ── PHASE 5: Cloud Adversarial Second-Opinion Code Review ─────────
        print(f"\n{UI.CLOUD_BOLD}[Phase 5/5: Cloud Adversarial Second-Opinion Review via Gemini]{UI.RST}")
        cloud_review_prompt = (
            f"You are the Principal Security & Architecture Reviewer. Scrutinize the following git diff for a feature titled '{idea}':\n\n"
            f"```diff\n{diff_text[:7000]}\n```\n\n"
            f"Audit for:\n"
            f"1. Subtle boundary bugs or silent failure modes\n"
            f"2. Security vulnerabilities\n"
            f"3. Performance regressions\n"
            f"If critical bugs exist, list the exact fix. If clean, state 'VERIFIED_CLEAN'."
        )
        cloud_review = await session.chat_cloud_oauth(cloud_review_prompt, with_project_context=False)

        if "VERIFIED_CLEAN" not in cloud_review.upper() and len(cloud_review.strip()) > 30:
            print(f"\n{UI.AMBER_BOLD}Cloud Reviewer identified potential improvements or edge cases:{UI.RST}\n")
            print(cloud_review)
            try:
                apply_fixes = (await asyncio.to_thread(prompt_input, f"\n{UI.AMBER_BOLD}Apply Cloud Reviewer fixes locally on ROCm? [y/N]: {UI.RST}")).strip().lower()
                if apply_fixes in ("y", "yes"):
                    print(f"{UI.ROCM_BOLD}Applying reviewer recommendations...{UI.RST}")
                    fix_cloud_prompt = (
                        f"The Cloud Reviewer provided these recommendations on our changes:\n\n{cloud_review}\n\n"
                        f"Use tools to apply these fixes to the files."
                    )
                    await session.chat_turn(fix_cloud_prompt, target="9000")
            except (KeyboardInterrupt, EOFError):
                pass
        else:
            print(f"{UI.GREEN_BOLD}✓ Cloud Reviewer confirmed changes are sound.{UI.RST}")

        # ── GATE 2: Git Commit Approval ───────────────────────────────────
        final_diff = cls._get_git_diff(repo_root)
        diff_lines = len(final_diff.splitlines())
        print(f"\n{UI.GREEN_BOLD}╭─── WORKFLOW COMPLETE: READY TO COMMIT ({diff_lines} lines changed) ─────────╮{UI.RST}")
        commit_msg = f"feat({slug[:20]}): {idea}"
        print(f"{UI.DARK_GRAY}│{UI.RST}  Proposed Commit Message: {UI.WHITE}{commit_msg}{UI.RST}")
        print(f"{UI.DARK_GRAY}╰────────────────────────────────────────────────────────────────────────╯{UI.RST}")

        try:
            do_commit = (await asyncio.to_thread(prompt_input, f"{UI.GREEN_BOLD}Commit these changes to git? [y/N]: {UI.RST}")).strip().lower()
            if do_commit in ("y", "yes"):
                to_add = sorted(cls._git_paths(repo_root) - dirty_before)
                if not to_add:
                    print(UI.warn("No new changes to commit."))
                    return
                subprocess.run(["git", "add", "-A", "--", *to_add], cwd=repo_root, check=True)
                subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_root, check=True)
                print(UI.ok(f"Changes successfully committed: '{commit_msg}'"))
            else:
                print(UI.warn("Changes left in working directory uncommitted."))
        except Exception as e:
            print(UI.err(f"Commit operation failed or skipped: {e}"))
