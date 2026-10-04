"""Instant ESC-key interruption listener for terminal streaming and tool execution.

Similar to Claude Code and Antigravity CLI, this listener monitors stdin in cbreak
mode during model generation or tool execution. Tapping the ESC key immediately
aborts the active turn without needing Ctrl+C or triggering OS SIGINT signals.
"""

import os
import sys
import select
import threading
from typing import Callable, Optional

try:
    import termios
    import tty
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False


class EscListener:
    """
    Context manager that listens for the standalone ESC key (ASCII 27 / \\x1b).

    Usage:
        listener = EscListener(on_escape=lambda: current_task.cancel())
        with listener:
            await current_task
        if listener.interrupted:
            print("Interrupted by user.")
    """

    def __init__(self, on_escape: Optional[Callable[[], None]] = None):
        self.on_escape = on_escape
        self.interrupted: bool = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_settings = None
        self._fd: Optional[int] = None

    def __enter__(self):
        self.interrupted = False
        self._stop_event.clear()

        # Check if stdin is an interactive terminal and termios is available
        if not TERMIOS_AVAILABLE or not sys.stdin.isatty():
            return self

        try:
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)

            # Configure terminal: cbreak mode + disable ECHO
            # - ICANON disabled: characters available immediately without pressing Enter
            # - ECHO disabled: raw escape characters like ^[ don't print to the console
            # - ISIG preserved: Ctrl+C / SIGINT still functions as a secondary safety net
            new_settings = termios.tcgetattr(self._fd)
            new_settings[3] = new_settings[3] & ~termios.ICANON
            new_settings[3] = new_settings[3] & ~termios.ECHO
            new_settings[6][termios.VMIN] = 1
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSANOW, new_settings)

            # Start listener background thread
            self._thread = threading.Thread(target=self._listen_loop, daemon=True, name="EscListenerThread")
            self._thread.start()
        except Exception:
            self._cleanup_terminal()

        return self

    def _listen_loop(self):
        if self._fd is None:
            return

        while not self._stop_event.is_set():
            try:
                # Poll with short 50ms timeout so thread exits promptly on stop_event
                r, _, _ = select.select([self._fd], [], [], 0.05)
                if not r or self._stop_event.is_set():
                    continue

                # Read 1 raw byte
                b = os.read(self._fd, 1)
                if not b:
                    continue

                if b == b"\x1b":
                    # An ESC byte was received. Check if subsequent bytes follow (e.g., arrow key sequences \x1b[A)
                    r_trailing, _, _ = select.select([self._fd], [], [], 0.04)
                    if r_trailing:
                        # Swallow the rest of the multi-byte terminal sequence without interrupting
                        try:
                            os.read(self._fd, 32)
                        except Exception:
                            pass
                        continue

                    # Standalone ESC key pressed!
                    self.interrupted = True
                    self._stop_event.set()
                    if self.on_escape:
                        try:
                            self.on_escape()
                        except Exception:
                            pass
                    break

                elif b == b"\x03":
                    # In case Ctrl+C is passed through unhandled
                    self.interrupted = True
                    self._stop_event.set()
                    if self.on_escape:
                        try:
                            self.on_escape()
                        except Exception:
                            pass
                    break

            except Exception:
                break

    def _cleanup_terminal(self):
        if self._fd is not None and self._old_settings is not None:
            try:
                # Flush any leftover unread characters before restoring so they don't leak into readline
                termios.tcflush(self._fd, termios.TCIFLUSH)
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._cleanup_terminal()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.1)
