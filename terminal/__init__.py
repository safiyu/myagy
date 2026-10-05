"""Terminal UI styling, interactive REPL loop, and keyboard interrupt listeners."""

from .ui import UI, RICH_AVAILABLE, console
from .esc_listener import EscListener


def __getattr__(name):
    # cli pulls in the session (and through it agents/core), so load it only on demand
    if name in ("read_input_prompt", "interactive_loop", "main"):
        from . import cli
        return getattr(cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UI",
    "RICH_AVAILABLE",
    "console",
    "EscListener",
    "read_input_prompt",
    "interactive_loop",
    "main",
]
