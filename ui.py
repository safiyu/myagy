"""Terminal styling, curated 256-color palette, and box decorations."""

import os
import sys

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.live import Live
    from rich import box
    from rich.text import Text
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    Markdown = None
    Table = None
    Live = None
    box = None
    Text = None


class UI:
    """Terminal styling, curated 256-color palette, and box decorations."""
    _use_color = sys.stdout.isatty() and not os.getenv("NO_COLOR")

    RST = "\033[0m" if _use_color else ""
    BOLD = "\033[1m" if _use_color else ""
    DIM = "\033[2m" if _use_color else ""
    ITALIC = "\033[3m" if _use_color else ""

    # Curated harmonious color palette
    ROCM = "\033[38;5;39m" if _use_color else ""        # Vivid Cyan / Sky Blue (AMD)
    ROCM_BOLD = "\033[1;38;5;39m" if _use_color else ""
    CUDA = "\033[38;5;48m" if _use_color else ""        # NVIDIA Emerald Green
    CUDA_BOLD = "\033[1;38;5;48m" if _use_color else ""
    CLOUD = "\033[38;5;75m" if _use_color else ""       # Google Gemini Blue
    CLOUD_BOLD = "\033[1;38;5;75m" if _use_color else ""
    LAYA = "\033[38;5;141m" if _use_color else ""       # Soft Violet / Purple (ModernBERT)
    LAYA_BOLD = "\033[1;38;5;141m" if _use_color else ""
    AMBER = "\033[38;5;214m" if _use_color else ""      # Amber / Tools / Risk
    AMBER_BOLD = "\033[1;38;5;214m" if _use_color else ""
    GREEN = "\033[38;5;40m" if _use_color else ""       # Success Green
    GREEN_BOLD = "\033[1;38;5;40m" if _use_color else ""
    RED = "\033[38;5;203m" if _use_color else ""        # Error Red
    RED_BOLD = "\033[1;38;5;203m" if _use_color else ""
    CYAN = "\033[38;5;51m" if _use_color else ""        # Bright Cyan
    CYAN_BOLD = "\033[1;38;5;51m" if _use_color else ""
    WHITE = "\033[1;97m" if _use_color else ""          # Bright White
    GRAY = "\033[38;5;244m" if _use_color else ""       # Muted Gray
    DARK_GRAY = "\033[38;5;239m" if _use_color else ""  # Dark Border Gray

    @classmethod
    def target_color(cls, target: str) -> str:
        t = str(target).lower().strip()
        if t in ("9000", "rocm", "amd", "gpu0"):
            return cls.ROCM
        if t in ("9001", "cuda", "nvidia", "gpu1"):
            return cls.CUDA
        if t in ("cloud", "gemini", "oauth"):
            return cls.CLOUD
        if t in ("auto", "laya", "smart"):
            return cls.LAYA
        return cls.WHITE

    @classmethod
    def badge(cls, text: str, color: str = "") -> str:
        color = color or cls.GRAY
        return f"{color}{cls.BOLD}[{text}]{cls.RST}"

    @classmethod
    def ok(cls, text: str) -> str:
        return f"{cls.GREEN_BOLD}[✓]{cls.RST} {cls.WHITE}{text}{cls.RST}"

    @classmethod
    def warn(cls, text: str) -> str:
        return f"{cls.AMBER_BOLD}[⚠]{cls.RST} {cls.AMBER}{text}{cls.RST}"

    @classmethod
    def err(cls, text: str) -> str:
        return f"{cls.RED_BOLD}[✗]{cls.RST} {cls.RED}{text}{cls.RST}"
