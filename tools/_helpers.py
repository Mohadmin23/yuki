"""Shared helpers used by more than one tool.

Kept here (instead of duplicating into each tool subfolder) so that fixes
to argument parsing or path conventions only need to happen once."""
from __future__ import annotations

import re
from pathlib import Path

# yuki/ lives at the repo root, alongside ms_llama.py and tools/.
YUKI_DIR = Path(__file__).parent.parent / "yuki"

# tool_shell only permits these commands. Adding here updates every caller.
ALLOWED_COMMANDS = {
    "ls", "pwd", "whoami", "date", "uptime", "df", "uname",
    "cat", "head", "tail", "wc", "echo", "which", "hostname",
}


def split_filename_content(args: str) -> tuple[str, str] | None:
    """Accept filename|content OR "filename", "content" — returns (name, content) or None."""
    if "|" in args:
        parts = args.split("|", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    m = re.match(r'\s*"?([^",]+?)"?\s*,\s*"?(.*?)"?\s*$', args, re.DOTALL)
    if m:
        return m.group(1), m.group(2)
    return None


def fmt_bytes(n: float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def sanitize_yuki_name(filename: str) -> str:
    """Strip slashes and parent-dir traversal from a yuki filename."""
    return filename.strip().replace("/", "").replace("..", "")
