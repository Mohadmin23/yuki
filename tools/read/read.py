"""Read a local file."""
from __future__ import annotations

from pathlib import Path


def tool_read(filepath):
    """Read a local file (max 2000 chars)."""
    try:
        p = Path(filepath.strip()).expanduser()
        if not p.exists():
            return f"File not found: {filepath}"
        if not p.is_file():
            return f"Not a file: {filepath}"
        text = p.read_text(errors="replace")[:2000]
        return text if text else "(empty file)"
    except Exception as e:
        return f"Read error: {e}"
