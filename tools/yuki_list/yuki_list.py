"""List files in yuki/."""
from __future__ import annotations

from tools._helpers import YUKI_DIR


def tool_yuki_list(_=""):
    """List all files in the yuki folder."""
    files = sorted(YUKI_DIR.iterdir())
    if not files:
        return "yuki/ is empty"
    lines = [f"  {f.name} ({f.stat().st_size} bytes)" for f in files if f.is_file()]
    return "Files in yuki/:\n" + "\n".join(lines) if lines else "yuki/ is empty"
