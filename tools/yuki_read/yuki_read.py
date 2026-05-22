"""Read a file from yuki/."""
from __future__ import annotations

from tools._helpers import YUKI_DIR, sanitize_yuki_name


def tool_yuki_read(filename):
    """Read a file from the yuki folder."""
    filename = sanitize_yuki_name(filename)
    filepath = YUKI_DIR / filename
    if not filepath.exists():
        return f"File not found: yuki/{filename}"
    text = filepath.read_text(errors="replace")[:2000]
    return text if text else "(empty file)"
