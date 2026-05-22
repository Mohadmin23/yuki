"""Delete a file from yuki/."""
from __future__ import annotations

from tools._helpers import YUKI_DIR, sanitize_yuki_name


def tool_yuki_delete(filename):
    """Delete a file from the yuki folder."""
    filename = sanitize_yuki_name(filename)
    if not filename:
        return "Error: empty filename"
    filepath = YUKI_DIR / filename
    if not filepath.exists():
        return f"File not found: yuki/{filename}"
    if not filepath.is_file():
        return f"Not a file: yuki/{filename}"
    filepath.unlink()
    return f"Deleted yuki/{filename}"
