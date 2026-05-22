"""Append to a file in yuki/."""
from __future__ import annotations

from tools._helpers import YUKI_DIR, sanitize_yuki_name, split_filename_content


def tool_yuki_append(args):
    """Append to a file in the yuki folder. Format: filename|content (or "name","content")."""
    split = split_filename_content(args)
    if not split:
        return "Error: use format filename|content"
    filename, content = split
    filename = sanitize_yuki_name(filename)
    if not filename:
        return "Error: empty filename"
    filepath = YUKI_DIR / filename
    if not filepath.exists():
        return f"File not found: yuki/{filename} (use yuki_write to create it first)"
    with filepath.open("a") as f:
        f.write(content)
    return f"Appended {len(content)} chars to yuki/{filename}"
