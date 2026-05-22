"""Write a file in yuki/."""
from __future__ import annotations

from tools._helpers import YUKI_DIR, sanitize_yuki_name, split_filename_content


def tool_yuki_write(args):
    """Write a file in the yuki folder. Format: filename|content (or "name","content")."""
    split = split_filename_content(args)
    if not split:
        return "Error: use format filename|content"
    filename, content = split
    filename = sanitize_yuki_name(filename)
    if not filename:
        return "Error: empty filename"
    filepath = YUKI_DIR / filename
    filepath.write_text(content)
    return f"Wrote {len(content)} chars to yuki/{filename}"
