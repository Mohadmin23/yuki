"""Current date and time."""
from __future__ import annotations

import datetime


def tool_time(_=""):
    """Get current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y — %I:%M %p")
