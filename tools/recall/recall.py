"""Recall summaries of past sessions semantically similar to a query."""
from __future__ import annotations

import time


def tool_recall(query):
    """Look up summaries of similar past sessions.

    The current session is excluded — it's already in the bot's live history.
    """
    import ms_llama  # lazy
    import episodic

    query = (query or "").strip()
    if not query:
        return "Error: nothing to recall"

    bot = ms_llama._ACTIVE_BOT
    if bot is None:
        return "Recall error: no active session."

    try:
        sessions = episodic.recall_sessions(
            query,
            exclude_session=getattr(bot, "session_uuid", None),
        )
    except Exception as e:  # noqa: BLE001
        return f"Recall error: {e}"

    if not sessions:
        return "Nothing matching in past sessions."

    now = time.time()
    lines = []
    for s in sessions:
        when = episodic.fuzzy_when(now - s["ts"])
        lines.append(f"• {when}: {s['summary']}")
    return "\n".join(lines)
