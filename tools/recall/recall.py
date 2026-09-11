"""Retrieve related past sessions and Yuki-owned notes."""

from __future__ import annotations


def tool_recall(query):
    """Look up related long-term conversational memory and internal notes.

    The current session is excluded — it's already in the bot's live history.
    """
    import ms_llama  # lazy
    from retrieval import render_retrieval_context
    from retrieval.local import retrieve_local_memory

    query = (query or "").strip()
    if not query:
        return "Error: nothing to recall"

    bot = ms_llama._ACTIVE_BOT
    if bot is None:
        return "Recall error: no active session."

    hits, warnings = retrieve_local_memory(
        query,
        exclude_session=getattr(bot, "session_uuid", None),
    )
    if not hits:
        if warnings:
            return "Recall error: " + "; ".join(warnings)
        return "Nothing matching in past sessions or Yuki notes."
    return render_retrieval_context(
        query,
        hits,
        label="YUKI MEMORY",
        warnings=warnings,
    )
