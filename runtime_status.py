"""Answer explicit action-status questions from trusted session events."""

from __future__ import annotations

import re
from collections.abc import Sequence

# Full matches deliberately exclude requests to execute/retry an action, quoted
# examples, and compound questions requiring information beyond the ledger.
_LAST_ACTION_QUESTION = re.compile(
    r"what (?:was|is) (?:the|your) (?:last|latest|most recent) "
    r"(?:(?:successfully )?completed |successful )?(?:tool )?(?:action|operation)"
    r"(?: (?:you|that you) (?:performed|completed|successfully completed))?"
    r"[?.!]*",
    re.IGNORECASE,
)
_JUST_PERFORMED_QUESTION = re.compile(
    r"what (?:tool )?action did you just (?:perform|complete)"
    r"(?:,? and was it (?:successfully completed|successful))?[?.!]*",
    re.IGNORECASE,
)


def action_status_reply(question: str, events: Sequence[dict]) -> str | None:
    """Return recorded truth for a bounded status query, or defer to normal chat.

    Events are in runtime order, oldest first. A canceled reply need not appear
    in chat history for its completed tool outcome to remain authoritative.
    """
    normalized = " ".join(question.split())
    if not (
        _LAST_ACTION_QUESTION.fullmatch(normalized)
        or _JUST_PERFORMED_QUESTION.fullmatch(normalized)
    ):
        return None
    success_only = bool(_LAST_ACTION_QUESTION.fullmatch(normalized)) and bool(
        re.search(r"\bsuccessful(?:ly)?\b", normalized, re.IGNORECASE)
    )
    for event in reversed(events):
        if event.get("kind") != "verified_tool_outcome" or not event.get("tool"):
            continue
        if success_only and event.get("succeeded") is not True:
            continue
        status = "It succeeded." if event.get("succeeded") is True else (
            "It did not succeed." if event.get("succeeded") is False
            else "Its success wasn't recorded."
        )
        reply = f"My latest recorded {'successful ' if success_only else ''}tool action was {event['tool']}. {status}"
        result = str(event.get("result") or "").strip()
        if result:
            # Quoted evidence, not newly generated narration or instructions.
            excerpt = result[:1200] + ("…" if len(result) > 1200 else "")
            reply += "\n\nRecorded result:\n" + "\n".join(
                f"> {line}" for line in excerpt.splitlines()
            )
        return reply
    qualifier = "successful " if success_only else ""
    return f"I don't have a {qualifier}tool outcome in the available session records."
