"""Save a fact to long-term hash-fact memory (single-user, flat store)."""
from __future__ import annotations


def tool_remember(fact):
    """Save a fact to memory.json. Returns a status string."""
    import ms_llama  # lazy; avoids circular import

    fact = (fact or "").strip()
    if not fact:
        return "Error: nothing to remember"
    fact = fact[: ms_llama.MEMORY_MAX_CHARS]

    category, value = ms_llama._classify_fact(fact)
    try:
        facts = ms_llama._load_facts()
        if not ms_llama._add_fact(facts, category, value):
            return f"Already remembered: {fact}"
        ms_llama._save_facts(facts)
        return f"Remembered: {fact}"
    except Exception as e:
        return f"Memory save error: {e}"
