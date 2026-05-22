from .remember import tool_remember as tool_fn

META = {
    "cli_name": "/remember",
    "react_name": "remember",
    "help": "Save a fact to long-term memory (persists across chats)",
    "usage": "/remember <fact>",
    "description": (
        'remember("fact") — Save a lasting fact about the user to long-term memory '
        "that persists across ALL future chats. Call this quietly when the user shares "
        'something personal worth keeping: their name, job, location, preferences '
        '("I\'m vegetarian"), ongoing projects, names of pets/family/friends, things '
        "they love or hate, or anything they'd expect you to recall days from now. "
        'Keep each fact to one short sentence phrased as a standalone note ("user\'s '
        'dog is named Max", "user prefers dark mode"). Do NOT call for small talk, '
        "transient questions, one-off jokes, or facts already obvious from the current "
        "conversation. Do not save the same fact twice."
    ),
    "param_name": "fact",
    "param_description": "One short sentence about the user phrased as a standalone note, e.g. \"user's dog is named Max\" or \"user prefers dark mode\".",
    "is_factual": False,
}
