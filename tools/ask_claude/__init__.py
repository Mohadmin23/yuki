import re
from .ask_claude import tool_ask_claude as tool_fn

META = {
    "cli_name": "/ask-claude",
    "react_name": "ask_claude",
    "help": "Send a question to the other Claude (running in a neighboring tmux pane)",
    "usage": "/ask-claude <question>",
    "description": (
        'ask_claude("question") — Only when the user explicitly asks you to '
        'ask, tell, relay to, or check with Claude (the other AI running in '
        'another tmux pane). Returns what Claude replied so you can paraphrase '
        'it back to the user. IMPORTANT: if the tool result starts with '
        '"ASK_CLAUDE_ERROR:", do NOT pretend Claude answered — read the error '
        'out loud to the user verbatim so they can fix the setup.'
    ),
    "param_name": "question",
    "param_description": "The question or message to send to the other Claude.",
    "is_factual": False,
    "auto_detect": [
        {
            "pattern": re.compile(
                r"\b(?:ask|tell|relay\s+to|send\s+to|check\s+with)\s+claude\b[\s,:-]*(.+?)[\?\.\!]?\s*$",
                re.IGNORECASE,
            ),
            "group": 1,
        },
    ],
}
