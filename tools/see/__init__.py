import re
from .see import tool_see as tool_fn

META = {
    "cli_name": "/see",
    "react_name": "see",
    "help": "Look through the eye (webcam) and say what it sees",
    "usage": "/see [target]",
    "description": 'see("target") — Look at the user\'s world through your real eye (the webcam, opens in a window). Empty arg = describe the scene; name an object ("apple") to find and track it. Use when the user asks you to look at something.',
    "param_name": "target",
    "param_description": "Optional object to find and track, e.g. 'apple' or 'phone'. Omit to just describe the scene.",
    "is_factual": True,
    "auto_detect": [
        {
            "pattern": re.compile(
                r"\b(?:"
                r"open\s+your\s+eyes?"
                r"|use\s+your\s+eyes?"
                r"|have\s+a\s+look"
                r"|take\s+a\s+look(?:\s+at\s+(?:this|me))?"
                r"|(?:see|look\s+at)\s+(?:this|me|that)"
                r"|what\s+do\s+you\s+see"
                r"|can\s+you\s+see(?:\s+(?:me|this|that))?"
                r")\b",
                re.IGNORECASE,
            ),
            "arg": "",
        },
    ],
}
