import re
from .time import tool_time as tool_fn

META = {
    "cli_name": "/time",
    "react_name": "time",
    "help": "Get current date and time",
    "usage": "/time",
    "description": 'time() — Only when the user asks what time/day/date it is. Do NOT call this unprompted. Takes no arguments.',
    "param_name": None,
    "is_factual": True,
    "auto_detect": [
        {
            "pattern": re.compile(
                r"\b(?:what\s+time|what\s+day|what\s+date|what'?s\s+the\s+(?:time|date|day)"
                r"|what'?s\s+today|today'?s\s+date|current\s+(?:time|date)|what\s+year)\b",
                re.IGNORECASE,
            ),
            "arg": "",
        },
    ],
}
