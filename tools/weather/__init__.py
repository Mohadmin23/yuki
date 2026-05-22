import re
from .weather import tool_weather as tool_fn

META = {
    "cli_name": "/weather",
    "react_name": "weather",
    "help": "Get weather for a city",
    "usage": "/weather <city>",
    "description": 'weather("city") — Only when the user asks about the weather in a specific place.',
    "param_name": "city",
    "param_description": "City name, e.g. 'Tokyo' or 'New York'.",
    "is_factual": True,
    "auto_detect": [
        {
            "pattern": re.compile(
                r"\bweather\b.*?\bin\s+(.+?)[\?\.\!]?\s*$", re.IGNORECASE
            ),
            "group": 1,
        },
        {
            "pattern": re.compile(
                r"\b(?:temperature|how\s+(?:cold|hot|warm))\b.*?\bin\s+(.+?)[\?\.\!]?\s*$",
                re.IGNORECASE,
            ),
            "group": 1,
        },
    ],
}
