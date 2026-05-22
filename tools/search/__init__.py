import re
from .search import tool_search as tool_fn

META = {
    "cli_name": "/search",
    "react_name": "search",
    "help": "Search the web",
    "usage": "/search <query>",
    "description": 'search("query") — Only when the user asks you to look something up, search, google, or find info you don\'t already know.',
    "param_name": "query",
    "param_description": "Search query — what to look up on the web.",
    "is_factual": True,
    "auto_detect": [
        {
            "pattern": re.compile(
                r"\b(?:search\s+(?:for|up)|look\s+up|google)\s+(.+?)[\?\.\!]?\s*$",
                re.IGNORECASE,
            ),
            "group": 1,
        },
    ],
}
