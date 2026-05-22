import re
from .calc import tool_calc as tool_fn

META = {
    "cli_name": "/calc",
    "react_name": "calc",
    "help": "Calculate a math expression",
    "usage": "/calc <expression>",
    "description": 'calc("expression") — Only when there\'s an actual math expression to evaluate.',
    "param_name": "expression",
    "param_description": "Arithmetic expression to evaluate, e.g. '2+2' or '(15*4)/3'. Numbers and +-*/.()% only.",
    "is_factual": True,
    "auto_detect": [
        {
            "pattern": re.compile(
                r"\b(?:calculate|compute)\s+(.+?)[\?\.\!]?\s*$", re.IGNORECASE
            ),
            "group": 1,
        },
        {
            "pattern": re.compile(
                r"\b(?:what\s+is|what'?s|how\s+much\s+is)\s+([\d\s\+\-\*/\.\(\)%]+)[\?\.\!]?\s*$",
                re.IGNORECASE,
            ),
            "group": 1,
        },
    ],
}
