from .fetch import tool_fetch as tool_fn

META = {
    "cli_name": "/fetch",
    "react_name": "fetch",
    "help": "Download a webpage and return its readable text",
    "usage": "/fetch <url>",
    "description": (
        'fetch("url") — Download the page at a URL and return its readable text. '
        "Call this when the user asks about, summarizes, or references a specific URL "
        '(http:// or https://). Strictly better than `search` for "what does this page say" '
        "questions — search returns external snippets, fetch returns the actual page content. "
        "Do NOT use fetch for vague topic queries with no URL — use search for those. "
        "Max ~8000 chars returned; truncated cleanly. Returns 'Fetch error:' on network/DNS "
        "issues, in which case admit honestly that you couldn't read the page rather than "
        "guessing from training data."
    ),
    "param_name": "url",
    "param_description": "Full URL starting with http:// or https:// to download and extract text from.",
    "is_factual": True,
}
