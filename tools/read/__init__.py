from .read import tool_read as tool_fn

META = {
    "cli_name": "/read",
    "react_name": "read",
    "help": "Read a local file",
    "usage": "/read <filepath>",
    "description": 'read("filepath") — Only when the user asks you to read a specific local file.',
    "param_name": "filepath",
    "param_description": "Local filesystem path to read. Supports ~ expansion. Returns up to 2000 chars.",
    "is_factual": True,
}
