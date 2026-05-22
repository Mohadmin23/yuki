from .shell import tool_shell as tool_fn

META = {
    "cli_name": "/shell",
    "react_name": "shell",
    "help": "Run a safe shell command",
    "usage": "/shell <command>",
    "description": 'shell("command") — Only when the user asks to run a shell command (ls, pwd, etc.).',
    "param_name": "command",
    "param_description": "Shell command to run. Only a whitelist is allowed: ls, pwd, whoami, date, uptime, df, uname, cat, head, tail, wc, echo, which, hostname.",
    "is_factual": True,
}
