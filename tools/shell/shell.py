"""Run a whitelisted shell command."""
from __future__ import annotations

import subprocess

from tools._helpers import ALLOWED_COMMANDS


def tool_shell(command):
    """Run a safe shell command."""
    cmd_name = command.strip().split()[0] if command.strip() else ""
    if cmd_name not in ALLOWED_COMMANDS:
        return (
            f"Command '{cmd_name}' not allowed. "
            f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
        )
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        output = (result.stdout + result.stderr).strip()
        return output[:2000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out (10s limit)."
    except Exception as e:
        return f"Shell error: {e}"
