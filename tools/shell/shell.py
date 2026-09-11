"""Run a whitelisted shell command."""
from __future__ import annotations

import shlex
import subprocess

from tools._helpers import ToolResult, validate_strict_shell


def tool_shell(command):
    """Run a safe shell command."""
    errors = validate_strict_shell(command)
    if errors:
        return ToolResult("Error: command not allowed. " + " ".join(errors), succeeded=False)
    try:
        result = subprocess.run(
            shlex.split(command), shell=False, capture_output=True, text=True,
            timeout=10, check=False,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode:
            return ToolResult(
                f"Error: command exited with status {result.returncode}.\n{output[:2000]}",
                succeeded=False,
            )
        return ToolResult(output[:2000] if output else "(no output)", succeeded=True)
    except subprocess.TimeoutExpired:
        return ToolResult("Error: command timed out (10s limit).", succeeded=False)
    except OSError as e:
        return ToolResult(f"Shell error: {e}", succeeded=False)
