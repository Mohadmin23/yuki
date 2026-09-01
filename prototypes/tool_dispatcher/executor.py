"""Guarded direct execution of validated calls through Yuki's real functions."""
from __future__ import annotations

import json
import re
import shlex
import time
from typing import Any

from tools._helpers import ALLOWED_COMMANDS

from .registry import ToolRegistry

CONFIRMATION_TOOLS = {
    "shell",
    "see",
    "image",
    "yuki_write",
    "yuki_delete",
    "yuki_append",
    "remember",
    "recall",
    "ask_claude",
}

_STRICT_SHELL_CHARS = re.compile(r"^[A-Za-z0-9_./:=,@%+\-\s]+$")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def validate_strict_shell(command: str) -> list[str]:
    errors: list[str] = []
    if any(ord(char) < 32 for char in command):
        errors.append("Control characters and newlines are forbidden.")
    if not _STRICT_SHELL_CHARS.fullmatch(command):
        errors.append(
            "Shell operators, substitutions, quotes, escapes, and globs are forbidden."
        )
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        errors.append(f"Invalid shell tokenization: {exc}")
        parts = []
    if not parts:
        errors.append("Shell command must not be empty.")
    elif parts[0] not in ALLOWED_COMMANDS:
        errors.append(
            f"Command {parts[0]!r} is not allowed; expected one of "
            f"{sorted(ALLOWED_COMMANDS)}."
        )
    return errors


def execute_call(
    registry: ToolRegistry,
    call: dict[str, Any],
    available_names: tuple[str, ...],
    *,
    allow_side_effects: bool = False,
    shell_mode: str = "dry_run",
) -> dict[str, Any]:
    """Revalidate then execute one call; the small model never sees the result."""

    validation = registry.validate_call(call, available_names)
    if not validation["passed"]:
        return {
            "status": "rejected",
            "passed": False,
            "executed": False,
            "errors": validation["errors"],
            "raw_output": None,
            "raw_output_repr": None,
            "latency_ms": None,
        }

    name = call["tool"]
    spec = registry.get(name)
    assert spec is not None
    argument = ""
    if spec.param_name is not None:
        argument = call["arguments"].get(spec.param_name, "")

    if name == "shell" and shell_mode == "dry_run":
        return {
            "status": "dry_run",
            "passed": True,
            "executed": False,
            "errors": [],
            "raw_output": {"command": argument, "dry_run": True},
            "raw_output_repr": repr({"command": argument, "dry_run": True}),
            "latency_ms": 0.0,
        }

    if name in CONFIRMATION_TOOLS and not allow_side_effects:
        return {
            "status": "confirmation_required",
            "passed": False,
            "executed": False,
            "errors": [
                f"{name} may change local state, use an external service, or incur cost."
            ],
            "raw_output": None,
            "raw_output_repr": None,
            "latency_ms": None,
        }

    if name == "shell":
        if shell_mode != "strict":
            return {
                "status": "rejected",
                "passed": False,
                "executed": False,
                "errors": [f"Unknown shell mode: {shell_mode}"],
                "raw_output": None,
                "raw_output_repr": None,
                "latency_ms": None,
            }
        shell_errors = validate_strict_shell(argument)
        if shell_errors:
            return {
                "status": "rejected",
                "passed": False,
                "executed": False,
                "errors": shell_errors,
                "raw_output": None,
                "raw_output_repr": None,
                "latency_ms": None,
            }

    started = time.perf_counter()
    try:
        result = spec.fn(argument)
    except Exception as exc:  # noqa: BLE001 - tool failures must stay observable
        return {
            "status": "failed",
            "passed": False,
            "executed": True,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "raw_output": None,
            "raw_output_repr": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    return {
        "status": "passed",
        "passed": True,
        "executed": True,
        "errors": [],
        "raw_output": _json_safe(result),
        "raw_output_repr": repr(result),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }
