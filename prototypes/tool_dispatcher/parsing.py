"""Strict model-native tool-call parsing with no execution dependencies."""
from __future__ import annotations

import json
from typing import Any

from .dialects import (
    ARCH_XML_OBJECT,
    HAMMER_FENCED_ARRAY,
    OPENAI_NATIVE_TOOL_CALLS,
    TOOL_CALL_ARRAY,
    XLAM_V1_ENVELOPE,
)


def parse_model_output(
    raw_text: str,
    output_mode: str,
    native_dialect: str = XLAM_V1_ENVELOPE,
) -> dict[str, Any]:
    """Strictly parse one JSON-only model output and normalize it."""

    raw = raw_text.strip()
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "malformed": True,
            "rejected": False,
            "decoded": None,
            "canonical_call": None,
            "errors": [f"Invalid JSON-only output: {exc.msg} at position {exc.pos}."],
        }
    if output_mode == "native":
        if native_dialect == XLAM_V1_ENVELOPE:
            if not isinstance(decoded, dict) or set(decoded) != {"tool_calls"}:
                return {
                    "passed": False,
                    "malformed": True,
                    "rejected": False,
                    "decoded": decoded,
                    "canonical_call": None,
                    "errors": ["Native output must contain only a tool_calls array."],
                }
            calls = decoded["tool_calls"]
        elif native_dialect in {
            TOOL_CALL_ARRAY,
            OPENAI_NATIVE_TOOL_CALLS,
            HAMMER_FENCED_ARRAY,
        }:
            if not isinstance(decoded, list):
                return {
                    "passed": False,
                    "malformed": True,
                    "rejected": False,
                    "decoded": decoded,
                    "canonical_call": None,
                    "errors": ["Native output must be a JSON tool-call array."],
                }
            calls = decoded
        elif native_dialect == ARCH_XML_OBJECT:
            if decoded == {}:
                return {
                    "passed": True,
                    "malformed": False,
                    "rejected": True,
                    "decoded": decoded,
                    "canonical_call": None,
                    "errors": [],
                }
            if not isinstance(decoded, dict):
                return {
                    "passed": False,
                    "malformed": True,
                    "rejected": False,
                    "decoded": decoded,
                    "canonical_call": None,
                    "errors": [
                        "XML-wrapped native output must be one JSON tool-call object."
                    ],
                }
            calls = [decoded]
        else:
            raise ValueError(f"Unknown native output dialect: {native_dialect}")
        if not isinstance(calls, list):
            return {
                "passed": False,
                "malformed": True,
                "rejected": False,
                "decoded": decoded,
                "canonical_call": None,
                "errors": ["Native tool calls must be a JSON array."],
            }
        if not calls:
            return {
                "passed": True,
                "malformed": False,
                "rejected": True,
                "decoded": decoded,
                "canonical_call": None,
                "errors": [],
            }
        if len(calls) != 1 or not isinstance(calls[0], dict):
            return {
                "passed": False,
                "malformed": True,
                "rejected": False,
                "decoded": decoded,
                "canonical_call": None,
                "errors": ["Native output must contain exactly one tool call."],
            }
        native_call = calls[0]
        if "name" not in native_call or not set(native_call).issubset(
            {"name", "arguments"}
        ):
            return {
                "passed": False,
                "malformed": True,
                "rejected": False,
                "decoded": decoded,
                "canonical_call": None,
                "errors": ["Tool call must contain a name and optional arguments."],
            }
        call = {
            "tool": native_call.get("name"),
            "arguments": native_call.get("arguments", {}),
        }
    elif output_mode == "canonical":
        if not isinstance(decoded, dict) or set(decoded) != {"tool", "arguments"}:
            return {
                "passed": False,
                "malformed": True,
                "rejected": False,
                "decoded": decoded,
                "canonical_call": None,
                "errors": ["Canonical output must contain exactly tool and arguments."],
            }
        if decoded.get("tool") == "no_tool" and decoded.get("arguments") == {}:
            return {
                "passed": True,
                "malformed": False,
                "rejected": True,
                "decoded": decoded,
                "canonical_call": None,
                "errors": [],
            }
        call = decoded
    else:
        raise ValueError(f"Unknown output mode: {output_mode}")

    return {
        "passed": True,
        "malformed": False,
        "rejected": False,
        "decoded": decoded,
        "canonical_call": call,
        "errors": [],
    }
