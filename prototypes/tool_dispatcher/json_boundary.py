"""Detect one complete top-level JSON value without accepting surrounding prose."""
from __future__ import annotations

import json


def complete_json_value_end(text: str, root_type: str = "object") -> int | None:
    """Return the end index for exactly one complete object or array.

    Leading and trailing whitespace are allowed. Any prefix/suffix prose keeps
    generation running and will later be reported as malformed output.
    """

    expected_open = {"object": "{", "array": "["}.get(root_type)
    if expected_open is None:
        raise ValueError(f"Unknown JSON root type: {root_type}")
    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != expected_open:
        return None

    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "}]":
            if not stack or char != stack.pop():
                return None
            if not stack:
                if text[index + 1 :].strip():
                    return None
                candidate = text[start : index + 1]
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                expected_type = dict if root_type == "object" else list
                return index + 1 if isinstance(value, expected_type) else None
    return None


def complete_json_object_end(text: str) -> int | None:
    return complete_json_value_end(text, "object")


def is_complete_json_value(text: str, root_type: str = "object") -> bool:
    return complete_json_value_end(text, root_type) is not None


def is_complete_json_object(text: str) -> bool:
    return is_complete_json_value(text, "object")
