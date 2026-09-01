"""Deterministic per-tool argument scoring for routing benchmarks."""
from __future__ import annotations

import ast
import unicodedata
from typing import Any

EXACT = "exact"
NORMALIZED_TEXT = "normalized_text"
CALC_AST = "calc_ast"
NO_ARGUMENTS = "no_arguments"

TOOL_ARGUMENT_RULES: dict[str, str] = {
    "time": NO_ARGUMENTS,
    "weather": NORMALIZED_TEXT,
    "fetch": EXACT,
    "search": NORMALIZED_TEXT,
    "calc": CALC_AST,
    "hardware": NORMALIZED_TEXT,
    "see": NORMALIZED_TEXT,
    "image": NORMALIZED_TEXT,
    "read": EXACT,
    "shell": EXACT,
    "yuki_write": EXACT,
    "yuki_read": EXACT,
    "yuki_list": NO_ARGUMENTS,
    "yuki_delete": EXACT,
    "yuki_append": EXACT,
    "remember": NORMALIZED_TEXT,
    "recall": NORMALIZED_TEXT,
    "ask_claude": NORMALIZED_TEXT,
}


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def _calc_structure(value: str) -> str | None:
    allowed = set("0123456789+-*/.()% ")
    if not value or not all(character in allowed for character in value):
        return None
    try:
        tree = ast.parse(value, mode="eval")
    except (SyntaxError, ValueError, TypeError):
        return None
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _equivalent_value(rule: str, expected: Any, actual: Any) -> bool:
    if rule in {EXACT, NO_ARGUMENTS}:
        return expected == actual
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    if rule == NORMALIZED_TEXT:
        return _normalized_text(expected) == _normalized_text(actual)
    if rule == CALC_AST:
        expected_tree = _calc_structure(expected)
        actual_tree = _calc_structure(actual)
        return expected_tree is not None and expected_tree == actual_tree
    raise ValueError(f"Unknown argument scoring rule: {rule}")


def score_arguments(
    tool: str,
    expected: dict[str, Any],
    actual: Any,
) -> dict[str, Any]:
    """Return strict and execution-equivalent scores with auditable details."""

    rule = TOOL_ARGUMENT_RULES.get(tool, EXACT)
    strict = actual == expected
    if not isinstance(actual, dict):
        return {
            "rule": rule,
            "strict": strict,
            "execution_equivalent": False,
            "reason": "Actual arguments are not a JSON object.",
            "fields": {},
        }
    if set(expected) != set(actual):
        return {
            "rule": rule,
            "strict": strict,
            "execution_equivalent": False,
            "reason": "Argument keys differ.",
            "fields": {},
        }

    fields = {
        key: {
            "expected": expected[key],
            "actual": actual[key],
            "equivalent": _equivalent_value(rule, expected[key], actual[key]),
        }
        for key in expected
    }
    equivalent = all(field["equivalent"] for field in fields.values())
    return {
        "rule": rule,
        "strict": strict,
        "execution_equivalent": equivalent,
        "reason": (
            "Arguments are execution-equivalent under the tool policy."
            if equivalent
            else "At least one argument differs under the tool policy."
        ),
        "fields": fields,
    }
