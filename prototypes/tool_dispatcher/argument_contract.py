"""Typed Yuki argument transport contract and deterministic contract scoring.

The contract is prototype-only. It does not modify Yuki's live tool metadata,
implementations, or frozen benchmark datasets.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from .argument_scoring import score_arguments

CONTRACT_PATH = Path(__file__).with_name("argument-contract-v1.json")
CONTRACT_VERSION = "yuki-argument-contract-v1"

NO_ARGUMENT = "no_argument"
LITERAL_SOURCE = "literal_source"
SEMANTIC_ARGUMENT = "semantic_argument"

_VALID_MODES = {NO_ARGUMENT, LITERAL_SOURCE, SEMANTIC_ARGUMENT}
_VALID_COMPARISONS = {
    "empty_object",
    "literal_character_exact",
    "normalized_text",
    "calc_ast",
    "enum_exact",
    "referent_phrase",
    "semantic_reference",
}
_REFERENT_PREFIX = re.compile(
    r"^(?:(?:the|a|an|my|your|our|his|her|their|yuki's)\s+)+",
    flags=re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _normalize_referent(value: str) -> str:
    normalized = _normalize_text(value).strip(" \t\r\n.!?")
    return _REFERENT_PREFIX.sub("", normalized)


def _reference_values(
    references: dict[str, Any] | None,
    field: str,
) -> tuple[str, ...]:
    if not references or field not in references:
        return ()
    value = references[field]
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(
        isinstance(item, str) for item in values
    ):
        raise TypeError(
            f"Contract references for {field!r} must be a string or list of strings"
        )
    return tuple(dict.fromkeys(values))


@lru_cache(maxsize=1)
def load_argument_contract() -> dict[str, Any]:
    """Load and validate the versioned machine-readable contract."""

    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if payload.get("version") != CONTRACT_VERSION:
        raise RuntimeError(
            f"Unsupported argument contract: {payload.get('version')!r}"
        )
    tools = payload.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise RuntimeError("Argument contract must define a non-empty tools object")
    for tool, spec in tools.items():
        mode = spec.get("mode")
        comparison = spec.get("comparison")
        argument = spec.get("argument")
        if mode not in _VALID_MODES:
            raise RuntimeError(f"Unknown argument mode for {tool}: {mode!r}")
        if comparison not in _VALID_COMPARISONS:
            raise RuntimeError(
                f"Unknown argument comparison for {tool}: {comparison!r}"
            )
        if mode == NO_ARGUMENT and argument is not None:
            raise RuntimeError(f"No-argument tool {tool} declares {argument!r}")
        if mode != NO_ARGUMENT and not isinstance(argument, str):
            raise RuntimeError(f"Argument tool {tool} has no field name")
        if mode == LITERAL_SOURCE and comparison != "literal_character_exact":
            raise RuntimeError(f"Literal tool {tool} is not character-exact")
        if spec.get("components") and mode != LITERAL_SOURCE:
            raise RuntimeError(f"Composite tool {tool} must be literal-source")
    return payload


def tool_argument_contract(tool: str) -> dict[str, Any]:
    """Return a copy-safe view of one tool's contract entry."""

    try:
        spec = load_argument_contract()["tools"][tool]
    except KeyError as exc:
        raise KeyError(f"No typed argument contract for Yuki tool {tool!r}") from exc
    return dict(spec)


def contract_manifest() -> dict[str, Any]:
    """Return the JSON-compatible contract for reports and data builders."""

    return json.loads(json.dumps(load_argument_contract(), ensure_ascii=False))


def validate_contract_against_registry(registry: Any) -> None:
    """Fail if the typed contract drifts from the live Yuki registry."""

    contract_tools = set(load_argument_contract()["tools"])
    registry_tools = set(registry.names)
    if contract_tools != registry_tools:
        raise RuntimeError(
            "Typed argument contract and live registry differ: "
            f"missing={sorted(registry_tools - contract_tools)}, "
            f"extra={sorted(contract_tools - registry_tools)}"
        )
    for tool in registry.names:
        spec = registry.get(tool)
        contract = tool_argument_contract(tool)
        if contract["argument"] != spec.param_name:
            raise RuntimeError(
                f"Argument field mismatch for {tool}: "
                f"{contract['argument']!r} != {spec.param_name!r}"
            )
        required = spec.param_name in spec.argument_schema.get("required", [])
        if contract["required"] != required:
            raise RuntimeError(
                f"Requiredness mismatch for {tool}: "
                f"{contract['required']!r} != {required!r}"
            )


def _semantic_match(
    tool: str,
    field: str,
    comparison: str,
    expected_values: tuple[str, ...],
    actual: str,
) -> bool:
    if comparison == "enum_exact":
        return actual in expected_values
    if comparison == "referent_phrase":
        normalized_actual = _normalize_referent(actual)
        return any(
            normalized_actual == _normalize_referent(expected)
            for expected in expected_values
        )
    if comparison in {"normalized_text", "calc_ast", "semantic_reference"}:
        return any(
            score_arguments(
                tool,
                {field: expected},
                {field: actual},
            )["execution_equivalent"]
            for expected in expected_values
        )
    raise ValueError(f"Unsupported semantic comparison: {comparison}")


def score_contract_arguments(
    tool: str,
    expected: dict[str, Any],
    actual: Any,
    *,
    literal_references: dict[str, Any] | None = None,
    semantic_alternatives: dict[str, Any] | None = None,
    declared_version: str | None = None,
) -> dict[str, Any]:
    """Score arguments under the typed contract without changing official gold.

    Literal fields are intentionally unscoreable without an independent literal
    reference. The scorer never treats the old semantic gold as a source span.
    """

    if declared_version is not None and declared_version != CONTRACT_VERSION:
        raise ValueError(
            f"Case declares {declared_version!r}, expected {CONTRACT_VERSION!r}"
        )
    spec = tool_argument_contract(tool)
    mode = spec["mode"]
    comparison = spec["comparison"]
    field = spec["argument"]
    base = {
        "contract_version": CONTRACT_VERSION,
        "mode": mode,
        "comparison": comparison,
        "argument": field,
        "official_gold_unchanged": True,
    }

    if mode == NO_ARGUMENT:
        correct = actual == {}
        return {
            **base,
            "scoreable": True,
            "contract_correct": correct,
            "reason": (
                "Argument object is empty as required."
                if correct
                else "No-argument tool must emit an empty argument object."
            ),
            "accepted_values": [],
        }

    if not expected:
        if not spec.get("required"):
            omitted_default = spec.get("omitted_equivalent_to")
            correct = actual == {} or (
                omitted_default is not None
                and actual == {field: omitted_default}
            )
            return {
                **base,
                "scoreable": True,
                "contract_correct": correct,
                "reason": (
                    "Optional argument is correctly omitted or uses its declared default."
                    if correct
                    else "Optional argument changes the requested operation."
                ),
                "accepted_values": (
                    [] if omitted_default is None else [omitted_default]
                ),
            }
        return {
            **base,
            "scoreable": False,
            "contract_correct": None,
            "reason": "Required semantic reference is absent.",
            "accepted_values": [],
        }

    if mode == LITERAL_SOURCE:
        accepted = _reference_values(literal_references, field)
        if not accepted:
            return {
                **base,
                "scoreable": False,
                "contract_correct": None,
                "reason": (
                    "Literal-source reference is absent; official semantic gold "
                    "is deliberately not substituted."
                ),
                "accepted_values": [],
            }
        actual_value = actual.get(field) if isinstance(actual, dict) else None
        correct = (
            isinstance(actual, dict)
            and set(actual) == {field}
            and isinstance(actual_value, str)
            and actual_value in accepted
        )
        return {
            **base,
            "scoreable": True,
            "contract_correct": correct,
            "reason": (
                "Argument matches an independently annotated literal source value."
                if correct
                else "Argument is not character-exact under the literal-source contract."
            ),
            "accepted_values": list(accepted),
        }

    expected_value = expected.get(field)
    if not isinstance(expected_value, str):
        return {
            **base,
            "scoreable": False,
            "contract_correct": None,
            "reason": "Canonical semantic reference is absent or not text.",
            "accepted_values": [],
        }
    alternatives = _reference_values(semantic_alternatives, field)
    accepted = tuple(dict.fromkeys((expected_value, *alternatives)))
    actual_value = actual.get(field) if isinstance(actual, dict) else None
    correct = (
        isinstance(actual, dict)
        and set(actual) == {field}
        and isinstance(actual_value, str)
        and _semantic_match(tool, field, comparison, accepted, actual_value)
    )
    return {
        **base,
        "scoreable": True,
        "contract_correct": correct,
        "reason": (
            "Argument is equivalent under the declared semantic comparator."
            if correct
            else "Argument differs under the declared semantic comparator."
        ),
        "accepted_values": list(accepted),
    }
