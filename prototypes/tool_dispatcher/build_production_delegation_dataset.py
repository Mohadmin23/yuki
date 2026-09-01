"""Derive a versioned production-delegation evaluation from frozen Phase 2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .delegation_contract import expected_domain_for_tool
from .registry import ToolRegistry

SOURCE_SHA256 = "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466"
DATASET_VERSION = "production-delegation-source-v1"

_VERBATIM_TOOLS = {
    "fetch",
    "search",
    "image",
    "read",
    "shell",
    "yuki_write",
    "yuki_read",
    "yuki_delete",
    "yuki_append",
    "remember",
    "recall",
    "ask_claude",
}

_SOURCE_OVERRIDES: dict[str, list[str]] = {
    "p2-remember-12": ["my cat's name is Nori"],
    "p2-remember-13": ["I prefer tabs over spaces"],
    "p2-remember-14": ["my main laptop is an M1 Air"],
    "p2-remember-15": ["I avoid cilantro"],
    "p2-remember-16": ["my desk faces east"],
    "p2-remember-17": ["my project codename is Aurora-7"],
    "p2-recall-13": ["Metal ran out of RAM"],
    "p2-recall-20": ["notes.md"],
}

_ARGUMENT_ALTERNATIVES: dict[str, dict[str, list[str]]] = {
    "p2-remember-12": {"fact": ["my cat's name is Nori"]},
    "p2-remember-13": {"fact": ["I prefer tabs over spaces"]},
    "p2-remember-14": {"fact": ["my main laptop is an M1 Air"]},
    "p2-remember-15": {"fact": ["I avoid cilantro"]},
    "p2-remember-16": {"fact": ["my desk faces east"]},
    "p2-remember-17": {"fact": ["my project codename is Aurora-7"]},
    "p2-recall-13": {"topic": ["Metal ran out of RAM"]},
    "p2-recall-20": {"topic": ["notes.md"]},
}


def _source_slice(request: str, value: str) -> str | None:
    if value in request:
        return value
    start = request.casefold().find(value.casefold())
    if start >= 0:
        return request[start : start + len(value)]
    return None


def _required_verbatim(case: dict[str, Any]) -> list[str]:
    if case["case_id"] in _SOURCE_OVERRIDES:
        return list(_SOURCE_OVERRIDES[case["case_id"]])
    if case["expected_tool"] not in _VERBATIM_TOOLS:
        return []

    values: list[str] = []
    for value in case["expected_arguments"].values():
        if not isinstance(value, str) or not value:
            continue
        candidates = value.split("|", 1) if case["expected_tool"] in {
            "yuki_write",
            "yuki_append",
        } else [value]
        for candidate in candidates:
            source = _source_slice(case["request"], candidate)
            if source is not None and source not in values:
                values.append(source)
    return values


def _smoke_ids(cases: list[dict[str, Any]], registry: ToolRegistry) -> list[str]:
    categories = ("normal", "paraphrase", "confusion", "preservation", "adversarial")
    selected = []
    for index, tool in enumerate(registry.names):
        tool_cases = [case for case in cases if case["expected_tool"] == tool]
        preferred = categories[index % len(categories)]
        match = next(
            (case for case in tool_cases if case["tags"][1] == preferred),
            tool_cases[0],
        )
        selected.append(match["case_id"])
    return selected


def _fallback_ids(cases: list[dict[str, Any]], registry: ToolRegistry) -> list[str]:
    """Choose ten cases per tool with deterministic category-first coverage."""

    quotas = {
        "normal": 2,
        "paraphrase": 2,
        "confusion": 2,
        "preservation": 2,
        "adversarial": 1,
    }
    selected: list[str] = []
    for tool in registry.names:
        tool_cases = [case for case in cases if case["expected_tool"] == tool]
        tool_ids: list[str] = []
        for category, quota in quotas.items():
            category_ids = [
                case["case_id"]
                for case in tool_cases
                if case["tags"][1] == category
            ][:quota]
            tool_ids.extend(category_ids)
        for case in tool_cases:
            if len(tool_ids) >= 10:
                break
            if case["case_id"] not in tool_ids:
                tool_ids.append(case["case_id"])
        selected.extend(tool_ids[:10])
    if len(selected) != 180 or len(set(selected)) != 180:
        raise RuntimeError("Fallback selection must contain 180 unique cases")
    return selected


def build_dataset(source_path: str | Path) -> dict[str, Any]:
    source = Path(source_path)
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise RuntimeError(
            f"Frozen Phase 2 checksum mismatch: {actual_hash} != {SOURCE_SHA256}"
        )
    cases = json.loads(source.read_text(encoding="utf-8"))
    registry = ToolRegistry()
    if len(cases) != 450:
        raise RuntimeError(f"Expected 450 Phase 2 cases, found {len(cases)}")
    if {case["expected_tool"] for case in cases} != set(registry.names):
        raise RuntimeError("Derived dataset tool set differs from the live registry")

    derived = []
    for case in cases:
        required_verbatim = _required_verbatim(case)
        if any(value not in case["request"] for value in required_verbatim):
            raise RuntimeError(
                f"Verbatim annotation is not source-exact for {case['case_id']}"
            )
        derived.append(
            {
                **case,
                "source_dataset": "phase2-450",
                "source_dataset_sha256": actual_hash,
                "expected_domain": expected_domain_for_tool(
                    case["expected_tool"], registry
                ),
                "required_verbatim": required_verbatim,
                "argument_alternatives": _ARGUMENT_ALTERNATIVES.get(
                    case["case_id"], {}
                ),
                "ambiguous_or_adversarial": case["tags"][1] == "adversarial",
                "tool_should_be_used": True,
            }
        )

    return {
        "dataset_version": DATASET_VERSION,
        "source_dataset": source.name,
        "source_dataset_sha256": actual_hash,
        "selection_sets": {
            "smoke-18": _smoke_ids(derived, registry),
            "fallback-stratified-180": _fallback_ids(derived, registry),
            "full-450": [case["case_id"] for case in derived],
        },
        "cases": derived,
    }


def write_dataset(payload: dict[str, Any], output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(output.read_bytes()).hexdigest()
