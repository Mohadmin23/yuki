"""Canonical main-brain delegation contract and deterministic Stage A scoring."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from .registry import ToolRegistry

DELEGATION_DOMAINS = (
    "information",
    "computer",
    "media",
    "yuki_files",
    "memory",
    "external_agent",
)

_REGISTRY_GROUP_BY_DOMAIN = {
    "information": "information",
    "computer": "computer",
    "media": "media",
    "yuki_files": "yuki-files",
    "memory": "memory",
    "external_agent": "external-agent",
}

DELEGATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "request": {"type": "string", "minLength": 1},
        "domain_hint": {"type": "string", "enum": list(DELEGATION_DOMAINS)},
        "verbatim": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
    "required": ["request", "domain_hint", "verbatim"],
    "additionalProperties": False,
}

# These cues make semantic scoring deterministic and auditable. They are used only
# after generation and are never included as gold hints in the model prompt.
_SEMANTIC_CUES: dict[str, tuple[str, ...]] = {
    "time": ("current time", "current date", "day of the week", "what time"),
    "weather": ("weather", "temperature", "conditions", "forecast"),
    "fetch": ("url", "webpage", "page content", "retrieve content", "download"),
    "search": ("web search", "search online", "search for information", "find online", "look up", "find information"),
    "calc": ("calculate", "compute", "arithmetic", "evaluate", "mathematical"),
    "hardware": ("hardware", "system metric", "resource", "memory usage", "ram usage", "cpu usage", "disk usage", "processor", "storage", "temperature", "running processes"),
    "see": ("camera", "vision", "visually inspect", "look for", "locate the"),
    "image": ("generate an image", "create an image", "generate a", "create a", "render", "illustration", "visual artwork"),
    "read": ("normal filesystem", "local file", "file contents", "specified file", "open the file", "read the file"),
    "shell": ("terminal", "command", "directory listing", "filesystem operation", "command line"),
    "yuki_write": ("create a yuki", "new yuki", "overwrite a yuki", "store content in yuki"),
    "yuki_read": ("read a yuki", "yuki file contents", "open a yuki", "existing yuki", "read the contents of a file"),
    "yuki_list": ("list yuki", "show yuki", "yuki's files", "yuki files available"),
    "yuki_delete": ("delete a yuki", "remove a yuki", "erase a yuki", "delete a file", "remove a file"),
    "yuki_append": ("append to a yuki", "add to a yuki", "continue a yuki", "extend a yuki", "append the", "append text", "append content"),
    "remember": ("store", "remember for later", "retain", "save in memory", "future memory"),
    "recall": ("previously stored", "retrieve stored", "retrieve from memory", "past memory", "remembered information", "previous conversation"),
    "ask_claude": ("claude", "external agent", "delegate to another agent"),
}


def domain_tools(registry: ToolRegistry) -> dict[str, tuple[str, ...]]:
    """Resolve canonical delegation domains through the live registry groups."""

    return {
        domain: registry.resolve_names(group=registry_group)
        for domain, registry_group in _REGISTRY_GROUP_BY_DOMAIN.items()
    }


def expected_domain_for_tool(tool: str, registry: ToolRegistry) -> str:
    matches = [
        domain for domain, tools in domain_tools(registry).items() if tool in tools
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one delegation domain for {tool!r}, got {matches!r}"
        )
    return matches[0]


def parse_delegation(raw_text: str) -> dict[str, Any]:
    """Parse and validate exactly one canonical delegation JSON object."""

    try:
        decoded = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "malformed": True,
            "delegation": None,
            "decoded": None,
            "errors": [f"Invalid JSON: {exc.msg} at position {exc.pos}."],
        }

    errors: list[str] = []
    expected_keys = {"request", "domain_hint", "verbatim"}
    if not isinstance(decoded, dict):
        errors.append("Delegation must be a JSON object.")
    else:
        if set(decoded) != expected_keys:
            errors.append(
                "Delegation must contain exactly request, domain_hint, and verbatim."
            )
        request = decoded.get("request")
        if not isinstance(request, str) or not request.strip():
            errors.append("request must be a non-empty string.")
        domain_hint = decoded.get("domain_hint")
        if domain_hint not in DELEGATION_DOMAINS:
            errors.append("domain_hint is not one of the six canonical domains.")
        verbatim = decoded.get("verbatim")
        if not isinstance(verbatim, list):
            errors.append("verbatim must be an array of strings.")
        elif any(not isinstance(value, str) or not value for value in verbatim):
            errors.append("Every verbatim entry must be a non-empty string.")
        elif len(verbatim) != len(set(verbatim)):
            errors.append("verbatim entries must not be duplicated.")

    passed = not errors
    return {
        "passed": passed,
        "malformed": not passed,
        "delegation": decoded if passed else None,
        "decoded": decoded,
        "errors": errors,
    }


def final_tool_leakage(request: str, tool_names: Iterable[str]) -> list[str]:
    """Detect implementation-directed tool references, not ordinary action verbs."""

    leaks: list[str] = []
    for tool in tool_names:
        escaped = re.escape(tool)
        patterns = (
            rf"\b(?:use|call|invoke|select|choose)\s+(?:the\s+)?(?:tool\s+)?[`'\"]?{escaped}\b",
            rf"[`'\"]{escaped}[`'\"]\s*(?:tool|function)",
            rf"\b{escaped}\s*\(",
            rf"/{escaped}\b",
        )
        if tool.startswith("yuki_") or tool == "ask_claude":
            patterns += (rf"\b{escaped}\b",)
        if any(re.search(pattern, request, flags=re.IGNORECASE) for pattern in patterns):
            leaks.append(tool)
    return leaks


def semantic_action_match(request: str, expected_tool: str) -> bool:
    normalized = " ".join(request.casefold().split())
    return any(cue in normalized for cue in _SEMANTIC_CUES[expected_tool])


def score_delegation(
    parsed: dict[str, Any],
    *,
    expected_tool: str,
    expected_domain: str,
    required_verbatim: list[str],
    live_tool_names: Iterable[str],
) -> dict[str, Any]:
    """Score Stage A without an LLM judge or any tool execution."""

    delegation = parsed["delegation"]
    if delegation is None:
        return {
            "delegation_emitted": False,
            "malformed_delegation": True,
            "domain_hint_exact": False,
            "semantic_request_quality": False,
            "semantic_action_match": False,
            "verbatim_required": len(required_verbatim),
            "verbatim_emitted": 0,
            "verbatim_hits": 0,
            "verbatim_recall": 0.0 if required_verbatim else 1.0,
            "verbatim_precision": 0.0,
            "final_tool_leakage": False,
            "leaked_tools": [],
            "tool_should_be_used": True,
        }

    emitted = delegation["verbatim"]
    hits = sum(value in emitted for value in required_verbatim)
    recall = hits / len(required_verbatim) if required_verbatim else 1.0
    precision = hits / len(emitted) if emitted else (1.0 if not required_verbatim else 0.0)
    leaks = final_tool_leakage(delegation["request"], live_tool_names)
    action_match = semantic_action_match(delegation["request"], expected_tool)
    domain_exact = delegation["domain_hint"] == expected_domain
    return {
        "delegation_emitted": True,
        "malformed_delegation": False,
        "domain_hint_exact": domain_exact,
        "semantic_request_quality": domain_exact and action_match and not leaks,
        "semantic_action_match": action_match,
        "verbatim_required": len(required_verbatim),
        "verbatim_emitted": len(emitted),
        "verbatim_hits": hits,
        "verbatim_recall": recall,
        "verbatim_precision": precision,
        "final_tool_leakage": bool(leaks),
        "leaked_tools": leaks,
        "tool_should_be_used": True,
    }
