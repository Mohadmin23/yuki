"""Cheap local domain selection for automatic schema-exposure experiments."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from .registry import TOOL_GROUPS, ToolRegistry

STRATEGY_NAME = "deterministic-domain-rules-v1"

# Public experiment names use underscores. Values point at the existing live-registry
# groups, so this module never becomes a second independent tool-schema registry.
DOMAIN_TO_REGISTRY_GROUP = {
    "information": "information",
    "computer": "computer",
    "media": "media",
    "yuki_files": "yuki-files",
    "memory": "memory",
    "external_agent": "external-agent",
}

TOOL_TO_DOMAIN = {
    tool: domain
    for domain, group in DOMAIN_TO_REGISTRY_GROUP.items()
    for tool in TOOL_GROUPS[group]
}

_PATH_RE = re.compile(
    r"(?<![\w])(?:~/|/(?:Users|Volumes|private|tmp|var|etc|opt|home)/|\.{1,2}/)"
)
_FILENAME_RE = re.compile(
    r"\b[^\s/]+\.(?:txt|md|json|jsonl|yaml|yml|toml|py|log|csv|tsv)\b",
    re.IGNORECASE,
)
_YUKI_FILE_RE = re.compile(
    r"(?:"
    r"\byuki(?:'s)?\s+(?:personal\s+)?(?:file|files|folder|note|notes|read|write|list|delete|append)\b"
    r"|\byuki(?:'s)?\s+(?:\S+\s+){0,3}(?:file|files|folder|scratch space)\b"
    r"|\byuki[_ -](?:read|write|list|delete|append)\b"
    r"|\bfrom\s+yuki(?:'s)?\s+(?:personal\s+)?folder\b"
    r")",
    re.IGNORECASE,
)
_NORMAL_FILE_RE = re.compile(
    r"\b(?:filesystem|local file|from (?:the )?disk|on (?:this )?computer)\b",
    re.IGNORECASE,
)
_FILE_ACTION_RE = re.compile(
    r"\b(?:read|open|show|list|write|create|make|save|delete|remove|erase|append|add|extend)\b",
    re.IGNORECASE,
)
_SHELL_HARDWARE_RE = re.compile(
    r"\b(?:shell|terminal|command|cpu|ram|memory usage|disk usage|hardware|uname|pwd|df\s+-h|ls\s+-)\b",
    re.IGNORECASE,
)
_EXTERNAL_AGENT_RE = re.compile(
    r"(?:"
    r"\b(?:ask|send|have|tell)\s+(?:the\s+other\s+)?claude\b"
    r"|\b(?:send|delegate)\b.+\bto\s+claude\b"
    r"|\bconsult\s+claude\b"
    r"|\bclaude\s+(?:to|review|inspect)\b"
    r")",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"\b(?:remember|recall|remembered conversations?|previously tell|past discussion|previous conversations?|store this fact|keep this in memory)\b",
    re.IGNORECASE,
)
_MEDIA_RE = re.compile(
    r"\b(?:camera|webcam|vision|visually|yuki(?:'s)? eyes?|look through|look around the room|inspect my workspace|what you can see|spot my|locate my|generate an image|create a picture|make artwork|image tool|image prompt|picture using)\b",
    re.IGNORECASE,
)
_INFORMATION_RE = re.compile(
    r"\b(?:weather|time|clock|calculate|calculator|compute|search|look up|find information|fetch|exact url|https?://)\b",
    re.IGNORECASE,
)
_EXPLICIT_SEARCH_RE = re.compile(
    r"\b(?:search(?:\s+(?:the\s+)?(?:web|internet|online)|\s+for|\s+query)|web search|look up)\b",
    re.IGNORECASE,
)
_NEGATED_SEARCH_RE = re.compile(
    r"\b(?:do not|don't|without|rather than|not)\s+(?:search|searching)\b",
    re.IGNORECASE,
)
_EXCLUDE_YUKI_RE = re.compile(
    r"\bnot\s+(?:from\s+)?yuki(?:'s)?(?:\s+folder)?\b",
    re.IGNORECASE,
)
_EXCLUDE_COMPUTER_RE = re.compile(
    r"(?:"
    r"\bnot\s+(?:the\s+)?(?:general\s+)?filesystem(?:\s+reader)?\b"
    r"|\bnot\s+shell\b"
    r"|\bdo not run (?:a\s+)?shell command\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DomainSelection:
    """One selector decision, including reproducible diagnostics."""

    domains: tuple[str, ...]
    tools: tuple[str, ...]
    latency_ms: float
    matched_rules: tuple[str, ...]
    strategy: str = STRATEGY_NAME

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "domains": list(self.domains),
            "tools": list(self.tools),
            "latency_ms": self.latency_ms,
            "matched_rules": list(self.matched_rules),
        }


class AutomaticDomainSelector:
    """Deterministic, local, no-network selector for schema exposure only."""

    strategy_name = STRATEGY_NAME

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        missing = set(registry.names) - set(TOOL_TO_DOMAIN)
        if missing:
            raise RuntimeError(
                f"Automatic domain map omits live Yuki tools: {sorted(missing)}"
            )

    def expected_domain_for_tool(self, tool: str) -> str:
        try:
            return TOOL_TO_DOMAIN[tool]
        except KeyError as exc:
            raise ValueError(f"No automatic-selector domain contains {tool!r}") from exc

    def select_domains(self, request: str) -> DomainSelection:
        """Choose schema domains without deciding the final Yuki tool call."""

        started = time.perf_counter_ns()
        text = request.strip()
        if not text:
            raise ValueError("Automatic domain selection requires a non-empty request")

        domains, matched_rules = self._classify(text)
        requested_tools = {
            tool
            for domain in domains
            for tool in TOOL_GROUPS[DOMAIN_TO_REGISTRY_GROUP[domain]]
        }
        tools = tuple(
            tool for tool in self.registry.names if tool in requested_tools
        )
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        return DomainSelection(
            domains=domains,
            tools=tools,
            latency_ms=latency_ms,
            matched_rules=matched_rules,
        )

    @staticmethod
    def _classify(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        rules: list[str] = []

        if _EXTERNAL_AGENT_RE.search(text):
            return ("external_agent",), ("external-agent-phrase",)

        if _MEMORY_RE.search(text):
            return ("memory",), ("memory-action",)

        if _EXPLICIT_SEARCH_RE.search(text) and not _NEGATED_SEARCH_RE.search(text):
            return ("information",), ("explicit-search-action",)

        explicit_yuki = bool(_YUKI_FILE_RE.search(text))
        exclude_yuki = bool(_EXCLUDE_YUKI_RE.search(text))
        exclude_computer = bool(_EXCLUDE_COMPUTER_RE.search(text))
        explicit_computer = bool(
            _PATH_RE.search(text)
            or _NORMAL_FILE_RE.search(text)
            or _SHELL_HARDWARE_RE.search(text)
        )

        if explicit_yuki and not exclude_yuki:
            rules.append("explicit-yuki-file")
            domains = ["yuki_files"]
            if explicit_computer and not exclude_computer:
                rules.append("ambiguous-normal-and-yuki-file")
                domains.insert(0, "computer")
            return tuple(domains), tuple(rules)

        if explicit_computer:
            rules.append("explicit-computer")
            return ("computer",), tuple(rules)

        if _MEDIA_RE.search(text):
            return ("media",), ("media-action",)

        if _INFORMATION_RE.search(text):
            return ("information",), ("information-action",)

        if _FILE_ACTION_RE.search(text) and _FILENAME_RE.search(text):
            return (
                "computer",
                "yuki_files",
            ), ("ambiguous-unqualified-file",)

        # Delegation already asserts that a tool is needed. Information is the least
        # stateful fallback and preserves the common factual tool family.
        return ("information",), ("information-fallback",)
