"""Oracle-only schema subsets for diagnosing tool competition.

These profiles use benchmark labels that would not be available in production. They
must never be reused as an automatic router.
"""
from __future__ import annotations

from dataclasses import dataclass

from .registry import TOOL_GROUPS, ToolRegistry

LOGICAL_DOMAINS = "logical-domains"
TARGETED_CONFUSIONS = "targeted-confusions"
REMEMBER_VS_RECALL = "remember-vs-recall"

ORACLE_PROFILES = (
    LOGICAL_DOMAINS,
    TARGETED_CONFUSIONS,
    REMEMBER_VS_RECALL,
)


@dataclass(frozen=True)
class OracleSubset:
    """One benchmark-only subset selected using the expected answer."""

    name: str
    tools: tuple[str, ...]


_TOOL_TO_DOMAIN = {
    tool: group
    for group, tools in TOOL_GROUPS.items()
    for tool in tools
}

_TARGETED_TAG_SUBSETS: dict[str, OracleSubset] = {
    "weather-routing": OracleSubset(
        "weather-vs-search-fetch", ("weather", "search", "fetch")
    ),
    "see-camera-routing": OracleSubset("see-vs-image", ("see", "image")),
    "read-vs-yuki-read": OracleSubset(
        "read-vs-yuki-read", ("read", "yuki_read")
    ),
    "search-preservation": OracleSubset(
        "search-vs-fetch", ("search", "fetch")
    ),
    "filepath-preservation": OracleSubset(
        "normal-filesystem", ("hardware", "read", "shell")
    ),
    "ask-claude-preservation": OracleSubset(
        "external-agent", ("ask_claude",)
    ),
}


def oracle_subset_for_case(
    profile: str,
    expected_tool: str,
    tags: tuple[str, ...],
    registry: ToolRegistry,
) -> OracleSubset | None:
    """Return the benchmark-label-derived subset, or skip an irrelevant case."""

    if profile == LOGICAL_DOMAINS:
        try:
            group = _TOOL_TO_DOMAIN[expected_tool]
        except KeyError as exc:
            raise ValueError(
                f"No logical oracle domain contains tool {expected_tool!r}"
            ) from exc
        subset = OracleSubset(group, TOOL_GROUPS[group])
    elif profile == TARGETED_CONFUSIONS:
        matches = [_TARGETED_TAG_SUBSETS[tag] for tag in tags if tag in _TARGETED_TAG_SUBSETS]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError(
                f"Case matches multiple targeted oracle subsets: {tags!r}"
            )
        subset = matches[0]
    elif profile == REMEMBER_VS_RECALL:
        if expected_tool not in {"remember", "recall"}:
            return None
        subset = OracleSubset("remember-vs-recall", ("remember", "recall"))
    else:
        raise ValueError(f"Unknown oracle profile: {profile}")

    resolved = registry.resolve_names(selected=subset.tools)
    if expected_tool not in resolved:
        raise ValueError(
            f"Oracle subset {subset.name!r} omits expected tool {expected_tool!r}"
        )
    return OracleSubset(subset.name, resolved)
