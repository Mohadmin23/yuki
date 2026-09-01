"""Typed literal binder v2 with general external-agent boundary parsing.

The parser consumes exactly one top-level delegation envelope. It never scans
recursively for phrases inside the payload, so strings such as ``Ask Claude:``
or ``handle this exact question:`` remain intact when they occur after the
outer boundary.
"""

from __future__ import annotations

import re
from typing import Any

from .argument_contract import LITERAL_SOURCE, tool_argument_contract
from .typed_literal_binder import bind_typed_call as bind_v1

CLAUDE_MARKER = re.compile(r"\bClaude\b", flags=re.IGNORECASE)
CLAUDE_LEAD = re.compile(
    r"^(?:"
    r"Ask\s+|Have\s+|Tell\s+|Send\b.*?\s+to\s+|Send\s+|Pass\b.*?\s+to\s+|"
    r"Delegate\s+to\s+|Consult\s+|Could\s+you\s+ask\s+|"
    r"Please\s+have\s+|I\s+need\s+|I\s+want\s+"
    r")?Claude\b",
    flags=re.IGNORECASE,
)

# A delimiter is structural only while parsing the single outer envelope. The
# vocabulary describes roles (action and transport noun), not benchmark strings.
CLAUDE_BOUNDARY = re.compile(
    r"^\s*(?:"
    r"(?P<direct>[:—])"
    r"|(?:(?:to\s+)?(?:handle|answer|review|inspect|process|consider)\s+)?"
    r"(?:this|the)\s+(?:exact\s+)?(?:question|message|request|task)\s*:"
    r"|(?:to\s+)?(?:handle|answer|review|inspect|process|consider)\s*:"
    r")\s*",
    flags=re.IGNORECASE,
)

CLAUDE_DIRECT_VERB = re.compile(
    r"^\s*(?:to\s+)?(?P<payload>.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)

OUTER_DELIMITERS = {'"': '"', "'": "'", "`": "`"}


def _trim(raw: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end


def _payload_span(
    raw: str,
    start: int,
    end: int,
    *,
    allow_outer_delimiter: bool,
) -> tuple[int, int, str]:
    start, end = _trim(raw, start, end)
    if allow_outer_delimiter and end - start >= 2:
        opener = raw[start]
        if opener in OUTER_DELIMITERS and raw[end - 1] == OUTER_DELIMITERS[opener]:
            start, end = _trim(raw, start + 1, end - 1)
            return start, end, "outer_delimiter"
    if end > start and raw[end - 1] == ".":
        end -= 1
    start, end = _trim(raw, start, end)
    return start, end, "sentence_boundary"


def _source_spans(
    raw: str,
    start: int,
    end: int,
    components: tuple[str, ...],
    separator: str | None,
) -> tuple[bool, list[dict[str, Any]]]:
    value = raw[start:end]
    if not components:
        return True, [{"start": start, "end": end, "text": value}]
    if separator is None:
        return False, []
    parts = value.split(separator, len(components) - 1)
    if len(parts) != len(components):
        return False, []
    spans = []
    cursor = start
    for name, part in zip(components, parts, strict=True):
        part_start = raw.find(part, cursor, end)
        if part_start < 0:
            return False, []
        spans.append(
            {
                "name": name,
                "start": part_start,
                "end": part_start + len(part),
                "text": part,
            }
        )
        cursor = part_start + len(part) + len(separator)
    return separator.join(span["text"] for span in spans) == value, spans


def _bound_result(
    *,
    raw: str,
    tool: str,
    start: int,
    end: int,
    rule: str,
) -> dict[str, Any]:
    contract = tool_argument_contract(tool)
    argument_name = contract["argument"]
    components = tuple(contract.get("components", ()))
    separator = contract.get("deterministic_separator")
    source_valid, spans = _source_spans(
        raw,
        start,
        end,
        components,
        separator,
    )
    candidate = {"start": start, "end": end, "rule": rule, "text": raw[start:end]}
    return {
        "status": "bound_composite" if components else "bound",
        "selected_tool": tool,
        "argument_mode": LITERAL_SOURCE,
        "argument_name": argument_name,
        "source_span": candidate,
        "source_spans": spans,
        "deterministic_separator": separator,
        "candidates": [candidate],
        "final_call": {
            "tool": tool,
            "arguments": {argument_name: raw[start:end]},
        },
        "source_copy_valid": source_valid,
    }


def _bind_claude(raw: str, selected_call: dict[str, Any]) -> dict[str, Any] | None:
    marker = CLAUDE_MARKER.search(raw)
    if marker is None or CLAUDE_LEAD.match(raw[: marker.end()]) is None:
        return None
    tail_start = marker.end()
    tail = raw[tail_start:]
    boundary = CLAUDE_BOUNDARY.match(tail)
    if boundary is not None:
        start = tail_start + boundary.end()
        rule = "single_outer_claude_envelope"
    else:
        direct = CLAUDE_DIRECT_VERB.match(tail)
        if direct is None:
            return None
        start = tail_start + direct.start("payload")
        rule = "claude_direct_verb_payload"
    start, end, delimiter_rule = _payload_span(
        raw,
        start,
        len(raw),
        allow_outer_delimiter=True,
    )
    if start >= end:
        return None
    return _bound_result(
        raw=raw,
        tool=selected_call["tool"],
        start=start,
        end=end,
        rule=f"{rule}:{delimiter_rule}",
    )


def _refine_v1_literal(
    raw: str,
    selected_call: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    if binding["status"] not in {"bound", "bound_composite"}:
        return binding
    source = binding.get("source_span")
    if not isinstance(source, dict):
        return binding
    start = source["start"]
    end = source["end"]
    if end < len(raw) and raw[end] == "?":
        end += 1
    tool = selected_call["tool"]
    allow_outer = tool in {"search", "image"}
    start, end, delimiter_rule = _payload_span(
        raw,
        start,
        end,
        allow_outer_delimiter=allow_outer,
    )
    if start >= end:
        return binding
    return _bound_result(
        raw=raw,
        tool=tool,
        start=start,
        end=end,
        rule=f"v1_structural_refinement:{delimiter_rule}",
    )


def bind_typed_call(
    *,
    raw_request: str,
    selected_call: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind one typed call without annotations, gold, models, or recursion."""

    if selected_call is not None and selected_call.get("tool") == "ask_claude":
        claude = _bind_claude(raw_request, selected_call)
        if claude is not None:
            return claude
    baseline = bind_v1(
        raw_request=raw_request,
        selected_call=selected_call,
    )
    if selected_call is None:
        return baseline
    try:
        contract = tool_argument_contract(selected_call.get("tool"))
    except (KeyError, TypeError):
        return baseline
    if contract["mode"] != LITERAL_SOURCE:
        return baseline
    return _refine_v1_literal(raw_request, selected_call, baseline)
