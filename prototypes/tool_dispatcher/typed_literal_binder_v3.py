"""Grammar-oriented source-span binder for typed literal Yuki fields.

V3 was designed against the fresh 200-case development split only. It has no
filesystem, corpus, annotation, benchmark, model, or tool-execution access.
Every emitted argument is an immutable-request slice.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .argument_contract import LITERAL_SOURCE, tool_argument_contract

CLAUDE = re.compile(r"\bClaude\b", re.IGNORECASE)
OUTER_MARKS = {'"': '"', "'": "'", "`": "`"}
SEPARATORS = (" → ", " — ", ":")

ASK_ENVELOPE_WORDS = {
    "a",
    "along",
    "answer",
    "consider",
    "exact",
    "exactly",
    "following",
    "for",
    "gets",
    "handle",
    "inspect",
    "is",
    "me",
    "message",
    "not",
    "please",
    "question",
    "request",
    "review",
    "search",
    "task",
    "the",
    "this",
    "to",
    "web",
}
ASK_ACTIONS = {"answer", "consider", "handle", "inspect", "review"}
TRANSPORT_NOUNS = {"message", "question", "request", "task"}


def _trim(raw: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end


def _words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", value.casefold())


def _separator(raw: str, start: int) -> tuple[int, int] | None:
    matches = []
    for token in SEPARATORS:
        marker = token.strip()
        index = raw.find(marker, start)
        if index >= 0:
            end = index + len(marker)
            while end < len(raw) and raw[end].isspace():
                end += 1
            matches.append((index, end))
    return min(matches) if matches else None


def _finish_span(raw: str, start: int, end: int) -> tuple[int, int, str]:
    start, end = _trim(raw, start, end)
    rule = "plain"
    if end > start and raw[end - 1] in ".!":
        end -= 1
        start, end = _trim(raw, start, end)
        rule = "sentence_terminal"
    if end - start >= 2:
        opener = raw[start]
        if opener in OUTER_MARKS and raw[end - 1] == OUTER_MARKS[opener]:
            start, end = _trim(raw, start + 1, end - 1)
            rule = "outer_delimiter"
    return start, end, rule


def _ask_start(raw: str) -> tuple[int, str] | None:
    marker = CLAUDE.search(raw)
    if marker is None or marker.start() > 48:
        return None
    tail_start = marker.end()
    tail_start, _ = _trim(raw, tail_start, len(raw))
    tail = raw[tail_start:]
    if tail.casefold().startswith("to "):
        direct_start = tail_start + 3
        lead_words = _words(raw[: marker.start()])
        if any(word in {"ask", "have", "tell"} for word in lead_words):
            return direct_start, "claude_direct_infinitive"
        separator = _separator(raw, direct_start)
        if separator is not None:
            preamble = raw[direct_start : separator[0]]
            words = _words(preamble)
            if (
                words
                and words[0] in ASK_ACTIONS
                and any(word in TRANSPORT_NOUNS for word in words)
                and set(words) <= ASK_ENVELOPE_WORDS
            ):
                return separator[1], "claude_action_transport_envelope"
        return direct_start, "claude_direct_infinitive"

    separator = _separator(raw, tail_start)
    if separator is not None:
        preamble = raw[tail_start : separator[0]]
        words = _words(preamble)
        immediate = not preamble.strip(" \t\r\n,")
        grammatical = (
            bool(words)
            and len(words) <= 9
            and set(words) <= ASK_ENVELOPE_WORDS
        )
        if immediate or grammatical:
            return separator[1], "single_claude_envelope"
    return tail_start, "claude_direct_payload"


def _delimiter_start(
    raw: str,
    *,
    cues: tuple[str, ...],
    reject_direct_object: tuple[str, ...] = (),
) -> tuple[int, str] | None:
    separator = _separator(raw, 0)
    if separator is None:
        return None
    left = raw[: separator[0]].casefold()
    if not any(cue in left for cue in cues):
        return None
    for marker in reject_direct_object:
        found = left.rfind(marker)
        if found >= 0 and left[found + len(marker) :].strip():
            return None
    return separator[1], "structural_separator"


def _regex_start(raw: str, patterns: tuple[str, ...]) -> tuple[int, str] | None:
    for pattern in patterns:
        match = re.match(pattern, raw, re.IGNORECASE | re.DOTALL)
        if match is not None:
            return match.end(), "grammatical_prefix"
    return None


SEARCH_PATTERNS = (
    r"^(?:search|query)(?:\s+the)?(?:\s+web|\s+internet|\s+online)?\s+(?:for|about)\s*:?[ ]*",
    r"^find(?:\s+results)?(?:\s+this)?(?:\s+on\s+the\s+internet)?\s+(?:for|about)\s*:?[ ]*",
    r"^look(?:\s+this)?\s+up(?:\s+online)?(?:\s+for)?\s*:?[ ]*",
    r"^(?:i\s+need\s+)?information\s+about\s*:?[ ]*",
)

IMAGE_PATTERNS = (
    r"^(?:draw|render|illustrate)\s+(?:this|the)\s+(?:scene|prompt|description)\s*:\s*",
    r"^(?:generate|create|make)\s+(?:a\s+|an\s+|new\s+)?(?:image|picture|artwork)\s+(?:of|showing)\s+",
    r"^(?:render|draw|illustrate)(?:\s+exactly)?\s+",
)

READ_PATTERNS = (
    r"^read\s+exactly\s+",
    r"^(?:read|inspect|open|load|show\s+me)\s+(?:(?:the|this)\s+)?(?:(?:os|local|filesystem|disk)\s+)?(?:file|path|filepath)(?:\s+at)?\s+",
)

SHELL_PATTERNS = (
    r"^(?:run\s+(?:this|the)\s+(?:exact\s+)?(?:shell\s+)?command|execute\s+in\s+the\s+terminal)\s*(?::|—)\s*",
    r"^(?:use\s+the\s+terminal\s+to\s+)?(?:run|execute)\s+",
)

COMPOSITE_PATTERNS = (
    r"^(?:add\s+to\s+a\s+yuki\s+file\s+with|write\s+this\s+exact\s+yuki\s+file\s+payload|replace\s+a\s+yuki\s+note\s+using)\s+",
)

YUKI_FILE_PATTERNS = (
    r"^(?:open\s+yuki's\s+internal\s+note|read\s+this\s+file\s+from\s+yuki's\s+own\s+storage)\s+",
)


def _url_start(raw: str) -> tuple[int, int] | None:
    matches = list(re.finditer(r"https?://[^\s`\"']+", raw))
    if len(matches) != 1:
        return None
    return matches[0].span()


def _pipe_start(raw: str) -> tuple[int, str] | None:
    delimiter = _delimiter_start(
        raw,
        cues=("yuki", "note", "file", "storage", "append", "write"),
    )
    if delimiter is not None:
        return delimiter
    match = re.search(r"[A-Za-z0-9_. —:-]+\.[A-Za-z0-9._-]+\|", raw)
    if match is None:
        return None
    return match.start(), "filename_pipe_anchor"


def _structural_start(raw: str, tool: str) -> tuple[int, str] | None:
    if tool == "ask_claude":
        return _ask_start(raw)
    if tool == "fetch":
        span = _url_start(raw)
        return (span[0], "url_token") if span else None
    if tool == "search":
        delimiter = _delimiter_start(
            raw,
            cues=("search", "query", "look", "information", "web"),
            reject_direct_object=(" for ", " about "),
        )
        return delimiter or _regex_start(raw, SEARCH_PATTERNS)
    if tool == "image":
        direct = _regex_start(raw, IMAGE_PATTERNS)
        if direct is not None:
            return direct
        delimiter = _delimiter_start(
            raw,
            cues=("image", "picture", "artwork", "prompt", "render", "draw"),
            reject_direct_object=(" of ", " showing "),
        )
        return delimiter
    if tool == "read":
        direct = _regex_start(raw, READ_PATTERNS)
        if direct is not None:
            return direct
        delimiter = _delimiter_start(
            raw,
            cues=("read", "open", "file", "path", "filesystem", "disk"),
            reject_direct_object=(" at ",),
        )
        return delimiter
    if tool == "shell":
        direct = _regex_start(raw, SHELL_PATTERNS)
        if direct is not None:
            return direct
        delimiter = _delimiter_start(
            raw,
            cues=("shell", "terminal", "command", "execute", "run"),
        )
        return delimiter
    if tool in {"yuki_write", "yuki_append"}:
        direct = _regex_start(raw, COMPOSITE_PATTERNS)
        if direct is not None:
            return direct
        return _pipe_start(raw)
    if tool in {"yuki_read", "yuki_delete"}:
        direct = _regex_start(raw, YUKI_FILE_PATTERNS)
        if direct is not None:
            return direct
        return _delimiter_start(
            raw,
            cues=("yuki", "internal", "storage", "note", "filename"),
        )
    return None


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


def _empty(status: str, tool: str | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "selected_tool": tool,
        "argument_mode": None,
        "candidates": [],
        "final_call": None,
        "source_copy_valid": None,
    }


def bind_typed_call(
    *,
    raw_request: str,
    selected_call: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind one call from syntax and immutable source text only."""

    if selected_call is None:
        return _empty("no_selected_call")
    tool = selected_call.get("tool")
    if not isinstance(tool, str):
        return _empty("malformed_selected_call")
    try:
        contract = tool_argument_contract(tool)
    except KeyError:
        return _empty("unsupported_selected_tool", tool)
    if contract["mode"] != LITERAL_SOURCE:
        return {
            "status": "typed_passthrough",
            "selected_tool": tool,
            "argument_mode": contract["mode"],
            "candidates": [],
            "final_call": deepcopy(selected_call),
            "source_copy_valid": None,
        }

    if tool == "fetch":
        url_span = _url_start(raw_request)
        if url_span is None:
            return _empty("unbound", tool)
        start, end = url_span
        if end == len(raw_request) and raw_request[end - 1] in ".!":
            end -= 1
        rule = "url_token"
    else:
        boundary = _structural_start(raw_request, tool)
        if boundary is None:
            return _empty("unbound", tool)
        start, rule = boundary
        start, end, finish = _finish_span(raw_request, start, len(raw_request))
        rule = f"{rule}:{finish}"
    if start >= end:
        return _empty("unbound", tool)

    argument = contract["argument"]
    components = tuple(contract.get("components", ()))
    separator = contract.get("deterministic_separator")
    source_valid, spans = _source_spans(
        raw_request,
        start,
        end,
        components,
        separator,
    )
    candidate = {
        "start": start,
        "end": end,
        "rule": rule,
        "text": raw_request[start:end],
    }
    return {
        "status": "bound_composite" if components else "bound",
        "selected_tool": tool,
        "argument_mode": LITERAL_SOURCE,
        "argument_name": argument,
        "source_span": candidate,
        "source_spans": spans,
        "deterministic_separator": separator,
        "candidates": [candidate],
        "final_call": {
            "tool": tool,
            "arguments": {argument: raw_request[start:end]},
        },
        "source_copy_valid": source_valid,
    }
