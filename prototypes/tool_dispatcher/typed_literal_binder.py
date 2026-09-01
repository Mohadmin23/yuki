"""Pure typed deterministic binder for literal-source Yuki arguments.

The module has no filesystem, benchmark-gold, annotation, or model access. It
may use a selected tool/argument as a semantic anchor, but every emitted literal
character is copied from the immutable raw request (apart from a contract-
declared composite separator).
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from .argument_contract import LITERAL_SOURCE, tool_argument_contract


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    rule: str

    def as_dict(self, raw: str) -> dict[str, Any]:
        return {**asdict(self), "text": raw[self.start : self.end]}


SEARCH_PREFIXES = (
    r"Search the web for\s+",
    r"Find information online about\s+",
    r"Search the internet for\s+",
    r"Find online guides for\s+",
    r"Search online for\s+",
    r"Web search for\s+",
    r"Search query:\s*",
    r"Look up\s+",
    r"Search for\s+",
    r"Find\s+",
)

IMAGE_PREFIXES = (
    r"Generate an image with this exact prompt:\s*",
    r"Use the image tool with this exact prompt:\s*",
    r"Create a picture using exactly:\s*",
    r"Make artwork from this exact prompt, not a text file:\s*",
    r"Generate an image with this prompt:\s*",
    r"Use this exact image prompt:\s*",
    r"Create an image of\s+",
    r"Create a picture of\s+",
    r"Make artwork showing\s+",
    r"Generate this scene:\s*",
    r"Generate exactly:\s*",
    r"Render\s+",
    r"Draw\s+",
    r"Illustrate\s+",
)

ASK_PREFIXES = (
    r"Send this exact question to Claude:\s*",
    r"Delegate this exact question to Claude:\s*",
    r"Have the other Claude review this message:\s*",
    r"Ask the other Claude agent this:\s*",
    r"Send Claude this exact question:\s*",
    r"Send Claude the request:\s*",
    r"Ask Claude, not web search:\s*",
    r"Ask Claude exactly:\s*",
    r"Consult Claude:\s*",
    r"Ask Claude:\s*",
    r"Ask Claude to\s+",
    r"Have Claude\s+",
)

READ_PREFIXES = (
    r"Show me the contents of\s+",
    r"Read the filesystem file\s+",
    r"Open and read\s+",
    r"Open the local file\s+",
    r"Inspect the file at\s+",
    r"Read the file at\s+",
    r"Load the local file\s+",
    r"Read exactly\s+",
    r"Read\s+",
)

SHELL_PREFIXES = (
    r"Run this allowed shell command:\s*",
    r"Use the terminal command\s+",
    r"Execute\s+",
    r"Run\s+",
)

YUKI_READ_PATTERNS = (
    r"^Read my Yuki notes file (?P<payload>.+?)[.]?$",
    r"^Open (?P<payload>.+?) from Yuki's personal folder[.]?$",
    r"^Show the Yuki file (?P<payload>.+?)[.]?$",
    r"^Use Yuki read for (?P<payload>.+?),\s*not\b",
    r"^Open Yuki's (?P<payload>.+?) file[.]?$",
    r"^From Yuki's files, read (?P<payload>.+?)[.]?$",
    r"^Read (?P<payload>.+?) from Yuki's scratch space[.]?$",
    r"^Use yuki_read to open (?P<payload>.+?)[.]?$",
    r"^Retrieve (?P<payload>.+?) from Yuki's notes[.]?$",
    r"^Read (?P<payload>.+?) in Yuki's own files[.]?$",
)

YUKI_DELETE_PATTERNS = (
    r"^Delete (?P<payload>.+?) from Yuki's folder[.]?$",
    r"^Remove the Yuki file (?P<payload>.+?)[.]?$",
    r"^Erase (?P<payload>.+?) from Yuki's personal files[.]?$",
    r"^Use Yuki delete on (?P<payload>.+?);\s*do not\b",
)


def _trim(raw: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end


def _strip_sentence_terminal(
    raw: str,
    start: int,
    end: int,
    *,
    preserve_question: bool = False,
) -> tuple[int, int]:
    start, end = _trim(raw, start, end)
    terminal_is_syntax = end > start and (
        raw[end - 1] == "."
        or (raw[end - 1] == "?" and not preserve_question)
    )
    if terminal_is_syntax:
        end -= 1
    return _trim(raw, start, end)


def _cut_first(raw: str, start: int, end: int, patterns: tuple[str, ...]) -> int:
    segment = raw[start:end]
    matches = [
        match.start()
        for pattern in patterns
        if (match := re.search(pattern, segment, flags=re.IGNORECASE)) is not None
    ]
    return start + min(matches) if matches else end


def _prefix_candidate(
    raw: str,
    prefixes: tuple[str, ...],
    *,
    suffixes: tuple[str, ...] = (),
    preserve_question: bool = False,
    rule: str,
) -> list[Candidate]:
    for prefix in prefixes:
        match = re.match(rf"^(?:{prefix})", raw, flags=re.IGNORECASE)
        if match is None:
            continue
        start = match.end()
        end = _cut_first(raw, start, len(raw), suffixes)
        start, end = _strip_sentence_terminal(
            raw,
            start,
            end,
            preserve_question=preserve_question,
        )
        return [Candidate(start, end, rule)] if start < end else []
    return []


def _capture_candidates(
    raw: str,
    patterns: tuple[str, ...],
    *,
    rule: str,
) -> list[Candidate]:
    candidates = []
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match is None:
            continue
        start, end = _trim(raw, *match.span("payload"))
        if start < end:
            candidates.append(Candidate(start, end, rule))
    return _dedupe(candidates)


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple[int, int]] = set()
    result = []
    for candidate in candidates:
        key = (candidate.start, candidate.end)
        if candidate.start >= candidate.end or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _url_candidates(raw: str) -> list[Candidate]:
    candidates = []
    for match in re.finditer(r"https?://[^\s]+", raw):
        start, end = match.span()
        if end == len(raw) and raw[end - 1] in ".,":
            end -= 1
        candidates.append(Candidate(start, end, "url_token"))
    return _dedupe(candidates)


def _pipe_candidate(raw: str) -> list[Candidate]:
    match = re.search(r"[A-Za-z0-9_.-]+\.[A-Za-z0-9._-]+\|.+$", raw)
    if match is None:
        return []
    start, end = match.span()
    end = _cut_first(
        raw,
        start,
        end,
        (
            r",\s*not\b",
            r";\s*(?:do not|don't)\b",
            r"\s+(?:in|to|into)\s+(?:Yuki(?:'s)?|your own|the Yuki|a Yuki)",
        ),
    )
    start, end = _strip_sentence_terminal(raw, start, end)
    return [Candidate(start, end, "literal_filename_pipe_content")]


def _exact_hint_candidates(
    raw: str,
    selected_call: dict[str, Any],
    argument_name: str,
) -> list[Candidate]:
    arguments = selected_call.get("arguments")
    value = arguments.get(argument_name) if isinstance(arguments, dict) else None
    if not isinstance(value, str) or not value:
        return []
    candidates = []
    start = 0
    while True:
        index = raw.find(value, start)
        if index < 0:
            break
        candidates.append(Candidate(index, index + len(value), "exact_model_hint"))
        start = index + 1
    return _dedupe(candidates)


def _structural_candidates(raw: str, tool: str) -> list[Candidate]:
    if tool == "fetch":
        return _url_candidates(raw)
    if tool == "search":
        return _prefix_candidate(
            raw,
            SEARCH_PREFIXES,
            suffixes=(r";\s*there\b",),
            rule="search_payload",
        )
    if tool == "image":
        return _prefix_candidate(raw, IMAGE_PREFIXES, rule="image_payload")
    if tool == "read":
        return _prefix_candidate(
            raw,
            READ_PREFIXES,
            suffixes=(
                r"\s+from the filesystem\b",
                r"\s+from disk\b",
                r"\s+from the computer\b",
            ),
            rule="filesystem_path",
        )
    if tool == "shell":
        return _prefix_candidate(
            raw,
            SHELL_PREFIXES,
            suffixes=(r"\s+in the shell\b", r",\s*do not\b"),
            rule="shell_command",
        )
    if tool in {"yuki_write", "yuki_append"}:
        return _pipe_candidate(raw)
    if tool == "yuki_read":
        return _capture_candidates(raw, YUKI_READ_PATTERNS, rule="yuki_filename")
    if tool == "yuki_delete":
        return _capture_candidates(raw, YUKI_DELETE_PATTERNS, rule="yuki_filename")
    if tool == "ask_claude":
        return _prefix_candidate(
            raw,
            ASK_PREFIXES,
            preserve_question=True,
            rule="external_agent_payload",
        )
    return []


def _source_invariant(
    raw: str,
    candidate: Candidate,
    value: str,
    *,
    components: tuple[str, ...],
    separator: str | None,
) -> tuple[bool, list[dict[str, Any]]]:
    if not components:
        return value == raw[candidate.start : candidate.end], [
            candidate.as_dict(raw)
        ]
    if separator is None:
        return False, []
    parts = value.split(separator, len(components) - 1)
    if len(parts) != len(components):
        return False, []
    spans = []
    cursor = candidate.start
    for name, part in zip(components, parts, strict=True):
        index = raw.find(part, cursor, candidate.end)
        if index < 0:
            return False, []
        spans.append(
            {"name": name, "start": index, "end": index + len(part), "text": part}
        )
        cursor = index + len(part) + len(separator)
    return separator.join(part["text"] for part in spans) == value, spans


def bind_typed_call(
    *,
    raw_request: str,
    selected_call: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind literal fields, pass semantic fields through, or explicitly abstain."""

    if selected_call is None:
        return {
            "status": "no_selected_call",
            "selected_tool": None,
            "argument_mode": None,
            "candidates": [],
            "final_call": None,
            "source_copy_valid": None,
        }
    tool = selected_call.get("tool")
    if not isinstance(tool, str):
        return {
            "status": "malformed_selected_call",
            "selected_tool": None,
            "argument_mode": None,
            "candidates": [],
            "final_call": None,
            "source_copy_valid": None,
        }
    try:
        contract = tool_argument_contract(tool)
    except KeyError:
        return {
            "status": "unsupported_selected_tool",
            "selected_tool": tool,
            "argument_mode": None,
            "candidates": [],
            "final_call": None,
            "source_copy_valid": None,
        }
    if contract["mode"] != LITERAL_SOURCE:
        return {
            "status": "typed_passthrough",
            "selected_tool": tool,
            "argument_mode": contract["mode"],
            "candidates": [],
            "final_call": deepcopy(selected_call),
            "source_copy_valid": None,
        }

    argument_name = contract["argument"]
    candidates = _structural_candidates(raw_request, tool)
    if not candidates:
        candidates = _exact_hint_candidates(raw_request, selected_call, argument_name)
    candidates = _dedupe(candidates)
    candidate_data = [candidate.as_dict(raw_request) for candidate in candidates]
    if not candidates:
        return {
            "status": "unbound",
            "selected_tool": tool,
            "argument_mode": contract["mode"],
            "argument_name": argument_name,
            "candidates": [],
            "final_call": None,
            "source_copy_valid": None,
        }
    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "selected_tool": tool,
            "argument_mode": contract["mode"],
            "argument_name": argument_name,
            "candidates": candidate_data,
            "final_call": None,
            "source_copy_valid": None,
        }

    candidate = candidates[0]
    value = raw_request[candidate.start : candidate.end]
    components = tuple(contract.get("components", ()))
    separator = contract.get("deterministic_separator")
    source_valid, source_spans = _source_invariant(
        raw_request,
        candidate,
        value,
        components=components,
        separator=separator,
    )
    return {
        "status": "bound_composite" if components else "bound",
        "selected_tool": tool,
        "argument_mode": contract["mode"],
        "argument_name": argument_name,
        "source_span": candidate.as_dict(raw_request),
        "source_spans": source_spans,
        "deterministic_separator": separator,
        "candidates": candidate_data,
        "final_call": {"tool": tool, "arguments": {argument_name: value}},
        "source_copy_valid": source_valid,
    }
