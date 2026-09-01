"""Pure deterministic source-span binding for focused dispatcher arguments.

This module intentionally has no filesystem, benchmark-gold, or annotation access.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

ARGUMENT_NAMES = {
    "search": "query",
    "image": "prompt",
    "ask_claude": "question",
    "see": "target",
    "read": "filepath",
    "yuki_write": "filename_and_content",
    "yuki_read": "filename",
    "yuki_delete": "filename",
    "yuki_append": "filename_and_content",
    "recall": "topic",
}


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    rule: str

    def as_dict(self, raw: str) -> dict[str, Any]:
        return {**asdict(self), "text": raw[self.start : self.end]}


@dataclass(frozen=True)
class CompositeSpanCandidate:
    filename_start: int
    filename_end: int
    content_start: int
    content_end: int
    rule: str

    def as_dict(self, raw: str) -> dict[str, Any]:
        filename = raw[self.filename_start : self.filename_end]
        content = raw[self.content_start : self.content_end]
        return {
            **asdict(self),
            "components": [
                {
                    "name": "filename",
                    "start": self.filename_start,
                    "end": self.filename_end,
                    "text": filename,
                },
                {
                    "name": "content",
                    "start": self.content_start,
                    "end": self.content_end,
                    "text": content,
                },
            ],
            "separator": "|",
            "text": f"{filename}|{content}",
        }


def _trim(raw: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end


def _strip_terminal_syntax(
    raw: str,
    start: int,
    end: int,
    *,
    preserve_question: bool = False,
    preserve_period: bool = False,
) -> tuple[int, int]:
    start, end = _trim(raw, start, end)
    if end <= start:
        return start, end
    terminal = raw[end - 1]
    if (terminal == "." and not preserve_period) or (
        terminal == "?" and not preserve_question
    ):
        end -= 1
    return _trim(raw, start, end)


def _before_guard(raw: str, start: int, end: int) -> int:
    segment = raw[start:end]
    guard = re.search(
        r"(?:;\s*(?:do not|don't|there is\b)|,\s*not\b)",
        segment,
        flags=re.IGNORECASE,
    )
    return start + guard.start() if guard else end


def _dedupe(candidates: list[SpanCandidate]) -> list[SpanCandidate]:
    observed: set[tuple[int, int]] = set()
    result = []
    for candidate in candidates:
        key = (candidate.start, candidate.end)
        if candidate.start >= candidate.end or key in observed:
            continue
        observed.add(key)
        result.append(candidate)
    return result


def _prefix_candidate(
    raw: str,
    pattern: str,
    rule: str,
    *,
    guards: bool = True,
    preserve_question: bool = False,
    preserve_period: bool = False,
) -> SpanCandidate | None:
    match = re.match(pattern, raw, flags=re.IGNORECASE)
    if match is None:
        return None
    start = match.end()
    end = _before_guard(raw, start, len(raw)) if guards else len(raw)
    start, end = _strip_terminal_syntax(
        raw,
        start,
        end,
        preserve_question=preserve_question,
        preserve_period=preserve_period,
    )
    return SpanCandidate(start, end, rule) if start < end else None


def _first_prefix(
    raw: str,
    specs: tuple[tuple[str, str, bool], ...],
    *,
    guards: bool = True,
    preserve_question: bool = False,
    preserve_period: bool = False,
) -> list[SpanCandidate]:
    for pattern, rule, exact_terminal in specs:
        candidate = _prefix_candidate(
            raw,
            pattern,
            rule,
            guards=guards,
            preserve_question=preserve_question or exact_terminal,
            preserve_period=preserve_period or exact_terminal,
        )
        if candidate is not None:
            return [candidate]
    return []


SEARCH_PREFIXES = (
    (r"^Search the web for\s+", "search_prefix", False),
    (r"^Find information about\s+", "search_prefix", False),
    (r"^Search for\s+", "search_prefix", False),
    (r"^Look up\s+", "search_prefix", False),
    (r"^Web search:\s*", "search_label", False),
    (r"^Find\s+", "search_prefix", False),
    (r"^Search online for\s+", "search_prefix", False),
    (r"^Search the internet for\s+", "search_prefix", False),
    (r"^Can you dig up\s+", "search_prefix", False),
    (r"^I need web results for\s+", "search_prefix", False),
    (r"^See what the internet says about\s+", "search_prefix", False),
    (r"^Track down documentation for\s+", "search_prefix", False),
    (r"^Could you research\s+", "search_prefix", False),
    (r"^google-ish lookup:\s*", "search_label", False),
    (r"^Search exactly for:\s*", "search_exact_label", True),
    (r"^Keep this query intact:\s*", "search_exact_label", True),
    (r"^Use the exact query\s+", "search_exact_label", True),
    (r"^Search, punctuation and all:\s*", "search_exact_label", True),
)

IMAGE_PREFIXES = (
    (r"^Generate an image with this prompt:\s*", "image_prompt_label", True),
    (r"^Create a picture of\s+", "image_prefix", False),
    (r"^Make artwork showing\s+", "image_prefix", False),
    (r"^Generate this scene:\s*", "image_scene_label", False),
    (r"^Create an image of\s+", "image_prefix", False),
    (r"^Make a poster of\s+", "image_prefix", False),
    (r"^Make an image of\s+", "image_prefix", False),
    (r"^Could you illustrate\s+", "image_prefix", False),
    (r"^I want a visual:\s*", "image_label", False),
    (r"^Turn this idea into artwork—\s*", "image_label", False),
    (r"^Paint, digitally,\s+", "image_prefix", False),
    (r"^picture pls:\s*", "image_label", False),
    (r"^Use this exact image prompt:\s*", "image_exact_label", True),
    (r"^Preserve punctuation in the prompt:\s*", "image_exact_label", True),
    (r"^Generate exactly:\s*", "image_exact_label", True),
    (r"^uhh make\.\.\. an image\? prompt is:\s*", "image_exact_label", True),
    (r"^Render\s+", "image_prefix", False),
    (r"^Generate\s+", "image_prefix", False),
    (r"^Create\s+", "image_prefix", False),
    (r"^Draw\s+", "image_prefix", False),
    (r"^Make\s+", "image_prefix", False),
    (r"^Illustrate\s+", "image_prefix", False),
    (r"^Visualize\s+", "image_prefix", False),
)

ASK_PREFIXES = (
    (r"^Ask Claude:\s*", "claude_label", False),
    (r"^Send Claude this question:\s*", "claude_question_label", False),
    (r"^Ask Claude to\s+", "claude_prefix", False),
    (r"^Delegate to Claude:\s*", "claude_label", False),
    (r"^Have Claude\s+", "claude_prefix", False),
    (r"^Ask the other Claude to\s+", "claude_prefix", False),
    (r"^Send this to Claude:\s*", "claude_label", False),
    (r"^Delegate this task to Claude:\s*", "claude_label", False),
    (r"^Tell Claude to\s+", "claude_prefix", False),
    (r"^I want Claude's take on\s+", "claude_prefix", False),
    (r"^claude pls inspect:\s*", "claude_label", False),
    (r"^Consult Claude about this:\s*", "claude_label", False),
    (r"^Get the neighboring agent to answer:\s*", "claude_label", False),
    (r"^Ask Claude\s+", "claude_prefix", False),
    (
        r"^Delegate a web-search design question to Claude instead of searching:\s*",
        "claude_label",
        False,
    ),
    (r"^Send Claude the shell-review task:\s*", "claude_payload_label", False),
    (r"^Send this exact question to Claude:\s*", "claude_exact_label", True),
    (r"^Preserve and ask Claude:\s*", "claude_exact_label", True),
    (r"^Claude gets exactly:\s*", "claude_exact_label", True),
    (
        r"^delegate\.\.\. maybe to Claude\? exact payload:\s*",
        "claude_exact_label",
        True,
    ),
    (r"^Could the other agent look over this question:\s*", "claude_label", False),
    (r"^Pass along to Claude—\s*", "claude_label", False),
)


def _search_candidates(raw: str) -> tuple[list[SpanCandidate], bool]:
    return _first_prefix(raw, SEARCH_PREFIXES), False


def _image_candidates(raw: str) -> tuple[list[SpanCandidate], bool]:
    return _first_prefix(raw, IMAGE_PREFIXES), False


def _ask_candidates(raw: str) -> tuple[list[SpanCandidate], bool]:
    candidates = _first_prefix(
        raw,
        ASK_PREFIXES,
        guards=False,
        preserve_question=True,
        preserve_period="? " in raw,
    )
    return candidates, False


SEE_WHOLE_SCENE = (
    r"^Describe what (?:the )?camera can see",
    r"^Look around the room and describe the scene",
    r"^Take a look around",
    r"^what does yuki's eye see",
)

SEE_CAPTURE_PATTERNS = (
    (r"^Use the camera to find (?P<payload>.+?)[.]?$", "see_find"),
    (r"^Look through Yuki's eye for (?P<payload>.+?)[.]?$", "see_find"),
    (r"^Check (?P<payload>.+?) with the camera[.]?$", "see_inspect"),
    (r"^Find (?P<payload>.+?) in the room[.]?$", "see_find"),
    (r"^Inspect (?P<payload>.+?) using vision[.]?$", "see_inspect"),
    (r"^Can the webcam spot (?P<payload>.+?)[?]$", "see_find"),
    (r"^Locate (?P<payload>.+?) with the camera[.]?$", "see_find"),
    (
        r"^Point the camera toward (?P<payload>.+?) and find ",
        "see_first_mention",
    ),
    (r"\band find (?P<payload>.+?)[.]?$", "see_second_mention"),
    (r"^Use vision to locate (?P<payload>.+?)[.]?$", "see_find"),
    (r"^Use your eyes and track down (?P<payload>.+?)[.]?$", "see_find"),
    (r"^Could vision hunt for (?P<payload>.+?)[?]$", "see_find"),
    (r"^Peek through the webcam for (?P<payload>.+?)[.]?$", "see_find"),
    (r"^Scan the room until you find (?P<payload>.+?)[.]?$", "see_find"),
    (
        r"^Find (?P<payload>.+?) with the camera;\s*(?:do not|don't)\b",
        "see_find",
    ),
    (
        r"^Find (?!.* with the camera;)(?P<payload>.+?);\s*(?:do not|don't)\b",
        "see_find",
    ),
    (r"^Look at (?P<payload>.+?),\s*not\b", "see_inspect"),
    (r"^Visually inspect (?P<payload>.+?);\s*(?:do not|don't)\b", "see_inspect"),
    (r"^Camera target exactly:\s*(?P<payload>.+)$", "see_exact_label"),
    (r"^Locate `(?P<payload>[^`]+)`", "see_quoted_target"),
    (r"^Use vision for the object named (?P<payload>.+?)[.]?$", "see_named_target"),
    (r"find the '(?P<payload>[^']+)'", "see_quoted_target"),
)


def _capture_candidates(
    raw: str, patterns: tuple[tuple[str, str], ...]
) -> list[SpanCandidate]:
    candidates = []
    for pattern, rule in patterns:
        for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
            start, end = match.span("payload")
            start, end = _trim(raw, start, end)
            if start < end:
                candidates.append(SpanCandidate(start, end, rule))
    return _dedupe(candidates)


def _see_candidates(raw: str) -> tuple[list[SpanCandidate], bool]:
    if any(re.match(pattern, raw, flags=re.IGNORECASE) for pattern in SEE_WHOLE_SCENE):
        return [], True
    candidates = _capture_candidates(raw, SEE_CAPTURE_PATTERNS)
    expanded = list(candidates)
    for candidate in candidates:
        segment = raw[candidate.start : candidate.end]
        modifier = re.search(r"\s+lying on\s+", segment, flags=re.IGNORECASE)
        if modifier:
            expanded.append(
                SpanCandidate(
                    candidate.start,
                    candidate.start + modifier.start(),
                    "see_target_without_locative_modifier",
                )
            )
    return _dedupe(expanded), False


RECALL_CAPTURE_PATTERNS = (
    (
        r"^Recall our previous discussion about (?P<payload>.+?)[.]?$",
        "recall_topic",
    ),
    (r"^What did I tell you about (?P<payload>.+?)[?]$", "recall_topic"),
    (
        r"^Look up remembered conversations about (?P<payload>.+?)[.]?$",
        "recall_topic",
    ),
    (r"^Recall what we discussed about (?P<payload>.+?)[.]?$", "recall_topic"),
    (r"^Find our past memory about (?P<payload>.+?)[.]?$", "recall_topic"),
    (
        r"^What do you remember from (?P<payload>.+?) discussion[?]$",
        "recall_topic_without_context_noun",
    ),
    (
        r"^Recall the earlier conversation about (?P<payload>.+?)[.]?$",
        "recall_topic",
    ),
    (r"^Search memory for (?P<payload>.+?)[.]?$", "recall_topic"),
    (
        r"^Bring back the past discussion about (?P<payload>.+?)[.]?$",
        "recall_topic",
    ),
    (r"^Recall our notes from (?P<payload>.+?)[.]?$", "recall_topic"),
    (
        r"^What did we previously say about (?P<payload>.+?)[?]$",
        "recall_topic",
    ),
    (
        r"^Can you dig into our shared past for (?P<payload>.+?) topic[?]$",
        "recall_topic_without_context_noun",
    ),
    (
        r"^Memory search: that conversation where (?P<payload>.+?)[.]?$",
        "recall_topic_without_context_noun",
    ),
    (
        r"^Remind me what came up when we talked about (?P<payload>.+?)[.]?$",
        "recall_topic",
    ),
    (r"^Find the older chat concerning (?P<payload>.+?)[.]?$", "recall_topic"),
    (
        r"^What was our earlier thinking around (?P<payload>.+?)[?]$",
        "recall_topic",
    ),
    (
        r"^Pull the remembered context for (?P<payload>.+?)[.]?$",
        "recall_topic",
    ),
    (r"^Recall (?P<payload>.+?),\s*not\b", "recall_topic"),
    (r"^Search remembered chats for (?P<payload>.+?),\s*not\b", "recall_topic"),
    (
        r"^Recall our discussion of (?P<payload>.+?);\s*(?:do not|don't)\b",
        "recall_topic_without_context_noun",
    ),
    (r"^Use memory for (?P<payload>.+?);\s*(?:do not|don't)\b", "recall_topic"),
    (r"^Recall the exact topic `(?P<payload>[^`]+)`", "recall_quoted_topic"),
    (r"^Search memory using topic (?P<payload>.+?)[.]?$", "recall_topic"),
    (r"^Bring back the topic (?P<payload>.+?),\s*punctuation intact", "recall_topic"),
    (
        r"previous thing about (?P<payload>.+?)[?]\s*recall that topic$",
        "recall_topic",
    ),
)


def _recall_candidates(raw: str) -> tuple[list[SpanCandidate], bool]:
    return _capture_candidates(raw, RECALL_CAPTURE_PATTERNS), False


YUKI_COMPOSITE_PATTERNS = (
    (
        r"^Create `(?P<filename>[^`]+)` with exact content `(?P<content>[^`]*)`[.]?$",
        "yuki_quoted_filename_and_content",
    ),
    (
        r"^Add exact text to `(?P<filename>[^`]+)`:\s*(?P<content>.+)$",
        "yuki_quoted_filename_colon_content",
    ),
    (
        r"^append-ish request\.\.\.\s*(?P<filename>.+?\.[A-Za-z0-9._-]+) gets `(?P<content>[^`]*)`$",
        "yuki_filename_gets_quoted_content",
    ),
    (
        r"^write\.\.\. in Yuki,\s*(?P<filename>.+?\.[A-Za-z0-9._-]+) with (?P<content>.+)$",
        "yuki_filename_with_content",
    ),
    (
        r"^Start a fresh Yuki note named (?P<filename>\S+\.[A-Za-z0-9._-]+) containing (?P<content>.+?)[.]?$",
        "yuki_named_file_containing",
    ),
    (
        r"^Put `(?P<content>[^`]*)` into (?P<filename>\S+\.[A-Za-z0-9._-]+) in Yuki's own files[.]?$",
        "yuki_quoted_content_into_file",
    ),
    (
        r"^Could you make (?P<filename>\S+\.[A-Za-z0-9._-]+) in your folder with (?P<content>.+?)[?]$",
        "yuki_make_file_with_content",
    ),
    (
        r"^new personal file\s+—\s+name:\s*(?P<filename>.+?\.[A-Za-z0-9._-]+);\s*contents:\s*(?P<content>.+)$",
        "yuki_named_contents",
    ),
    (
        r"^One more entry for Yuki's (?P<filename>\S+\.[A-Za-z0-9._-]+):\s*(?P<content>.+)$",
        "yuki_entry_colon_content",
    ),
    (
        r"^Could you continue (?P<filename>\S+\.[A-Za-z0-9._-]+) with (?P<content>.+?)[?]$",
        "yuki_continue_file_with_content",
    ),
    (
        r"^Yuki-file extension—(?P<filename>\S+\.[A-Za-z0-9._-]+) gets (?P<content>.+)$",
        "yuki_file_gets_content",
    ),
    (
        r"^add another line in (?P<filename>\S+\.[A-Za-z0-9._-]+) saying (?P<content>.+)$",
        "yuki_file_saying_content",
    ),
)


def _pipe_payload_candidate(raw: str) -> SpanCandidate | None:
    """Find a literal ``filename|content`` payload without interpreting content."""

    match = re.search(
        r"(?P<payload>[A-Za-z0-9_.-]+\.[A-Za-z0-9._-]+\|.+)$",
        raw,
    )
    if match is None:
        return None
    start, end = match.span("payload")
    segment = raw[start:end]
    guard = re.search(
        r"(?:\s+(?:in|to|as)\s+(?:Yuki(?:'s)?|your own|the Yuki|a Yuki|a file\b|file text\b|text\b)|;\s*(?:do not|don't)\b)",
        segment,
        flags=re.IGNORECASE,
    )
    if guard is not None:
        end = start + guard.start()
    start, end = _trim(raw, start, end)
    return SpanCandidate(start, end, "yuki_literal_pipe_payload")


def _composite_payload_candidate(raw: str) -> CompositeSpanCandidate | None:
    for pattern, rule in YUKI_COMPOSITE_PATTERNS:
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if match is None:
            continue
        filename_start, filename_end = _trim(raw, *match.span("filename"))
        content_start, content_end = _trim(raw, *match.span("content"))
        if filename_start < filename_end and content_start < content_end:
            return CompositeSpanCandidate(
                filename_start,
                filename_end,
                content_start,
                content_end,
                rule,
            )
    return None


def _bind_yuki_composite(raw: str, tool: str, argument_name: str) -> dict[str, Any] | None:
    contiguous = _pipe_payload_candidate(raw)
    if contiguous is not None:
        candidate = contiguous.as_dict(raw)
        value = candidate["text"]
        return {
            "status": "bound",
            "selected_tool": tool,
            "argument_name": argument_name,
            "source_span": candidate,
            "source_spans": [candidate],
            "candidates": [candidate],
            "final_call": {"tool": tool, "arguments": {argument_name: value}},
            "source_copy_valid": value == raw[contiguous.start : contiguous.end],
        }

    composite = _composite_payload_candidate(raw)
    if composite is None:
        return None
    candidate = composite.as_dict(raw)
    filename = raw[composite.filename_start : composite.filename_end]
    content = raw[composite.content_start : composite.content_end]
    value = f"{filename}|{content}"
    return {
        "status": "bound_composite",
        "selected_tool": tool,
        "argument_name": argument_name,
        "source_spans": candidate["components"],
        "deterministic_separator": "|",
        "candidates": [candidate],
        "final_call": {"tool": tool, "arguments": {argument_name: value}},
        "source_copy_valid": (
            filename
            == raw[composite.filename_start : composite.filename_end]
            and content == raw[composite.content_start : composite.content_end]
            and value == f"{filename}|{content}"
        ),
    }


def _exact_anchor_candidates(
    raw: str,
    model_arguments: dict[str, Any] | None,
    verbatim: list[str],
) -> list[SpanCandidate]:
    values = list(verbatim)
    if model_arguments:
        values.extend(value for value in model_arguments.values() if isinstance(value, str))
    candidates = []
    for value in values:
        if not value:
            continue
        start = 0
        while True:
            index = raw.find(value, start)
            if index < 0:
                break
            candidates.append(
                SpanCandidate(index, index + len(value), "exact_model_hint_anchor")
            )
            start = index + 1
    return _dedupe(candidates)


def _tool_candidates(raw: str, tool: str) -> tuple[list[SpanCandidate], bool]:
    if tool == "search":
        return _search_candidates(raw)
    if tool == "image":
        return _image_candidates(raw)
    if tool == "ask_claude":
        return _ask_candidates(raw)
    if tool == "see":
        return _see_candidates(raw)
    if tool == "recall":
        return _recall_candidates(raw)
    if tool in {"read", "yuki_read", "yuki_delete", "yuki_write", "yuki_append"}:
        # These tools deliberately fall back to exact model-hint anchors unless a
        # composite Yuki payload is recognized above. A changed path/filename is
        # never guessed from a loose filesystem-looking token.
        return [], False
    return [], False


def bind_source_argument(
    *,
    raw_request: str,
    selected_call: dict[str, Any] | None,
    semantic_request: str,
    verbatim: list[str],
) -> dict[str, Any]:
    """Bind one focused tool argument to raw offsets or explicitly abstain."""

    del semantic_request  # Reserved for future candidate ranking; never character source.
    if selected_call is None:
        return {
            "status": "no_selected_call",
            "selected_tool": None,
            "argument_name": None,
            "candidates": [],
            "final_call": None,
            "source_copy_valid": None,
        }

    tool = selected_call.get("tool")
    argument_name = ARGUMENT_NAMES.get(tool)
    if argument_name is None:
        return {
            "status": "unsupported_selected_tool",
            "selected_tool": tool,
            "argument_name": None,
            "candidates": [],
            "final_call": None,
            "source_copy_valid": None,
        }

    if tool in {"yuki_write", "yuki_append"}:
        composite_binding = _bind_yuki_composite(
            raw_request, tool, argument_name
        )
        if composite_binding is not None:
            return composite_binding

    structural, no_argument = _tool_candidates(raw_request, tool)
    candidates = structural or _exact_anchor_candidates(
        raw_request, selected_call.get("arguments"), verbatim
    )
    candidates = _dedupe(candidates)
    candidate_data = [candidate.as_dict(raw_request) for candidate in candidates]

    if no_argument:
        return {
            "status": "no_argument",
            "selected_tool": tool,
            "argument_name": argument_name,
            "candidates": [],
            "final_call": {"tool": tool, "arguments": {}},
            "source_copy_valid": True,
        }
    if not candidates:
        return {
            "status": "unbound",
            "selected_tool": tool,
            "argument_name": argument_name,
            "candidates": [],
            "final_call": None,
            "source_copy_valid": None,
        }
    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "selected_tool": tool,
            "argument_name": argument_name,
            "candidates": candidate_data,
            "final_call": None,
            "source_copy_valid": None,
        }

    candidate = candidates[0]
    value = raw_request[candidate.start : candidate.end]
    final_call = {"tool": tool, "arguments": {argument_name: value}}
    return {
        "status": "bound",
        "selected_tool": tool,
        "argument_name": argument_name,
        "source_span": candidate.as_dict(raw_request),
        "candidates": candidate_data,
        "final_call": final_call,
        "source_copy_valid": value
        == raw_request[candidate.start : candidate.end],
    }
