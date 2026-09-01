"""Build a fresh non-sacred literal payload-boundary development probe."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .argument_contract import tool_argument_contract
from .benchmark import write_report

ROOT = Path(__file__).parent
REPORT_ROOT = (
    ROOT
    / "reports"
    / "qwen38-flash"
    / "direct-native"
    / "typed-binder"
    / "boundary-probe"
)
PROBE = REPORT_ROOT / "literal-boundary-development-probe-v1.json"
ANNOTATIONS = REPORT_ROOT / "literal-boundary-development-annotations-v1.json"


@dataclass(frozen=True)
class ProbeSpec:
    case_id: str
    tool: str
    request: str
    literal: str
    hint: str
    family: str


def _ask(
    index: int,
    request: str,
    literal: str,
    family: str = "ask_scaffolding",
) -> ProbeSpec:
    return ProbeSpec(
        f"boundary-ask-{index:02d}",
        "ask_claude",
        request,
        literal,
        "semantic external-agent request",
        family,
    )


SPECS = (
    _ask(1, "Ask Claude: why does this fail?", "why does this fail?"),
    _ask(
        2,
        "Ask Claude this exact question: why does this fail?",
        "why does this fail?",
    ),
    _ask(3, "Have Claude answer: why does this fail?", "why does this fail?"),
    _ask(4, "Send this to Claude: why does this fail?", "why does this fail?"),
    _ask(
        5,
        "Claude — check this for me: why does this fail?",
        "check this for me: why does this fail?",
    ),
    _ask(
        6,
        "I need Claude to handle this exact question: why does this fail?",
        "why does this fail?",
    ),
    _ask(7, "Tell Claude: inspect the retry loop.", "inspect the retry loop"),
    _ask(
        8,
        "Delegate to Claude: review the parser boundary.",
        "review the parser boundary",
    ),
    _ask(
        9,
        "Could you ask Claude: does this preserve_case?",
        "does this preserve_case?",
    ),
    _ask(
        10,
        "Please have Claude answer: is Foo.JSON different from foo.json?",
        "is Foo.JSON different from foo.json?",
    ),
    _ask(
        11,
        "Send Claude this question: where did the null come from?",
        "where did the null come from?",
    ),
    _ask(
        12,
        "Pass this to Claude: compare foo_bar() with fooBar().",
        "compare foo_bar() with fooBar()",
    ),
    _ask(
        13,
        "Consult Claude: check KeyError: city without rewriting it.",
        "check KeyError: city without rewriting it",
    ),
    _ask(
        14,
        "Claude: explain why x:y stays literal.",
        "explain why x:y stays literal",
    ),
    _ask(
        15,
        'Claude — why did it output {"tool": null}?',
        'why did it output {"tool": null}?',
    ),
    _ask(
        16,
        "I want Claude to review: A_B-17 vs a_b-17.",
        "A_B-17 vs a_b-17",
    ),
    _ask(17, "Have Claude inspect: /tmp/Foo Bar.txt", "/tmp/Foo Bar.txt"),
    _ask(
        18,
        "Ask Claude to check the race condition.",
        "check the race condition",
        "ask_to_payload",
    ),
    _ask(
        19,
        "Have Claude review the schema adapter.",
        "review the schema adapter",
        "verb_payload",
    ),
    _ask(
        20,
        "Tell Claude to inspect traceback line 42.",
        "inspect traceback line 42",
        "verb_payload",
    ),
    _ask(
        21,
        "Ask Claude: handle this exact question: why does this fail?",
        "handle this exact question: why does this fail?",
        "payload_contains_scaffolding",
    ),
    _ask(
        22,
        "Ask Claude: Ask Claude to retry after five seconds.",
        "Ask Claude to retry after five seconds",
        "payload_contains_scaffolding",
    ),
    _ask(
        23,
        "Send this to Claude: Send this to Claude: preserve both copies.",
        "Send this to Claude: preserve both copies",
        "payload_contains_scaffolding",
    ),
    _ask(
        24,
        "Claude — I need Claude to handle this exact question: why now?",
        "I need Claude to handle this exact question: why now?",
        "payload_contains_scaffolding",
    ),
    _ask(
        25,
        "Have Claude answer: Have Claude answer: nested text.",
        "Have Claude answer: nested text",
        "payload_contains_scaffolding",
    ),
    _ask(
        26,
        'Ask Claude this exact question: "handle this exact question: why?"',
        "handle this exact question: why?",
        "outer_quote_delimiter",
    ),
    _ask(
        27,
        "Send this to Claude: 'Ask Claude: keep this colon.'",
        "Ask Claude: keep this colon.",
        "outer_quote_delimiter",
    ),
    _ask(
        28,
        "Delegate to Claude: `Have Claude answer: yes?`",
        "Have Claude answer: yes?",
        "outer_quote_delimiter",
    ),
    _ask(
        29,
        'Ask Claude: explain the phrase "handle this exact question:" literally.',
        'explain the phrase "handle this exact question:" literally',
        "inner_quote_payload",
    ),
    _ask(
        30,
        'Claude — check this for me: keep the words "check this for me:" intact.',
        'check this for me: keep the words "check this for me:" intact',
        "inner_quote_payload",
    ),
    _ask(
        31,
        "Ask Claude this exact question: why: does this fail?",
        "why: does this fail?",
        "inner_colon_payload",
    ),
    _ask(
        32,
        "Have Claude answer: first: second: third",
        "first: second: third",
        "inner_colon_payload",
    ),
    _ask(
        33,
        'Send this to Claude: JSON {"message":"Ask Claude: hi"}',
        'JSON {"message":"Ask Claude: hi"}',
        "structured_payload",
    ),
    _ask(
        34,
        "I need Claude to handle this exact question: compare A:B with A::B.",
        "compare A:B with A::B",
        "inner_colon_payload",
    ),
    _ask(
        35,
        'Ask Claude: Do not strip "Ask Claude:" from this payload.',
        'Do not strip "Ask Claude:" from this payload',
        "negative_instruction_payload",
    ),
    _ask(
        36,
        'Claude — "Ask Claude: why?"',
        "Ask Claude: why?",
        "outer_quote_delimiter",
    ),
    _ask(
        37,
        "Send this to Claude: `why does foo_bar() fail?`",
        "why does foo_bar() fail?",
        "outer_quote_delimiter",
    ),
    _ask(
        38,
        'Have Claude answer: "why does this fail?" exactly as written.',
        '"why does this fail?" exactly as written',
        "inner_quote_payload",
    ),
    _ask(
        39,
        "Ask Claude: colon at end -> value:",
        "colon at end -> value:",
        "terminal_colon_payload",
    ),
    _ask(
        40,
        "Ask Claude: question mark stays?",
        "question mark stays?",
        "terminal_question_payload",
    ),
    ProbeSpec(
        "boundary-search-01",
        "search",
        "Search for handle this exact question: parser behavior.",
        "handle this exact question: parser behavior",
        "semantic search query",
        "cross_tool_prefix_text",
    ),
    ProbeSpec(
        "boundary-search-02",
        "search",
        'Search the web for "Ask Claude: why?"',
        "Ask Claude: why?",
        "semantic search query",
        "outer_quote_delimiter",
    ),
    ProbeSpec(
        "boundary-search-03",
        "search",
        'Search query: site:example.com "exact phrase"',
        'site:example.com "exact phrase"',
        "semantic search query",
        "inner_quote_payload",
    ),
    ProbeSpec(
        "boundary-search-04",
        "search",
        "Look up why:path handling differs.",
        "why:path handling differs",
        "semantic search query",
        "inner_colon_payload",
    ),
    ProbeSpec(
        "boundary-search-05",
        "search",
        "Find information online about Send this to Claude: conventions.",
        "Send this to Claude: conventions",
        "semantic search query",
        "cross_tool_prefix_text",
    ),
    ProbeSpec(
        "boundary-search-06",
        "search",
        "Search for C++ A::B vs A:B.",
        "C++ A::B vs A:B",
        "semantic search query",
        "inner_colon_payload",
    ),
    ProbeSpec(
        "boundary-image-01",
        "image",
        'Generate an image with this exact prompt: "Claude — neon rain"',
        "Claude — neon rain",
        "semantic image prompt",
        "outer_quote_delimiter",
    ),
    ProbeSpec(
        "boundary-image-02",
        "image",
        "Create a picture using exactly: handle this exact question: a red fox",
        "handle this exact question: a red fox",
        "semantic image prompt",
        "cross_tool_prefix_text",
    ),
    ProbeSpec(
        "boundary-image-03",
        "image",
        'Use the image tool with this exact prompt: text reading "Ask Claude:" in neon',
        'text reading "Ask Claude:" in neon',
        "semantic image prompt",
        "inner_quote_payload",
    ),
    ProbeSpec(
        "boundary-image-04",
        "image",
        "Make artwork from this exact prompt, not a text file: A:B — C::D",
        "A:B — C::D",
        "semantic image prompt",
        "inner_colon_payload",
    ),
    ProbeSpec(
        "boundary-read-01",
        "read",
        "Read exactly /tmp/handle this exact question.txt.",
        "/tmp/handle this exact question.txt",
        "semantic path",
        "path_boundary",
    ),
    ProbeSpec(
        "boundary-read-02",
        "read",
        "Read /Users/me/Send this to Claude/Plan.md from the computer.",
        "/Users/me/Send this to Claude/Plan.md",
        "semantic path",
        "path_boundary",
    ),
    ProbeSpec(
        "boundary-read-03",
        "read",
        "Open the local file ~/Claude — notes.md.",
        "~/Claude — notes.md",
        "semantic path",
        "path_boundary",
    ),
    ProbeSpec(
        "boundary-read-04",
        "read",
        "Read ./HandleThisExactQuestion.JSON.",
        "./HandleThisExactQuestion.JSON",
        "semantic path",
        "path_boundary",
    ),
    ProbeSpec(
        "boundary-fetch-01",
        "fetch",
        "Fetch this exact URL: https://example.com/Ask-Claude:why?x=A:B.",
        "https://example.com/Ask-Claude:why?x=A:B",
        "semantic URL",
        "url_boundary",
    ),
    ProbeSpec(
        "boundary-fetch-02",
        "fetch",
        "Open contents of https://example.org/path?q=handle+this+exact+question.",
        "https://example.org/path?q=handle+this+exact+question",
        "semantic URL",
        "url_boundary",
    ),
    ProbeSpec(
        "boundary-shell-01",
        "shell",
        "Run this allowed shell command: printf 'Ask Claude: %s' test",
        "printf 'Ask Claude: %s' test",
        "semantic shell command",
        "command_boundary",
    ),
    ProbeSpec(
        "boundary-shell-02",
        "shell",
        "Execute echo 'handle this exact question:' in the shell.",
        "echo 'handle this exact question:'",
        "semantic shell command",
        "command_boundary",
    ),
    ProbeSpec(
        "boundary-yuki-01",
        "yuki_write",
        "Write exactly claude.txt|Ask Claude: why? into Yuki's files.",
        "claude.txt|Ask Claude: why?",
        "claude.txt|semantic content",
        "composite_boundary",
    ),
    ProbeSpec(
        "boundary-yuki-02",
        "yuki_append",
        "Append exactly notes.txt|handle this exact question: later to Yuki's file.",
        "notes.txt|handle this exact question: later",
        "notes.txt|semantic content",
        "composite_boundary",
    ),
    ProbeSpec(
        "boundary-yuki-03",
        "yuki_read",
        "Read my Yuki notes file Ask Claude notes.txt.",
        "Ask Claude notes.txt",
        "semantic filename",
        "filename_boundary",
    ),
    ProbeSpec(
        "boundary-yuki-04",
        "yuki_delete",
        "Delete Handle This Exact Question.md from Yuki's folder.",
        "Handle This Exact Question.md",
        "semantic filename",
        "filename_boundary",
    ),
    ProbeSpec(
        "boundary-yuki-05",
        "yuki_write",
        "Create Yuki file task.txt with this exact payload: task.txt|Claude: review A:B",
        "task.txt|Claude: review A:B",
        "task.txt|semantic content",
        "composite_boundary",
    ),
    ProbeSpec(
        "boundary-yuki-06",
        "yuki_append",
        "Add this exact payload to a Yuki file: log.txt|Claude — keep this",
        "log.txt|Claude — keep this",
        "log.txt|semantic content",
        "composite_boundary",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> tuple[dict, dict]:
    if len(SPECS) != 64 or len({spec.case_id for spec in SPECS}) != 64:
        raise RuntimeError("Boundary probe must contain 64 unique cases")
    input_cases = []
    annotations = {}
    for spec in SPECS:
        argument_name = tool_argument_contract(spec.tool)["argument"]
        start = spec.request.find(spec.literal)
        if start < 0 or spec.request.find(spec.literal, start + 1) >= 0:
            raise RuntimeError(
                f"Probe literal must occur exactly once: {spec.case_id}: {spec.literal!r}"
            )
        input_cases.append(
            {
                "case_id": spec.case_id,
                "request": spec.request,
                "selected_call": {
                    "tool": spec.tool,
                    "arguments": {argument_name: spec.hint},
                },
                "family": spec.family,
            }
        )
        annotations[spec.case_id] = {
            "case_id": spec.case_id,
            "tool": spec.tool,
            "argument_name": argument_name,
            "raw_request_sha256": hashlib.sha256(spec.request.encode()).hexdigest(),
            "status": "reference",
            "span": {
                "start": start,
                "end": start + len(spec.literal),
                "text": spec.literal,
            },
        }
    probe = {
        "version": "literal-boundary-development-probe-v1",
        "status": "gold_and_annotations_free",
        "purpose": "Fresh non-sacred deterministic literal payload-boundary development",
        "case_count": 64,
        "sacred_450_source": False,
        "model_outputs_used": False,
        "cases": input_cases,
    }
    annotation_payload = {
        "version": "literal-boundary-development-annotations-v1",
        "status": "frozen_before_binder_probe",
        "probe_version": probe["version"],
        "case_count": 64,
        "binder_access": False,
        "model_outputs_used": False,
        "official_benchmark_gold_used": False,
        "annotations": annotations,
    }
    return probe, annotation_payload


def main() -> None:
    probe, annotations = build()
    write_report(probe, PROBE)
    write_report(annotations, ANNOTATIONS)
    for path in (PROBE, ANNOTATIONS):
        checksum = _sha256(path)
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{checksum}  {path.name}\n",
            encoding="utf-8",
        )
        print(checksum)


if __name__ == "__main__":
    main()
