"""Build the fresh 200-development/100-holdout literal-boundary corpus.

The two splits use disjoint outer-envelope templates. Both are materialized and
checksummed before binder v3 exists. Development and holdout runners read only
their own split, and holdout scoring is locked until a v3 freeze manifest exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .argument_contract import LITERAL_SOURCE, tool_argument_contract
from .benchmark import write_report

ROOT = Path(__file__).parent
REPORT_ROOT = (
    ROOT
    / "reports"
    / "qwen38-flash"
    / "direct-native"
    / "typed-binder"
    / "boundary-corpus-v1"
)
DEV_INPUT = REPORT_ROOT / "literal-boundary-dev-200-v1.json"
DEV_ANNOTATIONS = REPORT_ROOT / "literal-boundary-dev-annotations-200-v1.json"
HOLDOUT_INPUT = REPORT_ROOT / "literal-boundary-holdout-100-v1.json"
HOLDOUT_ANNOTATIONS = (
    REPORT_ROOT / "literal-boundary-holdout-annotations-100-v1.json"
)
CORPUS_MANIFEST = REPORT_ROOT / "LITERAL-BOUNDARY-CORPUS-v1-FROZEN-MANIFEST.json"
CHECKSUMS = REPORT_ROOT / "LITERAL-BOUNDARY-CORPUS-SHA256SUMS-v1.txt"


@dataclass(frozen=True)
class Envelope:
    before: str
    after: str
    family: str


@dataclass(frozen=True)
class Spec:
    split: str
    case_id: str
    tool: str
    request: str
    literal: str
    family: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n",
        encoding="utf-8",
    )
    return checksum


def _envelope(before: str, after: str = "", family: str = "colon") -> Envelope:
    return Envelope(before, after, family)


ASK_DEV = (
    _envelope("Ask Claude: "),
    _envelope("Send this question to Claude: ", family="transport_noun"),
    _envelope("Have Claude answer: ", family="action_colon"),
    _envelope("Claude — ", family="dash"),
    _envelope("Could you ask Claude this question: ", family="polite"),
    _envelope("Delegate the following message to Claude: ", family="transport_noun"),
    _envelope("Tell Claude to ", ".", "direct_infinitive"),
    _envelope("I need Claude to review this request: ", family="transport_noun"),
    _envelope('Ask Claude this exact question: "', '"', "outer_quotes"),
    _envelope("Pass this along to Claude: `", "`", "outer_backticks"),
    _envelope("For Claude, the question is: ", family="topic_fronting"),
    _envelope("Please have Claude handle this message: ", family="transport_noun"),
    _envelope("Claude, please answer this: ", family="comma_clause"),
)

ASK_HOLDOUT = (
    _envelope("Route this request to Claude: ", family="transport_noun"),
    _envelope("Would you have Claude inspect this question: ", family="polite"),
    _envelope("Claude gets this message — ", family="dash"),
    _envelope("Hand the following task to Claude: ", family="transport_noun"),
    _envelope("Let Claude consider: ", family="action_colon"),
    _envelope("Please ask Claude to ", "!", "direct_infinitive"),
    _envelope("The exact question for Claude is: '", "'", "outer_quotes"),
)

ASK_PAYLOADS = (
    "why does parse_v3() return null?",
    "question: should A::B remain unchanged?",
    "message: keep Foo.JSON and foo.json distinct",
    "handle this exact question: which retry owns error_code-17?",
    'explain the words "Ask Claude:" without removing them',
)

SEARCH_DEV = (
    _envelope("Search for "),
    _envelope("Search the web for: "),
    _envelope("Look this up online — ", family="dash"),
    _envelope("Use web search with this query: ", family="transport_noun"),
    _envelope('Find results for "', '"', "outer_quotes"),
    _envelope("Query the web for `", "`", "outer_backticks"),
    _envelope("Online search, please: ", family="comma_clause"),
    _envelope("I need information about: ", family="implicit_intent"),
    _envelope("Search this exact phrase: ", family="exact_marker"),
    _envelope("Web lookup → ", family="arrow"),
)

SEARCH_HOLDOUT = (
    _envelope("Find this on the internet: "),
    _envelope("Run an online query for "),
    _envelope("The web-search text is: ", family="transport_noun"),
    _envelope('Look online for `', '`', "outer_backticks"),
    _envelope("Search request — ", family="dash"),
)

SEARCH_PAYLOADS = (
    'question: "zero copy" parser notes',
    "search: C++ A::B vs A:B",
    "handle this message: RTX_4070 used prices",
)

IMAGE_DEV = (
    _envelope("Generate an image of "),
    _envelope("Create artwork from this prompt: "),
    _envelope("Image prompt — ", family="dash"),
    _envelope('Render exactly "', '"', "outer_quotes"),
    _envelope("Draw this scene: `", "`", "outer_backticks"),
)

IMAGE_HOLDOUT = (
    _envelope("Make a new picture showing "),
    _envelope("Use this description for the generated image: "),
    _envelope("New artwork → ", family="arrow"),
    _envelope('Illustrate "', '"', "outer_quotes"),
    _envelope("The image description is: ", family="transport_noun"),
)

IMAGE_PAYLOADS = (
    'a neon sign reading "question: stay literal"',
    "message in a bottle under an em dash — moon",
    "handle this exact prompt: fox_v2 beside A::B",
    "search lights over a rain-soaked terminal",
    "Claude saying: keep_this_case",
)

READ_DEV = (
    _envelope("Read the OS file at ", "."),
    _envelope("Open this local path: "),
    _envelope("Filesystem read — ", family="dash"),
    _envelope('Read exactly "', '"', "outer_quotes"),
    _envelope("Show me the disk file `", "`", "outer_backticks"),
    _envelope("From the computer, open: ", family="fronted_scope"),
    _envelope("Load this filepath: "),
    _envelope("OS path → ", family="arrow"),
)

READ_HOLDOUT = (
    _envelope("Inspect the local file ", "."),
    _envelope("Disk file to read: "),
    _envelope('Open the filesystem path "', '"', "outer_quotes"),
)

READ_PAYLOADS = (
    "/tmp/question: notes/Build Log.md",
    "~/Documents/Claude — message.txt",
    "/Users/me/search this/file_name.JSON",
)

SHELL_DEV = (
    _envelope("Run this exact shell command: "),
    _envelope("Execute in the terminal — ", family="dash"),
    _envelope("Shell command: `", "`", "outer_backticks"),
    _envelope("Use the terminal to run "),
    _envelope('Command to execute: "', '"', "outer_quotes"),
    _envelope("Terminal → ", family="arrow"),
    _envelope("In the shell, run: ", family="fronted_scope"),
)

SHELL_HOLDOUT = (
    _envelope("Please execute this command: "),
    _envelope("Run via shell — ", family="dash"),
    _envelope("The terminal command is: `", "`", "outer_backticks"),
)

SHELL_PAYLOADS = (
    "printf 'question: %s' value",
    "ls -la '/tmp/Claude — files' | head -n 3",
)

FETCH_DEV = (
    _envelope("Fetch this URL: "),
    _envelope("Open the exact web address — ", family="dash"),
    _envelope("Retrieve `", "`", "outer_backticks"),
    _envelope("HTTP request for: "),
    _envelope('Get the URL "', '"', "outer_quotes"),
    _envelope("Download from ", "."),
    _envelope("Exact URL → ", family="arrow"),
)

FETCH_HOLDOUT = (
    _envelope("Retrieve this web resource: "),
    _envelope("URL to fetch — ", family="dash"),
    _envelope('Request "', '"', "outer_quotes"),
)

FETCH_PAYLOADS = (
    "https://example.com/question:keep?q=A::B&case=Foo_JSON",
    "https://docs.example.net/message/handle-this#Ask-Claude",
)

COMPOSITE_DEV = (
    _envelope("Write this exact Yuki file payload: "),
    _envelope("Replace a Yuki note using: "),
    _envelope("Append this exact Yuki payload — ", family="dash"),
    _envelope("Add to a Yuki file with `", "`", "outer_backticks"),
    _envelope('Yuki file contents: "', '"', "outer_quotes"),
    _envelope("Store in Yuki's own files: "),
    _envelope("Preserve the old note and append: "),
    _envelope("Create or replace this note → ", family="arrow"),
    _envelope("For Yuki storage, write: ", family="fronted_scope"),
    _envelope("Yuki append payload: "),
)

COMPOSITE_HOLDOUT = (
    _envelope("Put this exact payload in Yuki storage: "),
    _envelope("Extend the existing Yuki note with: "),
    _envelope("Yuki write request — ", family="dash"),
    _envelope('Append exactly "', '"', "outer_quotes"),
    _envelope("Internal note payload → ", family="arrow"),
)

COMPOSITE_PAYLOADS = (
    "question notes.md|message: preserve A::B and Foo.JSON",
    "Claude — log.txt|handle this exact content: x|y stays split once",
)

YUKI_FILE_DEV = (
    _envelope("Read this file from Yuki's own storage: "),
    _envelope("Open Yuki's internal note `", "`", "outer_backticks"),
    _envelope("Yuki-file read — ", family="dash"),
    _envelope('Delete this Yuki filename: "', '"', "outer_quotes"),
    _envelope("Remove from Yuki storage: "),
    _envelope("The internal filename is: ", family="transport_noun"),
    _envelope("From Yuki's notes, read: ", family="fronted_scope"),
    _envelope("Yuki delete → ", family="arrow"),
)

YUKI_FILE_HOLDOUT = (
    _envelope("Retrieve Yuki's saved file: "),
    _envelope("Erase this internal Yuki note — ", family="dash"),
    _envelope('Yuki filename: `', '`', "outer_backticks"),
)

YUKI_FILE_PAYLOADS = (
    "question: archive.md",
    "Claude — private notes.JSON",
    "search message A::B.txt",
)


def _cross(
    *,
    split: str,
    tool: str,
    envelopes: tuple[Envelope, ...],
    payloads: tuple[str, ...],
    count: int,
    id_prefix: str,
    alternating_tool: str | None = None,
) -> list[Spec]:
    pairs = [(envelope, payload) for payload in payloads for envelope in envelopes]
    if len(pairs) < count:
        raise RuntimeError(f"Insufficient unique combinations for {id_prefix}")
    specs = []
    for index, (envelope, payload) in enumerate(pairs[:count], start=1):
        selected_tool = (
            alternating_tool
            if alternating_tool is not None and index % 2 == 0
            else tool
        )
        specs.append(
            Spec(
                split=split,
                case_id=f"boundary-{split}-{id_prefix}-{index:03d}",
                tool=selected_tool,
                request=f"{envelope.before}{payload}{envelope.after}",
                literal=payload,
                family=f"{id_prefix}:{envelope.family}",
            )
        )
    return specs


def _build_specs() -> tuple[list[Spec], list[Spec]]:
    dev = [
        *_cross(split="dev", tool="ask_claude", envelopes=ASK_DEV, payloads=ASK_PAYLOADS, count=65, id_prefix="ask"),
        *_cross(split="dev", tool="search", envelopes=SEARCH_DEV, payloads=SEARCH_PAYLOADS, count=30, id_prefix="search"),
        *_cross(split="dev", tool="image", envelopes=IMAGE_DEV, payloads=IMAGE_PAYLOADS, count=25, id_prefix="image"),
        *_cross(split="dev", tool="read", envelopes=READ_DEV, payloads=READ_PAYLOADS, count=16, id_prefix="read"),
        *_cross(split="dev", tool="shell", envelopes=SHELL_DEV, payloads=SHELL_PAYLOADS, count=14, id_prefix="shell"),
        *_cross(split="dev", tool="fetch", envelopes=FETCH_DEV, payloads=FETCH_PAYLOADS, count=14, id_prefix="fetch"),
        *_cross(split="dev", tool="yuki_write", alternating_tool="yuki_append", envelopes=COMPOSITE_DEV, payloads=COMPOSITE_PAYLOADS, count=20, id_prefix="composite"),
        *_cross(split="dev", tool="yuki_read", alternating_tool="yuki_delete", envelopes=YUKI_FILE_DEV, payloads=YUKI_FILE_PAYLOADS, count=16, id_prefix="yuki-file"),
    ]
    holdout = [
        *_cross(split="holdout", tool="ask_claude", envelopes=ASK_HOLDOUT, payloads=ASK_PAYLOADS, count=35, id_prefix="ask"),
        *_cross(split="holdout", tool="search", envelopes=SEARCH_HOLDOUT, payloads=SEARCH_PAYLOADS, count=15, id_prefix="search"),
        *_cross(split="holdout", tool="image", envelopes=IMAGE_HOLDOUT, payloads=IMAGE_PAYLOADS, count=10, id_prefix="image"),
        *_cross(split="holdout", tool="read", envelopes=READ_HOLDOUT, payloads=READ_PAYLOADS, count=9, id_prefix="read"),
        *_cross(split="holdout", tool="shell", envelopes=SHELL_HOLDOUT, payloads=SHELL_PAYLOADS, count=6, id_prefix="shell"),
        *_cross(split="holdout", tool="fetch", envelopes=FETCH_HOLDOUT, payloads=FETCH_PAYLOADS, count=6, id_prefix="fetch"),
        *_cross(split="holdout", tool="yuki_write", alternating_tool="yuki_append", envelopes=COMPOSITE_HOLDOUT, payloads=COMPOSITE_PAYLOADS, count=10, id_prefix="composite"),
        *_cross(split="holdout", tool="yuki_read", alternating_tool="yuki_delete", envelopes=YUKI_FILE_HOLDOUT, payloads=YUKI_FILE_PAYLOADS, count=9, id_prefix="yuki-file"),
    ]
    return dev, holdout


def _payload(specs: list[Spec], split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = []
    annotations = {}
    for spec in specs:
        contract = tool_argument_contract(spec.tool)
        if contract["mode"] != LITERAL_SOURCE:
            raise RuntimeError(f"Non-literal tool in corpus: {spec.tool}")
        start = spec.request.find(spec.literal)
        if start < 0 or spec.request.find(spec.literal, start + 1) >= 0:
            raise RuntimeError(f"Literal span is not unique: {spec.case_id}")
        end = start + len(spec.literal)
        field = contract["argument"]
        cases.append(
            {
                "case_id": spec.case_id,
                "family": spec.family,
                "request": spec.request,
                "request_sha256": hashlib.sha256(spec.request.encode()).hexdigest(),
                "selected_call": {
                    "tool": spec.tool,
                    "arguments": {field: "semantic model hint intentionally omitted"},
                },
                "execution": {
                    "capability": False,
                    "executed": False,
                    "status": "not_available",
                },
            }
        )
        annotations[spec.case_id] = {
            "case_id": spec.case_id,
            "tool": spec.tool,
            "argument_name": field,
            "request_sha256": hashlib.sha256(spec.request.encode()).hexdigest(),
            "span": {"start": start, "end": end, "text": spec.literal},
        }
    input_payload = {
        "version": f"literal-boundary-{split}-input-v1",
        "status": "gold_and_annotations_free",
        "split": split,
        "case_count": len(cases),
        "configuration": {
            "models_called": False,
            "official_benchmark_cases_used": False,
            "sacred_450_used": False,
            "annotations_available_to_binder": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "cases": cases,
    }
    annotation_payload = {
        "version": f"literal-boundary-{split}-annotations-v1",
        "status": "frozen_before_binder_v3",
        "split": split,
        "case_count": len(annotations),
        "binder_access": False,
        "annotations": annotations,
    }
    return input_payload, annotation_payload


def _counts(specs: list[Spec]) -> dict[str, int]:
    tools = sorted({spec.tool for spec in specs})
    return {tool: sum(spec.tool == tool for spec in specs) for tool in tools}


def build() -> None:
    v3_source = ROOT / "typed_literal_binder_v3.py"
    if v3_source.exists():
        raise RuntimeError("Corpus must be frozen before binder v3 exists")
    dev, holdout = _build_specs()
    if len(dev) != 200 or len(holdout) != 100:
        raise RuntimeError(f"Unexpected split sizes: {len(dev)}, {len(holdout)}")
    all_specs = [*dev, *holdout]
    if len({spec.case_id for spec in all_specs}) != 300:
        raise RuntimeError("Boundary corpus case IDs are not unique")
    if len({spec.request for spec in all_specs}) != 300:
        raise RuntimeError("Boundary corpus requests are not unique")
    dev_input, dev_annotations = _payload(dev, "dev")
    holdout_input, holdout_annotations = _payload(holdout, "holdout")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_report(dev_input, DEV_INPUT)
    write_report(dev_annotations, DEV_ANNOTATIONS)
    write_report(holdout_input, HOLDOUT_INPUT)
    write_report(holdout_annotations, HOLDOUT_ANNOTATIONS)
    hashes = {
        path.name: _write_checksum(path)
        for path in (
            DEV_INPUT,
            DEV_ANNOTATIONS,
            HOLDOUT_INPUT,
            HOLDOUT_ANNOTATIONS,
        )
    }
    manifest = {
        "version": "literal-boundary-corpus-v1-frozen-manifest",
        "status": "frozen_before_binder_v3_source_existed",
        "case_count": 300,
        "development": {"cases": 200, "by_tool": _counts(dev)},
        "holdout": {
            "cases": 100,
            "by_tool": _counts(holdout),
            "unavailable_during_binder_development": True,
            "score_locked_until_binder_v3_freeze": True,
        },
        "template_policy": {
            "outer_envelope_templates_disjoint_between_splits": True,
            "payload_vocabulary_shared_to_isolate_boundary_generalization": True,
            "official_benchmark_requests_used": False,
        },
        "artifacts": hashes,
        "safety": {
            "models_called": False,
            "sacred_450_used": False,
            "yuki_tools_executed": False,
        },
    }
    write_report(manifest, CORPUS_MANIFEST)
    manifest_sha = _write_checksum(CORPUS_MANIFEST)
    CHECKSUMS.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in (
            DEV_INPUT,
            DEV_ANNOTATIONS,
            HOLDOUT_INPUT,
            HOLDOUT_ANNOTATIONS,
            CORPUS_MANIFEST,
        )),
        encoding="utf-8",
    )
    _write_checksum(CHECKSUMS)
    print(manifest_sha)


def main() -> None:
    build()


if __name__ == "__main__":
    main()
