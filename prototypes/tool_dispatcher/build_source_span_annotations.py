"""Build frozen literal-source references independently of the binder."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import write_report

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "deterministic-binding"
POLICY = REPORT_ROOT / "SOURCE-SPAN-POLICY-v1.md"
OUTPUT = REPORT_ROOT / "literal-source-annotations-targeted100-recall25-v1.json"

PAYLOAD_BASELINE = (
    ROOT / "reports" / "focused-repair" / "hammer15-verbatim-old-dispatch-v1.json"
)
SEE_RECALL_BASELINE = (
    ROOT
    / "reports"
    / "focused-repair"
    / "hammer15-see-recall-schema-repair-v1.json"
)

FROZEN_SHA256 = {
    POLICY: "3f7f935fa0a302bf14012fd4cb5f9f1c5a6ae0d5e72c6e8957638873c339f1dc",
    PAYLOAD_BASELINE: "24cee3ce9ae8f0c85d53174fbc8f13b6777281cf080e9323b6564a37ad929f38",
    SEE_RECALL_BASELINE: "513df715b51102b4d1ce9aee408f935e6ae78d4e2c42aab9311cf3bf553a0c90",
}


@dataclass(frozen=True)
class SpanSpec:
    status: str
    texts: tuple[str, ...] = ()
    reason: str | None = None


def literal(text: str) -> SpanSpec:
    return SpanSpec("reference", (text,))


def ambiguous(*texts: str, reason: str) -> SpanSpec:
    return SpanSpec("ambiguous", texts, reason)


NO_ARGUMENT = SpanSpec("no_argument", reason="The request asks for the whole scene.")

SEARCH = (
    "the latest MLX documentation",
    "Apple Silicon unified memory",
    "current Python 3.14 release notes",
    "Hammer 2.1 function calling results",
    "best local speech synthesis models",
    "recent articles about JSON schema decoding",
    "M1 memory pressure benchmarks",
    "how APFS handles case-sensitive paths",
    "open-source tool-calling datasets",
    "the official Rust async book",
    "local LLM routing research",
    "recent notes on MLX-LM streaming",
    "compact function-calling models",
    "Metal GPU utilization",
    "Python pathlib expanduser",
    "deterministic JSON generation locally",
    "macOS unified memory swap behavior",
    "Tokyo weather APIs",
    "a website explaining URL fragments",
    "camera calibration guides",
    "Claude tool-use documentation",
    "C++ std::filesystem path case-sensitivity",
    'site:huggingface.co "Hammer2.1-3b" function calling',
    'foo_bar() "KeyError: x-y"',
    "??? macOS swap usage M1 -- 2026",
)

IMAGE = (
    "neon fox beneath a rainy sign",
    "a watercolor lighthouse at dawn",
    "a tiny robot watering sunflowers",
    "moonlit anime library",
    "a glass city floating above clouds",
    "a cozy pixel-art kitchen in winter",
    "an astronaut playing cello on Mars",
    "a charcoal sketch of an old cedar tree",
    "a cinematic desert train at sunset",
    "a blue dragon curled around a lighthouse",
    "bioluminescent jellyfish over a dark reef",
    "a sleepy cat running a ramen shop",
    "paper-cut mountains under a gold moon",
    "retro-futurist bicycle workshop",
    "a storm moving over lavender fields",
    "friendly moss-covered forest giant",
    "an isometric library inside a teacup",
    "a camera-shaped sculpture",
    "Tokyo weather icons",
    "poster artwork containing the words notes.md",
    "Claude as a brass automaton",
    "RED fox, blue_sky, 35mm—high contrast!",
    "'Home, 2:17 a.m.' in rainy cyberpunk style",
    "café façade; crème-and-teal palette; 4:3 composition",
    "almost-empty white room + one RED chair",
)

ASK_CLAUDE = (
    "why does this asyncio task never finish?",
    "can you review my parser design?",
    "inspect this error: KeyError in build_prompt",
    "explain the tradeoffs of a two-stage router",
    "review this idea: cache schemas by domain",
    "whether this test has a race condition",
    "propose edge cases for JSON early stopping",
    "analyze why the model emitted two calls",
    "critique the benchmark methodology",
    "check whether my regex is too broad",
    "what could make MLX latency measurements noisy?",
    "is the adapter stateless?",
    "find the flaw in my confidence-interval code",
    "whether these schemas overlap too much",
    "model chooses image when I mean see",
    "how should paired errors be grouped?",
    "should p95 include selector time?",
    "about Tokyo weather routing; do not use weather directly",
    "how should queries be normalized?",
    'is `cat "/tmp/a b"` safe? Do not execute it.',
    "discuss notes.md versus Yuki files; don't read either file",
    "check why foo_bar() crashes",
    "Does `A_B-17` differ from `a_b-17` on APFS?",
    "Review path ~/My Notes/Foo.JSON — keep case + spaces.",
    'Why did it output {"tool":null}?',
)

SEE = (
    literal("my keys"),
    NO_ARGUMENT,
    literal("my coffee mug"),
    literal("the door"),
    literal("my red backpack"),
    literal("the whiteboard"),
    literal("my phone"),
    NO_ARGUMENT,
    literal("my charging cable"),
    ambiguous(
        "the window",
        reason="The same target phrase occurs twice at equally plausible offsets.",
    ),
    literal("my wallet"),
    literal("my earbuds"),
    NO_ARGUMENT,
    literal("the blue notebook"),
    literal("my glasses"),
    literal("the desk lamp"),
    NO_ARGUMENT,
    literal("the printed image"),
    literal("the thermostat display"),
    literal("the terminal window"),
    ambiguous(
        "the notes file",
        "the notes file lying on my desk",
        reason="Both the head noun phrase and its identifying modifier are plausible targets.",
    ),
    literal("USB-C adapter"),
    literal("AirPods Pro"),
    literal("M.2 enclosure"),
    literal("red-ish cable"),
)

RECALL = (
    "the server crash",
    "my sister's wedding",
    "the garden project",
    "a laptop upgrade",
    "the Rust compiler error",
    "the kitchen renovation",
    "my travel plans",
    "our API timeout debugging",
    "Nori's vet visit",
    "the voice latency investigation",
    "the blue bicycle",
    "the database migration",
    "Metal ran out of RAM",
    "jasmine tea",
    "the parser refactor",
    "home-office lighting",
    "my holiday packing list",
    "weather API planning",
    "URL fetch failures",
    "notes.md",
    "Claude bridge debugging",
    "Project X-17",
    "foo_bar parser crash",
    "Mum's 60th birthday",
    "the red-ish cable",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_occurrences(raw: str, text: str) -> list[dict[str, Any]]:
    spans = []
    start = 0
    while True:
        index = raw.find(text, start)
        if index < 0:
            break
        spans.append({"start": index, "end": index + len(text), "text": text})
        start = index + 1
    return spans


def _case_annotation(case: dict[str, Any], spec: SpanSpec) -> dict[str, Any]:
    raw = case["request"]
    spans = [span for text in spec.texts for span in _all_occurrences(raw, text)]
    if spec.status == "reference" and len(spans) != 1:
        raise RuntimeError(
            f"Reference for {case['case_id']} must resolve once, got {spans}"
        )
    if spec.status == "ambiguous" and len(spans) < 2:
        raise RuntimeError(f"Ambiguity for {case['case_id']} needs multiple spans")
    if spec.status == "no_argument" and spans:
        raise RuntimeError(f"No-argument annotation has spans: {case['case_id']}")
    return {
        "case_id": case["case_id"],
        "tool": case["expected_tool"],
        "argument_name": next(iter(case["expected_arguments"]), None),
        "raw_request_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "status": spec.status,
        "spans": spans,
        "reason": spec.reason,
    }


def _indexed_specs(tool: str, values: tuple[str, ...]) -> dict[str, SpanSpec]:
    if len(values) != 25:
        raise RuntimeError(f"Expected 25 {tool} annotations, got {len(values)}")
    return {
        f"p2-{tool}-{index:02d}": literal(value)
        for index, value in enumerate(values, 1)
    }


def build_annotations() -> dict[str, Any]:
    observed = {path: _sha256(path) for path in FROZEN_SHA256}
    mismatches = {
        str(path): {"expected": FROZEN_SHA256[path], "actual": actual}
        for path, actual in observed.items()
        if actual != FROZEN_SHA256[path]
    }
    if mismatches:
        raise RuntimeError(f"Frozen annotation inputs changed: {mismatches}")

    payload = _load(PAYLOAD_BASELINE)
    memory_media = _load(SEE_RECALL_BASELINE)
    cases = {
        case["case_id"]: case
        for report in (payload, memory_media)
        for case in report["case_results"]
    }
    specs = {
        **_indexed_specs("search", SEARCH),
        **_indexed_specs("image", IMAGE),
        **_indexed_specs("ask_claude", ASK_CLAUDE),
        **{f"p2-see-{index:02d}": spec for index, spec in enumerate(SEE, 1)},
        **_indexed_specs("recall", RECALL),
    }
    if len(specs) != 125:
        raise RuntimeError(f"Expected 125 annotations, got {len(specs)}")
    missing = sorted(set(specs) - cases.keys())
    if missing:
        raise RuntimeError(f"Missing frozen cases: {missing}")

    primary_ids = [
        f"p2-{tool}-{index:02d}"
        for tool in ("search", "image", "ask_claude", "see")
        for index in range(1, 26)
    ]
    recall_ids = [f"p2-recall-{index:02d}" for index in range(1, 26)]
    annotations = {
        case_id: _case_annotation(cases[case_id], specs[case_id])
        for case_id in primary_ids + recall_ids
    }
    return {
        "version": "literal-source-annotations-targeted100-recall25-v1",
        "status": "frozen_before_binder_implementation",
        "policy_sha256": observed[POLICY],
        "source_reports": {
            str(PAYLOAD_BASELINE): observed[PAYLOAD_BASELINE],
            str(SEE_RECALL_BASELINE): observed[SEE_RECALL_BASELINE],
        },
        "binder_access": False,
        "official_gold_included": False,
        "splits": {
            "primary_targeted_100": primary_ids,
            "recall_diagnostic_25": recall_ids,
        },
        "summary": {
            "total": len(annotations),
            "reference": sum(
                item["status"] == "reference" for item in annotations.values()
            ),
            "ambiguous": sum(
                item["status"] == "ambiguous" for item in annotations.values()
            ),
            "no_argument": sum(
                item["status"] == "no_argument" for item in annotations.values()
            ),
        },
        "annotations": annotations,
    }


def main() -> None:
    payload = build_annotations()
    write_report(payload, OUTPUT)
    checksum = _sha256(OUTPUT)
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{checksum}  {OUTPUT.name}\n", encoding="utf-8"
    )
    print(checksum)


if __name__ == "__main__":
    main()
