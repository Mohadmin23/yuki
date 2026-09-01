"""Build independent literal-source annotations for the frozen Flash 168 suites.

This module reads only immutable benchmark requests and expected tool labels. It
does not read model generations, model arguments, benchmark gold arguments, or
the deterministic binder implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .argument_contract import LITERAL_SOURCE, tool_argument_contract
from .benchmark import write_report
from .qwen38_flash_direct import (
    BROAD_CASES,
    BROAD_SHA256,
    TARGETED_CASES,
    TARGETED_SHA256,
)

ROOT = Path(__file__).parent
OUTPUT_ROOT = ROOT / "reports" / "qwen38-flash" / "direct-native" / "typed-binder"
OUTPUT = OUTPUT_ROOT / "flash168-literal-source-annotations-v1.json"

# Human-curated from the immutable raw request text only. Coincidence with old
# official gold is not evidence of provenance; every value is verified below as
# a unique, character-exact raw-request substring.
LITERAL_REFERENCES = {
    "fetch-01": "https://example.com/docs",
    "fetch-02": "https://www.rfc-editor.org/rfc/rfc8259",
    "fetch-03": "http://example.org/page",
    "fetch-04": "https://example.net/api",
    "search-01": "weather APIs",
    "search-02": "the latest MLX documentation",
    "search-03": "local function-calling models",
    "search-04": "JSON schema constrained decoding",
    "image-01": "neon fox under a rainy sign",
    "image-02": "watercolor lighthouse at dawn",
    "image-03": "tiny robot watering sunflowers",
    "image-04": "moonlit anime library",
    "read-01": "/tmp/test.txt",
    "read-02": "~/notes.md",
    "read-03": "/Users/shared/report.json",
    "read-04": "/tmp/yuki_notes.txt",
    "shell-01": "ls -la",
    "shell-02": "pwd",
    "shell-03": "df -h",
    "shell-04": "uname -a",
    "yuki-write-01": "notes.txt|Buy oat milk",
    "yuki-write-02": "poem.md|stars over quiet water",
    "yuki-write-03": "todo.txt|Call Sam tomorrow",
    "yuki-write-04": "draft.txt|chapter one",
    "yuki-read-01": "notes.txt",
    "yuki-read-02": "poem.md",
    "yuki-read-03": "todo.txt",
    "yuki-read-04": "journal.md",
    "yuki-delete-01": "notes.txt",
    "yuki-delete-02": "poem.md",
    "yuki-delete-03": "todo.txt",
    "yuki-delete-04": "journal.md",
    "yuki-append-01": "notes.txt|and buy coffee",
    "yuki-append-02": "log.txt|second entry",
    "yuki-append-03": "poem.md|one more line",
    "yuki-append-04": "todo.txt|Buy tea",
    "ask-claude-01": "inspect this error: TypeError in server.py",
    "ask-claude-02": "Why is the test hanging?",
    "ask-claude-03": "check the schema adapter",
    "ask-claude-04": "Is this traceback caused by asyncio?",
    "target-read-kind-01": "/tmp/test.txt",
    "target-read-kind-02": "~/Documents/notes.md",
    "target-read-kind-03": "/Users/me/Config.JSON",
    "target-read-kind-04": "./README.md",
    "target-read-kind-05": "/tmp/report final.txt",
    "target-read-kind-06": "/var/log/system.log",
    "target-read-kind-07": "/Users/me/My Files/Plan.md",
    "target-read-kind-08": "/private/tmp/yuki-test.json",
    "target-read-kind-09": "notes.txt",
    "target-read-kind-10": "poem.md",
    "target-read-kind-11": "journal.md",
    "target-read-kind-12": "todo.txt",
    "target-read-kind-13": "Shopping List.txt",
    "target-read-kind-14": "README.md",
    "target-read-kind-15": "dream.log",
    "target-read-kind-16": "config.JSON",
    "target-search-01": "latest MLX documentation",
    "target-search-02": "Qwen3.8-27B release notes",
    "target-search-03": "Apple Silicon unified memory benchmarks",
    "target-search-04": "weather APIs with free tiers",
    "target-search-05": "current Python 3.14 documentation",
    "target-search-06": "Hammer 2.1 function calling results",
    "target-search-07": "case-sensitive APFS behavior",
    "target-search-08": "mlx-lm JSON stopping",
    "target-search-09": "site:huggingface.co xLAM-2-1b-fc-r",
    "target-search-10": "C++ std::filesystem path case sensitivity",
    "target-search-11": "New York weather API latency",
    "target-search-12": "Unicode NFKC normalization examples",
    "target-search-13": "read vs yuki_read routing",
    "target-search-14": "macOS M1 memory pressure command",
    "target-search-15": "the latest Qwen release",
    "target-search-16": "Salesforce APIGen-MT paper",
    "target-path-01": "/Users/me/foo.txt",
    "target-path-02": "/Users/me/Foo.txt",
    "target-path-03": "~/notes.md",
    "target-path-04": "~/Notes.md",
    "target-path-05": "/tmp/report final.txt",
    "target-path-06": "./config/Prod.JSON",
    "target-path-07": "../shared/README.md",
    "target-path-08": "/Users/mohamedlaminemennane/Projects/Yuki/data.json",
    "target-path-09": "/private/tmp/a-b_c.01.log",
    "target-path-10": "/Volumes/madisk/Models/xLAM-2/config.json",
    "target-path-11": "/Users/me/Café/menu.txt",
    "target-path-12": "/tmp/.hidden-file",
    "target-path-13": "/Users/me/Folder With Spaces/File Name.md",
    "target-path-14": "./src/tools/__init__.py",
    "target-path-15": "/var/tmp/CASE_sensitive.Path",
    "target-path-16": "~/Documents/2026-08-18 notes.txt",
    "target-claude-01": "check this error",
    "target-claude-02": "inspect this error: TypeError in server.py",
    "target-claude-03": "review the schema adapter",
    "target-claude-04": "explain why this test is flaky",
    "target-claude-05": "check whether this shell validator is safe",
    "target-claude-06": "inspect traceback line 42",
    "target-claude-07": "review my fix for the parser",
    "target-claude-08": "identify the race condition",
    "target-claude-09": "verify this JSON schema",
    "target-claude-10": "assess the memory leak",
    "target-claude-11": "examine the MLX generation issue",
    "target-claude-12": "check the function name mapping",
    "target-claude-13": "review this exception: KeyError city",
    "target-claude-14": "inspect the benchmark methodology",
    "target-claude-15": "tell me whether this regex is bypassable",
    "target-claude-16": "analyze why yuki_read was selected",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source_cases() -> list[dict[str, Any]]:
    checks = {
        BROAD_CASES: BROAD_SHA256,
        TARGETED_CASES: TARGETED_SHA256,
    }
    for path, expected in checks.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {path}: {actual} != {expected}")
    return [
        case
        for path in checks
        for case in json.loads(path.read_text(encoding="utf-8"))
    ]


def build_annotations() -> dict[str, Any]:
    cases = _load_source_cases()
    literal_cases = {
        case["case_id"]: {
            "request": case["request"],
            "tool": case["expected_tool"],
        }
        for case in cases
        if tool_argument_contract(case["expected_tool"])["mode"] == LITERAL_SOURCE
    }
    if set(literal_cases) != set(LITERAL_REFERENCES):
        raise RuntimeError(
            "Literal annotation coverage differs from the typed contract: "
            f"missing={sorted(set(literal_cases) - set(LITERAL_REFERENCES))}, "
            f"extra={sorted(set(LITERAL_REFERENCES) - set(literal_cases))}"
        )

    annotations = {}
    for case_id, case in literal_cases.items():
        raw = case["request"]
        value = LITERAL_REFERENCES[case_id]
        start = raw.find(value)
        if start < 0 or raw.find(value, start + 1) >= 0:
            raise RuntimeError(
                f"Literal reference must occur exactly once in {case_id}: {value!r}"
            )
        tool = case["tool"]
        contract = tool_argument_contract(tool)
        span = {"start": start, "end": start + len(value), "text": value}
        annotation = {
            "case_id": case_id,
            "tool": tool,
            "argument_name": contract["argument"],
            "raw_request_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "status": "reference",
            "spans": [span],
            "literal_value": value,
            "components": [],
        }
        if contract.get("components"):
            separator = contract["deterministic_separator"]
            filename, content = value.split(separator, 1)
            filename_start = start
            content_start = start + len(filename) + len(separator)
            annotation["components"] = [
                {
                    "name": "filename",
                    "start": filename_start,
                    "end": filename_start + len(filename),
                    "text": filename,
                },
                {
                    "name": "content",
                    "start": content_start,
                    "end": content_start + len(content),
                    "text": content,
                },
            ]
        annotations[case_id] = annotation

    return {
        "version": "flash168-literal-source-annotations-v1",
        "status": "frozen_before_typed_binder_replay",
        "argument_contract_version": "yuki-argument-contract-v1",
        "source_suites": {
            BROAD_CASES.name: BROAD_SHA256,
            TARGETED_CASES.name: TARGETED_SHA256,
        },
        "annotation_method": (
            "Human-curated character spans from immutable raw requests; old "
            "official expected_arguments and Flash outputs were not inputs."
        ),
        "binder_access": False,
        "model_outputs_accessed": False,
        "official_gold_arguments_used_as_literal_references": False,
        "summary": {
            "total_source_cases": len(cases),
            "literal_source_cases": len(annotations),
            "reference": len(annotations),
            "ambiguous": 0,
        },
        "annotations": annotations,
    }


def main() -> None:
    payload = build_annotations()
    write_report(payload, OUTPUT)
    checksum = _sha256(OUTPUT)
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{checksum}  {OUTPUT.name}\n",
        encoding="utf-8",
    )
    print(checksum)


if __name__ == "__main__":
    main()
