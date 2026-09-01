"""Prepare a gold-free input file from frozen dispatcher outputs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .benchmark import write_report

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "deterministic-binding"
PAYLOAD_BASELINE = (
    ROOT / "reports" / "focused-repair" / "hammer15-verbatim-old-dispatch-v1.json"
)
SEE_RECALL_BASELINE = (
    ROOT
    / "reports"
    / "focused-repair"
    / "hammer15-see-recall-schema-repair-v1.json"
)
OUTPUT = REPORT_ROOT / "deterministic-binding-input-targeted100-recall25-v1.json"

FROZEN_SHA256 = {
    PAYLOAD_BASELINE: "24cee3ce9ae8f0c85d53174fbc8f13b6777281cf080e9323b6564a37ad929f38",
    SEE_RECALL_BASELINE: "513df715b51102b4d1ce9aee408f935e6ae78d4e2c42aab9311cf3bf553a0c90",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_input(case: dict[str, Any], split: str) -> dict[str, Any]:
    dispatch = case["dispatcher"]
    return {
        "case_id": case["case_id"],
        "split": split,
        "raw_request": case["request"],
        "raw_request_sha256": hashlib.sha256(case["request"].encode()).hexdigest(),
        "semantic_delegation": deepcopy(case["delegation"]),
        "selected_call": deepcopy(dispatch["canonical_call"]),
        "hammer_record": {
            "model_id": dispatch["generation"]["model_id"],
            "raw_generation": dispatch["generation"]["raw_text"],
            "parse": deepcopy(dispatch["parse"]),
            "validation": deepcopy(dispatch["validation"]),
            "offered_tools": deepcopy(dispatch["offered_tools"]),
            "native_dialect": dispatch["native_dialect"],
            "stateless": dispatch["stateless"],
            "generation_count": dispatch["generation_count"],
            "execution_capability": dispatch["execution_capability"],
        },
    }


def build_input() -> dict[str, Any]:
    observed = {path: _sha256(path) for path in FROZEN_SHA256}
    mismatches = {
        str(path): {"expected": FROZEN_SHA256[path], "actual": actual}
        for path, actual in observed.items()
        if actual != FROZEN_SHA256[path]
    }
    if mismatches:
        raise RuntimeError(f"Frozen binding inputs changed: {mismatches}")

    payload = _load(PAYLOAD_BASELINE)
    media_memory = _load(SEE_RECALL_BASELINE)
    primary_ids = [
        f"p2-{tool}-{index:02d}"
        for tool in ("search", "image", "ask_claude", "see")
        for index in range(1, 26)
    ]
    recall_ids = [f"p2-recall-{index:02d}" for index in range(1, 26)]
    source_cases = {
        case["case_id"]: case
        for report in (payload, media_memory)
        for case in report["case_results"]
    }
    missing = sorted(set(primary_ids + recall_ids) - source_cases.keys())
    if missing:
        raise RuntimeError(f"Missing frozen binder cases: {missing}")

    cases = [
        _case_input(
            source_cases[case_id],
            "primary_targeted_100" if case_id in primary_ids else "recall_diagnostic_25",
        )
        for case_id in primary_ids + recall_ids
    ]
    encoded = json.dumps(cases, ensure_ascii=False)
    forbidden = (
        '"expected_tool"',
        '"expected_arguments"',
        '"argument_alternatives"',
        '"strict_exact_call"',
        '"literal-source-annotations"',
    )
    leaked = [token for token in forbidden if token in encoded]
    if leaked:
        raise RuntimeError(f"Gold/scoring data leaked into binder input: {leaked}")
    if any(case["hammer_record"]["execution_capability"] for case in cases):
        raise RuntimeError("Unexpected execution capability in frozen Hammer records")

    return {
        "version": "deterministic-binding-input-targeted100-recall25-v1",
        "status": "gold_and_literal_annotation_free",
        "configuration": {
            "models_called": False,
            "delegations_regenerated": False,
            "hammer_generations_regenerated": False,
            "tool_execution_capability": False,
            "case_count": 125,
        },
        "source_reports": {
            str(PAYLOAD_BASELINE): observed[PAYLOAD_BASELINE],
            str(SEE_RECALL_BASELINE): observed[SEE_RECALL_BASELINE],
        },
        "splits": {
            "primary_targeted_100": primary_ids,
            "recall_diagnostic_25": recall_ids,
        },
        "cases": cases,
    }


def main() -> None:
    payload = build_input()
    write_report(payload, OUTPUT)
    checksum = _sha256(OUTPUT)
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{checksum}  {OUTPUT.name}\n", encoding="utf-8"
    )
    print(checksum)


if __name__ == "__main__":
    main()
