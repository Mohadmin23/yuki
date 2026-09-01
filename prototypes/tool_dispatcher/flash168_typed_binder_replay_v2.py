"""Replay the probe-frozen typed literal binder v2 over frozen Flash 168 calls.

This module never performs inference or tool execution. Binder predictions are
created from the existing gold-free v1 input and frozen before annotations or
benchmark gold are loaded for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .argument_contract import CONTRACT_VERSION, LITERAL_SOURCE, tool_argument_contract
from .benchmark import write_report
from .build_flash168_literal_annotations import OUTPUT as ANNOTATIONS
from .flash168_typed_binder_replay import (
    ANNOTATIONS_SHA256,
    BINDER_INPUT,
    FROZEN_FLASH_REPORT,
    FROZEN_FLASH_REPORT_SHA256,
    PHASE2_CASES,
    PHASE2_SHA256,
    REPORT_ROOT,
    _metrics,
    _score_case,
)
from .registry import ToolRegistry
from .typed_literal_binder_v2 import bind_typed_call

V1_MANIFEST = REPORT_ROOT / "FLASH168-TYPED-BINDER-v1-FROZEN-MANIFEST.json"
V1_MANIFEST_SHA256 = (
    "3359e7c9aa4d1b9afa36b671eeb11a5ce700cbed1b4398e515f28f8366c774cf"
)
V1_COMPARISON = REPORT_ROOT / "flash168-typed-binder-comparison-v1.json"
V1_COMPARISON_SHA256 = (
    "662078150149ebf1f30ae444213c9e40204e062c1092a8157d3506bf9cef3ed0"
)
BINDER_INPUT_SHA256 = (
    "eb71a6373a1e75f193e180d3a187e3576c9752fb1890a6ff178177a4767e5507"
)

BOUNDARY_ROOT = REPORT_ROOT / "boundary-probe"
BOUNDARY_COMPARISON = BOUNDARY_ROOT / "literal-boundary-v1-vs-v2-comparison.json"
BOUNDARY_COMPARISON_SHA256 = (
    "3ef6d50a53a991efeb5a43bba7ed44cd9a0ee52c5e15e205a1d8c2ef28aa42d9"
)
V2_BINDER_SOURCE = Path(__file__).with_name("typed_literal_binder_v2.py")
V2_BINDER_SHA256 = (
    "6b1cc7da143b8d8986d0063ae3c06343719b00a59850498f58340695ba85174c"
)

PREDICTIONS = REPORT_ROOT / "flash168-typed-binder-predictions-v2.json"
ABSTENTIONS = REPORT_ROOT / "flash168-typed-binder-abstentions-v2.jsonl"
FREEZE = REPORT_ROOT / "flash168-typed-binder-freeze-v2.json"
COMPARISON = REPORT_ROOT / "flash168-typed-binder-comparison-v2.json"
CASES = REPORT_ROOT / "flash168-typed-binder-cases-v2.jsonl"
REPORT = REPORT_ROOT / "FLASH168-TYPED-BINDER-v2-REPLAY-REPORT.md"
CHECKSUMS = REPORT_ROOT / "FLASH168-TYPED-BINDER-SHA256SUMS-v2.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen input changed: {path}: {actual} != {expected}")
    return actual


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    return _verify(path, expected)


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n",
        encoding="utf-8",
    )
    return checksum


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _verify_v1_freeze() -> dict[str, str]:
    observed = {
        str(V1_MANIFEST): _verify(V1_MANIFEST, V1_MANIFEST_SHA256),
        str(V1_COMPARISON): _verify(V1_COMPARISON, V1_COMPARISON_SHA256),
        str(BINDER_INPUT): _verify(BINDER_INPUT, BINDER_INPUT_SHA256),
        str(FROZEN_FLASH_REPORT): _verify(
            FROZEN_FLASH_REPORT,
            FROZEN_FLASH_REPORT_SHA256,
        ),
        str(ANNOTATIONS): _verify(ANNOTATIONS, ANNOTATIONS_SHA256),
        str(PHASE2_CASES): _verify(PHASE2_CASES, PHASE2_SHA256),
    }
    manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "frozen_do_not_modify_or_rescore_in_place":
        raise RuntimeError("The v1 replay is not marked immutable")
    if manifest["sacred_phase2_sha256"] != PHASE2_SHA256:
        raise RuntimeError("Sacred-450 checksum differs from the frozen manifest")
    return observed


def _verify_probe_gate() -> dict[str, Any]:
    _verify(BOUNDARY_COMPARISON, BOUNDARY_COMPARISON_SHA256)
    comparison = json.loads(BOUNDARY_COMPARISON.read_text(encoding="utf-8"))
    gate = comparison["gate"]
    if not gate["passed"]:
        raise RuntimeError("The fresh payload-boundary probe did not pass")
    checks = gate["checks"]
    if not checks["zero_literal_regressions"] or not checks["zero_schema_regressions"]:
        raise RuntimeError("The fresh probe contains a regression")
    if comparison["v2"]["exact_successes"] != 64:
        raise RuntimeError("The fresh probe is not 64/64 literal exact")
    return comparison


def bind() -> None:
    _verify_v1_freeze()
    probe = _verify_probe_gate()
    _verify(V2_BINDER_SOURCE, V2_BINDER_SHA256)
    payload = json.loads(BINDER_INPUT.read_text(encoding="utf-8"))
    if payload["status"] != "gold_and_literal_annotation_free":
        raise RuntimeError("Frozen binder input is not gold/annotation-free")

    results = []
    abstentions = []
    for case in payload["cases"]:
        binding = bind_typed_call(
            raw_request=case["raw_request"],
            selected_call=case["selected_call"],
        )
        result = {
            "index": case["index"],
            "suite": case["suite"],
            "case_id": case["case_id"],
            "raw_request": case["raw_request"],
            "raw_request_sha256": case["raw_request_sha256"],
            "baseline_selected_call": case["selected_call"],
            "binding": binding,
            "execution": {
                "capability": False,
                "executed": False,
                "status": "not_available",
            },
        }
        results.append(result)
        if binding["final_call"] is None:
            abstentions.append(result)

    statuses = sorted({result["binding"]["status"] for result in results})
    predictions = {
        "version": "flash168-typed-binder-predictions-v2",
        "status": "predictions_frozen_before_scoring",
        "configuration": {
            "input_sha256": BINDER_INPUT_SHA256,
            "probe_comparison_sha256": BOUNDARY_COMPARISON_SHA256,
            "probe_gate_passed": probe["gate"]["passed"],
            "case_count": 168,
            "models_called": False,
            "new_generation": False,
            "flash_rerun": False,
            "gold_available_to_binder": False,
            "literal_annotations_available_to_binder": False,
            "sacred_450_used": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "status_counts": {
            status: sum(result["binding"]["status"] == status for result in results)
            for status in statuses
        },
        "case_results": results,
    }
    write_report(predictions, PREDICTIONS)
    _write_jsonl(ABSTENTIONS, abstentions)
    print(_write_checksum(PREDICTIONS))
    print(_write_checksum(ABSTENTIONS))


def freeze() -> None:
    observed = _verify_v1_freeze()
    _verify_probe_gate()
    _verify(V2_BINDER_SOURCE, V2_BINDER_SHA256)
    prediction_sha = _verify_sidecar(PREDICTIONS)
    abstention_sha = _verify_sidecar(ABSTENTIONS)
    payload = {
        "version": "flash168-typed-binder-freeze-v2",
        "status": "frozen_before_scoring",
        "frozen_sources": observed,
        "development_probe": {
            "path": str(BOUNDARY_COMPARISON),
            "sha256": BOUNDARY_COMPARISON_SHA256,
            "gate_passed": True,
        },
        "binder_source": {
            "path": str(V2_BINDER_SOURCE),
            "sha256": V2_BINDER_SHA256,
        },
        "binder_predictions": {
            "path": str(PREDICTIONS),
            "sha256": prediction_sha,
        },
        "binder_abstentions": {
            "path": str(ABSTENTIONS),
            "sha256": abstention_sha,
        },
        "literal_annotations": {
            "path": str(ANNOTATIONS),
            "sha256": ANNOTATIONS_SHA256,
            "available_to_binder": False,
        },
        "safety": {
            "models_called": False,
            "new_generation": False,
            "gold_available_to_binder": False,
            "literal_annotations_available_to_binder": False,
            "sacred_450_used": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
    }
    write_report(payload, FREEZE)
    print(_write_checksum(FREEZE))


def score() -> None:
    observed = _verify_v1_freeze()
    _verify_probe_gate()
    freeze_sha = _verify_sidecar(FREEZE)
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    if predictions["status"] != "predictions_frozen_before_scoring":
        raise RuntimeError("V2 predictions were not frozen before scoring")
    annotations_payload = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    if annotations_payload["binder_access"] is not False:
        raise RuntimeError("Literal annotations were exposed to the binder")
    annotations = annotations_payload["annotations"]
    baseline = json.loads(FROZEN_FLASH_REPORT.read_text(encoding="utf-8"))
    v1 = json.loads(V1_COMPARISON.read_text(encoding="utf-8"))
    baseline_by_id = {case["case_id"]: case for case in baseline["case_results"]}
    prediction_by_id = {
        case["case_id"]: case for case in predictions["case_results"]
    }
    v1_by_id = {case["case_id"]: case for case in v1["case_results"]}
    if not (list(baseline_by_id) == list(prediction_by_id) == list(v1_by_id)):
        raise RuntimeError("Raw, v1, and v2 case orders differ")

    registry = ToolRegistry()
    cases = []
    for case_id, baseline_case in baseline_by_id.items():
        mode = tool_argument_contract(baseline_case["expected_tool"])["mode"]
        annotation = annotations.get(case_id)
        if (mode == LITERAL_SOURCE) != (annotation is not None):
            raise RuntimeError(f"Literal annotation coverage mismatch: {case_id}")
        case = _score_case(
            baseline_case,
            prediction_by_id[case_id],
            annotation,
            registry,
        )
        old = v1_by_id[case_id]
        case.update(
            {
                "v1_bound_flash_call": old["bound_flash_call"],
                "v1_historical_strict_exact": old["bound_historical_strict_exact"],
                "v1_typed_contract_exact": old["bound_typed_contract_exact"],
                "v1_schema_valid": old["bound_schema_valid"],
                "v1_to_v2_historical_repaired": (
                    not old["bound_historical_strict_exact"]
                    and case["bound_historical_strict_exact"]
                ),
                "v1_to_v2_historical_broken": (
                    old["bound_historical_strict_exact"]
                    and not case["bound_historical_strict_exact"]
                ),
                "v1_to_v2_typed_repaired": (
                    not old["bound_typed_contract_exact"]
                    and case["bound_typed_contract_exact"]
                ),
                "v1_to_v2_typed_broken": (
                    old["bound_typed_contract_exact"]
                    and not case["bound_typed_contract_exact"]
                ),
                "v1_to_v2_schema_broken": (
                    old["bound_schema_valid"] and not case["bound_schema_valid"]
                ),
            }
        )
        cases.append(case)

    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_tool[case["expected_tool"]].append(case)
    overall = _metrics(cases)
    v1_to_v2 = {
        key: sum(case[key] for case in cases)
        for key in (
            "v1_to_v2_historical_repaired",
            "v1_to_v2_historical_broken",
            "v1_to_v2_typed_repaired",
            "v1_to_v2_typed_broken",
            "v1_to_v2_schema_broken",
        )
    }
    no_raw_regressions = (
        overall["historical_broken_count"] == 0
        and overall["typed_broken_count"] == 0
        and overall["schema_broken_count"] == 0
    )
    no_v1_regressions = (
        v1_to_v2["v1_to_v2_historical_broken"] == 0
        and v1_to_v2["v1_to_v2_typed_broken"] == 0
        and v1_to_v2["v1_to_v2_schema_broken"] == 0
    )
    gate = {
        "description": (
            ">=95% typed exact on independent literal annotations, zero raw-to-v2 "
            "regressions, and zero v1-to-v2 regressions"
        ),
        "literal_typed_exact_at_least_95": (
            overall["literal_bound_typed_exact_accuracy"] >= 0.95
        ),
        "no_raw_to_v2_regressions": no_raw_regressions,
        "no_v1_to_v2_regressions": no_v1_regressions,
    }
    gate["passed"] = all(value for key, value in gate.items() if key != "description")
    comparison = {
        "version": "flash168-typed-binder-comparison-v2",
        "status": "completed",
        "frozen_inputs": {**observed, str(FREEZE): freeze_sha},
        "configuration": {
            "architecture": (
                "frozen Qwen3.8-Flash native call -> typed field contract -> "
                "probe-frozen deterministic binder v2 -> validation -> score only"
            ),
            "argument_contract_version": CONTRACT_VERSION,
            "case_count": 168,
            "literal_annotation_count": len(annotations),
            "models_called": False,
            "flash_rerun": False,
            "new_generation": False,
            "sacred_450_used": False,
            "gold_available_to_binder": False,
            "literal_annotations_available_to_binder": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "raw_to_v2": overall,
        "v1_overall": v1["overall"],
        "v1_to_v2": v1_to_v2,
        "per_tool": {
            tool: _metrics(tool_cases)
            for tool, tool_cases in sorted(by_tool.items())
        },
        "authorization_gate": gate,
        "case_lists": {
            name: [case["case_id"] for case in cases if case[name]]
            for name in (
                "historical_repaired",
                "historical_broken",
                "typed_repaired",
                "typed_broken",
                "schema_broken",
                "v1_to_v2_historical_repaired",
                "v1_to_v2_historical_broken",
                "v1_to_v2_typed_repaired",
                "v1_to_v2_typed_broken",
                "v1_to_v2_schema_broken",
            )
        },
        "case_results": cases,
    }
    write_report(comparison, COMPARISON)
    _write_jsonl(CASES, cases)
    print(_write_checksum(COMPARISON))
    print(_write_checksum(CASES))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def report() -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    v2 = comparison["raw_to_v2"]
    v1 = comparison["v1_overall"]
    delta = comparison["v1_to_v2"]
    gate = comparison["authorization_gate"]
    decision = (
        "The zero-regression evidence gate passes. Flash is now authorized for a "
        "future sacred-450 evaluation; that evaluation was not run here."
        if gate["passed"]
        else "The zero-regression evidence gate fails. Flash is not authorized for "
        "the sacred-450 evaluation."
    )
    markdown = f"""# Qwen3.8-Flash frozen 168 typed-binder v2 replay

Date: 2026-08-26

## Result

**Authorization gate: {'PASS' if gate['passed'] else 'FAIL'}.** The fresh 64-case development probe first improved from **31/64** under frozen binder v1 to **64/64** under the general boundary parser, with zero regressions. Only then was binder v2 replayed once over the immutable Flash 168 calls.

Historical strict exact rose from raw **{v2['raw_historical_strict_successes']}/168 ({_pct(v2['raw_historical_strict_accuracy'])})** to v1 **{v1['bound_historical_strict_successes']}/168 ({_pct(v1['bound_historical_strict_accuracy'])})** and v2 **{v2['bound_historical_strict_successes']}/168 ({_pct(v2['bound_historical_strict_accuracy'])})**. Typed-contract exact rose from raw **{v2['raw_typed_contract_successes']}/168 ({_pct(v2['raw_typed_contract_exact_accuracy'])})** to v1 **{v1['bound_typed_contract_successes']}/168 ({_pct(v1['bound_typed_contract_exact_accuracy'])})** and v2 **{v2['bound_typed_contract_successes']}/168 ({_pct(v2['bound_typed_contract_exact_accuracy'])})**.

On the 104 independently annotated literal fields, v2 reached **{v2['literal_bound_typed_successes']}/104 ({_pct(v2['literal_bound_typed_exact_accuracy'])})** typed exact, with **{v2['source_copy_checked_cases']}/{v2['source_copy_checked_cases']}** emitted values satisfying the character-for-character source-copy invariant.

## Raw vs frozen binder versions

| Metric | Raw Flash | Binder v1 | Binder v2 |
|---|---:|---:|---:|
| Historical strict exact | {v2['raw_historical_strict_successes']}/168 ({_pct(v2['raw_historical_strict_accuracy'])}) | {v1['bound_historical_strict_successes']}/168 ({_pct(v1['bound_historical_strict_accuracy'])}) | **{v2['bound_historical_strict_successes']}/168 ({_pct(v2['bound_historical_strict_accuracy'])})** |
| Execution-equivalent exact | {v2['raw_execution_equivalent_successes']}/168 ({_pct(v2['raw_execution_equivalent_accuracy'])}) | {v1['bound_execution_equivalent_successes']}/168 ({_pct(v1['bound_execution_equivalent_accuracy'])}) | **{v2['bound_execution_equivalent_successes']}/168 ({_pct(v2['bound_execution_equivalent_accuracy'])})** |
| Typed-contract exact | {v2['raw_typed_contract_successes']}/168 ({_pct(v2['raw_typed_contract_exact_accuracy'])}) | {v1['bound_typed_contract_successes']}/168 ({_pct(v1['bound_typed_contract_exact_accuracy'])}) | **{v2['bound_typed_contract_successes']}/168 ({_pct(v2['bound_typed_contract_exact_accuracy'])})** |
| Annotated literal typed exact | {v2['literal_raw_typed_successes']}/104 ({_pct(v2['literal_raw_typed_exact_accuracy'])}) | {v1['literal_bound_typed_successes']}/104 ({_pct(v1['literal_bound_typed_exact_accuracy'])}) | **{v2['literal_bound_typed_successes']}/104 ({_pct(v2['literal_bound_typed_exact_accuracy'])})** |
| Schema valid | {v2['raw_schema_valid_successes']}/168 ({_pct(v2['raw_schema_valid_rate'])}) | {v1['bound_schema_valid_successes']}/168 ({_pct(v1['bound_schema_valid_rate'])}) | {v2['bound_schema_valid_successes']}/168 ({_pct(v2['bound_schema_valid_rate'])}) |

V2 repaired **{v2['historical_repaired_count']}** raw historical failures and **{v2['typed_repaired_count']}** raw typed failures. Relative to v1 it repaired **{delta['v1_to_v2_historical_repaired']}** historical and **{delta['v1_to_v2_typed_repaired']}** typed cases, with **{delta['v1_to_v2_historical_broken']} historical**, **{delta['v1_to_v2_typed_broken']} typed**, and **{delta['v1_to_v2_schema_broken']} schema regressions**.

## General boundary rule

The binder parses one outer delegation envelope and then stops. A structural delimiter or grammatical action-plus-transport phrase identifies the boundary. Text after that boundary is copied from the immutable request; prefix-like words inside it are never recursively stripped. Matching quotes/backticks are removed only when they wrap the entire payload. Ambiguous boundaries cause abstention rather than guessing.

This is why these remain different:

```text
Ask Claude this exact question: why does this fail?
                                └─ payload

Ask Claude: handle this exact question: why does this fail?
            └──────────────────────────────────────────── payload
```

## Gate and safety

- Literal typed exact >=95%: **{'PASS' if gate['literal_typed_exact_at_least_95'] else 'FAIL'}**
- Raw-to-v2 regressions: **{v2['historical_broken_count']} historical, {v2['typed_broken_count']} typed, {v2['schema_broken_count']} schema**
- V1-to-v2 regressions: **{delta['v1_to_v2_historical_broken']} historical, {delta['v1_to_v2_typed_broken']} typed, {delta['v1_to_v2_schema_broken']} schema**
- Models/API calls: **0**
- New Flash generations: **0**
- Yuki tool executions: **0**
- Sacred-450 evaluation cases run: **0**

{decision}

## Frozen source hashes

- Original v1 freeze manifest: `{V1_MANIFEST_SHA256}`
- Fresh probe comparison: `{BOUNDARY_COMPARISON_SHA256}`
- Binder v2 source: `{V2_BINDER_SHA256}`
- Frozen Flash 168 report: `{FROZEN_FLASH_REPORT_SHA256}`
- Independent literal annotations: `{ANNOTATIONS_SHA256}`
- Sacred 450 (unchanged): `{PHASE2_SHA256}`
"""
    REPORT.write_text(markdown, encoding="utf-8")
    _write_checksum(REPORT)
    artifacts = [
        V1_MANIFEST,
        BOUNDARY_COMPARISON,
        PREDICTIONS,
        ABSTENTIONS,
        FREEZE,
        COMPARISON,
        CASES,
        REPORT,
    ]
    CHECKSUMS.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts)
        + f"{FROZEN_FLASH_REPORT_SHA256}  SOURCE::{FROZEN_FLASH_REPORT.name}\n"
        + f"{PHASE2_SHA256}  SOURCE::{PHASE2_CASES.name}\n",
        encoding="utf-8",
    )
    print(_write_checksum(CHECKSUMS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bind", "freeze", "score", "report"))
    globals()[parser.parse_args().stage]()


if __name__ == "__main__":
    main()
