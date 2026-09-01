"""Zero-inference typed deterministic-binder replay over frozen Flash 168 calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from .argument_contract import (
    CONTRACT_VERSION,
    LITERAL_SOURCE,
    score_contract_arguments,
    tool_argument_contract,
)
from .argument_scoring import score_arguments
from .benchmark import write_report
from .build_flash168_literal_annotations import OUTPUT as ANNOTATIONS
from .qwen38_flash_direct import (
    BROAD_SHA256,
    PHASE2_CASES,
    PHASE2_SHA256,
    TARGETED_SHA256,
)
from .qwen38_flash_direct import REPORT_ROOT as FLASH_REPORT_ROOT
from .registry import ToolRegistry
from .typed_literal_binder import bind_typed_call

REPORT_ROOT = FLASH_REPORT_ROOT / "typed-binder"
FROZEN_FLASH_REPORT = (
    FLASH_REPORT_ROOT / "qwen38-flash-direct-native-screen-168-v1.json"
)
FROZEN_FLASH_REPORT_SHA256 = (
    "0c71c299cbd559a3c2a0f0dd67e90236620926cda1859db89072239a601baade"
)
ANNOTATIONS_SHA256 = (
    "3a5ee9f97f6771b6977a3652418cc89729cbf4d8341791543261c02ec6f9662e"
)

BINDER_INPUT = REPORT_ROOT / "flash168-typed-binder-input-v1.json"
BINDER_PREDICTIONS = REPORT_ROOT / "flash168-typed-binder-predictions-v1.json"
BINDER_ABSTENTIONS = REPORT_ROOT / "flash168-typed-binder-abstentions-v1.jsonl"
BINDER_FREEZE = REPORT_ROOT / "flash168-typed-binder-freeze-v1.json"
COMPARISON = REPORT_ROOT / "flash168-typed-binder-comparison-v1.json"
CASES_OUTPUT = REPORT_ROOT / "flash168-typed-binder-cases-v1.jsonl"
REPORT_OUTPUT = REPORT_ROOT / "FLASH168-TYPED-BINDER-REPLAY-REPORT.md"
CHECKSUM_OUTPUT = REPORT_ROOT / "FLASH168-TYPED-BINDER-SHA256SUMS-v1.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n",
        encoding="utf-8",
    )
    return checksum


def _verify_exact(path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen input changed: {path}: {actual} != {expected}")
    return actual


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch: {path}: {actual} != {expected}")
    return actual


def _verify_sources() -> dict[str, str]:
    return {
        str(FROZEN_FLASH_REPORT): _verify_exact(
            FROZEN_FLASH_REPORT,
            FROZEN_FLASH_REPORT_SHA256,
        ),
        str(ANNOTATIONS): _verify_exact(ANNOTATIONS, ANNOTATIONS_SHA256),
        str(PHASE2_CASES): _verify_exact(PHASE2_CASES, PHASE2_SHA256),
    }


def prepare() -> None:
    observed = _verify_sources()
    report = _load(FROZEN_FLASH_REPORT)
    cases = []
    for case in report["case_results"]:
        generation = case.get("generation") or {}
        cases.append(
            {
                "index": case["index"],
                "suite": case["suite"],
                "case_id": case["case_id"],
                "raw_request": case["request"],
                "raw_request_sha256": hashlib.sha256(
                    case["request"].encode()
                ).hexdigest(),
                "selected_call": deepcopy(case["parse"]["canonical_call"]),
                "offered_tools": deepcopy(case["offered_tools"]),
                "frozen_flash_record": {
                    "model_id": generation.get("model_id"),
                    "provider": (generation.get("extra") or {}).get("provider"),
                    "raw_generation": generation.get("raw_text"),
                    "generation_error": generation.get("error"),
                    "native_dialect": case["native_dialect"],
                    "parse": deepcopy(case["parse"]),
                    "validation": deepcopy(case["validation"]),
                    "execution": deepcopy(case["execution"]),
                },
            }
        )
    if len(cases) != 168 or len({case["case_id"] for case in cases}) != 168:
        raise RuntimeError("Frozen Flash input is not 168 unique cases")
    encoded = json.dumps(cases, ensure_ascii=False)
    forbidden = (
        '"expected_tool"',
        '"expected_arguments"',
        '"exact_call"',
        '"contract_scoring"',
        '"literal_reference"',
        '"literal_annotation"',
    )
    leaked = [token for token in forbidden if token in encoded]
    if leaked:
        raise RuntimeError(f"Gold or annotations leaked into binder input: {leaked}")
    if any(case["frozen_flash_record"]["execution"]["executed"] for case in cases):
        raise RuntimeError("Frozen Flash report contains tool execution")

    payload = {
        "version": "flash168-typed-binder-input-v1",
        "status": "gold_and_literal_annotation_free",
        "configuration": {
            "case_count": 168,
            "models_called": False,
            "new_generation": False,
            "flash_rerun": False,
            "sacred_450_used": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "source_reports": observed,
        "suite_sha256": {
            "broad-72": BROAD_SHA256,
            "targeted-96": TARGETED_SHA256,
        },
        "cases": cases,
    }
    write_report(payload, BINDER_INPUT)
    print(_write_checksum(BINDER_INPUT))


def bind() -> None:
    _verify_sources()
    input_sha = _verify_sidecar(BINDER_INPUT)
    payload = _load(BINDER_INPUT)
    if payload["status"] != "gold_and_literal_annotation_free":
        raise RuntimeError("Binder input isolation status is invalid")

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
        "version": "flash168-typed-binder-predictions-v1",
        "status": "predictions_frozen_before_scoring",
        "configuration": {
            "input_sha256": input_sha,
            "case_count": 168,
            "models_called": False,
            "new_generation": False,
            "flash_rerun": False,
            "gold_available_to_binder": False,
            "literal_annotations_available_to_binder": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "status_counts": {
            status: sum(
                result["binding"]["status"] == status for result in results
            )
            for status in statuses
        },
        "case_results": results,
    }
    write_report(predictions, BINDER_PREDICTIONS)
    _write_jsonl(BINDER_ABSTENTIONS, abstentions)
    print(_write_checksum(BINDER_PREDICTIONS))
    print(_write_checksum(BINDER_ABSTENTIONS))


def freeze() -> None:
    observed = _verify_sources()
    input_sha = _verify_sidecar(BINDER_INPUT)
    prediction_sha = _verify_sidecar(BINDER_PREDICTIONS)
    abstention_sha = _verify_sidecar(BINDER_ABSTENTIONS)
    binder_source = Path(__file__).with_name("typed_literal_binder.py")
    payload = {
        "version": "flash168-typed-binder-freeze-v1",
        "status": "frozen_before_scoring",
        "frozen_sources": observed,
        "binder_input": {"path": str(BINDER_INPUT), "sha256": input_sha},
        "binder_predictions": {
            "path": str(BINDER_PREDICTIONS),
            "sha256": prediction_sha,
        },
        "binder_abstentions": {
            "path": str(BINDER_ABSTENTIONS),
            "sha256": abstention_sha,
        },
        "binder_source": {
            "path": str(binder_source),
            "sha256": _sha256(binder_source),
        },
        "literal_annotations": {
            "path": str(ANNOTATIONS),
            "sha256": ANNOTATIONS_SHA256,
            "frozen_before_binder_predictions": True,
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
    write_report(payload, BINDER_FREEZE)
    print(_write_checksum(BINDER_FREEZE))


def _score_case(
    baseline: dict[str, Any],
    prediction: dict[str, Any],
    annotation: dict[str, Any] | None,
    registry: ToolRegistry,
) -> dict[str, Any]:
    expected_tool = baseline["expected_tool"]
    expected_arguments = baseline["expected_arguments"]
    baseline_call = baseline["parse"]["canonical_call"]
    final_call = prediction["binding"]["final_call"]
    baseline_tool = baseline_call.get("tool") if baseline_call else None
    final_tool = final_call.get("tool") if final_call else None
    baseline_arguments = baseline_call.get("arguments") if baseline_call else None
    final_arguments = final_call.get("arguments") if final_call else None
    baseline_schema = bool(baseline["validation"]["passed"])
    final_validation = (
        registry.validate_call(final_call, baseline["offered_tools"])
        if final_call is not None
        else {"passed": False, "errors": ["Typed binder abstained or no call existed."]}
    )
    final_schema = bool(final_validation["passed"])

    baseline_argument_score = score_arguments(
        expected_tool,
        expected_arguments,
        baseline_arguments,
    )
    final_argument_score = score_arguments(
        expected_tool,
        expected_arguments,
        final_arguments,
    )
    literal_references = None
    if annotation is not None:
        literal_references = {
            annotation["argument_name"]: annotation["literal_value"]
        }
    baseline_contract = score_contract_arguments(
        expected_tool,
        expected_arguments,
        baseline_arguments,
        literal_references=literal_references,
        declared_version=CONTRACT_VERSION,
    )
    final_contract = score_contract_arguments(
        expected_tool,
        expected_arguments,
        final_arguments,
        literal_references=literal_references,
        declared_version=CONTRACT_VERSION,
    )

    baseline_tool_correct = baseline_tool == expected_tool
    final_tool_correct = final_tool == expected_tool
    baseline_strict = (
        baseline_tool_correct and baseline_argument_score["strict"] and baseline_schema
    )
    final_strict = final_tool_correct and final_argument_score["strict"] and final_schema
    baseline_equivalent = (
        baseline_tool_correct
        and baseline_argument_score["execution_equivalent"]
        and baseline_schema
    )
    final_equivalent = (
        final_tool_correct
        and final_argument_score["execution_equivalent"]
        and final_schema
    )
    baseline_contract_exact = (
        baseline_tool_correct
        and baseline_contract["contract_correct"] is True
        and baseline_schema
    )
    final_contract_exact = (
        final_tool_correct
        and final_contract["contract_correct"] is True
        and final_schema
    )
    mode = tool_argument_contract(expected_tool)["mode"]
    result = {
        "index": baseline["index"],
        "suite": baseline["suite"],
        "case_id": baseline["case_id"],
        "request": baseline["request"],
        "expected_tool": expected_tool,
        "expected_arguments": expected_arguments,
        "argument_mode": mode,
        "literal_annotation": annotation,
        "raw_flash_call": baseline_call,
        "binder_status": prediction["binding"]["status"],
        "binder_candidates": prediction["binding"].get("candidates", []),
        "bound_flash_call": final_call,
        "source_copy_valid": prediction["binding"].get("source_copy_valid"),
        "raw_tool_correct": baseline_tool_correct,
        "bound_tool_correct": final_tool_correct,
        "raw_schema_valid": baseline_schema,
        "bound_schema_valid": final_schema,
        "bound_validation": final_validation,
        "raw_historical_strict_exact": baseline_strict,
        "bound_historical_strict_exact": final_strict,
        "raw_execution_equivalent_exact": baseline_equivalent,
        "bound_execution_equivalent_exact": final_equivalent,
        "raw_typed_contract": baseline_contract,
        "bound_typed_contract": final_contract,
        "raw_typed_contract_exact": baseline_contract_exact,
        "bound_typed_contract_exact": final_contract_exact,
        "raw_argument_scoring": baseline_argument_score,
        "bound_argument_scoring": final_argument_score,
        "execution": {
            "capability": False,
            "executed": False,
            "status": "not_available",
        },
    }
    result["historical_repaired"] = not baseline_strict and final_strict
    result["historical_broken"] = baseline_strict and not final_strict
    result["typed_repaired"] = not baseline_contract_exact and final_contract_exact
    result["typed_broken"] = baseline_contract_exact and not final_contract_exact
    result["schema_broken"] = baseline_schema and not final_schema
    return result


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"cases": 0}
    literal = [case for case in cases if case["argument_mode"] == LITERAL_SOURCE]
    source_checks = [
        case["source_copy_valid"]
        for case in literal
        if case["source_copy_valid"] is not None
    ]
    return {
        "cases": len(cases),
        "literal_annotated_cases": len(literal),
        "raw_tool_selection_successes": sum(case["raw_tool_correct"] for case in cases),
        "raw_tool_selection_accuracy": mean(case["raw_tool_correct"] for case in cases),
        "bound_tool_selection_successes": sum(case["bound_tool_correct"] for case in cases),
        "bound_tool_selection_accuracy": mean(case["bound_tool_correct"] for case in cases),
        "raw_historical_strict_successes": sum(case["raw_historical_strict_exact"] for case in cases),
        "raw_historical_strict_accuracy": mean(case["raw_historical_strict_exact"] for case in cases),
        "bound_historical_strict_successes": sum(case["bound_historical_strict_exact"] for case in cases),
        "bound_historical_strict_accuracy": mean(case["bound_historical_strict_exact"] for case in cases),
        "raw_execution_equivalent_successes": sum(case["raw_execution_equivalent_exact"] for case in cases),
        "raw_execution_equivalent_accuracy": mean(case["raw_execution_equivalent_exact"] for case in cases),
        "bound_execution_equivalent_successes": sum(case["bound_execution_equivalent_exact"] for case in cases),
        "bound_execution_equivalent_accuracy": mean(case["bound_execution_equivalent_exact"] for case in cases),
        "raw_typed_contract_successes": sum(case["raw_typed_contract_exact"] for case in cases),
        "raw_typed_contract_exact_accuracy": mean(case["raw_typed_contract_exact"] for case in cases),
        "bound_typed_contract_successes": sum(case["bound_typed_contract_exact"] for case in cases),
        "bound_typed_contract_exact_accuracy": mean(case["bound_typed_contract_exact"] for case in cases),
        "raw_schema_valid_successes": sum(case["raw_schema_valid"] for case in cases),
        "raw_schema_valid_rate": mean(case["raw_schema_valid"] for case in cases),
        "bound_schema_valid_successes": sum(case["bound_schema_valid"] for case in cases),
        "bound_schema_valid_rate": mean(case["bound_schema_valid"] for case in cases),
        "literal_binder_coverage_successes": sum(
            case["binder_status"] in {"bound", "bound_composite"}
            for case in literal
        ),
        "literal_binder_coverage_rate": mean(
            case["binder_status"] in {"bound", "bound_composite"}
            for case in literal
        ) if literal else None,
        "literal_raw_typed_successes": sum(case["raw_typed_contract_exact"] for case in literal),
        "literal_raw_typed_exact_accuracy": mean(case["raw_typed_contract_exact"] for case in literal) if literal else None,
        "literal_bound_typed_successes": sum(case["bound_typed_contract_exact"] for case in literal),
        "literal_bound_typed_exact_accuracy": mean(case["bound_typed_contract_exact"] for case in literal) if literal else None,
        "literal_raw_historical_successes": sum(case["raw_historical_strict_exact"] for case in literal),
        "literal_raw_historical_accuracy": mean(case["raw_historical_strict_exact"] for case in literal) if literal else None,
        "literal_bound_historical_successes": sum(case["bound_historical_strict_exact"] for case in literal),
        "literal_bound_historical_accuracy": mean(case["bound_historical_strict_exact"] for case in literal) if literal else None,
        "binder_abstention_count": sum(case["bound_flash_call"] is None for case in cases),
        "binder_ambiguity_count": sum(case["binder_status"] == "ambiguous" for case in cases),
        "typed_passthrough_count": sum(case["binder_status"] == "typed_passthrough" for case in cases),
        "source_copy_checked_cases": len(source_checks),
        "source_copy_invariant_rate": mean(source_checks) if source_checks else None,
        "historical_repaired_count": sum(case["historical_repaired"] for case in cases),
        "historical_broken_count": sum(case["historical_broken"] for case in cases),
        "typed_repaired_count": sum(case["typed_repaired"] for case in cases),
        "typed_broken_count": sum(case["typed_broken"] for case in cases),
        "schema_broken_count": sum(case["schema_broken"] for case in cases),
    }


def score() -> None:
    observed = _verify_sources()
    freeze_sha = _verify_sidecar(BINDER_FREEZE)
    predictions = _load(BINDER_PREDICTIONS)
    if predictions["status"] != "predictions_frozen_before_scoring":
        raise RuntimeError("Binder predictions were not frozen before scoring")
    annotations_payload = _load(ANNOTATIONS)
    if annotations_payload["binder_access"] is not False:
        raise RuntimeError("Literal annotations were not isolated from the binder")
    annotations = annotations_payload["annotations"]
    baseline = _load(FROZEN_FLASH_REPORT)
    baseline_by_id = {
        case["case_id"]: case for case in baseline["case_results"]
    }
    prediction_by_id = {
        case["case_id"]: case for case in predictions["case_results"]
    }
    if list(baseline_by_id) != list(prediction_by_id):
        raise RuntimeError("Frozen Flash and binder prediction case orders differ")
    registry = ToolRegistry()
    cases = []
    for case_id, baseline_case in baseline_by_id.items():
        expected_mode = tool_argument_contract(
            baseline_case["expected_tool"]
        )["mode"]
        annotation = annotations.get(case_id)
        if (expected_mode == LITERAL_SOURCE) != (annotation is not None):
            raise RuntimeError(f"Literal annotation coverage mismatch: {case_id}")
        cases.append(
            _score_case(
                baseline_case,
                prediction_by_id[case_id],
                annotation,
                registry,
            )
        )

    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_tool[case["expected_tool"]].append(case)
    overall = _metrics(cases)
    no_regressions = (
        overall["historical_broken_count"] == 0
        and overall["typed_broken_count"] == 0
        and overall["schema_broken_count"] == 0
    )
    threshold = {
        "description": (
            ">=95% typed-contract exact on independently annotated literal "
            "cases with no historical, typed, or schema regressions"
        ),
        "annotated_literal_typed_exact_at_least_95": (
            overall["literal_bound_typed_exact_accuracy"] >= 0.95
        ),
        "no_regressions": no_regressions,
    }
    threshold["passed"] = all(
        value for key, value in threshold.items() if key != "description"
    )
    comparison = {
        "version": "flash168-typed-binder-comparison-v1",
        "status": "completed",
        "frozen_inputs": {**observed, str(BINDER_FREEZE): freeze_sha},
        "configuration": {
            "architecture": (
                "frozen Qwen3.8-Flash native call -> typed field contract -> "
                "deterministic literal binder -> schema validation -> score only"
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
        "overall": overall,
        "per_tool": {
            tool: _metrics(tool_cases)
            for tool, tool_cases in sorted(by_tool.items())
        },
        "suggested_decision_threshold": threshold,
        "case_lists": {
            name: [case["case_id"] for case in cases if case[name]]
            for name in (
                "historical_repaired",
                "historical_broken",
                "typed_repaired",
                "typed_broken",
                "schema_broken",
            )
        },
        "abstentions": [
            {
                "case_id": case["case_id"],
                "expected_tool": case["expected_tool"],
                "binder_status": case["binder_status"],
            }
            for case in cases
            if case["bound_flash_call"] is None
        ],
        "case_results": cases,
    }
    write_report(comparison, COMPARISON)
    _write_jsonl(CASES_OUTPUT, cases)
    print(_write_checksum(COMPARISON))
    print(_write_checksum(CASES_OUTPUT))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def report() -> None:
    comparison = _load(COMPARISON)
    metrics = comparison["overall"]
    threshold = comparison["suggested_decision_threshold"]
    per_tool = comparison["per_tool"]
    rows = []
    for tool, values in per_tool.items():
        rows.append(
            f"| `{tool}` | {values['cases']} | "
            f"{values['raw_historical_strict_successes']}/{values['cases']} "
            f"({_pct(values['raw_historical_strict_accuracy'])}) | "
            f"{values['bound_historical_strict_successes']}/{values['cases']} "
            f"({_pct(values['bound_historical_strict_accuracy'])}) | "
            f"{values['raw_typed_contract_successes']}/{values['cases']} "
            f"({_pct(values['raw_typed_contract_exact_accuracy'])}) | "
            f"{values['bound_typed_contract_successes']}/{values['cases']} "
            f"({_pct(values['bound_typed_contract_exact_accuracy'])}) |"
        )
    decision = (
        "Flash has earned a future sacred-450 evaluation under the stated "
        "evidence rule."
        if threshold["passed"]
        else "Flash has not yet earned the sacred-450 run under the stated "
        "zero-regression rule. Freeze this result and repair the general "
        "external-agent payload boundary on a separate development probe first."
    )
    markdown = f"""# Qwen3.8-Flash frozen 168 typed-binder replay

Date: 2026-08-26

## Result

**Zero-inference binder threshold: {'PASS' if threshold['passed'] else 'FAIL'}.** Applying the typed deterministic binder to the already frozen 168 Flash calls raised historical strict exact from **{metrics['raw_historical_strict_successes']}/168 ({_pct(metrics['raw_historical_strict_accuracy'])})** to **{metrics['bound_historical_strict_successes']}/168 ({_pct(metrics['bound_historical_strict_accuracy'])})**. Fully covered typed-contract exact rose from **{metrics['raw_typed_contract_successes']}/168 ({_pct(metrics['raw_typed_contract_exact_accuracy'])})** to **{metrics['bound_typed_contract_successes']}/168 ({_pct(metrics['bound_typed_contract_exact_accuracy'])})**.

On the **104 independently annotated literal-source cases**, raw Flash was **{metrics['literal_raw_typed_successes']}/104 ({_pct(metrics['literal_raw_typed_exact_accuracy'])})** contract exact and bound Flash was **{metrics['literal_bound_typed_successes']}/104 ({_pct(metrics['literal_bound_typed_exact_accuracy'])})**. Historical strict on the same literal subset moved from **{metrics['literal_raw_historical_successes']}/104 ({_pct(metrics['literal_raw_historical_accuracy'])})** to **{metrics['literal_bound_historical_successes']}/104 ({_pct(metrics['literal_bound_historical_accuracy'])})**.

The replay repaired **{metrics['historical_repaired_count']} historical strict failures** and **{metrics['typed_repaired_count']} typed failures**, with **{metrics['historical_broken_count']} historical**, **{metrics['typed_broken_count']} typed**, and **{metrics['schema_broken_count']} schema regressions**.

## Controlled architecture

```text
immutable raw request ──────────────────────────────┐
        ↓                                           │
frozen Qwen3.8-Flash native all-18 tool call        │
        ↓                                           │
typed field contract                               │
        ├─ semantic/no-argument → unchanged         │
        └─ literal_source → deterministic raw span ◀┘
                         ↓
                  schema validation
                         ↓
                    scoring only
```

No model was called or loaded. Flash was not rerun, no output was resampled, the sacred 450 was not read as evaluation data, and no Yuki tool could execute.

## Isolation and annotation policy

- The 104 annotations were frozen before binder predictions.
- The annotation builder reads immutable requests and expected tool labels only; it does not read Flash outputs or old expected argument values.
- The gold-free binder input contains no expected tool, expected arguments, score, or literal annotation fields.
- The binder module has no filesystem, benchmark, annotation, or model access.
- Every emitted literal value must be a raw-request slice. Composite Yuki values may insert only the contract-declared `|` separator between independently copied components.
- Ambiguous or unbound spans cause abstention rather than guessing.

## Overall comparison

| Metric | Raw Flash | Bound Flash |
|---|---:|---:|
| Tool selection | {metrics['raw_tool_selection_successes']}/168 ({_pct(metrics['raw_tool_selection_accuracy'])}) | {metrics['bound_tool_selection_successes']}/168 ({_pct(metrics['bound_tool_selection_accuracy'])}) |
| Historical strict exact | {metrics['raw_historical_strict_successes']}/168 ({_pct(metrics['raw_historical_strict_accuracy'])}) | **{metrics['bound_historical_strict_successes']}/168 ({_pct(metrics['bound_historical_strict_accuracy'])})** |
| Execution-equivalent exact | {metrics['raw_execution_equivalent_successes']}/168 ({_pct(metrics['raw_execution_equivalent_accuracy'])}) | **{metrics['bound_execution_equivalent_successes']}/168 ({_pct(metrics['bound_execution_equivalent_accuracy'])})** |
| Typed-contract exact | {metrics['raw_typed_contract_successes']}/168 ({_pct(metrics['raw_typed_contract_exact_accuracy'])}) | **{metrics['bound_typed_contract_successes']}/168 ({_pct(metrics['bound_typed_contract_exact_accuracy'])})** |
| Schema valid | {metrics['raw_schema_valid_successes']}/168 ({_pct(metrics['raw_schema_valid_rate'])}) | {metrics['bound_schema_valid_successes']}/168 ({_pct(metrics['bound_schema_valid_rate'])}) |

Typed scoring now covers all 168 cases: 104 independent literal references plus the existing deterministic semantic/no-argument comparators. The two bound typed failures are `target-weather-06`, whose frozen Flash request received an upstream Alibaba 429 before inference, and `target-claude-03`, where the binder copied the broader phrase `handle this exact question: review the schema adapter` instead of the annotated payload `review the schema adapter`.

## Binder behavior

- Literal coverage: **{metrics['literal_binder_coverage_successes']}/104 ({_pct(metrics['literal_binder_coverage_rate'])})**
- Typed semantic/no-argument passthrough: **{metrics['typed_passthrough_count']} cases**
- Binder ambiguities: **{metrics['binder_ambiguity_count']}**
- Total final-call absences: **{metrics['binder_abstention_count']}**; this is the pre-existing provider-error no-call, not a source-span ambiguity
- Character-copy invariant: **{metrics['source_copy_checked_cases']}/{metrics['source_copy_checked_cases']} ({_pct(metrics['source_copy_invariant_rate'])})**

## Per-tool comparison

| Tool | Cases | Raw historical | Bound historical | Raw typed | Bound typed |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The binder repaired all 19 destructive literal rewrites: 9 `search.query`, 9 `ask_claude.question`, and 1 `image.prompt`. It also broke the previously correct `target-claude-03` call by treating delegation scaffolding as payload. The 104/104 source-copy invariant therefore proves literal copying, not correct span selection. Semantic fields such as `weather.city`, `calc.expression`, `see.target`, `remember.fact`, and `recall.topic` were unchanged. Paths, commands, URLs, Yuki filenames, and Yuki contents remained exact and non-regressive.

## Decision

The proposed threshold was at least 95% contract exact on independently annotated literal cases with no regressions. Accuracy reached **{metrics['literal_bound_typed_successes']}/104 ({_pct(metrics['literal_bound_typed_exact_accuracy'])})**, but the replay caused **{metrics['typed_broken_count']} typed regression**. The accuracy condition passes; the no-regression condition fails. Therefore the frozen v1 replay does **not** pass the complete threshold.

This supports the architecture:

```text
Qwen3.8-Flash native all-18 selection
    → typed field contract
    → deterministic literal binder
    → schema validation
    → Yuki tool
```

{decision} No sacred-450 run was performed here.

## Frozen checksums

- Frozen Flash 168 report: `{FROZEN_FLASH_REPORT_SHA256}`
- Literal annotations: `{ANNOTATIONS_SHA256}`
- Broad 72: `{BROAD_SHA256}`
- Targeted 96: `{TARGETED_SHA256}`
- Sacred 450 remains: `{PHASE2_SHA256}`
"""
    REPORT_OUTPUT.write_text(markdown, encoding="utf-8")
    _write_checksum(REPORT_OUTPUT)

    artifact_paths = [
        ANNOTATIONS,
        BINDER_INPUT,
        BINDER_PREDICTIONS,
        BINDER_ABSTENTIONS,
        BINDER_FREEZE,
        COMPARISON,
        CASES_OUTPUT,
        REPORT_OUTPUT,
    ]
    CHECKSUM_OUTPUT.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifact_paths)
        + f"{FROZEN_FLASH_REPORT_SHA256}  SOURCE::{FROZEN_FLASH_REPORT.name}\n"
        + f"{PHASE2_SHA256}  SOURCE::{PHASE2_CASES.name}\n",
        encoding="utf-8",
    )
    print(_write_checksum(CHECKSUM_OUTPUT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "bind", "freeze", "score", "report"),
    )
    stage = parser.parse_args().stage
    globals()[stage]()


if __name__ == "__main__":
    main()
