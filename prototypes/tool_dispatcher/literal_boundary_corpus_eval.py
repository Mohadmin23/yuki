"""Evaluate binder v3 on frozen fresh development and holdout boundary splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from .benchmark import write_report
from .build_literal_boundary_corpus import (
    CORPUS_MANIFEST,
    DEV_ANNOTATIONS,
    DEV_INPUT,
    HOLDOUT_ANNOTATIONS,
    HOLDOUT_INPUT,
    REPORT_ROOT,
)
from .registry import ToolRegistry
from .typed_literal_binder_v2 import bind_typed_call as bind_v2
from .typed_literal_binder_v3 import bind_typed_call as bind_v3

CORPUS_MANIFEST_SHA256 = (
    "e11282f91b61a5ae752b5a02e0f649ad2835c3dfc3fa5abada9f5d6cb44c8bf9"
)
DEV_INPUT_SHA256 = "f399f91f981e08443439676afe6bc61e8f4d9260bf0fa09b1c30ab884d71dfee"
DEV_ANNOTATIONS_SHA256 = (
    "39a3f6a059eb44dd5eefb5a9c90b4fea0e37174c2766773156487afcffbcec2b"
)
HOLDOUT_INPUT_SHA256 = (
    "1b5f4347f70d072564e60bc992c6f3533e58dac21f35961dee57ce8fc0579c2b"
)
HOLDOUT_ANNOTATIONS_SHA256 = (
    "36b390e9ca35c7a0153a067e8bbb2ca936c13f801893e651f23c706edabdba3d"
)
V2_SOURCE = Path(__file__).with_name("typed_literal_binder_v2.py")
V2_SOURCE_SHA256 = "6b1cc7da143b8d8986d0063ae3c06343719b00a59850498f58340695ba85174c"
V3_SOURCE = Path(__file__).with_name("typed_literal_binder_v3.py")

DEV_PREDICTIONS = REPORT_ROOT / "literal-boundary-dev-predictions-v3.json"
DEV_COMPARISON = REPORT_ROOT / "literal-boundary-dev-comparison-v3.json"
V3_FREEZE = REPORT_ROOT / "LITERAL-BOUNDARY-BINDER-v3-FROZEN-MANIFEST.json"
HOLDOUT_PREDICTIONS = REPORT_ROOT / "literal-boundary-holdout-predictions-v3.json"
HOLDOUT_COMPARISON = REPORT_ROOT / "literal-boundary-holdout-comparison-v3.json"
FINAL_COMPARISON = REPORT_ROOT / "literal-boundary-v3-dev-holdout-comparison.json"
REPORT = REPORT_ROOT / "LITERAL-BOUNDARY-v3-DEV-HOLDOUT-REPORT.md"
CHECKSUMS = REPORT_ROOT / "LITERAL-BOUNDARY-v3-SHA256SUMS.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen artifact changed: {path}: {actual} != {expected}")
    return actual


def _verify_sidecar(path: Path) -> str:
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    return _verify(path, expected)


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n",
        encoding="utf-8",
    )
    return checksum


def _verify_dev_inputs() -> None:
    _verify(CORPUS_MANIFEST, CORPUS_MANIFEST_SHA256)
    _verify(DEV_INPUT, DEV_INPUT_SHA256)
    _verify(V2_SOURCE, V2_SOURCE_SHA256)


def _verify_holdout_inputs() -> None:
    _verify(CORPUS_MANIFEST, CORPUS_MANIFEST_SHA256)
    _verify(HOLDOUT_INPUT, HOLDOUT_INPUT_SHA256)
    _verify(V2_SOURCE, V2_SOURCE_SHA256)


def _run_split(
    *,
    input_path: Path,
    output_path: Path,
    expected_count: int,
    split: str,
) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload["status"] != "gold_and_annotations_free":
        raise RuntimeError(f"{split} binder input is not isolated")
    binders: tuple[tuple[str, Callable[..., dict[str, Any]]], ...] = (
        ("v2", bind_v2),
        ("v3", bind_v3),
    )
    results = []
    for case in payload["cases"]:
        bindings = {
            name: binder(
                raw_request=case["request"],
                selected_call=case["selected_call"],
            )
            for name, binder in binders
        }
        results.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "request": case["request"],
                "request_sha256": case["request_sha256"],
                "selected_call": case["selected_call"],
                "bindings": bindings,
                "execution": case["execution"],
            }
        )
    if len(results) != expected_count:
        raise RuntimeError(f"Unexpected {split} prediction count: {len(results)}")
    output = {
        "version": f"literal-boundary-{split}-predictions-v3",
        "status": "predictions_frozen_before_scoring",
        "split": split,
        "case_count": len(results),
        "configuration": {
            "models_called": False,
            "annotations_available_to_binders": False,
            "official_benchmark_cases_used": False,
            "sacred_450_used": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "case_results": results,
    }
    write_report(output, output_path)
    print(_write_checksum(output_path))


def dev_run() -> None:
    _verify_dev_inputs()
    _run_split(
        input_path=DEV_INPUT,
        output_path=DEV_PREDICTIONS,
        expected_count=200,
        split="dev",
    )


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"cases": 0}
    source_checks = [
        case["v3_source_copy_valid"]
        for case in cases
        if case["v3_source_copy_valid"] is not None
    ]
    return {
        "cases": len(cases),
        "v2_exact_successes": sum(case["v2_exact"] for case in cases),
        "v2_exact_accuracy": mean(case["v2_exact"] for case in cases),
        "v3_exact_successes": sum(case["v3_exact"] for case in cases),
        "v3_exact_accuracy": mean(case["v3_exact"] for case in cases),
        "v2_schema_successes": sum(case["v2_schema"] for case in cases),
        "v2_schema_rate": mean(case["v2_schema"] for case in cases),
        "v3_schema_successes": sum(case["v3_schema"] for case in cases),
        "v3_schema_rate": mean(case["v3_schema"] for case in cases),
        "v3_source_copy_checked": len(source_checks),
        "v3_source_copy_successes": sum(source_checks),
        "v3_source_copy_rate": mean(source_checks) if source_checks else None,
        "v2_failure_to_v3_success": sum(
            not case["v2_exact"] and case["v3_exact"] for case in cases
        ),
        "v2_success_to_v3_failure": sum(
            case["v2_exact"] and not case["v3_exact"] for case in cases
        ),
        "schema_regressions": sum(
            case["v2_schema"] and not case["v3_schema"] for case in cases
        ),
        "v3_abstentions": sum(case["v3_call"] is None for case in cases),
    }


def _score_split(
    *,
    split: str,
    prediction_path: Path,
    annotation_path: Path,
    comparison_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    prediction_sha = _verify_sidecar(prediction_path)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if predictions["status"] != "predictions_frozen_before_scoring":
        raise RuntimeError(f"{split} predictions were not frozen")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotations["binder_access"] is not False:
        raise RuntimeError(f"{split} annotations were exposed to binders")
    annotation_by_id = annotations["annotations"]
    registry = ToolRegistry()
    cases = []
    for prediction in predictions["case_results"]:
        annotation = annotation_by_id[prediction["case_id"]]
        scored: dict[str, Any] = {
            "case_id": prediction["case_id"],
            "family": prediction["family"],
            "request": prediction["request"],
            "annotation": annotation,
            "execution": prediction["execution"],
        }
        for version in ("v2", "v3"):
            binding = prediction["bindings"][version]
            call = binding["final_call"]
            validation = (
                registry.validate_call(call, registry.names)
                if call is not None
                else {"passed": False, "errors": ["Binder abstained."]}
            )
            value = None
            if call is not None and isinstance(call.get("arguments"), dict):
                value = call["arguments"].get(annotation["argument_name"])
            scored[f"{version}_call"] = call
            scored[f"{version}_status"] = binding["status"]
            scored[f"{version}_source_copy_valid"] = binding.get(
                "source_copy_valid"
            )
            scored[f"{version}_schema"] = bool(validation["passed"])
            scored[f"{version}_exact"] = (
                call is not None
                and call.get("tool") == annotation["tool"]
                and value == annotation["span"]["text"]
                and validation["passed"]
            )
        cases.append(scored)
    if len(cases) != expected_count:
        raise RuntimeError(f"Unexpected {split} score count: {len(cases)}")
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_tool[case["annotation"]["tool"]].append(case)
        by_family[case["family"]].append(case)
    metrics = _metrics(cases)
    gate = {
        "v3_exact_at_least_99": metrics["v3_exact_accuracy"] >= 0.99,
        "v3_schema_100": metrics["v3_schema_rate"] == 1,
        "v3_source_copy_100": metrics["v3_source_copy_rate"] == 1,
        "zero_literal_regressions_vs_v2": (
            metrics["v2_success_to_v3_failure"] == 0
        ),
        "zero_schema_regressions_vs_v2": metrics["schema_regressions"] == 0,
    }
    gate["passed"] = all(gate.values())
    comparison = {
        "version": f"literal-boundary-{split}-comparison-v3",
        "status": "completed",
        "split": split,
        "frozen_inputs": {
            str(prediction_path): prediction_sha,
            str(annotation_path): _sha256(annotation_path),
        },
        "overall": metrics,
        "by_tool": {
            tool: _metrics(tool_cases)
            for tool, tool_cases in sorted(by_tool.items())
        },
        "by_family": {
            family: _metrics(family_cases)
            for family, family_cases in sorted(by_family.items())
        },
        "gate": gate,
        "case_flips": {
            "v2_failure_to_v3_success": [
                case["case_id"]
                for case in cases
                if not case["v2_exact"] and case["v3_exact"]
            ],
            "v2_success_to_v3_failure": [
                case["case_id"]
                for case in cases
                if case["v2_exact"] and not case["v3_exact"]
            ],
            "v3_failures": [
                case["case_id"] for case in cases if not case["v3_exact"]
            ],
        },
        "case_results": cases,
        "safety": predictions["configuration"],
    }
    write_report(comparison, comparison_path)
    _write_checksum(comparison_path)
    return comparison


def dev_score() -> None:
    _verify_dev_inputs()
    _verify(DEV_ANNOTATIONS, DEV_ANNOTATIONS_SHA256)
    comparison = _score_split(
        split="dev",
        prediction_path=DEV_PREDICTIONS,
        annotation_path=DEV_ANNOTATIONS,
        comparison_path=DEV_COMPARISON,
        expected_count=200,
    )
    print(json.dumps(comparison["overall"], indent=2))
    print(json.dumps(comparison["gate"], indent=2))


def freeze_v3() -> None:
    _verify_dev_inputs()
    _verify(DEV_ANNOTATIONS, DEV_ANNOTATIONS_SHA256)
    dev_sha = _verify_sidecar(DEV_COMPARISON)
    dev = json.loads(DEV_COMPARISON.read_text(encoding="utf-8"))
    if not dev["gate"]["passed"]:
        raise RuntimeError("Binder v3 did not pass the development gate")
    v3_sha = _sha256(V3_SOURCE)
    manifest = {
        "version": "literal-boundary-binder-v3-frozen-manifest",
        "status": "binder_v3_frozen_before_holdout_unlock",
        "binder_v3": {"path": str(V3_SOURCE), "sha256": v3_sha},
        "development_comparison": {
            "path": str(DEV_COMPARISON),
            "sha256": dev_sha,
            "gate_passed": True,
        },
        "corpus_manifest": {
            "path": str(CORPUS_MANIFEST),
            "sha256": CORPUS_MANIFEST_SHA256,
        },
        "sealed_holdout": {
            "input_sha256": HOLDOUT_INPUT_SHA256,
            "annotations_sha256": HOLDOUT_ANNOTATIONS_SHA256,
            "read_by_dev_runner": False,
            "scored_before_binder_freeze": False,
        },
        "safety": {
            "models_called": False,
            "sacred_450_used": False,
            "yuki_tools_executed": False,
        },
    }
    write_report(manifest, V3_FREEZE)
    print(_write_checksum(V3_FREEZE))


def _verify_v3_freeze() -> dict[str, Any]:
    freeze_sha = _verify_sidecar(V3_FREEZE)
    manifest = json.loads(V3_FREEZE.read_text(encoding="utf-8"))
    if manifest["status"] != "binder_v3_frozen_before_holdout_unlock":
        raise RuntimeError("Binder v3 is not frozen before holdout")
    _verify(V3_SOURCE, manifest["binder_v3"]["sha256"])
    _verify(DEV_COMPARISON, manifest["development_comparison"]["sha256"])
    manifest["sha256"] = freeze_sha
    return manifest


def holdout_run() -> None:
    _verify_v3_freeze()
    _verify_holdout_inputs()
    _run_split(
        input_path=HOLDOUT_INPUT,
        output_path=HOLDOUT_PREDICTIONS,
        expected_count=100,
        split="holdout",
    )


def holdout_score() -> None:
    _verify_v3_freeze()
    _verify_holdout_inputs()
    _verify(HOLDOUT_ANNOTATIONS, HOLDOUT_ANNOTATIONS_SHA256)
    comparison = _score_split(
        split="holdout",
        prediction_path=HOLDOUT_PREDICTIONS,
        annotation_path=HOLDOUT_ANNOTATIONS,
        comparison_path=HOLDOUT_COMPARISON,
        expected_count=100,
    )
    print(json.dumps(comparison["overall"], indent=2))
    print(json.dumps(comparison["gate"], indent=2))


def report() -> None:
    freeze = _verify_v3_freeze()
    dev = json.loads(DEV_COMPARISON.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_COMPARISON.read_text(encoding="utf-8"))
    dev_metrics = dev["overall"]
    holdout_metrics = holdout["overall"]
    holdout_failures = [
        case for case in holdout["case_results"] if not case["v3_exact"]
    ]
    failure_tools = sorted(
        {case["annotation"]["tool"] for case in holdout_failures}
    )
    holdout_failures_by_tool = {
        tool: sum(
            case["annotation"]["tool"] == tool for case in holdout_failures
        )
        for tool in failure_tools
    }
    failure_statuses = sorted({case["v3_status"] for case in holdout_failures})
    holdout_failure_statuses = {
        status: sum(case["v3_status"] == status for case in holdout_failures)
        for status in failure_statuses
    }
    final_gate = {
        "development_passed": dev["gate"]["passed"],
        "holdout_passed": holdout["gate"]["passed"],
        "holdout_scored_only_after_v3_freeze": True,
    }
    final_gate["passed"] = all(final_gate.values())
    comparison = {
        "version": "literal-boundary-v3-dev-holdout-comparison",
        "status": "completed",
        "frozen_inputs": {
            str(CORPUS_MANIFEST): CORPUS_MANIFEST_SHA256,
            str(V3_FREEZE): freeze["sha256"],
            str(DEV_COMPARISON): _sha256(DEV_COMPARISON),
            str(HOLDOUT_COMPARISON): _sha256(HOLDOUT_COMPARISON),
        },
        "development": dev_metrics,
        "holdout": holdout_metrics,
        "final_gate": final_gate,
        "holdout_failure_ids": holdout["case_flips"]["v3_failures"],
        "holdout_failures_by_tool": holdout_failures_by_tool,
        "holdout_failure_statuses": holdout_failure_statuses,
        "safety": {
            "models_called": False,
            "sacred_450_used": False,
            "historical_168_used": False,
            "yuki_tools_executed": False,
        },
    }
    write_report(comparison, FINAL_COMPARISON)
    _write_checksum(FINAL_COMPARISON)

    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    markdown = f"""# Literal payload binder v3: fresh development and holdout study

Date: 2026-08-26

## Result

**Fresh-corpus gate: {'PASS' if final_gate['passed'] else 'FAIL'}.** Binder v3 was developed against 200 fresh cases, frozen by source checksum, and only then evaluated on the untouched 100-case holdout. The historical Flash 168 was not used as an authorization set and the sacred 450 was not touched.

| Split | Binder v2 exact | Binder v3 exact | Schema valid | Source-copy invariant | V2 success → V3 failure |
|---|---:|---:|---:|---:|---:|
| Development 200 | {dev_metrics['v2_exact_successes']}/200 ({pct(dev_metrics['v2_exact_accuracy'])}) | **{dev_metrics['v3_exact_successes']}/200 ({pct(dev_metrics['v3_exact_accuracy'])})** | {dev_metrics['v3_schema_successes']}/200 ({pct(dev_metrics['v3_schema_rate'])}) | {dev_metrics['v3_source_copy_successes']}/{dev_metrics['v3_source_copy_checked']} ({pct(dev_metrics['v3_source_copy_rate'])}) | {dev_metrics['v2_success_to_v3_failure']} |
| Holdout 100 | {holdout_metrics['v2_exact_successes']}/100 ({pct(holdout_metrics['v2_exact_accuracy'])}) | **{holdout_metrics['v3_exact_successes']}/100 ({pct(holdout_metrics['v3_exact_accuracy'])})** | {holdout_metrics['v3_schema_successes']}/100 ({pct(holdout_metrics['v3_schema_rate'])}) | {holdout_metrics['v3_source_copy_successes']}/{holdout_metrics['v3_source_copy_checked']} ({pct(holdout_metrics['v3_source_copy_rate'])}) | {holdout_metrics['v2_success_to_v3_failure']} |

## Isolation protocol

1. All 300 requests and independent span annotations were generated and frozen before `typed_literal_binder_v3.py` existed.
2. Development and holdout use disjoint outer-envelope templates.
3. The development runner read only the 200 development cases and annotations.
4. Binder v3 and the passing development result were checksum-frozen before the holdout runner could execute.
5. Holdout predictions were frozen before holdout annotations were loaded for scoring.
6. Neither the historical 168 nor the sacred 450 participated in development or gating.

## Gate

- Development >=99% exact: **{'PASS' if dev['gate']['v3_exact_at_least_99'] else 'FAIL'}**
- Holdout >=99% exact: **{'PASS' if holdout['gate']['v3_exact_at_least_99'] else 'FAIL'}**
- Development/holdout schema validity 100%: **{'PASS' if dev['gate']['v3_schema_100'] and holdout['gate']['v3_schema_100'] else 'FAIL'}**
- Development/holdout source-copy invariant 100%: **{'PASS' if dev['gate']['v3_source_copy_100'] and holdout['gate']['v3_source_copy_100'] else 'FAIL'}**
- V2-success-to-v3-failure regressions: **{dev_metrics['v2_success_to_v3_failure']} dev, {holdout_metrics['v2_success_to_v3_failure']} holdout**

Passing this study demonstrates generalization across the fresh template-disjoint holdout. It does not itself run or rescore the sacred 450.

## Holdout failure analysis

The 15 misses were concentrated rather than random:

- `search`: **{holdout_failures_by_tool.get('search', 0)}**
- `image`: **{holdout_failures_by_tool.get('image', 0)}**
- `shell`: **{holdout_failures_by_tool.get('shell', 0)}**
- Yuki write/append composites: **{holdout_failures_by_tool.get('yuki_write', 0) + holdout_failures_by_tool.get('yuki_append', 0)}**

V3 abstained on **{holdout_failure_statuses.get('unbound', 0)}** cases and selected an incorrect source span on **{len(holdout_failures) - holdout_failure_statuses.get('unbound', 0)}**. The unseen envelopes exposed lexical/template dependence in search phrasing, image phrasing, shell scope wording, and quoted composite payloads. This is the precise generalization failure the isolated holdout was intended to detect.

The source-copy invariant was **{holdout_metrics['v3_source_copy_successes']}/{holdout_metrics['v3_source_copy_checked']} ({pct(holdout_metrics['v3_source_copy_rate'])}) among emitted calls**. One incorrectly bounded composite did not satisfy the declared filename/content reconstruction invariant. Therefore even the mechanical copy-safety condition missed its required 100% gate.

Binder v3 is frozen as a failed research candidate. Its holdout misses must not be patched in place, and the 100-case holdout must not be reused to authorize a v4 candidate.

## Safety

- Models/API calls: **0**
- Yuki tool executions: **0**
- Historical Flash-168 replays: **0**
- Sacred-450 cases read for evaluation: **0**
"""
    REPORT.write_text(markdown, encoding="utf-8")
    _write_checksum(REPORT)
    artifacts = [
        CORPUS_MANIFEST,
        DEV_PREDICTIONS,
        DEV_COMPARISON,
        V3_FREEZE,
        HOLDOUT_PREDICTIONS,
        HOLDOUT_COMPARISON,
        FINAL_COMPARISON,
        REPORT,
    ]
    CHECKSUMS.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    print(_write_checksum(CHECKSUMS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "dev-run",
            "dev-score",
            "freeze-v3",
            "holdout-run",
            "holdout-score",
            "report",
        ),
    )
    stage = parser.parse_args().stage.replace("-", "_")
    globals()[stage]()


if __name__ == "__main__":
    main()
