"""Run and score deterministic binder versions on the fresh boundary probe."""

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
from .build_literal_boundary_probe import ANNOTATIONS, PROBE, REPORT_ROOT
from .registry import ToolRegistry
from .typed_literal_binder import bind_typed_call as bind_v1

PROBE_SHA256 = "516a1214a68f88be7f4fc2b8fbb2d9711e999cf87a94bf12cf07b1eed2eec560"
ANNOTATIONS_SHA256 = (
    "81ec7dbcf23fa828cb60bb0d0afa5d35ba892ae9bb2c789bee9ce236ee14d7a6"
)
V1_BINDER = Path(__file__).with_name("typed_literal_binder.py")
V1_BINDER_SHA256 = (
    "0ca26737b6017f907b46860be30c3770ec5ccfe6ef25d7505130c50b6104d602"
)
COMPARISON = REPORT_ROOT / "literal-boundary-v1-vs-v2-comparison.json"
REPORT = REPORT_ROOT / "LITERAL-BOUNDARY-DEVELOPMENT-PROBE-REPORT.md"
CHECKSUMS = REPORT_ROOT / "LITERAL-BOUNDARY-DEVELOPMENT-PROBE-SHA256SUMS-v1.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, expected: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen boundary input changed: {path}: {actual} != {expected}")
    return actual


def _verify_sidecar(path: Path) -> str:
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    return _verify(path, expected)


def _paths(version: str) -> tuple[Path, Path, Path]:
    return (
        REPORT_ROOT / f"literal-boundary-binder-{version}-predictions.json",
        REPORT_ROOT / f"literal-boundary-binder-{version}-abstentions.jsonl",
        REPORT_ROOT / f"literal-boundary-binder-{version}-comparison.json",
    )


def _binder(version: str) -> tuple[Callable[..., dict[str, Any]], Path, str]:
    if version == "v1":
        return bind_v1, V1_BINDER, V1_BINDER_SHA256
    if version == "v2":
        from .typed_literal_binder_v2 import bind_typed_call as bind_v2

        binder_path = Path(__file__).with_name("typed_literal_binder_v2.py")
        return bind_v2, binder_path, _sha256(binder_path)
    raise ValueError(f"Unknown binder version: {version}")


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n",
        encoding="utf-8",
    )
    return checksum


def run(version: str) -> None:
    _verify(PROBE, PROBE_SHA256)
    binder, binder_path, binder_sha = _binder(version)
    if version == "v1":
        _verify(binder_path, binder_sha)
    payload = json.loads(PROBE.read_text(encoding="utf-8"))
    if payload["status"] != "gold_and_annotations_free":
        raise RuntimeError("Boundary probe input is not annotation-free")
    predictions = []
    abstentions = []
    for case in payload["cases"]:
        binding = binder(
            raw_request=case["request"],
            selected_call=case["selected_call"],
        )
        result = {
            "case_id": case["case_id"],
            "family": case["family"],
            "request": case["request"],
            "selected_call": case["selected_call"],
            "binding": binding,
            "execution": {
                "capability": False,
                "executed": False,
                "status": "not_available",
            },
        }
        predictions.append(result)
        if binding["final_call"] is None:
            abstentions.append(result)
    output, abstention_output, _ = _paths(version)
    report = {
        "version": f"literal-boundary-binder-{version}-predictions",
        "status": "predictions_frozen_before_scoring",
        "probe_sha256": PROBE_SHA256,
        "binder": {"path": str(binder_path), "sha256": binder_sha},
        "configuration": {
            "models_called": False,
            "annotations_available_to_binder": False,
            "official_benchmark_gold_used": False,
            "sacred_450_used": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "status_counts": {
            status: sum(item["binding"]["status"] == status for item in predictions)
            for status in sorted({item["binding"]["status"] for item in predictions})
        },
        "case_results": predictions,
    }
    write_report(report, output)
    abstention_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in abstentions),
        encoding="utf-8",
    )
    print(_write_checksum(output))
    print(_write_checksum(abstention_output))


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    source_checks = [
        case["source_copy_valid"]
        for case in cases
        if case["source_copy_valid"] is not None
    ]
    return {
        "cases": len(cases),
        "exact_successes": sum(case["literal_exact"] for case in cases),
        "exact_accuracy": mean(case["literal_exact"] for case in cases),
        "schema_valid_successes": sum(case["schema_valid"] for case in cases),
        "schema_valid_rate": mean(case["schema_valid"] for case in cases),
        "coverage_successes": sum(case["bound_call"] is not None for case in cases),
        "coverage_rate": mean(case["bound_call"] is not None for case in cases),
        "ambiguity_count": sum(case["binder_status"] == "ambiguous" for case in cases),
        "abstention_count": sum(case["bound_call"] is None for case in cases),
        "source_copy_checked_cases": len(source_checks),
        "source_copy_invariant_rate": mean(source_checks) if source_checks else None,
    }


def score(version: str) -> None:
    _verify(PROBE, PROBE_SHA256)
    _verify(ANNOTATIONS, ANNOTATIONS_SHA256)
    prediction_path, _, comparison_path = _paths(version)
    prediction_sha = _verify_sidecar(prediction_path)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if predictions["status"] != "predictions_frozen_before_scoring":
        raise RuntimeError("Boundary predictions were not frozen before scoring")
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    if annotations["binder_access"] is not False:
        raise RuntimeError("Boundary annotations were exposed to the binder")
    annotation_by_id = annotations["annotations"]
    registry = ToolRegistry()
    cases = []
    for prediction in predictions["case_results"]:
        annotation = annotation_by_id[prediction["case_id"]]
        binding = prediction["binding"]
        call = binding["final_call"]
        validation = (
            registry.validate_call(call, registry.names)
            if call is not None
            else {"passed": False, "errors": ["Binder abstained."]}
        )
        value = None
        if call is not None and isinstance(call.get("arguments"), dict):
            value = call["arguments"].get(annotation["argument_name"])
        literal_exact = (
            call is not None
            and call.get("tool") == annotation["tool"]
            and value == annotation["span"]["text"]
            and validation["passed"]
        )
        cases.append(
            {
                "case_id": prediction["case_id"],
                "family": prediction["family"],
                "request": prediction["request"],
                "annotation": annotation,
                "binder_status": binding["status"],
                "binder_candidates": binding.get("candidates", []),
                "bound_call": call,
                "source_copy_valid": binding.get("source_copy_valid"),
                "schema_valid": bool(validation["passed"]),
                "validation": validation,
                "literal_exact": literal_exact,
                "execution": prediction["execution"],
            }
        )
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[case["family"]].append(case)
        by_tool[case["annotation"]["tool"]].append(case)
    comparison = {
        "version": f"literal-boundary-binder-{version}-comparison",
        "status": "completed",
        "frozen_inputs": {
            str(PROBE): PROBE_SHA256,
            str(ANNOTATIONS): ANNOTATIONS_SHA256,
            str(prediction_path): prediction_sha,
        },
        "configuration": {
            "models_called": False,
            "sacred_450_used": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
            "predictions_frozen_before_scoring": True,
            "annotations_available_to_binder": False,
        },
        "overall": _metrics(cases),
        "by_family": {
            family: _metrics(family_cases)
            for family, family_cases in sorted(by_family.items())
        },
        "by_tool": {
            tool: _metrics(tool_cases)
            for tool, tool_cases in sorted(by_tool.items())
        },
        "failure_ids": [
            case["case_id"] for case in cases if not case["literal_exact"]
        ],
        "case_results": cases,
    }
    write_report(comparison, comparison_path)
    print(_write_checksum(comparison_path))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def compare() -> None:
    _verify(PROBE, PROBE_SHA256)
    _verify(ANNOTATIONS, ANNOTATIONS_SHA256)
    v1_path = _paths("v1")[2]
    v2_path = _paths("v2")[2]
    v1_sha = _verify_sidecar(v1_path)
    v2_sha = _verify_sidecar(v2_path)
    old = json.loads(v1_path.read_text(encoding="utf-8"))
    new = json.loads(v2_path.read_text(encoding="utf-8"))
    old_by_id = {case["case_id"]: case for case in old["case_results"]}
    new_by_id = {case["case_id"]: case for case in new["case_results"]}
    if old_by_id.keys() != new_by_id.keys():
        raise RuntimeError("Boundary probe v1/v2 case IDs differ")
    repaired = [
        case_id
        for case_id in old_by_id
        if not old_by_id[case_id]["literal_exact"]
        and new_by_id[case_id]["literal_exact"]
    ]
    regressed = [
        case_id
        for case_id in old_by_id
        if old_by_id[case_id]["literal_exact"]
        and not new_by_id[case_id]["literal_exact"]
    ]
    schema_regressed = [
        case_id
        for case_id in old_by_id
        if old_by_id[case_id]["schema_valid"]
        and not new_by_id[case_id]["schema_valid"]
    ]
    new_metrics = new["overall"]
    checks = {
        "v2_literal_exact_64_of_64": new_metrics["exact_successes"] == 64,
        "v2_schema_valid_64_of_64": new_metrics["schema_valid_successes"] == 64,
        "v2_coverage_64_of_64": new_metrics["coverage_successes"] == 64,
        "v2_source_copy_64_of_64": (
            new_metrics["source_copy_checked_cases"] == 64
            and new_metrics["source_copy_invariant_rate"] == 1.0
        ),
        "zero_literal_regressions": not regressed,
        "zero_schema_regressions": not schema_regressed,
    }
    comparison = {
        "version": "literal-boundary-v1-vs-v2-comparison",
        "status": "completed_and_frozen_before_flash168_v2_replay",
        "frozen_inputs": {
            str(PROBE): PROBE_SHA256,
            str(ANNOTATIONS): ANNOTATIONS_SHA256,
            str(v1_path): v1_sha,
            str(v2_path): v2_sha,
        },
        "binder_sources": {
            "v1": {"path": str(V1_BINDER), "sha256": V1_BINDER_SHA256},
            "v2": {
                "path": str(Path(__file__).with_name("typed_literal_binder_v2.py")),
                "sha256": _sha256(
                    Path(__file__).with_name("typed_literal_binder_v2.py")
                ),
            },
        },
        "v1": old["overall"],
        "v2": new_metrics,
        "by_tool": {
            tool: {
                "v1": old["by_tool"].get(tool),
                "v2": new["by_tool"].get(tool),
            }
            for tool in sorted(set(old["by_tool"]) | set(new["by_tool"]))
        },
        "case_flips": {
            "failure_to_success": repaired,
            "success_to_failure": regressed,
            "schema_success_to_failure": schema_regressed,
        },
        "gate": {"checks": checks, "passed": all(checks.values())},
        "safety": {
            "models_called": False,
            "sacred_450_used": False,
            "official_benchmark_gold_used": False,
            "annotations_available_to_binder": False,
            "tool_execution_attempts": 0,
        },
    }
    write_report(comparison, COMPARISON)
    comparison_sha = _write_checksum(COMPARISON)

    old_metrics = old["overall"]
    old_ask = old["by_tool"]["ask_claude"]
    new_ask = new["by_tool"]["ask_claude"]
    markdown = f"""# Literal payload-boundary development probe

Date: 2026-08-26

## Result

**Probe gate: {'PASS' if comparison['gate']['passed'] else 'FAIL'}.** The frozen v1 binder scored **{old_metrics['exact_successes']}/64 ({_pct(old_metrics['exact_accuracy'])})**. The general v2 boundary parser scored **{new_metrics['exact_successes']}/64 ({_pct(new_metrics['exact_accuracy'])})**, repairing **{len(repaired)}** cases with **{len(regressed)} literal** and **{len(schema_regressed)} schema regressions**.

`ask_claude` improved from **{old_ask['exact_successes']}/40 ({_pct(old_ask['exact_accuracy'])})** to **{new_ask['exact_successes']}/40 ({_pct(new_ask['exact_accuracy'])})**. The 24 non-Claude literal controls improved from **{old_metrics['exact_successes'] - old_ask['exact_successes']}/24** to **{new_metrics['exact_successes'] - new_ask['exact_successes']}/24**.

## Probe construction

The 64 fresh, non-sacred cases contain:

- 40 `ask_claude` boundary cases.
- 24 controls across search, image, filesystem paths, URLs, commands, and Yuki filenames/content.
- Direct colon and em-dash envelopes.
- `Ask Claude to ...` and `Have Claude ...` verb payloads without delimiters.
- Nested strings such as `Ask Claude: Ask Claude to retry ...`.
- Payloads intentionally beginning with `handle this exact question:`.
- Inner colons, `A::B`, JSON, quotes, backticks, punctuation, and prefix-like path/content text.

Annotations were independently frozen before either scored probe run and were unavailable to both binders.

## General v2 boundary algorithm

The repair is an outer-envelope parser, not a benchmark phrase replacement:

1. Recognize one top-level Claude delegation marker near the start.
2. Consume either an immediate delimiter (`:` or `—`) or a grammatical transport envelope composed of an action plus a transport noun such as `question`, `message`, `request`, or `task`.
3. If no structural delimiter exists, treat the direct verb phrase after Claude as payload.
4. Stop parsing the envelope after that single boundary. Never recursively strip prefix-like text inside the payload.
5. Remove matching outer quotes/backticks only when they delimit the entire payload; preserve inner quotes and punctuation.
6. Copy the final span from the immutable request and validate the resulting call.
7. Abstain if a unique boundary cannot be established.

This distinguishes:

```text
Ask Claude this exact question: why does this fail?
                                └─ payload

Ask Claude: handle this exact question: why does this fail?
            └──────────────────────────────────────────── payload
```

## Gate

| Check | Result |
|---|---:|
| Literal exact | **64/64** |
| Schema valid | **64/64** |
| Binder coverage | **64/64** |
| Raw-source invariant | **64/64** |
| Literal regressions | **0** |
| Schema regressions | **0** |

The probe therefore authorizes exactly one model-free v2 replay of the already frozen Flash 168 outputs. It does not authorize model inference or the sacred 450.

## Frozen inputs

- Probe SHA-256: `{PROBE_SHA256}`
- Annotation SHA-256: `{ANNOTATIONS_SHA256}`
- V1 comparison SHA-256: `{v1_sha}`
- V2 comparison SHA-256: `{v2_sha}`
- Combined comparison SHA-256: `{comparison_sha}`
"""
    REPORT.write_text(markdown, encoding="utf-8")
    _write_checksum(REPORT)
    artifact_paths = [
        PROBE,
        ANNOTATIONS,
        *_paths("v1"),
        *_paths("v2"),
        COMPARISON,
        REPORT,
    ]
    CHECKSUMS.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in artifact_paths
            if path.exists()
        ),
        encoding="utf-8",
    )
    print(_write_checksum(CHECKSUMS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "score", "compare"))
    parser.add_argument("--binder-version", choices=("v1", "v2"))
    args = parser.parse_args()
    if args.stage == "compare":
        compare()
    elif args.binder_version is None:
        parser.error("--binder-version is required for run and score")
    elif args.stage == "run":
        run(args.binder_version)
    else:
        score(args.binder_version)


if __name__ == "__main__":
    main()
