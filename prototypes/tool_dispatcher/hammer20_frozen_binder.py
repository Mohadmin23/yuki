"""Apply deterministic source binding to frozen Hammer 2.0 7B outputs.

The prepare/bind boundary deliberately removes benchmark gold before the binder
runs. Scoring is a separate command and cannot influence frozen predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from .argument_scoring import score_arguments
from .benchmark import write_report
from .deterministic_binding_experiment import run as run_binder
from .registry import ToolRegistry

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "hammer20-7b" / "binder-ontology-repair"
FROZEN_REPORT = (
    ROOT
    / "reports"
    / "hammer20-7b"
    / "direct-native"
    / "hammer20-7b-8bit-all18-native-cached-full450-v1.json"
)
PHASE2_DATASET = ROOT / "phase2-cases.json"
ANNOTATIONS = (
    ROOT
    / "reports"
    / "deterministic-binding"
    / "literal-source-annotations-targeted100-recall25-v1.json"
)

FROZEN_SHA256 = {
    FROZEN_REPORT: "97980e3a53d7e105046dda2a4bd6c982682c047c5dace89a55f5d77fd891cb32",
    PHASE2_DATASET: "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466",
    ANNOTATIONS: "9c33e6a9a2881bf44379ce1cb1ce7ac9367d25e542a1e203ac052f6f25d647f0",
}

BINDER_INPUT = REPORT_ROOT / "hammer20-7b-frozen450-binder-input-v1.json"
BINDER_PREDICTIONS = REPORT_ROOT / "hammer20-7b-frozen450-binder-predictions-v1.json"
BINDER_AMBIGUITIES = REPORT_ROOT / "hammer20-7b-frozen450-binder-abstentions-v1.jsonl"
BINDER_FREEZE = REPORT_ROOT / "hammer20-7b-frozen450-binder-freeze-v1.json"
BINDER_COMPARISON = REPORT_ROOT / "HAMMER20-7B-FROZEN450-BINDER-COMPARISON-v1.json"
BINDER_CASES = REPORT_ROOT / "hammer20-7b-frozen450-binder-cases-v1.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n", encoding="utf-8"
    )
    return checksum


def _verify_frozen() -> dict[str, str]:
    observed = {str(path): _sha256(path) for path in FROZEN_SHA256}
    mismatches = {
        str(path): {"expected": expected, "actual": observed[str(path)]}
        for path, expected in FROZEN_SHA256.items()
        if observed[str(path)] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen Hammer binder inputs changed: {mismatches}")
    return observed


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: {actual} != {expected}")
    return actual


def _binder_case(case: dict[str, Any]) -> dict[str, Any]:
    generation = case["generation"]
    return {
        "case_id": case["case_id"],
        "split": "frozen_full_450",
        "raw_request": case["request"],
        "raw_request_sha256": hashlib.sha256(case["request"].encode()).hexdigest(),
        "semantic_delegation": {
            "request": case["request"],
            "domain_hint": None,
            "verbatim": [],
        },
        "selected_call": deepcopy(case["parse"]["canonical_call"]),
        "hammer_record": {
            "model_id": generation["model_id"],
            "raw_generation": generation["raw_text"],
            "parse": deepcopy(case["parse"]),
            "validation": deepcopy(case["validation"]),
            "offered_tools": deepcopy(case["offered_tools"]),
            "native_dialect": case["native_dialect"],
            "execution": deepcopy(case["execution"]),
        },
    }


def prepare() -> None:
    observed = _verify_frozen()
    report = _load(FROZEN_REPORT)
    cases = [_binder_case(case) for case in report["case_results"]]
    if len(cases) != 450 or len({case["case_id"] for case in cases}) != 450:
        raise RuntimeError("Frozen Hammer report is not the expected 450 unique cases")
    encoded = json.dumps(cases, ensure_ascii=False)
    forbidden = (
        '"expected_tool"',
        '"expected_arguments"',
        '"strict_arguments_correct"',
        '"exact_call"',
        '"literal_annotation"',
    )
    leaked = [token for token in forbidden if token in encoded]
    if leaked:
        raise RuntimeError(f"Gold/scoring data leaked into binder input: {leaked}")
    if any(case["hammer_record"]["execution"]["executed"] for case in cases):
        raise RuntimeError("Frozen report unexpectedly contains Yuki tool execution")

    payload = {
        "version": "hammer20-7b-frozen450-binder-input-v1",
        "status": "gold_and_literal_annotation_free",
        "configuration": {
            "case_count": 450,
            "models_called": False,
            "hammer_loaded": False,
            "hammer_generations_regenerated": False,
            "new_sampling": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "source_reports": observed,
        "splits": {"frozen_full_450": [case["case_id"] for case in cases]},
        "cases": cases,
    }
    write_report(payload, BINDER_INPUT)
    print(_write_checksum(BINDER_INPUT))


def bind() -> None:
    _verify_frozen()
    _verify_sidecar(BINDER_INPUT)
    run_binder(
        BINDER_INPUT,
        BINDER_PREDICTIONS,
        BINDER_AMBIGUITIES,
        version="hammer20-7b-frozen450-binder-predictions-v1",
    )


def freeze() -> None:
    observed = _verify_frozen()
    input_sha = _verify_sidecar(BINDER_INPUT)
    prediction_sha = _verify_sidecar(BINDER_PREDICTIONS)
    abstention_sha = _verify_sidecar(BINDER_AMBIGUITIES)
    policy_path = ROOT / "source_span_policy.py"
    payload = {
        "version": "hammer20-7b-frozen450-binder-freeze-v1",
        "status": "frozen_before_scoring",
        "frozen_hammer_artifacts": observed,
        "binder_input": {"path": str(BINDER_INPUT), "sha256": input_sha},
        "binder_predictions": {
            "path": str(BINDER_PREDICTIONS),
            "sha256": prediction_sha,
        },
        "binder_abstentions": {
            "path": str(BINDER_AMBIGUITIES),
            "sha256": abstention_sha,
        },
        "binder_policy": {"path": str(policy_path), "sha256": _sha256(policy_path)},
        "safety": {
            "models_called": False,
            "hammer_loaded": False,
            "new_sampling": False,
            "gold_available_to_binder": False,
            "literal_annotations_available_to_binder": False,
            "yuki_tool_execution_capability": False,
            "yuki_tool_execution_attempts": 0,
        },
    }
    write_report(payload, BINDER_FREEZE)
    print(_write_checksum(BINDER_FREEZE))


def _literal_success(
    call: dict[str, Any] | None,
    annotation: dict[str, Any] | None,
    expected_tool: str,
) -> bool | None:
    if annotation is None or annotation["status"] == "ambiguous":
        return None
    if call is None or call.get("tool") != expected_tool:
        return False
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    if annotation["status"] == "no_argument":
        return arguments == {}
    value = arguments.get(annotation["argument_name"])
    return any(value == span["text"] for span in annotation["spans"])


def _final_call(
    baseline_call: dict[str, Any] | None, binding: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    status = binding["status"]
    if status in {"bound", "bound_composite", "no_argument"}:
        return binding["final_call"], "binder_replaced_arguments"
    if status == "unsupported_selected_tool":
        return baseline_call, "unsupported_tool_passthrough"
    return None, "binder_abstained"


def _score_case(
    baseline: dict[str, Any],
    prediction: dict[str, Any],
    annotation: dict[str, Any] | None,
    registry: ToolRegistry,
) -> dict[str, Any]:
    baseline_call = baseline["parse"]["canonical_call"]
    binder_call, binder_policy = _final_call(baseline_call, prediction["binding"])
    expected_tool = baseline["expected_tool"]
    actual_tool = binder_call.get("tool") if binder_call else None
    actual_arguments = binder_call.get("arguments") if binder_call else None
    tool_correct = actual_tool == expected_tool
    argument_scoring = score_arguments(
        expected_tool, baseline["expected_arguments"], actual_arguments
    )
    validation = (
        registry.validate_call(binder_call, baseline["offered_tools"])
        if binder_call is not None
        else {"passed": False, "errors": ["Deterministic binder abstained."]}
    )
    schema_valid = bool(validation["passed"])
    strict_exact = tool_correct and argument_scoring["strict"] and schema_valid
    equivalent_exact = (
        tool_correct and argument_scoring["execution_equivalent"] and schema_valid
    )
    baseline_literal = _literal_success(baseline_call, annotation, expected_tool)
    binder_literal = _literal_success(binder_call, annotation, expected_tool)
    result = {
        "case_id": baseline["case_id"],
        "request": baseline["request"],
        "expected_tool": expected_tool,
        "expected_arguments": baseline["expected_arguments"],
        "baseline_call": baseline_call,
        "binder_status": prediction["binding"]["status"],
        "binder_policy": binder_policy,
        "binder_candidates": prediction["binding"]["candidates"],
        "binder_call": binder_call,
        "binder_validation": validation,
        "baseline_tool_correct": baseline["tool_correct"],
        "binder_tool_correct": tool_correct,
        "baseline_schema_valid": baseline["schema_valid"],
        "binder_schema_valid": schema_valid,
        "baseline_strict_arguments": baseline["strict_arguments_correct"],
        "binder_strict_arguments": tool_correct and argument_scoring["strict"],
        "baseline_execution_equivalent_arguments": baseline[
            "execution_equivalent_arguments_correct"
        ],
        "binder_execution_equivalent_arguments": (
            tool_correct and argument_scoring["execution_equivalent"]
        ),
        "baseline_strict_exact": baseline["exact_call"],
        "binder_strict_exact": strict_exact,
        "baseline_execution_equivalent_exact": baseline[
            "execution_equivalent_call"
        ],
        "binder_execution_equivalent_exact": equivalent_exact,
        "baseline_literal_source_exact": baseline_literal,
        "binder_literal_source_exact": binder_literal,
        "literal_annotation_available": annotation is not None,
        "source_copy_character_exact": prediction["binding"].get(
            "source_copy_valid"
        ),
        "argument_scoring": argument_scoring,
    }
    result["official_repaired"] = not result["baseline_strict_exact"] and strict_exact
    result["official_broken"] = result["baseline_strict_exact"] and not strict_exact
    result["literal_repaired"] = baseline_literal is False and binder_literal is True
    result["literal_broken"] = baseline_literal is True and binder_literal is False
    return result


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"cases": 0}
    literal_cases = [
        case
        for case in cases
        if case["baseline_literal_source_exact"] is not None
    ]
    source_copy = [
        case["source_copy_character_exact"]
        for case in cases
        if case["source_copy_character_exact"] is not None
    ]
    return {
        "cases": len(cases),
        "baseline_tool_selection_accuracy": mean(
            case["baseline_tool_correct"] for case in cases
        ),
        "binder_tool_selection_accuracy": mean(
            case["binder_tool_correct"] for case in cases
        ),
        "baseline_strict_argument_accuracy": mean(
            case["baseline_strict_arguments"] for case in cases
        ),
        "binder_strict_argument_accuracy": mean(
            case["binder_strict_arguments"] for case in cases
        ),
        "baseline_execution_equivalent_argument_accuracy": mean(
            case["baseline_execution_equivalent_arguments"] for case in cases
        ),
        "binder_execution_equivalent_argument_accuracy": mean(
            case["binder_execution_equivalent_arguments"] for case in cases
        ),
        "baseline_strict_exact_accuracy": mean(
            case["baseline_strict_exact"] for case in cases
        ),
        "binder_strict_exact_accuracy": mean(
            case["binder_strict_exact"] for case in cases
        ),
        "baseline_execution_equivalent_exact_accuracy": mean(
            case["baseline_execution_equivalent_exact"] for case in cases
        ),
        "binder_execution_equivalent_exact_accuracy": mean(
            case["binder_execution_equivalent_exact"] for case in cases
        ),
        "baseline_schema_valid_rate": mean(
            case["baseline_schema_valid"] for case in cases
        ),
        "binder_schema_valid_rate": mean(
            case["binder_schema_valid"] for case in cases
        ),
        "binder_coverage_rate": mean(
            case["binder_status"] in {"bound", "bound_composite", "no_argument"}
            for case in cases
        ),
        "binder_abstention_count": sum(
            case["binder_policy"] == "binder_abstained" for case in cases
        ),
        "binder_ambiguous_count": sum(
            case["binder_status"] == "ambiguous" for case in cases
        ),
        "official_repaired_count": sum(case["official_repaired"] for case in cases),
        "official_broken_count": sum(case["official_broken"] for case in cases),
        "literal_reference_cases": len(literal_cases),
        "baseline_literal_source_exact_accuracy": (
            mean(case["baseline_literal_source_exact"] for case in literal_cases)
            if literal_cases
            else None
        ),
        "binder_literal_source_exact_accuracy": (
            mean(case["binder_literal_source_exact"] for case in literal_cases)
            if literal_cases
            else None
        ),
        "literal_repaired_count": sum(case["literal_repaired"] for case in cases),
        "literal_broken_count": sum(case["literal_broken"] for case in cases),
        "bound_source_copy_character_exact_rate": (
            mean(source_copy) if source_copy else None
        ),
    }


def score() -> None:
    frozen = _verify_frozen()
    freeze_sha = _verify_sidecar(BINDER_FREEZE)
    freeze_manifest = _load(BINDER_FREEZE)
    predictions_sha = _verify_sidecar(BINDER_PREDICTIONS)
    if predictions_sha != freeze_manifest["binder_predictions"]["sha256"]:
        raise RuntimeError("Binder predictions differ from the pre-score freeze")

    baseline = _load(FROZEN_REPORT)
    predictions = _load(BINDER_PREDICTIONS)
    annotations = _load(ANNOTATIONS)["annotations"]
    baseline_by_id = {case["case_id"]: case for case in baseline["case_results"]}
    prediction_by_id = {
        case["case_id"]: case for case in predictions["case_results"]
    }
    if baseline_by_id.keys() != prediction_by_id.keys():
        raise RuntimeError("Frozen Hammer and binder prediction IDs differ")

    registry = ToolRegistry()
    results = [
        _score_case(
            baseline_by_id[case_id],
            prediction_by_id[case_id],
            annotations.get(case_id),
            registry,
        )
        for case_id in baseline_by_id
    ]
    focused_44 = [
        case
        for case in results
        if case["baseline_tool_correct"] and not case["baseline_strict_arguments"]
    ]
    if len(focused_44) != 44:
        raise RuntimeError(f"Expected 44 correct-tool argument failures, got {len(focused_44)}")

    by_tool = {
        tool: _metrics([case for case in results if case["expected_tool"] == tool])
        for tool in sorted({case["expected_tool"] for case in results})
    }
    status_counts = Counter(case["binder_status"] for case in results)
    focused_status_counts = Counter(case["binder_status"] for case in focused_44)
    comparison = {
        "version": "hammer20-7b-frozen450-binder-comparison-v1",
        "status": "completed_offline_without_inference",
        "frozen_inputs": frozen,
        "binder_freeze_sha256": freeze_sha,
        "safety": {
            "models_called": False,
            "hammer_loaded": False,
            "hammer_generations_regenerated": False,
            "new_sampling": False,
            "official_gold_mutated": False,
            "gold_available_to_binder": False,
            "literal_annotations_available_to_binder": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "application_policy": {
            "bound": "replace the selected call argument with literal source span(s)",
            "unsupported_selected_tool": "preserve the frozen Hammer call",
            "unbound_or_ambiguous": "abstain rather than guess",
            "composite_yuki_payload": (
                "copy filename and content source spans and join them with the fixed "
                "schema separator |"
            ),
        },
        "all_450": _metrics(results),
        "correct_tool_wrong_argument_44": _metrics(focused_44),
        "by_expected_tool": by_tool,
        "binder_status_counts": dict(status_counts),
        "focused_44_binder_status_counts": dict(focused_status_counts),
        "case_lists": {
            "official_repaired": [
                case["case_id"] for case in results if case["official_repaired"]
            ],
            "official_broken": [
                case["case_id"] for case in results if case["official_broken"]
            ],
            "literal_repaired": [
                case["case_id"] for case in results if case["literal_repaired"]
            ],
            "literal_broken": [
                case["case_id"] for case in results if case["literal_broken"]
            ],
            "binder_abstained": [
                case["case_id"]
                for case in results
                if case["binder_policy"] == "binder_abstained"
            ],
        },
        "focused_44_cases": focused_44,
    }
    write_report(comparison, BINDER_COMPARISON)
    BINDER_CASES.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in results),
        encoding="utf-8",
    )
    print(_write_checksum(BINDER_COMPARISON))
    print(_write_checksum(BINDER_CASES))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline deterministic binder for frozen Hammer 2.0 7B outputs"
    )
    parser.add_argument("command", choices=("prepare", "bind", "freeze", "score"))
    return parser


def main() -> None:
    command = _parser().parse_args().command
    {"prepare": prepare, "bind": bind, "freeze": freeze, "score": score}[command]()


if __name__ == "__main__":
    main()
