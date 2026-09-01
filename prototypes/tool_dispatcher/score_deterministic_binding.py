"""Score frozen binder predictions against official and literal references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .benchmark import write_report
from .delegation_dispatcher import _preservation_aware_arguments
from .registry import ToolRegistry

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "deterministic-binding"
PREDICTIONS = REPORT_ROOT / "deterministic-binding-targeted100-recall25-v1.json"
ANNOTATIONS = REPORT_ROOT / "literal-source-annotations-targeted100-recall25-v1.json"
PAYLOAD_BASELINE = (
    ROOT / "reports" / "focused-repair" / "hammer15-verbatim-old-dispatch-v1.json"
)
SEE_RECALL_BASELINE = (
    ROOT
    / "reports"
    / "focused-repair"
    / "hammer15-see-recall-schema-repair-v1.json"
)
OUTPUT = REPORT_ROOT / "DETERMINISTIC-BINDING-COMPARISON-v1.json"
CASES_OUTPUT = REPORT_ROOT / "deterministic-binding-evaluation-cases-v1.jsonl"

FROZEN_SHA256 = {
    PREDICTIONS: "4da47871fe461530b2182ffc0d089ee2477f30cd0750b7489a816e1758550db4",
    ANNOTATIONS: "9c33e6a9a2881bf44379ce1cb1ce7ac9367d25e542a1e203ac052f6f25d647f0",
    PAYLOAD_BASELINE: "24cee3ce9ae8f0c85d53174fbc8f13b6777281cf080e9323b6564a37ad929f38",
    SEE_RECALL_BASELINE: "513df715b51102b4d1ce9aee408f935e6ae78d4e2c42aab9311cf3bf553a0c90",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify() -> dict[str, str]:
    observed = {str(path): _sha256(path) for path in FROZEN_SHA256}
    mismatches = {
        str(path): {"expected": expected, "actual": observed[str(path)]}
        for path, expected in FROZEN_SHA256.items()
        if observed[str(path)] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen scoring inputs changed: {mismatches}")
    return observed


def _literal_success(
    call: dict[str, Any] | None,
    annotation: dict[str, Any],
    expected_tool: str,
) -> bool | None:
    if annotation["status"] == "ambiguous":
        return None
    if call is None or call.get("tool") != expected_tool:
        return False
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False
    if annotation["status"] == "no_argument":
        return arguments == {}
    argument_name = annotation["argument_name"]
    value = arguments.get(argument_name)
    return any(value == span["text"] for span in annotation["spans"])


def _binder_source_invariant(case: dict[str, Any]) -> bool | None:
    binding = case["binding"]
    if binding["status"] != "bound":
        return None
    span = binding["source_span"]
    raw = case["raw_request"]
    value = binding["final_call"]["arguments"][binding["argument_name"]]
    return (
        value == span["text"]
        and value == raw[span["start"] : span["end"]]
        and binding["source_copy_valid"] is True
    )


def _score_case(
    baseline: dict[str, Any],
    prediction: dict[str, Any],
    annotation: dict[str, Any],
    registry: ToolRegistry,
) -> dict[str, Any]:
    expected_tool = baseline["expected_tool"]
    expected_arguments = baseline["expected_arguments"]
    baseline_call = baseline["dispatcher"]["canonical_call"]
    binder_call = prediction["binding"]["final_call"]
    binder_tool = binder_call.get("tool") if binder_call else None
    binder_arguments = binder_call.get("arguments") if binder_call else None
    binder_tool_correct = binder_tool == expected_tool
    argument_score = _preservation_aware_arguments(
        expected_tool,
        expected_arguments,
        binder_arguments,
        baseline["argument_alternatives"],
    )
    offered_tools = baseline["dispatcher"]["offered_tools"]
    validation = (
        registry.validate_call(binder_call, offered_tools)
        if binder_call is not None
        else {"passed": False, "errors": ["Binder abstained."]}
    )
    schema_valid = bool(validation["passed"])
    strict_arguments = binder_tool_correct and argument_score["strict"]
    preservation_arguments = (
        binder_tool_correct and argument_score["preservation_aware"]
    )
    baseline_literal = _literal_success(baseline_call, annotation, expected_tool)
    binder_literal = _literal_success(binder_call, annotation, expected_tool)

    official_literal_conflict = False
    official_not_source_span = False
    if annotation["status"] == "reference":
        argument_name = annotation["argument_name"]
        official_value = expected_arguments.get(argument_name)
        literal_values = [span["text"] for span in annotation["spans"]]
        official_literal_conflict = official_value not in literal_values
        official_not_source_span = (
            not isinstance(official_value, str)
            or official_value not in baseline["request"]
        )

    result = {
        "case_id": baseline["case_id"],
        "split": prediction["split"],
        "expected_tool": expected_tool,
        "raw_request": baseline["request"],
        "official_arguments": expected_arguments,
        "literal_annotation": annotation,
        "baseline_call": baseline_call,
        "binder_status": prediction["binding"]["status"],
        "binder_candidates": prediction["binding"]["candidates"],
        "binder_call": binder_call,
        "pre_binding_tool_correct": baseline["tool_correct"],
        "baseline_schema_valid": baseline["schema_valid"],
        "binder_schema_valid": schema_valid,
        "binder_validation": validation,
        "baseline_official_strict_arguments": baseline[
            "strict_arguments_correct"
        ],
        "binder_official_strict_arguments": strict_arguments,
        "baseline_official_preservation_arguments": baseline[
            "preservation_aware_arguments_correct"
        ],
        "binder_official_preservation_arguments": preservation_arguments,
        "baseline_official_strict_exact": baseline["strict_exact_call"],
        "binder_official_strict_exact": (
            binder_tool_correct and strict_arguments and schema_valid
        ),
        "baseline_official_preservation_exact": baseline[
            "preservation_aware_exact_call"
        ],
        "binder_official_preservation_exact": (
            binder_tool_correct and preservation_arguments and schema_valid
        ),
        "baseline_literal_source_exact": baseline_literal,
        "binder_literal_source_exact": binder_literal,
        "binder_source_copy_invariant": _binder_source_invariant(prediction),
        "official_literal_conflict": official_literal_conflict,
        "official_not_contiguous_source_span": official_not_source_span,
    }
    result["official_repaired"] = (
        not result["baseline_official_strict_exact"]
        and result["binder_official_strict_exact"]
    )
    result["official_broken"] = (
        result["baseline_official_strict_exact"]
        and not result["binder_official_strict_exact"]
    )
    result["literal_repaired"] = baseline_literal is False and binder_literal is True
    result["literal_broken"] = baseline_literal is True and binder_literal is False
    return result


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    literal_eligible = [
        case for case in cases if case["literal_annotation"]["status"] != "ambiguous"
    ]
    tool_correct = [case for case in cases if case["pre_binding_tool_correct"]]
    tool_correct_literal_eligible = [
        case
        for case in tool_correct
        if case["literal_annotation"]["status"] != "ambiguous"
    ]
    invariants = [
        case["binder_source_copy_invariant"]
        for case in cases
        if case["binder_source_copy_invariant"] is not None
    ]
    return {
        "cases": len(cases),
        "tool_selection_accuracy": mean(
            case["pre_binding_tool_correct"] for case in cases
        ),
        "baseline_official_strict_argument_accuracy": mean(
            case["baseline_official_strict_arguments"] for case in cases
        ),
        "binder_official_strict_argument_accuracy": mean(
            case["binder_official_strict_arguments"] for case in cases
        ),
        "baseline_official_preservation_aware_argument_accuracy": mean(
            case["baseline_official_preservation_arguments"] for case in cases
        ),
        "binder_official_preservation_aware_argument_accuracy": mean(
            case["binder_official_preservation_arguments"] for case in cases
        ),
        "baseline_official_strict_exact_accuracy": mean(
            case["baseline_official_strict_exact"] for case in cases
        ),
        "binder_official_strict_exact_accuracy": mean(
            case["binder_official_strict_exact"] for case in cases
        ),
        "baseline_official_preservation_aware_exact_accuracy": mean(
            case["baseline_official_preservation_exact"] for case in cases
        ),
        "binder_official_preservation_aware_exact_accuracy": mean(
            case["binder_official_preservation_exact"] for case in cases
        ),
        "baseline_schema_valid_rate": mean(
            case["baseline_schema_valid"] for case in cases
        ),
        "binder_schema_valid_rate": mean(case["binder_schema_valid"] for case in cases),
        "baseline_literal_source_exact_conservative": mean(
            case["baseline_literal_source_exact"] is True for case in cases
        ),
        "binder_literal_source_exact_conservative": mean(
            case["binder_literal_source_exact"] is True for case in cases
        ),
        "literal_reference_eligible_cases": len(literal_eligible),
        "baseline_literal_source_exact_on_eligible": mean(
            case["baseline_literal_source_exact"] is True for case in literal_eligible
        ),
        "binder_literal_source_exact_on_eligible": mean(
            case["binder_literal_source_exact"] is True for case in literal_eligible
        ),
        "tool_correct_cases": len(tool_correct),
        "baseline_literal_source_exact_when_tool_correct": mean(
            case["baseline_literal_source_exact"] is True for case in tool_correct
        ),
        "binder_literal_source_exact_when_tool_correct": mean(
            case["binder_literal_source_exact"] is True for case in tool_correct
        ),
        "tool_correct_literal_reference_eligible_cases": len(
            tool_correct_literal_eligible
        ),
        "binder_literal_source_exact_when_tool_correct_and_eligible": mean(
            case["binder_literal_source_exact"] is True
            for case in tool_correct_literal_eligible
        ),
        "binder_bound_string_count": len(invariants),
        "binder_raw_slice_invariant_rate": mean(invariants) if invariants else None,
        "binder_final_call_available_rate": mean(
            case["binder_call"] is not None for case in cases
        ),
        "annotation_ambiguous_count": sum(
            case["literal_annotation"]["status"] == "ambiguous" for case in cases
        ),
        "binder_ambiguous_count": sum(
            case["binder_status"] == "ambiguous" for case in cases
        ),
        "binder_unbound_or_no_call_count": sum(
            case["binder_call"] is None for case in cases
        ),
        "official_literal_conflict_count": sum(
            case["official_literal_conflict"] for case in cases
        ),
        "official_not_contiguous_source_span_count": sum(
            case["official_not_contiguous_source_span"] for case in cases
        ),
        "official_repaired_count": sum(case["official_repaired"] for case in cases),
        "official_broken_count": sum(case["official_broken"] for case in cases),
        "literal_repaired_count": sum(case["literal_repaired"] for case in cases),
        "literal_broken_count": sum(case["literal_broken"] for case in cases),
    }


def main() -> None:
    observed = _verify()
    predictions = _load(PREDICTIONS)
    if predictions["status"] != "predictions_frozen_before_scoring":
        raise RuntimeError("Binder predictions were not frozen before scoring")
    annotations = _load(ANNOTATIONS)
    payload = _load(PAYLOAD_BASELINE)
    media_memory = _load(SEE_RECALL_BASELINE)
    baseline_cases = {
        case["case_id"]: case
        for report in (payload, media_memory)
        for case in report["case_results"]
    }
    prediction_cases = {
        case["case_id"]: case for case in predictions["case_results"]
    }
    annotation_cases = annotations["annotations"]
    if prediction_cases.keys() != annotation_cases.keys():
        raise RuntimeError("Prediction and literal annotation IDs differ")

    registry = ToolRegistry()
    results = [
        _score_case(
            baseline_cases[case_id],
            prediction_cases[case_id],
            annotation_cases[case_id],
            registry,
        )
        for case_id in prediction_cases
    ]
    primary = [case for case in results if case["split"] == "primary_targeted_100"]
    recall = [case for case in results if case["split"] == "recall_diagnostic_25"]
    by_tool = {
        tool: _metrics([case for case in results if case["expected_tool"] == tool])
        for tool in ("search", "image", "ask_claude", "see", "recall")
    }
    noteworthy = [
        case
        for case in results
        if case["official_repaired"]
        or case["official_broken"]
        or case["literal_repaired"]
        or case["literal_broken"]
        or case["literal_annotation"]["status"] == "ambiguous"
        or case["binder_status"] in {"ambiguous", "unbound"}
    ]
    comparison = {
        "version": "deterministic-binding-comparison-v1",
        "status": "completed",
        "frozen_inputs": observed,
        "safety": {
            "models_called": False,
            "hammer_3b_loaded": False,
            "qwen_prompt_changed": False,
            "delegations_regenerated": False,
            "official_gold_mutated": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
            "binder_had_literal_annotation_access": False,
            "predictions_frozen_before_scoring": True,
        },
        "primary_targeted_100": _metrics(primary),
        "recall_diagnostic_25": _metrics(recall),
        "by_tool": by_tool,
        "case_lists": {
            name: [case["case_id"] for case in results if case[name]]
            for name in (
                "official_repaired",
                "official_broken",
                "literal_repaired",
                "literal_broken",
                "official_literal_conflict",
                "official_not_contiguous_source_span",
            )
        },
        "annotation_ambiguous_cases": [
            {
                "case_id": case["case_id"],
                "reason": case["literal_annotation"]["reason"],
                "spans": case["literal_annotation"]["spans"],
                "binder_status": case["binder_status"],
            }
            for case in results
            if case["literal_annotation"]["status"] == "ambiguous"
        ],
        "noteworthy_cases": noteworthy,
    }
    write_report(comparison, OUTPUT)
    CASES_OUTPUT.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in results),
        encoding="utf-8",
    )
    for path in (OUTPUT, CASES_OUTPUT):
        checksum = _sha256(path)
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{checksum}  {path.name}\n", encoding="utf-8"
        )
        print(checksum)


if __name__ == "__main__":
    main()
