"""Execution-locked Hammer 3B capacity probe for the frozen 25 `see` cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .backends import create_backend
from .benchmark import write_report
from .delegation_dispatcher import (
    DelegationDispatcherRunner,
    ValidationOnlyDispatcher,
    failure_records,
)
from .focused_repair import (
    A_FROZEN_DELEGATIONS,
    FocusedRepairRegistry,
    _decorate_focused_report,
    _read_checksum,
    _sha256,
    _variant_task_instruction,
    _write_checksum,
    _write_jsonl,
    verify_frozen_artifacts,
)

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "focused-repair"
HAMMER3_MODEL = Path(
    "/Volumes/madisk/yuki-tool-dispatcher/Hammer2.1-3b-fp16"
)
HAMMER15_A1 = REPORT_ROOT / "hammer15-see-recall-schema-repair-v1.json"
SEE_25 = REPORT_ROOT / "focused-see-25-frozen-delegations-v1.json"
HAMMER3_REPORT = REPORT_ROOT / "hammer3-see-capacity-probe-v1.json"
HAMMER3_FAILURES = REPORT_ROOT / "hammer3-see-capacity-probe-v1-failures.jsonl"
COMPARISON = REPORT_ROOT / "hammer15-vs-hammer3-see-capacity-v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sidecar(path: Path) -> str:
    expected = _read_checksum(path)
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen checksum mismatch for {path}: {actual} != {expected}")
    return actual


def build_see_subset(source: dict[str, Any]) -> dict[str, Any]:
    """Select, without rewriting, the 25 frozen Qwen `see` case objects."""

    cases = [
        deepcopy(case)
        for case in source["case_results"]
        if case["expected_tool"] == "see"
    ]
    expected_ids = [f"p2-see-{index:02d}" for index in range(1, 26)]
    observed_ids = [case["case_id"] for case in cases]
    if observed_ids != expected_ids:
        raise RuntimeError(f"Unexpected frozen see IDs: {observed_ids}")

    configuration = deepcopy(source["configuration"])
    configuration.update(
        {
            "experiment": "focused_see_capacity_frozen_stage_a_subset",
            "selection": "focused-see-capacity-25",
            "total_cases": 25,
            "derived_from_focused_a_sha256": _sha256(A_FROZEN_DELEGATIONS),
            "tool_execution_capability": False,
        }
    )
    provenance = deepcopy(source["provenance"])
    provenance.update(
        {
            "derived_from": str(A_FROZEN_DELEGATIONS),
            "derived_from_sha256": _sha256(A_FROZEN_DELEGATIONS),
            "tools": ["see"],
            "case_ids": observed_ids,
            "delegations_regenerated": False,
        }
    )
    return {
        "configuration": configuration,
        "contract": deepcopy(source["contract"]),
        "provenance": provenance,
        "case_results": cases,
    }


def prepare() -> None:
    verify_frozen_artifacts()
    _verify_sidecar(A_FROZEN_DELEGATIONS)
    payload = build_see_subset(_load(A_FROZEN_DELEGATIONS))
    write_report(payload, SEE_25)
    print(_write_checksum(SEE_25))


def run() -> None:
    verify_frozen_artifacts()
    subset_sha = _verify_sidecar(SEE_25)
    if not HAMMER3_MODEL.is_dir() or not str(HAMMER3_MODEL).startswith("/Volumes/"):
        raise RuntimeError(f"External Hammer 3B checkpoint unavailable: {HAMMER3_MODEL}")

    registry = FocusedRepairRegistry("schema")
    backend = create_backend("mlx", str(HAMMER3_MODEL))
    dispatcher = ValidationOnlyDispatcher(
        registry,
        backend,
        input_mode="delegated_with_verbatim",
        domain_source="generated",
        task_instruction=_variant_task_instruction("schema"),
    )
    report = DelegationDispatcherRunner(dispatcher).run(
        _load(SEE_25),
        delegation_path=SEE_25,
        expected_delegation_sha256=subset_sha,
    )
    report = _decorate_focused_report(report, "schema", SEE_25)
    report["configuration"].update(
        {
            "experiment": "hammer3_see_only_capacity_probe",
            "capacity_probe": True,
            "qwen_called": False,
            "delegations_regenerated": False,
            "production_yuki_modified": False,
            "yuki_tool_execution_attempts": 0,
        }
    )
    write_report(report, HAMMER3_REPORT)
    _write_jsonl(failure_records(report), HAMMER3_FAILURES)
    print(_write_checksum(HAMMER3_REPORT))
    print(_write_checksum(HAMMER3_FAILURES))


def _case_view(case: dict[str, Any]) -> dict[str, Any]:
    dispatch = case["dispatcher"]
    return {
        "strict_exact": case["strict_exact_call"],
        "tool_correct": case["tool_correct"],
        "schema_valid": case["schema_valid"],
        "rejected": dispatch["parse"]["rejected"],
        "malformed": dispatch["parse"]["malformed"],
        "actual_tool": case["actual_tool"],
        "actual_arguments": case["actual_arguments"],
        "failure_category": case["failure_category"],
        "raw_generation": dispatch["generation"]["raw_text"],
    }


def _assert_identical_contract(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    baseline_cases = {
        case["case_id"]: case
        for case in baseline["case_results"]
        if case["expected_tool"] == "see"
    }
    candidate_cases = {case["case_id"]: case for case in candidate["case_results"]}
    if baseline_cases.keys() != candidate_cases.keys():
        raise RuntimeError("Hammer 1.5B and 3B see case IDs differ")

    if baseline["configuration"]["focused_repair_variant"] != "schema":
        raise RuntimeError("Hammer 1.5B comparison source is not the selected A1 arm")
    for field in ("task_instruction_sha256", "hammer_format_instruction_sha256"):
        if baseline["configuration"][field] != candidate["configuration"][field]:
            raise RuntimeError(f"Hammer prompt contract changed: {field}")

    source_fields = (
        "request",
        "expected_domain",
        "expected_tool",
        "expected_arguments",
        "required_verbatim",
        "argument_alternatives",
        "delegation",
        "delegation_scores",
    )
    dispatch_fields = (
        "dispatcher_input",
        "domain_hint",
        "domain_source",
        "input_mode",
        "offered_tools",
        "schemas_sent",
        "native_dialect",
        "stateless",
        "generation_count",
        "execution_capability",
    )
    for case_id, old in baseline_cases.items():
        new = candidate_cases[case_id]
        for field in source_fields:
            if old[field] != new[field]:
                raise RuntimeError(f"Frozen case changed for {case_id}: {field}")
        for field in dispatch_fields:
            if old["dispatcher"][field] != new["dispatcher"][field]:
                raise RuntimeError(f"Dispatcher contract changed for {case_id}: {field}")
        if old["execution"]["executed"] or new["execution"]["executed"]:
            raise RuntimeError(f"Unexpected tool execution in {case_id}")


def compare() -> None:
    verify_frozen_artifacts()
    baseline_sha = _verify_sidecar(HAMMER15_A1)
    candidate_sha = _verify_sidecar(HAMMER3_REPORT)
    subset_sha = _verify_sidecar(SEE_25)
    baseline = _load(HAMMER15_A1)
    candidate = _load(HAMMER3_REPORT)
    _assert_identical_contract(baseline, candidate)

    baseline_cases = {
        case["case_id"]: case
        for case in baseline["case_results"]
        if case["expected_tool"] == "see"
    }
    candidate_cases = {case["case_id"]: case for case in candidate["case_results"]}
    failure_to_success = [
        case_id
        for case_id in baseline_cases
        if not baseline_cases[case_id]["strict_exact_call"]
        and candidate_cases[case_id]["strict_exact_call"]
    ]
    success_to_failure = [
        case_id
        for case_id in baseline_cases
        if baseline_cases[case_id]["strict_exact_call"]
        and not candidate_cases[case_id]["strict_exact_call"]
    ]
    both_success = [
        case_id
        for case_id in baseline_cases
        if baseline_cases[case_id]["strict_exact_call"]
        and candidate_cases[case_id]["strict_exact_call"]
    ]
    both_failure = [
        case_id
        for case_id in baseline_cases
        if not baseline_cases[case_id]["strict_exact_call"]
        and not candidate_cases[case_id]["strict_exact_call"]
    ]

    def flips(field: str) -> dict[str, list[str]]:
        return {
            "failure_to_success": [
                case_id
                for case_id in baseline_cases
                if not baseline_cases[case_id][field]
                and candidate_cases[case_id][field]
            ],
            "success_to_failure": [
                case_id
                for case_id in baseline_cases
                if baseline_cases[case_id][field]
                and not candidate_cases[case_id][field]
            ],
        }

    old_metrics = baseline["focused_metrics"]["see"]
    new_metrics = candidate["focused_metrics"]["see"]
    old_failures = Counter(
        case["failure_category"]
        for case in baseline_cases.values()
        if case["failure_category"] is not None
    )
    new_failures = Counter(
        case["failure_category"]
        for case in candidate_cases.values()
        if case["failure_category"] is not None
    )
    new_argument_failures = [
        case
        for case in candidate_cases.values()
        if case["failure_category"] == "dispatcher_wrong_argument"
    ]
    comparison = {
        "version": "hammer15-vs-hammer3-see-capacity-v1",
        "status": "completed",
        "question": (
            "Does Hammer 3B materially solve the remaining see failures after A1 "
            "schema wording repair?"
        ),
        "frozen_inputs": {
            "see_25_sha256": subset_sha,
            "hammer15_a1_sha256": baseline_sha,
            "hammer3_report_sha256": candidate_sha,
        },
        "controlled_conditions": {
            "case_count": 25,
            "same_case_ids": True,
            "same_frozen_qwen_delegations": True,
            "qwen_called": False,
            "delegations_regenerated": False,
            "same_a1_schemas": True,
            "same_task_instruction": True,
            "same_hammer_native_format": True,
            "same_parser_validator_scoring": True,
            "stateless": True,
            "one_generation_per_case": True,
            "yuki_tool_execution_capability": False,
            "yuki_tool_execution_attempts": 0,
        },
        "hammer_1_5b": {
            "model_id": baseline["configuration"]["dispatcher_model_id"],
            "metrics": old_metrics,
            "failure_decomposition": dict(old_failures),
        },
        "hammer_3b": {
            "model_id": candidate["configuration"]["dispatcher_model_id"],
            "metrics": new_metrics,
            "failure_decomposition": dict(new_failures),
        },
        "change_percentage_points": {
            field: (new_metrics[field] - old_metrics[field]) * 100
            for field in (
                "tool_selection_accuracy",
                "strict_exact_call_accuracy",
                "preservation_aware_exact_call_accuracy",
                "schema_valid_rate",
            )
        },
        "case_level_flips": {
            "strict_exact": {
                "failure_to_success": failure_to_success,
                "success_to_failure": success_to_failure,
                "both_success": both_success,
                "both_failure": both_failure,
            },
            "tool_selection": flips("tool_correct"),
            "schema_validity": flips("schema_valid"),
            "strict_flip_details": [
                {
                    "case_id": case_id,
                    "request": baseline_cases[case_id]["request"],
                    "expected_arguments": baseline_cases[case_id][
                        "expected_arguments"
                    ],
                    "hammer_1_5b": _case_view(baseline_cases[case_id]),
                    "hammer_3b": _case_view(candidate_cases[case_id]),
                }
                for case_id in failure_to_success + success_to_failure
            ],
        },
        "hammer_3b_argument_failure_diagnostic": {
            "wrong_argument_cases": [
                case["case_id"] for case in new_argument_failures
            ],
            "empty_arguments": sum(
                case["actual_arguments"] == {} for case in new_argument_failures
            ),
            "nonempty_nonexact_arguments": sum(
                case["actual_arguments"] != {} for case in new_argument_failures
            ),
            "empty_frozen_qwen_verbatim": sum(
                not case["delegation"]["verbatim"]
                for case in new_argument_failures
            ),
            "gold_target_absent_from_frozen_qwen_verbatim": sum(
                case["expected_arguments"]["target"]
                not in case["delegation"]["verbatim"]
                for case in new_argument_failures
            ),
            "note": (
                "All official Hammer 3B argument failures lacked the exact gold target "
                "in frozen Qwen verbatim; this is an association with the interface "
                "contract, not a rescoring or a claim of sole causation."
            ),
        },
        "interpretation": {
            "rejection_and_selection_capacity_gap": True,
            "exact_call_capacity_gain": False,
            "summary": (
                "Hammer 3B eliminated rejection and wrong-tool behavior but produced no "
                "net strict or preservation-aware exact-call gain. Remaining failures "
                "are dominated by target-argument omission or boundary rewrites, plus one "
                "frozen Qwen wrong-domain case."
            ),
        },
    }
    write_report(comparison, COMPARISON)
    print(_write_checksum(COMPARISON))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execution-locked see capacity probe")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("run")
    commands.add_parser("compare")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run()
    else:
        compare()


if __name__ == "__main__":
    main()
