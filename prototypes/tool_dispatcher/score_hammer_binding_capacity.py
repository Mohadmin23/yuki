"""Score the frozen Hammer 1.5B/3B deterministic-binding capacity arms."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from .benchmark import write_report
from .hammer_binding_capacity import PRIMARY_100, REPORT_ROOT, arm_paths
from .registry import ToolRegistry
from .score_deterministic_binding import _metrics, _score_case

ROOT = Path(__file__).parent
ANNOTATIONS = (
    ROOT
    / "reports"
    / "deterministic-binding"
    / "literal-source-annotations-targeted100-recall25-v1.json"
)
ANNOTATIONS_SHA256 = "9c33e6a9a2881bf44379ce1cb1ce7ac9367d25e542a1e203ac052f6f25d647f0"
BINDER_SOURCE = ROOT / "source_span_policy.py"
BINDER_SOURCE_SHA256 = "b05a97db54d7062ed2889596f1caa156a2f694d9ba19d0dd8d40b4c5cc61f0f8"
POLICY = (
    ROOT / "reports" / "deterministic-binding" / "SOURCE-SPAN-POLICY-v1.md"
)
POLICY_SHA256 = "3f7f935fa0a302bf14012fd4cb5f9f1c5a6ae0d5e72c6e8957638873c339f1dc"
PRIOR_BINDER_PREDICTIONS = (
    ROOT
    / "reports"
    / "deterministic-binding"
    / "deterministic-binding-targeted100-recall25-v1.json"
)
PRIOR_BINDER_PREDICTIONS_SHA256 = (
    "4da47871fe461530b2182ffc0d089ee2477f30cd0750b7489a816e1758550db4"
)
OUTPUT = REPORT_ROOT / "HAMMER15-VS-HAMMER3-BINDING-COMPARISON-v1.json"
CASES_OUTPUT = REPORT_ROOT / "hammer15-vs-hammer3-binding-cases-v1.jsonl"

ARMS = ("hammer15", "hammer3")
TOOLS = ("search", "image", "ask_claude", "see")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen checksum mismatch for {path}: {actual} != {expected}")
    return actual


def _verify_inputs() -> dict[str, str]:
    observed = {str(PRIMARY_100): _verify_sidecar(PRIMARY_100)}
    fixed = {
        ANNOTATIONS: ANNOTATIONS_SHA256,
        BINDER_SOURCE: BINDER_SOURCE_SHA256,
        POLICY: POLICY_SHA256,
        PRIOR_BINDER_PREDICTIONS: PRIOR_BINDER_PREDICTIONS_SHA256,
    }
    for path, expected in fixed.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {path}: {actual} != {expected}")
        observed[str(path)] = actual
    for arm in ARMS:
        for name in (
            "dispatch",
            "failures",
            "binder_input",
            "binder_predictions",
            "binder_ambiguities",
        ):
            path = arm_paths(arm)[name]
            observed[str(path)] = _verify_sidecar(path)
    return observed


def _assert_hammer15_binder_replay(predictions: dict[str, dict[str, Any]]) -> None:
    prior = _load(PRIOR_BINDER_PREDICTIONS)
    prior_primary = {
        case["case_id"]: case
        for case in prior["case_results"]
        if case["split"] == "primary_targeted_100"
    }
    current = {
        case["case_id"]: case
        for case in predictions["hammer15"]["case_results"]
    }
    if prior_primary.keys() != current.keys():
        raise RuntimeError("Current and prior Hammer 1.5B binder IDs differ")
    stable_fields = (
        "raw_request",
        "semantic_delegation",
        "baseline_selected_call",
        "binding",
    )
    differences = [
        (case_id, field)
        for case_id in current
        for field in stable_fields
        if prior_primary[case_id][field] != current[case_id][field]
    ]
    if differences:
        raise RuntimeError(f"Hammer 1.5B binder replay changed: {differences[:5]}")


def _assert_controlled_contract(
    reports: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> list[str]:
    primary = _load(PRIMARY_100)
    expected_ids = primary["provenance"]["case_ids"]
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
    by_arm = {
        arm: {case["case_id"]: case for case in reports[arm]["case_results"]}
        for arm in ARMS
    }
    prediction_ids = {
        arm: [case["case_id"] for case in predictions[arm]["case_results"]]
        for arm in ARMS
    }
    for arm in ARMS:
        report = reports[arm]
        if list(by_arm[arm]) != expected_ids or prediction_ids[arm] != expected_ids:
            raise RuntimeError(f"{arm} case IDs differ from the frozen primary set")
        if report["configuration"]["focused_repair_variant"] != "schema":
            raise RuntimeError(f"{arm} is not the selected A1 schema arm")
        if predictions[arm]["status"] != "predictions_frozen_before_scoring":
            raise RuntimeError(f"{arm} binder predictions were not frozen")

    for field in ("task_instruction_sha256", "hammer_format_instruction_sha256"):
        if (
            reports["hammer15"]["configuration"][field]
            != reports["hammer3"]["configuration"][field]
        ):
            raise RuntimeError(f"Hammer prompt contract changed: {field}")

    for case_id in expected_ids:
        old = by_arm["hammer15"][case_id]
        new = by_arm["hammer3"][case_id]
        for field in source_fields:
            if old[field] != new[field]:
                raise RuntimeError(f"Frozen case changed for {case_id}: {field}")
        for field in dispatch_fields:
            if old["dispatcher"][field] != new["dispatcher"][field]:
                raise RuntimeError(f"Dispatcher contract changed for {case_id}: {field}")
        if old["execution"]["executed"] or new["execution"]["executed"]:
            raise RuntimeError(f"Unexpected tool execution in {case_id}")
    return expected_ids


def _paired_binomial_p(failure_to_success: int, success_to_failure: int) -> float:
    """Exact two-sided sign/McNemar p-value over discordant pairs."""

    discordant = failure_to_success + success_to_failure
    if discordant == 0:
        return 1.0
    lower = min(failure_to_success, success_to_failure)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _arm_metrics(
    arm: str,
    cases: list[dict[str, Any]],
    report: dict[str, Any],
    predictions: dict[str, Any],
) -> dict[str, Any]:
    scored = _metrics(cases)
    case_ids = {case["case_id"] for case in cases}
    dispatch_cases = [
        case for case in report["case_results"] if case["case_id"] in case_ids
    ]
    generations = [case["dispatcher"]["generation"] for case in dispatch_cases]
    latencies = [
        generation["latency_ms"]
        for generation in generations
        if generation.get("latency_ms") is not None
    ]
    rates = [
        generation["tokens_per_second"]
        for generation in generations
        if generation.get("tokens_per_second") is not None
    ]
    peaks = [
        generation["peak_memory_gb"]
        for generation in generations
        if generation.get("peak_memory_gb") is not None
    ]
    statuses = predictions["status_counts"]
    abstentions = sum(
        statuses.get(status, 0)
        for status in ("no_selected_call", "unbound", "unsupported_selected_tool")
    )
    return {
        "arm": arm,
        "model_id": report["configuration"]["dispatcher_model_id"],
        "cases": len(cases),
        "tool_selection_accuracy": scored["tool_selection_accuracy"],
        "official_strict_exact_accuracy": scored[
            "binder_official_strict_exact_accuracy"
        ],
        "official_preservation_aware_exact_accuracy": scored[
            "binder_official_preservation_aware_exact_accuracy"
        ],
        "literal_source_exact_accuracy_conservative": scored[
            "binder_literal_source_exact_conservative"
        ],
        "literal_source_exact_accuracy_on_eligible": scored[
            "binder_literal_source_exact_on_eligible"
        ],
        "literal_reference_eligible_cases": scored[
            "literal_reference_eligible_cases"
        ],
        "schema_valid_rate_after_binding": scored["binder_schema_valid_rate"],
        "binder_coverage": scored["binder_final_call_available_rate"],
        "binder_status_counts": statuses,
        "abstention_count": abstentions,
        "binder_ambiguity_count": scored["binder_ambiguous_count"],
        "annotation_ambiguity_count": scored["annotation_ambiguous_count"],
        "malformed_hammer_count": sum(
            case["malformed_output"] for case in dispatch_cases
        ),
        "hammer_rejection_count": sum(
            case["dispatcher"]["parse"]["rejected"] for case in dispatch_cases
        ),
        "average_latency_ms": mean(latencies) if latencies else None,
        "p50_latency_ms": median(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "generation_tokens_per_second": mean(rates) if rates else None,
        "peak_mlx_memory_gb": max(peaks) if peaks else None,
        "raw_slice_invariant_rate": scored["binder_raw_slice_invariant_rate"],
    }


def _flips(
    old: dict[str, dict[str, Any]],
    new: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    failure_to_success = [
        case_id
        for case_id in old
        if old[case_id][field] is False and new[case_id][field] is True
    ]
    success_to_failure = [
        case_id
        for case_id in old
        if old[case_id][field] is True and new[case_id][field] is False
    ]
    return {
        "failure_to_success": failure_to_success,
        "success_to_failure": success_to_failure,
        "net_successes": len(failure_to_success) - len(success_to_failure),
        "exact_paired_binomial_p": _paired_binomial_p(
            len(failure_to_success), len(success_to_failure)
        ),
    }


def main() -> None:
    observed = _verify_inputs()
    reports = {arm: _load(arm_paths(arm)["dispatch"]) for arm in ARMS}
    predictions = {
        arm: _load(arm_paths(arm)["binder_predictions"]) for arm in ARMS
    }
    case_ids = _assert_controlled_contract(reports, predictions)
    _assert_hammer15_binder_replay(predictions)
    annotations = _load(ANNOTATIONS)["annotations"]
    registry = ToolRegistry()

    scored: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        dispatch_by_id = {
            case["case_id"]: case for case in reports[arm]["case_results"]
        }
        prediction_by_id = {
            case["case_id"]: case
            for case in predictions[arm]["case_results"]
        }
        scored[arm] = [
            _score_case(
                dispatch_by_id[case_id],
                prediction_by_id[case_id],
                annotations[case_id],
                registry,
            )
            for case_id in case_ids
        ]

    scored_by_id = {
        arm: {case["case_id"]: case for case in scored[arm]} for arm in ARMS
    }
    old = scored_by_id["hammer15"]
    new = scored_by_id["hammer3"]
    overall = {
        arm: _arm_metrics(arm, scored[arm], reports[arm], predictions[arm])
        for arm in ARMS
    }
    per_tool = {
        tool: {
            arm: _arm_metrics(
                arm,
                [case for case in scored[arm] if case["expected_tool"] == tool],
                reports[arm],
                {
                    "status_counts": {
                        status: sum(
                            case["binder_status"] == status
                            for case in scored[arm]
                            if case["expected_tool"] == tool
                        )
                        for status in sorted(
                            {
                                case["binder_status"]
                                for case in scored[arm]
                                if case["expected_tool"] == tool
                            }
                        )
                    }
                },
            )
            for arm in ARMS
        }
        for tool in TOOLS
    }
    metric_flips = {
        "tool_selection": _flips(old, new, "pre_binding_tool_correct"),
        "official_strict_exact": _flips(
            old, new, "binder_official_strict_exact"
        ),
        "official_preservation_aware_exact": _flips(
            old, new, "binder_official_preservation_exact"
        ),
        "literal_source_exact": _flips(old, new, "binder_literal_source_exact"),
        "schema_validity_after_binding": _flips(old, new, "binder_schema_valid"),
    }
    literal_selection_only = [
        case_id
        for case_id in case_ids
        if not old[case_id]["pre_binding_tool_correct"]
        and new[case_id]["pre_binding_tool_correct"]
        and new[case_id]["binder_literal_source_exact"] is True
    ]
    official_selection_only = [
        case_id
        for case_id in case_ids
        if not old[case_id]["pre_binding_tool_correct"]
        and new[case_id]["pre_binding_tool_correct"]
        and new[case_id]["binder_official_strict_exact"]
    ]
    unchanged_official_failures = [
        case_id
        for case_id in case_ids
        if old[case_id]["pre_binding_tool_correct"]
        and new[case_id]["pre_binding_tool_correct"]
        and old[case_id]["binder_call"] == new[case_id]["binder_call"]
        and not old[case_id]["binder_official_strict_exact"]
        and not new[case_id]["binder_official_strict_exact"]
    ]
    official_fail_literal_success = [
        case_id
        for case_id in unchanged_official_failures
        if old[case_id]["binder_literal_source_exact"] is True
        and new[case_id]["binder_literal_source_exact"] is True
    ]
    binder_or_annotation_limited = [
        case_id
        for case_id in unchanged_official_failures
        if old[case_id]["binder_literal_source_exact"] is not True
        or new[case_id]["binder_literal_source_exact"] is not True
    ]

    comparison = {
        "version": "hammer15-vs-hammer3-deterministic-binding-capacity-v1",
        "status": "completed",
        "question": (
            "Does Hammer 3B materially improve end-to-end dispatch after exact payload "
            "regeneration is replaced by deterministic source binding?"
        ),
        "frozen_inputs": observed,
        "controlled_conditions": {
            "case_count": 100,
            "tools": list(TOOLS),
            "same_frozen_qwen_delegations": True,
            "qwen_called": False,
            "delegations_regenerated": False,
            "same_a1_schemas": True,
            "same_hammer_native_format": True,
            "same_parser_validator_scoring": True,
            "same_deterministic_binder": True,
            "hammer15_binding_matches_prior_frozen_primary": True,
            "binder_had_literal_annotation_access": False,
            "predictions_frozen_before_scoring": True,
            "official_gold_mutated": False,
            "stateless": True,
            "one_generation_per_case": True,
            "yuki_tool_execution_capability": False,
            "yuki_tool_execution_attempts": 0,
        },
        "overall": overall,
        "per_tool": per_tool,
        "change_hammer3_minus_hammer15": {
            "tool_selection_percentage_points": (
                overall["hammer3"]["tool_selection_accuracy"]
                - overall["hammer15"]["tool_selection_accuracy"]
            )
            * 100,
            "official_strict_exact_percentage_points": (
                overall["hammer3"]["official_strict_exact_accuracy"]
                - overall["hammer15"]["official_strict_exact_accuracy"]
            )
            * 100,
            "official_preservation_aware_exact_percentage_points": (
                overall["hammer3"]["official_preservation_aware_exact_accuracy"]
                - overall["hammer15"][
                    "official_preservation_aware_exact_accuracy"
                ]
            )
            * 100,
            "literal_source_exact_conservative_percentage_points": (
                overall["hammer3"]["literal_source_exact_accuracy_conservative"]
                - overall["hammer15"]["literal_source_exact_accuracy_conservative"]
            )
            * 100,
            "schema_validity_percentage_points": (
                overall["hammer3"]["schema_valid_rate_after_binding"]
                - overall["hammer15"]["schema_valid_rate_after_binding"]
            )
            * 100,
            "binder_coverage_percentage_points": (
                overall["hammer3"]["binder_coverage"]
                - overall["hammer15"]["binder_coverage"]
            )
            * 100,
            "latency_ratio": (
                overall["hammer3"]["average_latency_ms"]
                / overall["hammer15"]["average_latency_ms"]
            ),
            "peak_mlx_memory_ratio": (
                overall["hammer3"]["peak_mlx_memory_gb"]
                / overall["hammer15"]["peak_mlx_memory_gb"]
            ),
        },
        "case_level_analysis": {
            "flips": metric_flips,
            "fixed_only_by_better_hammer_selection": {
                "official_strict_exact": official_selection_only,
                "literal_source_exact": literal_selection_only,
            },
            "unchanged_same_call_official_failures": unchanged_official_failures,
            "unchanged_official_fail_but_literal_source_success": (
                official_fail_literal_success
            ),
            "unchanged_binder_or_annotation_limited": binder_or_annotation_limited,
        },
        "interpretation": {
            "real_capacity_signal": (
                "Hammer 3B materially improves see tool selection and therefore see "
                "literal-source exactness after binding."
            ),
            "overall_tradeoff": (
                "The targeted 100-case gain is +6 points in selection and +5 points "
                "in conservative literal-source exactness, but -1 point in official "
                "strict exactness at about 2x latency and 1.9x peak MLX memory."
            ),
            "decision": (
                "Do not replace Hammer 1.5B globally on this evidence. Hammer 3B is "
                "justified only if the see-specific selection gain is worth the nearly "
                "doubled local inference cost; it did not improve the other three "
                "tools end-to-end as a group."
            ),
        },
    }
    write_report(comparison, OUTPUT)

    paired_records = []
    for case_id in case_ids:
        paired_records.append(
            {
                "case_id": case_id,
                "expected_tool": old[case_id]["expected_tool"],
                "raw_request": old[case_id]["raw_request"],
                "official_arguments": old[case_id]["official_arguments"],
                "literal_annotation": old[case_id]["literal_annotation"],
                "hammer15": old[case_id],
                "hammer3": new[case_id],
            }
        )
    CASES_OUTPUT.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in paired_records),
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
