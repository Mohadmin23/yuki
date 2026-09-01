"""Build final automatic-schema comparison artifacts from completed reports."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

REPORTS = Path(__file__).with_name("reports")
AUTOMATIC = REPORTS / "automatic"

MODELS: dict[str, dict[str, Any]] = {
    "hammer21_1_5b": {
        "label": "Hammer 2.1 1.5B",
        "all_18": {
            "broad_72": REPORTS
            / "hammer21-1.5b-phase1-native-official-final.json",
            "targeted_96": REPORTS / "hammer21-1.5b-targeted-96-final.json",
        },
        "oracle": {
            "broad_72": None,
            "targeted_96": REPORTS
            / "oracle/hammer21-1.5b-targeted-logical-domains.json",
        },
        "automatic": {
            "broad_72": AUTOMATIC
            / "hammer21-1.5b-broad-72-automatic.json",
            "targeted_96": AUTOMATIC
            / "hammer21-1.5b-targeted-96-automatic.json",
        },
    },
    "hammer21_3b": {
        "label": "Hammer 2.1 3B",
        "all_18": {
            "broad_72": REPORTS / "hammer21-3b-phase1-native-final.json",
            "targeted_96": REPORTS / "hammer21-3b-targeted-96-final.json",
        },
        "oracle": {
            "broad_72": None,
            "targeted_96": REPORTS
            / "oracle/hammer21-3b-targeted-logical-domains.json",
        },
        "automatic": {
            "broad_72": AUTOMATIC / "hammer21-3b-broad-72-automatic.json",
            "targeted_96": AUTOMATIC
            / "hammer21-3b-targeted-96-automatic.json",
        },
    },
    "xlam_2_3b": {
        "label": "xLAM-2 3B",
        "all_18": {
            "broad_72": REPORTS / "new-models/xlam-2-3b-broad-72.json",
            "targeted_96": REPORTS / "new-models/xlam-2-3b-targeted-96.json",
        },
        "oracle": {"broad_72": None, "targeted_96": None},
        "automatic": {
            "broad_72": AUTOMATIC / "xlam-2-3b-broad-72-automatic.json",
            "targeted_96": AUTOMATIC
            / "xlam-2-3b-targeted-96-automatic.json",
        },
    },
    "arch_agent_3b": {
        "label": "Arch-Agent 3B",
        "all_18": {
            "broad_72": REPORTS / "new-models/arch-agent-3b-broad-72.json",
            "targeted_96": REPORTS
            / "new-models/arch-agent-3b-targeted-96.json",
        },
        "oracle": {"broad_72": None, "targeted_96": None},
        "automatic": {
            "broad_72": AUTOMATIC
            / "arch-agent-3b-broad-72-automatic.json",
            "targeted_96": AUTOMATIC
            / "arch-agent-3b-targeted-96-automatic.json",
        },
    },
}


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _peak_memory(report: dict[str, Any]) -> float | None:
    values = [
        generation["peak_memory_gb"]
        for case in report["case_results"]
        if (generation := case.get("generation"))
        and generation.get("peak_memory_gb") is not None
    ]
    return max(values) if values else None


def _baseline_summary(
    report: dict[str, Any], path: Path, exposure: str
) -> dict[str, Any]:
    metrics = report["metrics"]
    cases = report["case_results"]
    total = len(cases)
    default_offered = report["configuration"]["available_tools"]
    tool_available = sum(
        case["expected_tool"] in case.get("offered_tools", default_offered)
        for case in cases
    )
    offered_counts = [
        len(case.get("offered_tools", default_offered)) for case in cases
    ]
    valid_success = sum(
        bool(case["tool_correct"] and case["schema_valid"]) for case in cases
    )
    return {
        "status": "completed",
        "source_report": str(path),
        "exposure": exposure,
        "total_cases": total,
        "correct_tool_in_selected_subset_rate": tool_available / total,
        "average_offered_schemas": mean(offered_counts),
        "minimum_offered_schemas": min(offered_counts),
        "maximum_offered_schemas": max(offered_counts),
        "tool_selection_accuracy": metrics["tool_selection_accuracy"],
        "strict_argument_accuracy": metrics["strict_argument_accuracy"],
        "execution_equivalent_argument_accuracy": metrics[
            "execution_equivalent_argument_accuracy"
        ],
        "strict_exact_call_accuracy": metrics["exact_call_accuracy"],
        "execution_equivalent_exact_call_accuracy": metrics[
            "execution_equivalent_call_accuracy"
        ],
        "schema_valid_output_rate": metrics["schema_valid_output_rate"],
        "schema_valid_successful_call_rate": valid_success / total,
        "malformed_output_rate": metrics["malformed_output_rate"],
        "average_total_latency_ms": None,
        "p50_total_latency_ms": None,
        "p95_total_latency_ms": None,
        "average_dispatcher_latency_ms": metrics.get(
            "average_inference_latency_ms"
        ),
        "p50_dispatcher_latency_ms": metrics.get("p50_inference_latency_ms"),
        "p95_dispatcher_latency_ms": metrics.get("p95_inference_latency_ms"),
        "generation_tokens_per_second": metrics.get(
            "average_tokens_per_second"
        ),
        "peak_mlx_memory_gb": _peak_memory(report),
        "total_latency_note": (
            "not recorded in the reused baseline; dispatcher inference latency "
            "is retained without fabrication"
        ),
    }


def _automatic_summary(report: dict[str, Any], path: Path) -> dict[str, Any]:
    selector = report["selector_metrics"]
    dispatcher = report["dispatcher_metrics"]
    end_to_end = report["end_to_end_metrics"]
    unsuccessful = [
        case
        for case in report["case_results"]
        if not case["end_to_end_execution_equivalent_exact_call"]
    ]
    failure_families = Counter(
        case["tags"][1] if len(case["tags"]) > 1 else "untagged"
        for case in unsuccessful
    )
    primary_stages = Counter(
        case["failure_stage"] or "none" for case in unsuccessful
    )
    return {
        "status": "completed",
        "source_report": str(path),
        "exposure": "automatic_subset",
        "total_cases": report["configuration"]["total_cases"],
        "exact_domain_selection_accuracy": selector[
            "exact_domain_selection_accuracy"
        ],
        "correct_tool_in_selected_subset_rate": selector[
            "correct_tool_in_selected_subset_rate"
        ],
        "average_selected_domains": selector["average_selected_domains"],
        "average_offered_schemas": selector["average_offered_schemas"],
        "minimum_offered_schemas": selector["minimum_offered_schemas"],
        "maximum_offered_schemas": selector["maximum_offered_schemas"],
        "average_selector_latency_ms": selector[
            "average_selector_latency_ms"
        ],
        "p50_selector_latency_ms": selector["p50_selector_latency_ms"],
        "p95_selector_latency_ms": selector["p95_selector_latency_ms"],
        "tool_selection_accuracy": dispatcher["tool_selection_accuracy"],
        "strict_argument_accuracy": dispatcher["strict_argument_accuracy"],
        "execution_equivalent_argument_accuracy": dispatcher[
            "execution_equivalent_argument_accuracy"
        ],
        "strict_exact_call_accuracy": end_to_end[
            "strict_exact_call_accuracy"
        ],
        "execution_equivalent_exact_call_accuracy": end_to_end[
            "execution_equivalent_exact_call_accuracy"
        ],
        "schema_valid_output_rate": dispatcher["schema_valid_output_rate"],
        "schema_valid_successful_call_rate": end_to_end[
            "schema_valid_successful_call_rate"
        ],
        "malformed_output_rate": dispatcher["malformed_output_rate"],
        "average_total_latency_ms": end_to_end["average_total_latency_ms"],
        "p50_total_latency_ms": end_to_end["p50_total_latency_ms"],
        "p95_total_latency_ms": end_to_end["p95_total_latency_ms"],
        "average_dispatcher_latency_ms": dispatcher[
            "average_dispatcher_latency_ms"
        ],
        "p50_dispatcher_latency_ms": dispatcher["p50_dispatcher_latency_ms"],
        "p95_dispatcher_latency_ms": dispatcher["p95_dispatcher_latency_ms"],
        "generation_tokens_per_second": dispatcher[
            "aggregate_generation_tokens_per_second"
        ],
        "peak_mlx_memory_gb": dispatcher["peak_mlx_memory_gb"],
        "primary_failure_stages": dict(primary_stages),
        "failed_targeted_families": dict(failure_families),
        "selector_failures_by_expected_tool": report[
            "selector_failures_by_expected_tool"
        ],
        "failures_by_expected_tool": report["failures_by_expected_tool"],
        "selector_confusion_matrix_by_expected_domain": report[
            "selector_confusion_matrix_by_expected_domain"
        ],
        "dispatcher_confusion_matrix": report[
            "dispatcher_confusion_matrix"
        ],
    }


def build_comparison() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models: dict[str, Any] = {}
    combined_failures: list[dict[str, Any]] = []
    all_automatic_reports: list[dict[str, Any]] = []

    for model_key, definition in MODELS.items():
        model_output: dict[str, Any] = {"label": definition["label"], "datasets": {}}
        for dataset in ("broad_72", "targeted_96"):
            all_path = definition["all_18"][dataset]
            oracle_path = definition["oracle"][dataset]
            automatic_path = definition["automatic"][dataset]
            automatic_report = _load(automatic_path)
            all_automatic_reports.append(automatic_report)
            combined_failures.extend(automatic_report["failure_records"])
            model_output["datasets"][dataset] = {
                "all_18": _baseline_summary(
                    _load(all_path), all_path, "all_18"
                ),
                "oracle_subset": (
                    _baseline_summary(
                        _load(oracle_path), oracle_path, "oracle_subset"
                    )
                    if oracle_path
                    else {
                        "status": "not_run",
                        "reason": "No compatible oracle report exists; not fabricated.",
                    }
                ),
                "automatic_subset": _automatic_summary(
                    automatic_report, automatic_path
                ),
            }
        models[model_key] = model_output

    unique_cases = [
        *all_automatic_reports[0]["case_results"],
        *all_automatic_reports[1]["case_results"],
    ]
    selector_latency_samples = [
        case["selector_latency_ms"]
        for report in all_automatic_reports
        for case in report["case_results"]
    ]
    unique_total = len(unique_cases)
    selector_summary = {
        "unique_cases": unique_total,
        "exact_domain_selection_accuracy": sum(
            case["exact_domain_selection"] for case in unique_cases
        )
        / unique_total,
        "correct_tool_in_selected_subset_rate": sum(
            case["correct_tool_in_selected_subset"] for case in unique_cases
        )
        / unique_total,
        "average_selected_domains": mean(
            len(case["selected_domains"]) for case in unique_cases
        ),
        "average_offered_schemas": mean(
            case["offered_schema_count"] for case in unique_cases
        ),
        "minimum_offered_schemas": min(
            case["offered_schema_count"] for case in unique_cases
        ),
        "maximum_offered_schemas": max(
            case["offered_schema_count"] for case in unique_cases
        ),
        "selector_latency_measurements": len(selector_latency_samples),
        "average_selector_latency_ms": mean(selector_latency_samples),
        "p50_selector_latency_ms": median(selector_latency_samples),
        "p95_selector_latency_ms": _percentile(selector_latency_samples, 0.95),
    }

    comparison = {
        "experiment": "Yuki automatic schema selection",
        "phase": "phase_1_automatic_subset",
        "safety": {
            "yuki_tool_execution": False,
            "execution_capability_in_automatic_runner": False,
            "network_or_external_service_calls_from_yuki_tools": False,
            "persistent_yuki_state_changes": False,
            "one_stateless_dispatcher_generation_per_case": True,
        },
        "selector": {
            "strategy": "deterministic-domain-rules-v1",
            "summary_across_unique_cases": selector_summary,
        },
        "models": models,
        "notes": [
            "All-18 and oracle results were reused from compatible completed reports.",
            "Only automatic-subset inference was newly run.",
            "Oracle results for xLAM-2 3B and Arch-Agent 3B were not run.",
            "Broad oracle results were not run for any finalist.",
            "Reused reports did not record full pipeline latency, so their total latency is null rather than inferred.",
        ],
    }
    return comparison, combined_failures


def main() -> None:
    comparison, failures = build_comparison()
    comparison_path = AUTOMATIC / "automatic-routing-comparison.json"
    failures_path = AUTOMATIC / "automatic-routing-failures.jsonl"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with failures_path.open("w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in failures
        )


if __name__ == "__main__":
    main()
