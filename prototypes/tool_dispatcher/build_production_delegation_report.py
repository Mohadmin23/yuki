"""Build the versioned production-delegation comparison and human report."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "production-delegation"

PATHS = {
    "derived_dataset": ROOT / "production-delegation-source-v1.json",
    "stage_a": REPORT_ROOT
    / "stage-a"
    / "production-delegation-mainbrain-qwen3-14b-surrogate-fallback180-v1.json",
    "hammer_1_5b": REPORT_ROOT
    / "stage-b"
    / "production-delegation-hammer1.5b-qwen3-14b-surrogate-fallback180-v1.json",
    "hammer_3b": REPORT_ROOT
    / "stage-b"
    / "production-delegation-hammer3b-qwen3-14b-surrogate-fallback180-v1.json",
    "no_verbatim": REPORT_ROOT
    / "ablations"
    / "hammer1.5b-no-verbatim-fallback180-v1.json",
    "raw_generated": REPORT_ROOT
    / "ablations"
    / "hammer1.5b-raw-user-generated-domain-fallback180-v1.json",
    "raw_gold": REPORT_ROOT
    / "ablations"
    / "hammer1.5b-raw-user-gold-domain-fallback180-v1.json",
    "phase2_hammer_3b": ROOT / "reports" / "phase2" / "hammer3b-phase2.json",
    "oracle": ROOT / "reports" / "oracle" / "hammer21-targeted-oracle-comparison.json",
}

COMPARISON_PATH = REPORT_ROOT / "production-delegation-comparison-v1.json"
MARKDOWN_PATH = REPORT_ROOT / "PRODUCTION-DELEGATION-REPORT.md"
MANIFEST_PATH = REPORT_ROOT / "production-delegation-artifact-manifest-v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    radius = (
        z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    )
    return [center - radius, center + radius]


def _paired(
    first: dict[str, Any], second: dict[str, Any], metric: str
) -> dict[str, Any]:
    first_cases = {case["case_id"]: case for case in first["case_results"]}
    second_cases = {case["case_id"]: case for case in second["case_results"]}
    if set(first_cases) != set(second_cases):
        raise RuntimeError("Paired reports do not contain identical case IDs")
    counts = {
        "both_correct": 0,
        "first_only": 0,
        "second_only": 0,
        "both_wrong": 0,
    }
    for case_id, first_case in first_cases.items():
        first_value = bool(first_case[metric])
        second_value = bool(second_cases[case_id][metric])
        if first_value and second_value:
            counts["both_correct"] += 1
        elif first_value:
            counts["first_only"] += 1
        elif second_value:
            counts["second_only"] += 1
        else:
            counts["both_wrong"] += 1
    discordant = counts["first_only"] + counts["second_only"]
    if discordant:
        smaller = min(counts["first_only"], counts["second_only"])
        probability = min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(smaller + 1))
            / 2**discordant,
        )
    else:
        probability = 1.0
    return {**counts, "exact_mcnemar_p": probability}


def _model_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    cases = report["case_results"]
    strict_successes = sum(case["strict_exact_call"] for case in cases)
    preservation_successes = sum(
        case["preservation_aware_exact_call"] for case in cases
    )
    upstream_categories = {
        "malformed_delegation",
        "main_brain_wrong_domain",
        "main_brain_lost_verbatim",
    }
    upstream_failures = sum(
        count
        for category, count in report["failure_decomposition"].items()
        if category in upstream_categories
    )
    strict_failures = len(cases) - strict_successes
    return {
        "model_id": report["configuration"]["dispatcher_model_id"],
        "cases": len(cases),
        "generated_dispatches": report["configuration"]["generated_dispatches"],
        "tool_selection_accuracy": summary["tool_selection_accuracy"],
        "tool_selection_accuracy_when_domain_correct": summary[
            "tool_selection_accuracy_when_domain_correct"
        ],
        "strict_argument_accuracy_when_tool_correct": summary[
            "strict_argument_accuracy_when_tool_correct"
        ],
        "preservation_aware_argument_accuracy_when_tool_correct": summary[
            "preservation_aware_argument_accuracy_when_tool_correct"
        ],
        "strict_exact_call_accuracy": summary["strict_exact_call_accuracy"],
        "strict_exact_call_successes": strict_successes,
        "strict_exact_call_wilson_95": _wilson(strict_successes, len(cases)),
        "preservation_aware_exact_call_accuracy": summary[
            "preservation_aware_exact_call_accuracy"
        ],
        "preservation_aware_exact_call_successes": preservation_successes,
        "preservation_aware_exact_call_wilson_95": _wilson(
            preservation_successes, len(cases)
        ),
        "strict_exact_call_accuracy_when_domain_correct": summary[
            "strict_exact_call_accuracy_when_domain_correct"
        ],
        "preservation_aware_exact_call_accuracy_when_domain_correct": summary[
            "preservation_aware_exact_call_accuracy_when_domain_correct"
        ],
        "schema_valid_rate": summary["schema_valid_rate"],
        "malformed_output_rate": summary["malformed_output_rate"],
        "conversational_output_rate": summary["conversational_output_rate"],
        "average_dispatcher_latency_ms": summary["average_dispatcher_latency_ms"],
        "p50_dispatcher_latency_ms": summary["p50_dispatcher_latency_ms"],
        "p95_dispatcher_latency_ms": summary["p95_dispatcher_latency_ms"],
        "average_generation_tokens_per_second": summary[
            "average_generation_tokens_per_second"
        ],
        "peak_mlx_memory_gb": summary["peak_mlx_memory_gb"],
        "non_ambiguous": report["non_ambiguous_summary"],
        "failure_decomposition": report["failure_decomposition"],
        "upstream_attributed_strict_failures": upstream_failures,
        "dispatcher_attributed_strict_failures": strict_failures - upstream_failures,
        "ambiguous_or_adversarial_strict_failures": sum(
            case["ambiguous_or_adversarial"] and not case["strict_exact_call"]
            for case in cases
        ),
        "by_tool": report["by_tool"],
        "by_family": report["by_family"],
        "by_dataset_category": report["by_dataset_category"],
        "confusion_matrix": report["confusion_matrix"],
    }


def _ablation_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "input_mode": report["configuration"]["dispatcher_input_mode"],
        "domain_source": report["configuration"]["routing_domain_source"],
        "uses_oracle_domain": report["configuration"]["uses_oracle_domain"],
        "generated_dispatches": report["configuration"]["generated_dispatches"],
        "routing_domain_accuracy": summary["routing_domain_accuracy"],
        "tool_selection_accuracy": summary["tool_selection_accuracy"],
        "strict_exact_call_accuracy": summary["strict_exact_call_accuracy"],
        "preservation_aware_exact_call_accuracy": summary[
            "preservation_aware_exact_call_accuracy"
        ],
        "strict_argument_accuracy_when_tool_correct": summary[
            "strict_argument_accuracy_when_tool_correct"
        ],
        "preservation_aware_argument_accuracy_when_tool_correct": summary[
            "preservation_aware_argument_accuracy_when_tool_correct"
        ],
        "schema_valid_rate": summary["schema_valid_rate"],
        "malformed_output_rate": summary["malformed_output_rate"],
        "average_dispatcher_latency_ms": summary["average_dispatcher_latency_ms"],
    }


def _oracle_reference(oracle: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in oracle["models"]:
        label = "hammer_1_5b" if "1.5B" in row["label"] else "hammer_3b"
        profile = row["oracle_profile"] or "all_18"
        result.setdefault(label, {})[profile] = {
            "cases": row["total_cases"],
            "strict_exact_call_accuracy": row["exact_call_accuracy"],
            "equivalent_exact_call_accuracy": row["execution_equivalent_call_accuracy"],
            "average_dispatcher_latency_ms": row["average_inference_latency_ms"],
            "peak_mlx_memory_gb": row["peak_memory_gb"],
        }
    return result


def _old_selector_same_subset(
    phase2: dict[str, Any], selected_ids: set[str]
) -> dict[str, Any]:
    cases = [case for case in phase2["case_results"] if case["case_id"] in selected_ids]
    if len(cases) != 180:
        raise RuntimeError("Phase 2 selector comparison did not resolve 180 cases")
    available = [case for case in cases if case["correct_tool_in_selected_subset"]]
    return {
        "cases": len(cases),
        "exact_domain_selection_accuracy": mean(
            case["exact_domain_selection"] for case in cases
        ),
        "correct_tool_in_selected_subset_rate": mean(
            case["correct_tool_in_selected_subset"] for case in cases
        ),
        "hammer_3b_tool_selection_accuracy": mean(
            case["tool_correct"] for case in cases
        ),
        "hammer_3b_strict_exact_call_accuracy": mean(
            case["strict_exact_call"] for case in cases
        ),
        "hammer_3b_equivalent_exact_call_accuracy": mean(
            case["execution_equivalent_exact_call"] for case in cases
        ),
        "hammer_3b_tool_selection_when_available": mean(
            case["tool_correct"] for case in available
        ),
        "hammer_3b_strict_exact_when_available": mean(
            case["strict_exact_call"] for case in available
        ),
        "average_dispatcher_latency_ms": mean(
            case["generation"]["latency_ms"] for case in cases
        ),
    }


def _failure_examples(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    for case in report["case_results"]:
        category = case["failure_category"]
        if category is None or len(examples.setdefault(category, [])) >= 2:
            continue
        examples[category].append(
            {
                "case_id": case["case_id"],
                "request": case["request"],
                "expected_tool": case["expected_tool"],
                "actual_tool": case["actual_tool"],
                "expected_arguments": case["expected_arguments"],
                "actual_arguments": case["actual_arguments"],
                "delegation": case["delegation"],
            }
        )
    return examples


def build_comparison() -> dict[str, Any]:
    reports = {name: _load(path) for name, path in PATHS.items()}
    dataset = reports["derived_dataset"]
    stage_a = reports["stage_a"]
    hammer_1_5b = reports["hammer_1_5b"]
    hammer_3b = reports["hammer_3b"]

    expected_stage_a_sha = hammer_1_5b["configuration"]["stage_a_delegation_sha256"]
    if _sha256(PATHS["stage_a"]) != expected_stage_a_sha:
        raise RuntimeError("Frozen Stage A checksum no longer matches Stage B")
    if hammer_3b["configuration"]["stage_a_delegation_sha256"] != expected_stage_a_sha:
        raise RuntimeError("Hammer models did not use the same Stage A report")

    stage_a_cases = stage_a["case_results"]
    domain_successes = sum(
        case["delegation_scores"]["domain_hint_exact"] for case in stage_a_cases
    )
    valid_cases = [case for case in stage_a_cases if case["delegation"] is not None]
    malformed_cases = [case for case in stage_a_cases if case["delegation"] is None]
    verbatim_miss_cases = [
        case
        for case in stage_a_cases
        if case["delegation_scores"]["verbatim_recall"] < 1
    ]
    stage_a_summary = {
        **stage_a["summary"],
        "valid_delegations": len(valid_cases),
        "malformed_delegations": len(malformed_cases),
        "correct_domain_hints": domain_successes,
        "domain_hint_accuracy_given_valid_delegation": domain_successes
        / len(valid_cases),
        "verbatim_miss_cases": len(verbatim_miss_cases),
        "malformed_case_ids": [case["case_id"] for case in malformed_cases],
        "verbatim_miss_case_ids": [case["case_id"] for case in verbatim_miss_cases],
        "whole_source_request_reproduced_in_delegation": sum(
            case["delegation"] is not None
            and (
                case["delegation"]["request"] == case["request"]
                or case["request"] in case["delegation"]["verbatim"]
            )
            for case in stage_a_cases
        ),
        "semantic_metric_false_but_hammer_1_5b_strict_successes": sum(
            not case["delegation_scores"]["semantic_request_quality"]
            and dispatch_case["strict_exact_call"]
            for case, dispatch_case in zip(
                stage_a_cases, hammer_1_5b["case_results"], strict=True
            )
        ),
        "by_expected_domain": stage_a["by_expected_domain"],
        "by_dataset_category": stage_a["by_dataset_category"],
    }

    stage_a_by_id = {case["case_id"]: case for case in stage_a_cases}
    end_to_end_performance = {}
    for name, report in (("hammer_1_5b", hammer_1_5b), ("hammer_3b", hammer_3b)):
        latencies = []
        for case in report["case_results"]:
            total = stage_a_by_id[case["case_id"]]["generation"]["latency_ms"]
            if case["dispatcher"] is not None:
                latency = case["dispatcher"]["generation"].get("latency_ms")
                if latency is not None:
                    total += latency
            latencies.append(total)
        end_to_end_performance[name] = {
            "average_latency_ms": mean(latencies),
            "p50_latency_ms": median(latencies),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "load_latency_excluded": True,
            "models_were_measured_separately": True,
        }

    selected_ids = set(dataset["selection_sets"]["fallback-stratified-180"])
    old_selector = _old_selector_same_subset(reports["phase2_hammer_3b"], selected_ids)
    primary_1_5b = _model_summary(hammer_1_5b)
    primary_3b = _model_summary(hammer_3b)
    raw_generated = reports["raw_generated"]
    raw_gold = reports["raw_gold"]

    return {
        "experiment": "yuki_production_interface_delegation_v1",
        "status": "completed_with_temporary_main_brain_surrogate",
        "main_brain_label": stage_a["configuration"]["surrogate_label"],
        "main_brain_is_surrogate": True,
        "intended_main_brain_not_tested": "Qwen3.8-27B",
        "safety": {
            "yuki_tool_execution_capability": False,
            "yuki_tool_execution_attempts": 0,
            "external_api_usage": False,
            "one_model_loaded_at_a_time": True,
            "one_stateless_generation_per_stage_per_case": True,
            "previous_tool_results_used": False,
            "original_request_sent_to_primary_dispatcher": False,
        },
        "dataset": {
            "source": "phase2-cases.json",
            "source_cases": 450,
            "source_sha256": dataset["source_dataset_sha256"],
            "derived_version": dataset["dataset_version"],
            "derived_sha256": _sha256(PATHS["derived_dataset"]),
            "selection": "fallback-stratified-180",
            "selection_cases": 180,
            "selection_rule": (
                "ten cases per tool, category-first quotas, then source-order fill"
            ),
            "original_case_ids_and_gold_preserved": True,
        },
        "stage_a": {
            "configuration": stage_a["configuration"],
            "summary": stage_a_summary,
        },
        "primary": {
            "hammer_1_5b": primary_1_5b,
            "hammer_3b": primary_3b,
            "paired_strict_exact_call": _paired(
                hammer_1_5b, hammer_3b, "strict_exact_call"
            ),
            "paired_preservation_aware_exact_call": _paired(
                hammer_1_5b, hammer_3b, "preservation_aware_exact_call"
            ),
            "end_to_end_performance": end_to_end_performance,
            "failure_examples_hammer_1_5b": _failure_examples(hammer_1_5b),
            "failure_examples_hammer_3b": _failure_examples(hammer_3b),
        },
        "ablations_hammer_1_5b": {
            "primary_generated_domain_semantic_plus_verbatim": {
                "input_mode": "delegated_with_verbatim",
                "domain_source": "generated",
                "strict_exact_call_accuracy": primary_1_5b[
                    "strict_exact_call_accuracy"
                ],
                "preservation_aware_exact_call_accuracy": primary_1_5b[
                    "preservation_aware_exact_call_accuracy"
                ],
                "average_dispatcher_latency_ms": primary_1_5b[
                    "average_dispatcher_latency_ms"
                ],
            },
            "generated_domain_semantic_without_verbatim": _ablation_summary(
                reports["no_verbatim"]
            ),
            "generated_domain_raw_user": _ablation_summary(raw_generated),
            "gold_domain_raw_user": _ablation_summary(raw_gold),
            "paired_primary_vs_no_verbatim_strict": _paired(
                hammer_1_5b, reports["no_verbatim"], "strict_exact_call"
            ),
            "paired_primary_vs_raw_generated_strict": _paired(
                hammer_1_5b, raw_generated, "strict_exact_call"
            ),
            "paired_primary_vs_raw_gold_strict": _paired(
                hammer_1_5b, raw_gold, "strict_exact_call"
            ),
            "paired_raw_generated_vs_raw_gold_strict": _paired(
                raw_generated, raw_gold, "strict_exact_call"
            ),
        },
        "same_180_previous_raw_text_selector": old_selector,
        "selector_change": {
            "exact_domain_accuracy_delta": stage_a_summary["domain_hint_accuracy"]
            - old_selector["exact_domain_selection_accuracy"],
            "correct_tool_coverage_delta": stage_a_summary["domain_hint_accuracy"]
            - old_selector["correct_tool_in_selected_subset_rate"],
            "hammer_3b_strict_exact_call_delta": primary_3b[
                "strict_exact_call_accuracy"
            ]
            - old_selector["hammer_3b_strict_exact_call_accuracy"],
        },
        "targeted_96_oracle_reference": {
            "warning": (
                "Reference uses a different 96-case targeted dataset; gaps are "
                "descriptive and are not paired accuracy estimates."
            ),
            "results": _oracle_reference(reports["oracle"]),
        },
        "conclusions": {
            "selector_generalization": (
                "Improved materially but not solved: exact domain coverage rose "
                "on the same 180 cases, while 22 cases still lacked a correct "
                "domain due to malformed or wrong Stage A output."
            ),
            "verbatim": (
                "Essential in this interface: removing it caused a large paired "
                "strict exact-call loss."
            ),
            "hammer_capacity": (
                "No measured 3B accuracy advantage on this frozen production "
                "interface set; 1.5B was slightly more accurate and about twice "
                "as fast/small, with a non-significant paired difference."
            ),
            "dominant_failure_source": (
                "Stage A was the primary attributed source of roughly four-fifths "
                "of strict failures for both Hammer models."
            ),
            "production_decision": (
                "No production integration recommendation: rerun Stage A with "
                "the intended Qwen3.8-27B and then reuse this exact Hammer "
                "evaluation before drawing deployment conclusions."
            ),
        },
        "input_artifact_sha256": {name: _sha256(path) for name, path in PATHS.items()},
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _seconds(value: float) -> str:
    return f"{value / 1000:.2f} s"


def render_markdown(comparison: dict[str, Any]) -> str:
    stage_a = comparison["stage_a"]["summary"]
    primary = comparison["primary"]
    h15 = primary["hammer_1_5b"]
    h3 = primary["hammer_3b"]
    ablations = comparison["ablations_hammer_1_5b"]
    old = comparison["same_180_previous_raw_text_selector"]
    selector_change = comparison["selector_change"]
    oracle = comparison["targeted_96_oracle_reference"]["results"]

    category_rows = []
    for category in sorted(h15["by_dataset_category"]):
        left = h15["by_dataset_category"][category]
        right = h3["by_dataset_category"][category]
        category_rows.append(
            f"| {category} | {left['cases']} | "
            f"{_pct(left['strict_exact_call_accuracy'])} | "
            f"{_pct(right['strict_exact_call_accuracy'])} |"
        )

    domain_rows = []
    for domain, metrics in comparison["stage_a"]["summary"][
        "by_expected_domain"
    ].items():
        domain_rows.append(
            f"| {domain} | {metrics['cases']} | "
            f"{_pct(metrics['valid_delegation_rate'])} | "
            f"{_pct(metrics['domain_hint_accuracy'])} | "
            f"{_pct(metrics['average_verbatim_recall'])} | "
            f"{_pct(metrics['average_verbatim_precision'])} |"
        )

    failure_rows = []
    all_failure_categories = sorted(
        set(h15["failure_decomposition"]) | set(h3["failure_decomposition"])
    )
    for category in all_failure_categories:
        failure_rows.append(
            f"| {category} | {h15['failure_decomposition'].get(category, 0)} | "
            f"{h3['failure_decomposition'].get(category, 0)} |"
        )

    return f"""# Yuki Production-Interface Delegation Experiment

Date: 2026-08-21  
Status: completed isolated research run  
Main brain: **{comparison["main_brain_label"]}**

## Executive result

This experiment is useful but not a production verdict. The intended Qwen3.8-27B main brain was not locally available, so every Stage A result below is explicitly a **temporary Qwen3-14B-MLX-4bit surrogate** result.

The surrogate materially outperformed the failed raw-text deterministic selector on the same 180 Phase 2 cases: correct domain/tool coverage rose from {_pct(old["correct_tool_in_selected_subset_rate"])} to {_pct(stage_a["domain_hint_accuracy"])}, and Hammer 3B end-to-end strict exact calls rose from {_pct(old["hammer_3b_strict_exact_call_accuracy"])} to {_pct(h3["strict_exact_call_accuracy"])}. It did not solve routing reliability: 9 delegations were malformed, 13 more valid delegations chose the wrong domain, and exact `verbatim` recall averaged {_pct(stage_a["average_verbatim_recall"])}.

Hammer 1.5B slightly beat Hammer 3B on the frozen production interface: {_pct(h15["strict_exact_call_accuracy"])} versus {_pct(h3["strict_exact_call_accuracy"])} strict exact calls. The paired difference was only 7 cases versus 5 (`p={primary["paired_strict_exact_call"]["exact_mcnemar_p"]:.3f}`), so there is no evidence of a real accuracy advantage for either size on this set. Operationally, 1.5B was clearly lighter: {_seconds(h15["average_dispatcher_latency_ms"])} and {h15["peak_mlx_memory_gb"]:.2f} GB versus {_seconds(h3["average_dispatcher_latency_ms"])} and {h3["peak_mlx_memory_gb"]:.2f} GB.

The dominant remaining failures moved upstream. Stage A was the primary attribution for {h15["upstream_attributed_strict_failures"]}/{180 - h15["strict_exact_call_successes"]} strict 1.5B failures and {h3["upstream_attributed_strict_failures"]}/{180 - h3["strict_exact_call_successes"]} strict 3B failures. `verbatim` is essential: removing it collapsed 1.5B strict exact calls from {_pct(h15["strict_exact_call_accuracy"])} to {_pct(ablations["generated_domain_semantic_without_verbatim"]["strict_exact_call_accuracy"])}.

No Yuki tool was executed. The benchmark had no execution API, both Hammer reports contain the exact same frozen Stage A checksum, and models were loaded one at a time.

## Controls and dataset

- Canonical source: frozen 450-case Phase 2 dataset, SHA-256 `{comparison["dataset"]["source_sha256"]}`.
- Derived version: `{comparison["dataset"]["derived_version"]}`, SHA-256 `{comparison["dataset"]["derived_sha256"]}`.
- Selected set: deterministic `fallback-stratified-180`, exactly 10 cases per tool, preserving source IDs and original gold calls.
- The 18-case smoke reached only 88.9% strict delegation validity after the prompt adjustment, so the predeclared 180-case fallback was used instead of all 450.
- Stage A: one stateless generation per original request, thinking disabled, 160-token ceiling, complete-JSON stopping, no output repair.
- Stage B: official/native Hammer prompt and fenced tool-call array, one stateless generation per valid delegation, strict normalization and schema validation, no repair.
- The Stage A common-prefix cache reused only the identical rendered system-prefix computation. Previous request tokens and generations were trimmed before every case.
- This tool-centric set contains no no-tool examples and no conversation-context cases; it cannot measure those production decisions.

## Stage A — temporary surrogate delegation

| Metric | Result |
|---|---:|
| Valid delegation | {stage_a["valid_delegations"]}/180 ({_pct(stage_a["valid_delegation_rate"])}) |
| Malformed delegation | {stage_a["malformed_delegations"]}/180 ({_pct(stage_a["malformed_delegation_rate"])}) |
| Exact domain, all cases | {stage_a["correct_domain_hints"]}/180 ({_pct(stage_a["domain_hint_accuracy"])}) |
| Exact domain, valid delegations | {_pct(stage_a["domain_hint_accuracy_given_valid_delegation"])} |
| Deterministic semantic-request score | {_pct(stage_a["semantic_request_quality_rate"])} |
| Average `verbatim` recall | {_pct(stage_a["average_verbatim_recall"])} |
| Average `verbatim` precision | {_pct(stage_a["average_verbatim_precision"])} |
| Whole source request reproduced in delegation | {stage_a["whole_source_request_reproduced_in_delegation"]}/180 |
| Final implementation-tool leakage | {_pct(stage_a["final_tool_leakage_rate"])} |
| Average / p50 / p95 latency | {_seconds(stage_a["average_latency_ms"])} / {_seconds(stage_a["p50_latency_ms"])} / {_seconds(stage_a["p95_latency_ms"])} |
| Average generated tokens / tok/s | {stage_a["average_generated_tokens"]:.1f} / {stage_a["average_tokens_per_second"]:.2f} |
| Peak MLX memory | {stage_a["peak_mlx_memory_gb"]:.2f} GB |

| Expected domain | N | Valid | Exact domain | Verbatim recall | Verbatim precision |
|---|---:|---:|---:|---:|---:|
{chr(10).join(domain_rows)}

The nine format failures were four scalar `verbatim` values instead of arrays, four extra closing braces, and one invalidly escaped shell string. They were rejected unchanged. In seven valid cases, the surrogate copied the whole source request into `request` or `verbatim`; Hammer therefore saw text identical to the raw request through Stage A output even though the Stage B harness never injected the raw request separately. The strict semantic-request cue scorer is intentionally transparent but conservative: 42 cases that it marked false still became strict-correct Hammer 1.5B calls, so its 61.7% score should not be read as a calibrated semantic-quality estimate.

The sharpest Stage A weaknesses were preservation cases (59.4% domain accuracy), the computer domain (66.7%), normal `read` formatting, Yuki write/append payload splitting, and recall/question span selection. Zero generated semantic requests leaked implementation-level tool names.

## Stage B and end-to-end results

All accuracy below is end to end over the same 180 original requests. Invalid or wrong-domain Stage A output remains a failure; conditional columns isolate Hammer after a correct domain was supplied.

| Metric | Hammer 1.5B | Hammer 3B |
|---|---:|---:|
| Tool selection | {_pct(h15["tool_selection_accuracy"])} | {_pct(h3["tool_selection_accuracy"])} |
| Tool selection, correct domain | {_pct(h15["tool_selection_accuracy_when_domain_correct"])} | {_pct(h3["tool_selection_accuracy_when_domain_correct"])} |
| Strict arguments, correct tool | {_pct(h15["strict_argument_accuracy_when_tool_correct"])} | {_pct(h3["strict_argument_accuracy_when_tool_correct"])} |
| Preservation-aware arguments, correct tool | {_pct(h15["preservation_aware_argument_accuracy_when_tool_correct"])} | {_pct(h3["preservation_aware_argument_accuracy_when_tool_correct"])} |
| Strict exact call | **{_pct(h15["strict_exact_call_accuracy"])}** ({h15["strict_exact_call_successes"]}/180) | {_pct(h3["strict_exact_call_accuracy"])} ({h3["strict_exact_call_successes"]}/180) |
| Preservation-aware exact call | **{_pct(h15["preservation_aware_exact_call_accuracy"])}** | {_pct(h3["preservation_aware_exact_call_accuracy"])} |
| Strict exact call, correct domain | **{_pct(h15["strict_exact_call_accuracy_when_domain_correct"])}** | {_pct(h3["strict_exact_call_accuracy_when_domain_correct"])} |
| Schema valid | {_pct(h15["schema_valid_rate"])} | {_pct(h3["schema_valid_rate"])} |
| Malformed / conversational Hammer output | 0% / 0% | 0% / 0% |
| Dispatcher average / p50 / p95 | {_seconds(h15["average_dispatcher_latency_ms"])} / {_seconds(h15["p50_dispatcher_latency_ms"])} / {_seconds(h15["p95_dispatcher_latency_ms"])} | {_seconds(h3["average_dispatcher_latency_ms"])} / {_seconds(h3["p50_dispatcher_latency_ms"])} / {_seconds(h3["p95_dispatcher_latency_ms"])} |
| Generated tok/s | {h15["average_generation_tokens_per_second"]:.2f} | {h3["average_generation_tokens_per_second"]:.2f} |
| Peak MLX memory | {h15["peak_mlx_memory_gb"]:.2f} GB | {h3["peak_mlx_memory_gb"]:.2f} GB |
| End-to-end average latency, resident models | {_seconds(primary["end_to_end_performance"]["hammer_1_5b"]["average_latency_ms"])} | {_seconds(primary["end_to_end_performance"]["hammer_3b"]["average_latency_ms"])} |

MLX memory was measured with one model loaded at a time and must not be interpreted as a measured simultaneous-residency total. Model load time is excluded from per-case latency.

### Accuracy by case category

| Category | N | 1.5B strict exact | 3B strict exact |
|---|---:|---:|---:|
{chr(10).join(category_rows)}

Excluding the 18 tagged adversarial/ambiguous cases does not improve the result: strict exact calls are {_pct(h15["non_ambiguous"]["strict_exact_call_accuracy"])} for 1.5B and {_pct(h3["non_ambiguous"]["strict_exact_call_accuracy"])} for 3B. Preservation cases are the weakest category for both at 50.0%.

### Primary strict-failure decomposition

| Primary attribution | Hammer 1.5B | Hammer 3B |
|---|---:|---:|
{chr(10).join(failure_rows)}
| Ambiguous/adversarial failure (secondary tag) | {h15["ambiguous_or_adversarial_strict_failures"]} | {h3["ambiguous_or_adversarial_strict_failures"]} |

This precedence attributes lost exact text to Stage A before assigning a downstream wrong-argument category. It is an auditable primary attribution, not proof that every lost-verbatim case alone caused the final failure.

## Same-set selector comparison

The old deterministic selector can be compared fairly on these exact 180 IDs using the frozen Phase 2 Hammer 3B report.

| Same 180 cases | Old raw-text selector | Temporary surrogate `domain_hint` | Change |
|---|---:|---:|---:|
| Exact domain | {_pct(old["exact_domain_selection_accuracy"])} | {_pct(stage_a["domain_hint_accuracy"])} | {selector_change["exact_domain_accuracy_delta"] * 100:+.1f} pp |
| Correct tool available | {_pct(old["correct_tool_in_selected_subset_rate"])} | {_pct(stage_a["domain_hint_accuracy"])} | {selector_change["correct_tool_coverage_delta"] * 100:+.1f} pp |
| Hammer 3B strict exact, full pipeline | {_pct(old["hammer_3b_strict_exact_call_accuracy"])} | {_pct(h3["strict_exact_call_accuracy"])} | {selector_change["hammer_3b_strict_exact_call_delta"] * 100:+.1f} pp |

Semantic main-brain delegation therefore substantially improves held-out schema selection and complete calls. It does **not** fully solve selector generalization: 12.2% of cases still lacked the correct generated domain, and the current result pays a {_seconds(stage_a["average_latency_ms"])} Stage A inference cost.

## Hammer 1.5B ablations

| Domain source | Dispatcher input | Strict exact | Preservation-aware exact | Avg dispatcher |
|---|---|---:|---:|---:|
| generated | semantic + `verbatim` (primary) | {_pct(h15["strict_exact_call_accuracy"])} | {_pct(h15["preservation_aware_exact_call_accuracy"])} | {_seconds(h15["average_dispatcher_latency_ms"])} |
| generated | semantic only, no `verbatim` | {_pct(ablations["generated_domain_semantic_without_verbatim"]["strict_exact_call_accuracy"])} | {_pct(ablations["generated_domain_semantic_without_verbatim"]["preservation_aware_exact_call_accuracy"])} | {_seconds(ablations["generated_domain_semantic_without_verbatim"]["average_dispatcher_latency_ms"])} |
| generated | raw user request | {_pct(ablations["generated_domain_raw_user"]["strict_exact_call_accuracy"])} | {_pct(ablations["generated_domain_raw_user"]["preservation_aware_exact_call_accuracy"])} | {_seconds(ablations["generated_domain_raw_user"]["average_dispatcher_latency_ms"])} |
| gold/oracle | raw user request | {_pct(ablations["gold_domain_raw_user"]["strict_exact_call_accuracy"])} | {_pct(ablations["gold_domain_raw_user"]["preservation_aware_exact_call_accuracy"])} | {_seconds(ablations["gold_domain_raw_user"]["average_dispatcher_latency_ms"])} |

- `verbatim` is decisively useful: primary beat no-`verbatim` on 87 discordant strict cases and lost only 1 (`p<1e-23`).
- Semantic + `verbatim` beat raw text under the same generated domains by 10.0 points strict; paired discordance was 26 wins to 8 losses (`p={ablations["paired_primary_vs_raw_generated_strict"]["exact_mcnemar_p"]:.4f}`).
- Replacing generated domains with gold domains raised raw-text strict accuracy from {_pct(ablations["generated_domain_raw_user"]["strict_exact_call_accuracy"])} to {_pct(ablations["gold_domain_raw_user"]["strict_exact_call_accuracy"])}; all 17 discordant strict cases favored gold.
- Primary semantic delegation and same-set gold-domain/raw-user control were effectively tied ({_pct(h15["strict_exact_call_accuracy"])} vs {_pct(ablations["gold_domain_raw_user"]["strict_exact_call_accuracy"])}, paired `p=1.0`). That does not mean Stage A domains were oracle-perfect; semantic normalization and explicit exact strings recovered different cases while malformed/wrong domains lost others.

## Oracle reference and gap

These historical oracle numbers use the different 96-case targeted suite. They are included because they were requested, but subtracting them from this 180-case result is descriptive—not a paired reliability estimate.

| Model | Production interface, 180 | Production, correct domain | Targeted logical oracle, 96 | Targeted focused oracle, 96 |
|---|---:|---:|---:|---:|
| Hammer 1.5B | {_pct(h15["strict_exact_call_accuracy"])} | {_pct(h15["strict_exact_call_accuracy_when_domain_correct"])} | {_pct(oracle["hammer_1_5b"]["logical-domains"]["strict_exact_call_accuracy"])} | {_pct(oracle["hammer_1_5b"]["targeted-confusions"]["strict_exact_call_accuracy"])} |
| Hammer 3B | {_pct(h3["strict_exact_call_accuracy"])} | {_pct(h3["strict_exact_call_accuracy_when_domain_correct"])} | {_pct(oracle["hammer_3b"]["logical-domains"]["strict_exact_call_accuracy"])} | {_pct(oracle["hammer_3b"]["targeted-confusions"]["strict_exact_call_accuracy"])} |

The same-set 1.5B gold-domain/raw-user control is the stronger local comparison: it scored {_pct(ablations["gold_domain_raw_user"]["strict_exact_call_accuracy"])}, essentially the same aggregate as the primary interface but with substantially different per-case outcomes. No same-set 3B gold-domain ablation was run.

## Answers and next step

1. **Does semantic main-brain delegation solve selector generalization?** Partly. It improves correct schema coverage by {selector_change["correct_tool_coverage_delta"] * 100:.1f} points on the same cases, but {_pct(1 - stage_a["domain_hint_accuracy"])} still lack a correct domain.
2. **How accurate is `domain_hint`?** {_pct(stage_a["domain_hint_accuracy"])} overall and {_pct(stage_a["domain_hint_accuracy_given_valid_delegation"])} after valid JSON.
3. **How reliable is `verbatim`?** {_pct(stage_a["average_verbatim_recall"])} average recall and {_pct(stage_a["average_verbatim_precision"])} precision. It is not yet reliable enough, but the ablation proves the field itself is essential.
4. **How close is production routing to oracle subsets?** For 1.5B it matches the same-set gold-domain/raw control in aggregate; comparison to the historical 96-case oracle is not apples-to-apples. Stage A errors are now the dominant gap.
5. **Does Hammer 1.5B become viable?** It becomes the stronger efficiency candidate in this isolated interface test: slightly higher measured accuracy than 3B at about half the latency and memory. {_pct(h15["strict_exact_call_accuracy"])} is still not production-grade evidence.
6. **Does Hammer 3B justify its cost here?** No measured accuracy benefit in this run. It remains a valid comparator, but this evidence alone does not justify its roughly 2× dispatch latency/memory.
7. **Can the old raw-text selector be removed?** The main-brain route is much stronger on the same subset, but replacement should wait for the intended 27B rerun, full 450 evaluation, and no-tool/context coverage.
8. **Which component dominates failures?** Stage A: malformed JSON, wrong domains, and lost exact strings account for roughly four-fifths of strict failures under the primary precedence.
9. **What next?** Rerun only Stage A with the intended Qwen3.8-27B using this same derived dataset/version and contract, freeze its new delegations, then reuse the exact Hammer 1.5B/3B evaluation. If its smoke is healthy, run all 450. Add separate no-tool and real conversation-context cases before any production recommendation.

No Yuki production routing was modified, and no integration recommendation is made.

## Artifacts

- `production-delegation-comparison-v1.json` — consolidated machine-readable results, paired analyses, prior-selector comparison, oracle reference, checksums, and failure examples.
- `production-delegation-artifact-manifest-v1.json` — SHA-256 manifest for datasets, reports, harness sources, and failure JSONL files.
- `stage-a/` — frozen temporary-surrogate delegations and raw generations.
- `stage-b/` — primary Hammer reports and failure JSONL.
- `ablations/` — three Hammer 1.5B control reports and failure JSONL.
"""


def main() -> None:
    comparison = build_comparison()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_PATH.write_text(render_markdown(comparison), encoding="utf-8")

    manifest_paths = {
        **PATHS,
        "comparison": COMPARISON_PATH,
        "human_report": MARKDOWN_PATH,
        "source_delegation_contract": ROOT / "delegation_contract.py",
        "source_mainbrain_runner": ROOT / "delegation_mainbrain.py",
        "source_validation_dispatcher": ROOT / "delegation_dispatcher.py",
        "source_dataset_builder": ROOT / "build_production_delegation_dataset.py",
        "tests": ROOT / "tests" / "test_production_delegation.py",
    }
    for path in sorted((REPORT_ROOT / "stage-b").glob("*failures.jsonl")):
        manifest_paths[f"failures_{path.stem}"] = path
    for path in sorted((REPORT_ROOT / "ablations").glob("*failures.jsonl")):
        manifest_paths[f"failures_{path.stem}"] = path
    manifest = {
        "version": "production-delegation-artifact-manifest-v1",
        "main_brain_is_temporary_surrogate": True,
        "main_brain_label": comparison["main_brain_label"],
        "files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in manifest_paths.items()
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(_sha256(COMPARISON_PATH))


if __name__ == "__main__":
    main()
