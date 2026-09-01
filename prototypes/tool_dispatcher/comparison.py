"""Compare completed dispatcher reports without rerunning model inference."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from .argument_scoring import score_arguments
from .dialects import native_dialect_for_model


def _model_label(model_id: str) -> str:
    normalized = model_id.lower()
    if "hammer2.1-3b" in normalized:
        return "Hammer2.1-3B"
    if "hammer2.1-1.5b" in normalized:
        return "Hammer2.1-1.5B"
    if "xlam-2-1b-fc-r" in normalized:
        return "xLAM-2-1b-fc-r"
    if "xlam-1b-fc-r" in normalized:
        return "xLAM-1b-fc-r"
    return Path(model_id).name


def _throughput(case_results: list[dict[str, Any]]) -> dict[str, float | None]:
    rates: list[float] = []
    measured_tokens = 0
    estimated_seconds = 0.0
    for case in case_results:
        generation = case.get("generation") or {}
        rate = generation.get("tokens_per_second")
        tokens = generation.get("generated_tokens") or 0
        if rate and tokens:
            rates.append(rate)
            measured_tokens += tokens
            estimated_seconds += tokens / rate
    return {
        "aggregate_generation_tokens_per_second": (
            measured_tokens / estimated_seconds if estimated_seconds else None
        ),
        "median_backend_reported_tokens_per_second": (
            median(rates) if rates else None
        ),
    }


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


def _latency(case_results: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [
        generation["latency_ms"]
        for case in case_results
        if (generation := case.get("generation"))
        and generation.get("latency_ms") is not None
    ]
    return {
        "p50_inference_latency_ms": _percentile(values, 0.50),
        "p95_inference_latency_ms": _percentile(values, 0.95),
    }


def _rescored_arguments(case_results: list[dict[str, Any]]) -> dict[str, float]:
    total = len(case_results)
    strict_arguments = 0
    equivalent_arguments = 0
    strict_calls = 0
    equivalent_calls = 0
    for case in case_results:
        scoring = score_arguments(
            case["expected_tool"],
            case["expected_arguments"],
            case.get("actual_arguments"),
        )
        tool_correct = case.get("actual_tool") == case["expected_tool"]
        schema_valid = bool(case.get("schema_valid"))
        strict_arguments += int(scoring["strict"])
        equivalent_arguments += int(scoring["execution_equivalent"])
        strict_calls += int(tool_correct and scoring["strict"] and schema_valid)
        equivalent_calls += int(
            tool_correct and scoring["execution_equivalent"] and schema_valid
        )
    return {
        "strict_argument_accuracy": strict_arguments / total,
        "execution_equivalent_argument_accuracy": equivalent_arguments / total,
        "strict_exact_call_accuracy": strict_calls / total,
        "execution_equivalent_call_accuracy": equivalent_calls / total,
    }


def _metrics_by_tag(case_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    tags = sorted({tag for case in case_results for tag in case.get("tags", [])})
    for tag in tags:
        tagged = [case for case in case_results if tag in case.get("tags", [])]
        total = len(tagged)
        strict_arguments = 0
        equivalent_arguments = 0
        strict_calls = 0
        equivalent_calls = 0
        tool_calls = 0
        schema_valid = 0
        malformed = 0
        for case in tagged:
            scoring = score_arguments(
                case["expected_tool"],
                case["expected_arguments"],
                case.get("actual_arguments"),
            )
            tool_correct = case.get("actual_tool") == case["expected_tool"]
            valid = bool(case.get("schema_valid"))
            tool_calls += int(tool_correct)
            strict_arguments += int(scoring["strict"])
            equivalent_arguments += int(scoring["execution_equivalent"])
            strict_calls += int(tool_correct and scoring["strict"] and valid)
            equivalent_calls += int(
                tool_correct and scoring["execution_equivalent"] and valid
            )
            schema_valid += int(valid)
            malformed += int(bool(case.get("malformed")))
        metrics[tag] = {
            "total_cases": total,
            "tool_selection_accuracy": tool_calls / total,
            "strict_argument_accuracy": strict_arguments / total,
            "execution_equivalent_argument_accuracy": equivalent_arguments / total,
            "strict_exact_call_accuracy": strict_calls / total,
            "execution_equivalent_call_accuracy": equivalent_calls / total,
            "schema_valid_output_rate": schema_valid / total,
            "malformed_output_rate": malformed / total,
        }
    return metrics


def _oracle_effects(models: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        by_model.setdefault(model["model_id"], []).append(model)

    effects: dict[str, Any] = {}
    for runs in by_model.values():
        baseline = next(
            (run for run in runs if run["oracle_profile"] is None), None
        )
        oracle_runs = [run for run in runs if run["oracle_profile"] is not None]
        if baseline is None or not oracle_runs:
            continue
        profiles = {}
        for run in oracle_runs:
            profiles[run["oracle_profile"]] = {
                "tool_selection_accuracy_delta": (
                    run["tool_selection_accuracy"]
                    - baseline["tool_selection_accuracy"]
                ),
                "strict_argument_accuracy_delta": (
                    run["strict_argument_accuracy"]
                    - baseline["strict_argument_accuracy"]
                ),
                "execution_equivalent_argument_accuracy_delta": (
                    run["execution_equivalent_argument_accuracy"]
                    - baseline["execution_equivalent_argument_accuracy"]
                ),
                "strict_exact_call_accuracy_delta": (
                    run["strict_exact_call_accuracy"]
                    - baseline["strict_exact_call_accuracy"]
                ),
                "execution_equivalent_call_accuracy_delta": (
                    run["execution_equivalent_call_accuracy"]
                    - baseline["execution_equivalent_call_accuracy"]
                ),
                "schema_valid_output_rate_delta": (
                    run["schema_valid_output_rate"]
                    - baseline["schema_valid_output_rate"]
                ),
                "average_latency_reduction_ms": (
                    baseline["average_inference_latency_ms"]
                    - run["average_inference_latency_ms"]
                ),
                "p50_latency_reduction_ms": (
                    baseline["p50_inference_latency_ms"]
                    - run["p50_inference_latency_ms"]
                ),
                "p95_latency_reduction_ms": (
                    baseline["p95_inference_latency_ms"]
                    - run["p95_inference_latency_ms"]
                ),
            }
        effects[baseline["label"]] = {
            "baseline": baseline["experiment_label"],
            "profiles": profiles,
        }
    return effects


def compare_reports(paths: list[str | Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("Comparison requires at least two benchmark reports")
    loaded = []
    for path in paths:
        report_path = Path(path)
        with report_path.open(encoding="utf-8") as handle:
            loaded.append((report_path, json.load(handle)))

    reference_cases = [
        (
            case["case_id"],
            case["request"],
            case["expected_tool"],
            case["expected_arguments"],
        )
        for case in loaded[0][1]["case_results"]
    ]
    models = []
    for path, report in loaded:
        cases = report["case_results"]
        signature = [
            (
                case["case_id"],
                case["request"],
                case["expected_tool"],
                case["expected_arguments"],
            )
            for case in cases
        ]
        if signature != reference_cases:
            raise ValueError(f"Benchmark cases do not match: {path}")
        configuration = report["configuration"]
        metrics = report["metrics"]
        model_id = configuration.get("model_id") or next(
            (
                case["generation"]["model_id"]
                for case in cases
                if case.get("generation")
            ),
            "unknown",
        )
        label = _model_label(model_id)
        oracle_profile = configuration.get("oracle_profile")
        experiment_label = (
            f"{label} — {oracle_profile}" if oracle_profile else label
        )
        normalized_outputs = sum(
            bool((case.get("generation") or {}).get("normalized_text"))
            for case in cases
        )
        json_complete = sum(
            (case.get("generation") or {}).get("finish_reason") == "json_complete"
            for case in cases
        )
        model = {
            "label": label,
            "experiment_label": experiment_label,
            "model_id": model_id,
            "report_path": str(path),
            "native_dialect": configuration.get("native_dialect")
            or native_dialect_for_model(model_id),
            "oracle_profile": oracle_profile,
            "oracle_subsets": configuration.get("oracle_subsets", {}),
            "total_cases": metrics["total_cases"],
            "tool_selection_accuracy": metrics["tool_selection_accuracy"],
            "argument_accuracy": metrics["argument_accuracy"],
            "strict_argument_accuracy": metrics.get(
                "strict_argument_accuracy", metrics["argument_accuracy"]
            ),
            "exact_call_accuracy": metrics["exact_call_accuracy"],
            "schema_valid_output_rate": metrics["schema_valid_output_rate"],
            "malformed_output_rate": metrics["malformed_output_rate"],
            "average_inference_latency_ms": metrics[
                "average_inference_latency_ms"
            ],
            "end_to_end_generated_tokens_per_second": metrics.get(
                "end_to_end_generated_tokens_per_second"
            ),
            "normalized_output_count": normalized_outputs,
            "json_complete_rate": json_complete / len(cases),
            "peak_memory_gb": max(
                (
                    (case.get("generation") or {}).get("peak_memory_gb") or 0
                    for case in cases
                ),
                default=0,
            ),
            "process_rss_mb": max(
                (
                    (case.get("generation") or {}).get("process_rss_mb") or 0
                    for case in cases
                ),
                default=0,
            ),
            "failures": sum(not case["exact_call"] for case in cases),
            "failures_by_expected_tool": report.get(
                "failures_by_expected_tool", {}
            ),
            "confusion_matrix": report.get("confusion_matrix", {}),
            "metrics_by_tag": _metrics_by_tag(cases),
            **_rescored_arguments(cases),
            **_latency(cases),
            **_throughput(cases),
        }
        models.append(model)

    ranking = sorted(
        models,
        key=lambda model: (
            model["exact_call_accuracy"],
            model["tool_selection_accuracy"],
        ),
        reverse=True,
    )
    return {
        "comparison": {
            "case_count": len(reference_cases),
            "same_cases_verified": True,
            "routing_only": True,
            "tool_execution_attempts": 0,
            "reliability_conclusion_allowed": False,
            "phase": "phase_1_smoke",
        },
        "ranking_by_exact_call_accuracy": [
            model["experiment_label"] for model in ranking
        ],
        "oracle_effects_by_model": _oracle_effects(models),
        "models": models,
    }


def write_comparison(report: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
