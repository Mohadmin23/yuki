"""Execution-locked benchmark for automatic Yuki schema selection."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from .argument_scoring import score_arguments
from .automatic_selector import AutomaticDomainSelector
from .benchmark import BenchmarkCase
from .dialects import native_dialect_for_model
from .dispatcher import Dispatcher


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


def _latency_summary(values: list[float], prefix: str) -> dict[str, float | None]:
    return {
        f"average_{prefix}_latency_ms": mean(values) if values else None,
        f"p50_{prefix}_latency_ms": _percentile(values, 0.50),
        f"p95_{prefix}_latency_ms": _percentile(values, 0.95),
    }


def _primary_failure_stage(
    *,
    correct_tool_available: bool,
    malformed: bool,
    expected_tool_selected: bool,
    schema_valid: bool,
    equivalent_arguments_correct: bool,
    exact_domain: bool,
) -> str | None:
    """Assign the earliest actionable stage responsible for a failed call."""

    if not correct_tool_available:
        return "tool_unavailable"
    if malformed:
        return "malformed_output"
    if not expected_tool_selected:
        return "tool_selection"
    if not schema_valid:
        return "schema_validation"
    if not equivalent_arguments_correct:
        return "argument_generation"
    if not exact_domain:
        return "domain_selection"
    return None


class AutomaticSubsetBenchmarkRunner:
    """Select schemas and score one dispatch generation, with no execution API.

    This runner intentionally has no ``execute``, side-effect, shell, or tool-result
    parameters. The only dispatcher entry point it uses always stops at validation.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        selector: AutomaticDomainSelector | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.selector = selector or AutomaticDomainSelector(dispatcher.registry)

    def run(
        self,
        cases: Iterable[BenchmarkCase],
        *,
        dataset: str,
        output_mode: str = "native",
        limit: int | None = None,
    ) -> dict[str, Any]:
        scoped = list(cases)
        if limit is not None:
            scoped = scoped[: max(limit, 0)]
        if not scoped:
            raise ValueError("Automatic benchmark requires at least one case")

        case_results: list[dict[str, Any]] = []
        failure_records: list[dict[str, Any]] = []
        selector_confusion: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        selector_failures: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        dispatcher_confusion: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        failures_by_tool: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        selector_latencies: list[float] = []
        dispatcher_pipeline_latencies: list[float] = []
        inference_latencies: list[float] = []
        total_latencies: list[float] = []
        selected_domain_counts: list[int] = []
        offered_schema_counts: list[int] = []
        generation_rates: list[float] = []
        rate_token_total = 0
        estimated_generation_seconds = 0.0
        generated_tokens = 0
        inference_seconds = 0.0
        peak_memory_values: list[float] = []

        exact_domain_count = 0
        coverage_count = 0
        tool_correct_count = 0
        strict_arguments_count = 0
        equivalent_arguments_count = 0
        strict_call_count = 0
        equivalent_call_count = 0
        schema_valid_count = 0
        malformed_count = 0
        end_to_end_strict_count = 0
        end_to_end_equivalent_count = 0
        end_to_end_valid_success_count = 0

        for index, case in enumerate(scoped):
            total_started = time.perf_counter_ns()
            selection = self.selector.select_domains(case.request)
            expected_domain = self.selector.expected_domain_for_tool(
                case.expected_tool
            )
            expected_domains = (expected_domain,)
            exact_domain = selection.domains == expected_domains
            correct_tool_available = case.expected_tool in selection.tools

            dispatcher_started = time.perf_counter_ns()
            result = self.dispatcher.route_and_validate(
                case.request,
                selected=selection.tools,
                output_mode=output_mode,
            )
            dispatcher_pipeline_ms = (
                time.perf_counter_ns() - dispatcher_started
            ) / 1_000_000
            total_latency_ms = (
                time.perf_counter_ns() - total_started
            ) / 1_000_000

            actual_tool = result["selected_tool"]
            actual_arguments = result["parsed_arguments"]
            expected_tool_selected = actual_tool == case.expected_tool
            argument_scoring = score_arguments(
                case.expected_tool,
                case.expected_arguments,
                actual_arguments,
            )
            strict_arguments = bool(argument_scoring["strict"])
            equivalent_arguments = bool(
                argument_scoring["execution_equivalent"]
            )
            schema_valid = bool(result["validation"]["passed"])
            malformed = bool(result["parse"]["malformed"])
            strict_call = expected_tool_selected and strict_arguments and schema_valid
            equivalent_call = (
                expected_tool_selected and equivalent_arguments and schema_valid
            )
            end_to_end_strict = correct_tool_available and strict_call
            end_to_end_equivalent = correct_tool_available and equivalent_call
            end_to_end_valid_success = (
                correct_tool_available and expected_tool_selected and schema_valid
            )

            failure_stages: list[str] = []
            if not exact_domain:
                failure_stages.append("domain_selection")
            if not correct_tool_available:
                failure_stages.append("tool_unavailable")
            if malformed:
                failure_stages.append("malformed_output")
            if correct_tool_available and not malformed and not expected_tool_selected:
                failure_stages.append("tool_selection")
            if expected_tool_selected and not schema_valid:
                failure_stages.append("schema_validation")
            if (
                expected_tool_selected
                and schema_valid
                and not equivalent_arguments
            ):
                failure_stages.append("argument_generation")
            primary_failure = _primary_failure_stage(
                correct_tool_available=correct_tool_available,
                malformed=malformed,
                expected_tool_selected=expected_tool_selected,
                schema_valid=schema_valid,
                equivalent_arguments_correct=equivalent_arguments,
                exact_domain=exact_domain,
            )

            exact_domain_count += int(exact_domain)
            coverage_count += int(correct_tool_available)
            tool_correct_count += int(expected_tool_selected)
            strict_arguments_count += int(strict_arguments)
            equivalent_arguments_count += int(equivalent_arguments)
            strict_call_count += int(strict_call)
            equivalent_call_count += int(equivalent_call)
            schema_valid_count += int(schema_valid)
            malformed_count += int(malformed)
            end_to_end_strict_count += int(end_to_end_strict)
            end_to_end_equivalent_count += int(end_to_end_equivalent)
            end_to_end_valid_success_count += int(end_to_end_valid_success)

            selected_domain_counts.append(len(selection.domains))
            offered_schema_counts.append(len(selection.tools))
            selector_latencies.append(selection.latency_ms)
            dispatcher_pipeline_latencies.append(dispatcher_pipeline_ms)
            total_latencies.append(total_latency_ms)
            selected_domain_label = "+".join(selection.domains)
            selector_confusion[expected_domain][selected_domain_label] += 1
            dispatcher_confusion[case.expected_tool][actual_tool or "<none>"] += 1
            if not exact_domain:
                selector_failures[case.expected_tool]["domain_selection"] += 1
            if not correct_tool_available:
                selector_failures[case.expected_tool]["tool_unavailable"] += 1
            for stage in failure_stages:
                failures_by_tool[case.expected_tool][stage] += 1

            generation = result.get("generation") or {}
            latency = generation.get("latency_ms")
            if latency is not None:
                inference_latencies.append(latency)
                inference_seconds += latency / 1000
            token_count = generation.get("generated_tokens") or 0
            generated_tokens += token_count
            rate = generation.get("tokens_per_second")
            if rate is not None:
                generation_rates.append(rate)
            if rate and token_count:
                rate_token_total += token_count
                estimated_generation_seconds += token_count / rate
            peak_memory = generation.get("peak_memory_gb")
            if peak_memory is not None:
                peak_memory_values.append(peak_memory)

            generation_for_report = {
                key: value
                for key, value in generation.items()
                if key != "rendered_prompt"
            }
            case_id = case.case_id or f"case-{index + 1}"
            case_result = {
                "index": index,
                "case_id": case_id,
                "tags": list(case.tags),
                "dataset": dataset,
                "request": case.request,
                "expected_domain": expected_domain,
                "expected_tool": case.expected_tool,
                "expected_arguments": case.expected_arguments,
                "selected_domains": list(selection.domains),
                "selector_strategy": selection.strategy,
                "selector_matched_rules": list(selection.matched_rules),
                "exact_domain_selection": exact_domain,
                "offered_tools": list(selection.tools),
                "offered_schema_count": len(selection.tools),
                "correct_tool_in_selected_subset": correct_tool_available,
                "actual_tool": actual_tool,
                "actual_arguments": actual_arguments,
                "tool_correct": expected_tool_selected,
                "strict_arguments_correct": strict_arguments,
                "execution_equivalent_arguments_correct": equivalent_arguments,
                "argument_scoring": argument_scoring,
                "schema_valid": schema_valid,
                "malformed": malformed,
                "strict_exact_call": strict_call,
                "execution_equivalent_exact_call": equivalent_call,
                "end_to_end_strict_exact_call": end_to_end_strict,
                "end_to_end_execution_equivalent_exact_call": (
                    end_to_end_equivalent
                ),
                "end_to_end_schema_valid_successful_call": (
                    end_to_end_valid_success
                ),
                "failure_stage": primary_failure,
                "failure_stages": failure_stages,
                "selector_latency_ms": selection.latency_ms,
                "dispatcher_pipeline_latency_ms": dispatcher_pipeline_ms,
                "total_latency_ms": total_latency_ms,
                "native_dialect": result["native_dialect"],
                "generation": generation_for_report,
                "parse": result["parse"],
                "normalized_call": result["canonical_call"],
                "validation": result["validation"],
                "execution": result["execution"],
            }
            case_results.append(case_result)

            if failure_stages:
                failure_records.append(
                    {
                        "case_id": case_id,
                        "dataset": dataset,
                        "model": self.dispatcher.backend.model_id,
                        "request": case.request,
                        "expected_domain": expected_domain,
                        "expected_tool": case.expected_tool,
                        "expected_arguments": case.expected_arguments,
                        "selected_domains": list(selection.domains),
                        "selector_strategy": selection.strategy,
                        "selector_matched_rules": list(selection.matched_rules),
                        "exact_domain_selection": exact_domain,
                        "offered_tools": list(selection.tools),
                        "schema_subset": list(selection.tools),
                        "correct_tool_was_available": correct_tool_available,
                        "raw_generation": generation.get("raw_text"),
                        "normalized_generation": generation.get("normalized_text"),
                        "parsed_call": result["parse"].get("decoded"),
                        "normalized_call": result["canonical_call"],
                        "schema_validation": result["validation"],
                        "strict_correct": end_to_end_strict,
                        "equivalent_correct": end_to_end_equivalent,
                        "argument_scoring": argument_scoring,
                        "failure_stage": primary_failure,
                        "failure_stages": failure_stages,
                        "selector_latency_ms": selection.latency_ms,
                        "dispatcher_pipeline_latency_ms": dispatcher_pipeline_ms,
                        "inference_latency_ms": latency,
                        "total_latency_ms": total_latency_ms,
                    }
                )

        total = len(scoped)
        available_total = coverage_count
        selector_metrics = {
            "exact_domain_selection_accuracy": exact_domain_count / total,
            "correct_tool_in_selected_subset_rate": coverage_count / total,
            "average_selected_domains": mean(selected_domain_counts),
            "average_offered_schemas": mean(offered_schema_counts),
            "minimum_offered_schemas": min(offered_schema_counts),
            "maximum_offered_schemas": max(offered_schema_counts),
            **_latency_summary(selector_latencies, "selector"),
        }
        dispatcher_metrics = {
            "tool_selection_accuracy": tool_correct_count / total,
            "tool_selection_accuracy_when_available": (
                tool_correct_count / available_total if available_total else None
            ),
            "strict_argument_accuracy": strict_arguments_count / total,
            "execution_equivalent_argument_accuracy": (
                equivalent_arguments_count / total
            ),
            "strict_exact_call_accuracy": strict_call_count / total,
            "execution_equivalent_exact_call_accuracy": (
                equivalent_call_count / total
            ),
            "schema_valid_output_rate": schema_valid_count / total,
            "malformed_output_rate": malformed_count / total,
            **_latency_summary(inference_latencies, "dispatcher"),
            **_latency_summary(
                dispatcher_pipeline_latencies, "dispatcher_pipeline"
            ),
            "aggregate_generation_tokens_per_second": (
                rate_token_total / estimated_generation_seconds
                if estimated_generation_seconds
                else None
            ),
            "mean_backend_reported_tokens_per_second": (
                mean(generation_rates) if generation_rates else None
            ),
            "median_backend_reported_tokens_per_second": (
                median(generation_rates) if generation_rates else None
            ),
            "end_to_end_generated_tokens_per_second": (
                generated_tokens / inference_seconds if inference_seconds else None
            ),
            "peak_mlx_memory_gb": (
                max(peak_memory_values) if peak_memory_values else None
            ),
        }
        end_to_end_metrics = {
            "strict_exact_call_accuracy": end_to_end_strict_count / total,
            "execution_equivalent_exact_call_accuracy": (
                end_to_end_equivalent_count / total
            ),
            "schema_valid_successful_call_rate": (
                end_to_end_valid_success_count / total
            ),
            **_latency_summary(total_latencies, "total"),
        }

        return {
            "configuration": {
                "experiment": "automatic_schema_selection",
                "exposure_condition": "automatic_subset",
                "selector_strategy": self.selector.strategy_name,
                "dataset": dataset,
                "output_mode": output_mode,
                "native_dialect": native_dialect_for_model(
                    self.dispatcher.backend.model_id
                ),
                "backend": self.dispatcher.backend.backend_name,
                "model_id": self.dispatcher.backend.model_id,
                "generation_count_per_case": 1,
                "stateless": True,
                "execute": False,
                "execution_capability": False,
                "tool_results_returned_to_model": False,
                "total_cases": total,
                "phase": "automatic_schema_selection_phase_1",
            },
            "selector_metrics": selector_metrics,
            "dispatcher_metrics": dispatcher_metrics,
            "end_to_end_metrics": end_to_end_metrics,
            "selector_confusion_matrix_by_expected_domain": {
                expected: dict(selected)
                for expected, selected in selector_confusion.items()
            },
            "selector_failures_by_expected_tool": {
                tool: dict(counts) for tool, counts in selector_failures.items()
            },
            "dispatcher_confusion_matrix": {
                expected: dict(actuals)
                for expected, actuals in dispatcher_confusion.items()
            },
            "failures_by_expected_tool": {
                tool: dict(counts) for tool, counts in failures_by_tool.items()
            },
            "case_results": case_results,
            "failure_records": failure_records,
        }


def write_automatic_failure_jsonl(
    report: dict[str, Any], path: str | Path
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in report["failure_records"]
        )
