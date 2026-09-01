"""Routing-only Phase 2 reliability metrics for the frozen Yuki dataset."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from .automatic_benchmark import AutomaticSubsetBenchmarkRunner
from .benchmark import BenchmarkCase
from .dispatcher import Dispatcher


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    """Return a deterministic two-sided 95% Wilson score interval."""

    if total <= 0:
        return {
            "successes": successes,
            "total": total,
            "confidence": 0.95,
            "lower": None,
            "upper": None,
        }
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "confidence": 0.95,
        "lower": center - margin,
        "upper": center + margin,
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


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "average": mean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _looks_conversational(case: dict[str, Any]) -> bool:
    if not case["malformed"]:
        return False
    generation = case.get("generation") or {}
    parser_text = (
        generation.get("normalized_text") or generation.get("raw_text") or ""
    ).lstrip()
    return bool(parser_text) and not parser_text.startswith(("{", "["))


def classify_failure(
    case: dict[str, Any], live_tool_names: Iterable[str]
) -> dict[str, Any]:
    """Classify one strict-call failure without semantic model judging."""

    categories: list[str] = []
    validation_errors = case["validation"].get("errors", [])
    parse_errors = case["parse"].get("errors", [])
    actual_tool = case["actual_tool"]
    correct_tool = case["tool_correct"]
    expected_available = case["correct_tool_in_selected_subset"]
    live_tools = set(live_tool_names)

    if not expected_available:
        categories.append("selector_tool_unavailable")
    if any("exactly one tool call" in error for error in parse_errors):
        categories.append("multiple_calls")
    if _looks_conversational(case):
        categories.append("conversational_output")
    if case["malformed"]:
        categories.append("malformed_output")
    if actual_tool is not None and actual_tool not in live_tools:
        categories.append("nonexistent_tool")
    if expected_available and actual_tool != case["expected_tool"]:
        categories.append("wrong_tool")
    if correct_tool:
        if any(error.startswith("Missing required argument:") for error in validation_errors):
            categories.append("missing_argument")
        if any(error.startswith("Unexpected arguments:") for error in validation_errors):
            categories.append("extra_argument")
        if not case["schema_valid"] and not any(
            category in categories for category in ("missing_argument", "extra_argument")
        ):
            categories.append("schema_invalid")
        if case["schema_valid"]:
            if (
                not case["strict_arguments_correct"]
                and case["execution_equivalent_arguments_correct"]
            ):
                categories.append("argument_rewrite")
            elif not case["execution_equivalent_arguments_correct"]:
                categories.append("wrong_argument")
    elif not case["schema_valid"] and not case["malformed"]:
        if any(error.startswith("Missing required argument:") for error in validation_errors):
            categories.append("missing_argument")
        if any(error.startswith("Unexpected arguments:") for error in validation_errors):
            categories.append("extra_argument")

    precedence = (
        "selector_tool_unavailable",
        "multiple_calls",
        "conversational_output",
        "malformed_output",
        "nonexistent_tool",
        "wrong_tool",
        "missing_argument",
        "extra_argument",
        "schema_invalid",
        "argument_rewrite",
        "wrong_argument",
    )
    primary = next((name for name in precedence if name in categories), None)
    return {
        "primary": primary,
        "secondary": [name for name in categories if name != primary],
        "all": categories,
        "diagnostics": (
            [] if case["exact_domain_selection"] else ["selector_domain_nonexact"]
        ),
    }


class Phase2ReliabilityRunner:
    """Run the frozen study with no execution controls or execution capability."""

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher
        self._automatic = AutomaticSubsetBenchmarkRunner(dispatcher)

    def run(
        self,
        cases: Iterable[BenchmarkCase],
        *,
        dataset_path: str | Path,
        expected_checksum: str,
        output_mode: str = "native",
    ) -> dict[str, Any]:
        actual_checksum = dataset_sha256(dataset_path)
        if actual_checksum != expected_checksum:
            raise RuntimeError(
                "Frozen Phase 2 dataset checksum mismatch: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
        scoped = list(cases)
        base = self._automatic.run(
            scoped,
            dataset="phase2-450",
            output_mode=output_mode,
        )
        return self._enrich(base, actual_checksum)

    def _enrich(self, report: dict[str, Any], checksum: str) -> dict[str, Any]:
        cases = report["case_results"]
        total = len(cases)
        live_names = self.dispatcher.registry.names
        category_counts = Counter(case["tags"][1] for case in cases)
        taxonomy_primary: Counter[str] = Counter()
        taxonomy_all: Counter[str] = Counter()
        taxonomy_by_tool: dict[str, Counter[str]] = defaultdict(Counter)
        taxonomy_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        failure_records: list[dict[str, Any]] = []

        for case in cases:
            taxonomy = classify_failure(case, live_names)
            case["phase2_failure"] = taxonomy
            if taxonomy["primary"]:
                taxonomy_primary[taxonomy["primary"]] += 1
            for category in taxonomy["all"]:
                taxonomy_all[category] += 1
                taxonomy_by_tool[case["expected_tool"]][category] += 1
                taxonomy_by_category[case["tags"][1]][category] += 1

            if not case["end_to_end_strict_exact_call"]:
                generation = case.get("generation") or {}
                failure_records.append(
                    {
                        "case_id": case["case_id"],
                        "dataset": "phase2-450",
                        "dataset_sha256": checksum,
                        "model": self.dispatcher.backend.model_id,
                        "request": case["request"],
                        "tags": case["tags"],
                        "expected_domain": case["expected_domain"],
                        "expected_tool": case["expected_tool"],
                        "expected_arguments": case["expected_arguments"],
                        "selected_domains": case["selected_domains"],
                        "offered_tools": case["offered_tools"],
                        "correct_tool_was_available": case[
                            "correct_tool_in_selected_subset"
                        ],
                        "raw_generation": generation.get("raw_text"),
                        "normalized_generation": generation.get("normalized_text"),
                        "parsed_native_output": case["parse"].get("decoded"),
                        "normalized_call": case["normalized_call"],
                        "actual_tool": case["actual_tool"],
                        "actual_arguments": case["actual_arguments"],
                        "schema_validation": case["validation"],
                        "tool_correct": case["tool_correct"],
                        "strict_arguments_correct": case[
                            "strict_arguments_correct"
                        ],
                        "execution_equivalent_arguments_correct": case[
                            "execution_equivalent_arguments_correct"
                        ],
                        "strict_correct": case["end_to_end_strict_exact_call"],
                        "equivalent_correct": case[
                            "end_to_end_execution_equivalent_exact_call"
                        ],
                        "argument_scoring": case["argument_scoring"],
                        "failure": taxonomy,
                        "selector_latency_ms": case["selector_latency_ms"],
                        "dispatcher_latency_ms": generation.get("latency_ms"),
                        "dispatcher_pipeline_latency_ms": case[
                            "dispatcher_pipeline_latency_ms"
                        ],
                        "total_latency_ms": case["total_latency_ms"],
                    }
                )

        tool_correct = sum(case["tool_correct"] for case in cases)
        strict_correct = sum(case["end_to_end_strict_exact_call"] for case in cases)
        equivalent_correct = sum(
            case["end_to_end_execution_equivalent_exact_call"] for case in cases
        )
        expected_available = [
            case for case in cases if case["correct_tool_in_selected_subset"]
        ]
        selected_correct_tool = [case for case in cases if case["tool_correct"]]
        core_metrics = {
            "correct_tool_in_selected_subset_rate": sum(
                case["correct_tool_in_selected_subset"] for case in cases
            )
            / total,
            "tool_selection_accuracy": tool_correct / total,
            "tool_selection_accuracy_when_expected_tool_available": (
                sum(case["tool_correct"] for case in expected_available)
                / len(expected_available)
                if expected_available
                else None
            ),
            "strict_argument_accuracy": sum(
                case["strict_arguments_correct"] for case in cases
            )
            / total,
            "execution_equivalent_argument_accuracy": sum(
                case["execution_equivalent_arguments_correct"] for case in cases
            )
            / total,
            "strict_argument_accuracy_when_expected_tool_selected": (
                sum(case["strict_arguments_correct"] for case in selected_correct_tool)
                / len(selected_correct_tool)
                if selected_correct_tool
                else None
            ),
            "execution_equivalent_argument_accuracy_when_expected_tool_selected": (
                sum(
                    case["execution_equivalent_arguments_correct"]
                    for case in selected_correct_tool
                )
                / len(selected_correct_tool)
                if selected_correct_tool
                else None
            ),
            "strict_exact_call_accuracy": strict_correct / total,
            "execution_equivalent_exact_call_accuracy": equivalent_correct / total,
            "strict_exact_call_accuracy_when_expected_tool_available": (
                sum(case["strict_exact_call"] for case in expected_available)
                / len(expected_available)
                if expected_available
                else None
            ),
            "execution_equivalent_exact_call_accuracy_when_expected_tool_available": (
                sum(
                    case["execution_equivalent_exact_call"]
                    for case in expected_available
                )
                / len(expected_available)
                if expected_available
                else None
            ),
            "schema_valid_rate": sum(case["schema_valid"] for case in cases) / total,
            "malformed_rate": sum(case["malformed"] for case in cases) / total,
            "conversational_output_rate": taxonomy_all["conversational_output"]
            / total,
            "nonexistent_tool_rate": taxonomy_all["nonexistent_tool"] / total,
            "multiple_call_rate": taxonomy_all["multiple_calls"] / total,
            "confidence_intervals_95": {
                "tool_selection_accuracy": wilson_interval(tool_correct, total),
                "strict_exact_call_accuracy": wilson_interval(strict_correct, total),
                "execution_equivalent_exact_call_accuracy": wilson_interval(
                    equivalent_correct, total
                ),
            },
        }

        selector_latencies = [case["selector_latency_ms"] for case in cases]
        dispatcher_latencies = [
            case["generation"]["latency_ms"]
            for case in cases
            if case.get("generation")
            and case["generation"].get("latency_ms") is not None
        ]
        total_latencies = [case["total_latency_ms"] for case in cases]
        generation_rates = [
            case["generation"]["tokens_per_second"]
            for case in cases
            if case.get("generation")
            and case["generation"].get("tokens_per_second") is not None
        ]
        prompt_rates = [
            case["generation"]["prompt_tokens_per_second"]
            for case in cases
            if case.get("generation")
            and case["generation"].get("prompt_tokens_per_second") is not None
        ]
        prompt_tokens = [
            case["generation"]["prompt_tokens"]
            for case in cases
            if case.get("generation")
            and case["generation"].get("prompt_tokens") is not None
        ]
        peak_memory = [
            case["generation"]["peak_memory_gb"]
            for case in cases
            if case.get("generation")
            and case["generation"].get("peak_memory_gb") is not None
        ]
        process_rss = [
            case["generation"]["process_rss_mb"]
            for case in cases
            if case.get("generation")
            and case["generation"].get("process_rss_mb") is not None
        ]
        performance_metrics = {
            "selector_latency_ms": _summary(selector_latencies),
            "dispatcher_latency_ms": _summary(dispatcher_latencies),
            "total_routing_latency_ms": _summary(total_latencies),
            "average_generation_tokens_per_second": (
                mean(generation_rates) if generation_rates else None
            ),
            "median_generation_tokens_per_second": (
                median(generation_rates) if generation_rates else None
            ),
            "average_prompt_tokens_per_second": (
                mean(prompt_rates) if prompt_rates else None
            ),
            "average_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
            "peak_mlx_memory_gb": max(peak_memory) if peak_memory else None,
            "maximum_process_rss_mb": max(process_rss) if process_rss else None,
        }

        per_tool = {
            tool: self._per_tool_metrics(
                [case for case in cases if case["expected_tool"] == tool]
            )
            for tool in live_names
        }
        latency_by_category = {
            category: _summary(
                [
                    case["total_latency_ms"]
                    for case in cases
                    if case["tags"][1] == category
                ]
            )
            for category in category_counts
        }

        report["configuration"].update(
            {
                "experiment": "phase2_reliability",
                "phase": "phase_2_reliability",
                "dataset": "phase2-450",
                "dataset_sha256": checksum,
                "dataset_frozen": True,
                "execution_capability": False,
                "execute": False,
            }
        )
        report["dataset"] = {
            "cases": total,
            "sha256": checksum,
            "composition_counts": dict(category_counts),
            "composition_rates": {
                category: count / total for category, count in category_counts.items()
            },
            "cases_per_tool": dict(Counter(case["expected_tool"] for case in cases)),
        }
        report["core_metrics"] = core_metrics
        report["performance_metrics"] = performance_metrics
        report["per_tool_results"] = per_tool
        report["latency_by_dataset_category_ms"] = latency_by_category
        report["failure_taxonomy"] = {
            "primary_counts": dict(taxonomy_primary),
            "all_tag_counts": dict(taxonomy_all),
            "by_expected_tool": {
                tool: dict(counts) for tool, counts in taxonomy_by_tool.items()
            },
            "by_dataset_category": {
                category: dict(counts)
                for category, counts in taxonomy_by_category.items()
            },
        }
        report["failure_records"] = failure_records
        return report

    @staticmethod
    def _per_tool_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(cases)
        if not cases:
            return {
                "cases": 0,
                "correct_tool_in_selected_subset_rate": None,
                "tool_selection_accuracy": None,
                "tool_selection_accuracy_when_available": None,
                "strict_exact_call_accuracy": None,
                "execution_equivalent_exact_call_accuracy": None,
                "schema_valid_rate": None,
                "malformed_rate": None,
                "average_total_latency_ms": None,
                "p95_total_latency_ms": None,
            }
        available_cases = [
            case for case in cases if case["correct_tool_in_selected_subset"]
        ]
        total_latencies = [case["total_latency_ms"] for case in cases]
        return {
            "cases": total,
            "correct_tool_in_selected_subset_rate": sum(
                case["correct_tool_in_selected_subset"] for case in cases
            )
            / total,
            "tool_selection_accuracy": sum(case["tool_correct"] for case in cases)
            / total,
            "tool_selection_accuracy_when_available": (
                sum(case["tool_correct"] for case in available_cases)
                / len(available_cases)
                if available_cases
                else None
            ),
            "strict_exact_call_accuracy": sum(
                case["end_to_end_strict_exact_call"] for case in cases
            )
            / total,
            "execution_equivalent_exact_call_accuracy": sum(
                case["end_to_end_execution_equivalent_exact_call"]
                for case in cases
            )
            / total,
            "schema_valid_rate": sum(case["schema_valid"] for case in cases) / total,
            "malformed_rate": sum(case["malformed"] for case in cases) / total,
            "average_total_latency_ms": mean(total_latencies),
            "p95_total_latency_ms": _percentile(total_latencies, 0.95),
        }


def write_phase2_failures(report: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in report["failure_records"]
        )
