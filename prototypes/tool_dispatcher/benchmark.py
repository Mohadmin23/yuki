"""Scalable benchmark runner for model, regex, and hybrid routing modes."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from .argument_contract import CONTRACT_VERSION, score_contract_arguments
from .argument_scoring import score_arguments
from .dialects import CANONICAL_OBJECT, native_dialect_for_model
from .dispatcher import Dispatcher
from .oracle_subsets import oracle_subset_for_case

DEFAULT_CASES_PATH = Path(__file__).with_name("benchmark_cases.json")
HAMMER_TARGETED_CASES_PATH = Path(__file__).with_name(
    "benchmark_hammer_targeted_cases.json"
)


@dataclass(frozen=True)
class BenchmarkCase:
    request: str
    expected_tool: str
    expected_arguments: dict[str, Any]
    case_id: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    argument_contract_version: str | None = None
    literal_references: dict[str, Any] = field(default_factory=dict)
    semantic_alternatives: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BenchmarkCase:
        contract_version = value.get("argument_contract_version")
        literal_references = value.get("literal_references", {})
        semantic_alternatives = value.get(
            "semantic_alternatives",
            value.get("argument_alternatives", {}),
        )
        if literal_references and contract_version is None:
            raise ValueError(
                "Cases with literal references must declare "
                "argument_contract_version"
            )
        return cls(
            request=value["request"],
            expected_tool=value["expected_tool"],
            expected_arguments=value["expected_arguments"],
            case_id=value.get("case_id", ""),
            tags=tuple(value.get("tags", [])),
            argument_contract_version=contract_version,
            literal_references=literal_references,
            semantic_alternatives=semantic_alternatives,
        )


def load_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[BenchmarkCase]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Benchmark dataset must be a JSON array")
    return [BenchmarkCase.from_dict(item) for item in data]


def load_contract_annotations(
    path: str | Path,
    *,
    dataset_path: str | Path,
) -> dict[str, Any]:
    """Load a sidecar bound to the immutable dataset checksum."""

    annotation_path = Path(path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if payload.get("argument_contract_version") != CONTRACT_VERSION:
        raise RuntimeError(
            "Contract annotation version mismatch: "
            f"{payload.get('argument_contract_version')!r} != {CONTRACT_VERSION!r}"
        )
    expected_dataset_sha = payload.get("dataset_sha256")
    actual_dataset_sha = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    if expected_dataset_sha != actual_dataset_sha:
        raise RuntimeError(
            "Contract annotations are not bound to this dataset: "
            f"{expected_dataset_sha!r} != {actual_dataset_sha!r}"
        )
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise TypeError("Contract annotation sidecar has no annotations object")
    return payload


def _case_contract_inputs(
    case: BenchmarkCase,
    annotation_payload: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    version = case.argument_contract_version
    literal = case.literal_references
    semantic = case.semantic_alternatives
    if annotation_payload is None:
        return version, literal, semantic

    annotation = annotation_payload["annotations"].get(case.case_id)
    if annotation is None:
        return version, literal, semantic
    external_literal = annotation.get("literal_references", {})
    external_semantic = annotation.get("semantic_alternatives", {})
    if literal and external_literal and literal != external_literal:
        raise RuntimeError(f"Conflicting literal references for {case.case_id}")
    if semantic and external_semantic and semantic != external_semantic:
        raise RuntimeError(f"Conflicting semantic alternatives for {case.case_id}")
    external_version = annotation_payload["argument_contract_version"]
    if version is not None and version != external_version:
        raise RuntimeError(f"Conflicting contract versions for {case.case_id}")
    return (
        external_version,
        external_literal or literal,
        external_semantic or semantic,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    """Calculate an interpolated percentile without adding a numeric dependency."""

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


class BenchmarkRunner:
    """Run any-size case file; Phase 1 and Phase 2 use the same path."""

    def __init__(self, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher

    def run(
        self,
        cases: Iterable[BenchmarkCase],
        *,
        group: str = "all",
        selected: Iterable[str] | None = None,
        router_mode: str = "model_only",
        output_mode: str = "native",
        execute: bool = False,
        allow_side_effects: bool = False,
        shell_mode: str = "dry_run",
        limit: int | None = None,
        oracle_profile: str | None = None,
        progress_label: str | None = None,
        case_callback: Callable[[dict[str, Any], dict[str, Any] | None], None]
        | None = None,
        contract_annotations: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if router_mode not in {"model_only", "regex_only", "regex_then_model"}:
            raise ValueError(f"Unknown router mode: {router_mode}")
        if oracle_profile and (group != "all" or selected is not None):
            raise ValueError(
                "Oracle profiles cannot be combined with group or selected filters"
            )
        if oracle_profile:
            scoped = []
            for case in cases:
                subset = oracle_subset_for_case(
                    oracle_profile,
                    case.expected_tool,
                    case.tags,
                    self.dispatcher.registry,
                )
                if subset is not None:
                    scoped.append((case, subset))
            offered_union = {
                tool for _, subset in scoped for tool in subset.tools
            }
            available = tuple(
                tool
                for tool in self.dispatcher.registry.names
                if tool in offered_union
            )
        else:
            available = self.dispatcher.registry.resolve_names(group, selected)
            available_set = set(available)
            scoped = [
                (case, None)
                for case in cases
                if case.expected_tool in available_set
            ]
        if limit is not None:
            scoped = scoped[: max(limit, 0)]
        if not scoped:
            raise ValueError("No benchmark cases match the selected tool scope")
        output_dialect = (
            native_dialect_for_model(self.dispatcher.backend.model_id)
            if output_mode == "native"
            else CANONICAL_OBJECT
        )

        results: list[dict[str, Any]] = []
        failure_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        latencies: list[float] = []
        generation_rates: list[float] = []
        rate_token_total = 0
        estimated_generation_seconds = 0.0
        generated_tokens = 0
        inference_seconds = 0.0
        model_invocations = 0
        malformed_count = 0
        schema_valid_count = 0
        tool_correct_count = 0
        argument_correct_count = 0
        execution_equivalent_argument_count = 0
        exact_call_count = 0
        execution_equivalent_call_count = 0
        contract_scoreable_count = 0
        contract_argument_correct_count = 0
        contract_exact_call_count = 0
        execution_attempts = 0
        execution_passes = 0
        fine_tuning_examples: list[dict[str, Any]] = []

        for index, (case, oracle_subset) in enumerate(scoped):
            if progress_label:
                print(
                    f"{progress_label} {index + 1}/{len(scoped)}: "
                    f"{case.case_id or f'case-{index + 1}'}",
                    flush=True,
                )
            case_group = "all" if oracle_subset else group
            case_selected = oracle_subset.tools if oracle_subset else selected
            case_available = oracle_subset.tools if oracle_subset else available
            if router_mode == "model_only":
                result = self.dispatcher.dispatch(
                    case.request,
                    group=case_group,
                    selected=case_selected,
                    output_mode=output_mode,
                    execute=execute,
                    allow_side_effects=allow_side_effects,
                    shell_mode=shell_mode,
                )
            elif router_mode == "regex_only":
                result = self.dispatcher.dispatch_regex(
                    case.request, group=case_group, selected=case_selected
                )
            else:
                regex_result = self.dispatcher.dispatch_regex(
                    case.request, group=case_group, selected=case_selected
                )
                if regex_result["validation"]["passed"]:
                    result = regex_result
                    result["router_mode"] = "regex_then_model:regex"
                else:
                    result = self.dispatcher.dispatch(
                        case.request,
                        group=case_group,
                        selected=case_selected,
                        output_mode=output_mode,
                        execute=execute,
                        allow_side_effects=allow_side_effects,
                        shell_mode=shell_mode,
                    )
                    result["router_mode"] = "regex_then_model:model"

            if execute and result["validation"]["passed"] and not result["execution"]["executed"]:
                result["execution"] = self.dispatcher.execute_validated(
                    result["canonical_call"],
                    group=case_group,
                    selected=case_selected,
                    allow_side_effects=allow_side_effects,
                    shell_mode=shell_mode,
                )

            actual_tool = result["selected_tool"]
            actual_arguments = result["parsed_arguments"]
            tool_correct = actual_tool == case.expected_tool
            argument_scoring = score_arguments(
                case.expected_tool, case.expected_arguments, actual_arguments
            )
            (
                contract_version,
                literal_references,
                semantic_alternatives,
            ) = _case_contract_inputs(case, contract_annotations)
            contract_scoring = score_contract_arguments(
                case.expected_tool,
                case.expected_arguments,
                actual_arguments,
                literal_references=literal_references,
                semantic_alternatives=semantic_alternatives,
                declared_version=contract_version,
            )
            arguments_correct = argument_scoring["strict"]
            execution_equivalent_arguments = argument_scoring[
                "execution_equivalent"
            ]
            schema_valid = bool(result["validation"]["passed"])
            malformed = bool(result["parse"]["malformed"])
            exact_call = tool_correct and arguments_correct and schema_valid
            execution_equivalent_call = (
                tool_correct and execution_equivalent_arguments and schema_valid
            )
            contract_scoreable = bool(contract_scoring["scoreable"])
            contract_arguments_correct = contract_scoring["contract_correct"]
            contract_exact_call = (
                tool_correct
                and bool(contract_arguments_correct)
                and schema_valid
                if contract_scoreable
                else None
            )
            tool_correct_count += int(tool_correct)
            argument_correct_count += int(arguments_correct)
            execution_equivalent_argument_count += int(
                execution_equivalent_arguments
            )
            schema_valid_count += int(schema_valid)
            malformed_count += int(malformed)
            exact_call_count += int(exact_call)
            execution_equivalent_call_count += int(execution_equivalent_call)
            contract_scoreable_count += int(contract_scoreable)
            contract_argument_correct_count += int(
                contract_arguments_correct is True
            )
            contract_exact_call_count += int(contract_exact_call is True)
            confusion[case.expected_tool][actual_tool or "<none>"] += 1

            generation = result.get("generation")
            generation_for_report = None
            if generation:
                generation_for_report = {
                    key: value
                    for key, value in generation.items()
                    if key != "rendered_prompt"
                }
                model_invocations += 1
                latency = generation.get("latency_ms")
                if latency is not None:
                    latencies.append(latency)
                    inference_seconds += latency / 1000
                rate = generation.get("tokens_per_second")
                if rate is not None:
                    generation_rates.append(rate)
                token_count = generation.get("generated_tokens") or 0
                generated_tokens += token_count
                if rate and token_count:
                    rate_token_total += token_count
                    estimated_generation_seconds += token_count / rate

            execution = result["execution"]
            if execute:
                execution_attempts += int(
                    execution["executed"] or execution["status"] == "dry_run"
                )
                execution_passes += int(execution["passed"])

            failure_types: list[str] = []
            if malformed:
                failure_types.append("malformed_output")
            if result["parse"]["rejected"]:
                failure_types.append("model_reject")
            if not tool_correct:
                failure_types.append("wrong_tool")
            if not arguments_correct:
                failure_types.append("wrong_arguments")
            if not execution_equivalent_arguments:
                failure_types.append("execution_non_equivalent_arguments")
            if not schema_valid:
                failure_types.append("schema_invalid")
            if execute and execution["status"] == "failed":
                failure_types.append("execution_failed")
            for failure_type in failure_types:
                failure_counts[case.expected_tool][failure_type] += 1

            case_result = {
                "index": index,
                "case_id": case.case_id or f"case-{index + 1}",
                "tags": list(case.tags),
                "oracle_subset": oracle_subset.name if oracle_subset else None,
                "offered_tools": list(case_available),
                "request": case.request,
                "expected_tool": case.expected_tool,
                "expected_arguments": case.expected_arguments,
                "actual_tool": actual_tool,
                "actual_arguments": actual_arguments,
                "tool_correct": tool_correct,
                "arguments_correct": arguments_correct,
                "strict_arguments_correct": arguments_correct,
                "execution_equivalent_arguments_correct": (
                    execution_equivalent_arguments
                ),
                "argument_scoring": argument_scoring,
                "argument_contract_version": CONTRACT_VERSION,
                "literal_references": literal_references,
                "semantic_alternatives": semantic_alternatives,
                "contract_scoring": contract_scoring,
                "contract_arguments_correct": contract_arguments_correct,
                "contract_exact_call": contract_exact_call,
                "schema_valid": schema_valid,
                "malformed": malformed,
                "exact_call": exact_call,
                "execution_equivalent_call": execution_equivalent_call,
                "router_path": result["router_mode"],
                "native_dialect": result["native_dialect"],
                "generation": generation_for_report,
                "parse": result["parse"],
                "validation": result["validation"],
                "execution": execution,
            }
            results.append(case_result)
            fine_tuning_example = None
            if not exact_call:
                fine_tuning_example = {
                    "case_id": case_result["case_id"],
                    "request": case.request,
                    "offered_tools": list(case_available),
                    "output_mode": output_mode,
                    "native_dialect": output_dialect,
                    "expected_call": {
                        "tool": case.expected_tool,
                        "arguments": case.expected_arguments,
                    },
                    "raw_generation": (
                        generation.get("raw_text") if generation else None
                    ),
                    "actual_call": result["canonical_call"],
                    "argument_scoring": argument_scoring,
                    "argument_contract": contract_scoring,
                    "training_eligible_under_argument_contract": (
                        contract_scoreable
                    ),
                    "failure_types": failure_types,
                }
                fine_tuning_examples.append(fine_tuning_example)
            if case_callback is not None:
                case_callback(case_result, fine_tuning_example)

        total = len(scoped)
        metrics = {
            "total_cases": total,
            "model_invocations": model_invocations,
            "tool_selection_accuracy": tool_correct_count / total,
            "argument_accuracy": argument_correct_count / total,
            "strict_argument_accuracy": argument_correct_count / total,
            "execution_equivalent_argument_accuracy": (
                execution_equivalent_argument_count / total
            ),
            "exact_call_accuracy": exact_call_count / total,
            "execution_equivalent_call_accuracy": (
                execution_equivalent_call_count / total
            ),
            "argument_contract_version": CONTRACT_VERSION,
            "contract_scoreable_cases": contract_scoreable_count,
            "contract_scoring_coverage_rate": contract_scoreable_count / total,
            "contract_argument_accuracy": (
                contract_argument_correct_count / contract_scoreable_count
                if contract_scoreable_count
                else None
            ),
            "contract_exact_call_accuracy": (
                contract_exact_call_count / contract_scoreable_count
                if contract_scoreable_count
                else None
            ),
            "schema_valid_output_rate": schema_valid_count / total,
            "malformed_output_rate": (
                malformed_count / model_invocations if model_invocations else 0.0
            ),
            "average_inference_latency_ms": mean(latencies) if latencies else None,
            "p50_inference_latency_ms": _percentile(latencies, 0.50),
            "p95_inference_latency_ms": _percentile(latencies, 0.95),
            "average_tokens_per_second": (
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
            "execution_attempts": execution_attempts,
            "execution_passes": execution_passes,
        }
        subset_summary: dict[str, dict[str, Any]] = {}
        for _, subset in scoped:
            if subset is None:
                continue
            summary = subset_summary.setdefault(
                subset.name,
                {"tools": list(subset.tools), "case_count": 0},
            )
            summary["case_count"] += 1

        return {
            "configuration": {
                "router_mode": router_mode,
                "output_mode": output_mode,
                "native_dialect": output_dialect,
                "backend": self.dispatcher.backend.backend_name,
                "model_id": self.dispatcher.backend.model_id,
                "group": group,
                "available_tools": list(available),
                "oracle_profile": oracle_profile,
                "oracle_subsets": subset_summary,
                "execute": execute,
                "argument_contract_version": CONTRACT_VERSION,
                "contract_annotation_sidecar": (
                    contract_annotations.get("version")
                    if contract_annotations
                    else None
                ),
                "phase": (
                    "phase_1_smoke"
                    if total <= 100
                    else "expanded_smoke"
                    if total < 300
                    else "phase_2_reliability"
                ),
            },
            "metrics": metrics,
            "failures_by_expected_tool": {
                tool: dict(counts) for tool, counts in failure_counts.items()
            },
            "confusion_matrix": {
                expected: dict(actuals) for expected, actuals in confusion.items()
            },
            "case_results": results,
            "fine_tuning_examples": fine_tuning_examples,
        }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_fine_tuning_jsonl(report: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(example, ensure_ascii=False) + "\n"
            for example in report["fine_tuning_examples"]
        )
