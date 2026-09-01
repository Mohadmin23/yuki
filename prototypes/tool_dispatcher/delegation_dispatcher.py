"""Execution-locked Stage B routing from frozen main-brain delegations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .argument_scoring import score_arguments
from .backends import MAX_GENERATION_TOKENS, DispatcherBackend
from .delegation_contract import domain_tools
from .parsing import parse_model_output
from .prompting import build_prompt
from .registry import ToolRegistry

DISPATCH_INPUT_MODES = (
    "delegated_with_verbatim",
    "delegated_without_verbatim",
    "raw_user",
)
DOMAIN_SOURCES = ("generated", "gold")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def render_delegation_for_dispatcher(delegation: dict[str, Any]) -> str:
    """Render only the production delegation object, never the raw user request."""

    verbatim_json = json.dumps(delegation["verbatim"], ensure_ascii=False)
    return (
        "Delegated semantic action:\n"
        f"{delegation['request']}\n\n"
        "Exact user strings to preserve unchanged in arguments:\n"
        f"{verbatim_json}"
    )


def render_semantic_request_only(delegation: dict[str, Any]) -> str:
    return f"Delegated semantic action:\n{delegation['request']}"


class ValidationOnlyDispatcher:
    """Generate, parse, and validate one call with no execution API or import."""

    def __init__(
        self,
        registry: ToolRegistry,
        backend: DispatcherBackend,
        *,
        input_mode: str = "delegated_with_verbatim",
        domain_source: str = "generated",
        task_instruction: str | None = None,
        hammer_format_instruction: str | None = None,
    ) -> None:
        if input_mode not in DISPATCH_INPUT_MODES:
            raise ValueError(f"Unknown dispatcher input mode: {input_mode}")
        if domain_source not in DOMAIN_SOURCES:
            raise ValueError(f"Unknown domain source: {domain_source}")
        self.registry = registry
        self.backend = backend
        self.input_mode = input_mode
        self.domain_source = domain_source
        self.task_instruction = task_instruction
        self.hammer_format_instruction = hammer_format_instruction
        self._domains = domain_tools(registry)

    def route(
        self,
        delegation: dict[str, Any],
        *,
        raw_request: str | None = None,
        gold_domain: str | None = None,
    ) -> dict[str, Any]:
        domain = (
            delegation["domain_hint"]
            if self.domain_source == "generated"
            else gold_domain
        )
        if domain is None:
            raise ValueError("The configured domain source is unavailable")
        names = self._domains[domain]
        if self.input_mode == "delegated_with_verbatim":
            dispatcher_input = render_delegation_for_dispatcher(delegation)
        elif self.input_mode == "delegated_without_verbatim":
            dispatcher_input = render_semantic_request_only(delegation)
        else:
            if raw_request is None:
                raise ValueError("raw_user input mode requires the original request")
            dispatcher_input = raw_request
        package = build_prompt(
            self.registry,
            dispatcher_input,
            names,
            output_mode="native",
            model_id=self.backend.model_id,
            task_instruction=self.task_instruction,
            hammer_format_instruction=self.hammer_format_instruction,
        )
        base = {
            "dispatcher_input": dispatcher_input,
            "domain_hint": domain,
            "domain_source": self.domain_source,
            "input_mode": self.input_mode,
            "offered_tools": list(names),
            "schemas_sent": package.schemas_sent,
            "native_dialect": package.native_dialect,
            "stateless": True,
            "generation_count": 1,
            "execution_capability": False,
        }
        try:
            generation = self.backend.generate(
                package.content,
                package.output_schema,
                max_tokens=MAX_GENERATION_TOKENS,
                system_prompt=package.system_content,
                chat_template_tools=package.chat_template_tools,
                json_root=package.json_root,
                pre_rendered=package.pre_rendered,
            )
        except Exception as exc:  # noqa: BLE001 - inference failure is benchmark data
            return {
                **base,
                "generation": {
                    "backend": self.backend.backend_name,
                    "model_id": self.backend.model_id,
                    "raw_text": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": None,
                    "generated_tokens": None,
                    "tokens_per_second": None,
                },
                "parse": {
                    "passed": False,
                    "malformed": False,
                    "rejected": False,
                    "decoded": None,
                    "canonical_call": None,
                    "errors": ["Dispatcher generation failed before parsing."],
                },
                "canonical_call": None,
                "validation": {"passed": False, "errors": ["Generation failed."]},
            }

        parsed = parse_model_output(
            generation.normalized_text or generation.raw_text,
            "native",
            package.native_dialect,
        )
        call = parsed["canonical_call"]
        if call is None:
            errors = parsed["errors"] or (
                ["Model rejected the delegated request."] if parsed["rejected"] else []
            )
            validation = {"passed": False, "errors": errors}
        else:
            validation = self.registry.validate_call(call, names)
        generation_data = generation.as_dict()
        generation_data.pop("rendered_prompt", None)
        return {
            **base,
            "generation": generation_data,
            "parse": parsed,
            "canonical_call": call,
            "validation": validation,
        }


def _preservation_aware_arguments(
    tool: str,
    expected: dict[str, Any],
    actual: Any,
    alternatives: dict[str, list[str]],
) -> dict[str, Any]:
    base = score_arguments(tool, expected, actual)
    if base["execution_equivalent"] or not isinstance(actual, dict):
        return {**base, "preservation_aware": base["execution_equivalent"]}
    if set(actual) != set(expected):
        return {**base, "preservation_aware": False}

    candidate_expected = dict(expected)
    for key, values in alternatives.items():
        if key not in actual:
            continue
        for value in values:
            candidate_expected[key] = value
            if score_arguments(tool, candidate_expected, actual)[
                "execution_equivalent"
            ]:
                return {
                    **base,
                    "preservation_aware": True,
                    "matched_alternative": {key: value},
                }
        candidate_expected[key] = expected[key]
    return {**base, "preservation_aware": False}


def _looks_conversational(result: dict[str, Any]) -> bool:
    if not result["parse"]["malformed"]:
        return False
    raw = result["generation"].get("raw_text", "").lstrip()
    return bool(raw) and not raw.startswith(("[", "{", "```"))


def _failure_category(case: dict[str, Any]) -> str | None:
    if case["strict_exact_call"]:
        return None
    if case["delegation"] is None and case["dispatcher_input_mode"] != "raw_user":
        return "malformed_delegation"
    if not case["routing_domain_gold_correct"]:
        return "main_brain_wrong_domain"
    if (
        case["dispatcher_input_mode"] == "delegated_with_verbatim"
        and case["delegation_scores"]["verbatim_recall"] < 1.0
    ):
        return "main_brain_lost_verbatim"
    if case["dispatcher"]["generation"].get("error"):
        return "dispatcher_generation_error"
    if case["dispatcher"]["parse"]["malformed"]:
        return (
            "dispatcher_conversational_output"
            if case["conversational_output"]
            else "dispatcher_malformed_output"
        )
    if not case["tool_correct"]:
        return "dispatcher_wrong_tool"
    if not case["schema_valid"]:
        return "dispatcher_schema_invalid"
    if case["preservation_aware_arguments_correct"]:
        return "dispatcher_argument_rewrite"
    return "dispatcher_wrong_argument"


def _scope_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    if not total:
        return {"cases": 0}
    domain_correct = [case for case in cases if case["domain_hint_exact"]]
    routing_domain_correct = [
        case for case in cases if case["routing_domain_gold_correct"]
    ]
    tool_correct = [case for case in cases if case["tool_correct"]]
    latencies = [
        case["dispatcher"]["generation"].get("latency_ms")
        for case in cases
        if case["dispatcher"] is not None
        and case["dispatcher"]["generation"].get("latency_ms") is not None
    ]
    return {
        "cases": total,
        "valid_delegation_rate": mean(case["delegation"] is not None for case in cases),
        "domain_hint_accuracy": mean(case["domain_hint_exact"] for case in cases),
        "routing_domain_accuracy": mean(
            case["routing_domain_gold_correct"] for case in cases
        ),
        "tool_selection_accuracy": mean(case["tool_correct"] for case in cases),
        "tool_selection_accuracy_when_domain_correct": (
            mean(case["tool_correct"] for case in domain_correct)
            if domain_correct
            else None
        ),
        "tool_selection_accuracy_when_routing_domain_correct": (
            mean(case["tool_correct"] for case in routing_domain_correct)
            if routing_domain_correct
            else None
        ),
        "strict_argument_accuracy": mean(
            case["strict_arguments_correct"] for case in cases
        ),
        "equivalent_argument_accuracy": mean(
            case["equivalent_arguments_correct"] for case in cases
        ),
        "preservation_aware_argument_accuracy": mean(
            case["preservation_aware_arguments_correct"] for case in cases
        ),
        "strict_argument_accuracy_when_tool_correct": (
            mean(case["strict_arguments_correct"] for case in tool_correct)
            if tool_correct
            else None
        ),
        "preservation_aware_argument_accuracy_when_tool_correct": (
            mean(case["preservation_aware_arguments_correct"] for case in tool_correct)
            if tool_correct
            else None
        ),
        "strict_exact_call_accuracy": mean(case["strict_exact_call"] for case in cases),
        "equivalent_exact_call_accuracy": mean(
            case["equivalent_exact_call"] for case in cases
        ),
        "preservation_aware_exact_call_accuracy": mean(
            case["preservation_aware_exact_call"] for case in cases
        ),
        "strict_exact_call_accuracy_when_domain_correct": (
            mean(case["strict_exact_call"] for case in domain_correct)
            if domain_correct
            else None
        ),
        "preservation_aware_exact_call_accuracy_when_domain_correct": (
            mean(case["preservation_aware_exact_call"] for case in domain_correct)
            if domain_correct
            else None
        ),
        "strict_exact_call_accuracy_when_routing_domain_correct": (
            mean(case["strict_exact_call"] for case in routing_domain_correct)
            if routing_domain_correct
            else None
        ),
        "preservation_aware_exact_call_accuracy_when_routing_domain_correct": (
            mean(
                case["preservation_aware_exact_call"] for case in routing_domain_correct
            )
            if routing_domain_correct
            else None
        ),
        "schema_valid_rate": mean(case["schema_valid"] for case in cases),
        "malformed_output_rate": mean(case["malformed_output"] for case in cases),
        "conversational_output_rate": mean(
            case["conversational_output"] for case in cases
        ),
        "average_dispatcher_latency_ms": mean(latencies) if latencies else None,
        "p50_dispatcher_latency_ms": median(latencies) if latencies else None,
        "p95_dispatcher_latency_ms": _percentile(latencies, 0.95),
    }


def _grouped_metrics(
    cases: list[dict[str, Any]], value_getter: Any
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[value_getter(case)].append(case)
    return {name: _scope_metrics(scoped) for name, scoped in sorted(grouped.items())}


class DelegationDispatcherRunner:
    """Score frozen delegations through a validation-only Hammer dispatcher."""

    def __init__(self, dispatcher: ValidationOnlyDispatcher) -> None:
        self.dispatcher = dispatcher

    def run(
        self,
        delegation_report: dict[str, Any],
        *,
        delegation_path: str | Path,
        expected_delegation_sha256: str,
    ) -> dict[str, Any]:
        path = Path(delegation_path)
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_delegation_sha256:
            raise RuntimeError(
                f"Frozen delegation checksum mismatch: {actual_sha} != "
                f"{expected_delegation_sha256}"
            )
        stage_a_configuration = delegation_report["configuration"]
        if not isinstance(stage_a_configuration.get("surrogate"), bool):
            raise TypeError(
                "Stage A report must explicitly identify whether it is a surrogate run"
            )

        results = []
        sources = delegation_report["case_results"]
        for index, source in enumerate(sources):
            print(
                f"Stage B {index + 1}/{len(sources)}: {source['case_id']}",
                flush=True,
            )
            delegation = source["delegation"]
            can_dispatch = delegation is not None or (
                self.dispatcher.input_mode == "raw_user"
                and self.dispatcher.domain_source == "gold"
            )
            routing_delegation = delegation or {
                "request": source["request"],
                "domain_hint": source["expected_domain"],
                "verbatim": [],
            }
            dispatch = (
                self.dispatcher.route(
                    routing_delegation,
                    raw_request=source["request"],
                    gold_domain=source["expected_domain"],
                )
                if can_dispatch
                else None
            )
            call = dispatch["canonical_call"] if dispatch else None
            actual_tool = call.get("tool") if call else None
            actual_arguments = call.get("arguments") if call else None
            tool_correct = actual_tool == source["expected_tool"]
            argument_score = _preservation_aware_arguments(
                source["expected_tool"],
                source["expected_arguments"],
                actual_arguments,
                source["argument_alternatives"],
            )
            schema_valid = bool(dispatch and dispatch["validation"]["passed"])
            strict_arguments = tool_correct and argument_score["strict"]
            equivalent_arguments = (
                tool_correct and argument_score["execution_equivalent"]
            )
            preservation_arguments = (
                tool_correct and argument_score["preservation_aware"]
            )
            malformed = bool(dispatch and dispatch["parse"]["malformed"])
            result = {
                "case_id": source["case_id"],
                "request": source["request"],
                "tags": source["tags"],
                "expected_domain": source["expected_domain"],
                "expected_tool": source["expected_tool"],
                "expected_arguments": source["expected_arguments"],
                "required_verbatim": source["required_verbatim"],
                "argument_alternatives": source["argument_alternatives"],
                "ambiguous_or_adversarial": source["ambiguous_or_adversarial"],
                "delegation": delegation,
                "delegation_scores": source["delegation_scores"],
                "domain_hint_exact": source["delegation_scores"]["domain_hint_exact"],
                "routing_domain": (
                    dispatch["domain_hint"] if dispatch is not None else None
                ),
                "routing_domain_source": self.dispatcher.domain_source,
                "dispatcher_input_mode": self.dispatcher.input_mode,
                "routing_domain_gold_correct": bool(
                    dispatch and dispatch["domain_hint"] == source["expected_domain"]
                ),
                "dispatcher": dispatch,
                "actual_tool": actual_tool,
                "actual_arguments": actual_arguments,
                "argument_scoring": argument_score,
                "tool_correct": tool_correct,
                "strict_arguments_correct": strict_arguments,
                "equivalent_arguments_correct": equivalent_arguments,
                "preservation_aware_arguments_correct": preservation_arguments,
                "schema_valid": schema_valid,
                "malformed_output": malformed,
                "conversational_output": bool(
                    dispatch and _looks_conversational(dispatch)
                ),
                "strict_exact_call": tool_correct and strict_arguments and schema_valid,
                "equivalent_exact_call": tool_correct
                and equivalent_arguments
                and schema_valid,
                "preservation_aware_exact_call": tool_correct
                and preservation_arguments
                and schema_valid,
                "execution": {
                    "capability": False,
                    "executed": False,
                    "status": "not_available",
                },
            }
            result["failure_category"] = _failure_category(result)
            results.append(result)

        dispatch_generations = [
            case["dispatcher"]["generation"]
            for case in results
            if case["dispatcher"] is not None
        ]
        rates = [
            generation["tokens_per_second"]
            for generation in dispatch_generations
            if generation.get("tokens_per_second") is not None
        ]
        peaks = [
            generation["peak_memory_gb"]
            for generation in dispatch_generations
            if generation.get("peak_memory_gb") is not None
        ]
        failures = Counter(
            case["failure_category"]
            for case in results
            if case["failure_category"] is not None
        )
        matrix: dict[str, Counter[str]] = defaultdict(Counter)
        for case in results:
            matrix[case["expected_tool"]][case["actual_tool"] or "<none>"] += 1

        non_ambiguous = [
            case for case in results if not case["ambiguous_or_adversarial"]
        ]
        return {
            "configuration": {
                "experiment": "production_delegation_stage_b",
                "stage_a_main_brain_role": stage_a_configuration[
                    "main_brain_role"
                ],
                "stage_a_surrogate": stage_a_configuration["surrogate"],
                "stage_a_model_id": stage_a_configuration["model_id"],
                "stage_a_model_label": stage_a_configuration.get("model_label")
                or stage_a_configuration.get("surrogate_label"),
                "stage_a_delegation_contract_version": stage_a_configuration.get(
                    "delegation_contract_version"
                ),
                "stage_a_delegation_sha256": actual_sha,
                "source_dataset_sha256": stage_a_configuration[
                    "source_dataset_sha256"
                ],
                "dispatcher_model_id": self.dispatcher.backend.model_id,
                "dispatcher_backend": self.dispatcher.backend.backend_name,
                "dispatcher_input_mode": self.dispatcher.input_mode,
                "routing_domain_source": self.dispatcher.domain_source,
                "ablation": not (
                    self.dispatcher.input_mode == "delegated_with_verbatim"
                    and self.dispatcher.domain_source == "generated"
                ),
                "uses_oracle_domain": self.dispatcher.domain_source == "gold",
                "native_dialect": (
                    results[0]["dispatcher"]["native_dialect"]
                    if results and results[0]["dispatcher"]
                    else None
                ),
                "stateless": True,
                "one_generation_per_valid_delegation": True,
                "original_request_sent_to_dispatcher": (
                    self.dispatcher.input_mode == "raw_user"
                ),
                "tool_execution_capability": False,
                "total_cases": len(results),
                "generated_dispatches": len(dispatch_generations),
            },
            "summary": {
                **_scope_metrics(results),
                "average_generation_tokens_per_second": mean(rates) if rates else None,
                "peak_mlx_memory_gb": max(peaks) if peaks else None,
            },
            "non_ambiguous_summary": _scope_metrics(non_ambiguous),
            "by_tool": _grouped_metrics(results, lambda case: case["expected_tool"]),
            "by_family": _grouped_metrics(
                results, lambda case: case["expected_domain"]
            ),
            "by_dataset_category": _grouped_metrics(
                results, lambda case: case["tags"][1]
            ),
            "failure_decomposition": dict(failures),
            "confusion_matrix": {
                expected: dict(actual) for expected, actual in matrix.items()
            },
            "case_results": results,
        }


def failure_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for case in report["case_results"] if not case["strict_exact_call"]]
