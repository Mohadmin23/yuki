"""Execution-locked GPT-OSS native tool-call experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .backends import DispatcherBackend, create_backend
from .benchmark import BenchmarkCase, BenchmarkRunner, load_cases, write_report
from .dispatcher import Dispatcher
from .registry import ToolRegistry

ROOT = Path(__file__).parent
PHASE2_CASES = ROOT / "phase2-cases.json"
SELECTION_SOURCE = ROOT / "production-delegation-source-v1.json"
PHASE2_SHA256 = "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466"
SELECTION_SOURCE_SHA256 = (
    "ea6711ce0548d3771166f8f2f9a5e59ebd73acb14a9ddf7ecbf2f6589f1744c3"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sources() -> None:
    expected = {
        PHASE2_CASES: PHASE2_SHA256,
        SELECTION_SOURCE: SELECTION_SOURCE_SHA256,
    }
    mismatches = {
        str(path): {"expected": checksum, "actual": _sha256(path)}
        for path, checksum in expected.items()
        if _sha256(path) != checksum
    }
    if mismatches:
        raise RuntimeError(f"Frozen direct-dispatch inputs changed: {mismatches}")


def selected_cases(selection: str) -> list[BenchmarkCase]:
    """Return the exact frozen smoke IDs or all 450 Phase 2 cases."""

    _verify_sources()
    cases = load_cases(PHASE2_CASES)
    if selection == "full-450":
        return cases
    if selection != "smoke-18":
        raise ValueError(f"Unknown selection: {selection}")
    source = json.loads(SELECTION_SOURCE.read_text(encoding="utf-8"))
    ids = source["selection_sets"]["smoke-18"]
    by_id = {case.case_id: case for case in cases}
    selected = [by_id[case_id] for case_id in ids]
    if len(selected) != 18 or len({case.expected_tool for case in selected}) != 18:
        raise RuntimeError("Frozen smoke selection is not one case per Yuki tool")
    return selected


def _group_metrics(cases: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        name = case["tags"][1] if field == "tags_category" else case[field]
        grouped[name].append(case)
    return {
        name: {
            "cases": len(scoped),
            "tool_selection_accuracy": mean(c["tool_correct"] for c in scoped),
            "strict_argument_accuracy": mean(
                c["strict_arguments_correct"] for c in scoped
            ),
            "execution_equivalent_argument_accuracy": mean(
                c["execution_equivalent_arguments_correct"] for c in scoped
            ),
            "strict_exact_call_accuracy": mean(c["exact_call"] for c in scoped),
            "execution_equivalent_exact_call_accuracy": mean(
                c["execution_equivalent_call"] for c in scoped
            ),
            "schema_valid_rate": mean(c["schema_valid"] for c in scoped),
            "malformed_rate": mean(c["malformed"] for c in scoped),
            "rejection_rate": mean(c["parse"]["rejected"] for c in scoped),
        }
        for name, scoped in sorted(grouped.items())
    }


def _write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n", encoding="utf-8"
    )
    return checksum


def run_experiment(
    backend: DispatcherBackend,
    *,
    selection: str,
    output: Path,
    failures_output: Path,
) -> dict[str, Any]:
    """Run raw request -> native call -> validation with no execution path."""

    cases = selected_cases(selection)
    registry = ToolRegistry()
    schema_bytes = json.dumps(
        registry.strict_openai_schemas(registry.names),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    report = BenchmarkRunner(Dispatcher(registry, backend)).run(
        cases,
        group="all",
        router_mode="model_only",
        output_mode="native",
        execute=False,
        progress_label="GPT-OSS direct",
    )
    if report["configuration"]["execute"]:
        raise RuntimeError("Execution lock unexpectedly disabled")
    if any(case["execution"]["executed"] for case in report["case_results"]):
        raise RuntimeError("A Yuki tool execution was attempted")

    generations = [case["generation"] for case in report["case_results"]]
    generation_extras = [generation.get("extra") or {} for generation in generations]
    costs = [
        extra.get("api_cost_usd")
        for extra in generation_extras
        if isinstance(extra.get("api_cost_usd"), int | float)
    ]
    reasoning_tokens = [
        extra.get("reasoning_tokens", 0) or 0 for extra in generation_extras
    ]
    providers = sorted(
        {
            extra.get("provider")
            for extra in generation_extras
            if extra.get("provider")
        }
    )
    report["configuration"].update(
        {
            "experiment": "gpt_oss_direct_all_18_native_tools",
            "selection": selection,
            "exposure_condition": "all_18_real_yuki_schemas",
            "schema_profile": "live_registry_strict_validation_adapter",
            "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
            "phase2_dataset_sha256": PHASE2_SHA256,
            "selection_source_sha256": SELECTION_SOURCE_SHA256,
            "native_api_tool_calls": True,
            "tool_choice": "required",
            "parallel_tool_calls_parameter_sent": False,
            "multiple_calls_rejected_by_parser": True,
            "reasoning_effort": "low",
            "reasoning_excluded_from_output": True,
            "generation_count_per_case": 1,
            "stateless": True,
            "execution_capability": False,
            "yuki_tool_execution_attempts": 0,
            "binder_used": False,
            "qwen_used": False,
            "hammer_used": False,
            "regex_router_used": False,
        }
    )
    report["api_usage"] = {
        "cost_usd": round(sum(costs), 10),
        "cost_reported_cases": len(costs),
        "providers": providers,
        "average_reasoning_tokens": mean(reasoning_tokens),
        "maximum_reasoning_tokens": max(reasoning_tokens),
    }
    report["by_expected_tool"] = _group_metrics(
        report["case_results"], "expected_tool"
    )
    report["by_dataset_category"] = _group_metrics(
        report["case_results"], "tags_category"
    )
    write_report(report, output)
    _write_jsonl(report["fine_tuning_examples"], failures_output)
    _write_checksum(output)
    _write_checksum(failures_output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execution-locked GPT-OSS direct native tool benchmark"
    )
    parser.add_argument(
        "--selection", choices=("smoke-18", "full-450"), required=True
    )
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--provider", default="DeepInfra")
    parser.add_argument("--max-api-cost-usd", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    backend = create_backend(
        "openrouter-native-tools",
        args.model,
        provider=args.provider,
        max_api_cost_usd=args.max_api_cost_usd,
    )
    report = run_experiment(
        backend,
        selection=args.selection,
        output=args.output,
        failures_output=args.failures_output,
    )
    print(_sha256(args.output))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
