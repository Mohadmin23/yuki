"""Execution-locked Hammer 2.0 direct native dispatcher experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .backends import DispatcherBackend, MLXBackend
from .benchmark import BenchmarkRunner, _percentile, write_report
from .dispatcher import Dispatcher
from .gpt_oss_direct import PHASE2_SHA256, selected_cases
from .registry import ToolRegistry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _checkpoint_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid checkpoint JSON on line {line_number}: {exc}"
            ) from exc
    return records


def _aggregate_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    generations = [case.get("generation") for case in cases]
    generations = [generation for generation in generations if generation]
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
    generated_tokens = sum(
        generation.get("generated_tokens") or 0 for generation in generations
    )
    inference_seconds = sum(latencies) / 1000
    rate_token_total = 0
    estimated_generation_seconds = 0.0
    for generation in generations:
        token_count = generation.get("generated_tokens") or 0
        rate = generation.get("tokens_per_second")
        if rate and token_count:
            rate_token_total += token_count
            estimated_generation_seconds += token_count / rate
    total = len(cases)
    return {
        "total_cases": total,
        "model_invocations": len(generations),
        "tool_selection_accuracy": mean(case["tool_correct"] for case in cases),
        "argument_accuracy": mean(
            case["strict_arguments_correct"] for case in cases
        ),
        "strict_argument_accuracy": mean(
            case["strict_arguments_correct"] for case in cases
        ),
        "execution_equivalent_argument_accuracy": mean(
            case["execution_equivalent_arguments_correct"] for case in cases
        ),
        "exact_call_accuracy": mean(case["exact_call"] for case in cases),
        "execution_equivalent_call_accuracy": mean(
            case["execution_equivalent_call"] for case in cases
        ),
        "schema_valid_output_rate": mean(case["schema_valid"] for case in cases),
        "malformed_output_rate": mean(case["malformed"] for case in cases),
        "average_inference_latency_ms": mean(latencies) if latencies else None,
        "p50_inference_latency_ms": _percentile(latencies, 0.50),
        "p95_inference_latency_ms": _percentile(latencies, 0.95),
        "average_tokens_per_second": (
            rate_token_total / estimated_generation_seconds
            if estimated_generation_seconds
            else None
        ),
        "mean_backend_reported_tokens_per_second": mean(rates) if rates else None,
        "median_backend_reported_tokens_per_second": (
            median(rates) if rates else None
        ),
        "end_to_end_generated_tokens_per_second": (
            generated_tokens / inference_seconds if inference_seconds else None
        ),
        "execution_attempts": 0,
        "execution_passes": 0,
    }


def _assemble_report(
    *,
    backend: DispatcherBackend,
    registry: ToolRegistry,
    selection: str,
    schema_sha256: str,
    case_results: list[dict[str, Any]],
    fine_tuning_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: dict[str, Counter[str]] = defaultdict(Counter)
    for example in fine_tuning_examples:
        expected_tool = example["expected_call"]["tool"]
        failures[expected_tool].update(example["failure_types"])
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for case in case_results:
        confusion[case["expected_tool"]][case["actual_tool"] or "<none>"] += 1
    return {
        "configuration": {
            "router_mode": "model_only",
            "output_mode": "native",
            "native_dialect": "hammer_fenced_tool_call_array",
            "backend": backend.backend_name,
            "model_id": backend.model_id,
            "group": "all",
            "available_tools": list(registry.names),
            "oracle_profile": None,
            "oracle_subsets": {},
            "execute": False,
            "phase": (
                "phase_2_reliability"
                if len(case_results) >= 300
                else "phase_1_smoke"
            ),
            "experiment": "hammer20_7b_direct_all_18_native_tools",
            "selection": selection,
            "exposure_condition": "all_18_real_yuki_schemas",
            "schema_profile": "live_registry_hammer_native_validation_adapter",
            "schema_sha256": schema_sha256,
            "phase2_dataset_sha256": PHASE2_SHA256,
            "native_hammer_format": True,
            "generation_count_per_case": 1,
            "stateless": True,
            "execution_capability": False,
            "yuki_tool_execution_attempts": 0,
            "binder_used": False,
            "qwen_used": False,
            "regex_router_used": False,
            "resumable_case_checkpoint": True,
            "static_prefix_cache": backend.status()["model_metadata"],
        },
        "metrics": _aggregate_metrics(case_results),
        "failures_by_expected_tool": {
            tool: dict(counts) for tool, counts in failures.items()
        },
        "confusion_matrix": {
            expected: dict(actuals) for expected, actuals in confusion.items()
        },
        "case_results": case_results,
        "fine_tuning_examples": fine_tuning_examples,
    }


def run_experiment(
    backend: DispatcherBackend,
    *,
    selection: str,
    output: Path,
    failures_output: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run raw request -> Hammer call -> validation with no execution path."""

    cases = selected_cases(selection)
    registry = ToolRegistry()
    schema_bytes = json.dumps(
        registry.hammer_schemas(registry.names),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    checkpoint_records = _checkpoint_records(checkpoint) if checkpoint else []
    case_order = {case.case_id: index for index, case in enumerate(cases)}
    completed: dict[str, dict[str, Any]] = {}
    for record in checkpoint_records:
        if record.get("version") != 1:
            raise RuntimeError("Unsupported Hammer checkpoint version")
        if record.get("phase2_dataset_sha256") != PHASE2_SHA256:
            raise RuntimeError("Checkpoint dataset checksum mismatch")
        if record.get("schema_sha256") != schema_sha256:
            raise RuntimeError("Checkpoint schema checksum mismatch")
        if record.get("model_id") != backend.model_id:
            raise RuntimeError("Checkpoint model mismatch")
        case_id = record["case_result"]["case_id"]
        if case_id not in case_order or case_id in completed:
            raise RuntimeError(f"Invalid or duplicate checkpoint case: {case_id}")
        completed[case_id] = record

    pending = [case for case in cases if case.case_id not in completed]

    def save_case(
        case_result: dict[str, Any],
        fine_tuning_example: dict[str, Any] | None,
    ) -> None:
        case_result["index"] = case_order[case_result["case_id"]]
        record = {
            "version": 1,
            "phase2_dataset_sha256": PHASE2_SHA256,
            "schema_sha256": schema_sha256,
            "model_id": backend.model_id,
            "case_result": case_result,
            "fine_tuning_example": fine_tuning_example,
        }
        completed[case_result["case_id"]] = record
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if pending:
        BenchmarkRunner(Dispatcher(registry, backend)).run(
            pending,
            group="all",
            router_mode="model_only",
            output_mode="native",
            execute=False,
            progress_label=(
                f"Hammer 2.0 7B direct (resume {len(completed)}/{len(cases)})"
            ),
            case_callback=save_case,
        )

    ordered_records = [completed[case.case_id] for case in cases]
    case_results = [record["case_result"] for record in ordered_records]
    fine_tuning_examples = [
        record["fine_tuning_example"]
        for record in ordered_records
        if record.get("fine_tuning_example") is not None
    ]
    report = _assemble_report(
        backend=backend,
        registry=registry,
        selection=selection,
        schema_sha256=schema_sha256,
        case_results=case_results,
        fine_tuning_examples=fine_tuning_examples,
    )
    if report["configuration"]["execute"]:
        raise RuntimeError("Execution lock unexpectedly disabled")
    if any(case["execution"]["executed"] for case in report["case_results"]):
        raise RuntimeError("A Yuki tool execution was attempted")

    report["configuration"].update(
        {
            "experiment": "hammer20_7b_direct_all_18_native_tools",
            "selection": selection,
            "exposure_condition": "all_18_real_yuki_schemas",
            "schema_profile": "live_registry_hammer_native_validation_adapter",
            "schema_sha256": schema_sha256,
            "phase2_dataset_sha256": PHASE2_SHA256,
            "native_hammer_format": True,
            "generation_count_per_case": 1,
            "stateless": True,
            "execution_capability": False,
            "yuki_tool_execution_attempts": 0,
            "binder_used": False,
            "qwen_used": False,
            "regex_router_used": False,
        }
    )
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
    if checkpoint is not None:
        _write_checksum(checkpoint)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execution-locked Hammer 2.0 7B direct native benchmark"
    )
    parser.add_argument(
        "--selection", choices=("smoke-18", "full-450"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    backend = MLXBackend(
        str(args.model), static_prefix_marker="[BEGIN OF QUERY]\n"
    )
    report = run_experiment(
        backend,
        selection=args.selection,
        output=args.output,
        failures_output=args.failures_output,
        checkpoint=args.checkpoint,
    )
    print(_sha256(args.output))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
