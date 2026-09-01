"""Command-line entry points for registry inspection and repeatable benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .automatic_benchmark import (
    AutomaticSubsetBenchmarkRunner,
    write_automatic_failure_jsonl,
)
from .backends import DEFAULT_XLAM_MODEL, FixtureBackend, create_backend
from .benchmark import (
    BenchmarkRunner,
    load_cases,
    load_contract_annotations,
    write_fine_tuning_jsonl,
    write_report,
)
from .build_production_delegation_dataset import build_dataset, write_dataset
from .comparison import compare_reports, write_comparison
from .delegation_dispatcher import (
    DISPATCH_INPUT_MODES,
    DOMAIN_SOURCES,
    DelegationDispatcherRunner,
    ValidationOnlyDispatcher,
    failure_records,
)
from .delegation_mainbrain import (
    MainBrainDelegationRunner,
    MLXDelegationBackend,
    OpenRouterDelegationBackend,
    prepare_dataset_cases,
)
from .dispatcher import Dispatcher
from .oracle_subsets import ORACLE_PROFILES
from .phase2_benchmark import Phase2ReliabilityRunner, write_phase2_failures
from .registry import ToolRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yuki tool-dispatch prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Print the live adapted Yuki registry")
    compare = commands.add_parser("compare", help="Compare completed reports")
    compare.add_argument("reports", nargs="+")
    compare.add_argument("--output")

    benchmark = commands.add_parser("benchmark", help="Run a routing benchmark")
    benchmark.add_argument("--cases", default=None)
    benchmark.add_argument(
        "--argument-contract-annotations",
        help=(
            "Optional versioned literal/semantic annotation sidecar bound to "
            "the case-file SHA-256"
        ),
    )
    benchmark.add_argument(
        "--backend",
        choices=[
            "mlx",
            "transformers",
            "openai-compatible-local",
            "openrouter-native-tools",
        ],
        default="mlx",
    )
    benchmark.add_argument("--model", default=DEFAULT_XLAM_MODEL)
    benchmark.add_argument("--base-url")
    benchmark.add_argument("--provider")
    benchmark.add_argument("--max-api-cost-usd", type=float, default=0.1)
    benchmark.add_argument(
        "--router-mode",
        choices=["model_only", "regex_only", "regex_then_model"],
        default="model_only",
    )
    benchmark.add_argument(
        "--output-mode", choices=["native", "canonical"], default="native"
    )
    benchmark.add_argument("--group", default="all")
    benchmark.add_argument(
        "--oracle-profile",
        choices=ORACLE_PROFILES,
        help=(
            "Benchmark-only schema selection using expected labels; never a "
            "production router"
        ),
    )
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--execute", action="store_true")
    benchmark.add_argument("--allow-side-effects", action="store_true")
    benchmark.add_argument(
        "--shell-mode", choices=["dry_run", "strict"], default="dry_run"
    )
    benchmark.add_argument("--no-schema-constraints", action="store_true")
    benchmark.add_argument("--output")
    benchmark.add_argument("--failures-output")

    automatic = commands.add_parser(
        "automatic-benchmark",
        help="Run execution-locked automatic schema selection",
    )
    automatic.add_argument("--cases", default=None)
    automatic.add_argument(
        "--dataset",
        required=True,
        help="Stable dataset label stored in every result and failure record",
    )
    automatic.add_argument(
        "--backend",
        choices=["mlx", "transformers", "openai-compatible-local"],
        default="mlx",
    )
    automatic.add_argument("--model", default=DEFAULT_XLAM_MODEL)
    automatic.add_argument("--base-url")
    automatic.add_argument(
        "--output-mode", choices=["native", "canonical"], default="native"
    )
    automatic.add_argument("--limit", type=int)
    automatic.add_argument("--no-schema-constraints", action="store_true")
    automatic.add_argument("--output")
    automatic.add_argument("--failures-output")

    phase2 = commands.add_parser(
        "phase2-benchmark",
        help="Run the frozen execution-locked Phase 2 reliability study",
    )
    phase2.add_argument("--cases", required=True)
    phase2.add_argument("--dataset-sha256", required=True)
    phase2.add_argument(
        "--backend",
        choices=["mlx", "transformers", "openai-compatible-local"],
        default="mlx",
    )
    phase2.add_argument("--model", required=True)
    phase2.add_argument("--base-url")
    phase2.add_argument(
        "--output-mode", choices=["native", "canonical"], default="native"
    )
    phase2.add_argument("--no-schema-constraints", action="store_true")
    phase2.add_argument("--output", required=True)
    phase2.add_argument("--failures-output", required=True)

    derive = commands.add_parser(
        "build-delegation-dataset",
        help="Derive the versioned production-delegation source from Phase 2",
    )
    derive.add_argument("--source", required=True)
    derive.add_argument("--output", required=True)
    derive.add_argument("--checksum-output", required=True)

    stage_a = commands.add_parser(
        "delegation-mainbrain",
        help="Run execution-free Stage A semantic delegation generation",
    )
    stage_a.add_argument("--dataset", required=True)
    stage_a.add_argument("--dataset-sha256", required=True)
    stage_a.add_argument(
        "--selection",
        choices=["smoke-18", "fallback-stratified-180", "full-450"],
        required=True,
    )
    stage_a.add_argument(
        "--backend", choices=["mlx", "openrouter"], default="mlx"
    )
    stage_a.add_argument("--model", required=True)
    stage_a.add_argument(
        "--max-api-cost-usd",
        type=float,
        default=1.0,
        help="OpenRouter-only cumulative spend guard for this run",
    )
    stage_a.add_argument(
        "--generation-ceiling",
        type=int,
        default=160,
        help="Maximum completion tokens per delegation generation",
    )
    stage_a.add_argument(
        "--provider",
        help="Optional exact OpenRouter provider; fallback is disabled when set",
    )
    stage_a.add_argument("--output", required=True)

    stage_b = commands.add_parser(
        "delegation-dispatch",
        help="Run execution-free Stage B against frozen delegations",
    )
    stage_b.add_argument("--delegations", required=True)
    stage_b.add_argument("--delegation-sha256", required=True)
    stage_b.add_argument("--model", required=True)
    stage_b.add_argument(
        "--input-mode",
        choices=DISPATCH_INPUT_MODES,
        default="delegated_with_verbatim",
    )
    stage_b.add_argument("--domain-source", choices=DOMAIN_SOURCES, default="generated")
    stage_b.add_argument("--output", required=True)
    stage_b.add_argument("--failures-output", required=True)
    return parser


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_failure_jsonl(records: list[dict], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    args = _parser().parse_args()
    registry = ToolRegistry()
    if args.command == "inspect":
        print(json.dumps(registry.describe(), indent=2, ensure_ascii=False))
        return
    if args.command == "compare":
        report = compare_reports(args.reports)
        if args.output:
            write_comparison(report, args.output)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.command == "build-delegation-dataset":
        checksum = write_dataset(build_dataset(args.source), args.output)
        checksum_path = Path(args.checksum_output)
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        checksum_path.write_text(
            f"{checksum}  {Path(args.output).name}\n", encoding="utf-8"
        )
        print(checksum)
        return
    if args.command == "delegation-mainbrain":
        dataset = prepare_dataset_cases(_load_json(args.dataset))
        if args.backend == "openrouter":
            backend = OpenRouterDelegationBackend(
                args.model,
                max_api_cost_usd=args.max_api_cost_usd,
                generation_ceiling=args.generation_ceiling,
                provider=args.provider,
            )
        else:
            backend = MLXDelegationBackend(args.model)
        runner = MainBrainDelegationRunner(backend, registry)
        report = runner.run(
            dataset,
            derived_dataset_path=args.dataset,
            expected_dataset_sha256=args.dataset_sha256,
            selection=args.selection,
        )
        write_report(report, args.output)
        print(hashlib.sha256(Path(args.output).read_bytes()).hexdigest())
        return
    if args.command == "delegation-dispatch":
        backend = create_backend("mlx", args.model)
        runner = DelegationDispatcherRunner(
            ValidationOnlyDispatcher(
                registry,
                backend,
                input_mode=args.input_mode,
                domain_source=args.domain_source,
            )
        )
        report = runner.run(
            _load_json(args.delegations),
            delegation_path=args.delegations,
            expected_delegation_sha256=args.delegation_sha256,
        )
        write_report(report, args.output)
        _write_failure_jsonl(failure_records(report), args.failures_output)
        return
    if args.command == "automatic-benchmark":
        backend = create_backend(
            args.backend,
            args.model,
            base_url=args.base_url,
            constrain_json=not args.no_schema_constraints,
            provider=args.provider,
            max_api_cost_usd=args.max_api_cost_usd,
        )
        dispatcher = Dispatcher(registry, backend)
        runner = AutomaticSubsetBenchmarkRunner(dispatcher)
        report = runner.run(
            load_cases(args.cases) if args.cases else load_cases(),
            dataset=args.dataset,
            output_mode=args.output_mode,
            limit=args.limit,
        )
        if args.output:
            write_report(report, args.output)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.failures_output:
            write_automatic_failure_jsonl(report, args.failures_output)
        return
    if args.command == "phase2-benchmark":
        backend = create_backend(
            args.backend,
            args.model,
            base_url=args.base_url,
            constrain_json=not args.no_schema_constraints,
        )
        dispatcher = Dispatcher(registry, backend)
        runner = Phase2ReliabilityRunner(dispatcher)
        report = runner.run(
            load_cases(args.cases),
            dataset_path=args.cases,
            expected_checksum=args.dataset_sha256,
            output_mode=args.output_mode,
        )
        write_report(report, args.output)
        write_phase2_failures(report, args.failures_output)
        return

    if args.router_mode == "regex_only":
        backend = FixtureBackend([])
    else:
        backend = create_backend(
            args.backend,
            args.model,
            base_url=args.base_url,
            constrain_json=not args.no_schema_constraints,
        )
    dispatcher = Dispatcher(registry, backend)
    runner = BenchmarkRunner(dispatcher)
    case_path = args.cases
    contract_annotations = None
    if args.argument_contract_annotations:
        if case_path is None:
            raise ValueError(
                "--argument-contract-annotations requires an explicit --cases file"
            )
        contract_annotations = load_contract_annotations(
            args.argument_contract_annotations,
            dataset_path=case_path,
        )
    report = runner.run(
        load_cases(case_path) if case_path else load_cases(),
        group=args.group,
        router_mode=args.router_mode,
        output_mode=args.output_mode,
        execute=args.execute,
        allow_side_effects=args.allow_side_effects,
        shell_mode=args.shell_mode,
        limit=args.limit,
        oracle_profile=args.oracle_profile,
        contract_annotations=contract_annotations,
    )
    if args.output:
        write_report(report, args.output)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.failures_output:
        write_fine_tuning_jsonl(report, args.failures_output)


if __name__ == "__main__":
    main()
