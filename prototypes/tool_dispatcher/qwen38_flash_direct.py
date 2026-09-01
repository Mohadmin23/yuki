"""Execution-locked Qwen3.8-Flash direct-native screening experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .argument_contract import LITERAL_SOURCE, tool_argument_contract
from .backends import MAX_GENERATION_TOKENS, OpenRouterNativeToolBackend
from .benchmark import BenchmarkCase, BenchmarkRunner, _percentile, load_cases
from .dispatcher import Dispatcher
from .gpt_oss_direct import PHASE2_CASES, PHASE2_SHA256
from .registry import ToolRegistry

ROOT = Path(__file__).parent
BROAD_CASES = ROOT / "benchmark_cases.json"
TARGETED_CASES = ROOT / "benchmark_hammer_targeted_cases.json"
BROAD_SHA256 = "9c4317eee6b271c50bf364f8f96a26b5904899e8a3cde83387a27de59de4c0ba"
TARGETED_SHA256 = "b416a10cec7b7fdac805b0114d8a2be642b4434ccd03d8e4d833df79b8da013d"
MODEL_ID = "qwen/qwen3.8-flash"
PROVIDER = "Alibaba"
BASE_URL = "https://openrouter.ai/api/v1"
REASONING = {"effort": "none", "exclude": True}
TEMPERATURE = 0
SEED = 0
MAX_RETRIES = 0
TOOL_CHOICE = "auto"

REPORT_ROOT = ROOT / "reports" / "qwen38-flash" / "direct-native"
RUN_VERSION = "v2"
SUITES = {
    "broad-72": (BROAD_CASES, BROAD_SHA256, 72),
    "targeted-96": (TARGETED_CASES, TARGETED_SHA256, 96),
}

OVER_REJECTION_MAX_TOTAL = 8
OVER_REJECTION_MAX_TOOL_RATE = 0.50
CATASTROPHIC_LITERAL_MAX_RATE = 0.50
CATASTROPHIC_LITERAL_MAX_TOOL_RATE = 0.75


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> dict[str, str]:
    expected = {
        PHASE2_CASES: PHASE2_SHA256,
        BROAD_CASES: BROAD_SHA256,
        TARGETED_CASES: TARGETED_SHA256,
    }
    actual = {str(path): _sha256(path) for path in expected}
    mismatches = {
        str(path): {"expected": checksum, "actual": actual[str(path)]}
        for path, checksum in expected.items()
        if actual[str(path)] != checksum
    }
    if mismatches:
        raise RuntimeError(f"Frozen screening inputs changed: {mismatches}")
    return actual


def suite_cases(name: str) -> list[BenchmarkCase]:
    verify_sources()
    try:
        path, _, expected_count = SUITES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown screening suite: {name}") from exc
    cases = load_cases(path)
    if len(cases) != expected_count:
        raise RuntimeError(
            f"Suite {name} changed size: {len(cases)} != {expected_count}"
        )
    return cases


def schema_snapshot(registry: ToolRegistry) -> tuple[list[dict[str, Any]], str]:
    schemas = registry.strict_openai_schemas(registry.names)
    encoded = json.dumps(
        schemas,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return schemas, hashlib.sha256(encoded).hexdigest()


def create_backend(max_api_cost_usd: float) -> OpenRouterNativeToolBackend:
    return OpenRouterNativeToolBackend(
        MODEL_ID,
        provider=PROVIDER,
        max_api_cost_usd=max_api_cost_usd,
        max_retries=MAX_RETRIES,
        reasoning_effort=REASONING["effort"],
        reasoning_exclude=REASONING["exclude"],
        temperature=TEMPERATURE,
        seed=SEED,
        tool_choice=TOOL_CHOICE,
    )


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid checkpoint line {line_number}: {exc}"
            ) from exc
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n", encoding="utf-8"
    )
    return checksum


def _multiple_calls(case: dict[str, Any]) -> bool:
    decoded = case["parse"].get("decoded")
    return isinstance(decoded, list) and len(decoded) > 1


def _invented_argument(case: dict[str, Any]) -> bool:
    actual = case.get("actual_arguments")
    expected = case["expected_arguments"]
    return isinstance(actual, dict) and bool(set(actual) - set(expected))


def _failure_taxonomy(case: dict[str, Any]) -> list[str]:
    if case["exact_call"]:
        return []
    labels: list[str] = []
    if (case.get("generation") or {}).get("error"):
        labels.append("generation_or_provider_error")
    if _multiple_calls(case):
        labels.append("multiple_calls")
    if case["malformed"]:
        labels.append("malformed_native_call")
    if case["parse"]["rejected"]:
        labels.append("no_call_or_explicit_rejection")
    if case["actual_tool"] != case["expected_tool"]:
        labels.append("wrong_tool")
    elif not case["strict_arguments_correct"]:
        labels.append("correct_tool_wrong_argument")
        if case["execution_equivalent_arguments_correct"]:
            labels.append("harmless_normalization")
        elif tool_argument_contract(case["expected_tool"])["mode"] == LITERAL_SOURCE:
            labels.append("destructive_literal_rewrite")
    if _invented_argument(case):
        labels.append("invented_argument")
    if not case["schema_valid"]:
        labels.append("schema_invalid_argument")
    return labels


def _enrich_case(case: dict[str, Any], suite: str) -> dict[str, Any]:
    enriched = dict(case)
    enriched["suite"] = suite
    enriched["failure_taxonomy"] = _failure_taxonomy(enriched)
    enriched["multiple_calls"] = _multiple_calls(enriched)
    enriched["invented_argument"] = _invented_argument(enriched)
    enriched["preservation_aware_arguments_correct"] = None
    enriched["preservation_aware_exact_call"] = None
    enriched["preservation_aware_note"] = (
        "Unavailable: these frozen development suites contain no independent "
        "preservation-alternative annotations."
    )
    return enriched


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    generations = [case.get("generation") for case in cases]
    generations = [generation for generation in generations if generation]
    latencies = [
        generation["latency_ms"]
        for generation in generations
        if generation.get("latency_ms") is not None
    ]
    token_rates = [
        generation["tokens_per_second"]
        for generation in generations
        if generation.get("tokens_per_second") is not None
    ]
    prompt_tokens = sum(generation.get("prompt_tokens") or 0 for generation in generations)
    output_tokens = sum(generation.get("generated_tokens") or 0 for generation in generations)
    costs = [
        (generation.get("extra") or {}).get("api_cost_usd")
        for generation in generations
    ]
    costs = [float(cost) for cost in costs if isinstance(cost, int | float)]
    scoreable = [case for case in cases if case["contract_scoring"]["scoreable"]]
    return {
        "total_cases": total,
        "tool_selection_successes": sum(case["tool_correct"] for case in cases),
        "tool_selection_accuracy": mean(case["tool_correct"] for case in cases),
        "strict_exact_successes": sum(case["exact_call"] for case in cases),
        "strict_exact_call_accuracy": mean(case["exact_call"] for case in cases),
        "execution_equivalent_exact_successes": sum(case["execution_equivalent_call"] for case in cases),
        "execution_equivalent_exact_call_accuracy": mean(case["execution_equivalent_call"] for case in cases),
        "preservation_aware_exact_call_accuracy": None,
        "contract_scoreable_cases": len(scoreable),
        "contract_scoring_coverage_rate": len(scoreable) / total,
        "contract_exact_successes": sum(case["contract_exact_call"] is True for case in scoreable),
        "contract_exact_call_accuracy": mean(case["contract_exact_call"] is True for case in scoreable) if scoreable else None,
        "schema_valid_successes": sum(case["schema_valid"] for case in cases),
        "schema_valid_output_rate": mean(case["schema_valid"] for case in cases),
        "malformed_count": sum(case["malformed"] for case in cases),
        "malformed_output_rate": mean(case["malformed"] for case in cases),
        "explicit_rejection_count": sum(case["parse"]["rejected"] for case in cases),
        "no_call_count": sum(case["actual_tool"] is None for case in cases),
        "explicit_rejection_or_no_call_count": sum(
            case["parse"]["rejected"] or case["actual_tool"] is None
            for case in cases
        ),
        "multiple_call_count": sum(case["multiple_calls"] for case in cases),
        "wrong_tool_count": sum(not case["tool_correct"] for case in cases),
        "correct_tool_wrong_argument_count": sum(case["tool_correct"] and not case["strict_arguments_correct"] for case in cases),
        "average_inference_latency_ms": mean(latencies) if latencies else None,
        "p50_inference_latency_ms": _percentile(latencies, 0.50),
        "p95_inference_latency_ms": _percentile(latencies, 0.95),
        "average_output_tokens_per_second": mean(token_rates) if token_rates else None,
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "observed_api_cost_usd": round(sum(costs), 10),
        "cost_reported_cases": len(costs),
    }


def _per_tool(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["expected_tool"]].append(case)
    return {tool: _metrics(scoped) for tool, scoped in sorted(grouped.items())}


def _failure_summary(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for case in cases:
        counts.update(case["failure_taxonomy"])
    return dict(sorted(counts.items()))


def _literal_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    literal = [
        case
        for case in cases
        if tool_argument_contract(case["expected_tool"])["mode"] == LITERAL_SOURCE
        and case["tool_correct"]
        and case["schema_valid"]
    ]
    failures = [case for case in literal if not case["strict_arguments_correct"]]
    by_tool = {}
    for tool in sorted({case["expected_tool"] for case in literal}):
        scoped = [case for case in literal if case["expected_tool"] == tool]
        broken = sum(not case["strict_arguments_correct"] for case in scoped)
        by_tool[tool] = {
            "correct_tool_schema_valid_cases": len(scoped),
            "official_exact_successes": len(scoped) - broken,
            "destructive_rewrite_count": broken,
            "destructive_rewrite_rate": broken / len(scoped) if scoped else None,
        }
    return {
        "basis": "official exact-sensitive gold; no missing literal annotation was fabricated",
        "correct_tool_schema_valid_cases": len(literal),
        "official_exact_successes": len(literal) - len(failures),
        "destructive_rewrite_count": len(failures),
        "destructive_rewrite_rate": len(failures) / len(literal) if literal else None,
        "by_tool": by_tool,
        "failures": [
            {
                "suite": case["suite"],
                "case_id": case["case_id"],
                "tool": case["expected_tool"],
                "request": case["request"],
                "expected_arguments": case["expected_arguments"],
                "actual_arguments": case["actual_arguments"],
            }
            for case in failures
        ],
    }


def _qualitative_gate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = [case for case in cases if case["parse"]["rejected"]]
    rejection_by_tool = Counter(case["expected_tool"] for case in rejected)
    tool_totals = Counter(case["expected_tool"] for case in cases)
    over_rejection = len(rejected) > OVER_REJECTION_MAX_TOTAL or any(
        rejection_by_tool[tool] / tool_totals[tool] >= OVER_REJECTION_MAX_TOOL_RATE
        for tool in rejection_by_tool
        if tool_totals[tool] >= 4
    )
    literal = _literal_summary(cases)
    catastrophic_literal = bool(
        literal["destructive_rewrite_rate"] is not None
        and literal["destructive_rewrite_rate"] > CATASTROPHIC_LITERAL_MAX_RATE
    ) or any(
        value["correct_tool_schema_valid_cases"] >= 4
        and value["destructive_rewrite_rate"] > CATASTROPHIC_LITERAL_MAX_TOOL_RATE
        for value in literal["by_tool"].values()
    )
    return {
        "over_rejection_pass": not over_rejection,
        "over_rejection_definition": {
            "maximum_total": OVER_REJECTION_MAX_TOTAL,
            "maximum_per_tool_rate": OVER_REJECTION_MAX_TOOL_RATE,
            "minimum_tool_cases": 4,
        },
        "catastrophic_literal_rewrite_pass": not catastrophic_literal,
        "catastrophic_literal_definition": {
            "maximum_aggregate_rate": CATASTROPHIC_LITERAL_MAX_RATE,
            "maximum_per_tool_rate": CATASTROPHIC_LITERAL_MAX_TOOL_RATE,
            "minimum_tool_cases": 4,
        },
    }


def _assemble_report(
    cases: list[dict[str, Any]],
    *,
    suite: str,
    schemas: list[dict[str, Any]],
    schema_sha256: str,
) -> dict[str, Any]:
    suite_checksum: str | dict[str, str]
    if suite in SUITES:
        suite_checksum = SUITES[suite][1]
    else:
        suite_checksum = {
            "broad-72": BROAD_SHA256,
            "targeted-96": TARGETED_SHA256,
        }
    return {
        "configuration": {
            "experiment": "qwen38_flash_direct_native_screen",
            "suite": suite,
            "model_id": MODEL_ID,
            "provider": PROVIDER,
            "base_url": BASE_URL,
            "api_mode": "openai_compatible_chat_completions_provider_native_tool_calls",
            "native_dialect": "openai_native_tool_calls",
            "tool_choice": TOOL_CHOICE,
            "historical_profile_difference": (
                "Alibaba rejected tool_choice='required' before inference; the "
                "screen uses the documented provider-compatible 'auto' value."
            ),
            "parallel_tool_calls_parameter_sent": False,
            "reasoning": REASONING,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_GENERATION_TOKENS,
            "max_retries": MAX_RETRIES,
            "generation_count_per_case": 1,
            "stateless": True,
            "conversation_history": False,
            "all_18_schemas_every_case": True,
            "schema_profile": "historical_live_registry_strict_validation_adapter",
            "schema_sha256": schema_sha256,
            "schemas_sent": schemas,
            "schema_profile_matches_gpt_oss_full450_sha256": schema_sha256 == "22139a7c2a7ad583ec38b1fb98715e876c1a9b51554607859bde75688dc6523d",
            "semantic_delegation": False,
            "hammer_used": False,
            "domain_selector_used": False,
            "regex_router_used": False,
            "post_hoc_tool_substitution": False,
            "retry_or_self_correction": False,
            "execution_capability": False,
            "yuki_tool_execution_attempts": 0,
            "sacred_450_used_as_evaluation": False,
            "sacred_450_sha256": PHASE2_SHA256,
            "suite_sha256": suite_checksum,
        },
        "metrics": _metrics(cases),
        "per_tool": _per_tool(cases),
        "failure_taxonomy": _failure_summary(cases),
        "literal_preservation": _literal_summary(cases),
        "qualitative_gate": _qualitative_gate(cases),
        "case_results": cases,
    }


def run_suite(
    backend: OpenRouterNativeToolBackend,
    *,
    suite: str,
    output: Path,
    failures_output: Path,
    raw_output: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    verify_sources()
    cases = suite_cases(suite)
    registry = ToolRegistry()
    schemas, schema_sha256 = schema_snapshot(registry)
    order = {case.case_id: index for index, case in enumerate(cases)}
    completed: dict[str, dict[str, Any]] = {}
    for row in _load_checkpoint(checkpoint):
        if row.get("version") != 1 or row.get("suite") != suite:
            raise RuntimeError("Checkpoint version or suite mismatch")
        if row.get("suite_sha256") != SUITES[suite][1]:
            raise RuntimeError("Checkpoint suite checksum mismatch")
        if row.get("schema_sha256") != schema_sha256:
            raise RuntimeError("Checkpoint schema checksum mismatch")
        if row.get("model_id") != MODEL_ID or row.get("provider") != PROVIDER:
            raise RuntimeError("Checkpoint model/provider mismatch")
        case_id = row["case_result"]["case_id"]
        if case_id not in order or case_id in completed:
            raise RuntimeError(f"Invalid or duplicate checkpoint case: {case_id}")
        completed[case_id] = row

    pending = [case for case in cases if case.case_id not in completed]

    def save_case(case_result: dict[str, Any], fine_tuning_example: dict[str, Any] | None) -> None:
        case_result["index"] = order[case_result["case_id"]]
        enriched = _enrich_case(case_result, suite)
        generation = enriched.get("generation") or {}
        extra = generation.get("extra") or {}
        returned_provider = extra.get("provider")
        returned_model = extra.get("returned_model")
        row = {
            "version": 1,
            "suite": suite,
            "suite_sha256": SUITES[suite][1],
            "schema_sha256": schema_sha256,
            "model_id": MODEL_ID,
            "provider": PROVIDER,
            "case_result": enriched,
            "failure_record": fine_tuning_example,
        }
        completed[enriched["case_id"]] = row
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if returned_provider is not None and returned_provider != PROVIDER:
            raise RuntimeError(
                f"Provider changed on {enriched['case_id']}: {returned_provider!r}"
            )
        if returned_model is not None and "qwen3.8-flash" not in returned_model:
            raise RuntimeError(
                f"Returned model changed on {enriched['case_id']}: {returned_model!r}"
            )

    if pending:
        BenchmarkRunner(Dispatcher(registry, backend)).run(
            pending,
            group="all",
            router_mode="model_only",
            output_mode="native",
            execute=False,
            progress_label=f"Qwen3.8-Flash direct {suite}",
            case_callback=save_case,
        )

    ordered = [completed[case.case_id] for case in cases]
    case_results = [
        _enrich_case(row["case_result"], suite)
        for row in ordered
    ]
    report = _assemble_report(
        case_results,
        suite=suite,
        schemas=schemas,
        schema_sha256=schema_sha256,
    )
    if any(case["execution"]["executed"] for case in case_results):
        raise RuntimeError("Execution lock was violated")
    if any(len(case["offered_tools"]) != 18 for case in case_results):
        raise RuntimeError("A case did not expose all 18 tools")
    _write_json(output, report)
    failures = [case for case in case_results if not case["exact_call"]]
    _write_jsonl(failures_output, failures)
    raw_rows = [
        {
            "suite": suite,
            "case_id": case["case_id"],
            "raw_model_generation": (case.get("generation") or {}).get("raw_text"),
            "generation_error": (case.get("generation") or {}).get("error"),
            "raw_api_response": ((case.get("generation") or {}).get("extra") or {}).get("raw_api_response"),
        }
        for case in case_results
    ]
    _write_jsonl(raw_output, raw_rows)
    for path in (output, failures_output, raw_output, checkpoint):
        _write_checksum(path)
    return report


def combine_reports(broad: dict[str, Any], targeted: dict[str, Any]) -> dict[str, Any]:
    schemas = broad["configuration"]["schemas_sent"]
    schema_sha256 = broad["configuration"]["schema_sha256"]
    if targeted["configuration"]["schema_sha256"] != schema_sha256:
        raise RuntimeError("Broad and targeted schema profiles differ")
    cases = broad["case_results"] + targeted["case_results"]
    report = _assemble_report(
        cases,
        suite="combined-168",
        schemas=schemas,
        schema_sha256=schema_sha256,
    )
    report["configuration"]["suite_sha256"] = {
        "broad-72": BROAD_SHA256,
        "targeted-96": TARGETED_SHA256,
    }
    metrics = report["metrics"]
    qualitative = report["qualitative_gate"]
    strict_pass = metrics["strict_exact_successes"] >= 135
    contract_pass = (
        metrics["contract_scoreable_cases"] == 168
        and metrics["contract_exact_successes"] >= 135
    )
    checks = {
        "tool_selection_at_least_158": metrics["tool_selection_successes"] >= 158,
        "strict_or_full_coverage_contract_exact_at_least_135": strict_pass or contract_pass,
        "schema_valid_at_least_165": metrics["schema_valid_successes"] >= 165,
        "malformed_at_most_1": metrics["malformed_count"] <= 1,
        "no_major_over_rejection": qualitative["over_rejection_pass"],
        "no_catastrophic_literal_rewriting": qualitative["catastrophic_literal_rewrite_pass"],
    }
    report["screening_gate"] = {
        "predeclared_thresholds": {
            "tool_selection": ">=158/168 (94%)",
            "strict_or_contract_exact": ">=135/168 (80%); contract arm requires all 168 cases scoreable",
            "schema_valid": ">=165/168 (98%)",
            "malformed": "<=1/168 (1%)",
            "over_rejection": qualitative["over_rejection_definition"],
            "catastrophic_literal_rewriting": qualitative["catastrophic_literal_definition"],
        },
        "checks": checks,
        "strict_arm_passed": strict_pass,
        "contract_arm_passed": contract_pass,
        "passed": all(checks.values()),
    }
    return report


def _default_path(suite: str, kind: str) -> Path:
    stem = suite.replace("-", "")
    suffix = "json" if kind == "report" else "jsonl"
    return REPORT_ROOT / (
        f"qwen38-flash-all18-native-{stem}-{RUN_VERSION}-{kind}.{suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.8-Flash direct-native screen")
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--max-api-cost-usd", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failures-output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    output = args.output or _default_path(args.suite, "report")
    failures = args.failures_output or _default_path(args.suite, "failures")
    raw = args.raw_output or _default_path(args.suite, "raw")
    checkpoint = args.checkpoint or _default_path(args.suite, "checkpoint")
    backend = create_backend(args.max_api_cost_usd)
    report = run_suite(
        backend,
        suite=args.suite,
        output=output,
        failures_output=failures,
        raw_output=raw,
        checkpoint=checkpoint,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
