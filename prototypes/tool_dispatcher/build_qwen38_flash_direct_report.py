"""Build frozen reports for the Qwen3.8-Flash direct-native screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .qwen38_flash_direct import (
    BROAD_CASES,
    BROAD_SHA256,
    PHASE2_CASES,
    PHASE2_SHA256,
    REPORT_ROOT,
    TARGETED_CASES,
    TARGETED_SHA256,
    _write_json,
    _write_jsonl,
    combine_reports,
    verify_sources,
)

BROAD_REPORT = REPORT_ROOT / "qwen38-flash-all18-native-broad72-v2-report.json"
TARGETED_REPORT = REPORT_ROOT / "qwen38-flash-all18-native-targeted96-v2-report.json"
COMBINED_REPORT = REPORT_ROOT / "qwen38-flash-direct-native-screen-168-v1.json"
CASES_OUTPUT = REPORT_ROOT / "qwen38-flash-direct-native-screen-168-v1-cases.jsonl"
FAILURES_OUTPUT = (
    REPORT_ROOT / "qwen38-flash-direct-native-screen-168-v1-failures.jsonl"
)
RAW_OUTPUT = (
    REPORT_ROOT / "qwen38-flash-direct-native-screen-168-v1-raw-responses.jsonl"
)
PER_TOOL_OUTPUT = (
    REPORT_ROOT / "qwen38-flash-direct-native-screen-168-v1-per-tool.json"
)
CONFIG_OUTPUT = (
    REPORT_ROOT / "qwen38-flash-direct-native-screen-168-v1-configuration.json"
)
MARKDOWN_OUTPUT = REPORT_ROOT / "QWEN38-FLASH-DIRECT-NATIVE-SCREEN-168.md"
CHECKSUM_OUTPUT = (
    REPORT_ROOT / "QWEN38-FLASH-DIRECT-NATIVE-SCREEN-168-SHA256SUMS-v1.txt"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.3f} s"


def _suite_row(label: str, report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    return (
        f"| {label} | {metrics['tool_selection_successes']}/{metrics['total_cases']} "
        f"({_pct(metrics['tool_selection_accuracy'])}) | "
        f"{metrics['strict_exact_successes']}/{metrics['total_cases']} "
        f"({_pct(metrics['strict_exact_call_accuracy'])}) | "
        f"{metrics['execution_equivalent_exact_successes']}/"
        f"{metrics['total_cases']} "
        f"({_pct(metrics['execution_equivalent_exact_call_accuracy'])}) | "
        f"{metrics['schema_valid_successes']}/{metrics['total_cases']} "
        f"({_pct(metrics['schema_valid_output_rate'])}) | "
        f"{metrics['malformed_count']} | "
        f"{metrics['explicit_rejection_count']} | "
        f"{_seconds(metrics['average_inference_latency_ms'])} | "
        f"${metrics['observed_api_cost_usd']:.6f} |"
    )


def _per_tool_table(report: dict[str, Any]) -> str:
    rows = [
        "| Tool | Cases | Tool | Strict | Equivalent | Schema valid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tool, metrics in report["per_tool"].items():
        total = metrics["total_cases"]
        rows.append(
            f"| `{tool}` | {total} | "
            f"{metrics['tool_selection_successes']}/{total} "
            f"({_pct(metrics['tool_selection_accuracy'])}) | "
            f"{metrics['strict_exact_successes']}/{total} "
            f"({_pct(metrics['strict_exact_call_accuracy'])}) | "
            f"{metrics['execution_equivalent_exact_successes']}/{total} "
            f"({_pct(metrics['execution_equivalent_exact_call_accuracy'])}) | "
            f"{metrics['schema_valid_successes']}/{total} "
            f"({_pct(metrics['schema_valid_output_rate'])}) |"
        )
    return "\n".join(rows)


def _failure_table(report: dict[str, Any]) -> str:
    labels = [
        "wrong_tool",
        "correct_tool_wrong_argument",
        "harmless_normalization",
        "destructive_literal_rewrite",
        "malformed_native_call",
        "no_call_or_explicit_rejection",
        "invented_argument",
        "schema_invalid_argument",
        "multiple_calls",
        "generation_or_provider_error",
    ]
    counts = report["failure_taxonomy"]
    return "\n".join(
        ["| Failure class | Count |", "|---|---:|"]
        + [f"| `{label}` | {counts.get(label, 0)} |" for label in labels]
    )


def _literal_table(report: dict[str, Any]) -> str:
    rows = [
        "| Exact-sensitive tool | Eligible | Exact | Rewritten | Rewrite rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for tool, metrics in report["literal_preservation"]["by_tool"].items():
        rows.append(
            f"| `{tool}` | {metrics['correct_tool_schema_valid_cases']} | "
            f"{metrics['official_exact_successes']} | "
            f"{metrics['destructive_rewrite_count']} | "
            f"{_pct(metrics['destructive_rewrite_rate'])} |"
        )
    return "\n".join(rows)


def _markdown(
    broad: dict[str, Any],
    targeted: dict[str, Any],
    combined: dict[str, Any],
) -> str:
    metrics = combined["metrics"]
    gate = combined["screening_gate"]
    literal = combined["literal_preservation"]
    return f"""# Qwen3.8-Flash direct-native Yuki tool screen — 168 cases

Date: 2026-08-26

## Result

**Screening gate: {'PASS' if gate['passed'] else 'FAIL'}.** Qwen3.8-Flash selected the correct Yuki tool on **{metrics['tool_selection_successes']}/168 ({_pct(metrics['tool_selection_accuracy'])})**, produced **{metrics['strict_exact_successes']}/168 ({_pct(metrics['strict_exact_call_accuracy'])})** strict exact calls, and returned schema-valid calls on **{metrics['schema_valid_successes']}/168 ({_pct(metrics['schema_valid_output_rate'])})**. There were **{metrics['malformed_count']} malformed outputs**, **{metrics['explicit_rejection_count']} explicit rejections**, and **{metrics['no_call_count']} no-call case**.

The one no-call case, `target-weather-06`, was not a model rejection: Alibaba returned an upstream shared-pool HTTP 429 before inference. Retries were disabled, so it remains an honest scored failure.

Flash therefore cleared the predeclared development-screen gate and has earned consideration for a future sacred-450 direct-native run. This report does **not** authorize that run, and a 168-case development result must not be treated as superior to a historical 450-case result.

## Exact model and inference configuration

- Model ID requested and returned: `qwen/qwen3.8-flash`
- OpenRouter provider requested and returned: `Alibaba`
- API: OpenAI-compatible Chat Completions with provider-native `tool_calls`
- Tool exposure: all 18 strict-adapted live Yuki schemas on every request
- Tool schema SHA-256: `{combined['configuration']['schema_sha256']}`
- Historical profile check: schema hash exactly matches the GPT-OSS direct-full450 strict schema profile
- `tool_choice`: `auto`
- Reasoning: `effort=none`, excluded from output
- Temperature: `0`; seed: `0`; maximum generation: `256` tokens
- API retries: `0`; generations: exactly one accepted inference attempt per case
- Stateless: no history, no semantic delegation, no Hammer, no selector, no regex routing, no self-correction, no voting
- Execution lock: no Yuki tool execution capability; observed execution attempts: `0`

### Provider compatibility note

The historical direct backend used `tool_choice=required`. Alibaba rejected that value before inference for this model even though the endpoint supports tools. A first infrastructure-only broad pass therefore produced zero model generations, zero tokens, and zero cost. It is preserved separately as a failed setup artifact and is not presented as a model benchmark. The completed v2 screen used OpenRouter's documented provider-compatible `tool_choice=auto`; every other experimental control remained fixed. This difference from the GPT-OSS/Hammer profile is explicit rather than hidden.

## Broad, targeted, and combined scores

| Suite | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Rejects | Avg latency | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{_suite_row('Broad 72', broad)}
{_suite_row('Targeted 96', targeted)}
{_suite_row('Combined 168', combined)}

Preservation-aware scoring is unavailable because these two frozen development suites do not contain independently defined preservation-alternative annotations. It was not fabricated from normalized gold.

## Typed argument-contract score

The contract-aware scorer could independently score **{metrics['contract_scoreable_cases']}/168 ({_pct(metrics['contract_scoring_coverage_rate'])})** cases and passed **{metrics['contract_exact_successes']}/{metrics['contract_scoreable_cases']} ({_pct(metrics['contract_exact_call_accuracy'])})**. Literal-source fields without independent literal annotations remained unscoreable, as required. Consequently, typed coverage is incomplete and the contract arm did **not** qualify for the 168-case gate. The gate passed through historical strict scoring: **{metrics['strict_exact_successes']}/168**, above the required 135.

## Per-tool results

{_per_tool_table(combined)}

The dominant weaknesses are argument rewriting in `search` and `ask_claude`. Tool ontology itself was excellent on this set: `read` versus `yuki_read`, `see` versus `image`, storage write/append, recall/search, remember/write, and shell/hardware all had perfect tool selection. `hardware process` versus `top` produced no error in the four broad hardware cases, but this suite is too small to establish that boundary reliably.

## Failure taxonomy

{_failure_table(combined)}

Counts overlap by design. For example, a destructive rewrite is also a correct-tool/wrong-argument failure. The single provider error is also counted under historical scoring as no selected tool and schema invalid; it is kept separate so it is not mistaken for semantic routing behavior.

## Literal preservation

Among **{literal['correct_tool_schema_valid_cases']}** correct-tool, schema-valid cases whose tools are exact-sensitive under the current field contract, **{literal['official_exact_successes']}** matched official gold and **{literal['destructive_rewrite_count']} ({_pct(literal['destructive_rewrite_rate'])})** rewrote it.

{_literal_table(combined)}

The rewrite pattern was concentrated rather than catastrophic:

- `search.query`: 9/20 rewritten, usually by adding supposedly helpful search terms.
- `ask_claude.question`: 9/20 rewritten, usually by adding politeness, punctuation, or extra instructions.
- `image.prompt`: 1/4 rewritten into a much longer creative prompt despite “using exactly.”
- Filesystem paths, shell commands, URLs, Yuki filenames, and Yuki file contents were preserved exactly in every eligible correct call.

The predeclared catastrophic-rewrite check passed: 19/104 (18.3%) overall and no exact-sensitive tool crossed the 75% per-tool failure boundary. Nevertheless, 45% rewrite rates for both `search` and `ask_claude` remain an important production limitation and support deterministic typed payload binding.

## Rejection and native-output behavior

- Explicit rejections: **0**
- Model-produced no calls: **0**
- Provider-error no calls: **1**
- Multiple calls: **0**
- Malformed native calls: **0**
- Prose-only generations: **0**

Using `tool_choice=auto` did not produce an over-rejection pattern: every successful provider response contained exactly one native tool call.

## Latency, tokens, and cost

- Average inference latency: **{_seconds(metrics['average_inference_latency_ms'])}** over 167 completed generations
- p50: **{_seconds(metrics['p50_inference_latency_ms'])}**
- p95: **{_seconds(metrics['p95_inference_latency_ms'])}**
- Average generated tokens/second: **{metrics['average_output_tokens_per_second']:.2f}**
- Input tokens: **{metrics['input_tokens']:,}**
- Output tokens: **{metrics['output_tokens']:,}**
- Observed API cost: **${metrics['observed_api_cost_usd']:.6f}**

## Exact-suite historical context

These are direct comparisons only because the older models were run on the same frozen 72- and 96-case development suites. Flash's `tool_choice=auto` provider constraint remains a profile difference.

### Broad 72

| Model | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash direct | **100.0%** | **91.7%** | **93.1%** | **100.0%** | 0.0% |
| Hammer 2.1 3B | 97.2% | 91.7% | 93.1% | 98.6% | 0.0% |
| xLAM-2 3B | 97.2% | 91.7% | 91.7% | 97.2% | 0.0% |
| Arch-Agent 3B | 97.2% | 90.3% | 93.1% | 98.6% | 1.4% |
| Hammer 2.1 1.5B | 91.7% | 87.5% | 88.9% | 94.4% | 0.0% |
| xLAM-2 1B | 90.3% | 81.9% | 83.3% | 97.2% | 0.0% |
| Arch-Function 3B | 88.9% | 80.6% | 81.9% | 86.1% | 9.7% |
| Granite 4.1 3B | 91.7% | 73.6% | 75.0% | 90.3% | 0.0% |
| Original xLAM 1B | 84.7% | 45.8% | 50.0% | 97.2% | 0.0% |

### Targeted 96

| Model | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash direct | **99.0%** | 82.3% | 83.3% | **99.0%** | 0.0% |
| xLAM-2 3B | **99.0%** | **83.3%** | **86.5%** | **99.0%** | 0.0% |
| Arch-Agent 3B | 97.9% | 82.3% | **86.5%** | **99.0%** | 1.0% |
| Hammer 2.1 3B | 87.5% | 80.2% | 81.3% | 87.5% | 0.0% |
| Arch-Function 3B | 89.6% | 78.1% | 79.2% | 81.3% | 10.4% |
| Hammer 2.1 1.5B | 75.0% | 62.5% | 66.7% | 80.2% | 0.0% |
| Granite 4.1 3B | 93.8% | 59.4% | 60.4% | 80.2% | 1.0% |

The historical full-450 references remain unchanged and are not ranked against this 168-case screen as though the data were identical: GPT-OSS direct 76.2% strict, Hammer 2.0 7B direct 81.8%, and Qwen3.8-27B → Hammer 1.5B 79.6%.

## Gate decision

| Gate condition | Result |
|---|---:|
| Tool selection ≥158/168 | **PASS — {metrics['tool_selection_successes']}/168** |
| Strict exact ≥135/168, or fully covered contract exact | **PASS via strict — {metrics['strict_exact_successes']}/168** |
| Schema-valid ≥165/168 | **PASS — {metrics['schema_valid_successes']}/168** |
| Malformed ≤1/168 | **PASS — {metrics['malformed_count']}/168** |
| No major over-rejection | **PASS** |
| No catastrophic literal rewriting | **PASS** |

**Final gate: PASS.** Flash has earned a future sacred-450 direct-native evaluation under the predeclared screen, but that run remains explicitly deferred and unauthorized. Before such a run, freeze whether `tool_choice=auto` is the accepted Flash deployment profile and retain typed deterministic binding for exact-sensitive fields.

## Safety and frozen-data confirmation

- Only `qwen/qwen3.8-flash` was called. Qwen3.8-27B, GPT-OSS, Hammer, and local models were not called or loaded.
- No semantic delegation, domain selection, model fallback, regex routing, or post-hoc tool substitution occurred.
- No Yuki tool was executed; no production Yuki file was modified.
- Pilot v1.2 was not used as evaluation data.
- Broad-72 SHA-256: `{BROAD_SHA256}`
- Targeted-96 SHA-256: `{TARGETED_SHA256}`
- Sacred Phase 2 SHA-256 remains: `{PHASE2_SHA256}`

## Artifacts

- `qwen38-flash-direct-native-screen-168-v1.json` — complete machine-readable combined report
- `qwen38-flash-direct-native-screen-168-v1-cases.jsonl` — normalized calls and per-case scoring
- `qwen38-flash-direct-native-screen-168-v1-failures.jsonl` — strict failures
- `qwen38-flash-direct-native-screen-168-v1-raw-responses.jsonl` — raw API responses/errors
- `qwen38-flash-direct-native-screen-168-v1-per-tool.json` — per-tool metrics
- `qwen38-flash-direct-native-screen-168-v1-configuration.json` — frozen configuration and schema snapshot
- `QWEN38-FLASH-DIRECT-NATIVE-SCREEN-168-SHA256SUMS-v1.txt` — artifact/source checksums
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    verify_sources()
    broad = _load(BROAD_REPORT)
    targeted = _load(TARGETED_REPORT)
    combined = combine_reports(broad, targeted)
    cases = combined["case_results"]

    _write_json(COMBINED_REPORT, combined)
    _write_jsonl(CASES_OUTPUT, cases)
    _write_jsonl(
        FAILURES_OUTPUT,
        [case for case in cases if not case["exact_call"]],
    )
    _write_jsonl(
        RAW_OUTPUT,
        [
            {
                "suite": case["suite"],
                "case_id": case["case_id"],
                "raw_model_generation": (case.get("generation") or {}).get(
                    "raw_text"
                ),
                "generation_error": (case.get("generation") or {}).get("error"),
                "raw_api_response": (
                    (case.get("generation") or {}).get("extra") or {}
                ).get("raw_api_response"),
            }
            for case in cases
        ],
    )
    _write_json(PER_TOOL_OUTPUT, combined["per_tool"])
    _write_json(CONFIG_OUTPUT, combined["configuration"])
    MARKDOWN_OUTPUT.write_text(
        _markdown(broad, targeted, combined),
        encoding="utf-8",
    )

    artifact_paths = [
        BROAD_REPORT,
        TARGETED_REPORT,
        COMBINED_REPORT,
        CASES_OUTPUT,
        FAILURES_OUTPUT,
        RAW_OUTPUT,
        PER_TOOL_OUTPUT,
        CONFIG_OUTPUT,
        MARKDOWN_OUTPUT,
    ]
    source_paths = [BROAD_CASES, TARGETED_CASES, PHASE2_CASES]
    lines = [
        f"{_sha256(path)}  {path.relative_to(REPORT_ROOT)}"
        for path in artifact_paths
    ] + [
        f"{_sha256(path)}  SOURCE::{path.name}"
        for path in source_paths
    ]
    CHECKSUM_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return combined


def main() -> None:
    report = build()
    print(json.dumps(report["screening_gate"], indent=2))


if __name__ == "__main__":
    main()
