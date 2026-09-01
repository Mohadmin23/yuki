# Qwen3.8-Flash direct-native Yuki tool screen — 168 cases

Date: 2026-08-26

## Result

**Screening gate: PASS.** Qwen3.8-Flash selected the correct Yuki tool on **167/168 (99.4%)**, produced **145/168 (86.3%)** strict exact calls, and returned schema-valid calls on **167/168 (99.4%)**. There were **0 malformed outputs**, **0 explicit rejections**, and **1 no-call case**.

The one no-call case, `target-weather-06`, was not a model rejection: Alibaba returned an upstream shared-pool HTTP 429 before inference. Retries were disabled, so it remains an honest scored failure.

Flash therefore cleared the predeclared development-screen gate and has earned consideration for a future sacred-450 direct-native run. This report does **not** authorize that run, and a 168-case development result must not be treated as superior to a historical 450-case result.

## Exact model and inference configuration

- Model ID requested and returned: `qwen/qwen3.8-flash`
- OpenRouter provider requested and returned: `Alibaba`
- API: OpenAI-compatible Chat Completions with provider-native `tool_calls`
- Tool exposure: all 18 strict-adapted live Yuki schemas on every request
- Tool schema SHA-256: `22139a7c2a7ad583ec38b1fb98715e876c1a9b51554607859bde75688dc6523d`
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
| Broad 72 | 72/72 (100.0%) | 66/72 (91.7%) | 67/72 (93.1%) | 72/72 (100.0%) | 0 | 0 | 1.124 s | $0.011622 |
| Targeted 96 | 95/96 (99.0%) | 79/96 (82.3%) | 80/96 (83.3%) | 95/96 (99.0%) | 0 | 0 | 1.592 s | $0.015034 |
| Combined 168 | 167/168 (99.4%) | 145/168 (86.3%) | 147/168 (87.5%) | 167/168 (99.4%) | 0 | 0 | 1.390 s | $0.026655 |

Preservation-aware scoring is unavailable because these two frozen development suites do not contain independently defined preservation-alternative annotations. It was not fabricated from normalized gold.

## Typed argument-contract score

The contract-aware scorer could independently score **64/168 (38.1%)** cases and passed **63/64 (98.4%)**. Literal-source fields without independent literal annotations remained unscoreable, as required. Consequently, typed coverage is incomplete and the contract arm did **not** qualify for the 168-case gate. The gate passed through historical strict scoring: **145/168**, above the required 135.

## Per-tool results

| Tool | Cases | Tool | Strict | Equivalent | Schema valid |
|---|---:|---:|---:|---:|---:|
| `ask_claude` | 20 | 20/20 (100.0%) | 11/20 (55.0%) | 11/20 (55.0%) | 20/20 (100.0%) |
| `calc` | 4 | 4/4 (100.0%) | 3/4 (75.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `fetch` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `hardware` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `image` | 4 | 4/4 (100.0%) | 3/4 (75.0%) | 3/4 (75.0%) | 4/4 (100.0%) |
| `read` | 28 | 28/28 (100.0%) | 28/28 (100.0%) | 28/28 (100.0%) | 28/28 (100.0%) |
| `recall` | 4 | 4/4 (100.0%) | 3/4 (75.0%) | 3/4 (75.0%) | 4/4 (100.0%) |
| `remember` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `search` | 20 | 20/20 (100.0%) | 11/20 (55.0%) | 11/20 (55.0%) | 20/20 (100.0%) |
| `see` | 20 | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) |
| `shell` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `time` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `weather` | 20 | 19/20 (95.0%) | 18/20 (90.0%) | 19/20 (95.0%) | 19/20 (95.0%) |
| `yuki_append` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `yuki_delete` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `yuki_list` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `yuki_read` | 12 | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `yuki_write` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |

The dominant weaknesses are argument rewriting in `search` and `ask_claude`. Tool ontology itself was excellent on this set: `read` versus `yuki_read`, `see` versus `image`, storage write/append, recall/search, remember/write, and shell/hardware all had perfect tool selection. `hardware process` versus `top` produced no error in the four broad hardware cases, but this suite is too small to establish that boundary reliably.

## Failure taxonomy

| Failure class | Count |
|---|---:|
| `wrong_tool` | 1 |
| `correct_tool_wrong_argument` | 22 |
| `harmless_normalization` | 2 |
| `destructive_literal_rewrite` | 19 |
| `malformed_native_call` | 0 |
| `no_call_or_explicit_rejection` | 0 |
| `invented_argument` | 0 |
| `schema_invalid_argument` | 1 |
| `multiple_calls` | 0 |
| `generation_or_provider_error` | 1 |

Counts overlap by design. For example, a destructive rewrite is also a correct-tool/wrong-argument failure. The single provider error is also counted under historical scoring as no selected tool and schema invalid; it is kept separate so it is not mistaken for semantic routing behavior.

## Literal preservation

Among **104** correct-tool, schema-valid cases whose tools are exact-sensitive under the current field contract, **85** matched official gold and **19 (18.3%)** rewrote it.

| Exact-sensitive tool | Eligible | Exact | Rewritten | Rewrite rate |
|---|---:|---:|---:|---:|
| `ask_claude` | 20 | 11 | 9 | 45.0% |
| `fetch` | 4 | 4 | 0 | 0.0% |
| `image` | 4 | 3 | 1 | 25.0% |
| `read` | 28 | 28 | 0 | 0.0% |
| `search` | 20 | 11 | 9 | 45.0% |
| `shell` | 4 | 4 | 0 | 0.0% |
| `yuki_append` | 4 | 4 | 0 | 0.0% |
| `yuki_delete` | 4 | 4 | 0 | 0.0% |
| `yuki_read` | 12 | 12 | 0 | 0.0% |
| `yuki_write` | 4 | 4 | 0 | 0.0% |

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

- Average inference latency: **1.390 s** over 167 completed generations
- p50: **1.016 s**
- p95: **3.631 s**
- Average generated tokens/second: **25.24**
- Input tokens: **458,608**
- Output tokens: **4,752**
- Observed API cost: **$0.026655**

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
| Tool selection ≥158/168 | **PASS — 167/168** |
| Strict exact ≥135/168, or fully covered contract exact | **PASS via strict — 145/168** |
| Schema-valid ≥165/168 | **PASS — 167/168** |
| Malformed ≤1/168 | **PASS — 0/168** |
| No major over-rejection | **PASS** |
| No catastrophic literal rewriting | **PASS** |

**Final gate: PASS.** Flash has earned a future sacred-450 direct-native evaluation under the predeclared screen, but that run remains explicitly deferred and unauthorized. Before such a run, freeze whether `tool_choice=auto` is the accepted Flash deployment profile and retain typed deterministic binding for exact-sensitive fields.

## Safety and frozen-data confirmation

- Only `qwen/qwen3.8-flash` was called. Qwen3.8-27B, GPT-OSS, Hammer, and local models were not called or loaded.
- No semantic delegation, domain selection, model fallback, regex routing, or post-hoc tool substitution occurred.
- No Yuki tool was executed; no production Yuki file was modified.
- Pilot v1.2 was not used as evaluation data.
- Broad-72 SHA-256: `9c4317eee6b271c50bf364f8f96a26b5904899e8a3cde83387a27de59de4c0ba`
- Targeted-96 SHA-256: `b416a10cec7b7fdac805b0114d8a2be642b4434ccd03d8e4d833df79b8da013d`
- Sacred Phase 2 SHA-256 remains: `5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466`

## Artifacts

- `qwen38-flash-direct-native-screen-168-v1.json` — complete machine-readable combined report
- `qwen38-flash-direct-native-screen-168-v1-cases.jsonl` — normalized calls and per-case scoring
- `qwen38-flash-direct-native-screen-168-v1-failures.jsonl` — strict failures
- `qwen38-flash-direct-native-screen-168-v1-raw-responses.jsonl` — raw API responses/errors
- `qwen38-flash-direct-native-screen-168-v1-per-tool.json` — per-tool metrics
- `qwen38-flash-direct-native-screen-168-v1-configuration.json` — frozen configuration and schema snapshot
- `QWEN38-FLASH-DIRECT-NATIVE-SCREEN-168-SHA256SUMS-v1.txt` — artifact/source checksums
