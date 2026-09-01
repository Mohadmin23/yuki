# xLAM-2 3B FC-R — Yuki Tool Dispatcher Report

Date: 2026-08-19

## Verdict

`Salesforce/xLAM-2-3b-fc-r` is the strongest model tested so far on the 96-case targeted failure suite. It reached 83.3% strict exact-call accuracy and 86.5% execution-equivalent exact-call accuracy, ahead of Hammer 3B, Arch-Function 3B, and Hammer 1.5B on the identical cases.

On the 72-case all-tools smoke suite, it tied Hammer 3B at 91.7% strict exact-call accuracy. Hammer 3B retained a small lead in execution-equivalent exact calls and was substantially faster. xLAM-2 3B therefore improves robustness on the hard targeted prompts, but it is not an unconditional replacement for Hammer 3B.

This remains Phase 1 smoke/diagnostic evidence. The harness correctly marks these results as insufficient for a final reliability conclusion.

## Test configuration

- Model: `Salesforce/xLAM-2-3b-fc-r`
- Local FP16 conversion: `/Volumes/madisk/yuki-tool-dispatcher/xLAM-2-3b-fc-r-fp16`
- Source cache: `/Volumes/madisk/huggingface/hub/models--Salesforce--xLAM-2-3b-fc-r`
- Source model: BF16, Qwen2 architecture
- License: CC-BY-NC-4.0 research release
- Native output: xLAM-2 JSON tool-call array
- Available schemas: all 18 real Yuki tool schemas
- Generation ceiling: 256 tokens with immediate stop after one complete structured call
- State: stateless; one dispatcher generation per request
- Router: model only; no regex auto-detection or fallback
- Tool execution: disabled for routing benchmarks
- Tool results returned to model: none
- Peak MLX memory observed: 6.6521 GB

The existing xLAM-2 adapter was reused without model-specific prompt or scoring changes. Its native output is normalized internally only after generation, then checked against the selected live Yuki schema.

## Native-format verification

The initial weather check produced exactly:

```json
[{"name":"weather","arguments":{"city":"Algiers"}}]
```

It parsed and validated successfully, stopped after the complete 14-token call, and used the expected `tool_call_array` dialect. Cold latency was 12.31 seconds at 9.46 generated tokens/second.

## Broad all-18 benchmark — 72 cases

| Metric | Result |
|---|---:|
| Tool selection | 97.2% |
| Strict argument accuracy | 94.4% |
| Execution-equivalent argument accuracy | 94.4% |
| Strict exact-call accuracy | 91.7% |
| Execution-equivalent exact-call accuracy | 91.7% |
| Schema-valid output | 97.2% |
| Malformed output | 0.0% |
| Average latency | 16.13 s |
| p50 latency | 16.37 s |
| p95 latency | 17.10 s |
| Aggregate generation speed | 9.58 tok/s |
| Peak MLX memory | 6.6521 GB |

All 72 cases ran exactly once. No Yuki tool was executed, so external service availability did not affect routing scores.

### Verified six-model broad comparison

The harness verified that every report used the same 72 cases.

| Model | Tool selection | Strict exact call | Equivalent exact call | Schema valid | Malformed | Avg latency | tok/s | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hammer 2.1 3B | 97.2% | **91.7%** | **93.1%** | **98.6%** | 0.0% | 11.92 s | 9.64 | 6.651 GB |
| **xLAM-2 3B** | 97.2% | **91.7%** | 91.7% | 97.2% | 0.0% | 16.13 s | 9.58 | 6.652 GB |
| Hammer 2.1 1.5B | 91.7% | 87.5% | 88.9% | 94.4% | 0.0% | **4.69 s** | **18.47** | 3.624 GB |
| xLAM-2 1B | 90.3% | 81.9% | 83.3% | 97.2% | 0.0% | 6.22 s | 18.43 | 3.624 GB |
| Arch-Function 3B | 88.9% | 80.6% | 81.9% | 86.1% | 9.7% | 13.58 s | 9.50 | 6.652 GB |
| xLAM 1B | 84.7% | 45.8% | 50.0% | 97.2% | 0.0% | 5.65 s | 18.39 | 3.594 GB |

## Targeted failure suite — 96 cases

| Metric | xLAM-2 3B | Hammer 3B | Arch 3B | Hammer 1.5B |
|---|---:|---:|---:|---:|
| Tool selection | **99.0%** | 87.5% | 89.6% | 75.0% |
| Strict argument accuracy | **83.3%** | 80.2% | 78.1% | 62.5% |
| Equivalent argument accuracy | **86.5%** | 81.3% | 79.2% | 66.7% |
| Strict exact-call accuracy | **83.3%** | 80.2% | 78.1% | 62.5% |
| Equivalent exact-call accuracy | **86.5%** | 81.3% | 79.2% | 66.7% |
| Schema-valid output | **99.0%** | 87.5% | 81.3% | 80.2% |
| Malformed output | **0.0%** | **0.0%** | 10.4% | **0.0%** |
| Average latency | 14.22 s | 10.95 s | 13.81 s | **4.64 s** |
| p50 latency | 13.69 s | 10.34 s | 14.38 s | **4.72 s** |
| p95 latency | 16.83 s | 14.23 s | 18.01 s | **5.38 s** |
| Aggregate generation speed | 9.11 tok/s | 9.41 tok/s | 9.36 tok/s | **18.44 tok/s** |

The harness verified the same 96 targeted cases for all four models.

### xLAM-2 3B results by targeted tool

| Expected tool | Cases | Tool selection | Strict exact call | Equivalent exact call | Schema valid | Malformed |
|---|---:|---:|---:|---:|---:|---:|
| `weather` | 16 | 100.0% | 93.8% | 100.0% | 100.0% | 0.0% |
| `see` | 16 | 100.0% | 87.5% | 87.5% | 100.0% | 0.0% |
| `read` | 24 | 100.0% | 91.7% | 91.7% | 100.0% | 0.0% |
| `yuki_read` | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `search` | 16 | 93.8% | 75.0% | 75.0% | 93.8% | 0.0% |
| `ask_claude` | 16 | 100.0% | 56.3% | 68.8% | 100.0% | 0.0% |

The arithmetic mean of per-case backend-reported token rates is not used here because one one-token rejection reported an artificial 20,495 tok/s generation-only rate. The aggregate rate and median were both approximately 9.11 tok/s and reflect normal generation.

## Failure analysis

Only six broad cases missed strict exactness:

- `see`: selected the correct tool but emitted `{ "target": null }` instead of an empty argument object, making the call schema-invalid.
- `read`: changed `~/notes.md` to `~notes.md`, a real filesystem-breaking error.
- `yuki_append`: preserved the exact payload but selected `yuki_write` instead of append.
- `ask_claude`: two calls changed how much surrounding instruction text was retained in the question.
- `ask_claude`: one call preserved the exact question but misspelled the tool as `ask_clauke`.

The targeted suite adds several useful findings:

- Tool ontology is very strong: 95 of 96 tools were selected correctly.
- `read` versus `yuki_read` is not a capacity problem for this model: all 32 targeted selections were correct, with 30 strict exact calls.
- All 16 weather selections were correct; the sole strict argument difference was execution-equivalent.
- `ask_claude` remains the largest argument-preservation weakness even though all 16 tool selections were correct.
- Search produced the only targeted reject and three additional argument rewrites.
- No targeted generation was malformed.

## Recommendation

xLAM-2 3B should remain a serious finalist. It currently has the best targeted robustness and ties the best broad strict accuracy, with excellent native-output discipline. Its cost is latency: it used essentially the same memory as Hammer 3B but was about 35% slower on the broad suite and about 30% slower on the targeted suite.

Hammer 1.5B remains the efficiency leader. Hammer 3B remains the strongest balanced 3B result because it matches xLAM-2 3B on broad strict accuracy, slightly leads broad equivalent accuracy, and is faster. The remaining Arch-Agent 3B and Granite 4.1 3B tests should be completed before choosing a final dispatcher.

## Artifacts

- `xlam-2-3b-broad-72.json` — full 72-case broad report
- `xlam-2-3b-broad-72-failures.jsonl` — broad failures
- `xlam-2-3b-targeted-96.json` — full targeted report
- `xlam-2-3b-targeted-96-failures.jsonl` — targeted failures
- `xlam-2-3b-broad-six-model-comparison.json` — verified six-model broad comparison
- `xlam-2-3b-targeted-four-model-comparison.json` — verified targeted comparison

