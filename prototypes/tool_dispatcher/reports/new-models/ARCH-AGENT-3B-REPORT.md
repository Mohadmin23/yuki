# Arch-Agent-3B — Yuki Tool Dispatcher Report

Date: 2026-08-19

## Verdict

`katanemo/Arch-Agent-3B` is the strongest balanced 3B result tested so far. It matches xLAM-2 3B's 86.5% execution-equivalent exact-call score on the 96-case targeted suite while running about 16% faster, and it matches Hammer 3B's leading 93.1% execution-equivalent exact-call score on the 72-case broad suite.

Strict scoring exposes more argument rewriting: Arch-Agent reached 82.3% targeted and 90.3% broad strict exact-call accuracy. It produced one conversational refusal in each suite, so its malformed rate is low but not zero.

This remains Phase 1 smoke/diagnostic evidence. The harness correctly marks it as insufficient for a final reliability conclusion.

## Test configuration

- Model: `katanemo/Arch-Agent-3B`
- Local FP16 conversion: `/Volumes/madisk/yuki-tool-dispatcher/Arch-Agent-3B-fp16`
- Source cache: `/Volumes/madisk/huggingface/hub/models--katanemo--Arch-Agent-3B`
- Base architecture: Qwen2.5-Coder-3B-Instruct
- Source dtype: BF16
- License: Katanemo/DigitalOcean Community License; commercial use requires a separate commercial license
- Native output: one JSON call inside `<tool_call>` XML tags
- Available schemas: all 18 real Yuki tool schemas
- Generation ceiling: 256 tokens with immediate stop after one complete structured call
- State: stateless; one dispatcher generation per request
- Router: model only; no regex auto-detection or fallback
- Tool execution: disabled for routing benchmarks
- Tool results returned to model: none
- Peak MLX memory observed: 6.6521 GB

The prototype added only model-name recognition for Arch-Agent's documented native wrapper. It reused the existing Arch XML prompt, parser, canonical normalization, schema validation, scoring, and early JSON stop. No Yuki core code was changed.

## Native-format verification

The initial weather check produced:

```text
<tool_call>
{"name":"weather","arguments":{"city":"Algiers"}}
```

The call parsed and validated exactly, stopped after 14 generated tokens, and contained no assistant prose. Cold latency was 9.87 seconds at 9.94 generated tokens/second.

## Broad all-18 benchmark — 72 cases

| Metric | Result |
|---|---:|
| Tool selection | 97.2% |
| Strict argument accuracy | 90.3% |
| Execution-equivalent argument accuracy | 93.1% |
| Strict exact-call accuracy | 90.3% |
| Execution-equivalent exact-call accuracy | 93.1% |
| Schema-valid output | 98.6% |
| Malformed output | 1.4% |
| Average latency | 12.72 s |
| p50 latency | 12.95 s |
| p95 latency | 14.06 s |
| Generation speed | 9.73 tok/s |
| Peak MLX memory | 6.6521 GB |

All 72 cases ran exactly once. No Yuki tool was executed, so external services did not affect routing scores.

### Verified seven-model broad comparison

The harness verified that all seven reports used the same 72 cases.

| Model | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg latency | tok/s | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hammer 2.1 3B | 97.2% | **91.7%** | **93.1%** | **98.6%** | **0.0%** | 11.92 s | 9.64 | 6.651 GB |
| xLAM-2 3B | 97.2% | **91.7%** | 91.7% | 97.2% | **0.0%** | 16.13 s | 9.58 | 6.652 GB |
| **Arch-Agent 3B** | 97.2% | 90.3% | **93.1%** | **98.6%** | 1.4% | 12.72 s | 9.73 | 6.652 GB |
| Hammer 2.1 1.5B | 91.7% | 87.5% | 88.9% | 94.4% | **0.0%** | **4.69 s** | **18.47** | 3.624 GB |
| xLAM-2 1B | 90.3% | 81.9% | 83.3% | 97.2% | **0.0%** | 6.22 s | 18.43 | 3.624 GB |
| Arch-Function 3B | 88.9% | 80.6% | 81.9% | 86.1% | 9.7% | 13.58 s | 9.50 | 6.652 GB |
| xLAM 1B | 84.7% | 45.8% | 50.0% | 97.2% | **0.0%** | 5.65 s | 18.39 | 3.594 GB |

## Targeted failure suite — 96 cases

| Metric | xLAM-2 3B | Arch-Agent 3B | Hammer 3B | Arch-Function 3B | Hammer 1.5B |
|---|---:|---:|---:|---:|---:|
| Tool selection | **99.0%** | 97.9% | 87.5% | 89.6% | 75.0% |
| Strict exact call | **83.3%** | 82.3% | 80.2% | 78.1% | 62.5% |
| Equivalent exact call | **86.5%** | **86.5%** | 81.3% | 79.2% | 66.7% |
| Schema valid | **99.0%** | **99.0%** | 87.5% | 81.3% | 80.2% |
| Malformed | **0.0%** | 1.0% | **0.0%** | 10.4% | **0.0%** |
| Average latency | 14.22 s | 11.98 s | **10.95 s** | 13.81 s | **4.64 s** |
| p50 latency | 13.69 s | 12.67 s | 10.34 s | 14.38 s | **4.72 s** |
| p95 latency | 16.83 s | **13.76 s** | 14.23 s | 18.01 s | **5.38 s** |
| Generation speed | 9.11 | **9.54** | 9.41 | 9.36 | **18.44** |

The harness verified the same 96 targeted cases for all five models.

### Arch-Agent results by targeted tool

| Expected tool | Cases | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed |
|---|---:|---:|---:|---:|---:|---:|
| `weather` | 16 | 100.0% | 93.8% | 100.0% | 100.0% | 0.0% |
| `see` | 16 | 100.0% | 87.5% | 87.5% | 100.0% | 0.0% |
| `read` | 24 | 95.8% | 95.8% | 95.8% | 100.0% | 0.0% |
| `yuki_read` | 8 | 100.0% | 87.5% | 87.5% | 100.0% | 0.0% |
| `search` | 16 | 100.0% | 75.0% | 75.0% | 100.0% | 0.0% |
| `ask_claude` | 16 | 93.8% | 50.0% | 68.8% | 93.8% | 6.3% |

## Failure analysis

Seven broad cases missed strict exactness:

- `fetch`: one prompt caused a conversational refusal instead of a tool call.
- `calc`: removed harmless spaces from `17 * 23`; this passed execution-equivalent scoring.
- `yuki_write` and `yuki_append`: copied surrounding request wording into exact file payloads.
- `yuki_read`: routed `poem.md` through normal `read` as `yuki/poem.md`.
- `recall`: compressed `my sister's wedding` to `sister wedding`, losing exact phrasing.
- `ask_claude`: capitalized the first word only; this passed execution-equivalent scoring.

The targeted suite clarifies the model's behavior:

- It selected 94 of 96 tools correctly.
- The only targeted ontology error in the filesystem family routed one normal `read` case to `shell`; all eight `yuki_read` selections were correct.
- Weather was perfect under execution-equivalent scoring.
- Search selected the correct tool all 16 times, but rewrote four queries.
- `ask_claude` remains the largest text-preservation weakness: eight strict argument misses, three of which were execution-equivalent, plus one conversational malformed output.
- No multi-call output occurred despite the model's multi-step agent training.

## Recommendation

Arch-Agent 3B should remain a finalist. It currently offers the best hard-case balance among the 3B models: xLAM-2 3B is one strict call better but slower, while Hammer 3B is slightly faster and has zero malformed output but trails on the targeted suite.

Hammer 1.5B remains far ahead on speed and memory. The remaining Granite 4.1 3B test should be completed before selecting a dispatcher. The Katanemo/DigitalOcean license restriction should also be considered separately from technical quality if Yuki may ever be distributed or used commercially.

## Artifacts

- `arch-agent-3b-broad-72.json` — full broad report
- `arch-agent-3b-broad-72-failures.jsonl` — broad failures
- `arch-agent-3b-targeted-96.json` — full targeted report
- `arch-agent-3b-targeted-96-failures.jsonl` — targeted failures
- `arch-agent-3b-broad-seven-model-comparison.json` — verified seven-model broad comparison
- `arch-agent-3b-targeted-five-model-comparison.json` — verified targeted comparison

