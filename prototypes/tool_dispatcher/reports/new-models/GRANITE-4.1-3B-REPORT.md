# Granite 4.1 3B — Yuki Tool Dispatcher Report

Date: 2026-08-19

## Verdict

`ibm-granite/granite-4.1-3b` is not a competitive Yuki dispatcher under the current 18-tool contract. It selected the right general tool surprisingly often, but repeatedly ignored exact argument schemas, altered paths and URLs, and misspelled tool names.

It scored 59.4% strict exact calls on the targeted suite and 73.6% on the broad suite while being slower and larger than the leading 3B models. Its permissive Apache 2.0 license is attractive, but the technical result does not justify keeping it in the current dispatcher shortlist.

This remains Phase 1 smoke/diagnostic evidence, not a final reliability study.

## Test configuration

- Model: `ibm-granite/granite-4.1-3b`
- Local FP16 conversion: `/Volumes/madisk/yuki-tool-dispatcher/granite-4.1-3b-fp16`
- Source cache: `/Volumes/madisk/huggingface/hub/models--ibm-granite--granite-4.1-3b`
- Architecture: GraniteForCausalLM
- Source dtype: BF16
- License: Apache 2.0
- Native output: one JSON call inside `<tool_call>` XML tags
- Available schemas: all 18 real Yuki tool schemas
- Generation ceiling: 256 tokens with immediate stop after one complete structured call
- State: stateless; one dispatcher generation per request
- Router: model only; no regex auto-detection or fallback
- Tool execution: disabled
- Tool results returned to model: none
- Peak MLX memory observed: 7.4411 GB

Granite reused the same XML-native parser path as Arch after its official model card and chat template confirmed the identical call envelope. No Yuki core code or scoring rule was changed.

## Native-format verification

The first call produced the expected native output:

```text
<tool_call>
{"name":"weather","arguments":{"city":"Algiers"}}
```

It parsed and validated exactly, stopping after 14 generated tokens. Cold latency was 12.17 seconds at 8.40 generated tokens/second.

## Broad all-18 benchmark — 72 cases

| Metric | Result |
|---|---:|
| Tool selection | 91.7% |
| Strict argument accuracy | 79.2% |
| Execution-equivalent argument accuracy | 80.6% |
| Strict exact-call accuracy | 73.6% |
| Execution-equivalent exact-call accuracy | 75.0% |
| Schema-valid output | 90.3% |
| Malformed output | 0.0% |
| Average latency | 15.71 s |
| p50 latency | 15.81 s |
| p95 latency | 17.08 s |
| Generation speed | 8.61 tok/s |
| Peak MLX memory | 7.4411 GB |

All 72 cases ran exactly once, and no Yuki tool was executed.

## Targeted failure suite — 96 cases

| Metric | Result |
|---|---:|
| Tool selection | 93.8% |
| Strict argument accuracy | 59.4% |
| Execution-equivalent argument accuracy | 60.4% |
| Strict exact-call accuracy | 59.4% |
| Execution-equivalent exact-call accuracy | 60.4% |
| Schema-valid output | 80.2% |
| Malformed output | 1.0% |
| Average latency | 14.70 s |
| p50 latency | 14.90 s |
| p95 latency | 17.55 s |
| Generation speed | 8.53 tok/s |
| Peak MLX memory | 7.4411 GB |

### Targeted results by tool

| Expected tool | Cases | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed |
|---|---:|---:|---:|---:|---:|---:|
| `weather` | 16 | 100.0% | 93.8% | 100.0% | 100.0% | 0.0% |
| `see` | 16 | 100.0% | 87.5% | 87.5% | 100.0% | 0.0% |
| `read` | 24 | 79.2% | 4.2% | 4.2% | 25.0% | 0.0% |
| `yuki_read` | 8 | 100.0% | 87.5% | 87.5% | 100.0% | 0.0% |
| `search` | 16 | 100.0% | 75.0% | 75.0% | 100.0% | 0.0% |
| `ask_claude` | 16 | 93.8% | 50.0% | 50.0% | 93.8% | 6.3% |

## Failure analysis

The dominant failure is normal filesystem reading:

- Five of 24 targeted `read` prompts were routed to `shell` with `cat` commands.
- Eighteen more selected `read` but invented `{ "command": "cat …" }` instead of the offered `{ "filepath": "…" }` schema.
- Only one of 24 targeted normal `read` calls was strictly correct.
- The model also inserted spaces or invisible characters into exact paths.

The broad suite exposed additional systematic problems:

- All four `yuki_list` calls were emitted as the nonexistent tool name `yuki_ list`.
- Three of four `fetch` URLs were corrupted with inserted visible or zero-width spaces.
- Two search queries were rewritten, including insertion of a space inside `function-calling`.
- `see` emitted `target: null` instead of an empty argument object.
- Yuki append payloads lost filenames or copied surrounding request text.
- Recall and `ask_claude` text was rewritten beyond the execution-equivalent rules.

The outputs were generally syntactically structured, which explains the low malformed rate. The failures are semantic and schema-level, making them more serious for actual execution.

## Recommendation

Remove Granite 4.1 3B from the current dispatcher shortlist. It is the largest tested model by both FP16 footprint and peak MLX allocation, among the slowest, and substantially less accurate than Hammer 1.5B despite using roughly twice the memory.

Its Apache 2.0 license could justify future fine-tuning research, particularly around Yuki's exact schemas. That would be a different experiment; the current pretrained checkpoint should not be selected as the dispatcher.

## Artifacts

- `granite-4.1-3b-broad-72.json` — full broad report
- `granite-4.1-3b-broad-72-failures.jsonl` — broad failures
- `granite-4.1-3b-targeted-96.json` — full targeted report
- `granite-4.1-3b-targeted-96-failures.jsonl` — targeted failures
- `final-eight-model-broad-comparison.json` — verified broad comparison
- `final-six-model-targeted-comparison.json` — verified targeted comparison

