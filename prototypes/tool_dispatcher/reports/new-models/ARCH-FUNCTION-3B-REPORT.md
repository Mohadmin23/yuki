# Arch-Function-3B — Yuki Tool Dispatcher Report

Date: 2026-08-19

## Verdict

`katanemo/Arch-Function-3B` is a capable tool selector, but it is not the best current Yuki dispatcher candidate. On the 72-case all-tools smoke suite it placed behind both Hammer models and xLAM-2 1B in strict exact-call accuracy. It was also slower than Hammer 3B and produced malformed or conversational output in about 10% of cases.

Its strongest result is tool selection on the targeted suite: 89.6%, slightly above Hammer 3B's 87.5%. Its main weakness is output discipline and argument-object construction, especially for `ask_claude` and adversarially worded `search` cases.

This is still Phase 1 smoke/diagnostic evidence. It is not enough to draw a final reliability conclusion.

## Test configuration

- Model: `katanemo/Arch-Function-3B`
- Local FP16 conversion: `/Volumes/madisk/yuki-tool-dispatcher/Arch-Function-3B-fp16`
- Source cache: `/Volumes/madisk/huggingface/hub/models--katanemo--Arch-Function-3B`
- Native format: official Arch `<tool_call>` wrapper containing one `{ "name", "arguments" }` JSON object
- Available schemas: all 18 real Yuki tool schemas
- Generation ceiling: 256 tokens, with immediate stop after one complete structured call
- State: stateless; one dispatcher generation per request
- Tool execution: disabled for these routing benchmarks
- Tool results returned to model: none
- Peak MLX memory observed: 6.6521 GB

The prototype preserves the raw Arch generation, removes only the exact native wrapper for parsing, normalizes the call to Yuki's canonical representation, and validates it against the selected live Yuki schema.

## Broad all-18 benchmark — 72 cases

| Metric | Result |
|---|---:|
| Tool selection | 88.9% |
| Strict argument accuracy | 80.6% |
| Execution-equivalent argument accuracy | 81.9% |
| Strict exact-call accuracy | 80.6% |
| Execution-equivalent exact-call accuracy | 81.9% |
| Schema-valid output | 86.1% |
| Malformed output | 9.7% |
| Average latency | 13.58 s |
| p50 latency | 13.52 s |
| p95 latency | 15.93 s |
| Generation speed | 9.50 tok/s |

All 72 cases were invoked exactly once. No Yuki tool was executed, so external service availability did not affect routing scores.

### Same-case broad comparison

The harness verified that all five reports below used the same 72 cases.

| Model | Tool selection | Strict exact call | Equivalent exact call | Schema valid | Malformed | Avg latency | tok/s | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hammer 2.1 3B | 97.2% | **91.7%** | **93.1%** | 98.6% | 0.0% | 11.92 s | 9.64 | 6.651 GB |
| Hammer 2.1 1.5B | 91.7% | 87.5% | 88.9% | 94.4% | 0.0% | **4.69 s** | **18.47** | 3.624 GB |
| xLAM-2 1B | 90.3% | 81.9% | 83.3% | 97.2% | 0.0% | 6.22 s | 18.43 | 3.624 GB |
| **Arch-Function 3B** | 88.9% | 80.6% | 81.9% | 86.1% | 9.7% | 13.58 s | 9.50 | 6.652 GB |
| xLAM 1B | 84.7% | 45.8% | 50.0% | 97.2% | 0.0% | 5.65 s | 18.39 | 3.594 GB |

## Targeted failure suite — 96 cases

| Metric | Arch 3B | Hammer 3B | Hammer 1.5B |
|---|---:|---:|---:|
| Tool selection | **89.6%** | 87.5% | 75.0% |
| Strict argument accuracy | 78.1% | **80.2%** | 62.5% |
| Equivalent argument accuracy | 79.2% | **81.3%** | 66.7% |
| Strict exact-call accuracy | 78.1% | **80.2%** | 62.5% |
| Equivalent exact-call accuracy | 79.2% | **81.3%** | 66.7% |
| Schema-valid output | 81.3% | **87.5%** | 80.2% |
| Malformed output | 10.4% | **0.0%** | **0.0%** |
| Average latency | 13.81 s | 10.95 s | **4.64 s** |
| p50 latency | 14.38 s | 10.34 s | **4.72 s** |
| p95 latency | 18.01 s | 14.23 s | **5.38 s** |
| Generation speed | 9.36 tok/s | 9.41 tok/s | **18.44 tok/s** |

The harness verified the same 96 targeted cases for all three models.

### Arch results by targeted tool

| Expected tool | Cases | Tool selection | Strict exact call | Equivalent exact call | Schema valid | Malformed |
|---|---:|---:|---:|---:|---:|---:|
| `weather` | 16 | 100.0% | 93.8% | 100.0% | 100.0% | 0.0% |
| `see` | 16 | 87.5% | 81.3% | 81.3% | 87.5% | 12.5% |
| `read` | 24 | 95.8% | 91.7% | 91.7% | 91.7% | 4.2% |
| `yuki_read` | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| `search` | 16 | 68.8% | 68.8% | 68.8% | 68.8% | 31.3% |
| `ask_claude` | 16 | 87.5% | 37.5% | 37.5% | 43.8% | 12.5% |

## Failure analysis

The main broad-suite failures were:

- `time`: three of four calls emitted malformed JSON: `"arguments:{}` instead of `"arguments": {}`.
- `ask_claude`: it frequently emitted the question as a bare string instead of the required `{ "question": "..." }` object.
- `yuki_write` / `yuki_append`: it sometimes copied surrounding request wording into the exact payload or emitted a bare argument string.
- `read`: one absolute filesystem path was incorrectly routed to `yuki_read`.
- `calc`: one request was answered directly instead of dispatching, while another only removed harmless whitespace and therefore passed execution-equivalent scoring.
- A few `weather`, `search`, `see`, and file cases produced conversational refusals instead of tool calls. These are genuine model outputs, not parser failures.

The targeted suite clarifies the picture:

- The `read` versus `yuki_read` distinction is not a general Arch weakness; it achieved 31/32 strict exact calls across those targeted cases.
- Weather routing is excellent; the only strict miss was an execution-equivalent text normalization.
- Search failures are strongly wording-sensitive: five of sixteen prompts caused conversational output rather than a dispatch.
- `ask_claude` is primarily an argument-shape problem, not a selection problem. It selected the tool 14/16 times but produced only 6/16 strict valid calls.

## Recommendation

Do not replace Hammer 1.5B or Hammer 3B with Arch-Function 3B at this stage. Arch uses roughly the same memory as Hammer 3B, is slower, and loses output validity. Hammer 1.5B remains the efficiency leader; Hammer 3B remains the broad accuracy leader among the tested models.

Arch's high targeted tool-selection result is useful evidence that its semantic routing capacity is competitive. If revisited later, the highest-value experiments would be grammar-constrained generation and a narrowly corrected schema/prompt treatment for no-argument tools and `ask_claude`. Those changes were deliberately not made here, to preserve the requested same-prompt comparison.

## Artifacts

- `arch-function-3b-broad-72.json` — full 72-case broad report
- `arch-function-3b-broad-72-failures.jsonl` — broad failures
- `arch-function-3b-targeted-96.json` — full targeted report
- `arch-function-3b-targeted-96-failures.jsonl` — targeted failures
- `arch-function-broad-five-model-comparison.json` — verified same-case five-model broad comparison
- `arch-function-targeted-hammer-comparison.json` — verified same-case targeted comparison

