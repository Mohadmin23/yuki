# Qwen3.8-27B → Hammer 1.5B production-delegation study

Date: 2026-08-21

## Outcome

The intended `qwen/qwen3.8-27b` main brain is a strong domain selector, but the complete Qwen → Hammer 1.5B pipeline is not yet reliable enough for unattended dispatch across all 18 Yuki tools.

Across the frozen 450-case Phase 2-derived dataset, Qwen produced valid delegation JSON in every case and selected the correct domain in 446/450 cases (99.1%). Hammer 1.5B selected the correct final tool in 408/450 cases (90.7%). The complete pipeline produced 358/450 strict exact calls (79.6%) and 364/450 preservation-aware exact calls (80.9%).

Hammer 3B was intentionally excluded at the user's direction. No Yuki tool was executed.

## Experiment contract

```text
raw request
  → Qwen3.8-27B delegation JSON
  → deterministic domain-to-live-Yuki-schema mapping
  → Hammer 2.1 1.5B native tool call
  → parse, validate, and score
  → stop
```

- Qwen model: `qwen/qwen3.8-27b` through OpenRouter, labeled as the intended main brain rather than a surrogate.
- Dispatcher: `MadeAgents/Hammer2.1-1.5b` FP16 from the external disk.
- Dataset: all 450 frozen derived cases, 25 per Yuki tool.
- Stateless: no conversation history or previous tool result.
- One Qwen generation and one Hammer generation per valid case.
- Qwen reasoning disabled; all recorded reasoning-token counts were zero.
- Qwen used strict JSON-schema output.
- Hammer used its native fenced tool-call array format.
- Tool execution capability was absent from both benchmark stages.

## Stage A: Qwen3.8-27B delegation

| Metric | Result |
|---|---:|
| Valid delegation JSON | 100.0% |
| Malformed delegation | 0.0% |
| Correct domain | 99.1% |
| Verbatim recall | 90.6% |
| Verbatim precision | 75.9% |
| Final-tool leakage | 0.2% (1 case) |
| Average latency | 2.145 s |
| p50 / p95 latency | 1.210 s / 6.976 s |
| Average generated tokens | 28.96 |
| Average generation speed | 21.25 tok/s |

The four domain errors were:

- `search` interpreted as stored-memory retrieval.
- `see` interpreted as information search.
- One `yuki_write` and one `yuki_append` preservation prompt interpreted as memory storage.

The deterministic semantic-quality score was 69.3%, but this is a conservative lexical-cue heuristic rather than a semantic judge. It flags plainly useful delegations such as “Search for the exact phrase…” when they do not contain one of its narrow registered phrases. It should not be treated as a direct model-accuracy measure.

Verbatim preservation remains a real concern. Forty-four cases had less than complete gold-string recall, concentrated in `search`, `image`, `ask_claude`, `recall`, and a few exact shell/Yuki-file payloads. Some missing strings were still present inside Qwen's semantic `request`, so not every verbatim miss caused a bad final call.

## Stage B and end-to-end result

| Metric | Result |
|---|---:|
| Tool selection | 90.7% |
| Strict argument accuracy | 79.6% |
| Execution-equivalent argument accuracy | 79.8% |
| Preservation-aware argument accuracy | 80.9% |
| Strict exact call | 79.6% (358/450) |
| Execution-equivalent exact call | 79.8% (359/450) |
| Preservation-aware exact call | 80.9% (364/450) |
| Schema-valid output | 92.4% |
| Malformed output | 0.4% (2/450) |
| Conversational output | 0.0% |
| Average latency | 2.486 s |
| p50 / p95 latency | 2.512 s / 3.197 s |
| Average generation speed | 19.39 tok/s |
| Peak MLX memory | 3.473 GB |

The 92 strict failures decompose as follows. This taxonomy is priority ordered, so “lost verbatim” means the failure was associated with an upstream miss; it does not prove that miss was the sole cause.

| Failure category | Cases |
|---|---:|
| Main-brain wrong domain | 4 |
| Main-brain lost-verbatim association | 37 |
| Dispatcher wrong tool | 29 |
| Dispatcher wrong argument | 14 |
| Dispatcher argument rewrite | 6 |
| Dispatcher malformed output | 2 |

## Where the pipeline works

Strict exact-call accuracy by family:

| Family | Strict exact call |
|---|---:|
| Yuki files | 90.4% |
| Information | 89.6% |
| Computer | 89.3% |
| Memory | 60.0% |
| External agent | 60.0% |
| Media | 42.0% |

Several individual tools were excellent: `time`, `weather`, and `yuki_list` reached 100%; `calc`, `read`, and `yuki_read` reached 96%. The normal-filesystem versus Yuki-filesystem separation therefore worked well under the production delegation contract.

## Dominant weaknesses

### Camera/vision and image generation

`see` reached only 40% tool selection and 36% strict exact calls. Hammer rejected 13/25 `see` requests, selected `image` once, and selected `search` once. `image` selection was much better at 88%, but strict exact calls were only 48% because image descriptions were often lost or rewritten upstream.

### Memory recall

`remember` selected the correct tool in all 25 cases and reached 96% preservation-aware exact calls. `recall`, however, reached only 52% tool selection and 44% strict exact calls: Hammer rejected 11 requests and selected `remember` once. This is now a focused Hammer/schema/prompt issue rather than broad 18-schema overload, because Hammer saw only the two-tool memory subset.

### Exact payload preservation

`ask_claude` selected the correct tool in 24/25 cases, yet strict exact calls were only 60%. `search` similarly selected correctly in 23/25 but reached 60% strict exact calls. These results point to Qwen's `verbatim` extraction and semantic rewriting as a larger issue than final tool choice for those families.

### Reject behavior

Thirty-four outputs were schema-invalid. Only two were malformed JSON/tool-call syntax; 32 were explicit Hammer rejections in a benchmark where every request required a tool. The largest rejection clusters were `see` (13) and `recall` (11). These are the best focused examples for prompt/schema experiments or fine-tuning.

## Fair comparison with the temporary 14B surrogate

The following comparison uses the identical deterministic 180 case IDs used in the earlier surrogate experiment:

| Metric | Qwen3-14B surrogate + Hammer 1.5B | Qwen3.8-27B + Hammer 1.5B | Change |
|---|---:|---:|---:|
| Domain accuracy | 87.8% | 97.8% | +10.0 pp |
| Tool selection | 83.9% | 90.6% | +6.7 pp |
| Strict exact call | 73.3% | 78.3% | +5.0 pp |
| Preservation-aware exact call | 73.9% | 79.4% | +5.6 pp |
| Schema-valid output | 90.0% | 93.9% | +3.9 pp |

The intended 27B model clearly improves the production interface, especially domain routing. It does not by itself solve exact argument preservation or Hammer's `see`/`recall` rejection behavior.

## Category difficulty

| Dataset category | Strict exact call |
|---|---:|
| Normal | 85.1% |
| Paraphrase | 80.9% |
| Confusion | 77.8% |
| Preservation | 64.6% |
| Adversarial | 55.6% |

The monotonic decline is useful evidence that the benchmark is exposing robustness limitations rather than only random failures.

## Cost and hardware

- Pre-run exact-model verification: `$0.0002668`.
- 18-case smoke: `$0.0053244`.
- Full 450-case Qwen run: `$0.13306525`.
- Total observed OpenRouter cost: `$0.13865645`.
- Approximate balance remaining from the user-reported `$1.40`: `$1.26134`.
- Hammer was local, so it added no API cost.
- Hammer model storage was on `/Volumes/madisk`, not the internal disk.
- Hammer peak MLX memory was 3.473 GB.

## Recommendation

Keep Hammer 1.5B as the active small-model candidate for the next focused iteration, but do not call the current 79.6% strict pipeline production-ready.

The next experiment should stay narrow:

1. Repair Qwen `verbatim` extraction for `search`, `image`, and `ask_claude` without changing domain routing.
2. Diagnose why Hammer rejects `see` and `recall` even with only the correct two-tool family exposed.
3. Re-score those fixed failure families on a frozen targeted set before considering another broad run.

No Hammer 3B run is needed for that diagnosis.

## Artifacts

- `stage-a/production-delegation-mainbrain-qwen3.8-27b-openrouter-full450-v1.json`
- `stage-b/production-delegation-hammer1.5b-qwen3.8-27b-full450-v1.json`
- `stage-b/production-delegation-hammer1.5b-qwen3.8-27b-full450-v1-failures.jsonl`
- `qwen3.8-hammer1.5-full450-comparison-v1.json`

