# Hammer 1.5B vs Hammer 3B with deterministic source binding

## Result

Hammer 3B shows a real capacity advantage for selecting `see`, but it does **not**
justify replacing Hammer 1.5B globally on this focused evidence.

Across the frozen 100 cases, 3B improved tool selection by 6 percentage points and
literal-source exactness by 5 points. Official strict exactness decreased by 1 point.
The 3B arm required 1.97x the average latency and 1.88x the peak MLX memory.

The practical interpretation is therefore split:

- For `see`, 3B materially improves routing after A1 schema repair.
- For `search`, `image`, and `ask_claude` as a group, 3B provides no net
  end-to-end gain after deterministic binding.
- The overall paired gains are not conclusive on this 100-case focused set
  (tool-selection exact paired p=0.109; literal-source exact p=0.180).

## Controlled setup

- Frozen Qwen3.8-27B delegations; Qwen was not called.
- Exactly 100 frozen cases: 25 each for `search`, `image`, `ask_claude`, and `see`.
- A1 repaired schemas, native Hammer output, identical prompt, parser, validator,
  scorer, and deterministic source binder.
- One stateless generation per case.
- Hammer 1.5B and Hammer 3B were loaded in separate processes and unloaded after
  their respective runs.
- No Yuki tool execution capability was present; execution attempts were zero.
- Official gold was not modified. Literal annotations were hidden from the binder
  and used only after predictions were frozen.
- The Hammer 1.5B binding outputs exactly replayed the prior frozen primary-100
  binding outputs.

## Overall comparison

| Metric | Hammer 1.5B | Hammer 3B | 3B change |
|---|---:|---:|---:|
| Tool selection | 89% | 95% | +6 pp |
| Official strict exact | 64% | 63% | -1 pp |
| Official preservation-aware exact | 64% | 63% | -1 pp |
| Literal-source exact, conservative | 87% | 92% | +5 pp |
| Literal-source exact, eligible only | 87/98 (88.8%) | 92/98 (93.9%) | +5 cases |
| Schema valid after binding | 90% | 95% | +5 pp |
| Binder coverage | 90% | 95% | +5 pp |
| Binder abstentions | 9 | 3 | -6 |
| Binder ambiguities | 1 | 2 | +1 |
| Hammer rejections | 7 | 3 | -4 |
| Malformed Hammer generations | 0 | 0 | — |
| Average latency | 2.229 s | 4.398 s | 1.97x |
| p50 latency | 2.294 s | 4.486 s | 1.96x |
| p95 latency | 2.741 s | 5.365 s | 1.96x |
| Generation speed | 19.98 tok/s | 9.90 tok/s | 0.50x |
| Peak MLX memory | 3.470 GB | 6.537 GB | 1.88x |

Every emitted bound string passed the raw-source slice invariant in both arms.

## Per-tool comparison

| Tool | Model | Tool selection | Official strict exact | Preservation-aware exact | Literal-source exact | Schema valid | Binder coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| search | 1.5B | 92% | 80% | 80% | 92% | 96% | 96% |
| search | 3B | 92% | 80% | 80% | 92% | 96% | 96% |
| image | 1.5B | 100% | 52% | 52% | 100% | 100% | 100% |
| image | 3B | 100% | 52% | 52% | 100% | 100% | 100% |
| ask_claude | 1.5B | 96% | 96% | 96% | 96% | 96% | 96% |
| ask_claude | 3B | 92% | 92% | 92% | 92% | 92% | 92% |
| see | 1.5B | 68% | 28% | 28% | 60% conservative; 15/23 eligible (65.2%) | 68% | 68% |
| see | 3B | 96% | 28% | 28% | 84% conservative; 21/23 eligible (91.3%) | 92% | 92% |

The decisive capacity signal is `see`: 3B gained 28 points in selection and 24
points in conservative literal-source exactness. Its official `see` exact score did
not rise because deterministic source preservation often retains articles or
locative wording that the unchanged official gold strips.

## Paired case analysis

Tool-selection flips from 1.5B failure to 3B success occurred in eight cases:

- `p2-search-18`
- `p2-see-04`, `p2-see-06`, `p2-see-07`, `p2-see-09`, `p2-see-15`,
  `p2-see-19`, `p2-see-21`

The reverse occurred in two cases:

- `p2-search-16`
- `p2-ask_claude-25`

Seven of the positive selection flips became literal-source exact final calls:
`p2-search-18` plus six `see` cases. `p2-see-21` correctly selected `see` under 3B,
but the binder abstained because two source spans were legitimately ambiguous.

The two negative flips both lost previously exact final calls, so the net changes
were:

- Tool selection: 8 repaired, 2 broken, net +6.
- Literal-source exact: 7 repaired, 2 broken, net +5.
- Official strict exact: 1 repaired, 2 broken, net -1.

## Binder/interface limits

There were 25 cases where both models selected the correct tool, produced the same
bound call, and still failed unchanged official strict gold. In 23 of those cases,
the bound call was literal-source exact. These are official-gold versus literal-span
boundary differences, not Hammer capacity failures.

Only two shared failures remained attributable to the binder/annotation interface:

- `p2-see-10`: the request repeats “the window”; both source choices are valid, so
  the binder correctly abstained as ambiguous.
- `p2-see-18`: binder v1 included “with the camera” in the target span, so neither
  model could repair it.

For 3B alone, `p2-see-21` exposed a second annotation ambiguity after 3B fixed the
tool selection. That is a safe abstention, not malformed model output.

## Answer to the main question

After deterministic binding removes string-copy failures, Hammer 3B does materially
outperform Hammer 1.5B on the remaining `see` selection problem. It does not provide
a compelling overall replacement: official exactness is 1 point lower, literal-source
exactness is only 5 points higher, and the local inference cost is nearly doubled.

Continue with Hammer 1.5B as the default experimental dispatcher. Treat Hammer 3B as
a justified `see`-focused capacity option or a reference arm, not as the new global
default based on this experiment alone.
