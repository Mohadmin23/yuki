# Hammer Oracle Schema-Subset Experiment

## Question

Is Hammer 1.5B intrinsically too weak for the targeted Yuki dispatch cases, or is it mainly
overloaded by seeing all 18 schemas at once?

## Controls

- Same Hammer FP16 checkpoints and MLX backend.
- Same 96 targeted cases, model-native Hammer format, prompt style, 256-token ceiling, JSON
  completion stopping, validation, and scoring.
- One stateless generation per case; no tool execution and no result feedback.
- The only experimental variable was the schema list selected from the expected benchmark label.
- Models were run separately, one model at a time.
- No automatic domain router and no Phase 2 cases were introduced.

The logical-domain oracle used the existing information, computer, media, Yuki-files, memory,
and external-agent groups. The more aggressive targeted-confusions oracle used:

- `weather`, `fetch`, `search`
- `see`, `image`
- `read`, `yuki_read`
- `search`, `fetch`
- `hardware`, `read`, `shell`
- `ask_claude` alone

The existing eight broad `remember`/`recall` cases were run separately with only those two
schemas.

## Targeted 96-case results

| Model and exposure | Tool selection | Strict arguments | Equivalent arguments | Exact call | Equivalent call | Schema valid | Malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hammer 1.5B — all 18 | 75.0% | 62.5% | 66.7% | 62.5% | 66.7% | 80.2% | 0% |
| Hammer 1.5B — logical domains | 85.4% | 71.9% | 72.9% | 71.9% | 72.9% | 86.5% | 0% |
| Hammer 1.5B — focused confusions | 89.6% | 76.0% | 77.1% | 76.0% | 77.1% | 89.6% | 0% |
| Hammer 3B — all 18 | 87.5% | 80.2% | 81.3% | 80.2% | 81.3% | 87.5% | 0% |
| Hammer 3B — logical domains | 95.8% | 85.4% | 86.5% | 85.4% | 86.5% | 95.8% | 0% |
| Hammer 3B — focused confusions | 96.9% | 87.5% | 88.5% | 87.5% | 88.5% | 96.9% | 0% |

## Performance

| Model and exposure | Average latency | p50 | p95 | Generated tokens/sec | Peak MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hammer 1.5B — all 18 | 4.64 s | 4.72 s | 5.38 s | 18.44 | 3.62 GB |
| Hammer 1.5B — logical domains | 2.12 s | 2.24 s | 2.65 s | 19.44 | 3.45 GB |
| Hammer 1.5B — focused confusions | 2.04 s | 2.08 s | 2.47 s | 19.46 | 3.40 GB |
| Hammer 3B — all 18 | 10.95 s | 10.34 s | 14.23 s | 9.41 | 6.65 GB |
| Hammer 3B — logical domains | 4.42 s | 4.45 s | 5.39 s | 9.76 | 6.52 GB |
| Hammer 3B — focused confusions | 4.59 s | 4.55 s | 5.65 s | 9.51 | 6.46 GB |

Schema reduction cuts average latency by about 56% for 1.5B and 58–60% for 3B. Generated-token
throughput changes little; the gain comes primarily from processing a much shorter schema prompt.

## Focused family exact-call accuracy

| Family | 1.5B all 18 | 1.5B focused | 3B all 18 | 3B focused |
| --- | ---: | ---: | ---: | ---: |
| Weather vs search/fetch | 75.0% | 81.3% | 87.5% | 93.8% |
| See vs image | 18.8% | 43.8% | 75.0% | 81.3% |
| Read vs Yuki read | 62.5% | **100%** | 87.5% | 93.8% |
| Search vs fetch | 75.0% | 81.3% | 62.5% | 93.8% |
| Normal filepath preservation | 75.0% | 87.5% | 100% | 100% |
| Ask-Claude text preservation | 68.8% | 62.5% | 68.8% | 62.5% |

The `remember`/`recall` pair scored 8/8 exact for both models. Hammer 1.5B averaged 2.35 seconds
and Hammer 3B averaged 5.33 seconds on those eight cases.

## Conclusion

Hammer 1.5B is **partly overloaded and partly capacity-limited**.

Evidence for schema competition:

- Tool-selection accuracy rises from 75.0% to 89.6% with focused subsets.
- Exact-call accuracy rises 13.5 points, from 62.5% to 76.0%.
- `read` versus `yuki_read` rises from 62.5% to 100%, showing that this family is primarily an
  ontology/schema-competition problem for 1.5B.
- Latency falls from 4.64 seconds to about 2.04 seconds.

Evidence for meaningful 3B capacity:

- 1.5B does not approach the proposed 85–90% targeted exact-call range even with oracle-focused
  subsets.
- 3B retains an 11.5-point exact-call lead under identical focused exposure: 87.5% versus 76.0%.
- Camera routing remains 43.8% for 1.5B with only `see` and `image`, while 3B reaches 81.3%.
- Singleton `ask_claude` exposure does not improve text preservation, so those failures cannot be
  blamed on competing schemas.
- The remaining 1.5B failures are increasingly argument-preservation and semantic interpretation
  failures rather than selection among unrelated tools.

Therefore, schema subsets make Hammer 1.5B substantially faster and selectively much more
reliable, but they do not recover most of the 3B model's targeted accuracy. A future automatic
domain-selection experiment is justified for latency and for specific ontologies such as normal
files versus Yuki files, but it should not be expected to erase the overall capacity gap.

## Machine-readable reports

- `hammer21-targeted-oracle-comparison.json` — all six 96-case runs, deltas, tag metrics,
  per-tool failures, and confusion matrices.
- `hammer21-remember-vs-recall-comparison.json` — the two memory-pair runs.
- Individual model/profile JSON and failure JSONL files preserve every raw generation.
