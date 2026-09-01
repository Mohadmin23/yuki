# Hammer `see` capacity probe: 1.5B versus 3B

Date: 2026-08-21

## Outcome

Hammer 3B materially fixes `see` selection confidence and rejection behavior, but it does **not** improve strict end-to-end calls on these 25 frozen cases.

The result is therefore mixed but decisive:

- There is a real capacity gap in recognizing and committing to the repaired `see` capability.
- There is no measured 3B capacity advantage in exact `target` extraction.
- The remaining strict failures are primarily associated with the frozen delegation/optional-argument interface, not unresolved `see` versus `image` ontology.

## Controlled result

| Metric | Hammer 1.5B A1 | Hammer 3B A1 | Change |
|---|---:|---:|---:|
| Tool selection | 68% | **96%** | **+28 pp** |
| Strict argument accuracy | 56% | 56% | 0 pp |
| Strict exact call | 56% | 56% | 0 pp |
| Preservation-aware exact call | 56% | 56% | 0 pp |
| Schema valid | 76% | **100%** | **+24 pp** |
| Rejections | 6/25 | **0/25** | −6 |
| Malformed | 0/25 | 0/25 | 0 |
| Average latency | **1.755 s** | 3.726 s | 2.12× |
| p50 latency | **1.960 s** | 3.818 s | 1.95× |
| p95 latency | **2.250 s** | 4.542 s | 2.02× |
| Generated tokens/s | **21.78** | 10.03 | 0.46× |
| Peak MLX memory | **3.459 GB** | 6.526 GB | 1.89× |

Both arms used the exact same 25 case IDs, frozen Qwen3.8-27B delegation objects, A1 media schemas, default task instruction, Hammer-native format, parser, validator, scorer, generation ceiling, stateless one-generation policy, and validation-only execution lock.

## Case-level flips

Strict failure to success:

- `p2-see-04`: rejected → `see(target="door")`
- `p2-see-06`: rejected → `see(target="whiteboard")`

Strict success to failure:

- `p2-see-18`: `see(target="printed image")` → `see()`
- `p2-see-20`: `see(target="terminal window")` → `see()`

Net strict change: **zero**. Twelve cases succeeded under both models and nine failed under both.

At the tool-selection level, 3B fixed seven cases—`04`, `06`, `07`, `09`, `15`, `19`, and `21`—with no reverse selection flips. Six of those were the remaining 1.5B rejections; the seventh changed an incorrect `image` call to `see`.

## Why strict accuracy did not move

Hammer 3B's eleven official strict failures decompose into:

- 10 correct-`see` calls with incorrect `target` arguments.
- 1 frozen Qwen wrong-domain case (`p2-see-23`), where the dispatcher was offered information tools and selected `search`.

Among the ten argument failures:

- Six emitted an empty optional argument object.
- Four emitted a nonempty but nonexact target.
- Seven had an empty frozen Qwen `verbatim` array.
- In all ten, the exact gold target was absent from frozen Qwen `verbatim`.

Three nonexact outputs—`my keys`, `the window`, and `my wallet`—preserved Qwen's supplied `verbatim` exactly even though the frozen gold expected `keys`, `window`, and `wallet`. Official scoring was not changed.

This does not prove Qwen is the sole cause, but it shows why added Hammer capacity cannot reliably recover the benchmark's exact span: the schema permits omitting `target`, while the exact gold target is not transported in the frozen delegation for any of the ten argument failures.

## Answer to the capacity question

Hammer 3B **does solve the remaining rejection/tool-selection problem**, so that part was capacity-sensitive after schema repair.

Hammer 3B **does not materially solve the remaining overall `see` failures**: strict and preservation-aware exact accuracy remain 56%, while latency and memory roughly double. The remaining end-to-end issue is more likely the interface between Qwen `verbatim`, optional `see.target`, and exact-span scoring than insufficient Hammer capacity.

On this evidence, Hammer 3B is not justified as a replacement for 1.5B solely for `see`. The next useful change would be an independently evaluated payload/argument-interface repair, not another model-size benchmark.

## Safety record

- Qwen calls: 0.
- Delegations regenerated: 0.
- Full-450 cases run: 0.
- Hammer generations: exactly 25.
- Yuki tool execution capability or attempts: 0.
- Production Yuki changes: 0.
- Hammer 3B checkpoint: external disk.
- Hammer 3B process after run: unloaded.
