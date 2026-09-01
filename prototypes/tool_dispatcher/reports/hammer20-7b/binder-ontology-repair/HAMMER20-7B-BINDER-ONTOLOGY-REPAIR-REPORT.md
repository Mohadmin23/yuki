# Hammer 2.0 7B — Frozen Binder + Ontology Repair

## Outcome

The original Hammer 2.0 7B run remains frozen byte-for-byte at **368/450 strict exact calls (81.8%)** and **412/450 correct tool selections (91.6%)**. No baseline case was regenerated.

The deterministic binder successfully repairs literal-copy failures, but it does **not** increase the official full-450 strict score when applied uniformly. It repairs 24 official calls and breaks 31 calls whose official gold intentionally removes or rewrites literal source wording. Uniform strict exact therefore changes from **368/450 (81.8%)** to **361/450 (80.2%)**. This is a scoring/interface distinction, not a character-copy failure: literal-source exact on the 123 unambiguous annotated cases rises from **64/123 (52.0%)** to **117/123 (95.1%)**, and every emitted bound value satisfies the character-for-character source-copy invariant.

The two minimal schema boundaries produced a strong targeted improvement with no strict regressions, but they fixed only 6 of the 14 dominant ontology confusions. The predeclared full-run gate required at least 7, so the 450-case repaired rerun was not started.

## Safety and freeze guarantees

- Frozen source: `hammer20-7b-8bit-all18-native-cached-full450-v1.json`
- Frozen SHA-256: `97980e3a53d7e105046dda2a4bd6c982682c047c5dace89a55f5d77fd891cb32`
- No Qwen call, Hammer 3B load, Yuki execution, production-Yuki edit, gold mutation, or baseline resampling occurred.
- Binder input contained raw requests and frozen Hammer calls, but no expected tools, expected arguments, exact-call labels, or literal annotations.
- Binder predictions and binder source were checksummed before scoring.
- The targeted Hammer run used the same 8-bit checkpoint, native Hammer format, prompt instructions, parser, validator, scoring, all-18 exposure, and one stateless generation per case.

## Experiment 1 — deterministic source binder

### The 44 correct-tool/wrong-argument cases

The binder covered 39 of the 44 cases. The five uncovered cases were `remember.fact` cases requiring semantic perspective rewriting (`my`/`I` to `user`), not lossless source transport.

Within these 44 cases:

| Measure | Baseline | Binder |
|---|---:|---:|
| Official strict exact | 0/44 | 24/44 (54.5%) |
| Literal-source exact, annotated cases | 7/31 (22.6%) | 31/31 (100.0%) |
| Official calls broken | — | 0 |
| Bound source-copy invariant | — | 100.0% |

Official repairs by tool were:

- `search`: 5
- `image`: 3
- `ask_claude`: 8
- `yuki_write`: 4
- `yuki_append`: 4

`see` and `recall` gain literal-source correctness but not official strict correctness because their official gold generally removes articles, possessives, locative wording, or contextual nouns. That is exactly the conflict the earlier source-span policy warned about.

### Uniform application to all 450

| Measure | Frozen Hammer | Binder pipeline |
|---|---:|---:|
| Upstream Hammer tool selection | 412/450 | unchanged (frozen) |
| Final-call tool correctness after binder abstention | 412/450 | 410/450 |
| Official strict exact | 368/450 (81.8%) | 361/450 (80.2%) |
| Official repairs | — | 24 |
| Official breaks | — | 31 |
| Literal-source exact, eligible annotations | 64/123 (52.0%) | 117/123 (95.1%) |
| Binder abstentions/no-call outcomes | — | 29 |

The 31 official breaks are dominated by benchmark gold that expects semantic normalization while the source policy requires literal transport. Examples include:

- `image.prompt`: source “a watercolor lighthouse…” versus gold “watercolor lighthouse…”
- `see.target`: source “my keys” versus gold “keys”
- `recall.topic`: source “the garden project” versus gold “garden project”

An oracle that applied the binder only to the known 44 failures would reach **392/450 (87.1%)**, but that is not deployable because production code does not know the gold failure label. It is reported only as an upper bound.

### Binder conclusion

Deterministic binding works for genuinely exact-sensitive fields—queries, external-agent questions, quoted prompts, paths, case-sensitive filenames, and file content. It should not be indiscriminately applied to semantically normalized fields. A production contract needs an explicit per-field mode such as:

```text
literal_source     → copy/assemble raw spans only
semantic_argument → retain normalized semantic extraction
```

`remember.fact`, most `recall.topic`, and many generic `see.target`/`image.prompt` cases belong to the second category under the current official benchmark. Exact labels or explicit user wording can opt those fields into literal mode.

## Experiment 2 — schema ontology repair

Only four prototype-facing tool descriptions changed:

- `read`: normal operating-system filesystem paths only; not Yuki's internal store.
- `yuki_read`: Yuki's internal managed file store only; not arbitrary OS paths.
- `yuki_write`: create or replace the complete content; do not preserve existing content.
- `yuki_append`: preserve existing content and add after it; do not replace/create fresh content.

No parameter, schema shape, prompt instruction, native format, model weight, or production tool metadata changed.

### Frozen targeted set

The 49 cases were frozen before inference and included:

- all 8 frozen `yuki_read → read` confusions;
- all 6 frozen `yuki_write → yuki_append` confusions;
- 6 normal-read controls;
- 6 write-preservation controls;
- 6 append-preservation controls;
- all 20 prior explicit rejections (with three overlaps).

### Results

| Measure | Frozen baseline | Repaired schemas |
|---|---:|---:|
| Tool selection | 15/49 (30.6%) | 28/49 (57.1%) |
| Strict exact | 9/49 (18.4%) | 22/49 (44.9%) |
| Schema valid | 29/49 (59.2%) | 36/49 (73.5%) |
| Explicit rejections | 20 | 13 |
| Malformed | 0 | 0 |
| New strict regressions | — | 0 |

Runtime for the repaired arm was 6.47 s average, 6.69 s p50, 8.98 s p95, 7.76 generated tokens/s, and 8.91 GB peak MLX memory.

Confusion-specific results:

| Confusion | Baseline | Repaired | Fixed |
|---|---:|---:|---:|
| `yuki_read → read` | 8 | 5 | 3 |
| `yuki_write → yuki_append` | 6 | 3 | 3 |
| Combined | 14 | 8 | 6 |

The remaining cases show that wording alone is not a complete solution. Hammer still maps internal-store language to invented OS paths in five cases, and it still interprets three create/write requests as append operations.

### Why no full 450 rerun happened

Before inference, the frozen gate required:

1. higher targeted tool selection;
2. higher targeted strict exact;
3. at least 7 fewer primary confusions;
4. no increase in rejections;
5. at most one new strict regression.

Four checks passed. The primary confusion reduction was 6 rather than 7. Because the gate was frozen before seeing results, it was not relaxed after the fact. The full 450 rerun was therefore blocked, preserving the user's “one rerun only if clean” condition.

## Bottom line

1. **The frozen 81.8% result remains the official Hammer 2.0 7B baseline.**
2. **Lossless source binding is highly effective at character preservation**, but official strict accuracy falls under uniform application because the current benchmark mixes literal and semantic argument contracts.
3. **The schema boundary repair is directionally strong and non-regressive**, fixing 6/14 dominant confusions and seven prior rejections, but it does not yet clear the predeclared full-run gate.
4. **No additional 450-case inference was spent.** The next decision is whether to accept the targeted evidence and explicitly authorize the one full rerun, or first define a stronger interface signal that distinguishes literal-source arguments from semantic arguments.

