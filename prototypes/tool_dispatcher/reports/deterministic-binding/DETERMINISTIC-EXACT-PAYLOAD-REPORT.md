# Yuki deterministic exact-payload transport — focused report

Date: 2026-08-21

## Outcome

Deterministic source binding substantially removes the exact-payload bottleneck once Hammer has selected the correct tool.

On the 100-case primary set:

- Official strict exact accuracy increased from **58% to 64%**.
- Literal-source exact accuracy increased from **51% to 87%** conservatively, counting ambiguity and abstention as failure.
- On the 98 cases with a unique/no-argument literal reference, literal accuracy increased from **52.0% to 88.8%**.
- Given a correct Hammer tool selection, the binder achieved **97.8%** conservative literal accuracy and **98.9%** on literal-reference-eligible cases.
- Every emitted bound string was copied character-for-character from its recorded raw source span: **110/110, 100%** across primary plus recall.

No Qwen or Hammer inference was run. The experiment reused frozen Qwen3.8-27B delegations and frozen Hammer 1.5B A1 outputs.

## Primary 100-case result

| Metric | Frozen Qwen verbatim → Hammer | Deterministic binder |
|---|---:|---:|
| Hammer tool selection | 89% | 89% |
| Official strict arguments | 58% | **64%** |
| Official preservation-aware arguments | 58% | **64%** |
| Official strict exact calls | 58% | **64%** |
| Official preservation-aware exact calls | 58% | **64%** |
| Literal-source exact, conservative | 51% | **87%** |
| Literal-source exact, 98 eligible cases | 52.0% | **88.8%** |
| Schema valid | **93%** | 90% |
| Final bound call available | — | 90% |
| Annotated ambiguous spans | — | 2 |
| Binder ambiguous outputs | — | 1 |

The binder repaired 18 official exact calls and broke 12, for a net official gain of six cases. Under the frozen literal policy it repaired 36 cases and broke none.

Schema validity decreased because the binder explicitly abstains instead of retaining a generated required argument. The ten primary cases without a final bound call were seven pre-existing Hammer rejections/no-calls, one unsupported wrong-tool selection, one wrong-tool input with no defensible payload for that tool, and one explicit span ambiguity.

## By tool

| Tool | Official exact baseline → binder | Literal exact baseline → binder | Literal exact when tool correct |
|---|---:|---:|---:|
| `search` | 60% → **80%** | 56% → **92%** | **100%** |
| `image` | 56% → 52% | 52% → **100%** | **100%** |
| `ask_claude` | 60% → **96%** | 60% → **96%** | **100%** |
| `see` | 56% → 28% | 36% → **60%** | 88.2% conservative / **93.8% eligible** |

The search and external-agent bottlenecks are largely removed. Image reaches 100% literal accuracy, even though official accuracy falls slightly because the literal policy preserves articles that official gold removes.

`see` remains dominated by Hammer selection/rejection: only 17/25 cases had the correct selected tool. Of those, the binder achieved 15 literal successes, explicitly abstained on the repeated `the window` span, and made one policy-conformance miss by including `with the camera` in the `p2-see-18` target. Predictions were already frozen before scoring, so that miss was not tuned away on this test set.

## Official gold versus literal-source policy

The official gold records were not changed.

The primary set contains 30 unambiguous official/literal conflicts, mostly because official gold removes articles or possessives:

- `the latest MLX documentation` versus official `latest MLX documentation`;
- `a glass city floating above clouds` versus official `glass city floating above clouds`;
- `my keys` versus official `keys`.

Two additional `see` cases are explicitly ambiguous rather than assigned a canonical literal value:

- `p2-see-10`: `the window` occurs at two equally plausible offsets. The binder abstained.
- `p2-see-21`: `the notes file` versus `the notes file lying on my desk`. Hammer had already rejected the request, so the binder received no selected call.

This explains why literal preservation can improve sharply while some official scores regress. The official metric remains useful for compatibility with prior reports; the literal metric measures the new transport contract.

## Recall diagnostic

| Metric | Frozen baseline | Deterministic binder |
|---|---:|---:|
| Tool selection | 96% | 96% |
| Official strict exact | **76%** | 36% |
| Official preservation-aware exact | **80%** | 44% |
| Literal-source exact | 48% | **96%** |
| Literal-source exact when tool correct | 50% | **100%** |
| Schema valid | **100%** | 96% |

Recall has 15 official/literal conflicts. Two official values are not contiguous raw spans at all:

- `Metal out of RAM` is synthesized from raw `Metal ran out of RAM`.
- `notes.md discussion` reorders raw `discussion of notes.md`.

The binder correctly refuses to synthesize those forms. Its only literal failure is `p2-recall-21`, where Hammer selected `remember`; a post-selection binder cannot repair that tool decision.

## Does deterministic binding remove the bottleneck?

**Yes, for correctly routed calls.** Search, image, ask-Claude, and recall achieved 100% literal accuracy conditional on correct tool selection. The combined primary result was 98.9% on correct-tool, literal-reference-eligible cases.

The remaining limitations are no longer primarily string regeneration:

1. Hammer rejection or wrong-tool selection prevents binding.
2. Some raw requests have genuinely ambiguous source boundaries.
3. The old official gold intentionally normalizes spans and therefore measures a different contract.
4. One frozen `see` result exposed a general camera-scaffolding suffix the v1 binder failed to exclude.

The result supports replacing Qwen/Hammer exact-string reproduction with deterministic raw-source binding. It does not yet justify production integration: broader tool grammars, independent evaluation data, and a defined fallback policy for abstention are still required.

## Isolation and safety

- Primary cases: exactly 100 frozen cases (`search`, `image`, `ask_claude`, `see`).
- Recall diagnostic: exactly 25 frozen cases.
- Models loaded or called: 0.
- Qwen prompt changes: 0.
- Delegations or Hammer generations regenerated: 0.
- Full-450 reruns: 0.
- Yuki tool execution capability or attempts: 0.
- Frozen gold mutations: 0.
- Binder access to literal annotations or gold: 0.
- Binder predictions were checksummed before the scoring process opened the annotation file.
