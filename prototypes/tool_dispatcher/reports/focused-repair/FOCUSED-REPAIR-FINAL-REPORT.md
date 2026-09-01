# Yuki dispatcher focused repair — final report

Date: 2026-08-21

## Outcome

The focused repair produced one clear success and one clear negative result:

- Minimal Hammer-facing schema descriptions fixed `recall` and improved—but did not fully fix—`see`.
- The attempted lossless Qwen `verbatim` prompt did not improve the frozen official metric or downstream exact-call accuracy. It is rejected.

No 450-case benchmark was rerun. Hammer 3B was never loaded. Production Yuki was not modified. No Yuki tool was executed.

## Experiment A: Hammer `see` and `recall`

| Tool / metric | Frozen baseline | Schema repair | Schema + rejection clarification |
|---|---:|---:|---:|
| `see` tool selection | 40% | **68%** | 68% |
| `see` strict exact | 36% | **56%** | 56% |
| `see` schema valid | 48% | **76%** | 72% |
| `see` rejections | 13 | **6** | 7 |
| `recall` tool selection | 52% | **96%** | 96% |
| `recall` strict exact | 44% | **76%** | 76% |
| `recall` preservation-aware exact | 44% | **80%** | 80% |
| `recall` schema valid | 56% | **100%** | 100% |
| `recall` rejections | 11 | **0** | 0 |

The selected variant is schema descriptions only. The rejection-instruction clarification had no independent benefit and slightly degraded `see`.

| Variant / tool | Avg latency | p50 | p95 | Generated tok/s | Peak MLX |
|---|---:|---:|---:|---:|---:|
| A0 `see` | 1.527 s | 1.167 s | 2.139 s | 23.41 | 3.461 GB |
| A0 `recall` | 2.341 s | 2.541 s | 3.129 s | 20.80 | 3.449 GB |
| A1 `see` | 1.755 s | 1.960 s | 2.250 s | 21.78 | 3.459 GB |
| A1 `recall` | 2.050 s | 2.029 s | 2.241 s | 19.56 | 3.351 GB |
| A2 `see` | 1.713 s | 1.965 s | 2.193 s | 22.09 | 3.473 GB |
| A2 `recall` | 2.059 s | 2.034 s | 2.249 s | 19.52 | 3.345 GB |

The selected descriptions are capability-only and remain prototype-local. `see` explicitly inspects an existing visual source and is not generation; `image` creates a new image and is not inspection. `recall` retrieves persistent memory and does not rely on dispatcher conversation context; `remember` stores a new persistent fact and does not retrieve one. Argument names and schema structure are unchanged.

### 1. Why was Hammer rejecting `see`?

All 13 baseline failures were intentional native empty-array rejections. They were not malformed and were not explained by missing arguments: `target` is optional, and failures occurred both with and without exact target strings.

The old description mixed persona and UI details—Yuki's “real eye,” a webcam window, and “when the user asks you”—without stating the central capability boundary. Hammer was being asked to infer that a listed function grants access to existing visual input. The capability-only repair explicitly separated inspecting an existing camera/view/screen/image source from generating a new image. That removed 7 of 13 rejections.

Six rejections remain, so `see` is still partly a Hammer semantic-confidence/capacity problem. The repair did not reach the encouraging 75–80% strict target.

### 2. Why was Hammer rejecting `recall`?

The old schema required a stateless dispatcher to determine whether information was absent from “the current chat,” then told it how to use returned summaries in an assistant reply. Neither concept belongs in a function router's capability description.

The repaired schema simply says to retrieve information previously stored in Yuki's persistent memory about a topic and contrasts that with storing a new fact. This eliminated all 11 rejections. The resulting 76% strict and 80% preservation-aware exact accuracy reached the encouraging range.

### 3. Did schema-description repair work?

Yes. It was decisive for `recall` and materially helpful for `see`.

### 4. Did rejection clarification help independently?

No. It changed neither target's accuracy, increased `see` rejections from 6 to 7, and reduced `see` schema validity from 76% to 72%. It is not selected.

### 5. Did the fixes degrade `image` or `remember`?

No.

| Tool / metric | Baseline | Selected repair |
|---|---:|---:|
| `image` tool selection | 88% | **100%** |
| `image` strict exact | 48% | **56%** |
| `image` schema valid | 88% | **100%** |
| `remember` tool selection | 100% | 100% |
| `remember` strict exact | 76% | 76% |
| `remember` preservation-aware exact | 96% | 96% |

No A1 target or regression case was malformed.

## Experiment B: Qwen exact-payload transport

The exact old 75-case subset baseline was 98.7% domain accuracy and 61.3% exact character preservation.

| Metric | Old frozen contract | Lossless-prompt candidate | Change |
|---|---:|---:|---:|
| Valid JSON | 100% | 100% | 0 pp |
| Domain accuracy | 98.7% | 97.3% | −1.3 pp |
| Verbatim recall | 61.3% | 53.3% | −8.0 pp |
| Verbatim precision | 61.3% | 52.7% | −8.7 pp |
| Exact character preservation | 61.3% | 53.3% | −8.0 pp |
| Final-tool leakage | 0% | 0% | 0 pp |

The new 75-case API run cost `$0.0219556`, used the exact `qwen/qwen3.8-27b` model, remained schema-constrained, and recorded zero reasoning tokens.

### 6. What fraction of target failures originated in Qwen loss?

Under the selected Hammer schema repair and old frozen Qwen delegations:

- Search: 8/10 strict failures (80%) were associated with incomplete Qwen `verbatim`.
- Image: 10/11 (90.9%).
- Ask Claude: 9/10 (90%).
- Combined: 27/31 (87.1%).

This is an association, not proof of sole causation: the priority-ordered taxonomy labels upstream loss before checking every possible downstream error. Still, the concentration is strong enough to identify payload transport as the dominant B-family limitation.

### 7. Did the revised Qwen contract improve exact preservation?

No under the unchanged frozen scoring. It reduced official exact preservation by eight percentage points.

The failure pattern changed rather than disappearing. Qwen often inserted a leading `a/an`, retained sentence-ending punctuation, omitted ordinary search queries, or still shortened complex payloads.

There is an important benchmark-boundary issue: 13 official new failures are exact substrings of the raw user request and differ from gold only because gold removed a leading article in nine image prompts or a final period in four external-agent messages. For example, source text contains “a watercolor lighthouse,” while gold expects “watercolor lighthouse.” Official scoring remains unchanged, as required, but the frozen gold span definition is not fully aligned with a literal source-copy contract.

This ambiguity is why no second prompt was tuned on the same frozen set.

### 8. Did improvements survive downstream Hammer?

There was no aggregate improvement to survive.

| Tool | Old Qwen + selected Hammer repair | New Qwen + selected Hammer repair |
|---|---:|---:|
| Search strict exact | 60% | 64% |
| Image strict exact | 56% | 48% |
| Ask Claude strict exact | 60% | 52% |
| Combined strict exact | 58.7% | 54.7% |
| Combined tool selection | 96.0% | 94.7% |
| Combined schema valid | 98.7% | 97.3% |

Strict argument and preservation-aware argument/exact accuracy equal the strict exact figures in this 75-case subset. The old arm had one rejection and no malformed output; the new arm had one rejection and one malformed output. Average Hammer latency was 2.420 s old versus 2.506 s new (p50 2.382/2.457 s, p95 2.950/3.073 s), at 19.15/18.53 generated tok/s and 3.470/3.464 GB peak MLX memory.

One search failure became a success, but three image and two external-agent successes became failures. The new Qwen prompt is rejected.

### 9. New targeted strict accuracies

The selected focused configuration retains the old Qwen contract and uses the repaired Hammer schema descriptions:

| Tool | Selected strict exact |
|---|---:|
| `see` | 56% |
| `recall` | 76% |
| `search` | 60% |
| `image` | 56% |
| `ask_claude` | 60% |

For transparency, the rejected experimental Qwen contract produced search 64%, image 48%, and ask-Claude 52%.

### 10. What causes the remaining failures?

- `recall`: the large rejection cluster was schema ontology, now fixed. Remaining failures are mostly Qwen verbatim/semantic issues plus one argument rewrite.
- `see`: mixed Hammer confidence/capacity, residual ontology sensitivity, one wrong Qwen domain, and target-form differences.
- Search/image/ask-Claude: primarily Qwen payload extraction under official scoring, with a smaller Hammer argument-preservation component.
- Benchmark ambiguity: material for unquoted leading articles and terminal punctuation; it affects interpretation of the Qwen experiment but does not explain omitted or shortened search queries.

### 11. Is Hammer 1.5B worth continuing with?

Yes. `recall` moved into the target range through a general schema repair, competing tools did not regress, and B-family tool selection under the selected repair is 96%. The evidence says many failures are interface/data-contract problems rather than a blanket 1.5B capacity ceiling.

Hammer 1.5B is still not ready for unattended tool execution: `see` remains only 56% strict and exact payload transport remains unresolved.

### 12. Is there enough evidence to run Hammer 3B?

Not for another broad benchmark. A future 25-case `see`-only capacity probe would now be scientifically defensible because schema wording has been isolated and six rejections remain. It should not precede resolution of the Qwen span-boundary contract and an independent evaluation set for exact extraction. Hammer 3B would not repair upstream Qwen payload loss.

## Recommendation

Adopt the four prototype-only Hammer schema descriptions as the current research candidate; do not adopt the rejection-prompt clarification or the new Qwen prompt.

The next useful work is not another full 450 run. Define an unambiguous span-annotation policy—especially articles and terminal punctuation—then evaluate a deterministic extractor or a new Qwen prompt on an independent frozen payload set. Separately, a small `see`-only 1.5B versus 3B capacity probe can be considered later.

## Safety and resource record

- Full-450 reruns: 0.
- New paid Qwen cases: 75.
- New OpenRouter cost: `$0.0219556`.
- Approximate balance remaining from the earlier reported account balance: `$1.23939`.
- Hammer model location: external disk.
- Yuki tool execution attempts: 0.
- Hammer 3B loads: 0.
- All model processes were unloaded between experiment arms.
