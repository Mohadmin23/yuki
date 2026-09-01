# Yuki Tool Dispatcher — Automatic Schema Selection Experiment

## Executive result

Automatic schema reduction works well on the existing Phase 1 data. The deterministic selector preserved the expected tool in **168/168 cases (100%)**, reduced exposure from 18 schemas to **3.54 schemas on average**, and added only **0.042 ms average / 0.074 ms p95** latency. Across the four models, average dispatcher latency fell by **53%–70%**.

The strongest practical 3B candidates are **Arch-Agent 3B** and **Hammer 2.1 3B**. xLAM-2 3B remains valuable as a third Phase 2 candidate because it reached 100% targeted tool selection and schema validity, but its argument preservation and latency were weaker. Hammer 1.5B became much faster and more accurate, but its targeted equivalent accuracy of 72.9% remains too far behind the 3B models for a reliability-first deployment.

No Yuki tool was executed. No external service, network-backed Yuki tool, shell command, file tool, camera, image worker, Claude bridge, or persistent memory path was called.

## Experiment controls

- Real source of truth: Yuki's live 18-tool registry, metadata, descriptions, parameters, and adapted schemas.
- New experimental variable: `deterministic-domain-rules-v1`, a local regex/phrase selector.
- Models: Hammer 2.1 1.5B, Hammer 2.1 3B, xLAM-2 3B, and Arch-Agent 3B, all FP16 through MLX from the external disk.
- Data: the unchanged 72-case broad suite and unchanged 96-case targeted suite.
- Generation: each model's verified native adapter, one stateless generation, 256-token ceiling, and complete-JSON early stopping.
- Scoring: unchanged strict and execution-equivalent argument rules.
- Execution: impossible through the automatic runner's public interface; its command has no execution, side-effect, or shell controls.
- Reuse: compatible all-18 and logical-oracle reports were reused. Only automatic-subset inference was newly run.
- Missing data: broad oracle runs do not exist. Logical-oracle runs for xLAM-2 3B and Arch-Agent 3B do not exist and are shown as **not run**.

The selector rules were developed and checked against these existing Phase 1 cases. Therefore, selector accuracy here is development-set performance, not a held-out reliability estimate. No cases were changed, but a future held-out suite is still required before deployment conclusions.

## Selector result

| Metric | Broad 72 | Targeted 96 | Combined unique cases |
|---|---:|---:|---:|
| Exact domain selection | 98.6% | 100.0% | 99.4% (167/168) |
| Correct tool available | 100.0% | 100.0% | 100.0% (168/168) |
| Average selected domains | 1.01 | 1.00 | 1.01 |
| Average schemas offered | 3.82 | 3.33 | 3.54 |
| Minimum / maximum schemas | 1 / 8 | 1 / 5 | 1 / 8 |

Across 672 measured selector calls (168 cases × four models), latency was **0.042 ms average, 0.042 ms p50, and 0.074 ms p95**.

The only non-exact classification was the intentionally ambiguous request `Extend poem.md using exactly poem.md|one more line`. The selector exposed both `computer` and `yuki_files`, keeping the expected `yuki_append` tool available. There were **zero `tool_unavailable` failures**.

Selector confusion by expected domain over the 168 unique cases:

| Expected domain | Selected domain(s) | Cases |
|---|---|---:|
| information | information | 52 |
| computer | computer | 36 |
| media | media | 24 |
| yuki_files | yuki_files | 27 |
| yuki_files | computer + yuki_files | 1 |
| memory | memory | 8 |
| external_agent | external_agent | 20 |

## Broad 72 comparison

`Total latency` was not recorded by the reused all-18 reports, so it is left as `n/r`; their measured dispatcher inference latency is preserved. Broad oracle results were not run.

| Model | Exposure | Tool available | Strict exact | Equivalent exact | Schema valid | Malformed | Avg total | Avg dispatcher | Schemas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hammer 1.5B | all 18 | 100% | 87.5% | 88.9% | 94.4% | 0.0% | n/r | 4.69 s | 18.00 |
| Hammer 1.5B | oracle | not run | — | — | — | — | — | — | — |
| Hammer 1.5B | automatic | 100% | 90.3% | 90.3% | 94.4% | 0.0% | 2.20 s | 2.20 s | 3.82 |
| Hammer 3B | all 18 | 100% | 91.7% | 93.1% | 98.6% | 0.0% | n/r | 11.92 s | 18.00 |
| Hammer 3B | oracle | not run | — | — | — | — | — | — | — |
| Hammer 3B | automatic | 100% | **95.8%** | 95.8% | 98.6% | 0.0% | 4.42 s | 4.41 s | 3.82 |
| xLAM-2 3B | all 18 | 100% | 91.7% | 91.7% | 97.2% | 0.0% | n/r | 16.13 s | 18.00 |
| xLAM-2 3B | oracle | not run | — | — | — | — | — | — | — |
| xLAM-2 3B | automatic | 100% | 93.1% | 93.1% | 98.6% | 0.0% | 4.80 s | 4.80 s | 3.82 |
| Arch-Agent 3B | all 18 | 100% | 90.3% | 93.1% | 98.6% | 1.4% | n/r | 12.72 s | 18.00 |
| Arch-Agent 3B | oracle | not run | — | — | — | — | — | — | — |
| Arch-Agent 3B | automatic | 100% | 93.1% | **95.8%** | **100%** | 0.0% | 4.45 s | 4.45 s | 3.82 |

Broad automatic exposure improved strict exact-call accuracy for every model: +2.8 points for Hammer 1.5B, +4.2 for Hammer 3B, +1.4 for xLAM-2 3B, and +2.8 for Arch-Agent 3B.

## Targeted 96 comparison

The comparable oracle condition is the logical-domain oracle. The earlier special confusion-pair oracle used smaller, case-specific subsets and is not mixed into this primary table.

| Model | Exposure | Tool available | Strict exact | Equivalent exact | Schema valid | Malformed | Avg total | Avg dispatcher | Schemas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hammer 1.5B | all 18 | 100% | 62.5% | 66.7% | 80.2% | 0.0% | n/r | 4.64 s | 18.00 |
| Hammer 1.5B | logical oracle | 100% | 71.9% | 72.9% | 86.5% | 0.0% | n/r | 2.12 s | 3.33 |
| Hammer 1.5B | automatic | 100% | **71.9%** | **72.9%** | 86.5% | 0.0% | 2.10 s | 2.10 s | 3.33 |
| Hammer 3B | all 18 | 100% | 80.2% | 81.3% | 87.5% | 0.0% | n/r | 10.95 s | 18.00 |
| Hammer 3B | logical oracle | 100% | 85.4% | 86.5% | 95.8% | 0.0% | n/r | 4.42 s | 3.33 |
| Hammer 3B | automatic | 100% | **85.4%** | **86.5%** | 95.8% | 0.0% | 4.54 s | 4.54 s | 3.33 |
| xLAM-2 3B | all 18 | 100% | 83.3% | 86.5% | 99.0% | 0.0% | n/r | 14.22 s | 18.00 |
| xLAM-2 3B | logical oracle | not run | — | — | — | — | — | — | — |
| xLAM-2 3B | automatic | 100% | 82.3% | 85.4% | **100%** | 0.0% | 4.71 s | 4.71 s | 3.33 |
| Arch-Agent 3B | all 18 | 100% | 82.3% | 86.5% | 99.0% | 1.0% | n/r | 11.98 s | 18.00 |
| Arch-Agent 3B | logical oracle | not run | — | — | — | — | — | — | — |
| Arch-Agent 3B | automatic | 100% | 84.4% | **88.5%** | **100%** | 0.0% | 4.33 s | 4.33 s | 3.33 |

Automatic exposure reproduced **100% of the measured logical-oracle strict and equivalent accuracy gain** for both Hammer models. xLAM-2 lost one strict and one equivalent case versus all 18, while Arch-Agent gained two cases.

## Automatic latency, throughput, and memory

| Model | Dataset | Avg / p50 / p95 total latency | Generated tok/s | Peak MLX memory |
|---|---|---:|---:|---:|
| Hammer 1.5B | broad | 2.20 / 2.25 / 2.73 s | 19.63 | 3.46 GB |
| Hammer 1.5B | targeted | 2.10 / 2.22 / 2.60 s | 19.72 | 3.45 GB |
| Hammer 3B | broad | 4.42 / 4.41 / 5.36 s | 9.84 | 6.52 GB |
| Hammer 3B | targeted | 4.54 / 4.56 / 5.53 s | 9.63 | 6.52 GB |
| xLAM-2 3B | broad | 4.80 / 4.65 / 6.43 s | 9.60 | 6.60 GB |
| xLAM-2 3B | targeted | 4.71 / 4.50 / 6.02 s | 9.63 | 6.58 GB |
| Arch-Agent 3B | broad | 4.45 / 4.38 / 5.86 s | 9.55 | 6.53 GB |
| Arch-Agent 3B | targeted | 4.33 / 4.14 / 5.69 s | 9.56 | 6.53 GB |

Compared with all-18 dispatcher latency, automatic reduction cut the broad/targeted averages by approximately:

- Hammer 1.5B: **53% / 55%**
- Hammer 3B: **63% / 59%**
- xLAM-2 3B: **70% / 67%**
- Arch-Agent 3B: **65% / 64%**

## Targeted failure diagnosis

These counts use execution-equivalent correctness, so harmless normalization is not treated as a practical failure.

| Model | Tool-selection failures | Argument failures | Main remaining families |
|---|---:|---:|---|
| Hammer 1.5B | 14 | 12 | see/camera 9; Claude 6; weather 4; search 4 |
| Hammer 3B | 4 | 9 | Claude 6; see/camera 3; read/Yuki-read 2; search 2 |
| xLAM-2 3B | 0 | 14 | Claude 6; filepath 3; see/camera 2; read/Yuki-read 2 |
| Arch-Agent 3B | 0 | 11 | Claude 4; search 4; see/camera 2; read/Yuki-read 1 |

Across automatic runs there were no targeted `tool_unavailable`, `malformed_output`, or selector failures. The benchmark supports and records all required stages; the only broad schema-validation failure was one xLAM call.

Interpretation:

- **Routing problems:** the selector caused no tool removal on this data. Its one broad multi-domain choice was conservative and successful. Held-out testing is still needed because the rules were developed on these cases.
- **Ontology problems:** separating normal filesystem and Yuki-file domains largely removes direct `read` versus `yuki_read` competition. Hammer 3B still made two tool-choice errors within the selected domains; the other models' remaining cases were argument errors rather than read/Yuki-read selection errors.
- **Model-capacity problems:** Hammer 1.5B still made 14 targeted tool-selection errors after perfect selector coverage, especially on camera, weather, and search prompts. Its remaining gap is not explainable by 18-schema overload alone.
- **Argument-preservation problems:** this is now the dominant 3B failure type. `ask_claude` text preservation failed 22 times across the four models; search text, filepaths, and optional camera targets are the other recurring families. Schema reduction alone does not solve copying fidelity.

## Final decisions

### 1. How accurate is the automatic selector itself?

It achieved **99.4% exact domain classification** (167/168). The one non-exact case deliberately selected two plausible domains.

### 2. How often does it preserve the correct tool?

**100% (168/168)**. No expected tool was removed.

### 3. How many schemas does each request expose?

**3.54 on average**, with 3.82 on broad and 3.33 on targeted, versus 18 in the baseline. The range was 1–8.

### 4. Does automatic schema reduction materially improve latency?

Yes. Average dispatcher latency fell by **53%–70%**, while selector overhead was effectively zero compared with generation time.

### 5. How much oracle accuracy improvement survives automatic selection?

For the two models with comparable logical-oracle data, **all of it**: Hammer 1.5B and Hammer 3B exactly matched their oracle strict and equivalent targeted scores. Oracle data for xLAM-2 and Arch-Agent was not run.

### 6. Does Hammer 1.5B become competitive enough to justify its efficiency advantage?

Not as the sole reliability-first dispatcher. It is excellent operationally—about **2.1 s** and **3.45 GB**, roughly half the latency and memory of the 3B models—but its targeted equivalent accuracy is **72.9%**, 12.5–15.6 points below the 3B finalists. Schema competition was a real problem, but model capacity remains meaningful. It remains attractive as an efficiency baseline or a future confidence-gated tier.

### 7. Which models should advance to Phase 2?

1. **Arch-Agent 3B** — best targeted equivalent accuracy (88.5%), fastest automatic 3B result (4.33 s), 100% tool selection/schema validity.
2. **Hammer 2.1 3B** — best targeted strict accuracy (85.4%) and best broad strict accuracy (95.8%); strongest balanced alternative.
3. **xLAM-2 3B**, if Phase 2 budget permits a third model — perfect targeted tool selection/schema validity and strong hard-routing behavior, though argument accuracy and latency trail the other 3B finalists.

Hammer 1.5B should be parked as the efficiency reference, not discarded.

### 8. What failure types remain?

- Selector routing: no tool-availability failures on current cases; held-out validation remains necessary.
- Ontology: greatly reduced by domain separation; small residual within-domain selection errors remain for Hammer.
- Capacity: clearly visible in Hammer 1.5B's camera/weather/search tool choices.
- Argument preservation: the main 3B limitation, led by Claude questions, search text, exact paths, and optional camera targets.

## Artifacts

- `automatic-routing-comparison.json` — normalized all-18/oracle/automatic comparison with missing conditions explicitly marked `not_run`.
- Eight per-model/per-dataset automatic JSON reports — full case results, selector metrics, dispatcher metrics, end-to-end metrics, confusion matrices, and stage counts.
- Eight per-model/per-dataset failure JSONL files — model-native raw output and detailed diagnostics.
- `automatic-routing-failures.jsonl` — combined automatic failure diagnostics across all models and datasets.

Phase 2 has **not** been started.
