# Yuki Tool Dispatcher — Complete Model and Experiment History

Date: 2026-08-26

## Executive summary

This report consolidates the complete Yuki tool-dispatcher research history from the first 72-case smoke benchmark through the typed argument contract and the model-free Hammer 1.5B pilot-dataset stage.

Thirteen distinct model checkpoints were tested. They did not all perform the same job or advance through every phase, so there is no honest single leaderboard covering all of them. Early checkpoints were direct local dispatchers; Qwen and GPT-OSS were later tested as semantic main brains; GPT-OSS, Hammer 7B, and Qwen3.8-Flash were also tested as direct all-tools callers. Scores below are grouped by the exact phase, dataset, schema exposure, and architecture that produced them. A dash means **not run**, not zero.

The current headline results are:

- Best local full-450 strict exact result: **Hammer 2.0 7B direct, 368/450 (81.8%)**.
- Best full-450 tool selection: **GPT-OSS-20B direct, 427/450 (94.9%)**.
- Best tested main-brain domain routing: **Qwen3.8-27B, 446/450 (99.1%)**.
- Best efficiency among local dedicated dispatchers: **Hammer 2.1 1.5B**, about 2.5 seconds and 3.47 GB in the Qwen full-450 pipeline.
- Best current two-model full-450 strict result: **Qwen3.8-27B → Hammer 2.1 1.5B, 358/450 (79.6%)**.
- Deterministic binding proved that exact character transport can reach **95.1% literal-source accuracy** on 123 eligible frozen Hammer 7B cases, but applying it blindly lowered official strict accuracy because the old gold sometimes expects semantic normalization rather than literal copying.
- The latest Hammer 7B schema repair improved a difficult 49-case ontology set from **18.4% to 44.9% strict exact** without regressions, but fixed only 6/14 primary confusions. The predeclared gate required 7, so no repaired full-450 rerun occurred.
- The typed argument contract is now followed by immutable, human-approved **150-record Hammer 1.5B Pilot v1.2** covering all 18 tools. V1.2 applies exactly five final human revisions to checksum-locked v1.1 and retains zero sacred-450 leakage flags, diversified ontology language, and structured `filename`/`content` training fields through a prototype-only adapter. This was a data-format/validation stage only: no model was trained and the frozen 450 was not rerun.
- Qwen3.8-Flash passed a direct-native all-18 development screen at **167/168 (99.4%) tool selection** and **145/168 (86.3%) strict exact**, with **0 malformed calls**. This used only the non-sacred broad-72 and targeted-96 suites; no sacred-450 run followed.

No result in this report includes Yuki tool execution. Every benchmark stopped after generation, parsing, validation, and scoring.

## How to read the scores

These metrics answer different questions:

| Metric | Meaning |
|---|---|
| Tool selection | The model chose the correct Yuki tool, regardless of argument correctness. |
| Strict exact call | Both tool and argument object exactly match the frozen official gold. |
| Execution-equivalent exact | Tool is correct and arguments are equivalent under conservative per-tool normalization. Paths, commands, filenames, and file contents remain exact-sensitive. |
| Preservation-aware exact | A later metric that tolerates carefully defined harmless differences while still exposing unwanted rewriting. |
| Literal-source exact | The final argument matches a frozen annotation copied literally from the immutable raw request. This measures the deterministic binder contract, not the older normalized gold contract. |
| Schema valid | The emitted call exists and passes the adapted live Yuki schema. |
| Malformed | Output cannot be parsed as the expected native/canonical structured format. |
| Domain accuracy | A main brain chose the correct schema family, not necessarily the final tool. |
| Verbatim recall | A main brain preserved the expected exact-sensitive strings in its delegation object. |

Strict exact scores are comparable only when the dataset and architecture are the same. For example, 91.7% on the 72-case smoke set is not stronger evidence than 81.8% on the much harder 450-case reliability set.

## Model inventory

| # | Checkpoint | Size / runtime | Tested roles | Furthest phase | Outcome |
|---:|---|---|---|---|---|
| 1 | `Salesforce/xLAM-1b-fc-r` | 1B FP16, local MLX | Direct dispatcher | Broad 72 | Retired; weak argument fidelity. |
| 2 | `Salesforce/xLAM-2-1b-fc-r` | 1B FP16, local MLX | Direct dispatcher | Broad 72 | Retired; Hammer 1.5B was better at similar memory. |
| 3 | `MadeAgents/Hammer2.1-1.5b` | 1.5B FP16, local MLX | Direct dispatcher; Qwen downstream dispatcher; binder arm | Full 450 plus focused repairs | Current efficiency/reference dispatcher. |
| 4 | `MadeAgents/Hammer2.1-3b` | 3B FP16, local MLX | Direct dispatcher; oracle/automatic subsets; focused capacity reference | Phase 2 450 and focused probes | Real `see` capacity advantage, but not enough global gain to replace 1.5B. |
| 5 | `katanemo/Arch-Function-3B` | 3B FP16, local MLX | Direct dispatcher | Targeted 96 | Retired; too many malformed/conversational outputs. |
| 6 | `Salesforce/xLAM-2-3b-fc-r` | 3B FP16, local MLX | Direct dispatcher; automatic subsets | Automatic 96 | Excellent difficult-prompt selection; slower and did not enter Phase 2 final pair. |
| 7 | `katanemo/Arch-Agent-3B` | 3B FP16, local MLX | Direct dispatcher; automatic subsets | Phase 2 450 | Strong semantic selection; malformed/conversational behavior and license caveat. |
| 8 | `ibm-granite/granite-4.1-3b` | 3B FP16, local MLX | Direct dispatcher | Targeted 96 | Retired; weak exact arguments/schema adherence and highest memory in its cohort. |
| 9 | `Qwen/Qwen3-14B-MLX-4bit` | 14B 4-bit, local MLX | Temporary semantic main-brain surrogate | Production-interface 180 | Useful architecture proof only; explicitly superseded by intended Qwen3.8-27B. |
| 10 | `qwen/qwen3.8-27b` | 27B, OpenRouter | Intended semantic main brain | Production-interface 450 | Best domain router; kept as the main-brain reference. |
| 11 | `openai/gpt-oss-20b` | 20B, OpenRouter/DeepInfra | Semantic main brain and direct native caller | Two full-450 runs | Poorer than Qwen as delegator; extremely fast and strong at direct tool selection. |
| 12 | `MadeAgents/Hammer2.0-7b` | 7B 8-bit, local MLX | Direct native caller | Full 450 plus frozen binder/ontology repair | Best local full-450 strict score; expensive in latency/memory. |
| 13 | `qwen/qwen3.8-flash` | API, OpenRouter/Alibaba | Direct native caller | Broad 72 + targeted 96 screen | Passed the 168-case gate; exact payload rewriting remains concentrated in search and external-agent questions. |

The local model files and source caches used for the completed model-comparison work were moved to `/Volumes/madisk`; compatibility links may exist on the internal disk.

## Research timeline in one view

```text
Phase 1A: 8 local models, broad 72, all 18 schemas
    ↓ find viable checkpoints
Phase 1B: 6 models, targeted 96 failure suite
    ↓ separate easy coverage from robustness
Phase 1C: Hammer 1.5B vs 3B, oracle schema subsets
    ↓ measure schema competition versus capacity
Phase 1D: 4 finalists, automatic schema selector
    ↓ test a realistic smaller-schema pipeline
Phase 2: Hammer 3B vs Arch-Agent 3B, held-out 450
    ↓ selector fails to generalize; redesign interface
Production-interface: Qwen semantic delegation → Hammer
    ↓ Qwen3.8 fixes domain selection; payload/rejection failures remain
Focused repair: schema wording, Qwen verbatim prompt, capacity probes
    ↓ schema repair helps; prompt-only exact copying fails
Deterministic binder: copy spans from raw user request
    ↓ literal transport works; reveals gold/interface mismatch
Alternative direct callers: GPT-OSS-20B and Hammer 2.0 7B
    ↓ 7B becomes strict leader; ontology remains the latest problem
Typed contract: classify every field as literal, semantic, or no-argument
    ↓ preserve exact source data without penalizing useful semantic extraction
Current: immutable human-approved 150-record Hammer 1.5B Pilot v1.2
    ↓ staged dataset work paused for a controlled direct-caller screen
Phase 8: Qwen3.8-Flash direct native, broad 72 + targeted 96
    ↓ passes screening gate; sacred 450 remains deferred and unauthorized
```

## Phase 1A — broad 72-case local-model smoke test

### Why this phase existed

The first question was whether a roughly 1B–3B local model could act only as a learned function router/parser across Yuki's 18 real tools. The test deliberately used all 18 schemas, one stateless generation, native model output, a 256-token ceiling, strict adapted schemas, and no regex fallback or tool execution.

### Results

| Model | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg latency | tok/s | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hammer 2.1 3B | **97.2%** | **91.7%** | **93.1%** | **98.6%** | 0.0% | 11.92 s | 9.64 | 6.651 GB |
| xLAM-2 3B | **97.2%** | **91.7%** | 91.7% | 97.2% | 0.0% | 16.13 s | 9.58 | 6.652 GB |
| Arch-Agent 3B | **97.2%** | 90.3% | **93.1%** | **98.6%** | 1.4% | 12.72 s | 9.73 | 6.652 GB |
| Hammer 2.1 1.5B | 91.7% | 87.5% | 88.9% | 94.4% | 0.0% | **4.69 s** | **18.47** | 3.624 GB |
| xLAM-2 1B | 90.3% | 81.9% | 83.3% | 97.2% | 0.0% | 6.22 s | 18.43 | 3.624 GB |
| Arch-Function 3B | 88.9% | 80.6% | 81.9% | 86.1% | 9.7% | 13.58 s | 9.50 | 6.652 GB |
| Granite 4.1 3B | 91.7% | 73.6% | 75.0% | 90.3% | 0.0% | 15.71 s | 8.61 | 7.441 GB |
| Original xLAM 1B | 84.7% | 45.8% | 50.0% | 97.2% | 0.0% | 5.65 s | 18.39 | 3.594 GB |

### Successor decision

Hammer 1.5B became the efficiency candidate. Hammer 3B became the stable 3B baseline. xLAM-2 3B and Arch-Agent 3B stayed as serious accuracy contenders. Original xLAM 1B, xLAM-2 1B, Arch-Function 3B, and Granite 3B were removed from the main shortlist.

The 72 cases were only a smoke test, so the next phase concentrated on known failure families rather than drawing a reliability conclusion.

## Phase 1B — targeted 96-case failure suite

### Why this phase existed

Strict argument equality was mixing harmless casing changes with destructive changes to file paths, commands, filenames, and file content. The scorer therefore retained strict accuracy and added conservative execution-equivalent scoring. The 96 cases emphasized weather routing, `see`, `read` versus `yuki_read`, query/path preservation, and `ask_claude` payloads.

The two earlier 1B xLAM models were **not run** on this later suite.

### Results

| Model | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg latency | tok/s | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xLAM-2 3B | **99.0%** | **83.3%** | **86.5%** | **99.0%** | 0.0% | 14.22 s | 9.11 | 6.652 GB |
| Arch-Agent 3B | 97.9% | 82.3% | **86.5%** | **99.0%** | 1.0% | 11.98 s | 9.54 | 6.652 GB |
| Hammer 2.1 3B | 87.5% | 80.2% | 81.3% | 87.5% | 0.0% | 10.95 s | 9.41 | 6.651 GB |
| Arch-Function 3B | 89.6% | 78.1% | 79.2% | 81.3% | 10.4% | 13.81 s | 9.36 | 6.652 GB |
| Hammer 2.1 1.5B | 75.0% | 62.5% | 66.7% | 80.2% | 0.0% | **4.64 s** | **18.44** | 3.624 GB |
| Granite 4.1 3B | 93.8% | 59.4% | 60.4% | 80.2% | 1.0% | 14.70 s | 8.53 | 7.441 GB |

### What changed in our understanding

The easy broad suite had hidden large differences. xLAM-2 3B and Arch-Agent handled difficult selection well. Hammer 1.5B remained efficient but fell sharply when all 18 tools competed. This created the next question: was 1.5B too small, or merely overloaded by schemas?

### Successor decision

Only Hammer 1.5B and Hammer 3B entered the controlled oracle-capacity experiment. That omission was experimental isolation, not a claim that the other models scored zero.

## Phase 1C — oracle schema-subset experiment

### Why this phase existed

The harness used the known gold family to expose only the relevant logical or confusion-focused subset. This deliberately unrealistic oracle removed the need for a second router and isolated the effect of schema competition.

Only Hammer 2.1 1.5B and Hammer 2.1 3B were run.

### Results on the same targeted 96 cases

| Model / exposure | Tool selection | Strict exact | Equivalent exact | Schema valid | Avg latency | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|
| Hammer 1.5B, all 18 | 75.0% | 62.5% | 66.7% | 80.2% | 4.64 s | 3.62 GB |
| Hammer 1.5B, logical oracle | 85.4% | 71.9% | 72.9% | 86.5% | 2.12 s | 3.45 GB |
| Hammer 1.5B, focused oracle | 89.6% | 76.0% | 77.1% | 89.6% | 2.04 s | 3.40 GB |
| Hammer 3B, all 18 | 87.5% | 80.2% | 81.3% | 87.5% | 10.95 s | 6.65 GB |
| Hammer 3B, logical oracle | 95.8% | 85.4% | 86.5% | 95.8% | 4.42 s | 6.52 GB |
| Hammer 3B, focused oracle | 96.9% | 87.5% | 88.5% | 96.9% | 4.59 s | 6.46 GB |

### Conclusion and successor

Schema competition was real: smaller subsets improved accuracy and roughly halved latency. Capacity was also real: 1.5B never caught 3B under identical oracle exposure. The especially poor `see` versus `image` performance remained a 1.5B warning.

The next phase replaced the unrealistic gold oracle with a deterministic automatic domain selector.

## Phase 1D — automatic schema selection on the development suites

### Why this phase existed

This was the first realistic version of:

```text
request → automatic domain selection → smaller schema set → model dispatcher
```

The selector was deterministic and did not load another model. It offered the expected tool in all 168 development cases and exposed about 3.54 schemas on average. Only the four active finalists were run: Hammer 1.5B, Hammer 3B, xLAM-2 3B, and Arch-Agent 3B.

### Broad 72 strict exact

| Model | All 18 | Automatic subset | Automatic schema valid | Automatic avg latency |
|---|---:|---:|---:|---:|
| Hammer 1.5B | 87.5% | **90.3%** | 94.4% | 2.20 s |
| Hammer 3B | 91.7% | **95.8%** | 98.6% | 4.42 s |
| xLAM-2 3B | 91.7% | **93.1%** | 98.6% | 4.80 s |
| Arch-Agent 3B | 90.3% | **93.1%** | 100% | 4.45 s |

### Targeted 96 strict exact

| Model | All 18 | Automatic subset | Equivalent exact | Schema valid | Avg latency |
|---|---:|---:|---:|---:|---:|
| Hammer 1.5B | 62.5% | **71.9%** | 72.9% | 86.5% | 2.10 s |
| Hammer 3B | 80.2% | **85.4%** | 86.5% | 95.8% | 4.54 s |
| xLAM-2 3B | **83.3%** | 82.3% | 85.4% | 100% | 4.71 s |
| Arch-Agent 3B | 82.3% | **84.4%** | **88.5%** | 100% | 4.33 s |

### Conclusion and successor

Automatic subsets reproduced most oracle gains on the development cases. Hammer 1.5B improved but remained below the 3B models, while Hammer 3B and Arch-Agent offered the best combined tradeoffs.

This apparent success later proved optimistic: the deterministic selector had been developed around these cases. Phase 2 therefore froze a new 450-case dataset to test generalization. Hammer 3B and Arch-Agent 3B were the only two Phase 2 finalists. xLAM-2 3B was not run in Phase 2; it was deferred rather than disproven.

## Phase 2 — held-out 450-case automatic-routing reliability study

### Why this phase existed

The small suites could rank candidates but could not support reliability claims. Phase 2 created 450 frozen cases—25 per real Yuki tool—with normal, paraphrase, confusion, preservation, and adversarial categories. It tested the full automatic-subset pipeline, not just the dispatcher conditional on a correct subset.

### Selector result

- Expected tool offered: **321/450 (71.3%)**.
- Exact domain selection: **291/450 (64.7%)**.
- Average schemas offered: **4.347**.

This was the central failure: the development-set selector did not generalize.

### End-to-end results

| Model | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg / p50 / p95 | tok/s | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hammer 2.1 3B | 66.7% | 60.0% | 60.4% | 67.6% | **0.0%** | 4.466 / 4.483 / 6.818 s | 10.54 | 6.533 GB |
| Arch-Agent 3B | **70.7%** | **60.9%** | **63.3%** | **85.3%** | 13.6% | 5.372 / 5.091 / 8.140 s | 9.29 | 6.541 GB |

Conditional on being shown the expected tool, Hammer selected correctly 93.5% of the time and Arch-Agent 99.1% of the time. Their strict exact results were statistically tied on this set; the dominant bottleneck was the selector.

### Successor decision

This phase ended the idea that a hand-built domain selector was already production-ready. Instead of tuning it further on the same data, the architecture changed: the main conversational model would create a semantic delegation, and the small model would select within the resulting live schema subset.

Hammer 3B remained the stable direct-reference model, but Hammer 1.5B returned because the next architecture might move enough semantic work upstream to make its efficiency worthwhile.

## Phase 3 — production-interface experiment with a temporary 14B surrogate

### Why this phase existed

The intended main brain was unavailable locally, so `Qwen3-14B-MLX-4bit` was explicitly labeled as a temporary surrogate. It generated only:

```json
{
  "request": "semantic delegated action",
  "domain_hint": "schema family",
  "verbatim": ["exact-sensitive strings"]
}
```

Hammer then received the corresponding live Yuki subset. The initial smoke gate passed, but resource constraints led to the predeclared deterministic 180-case fallback rather than all 450.

### Stage A surrogate result, 180 cases

| Valid delegation | Domain accuracy | Verbatim recall | Verbatim precision | Avg latency | tok/s | Peak MLX |
|---:|---:|---:|---:|---:|---:|---:|
| 95.0% | 87.8% | 84.2% | 61.7% | 5.54 s | 6.68 | 8.35 GB |

### End-to-end results

| Dispatcher | Tool selection | Strict exact | Preservation-aware exact | Schema valid | Avg latency | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|
| Hammer 2.1 1.5B | 83.9% | **73.3% (132/180)** | **73.9%** | 90.0% | 2.71 s | 3.46 GB |
| Hammer 2.1 3B | **84.4%** | 72.2% (130/180) | 72.8% | 90.0% | 5.21 s | 6.53 GB |

### Conclusion and successor

The new semantic-delegation architecture beat the same-set hand-built selector, and Hammer 1.5B matched or slightly beat 3B at about half the local dispatch cost. But the surrogate's 87.8% domain accuracy dominated many failures. The artifacts were versioned so the exact downstream evaluation could be repeated with the intended 27B model.

## Phase 4 — Qwen3.8-27B main brain → Hammer 1.5B, full 450

### Why this phase existed

This was the intended architecture test with the exact `qwen/qwen3.8-27b` OpenRouter model, reasoning disabled, strict JSON-schema delegation, and the same frozen 450 cases. At the user's direction, Hammer 3B was **not run**.

### Qwen Stage A

| Valid JSON | Domain accuracy | Verbatim recall | Verbatim precision | Avg latency | tok/s |
|---:|---:|---:|---:|---:|---:|
| 100.0% | **99.1% (446/450)** | 90.6% | 75.9% | 2.145 s | 21.25 |

### Qwen → Hammer 1.5B result

| Tool selection | Strict exact | Execution-equivalent | Preservation-aware | Schema valid | Malformed | Hammer avg latency | Peak MLX |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 90.7% | **79.6% (358/450)** | 79.8% | 80.9% | 92.4% | 0.4% | 2.486 s | 3.473 GB |

Sequential Qwen plus Hammer latency was about 4.631 seconds. The full Qwen run cost approximately `$0.1331`.

### What succeeded

Qwen nearly solved domain routing. `time`, `weather`, and `yuki_list` reached 100% strict downstream calls; `calc`, `read`, and `yuki_read` reached 96%. The production-style interface was much stronger than the failed automatic selector.

### What failed

- `see`: 40% tool selection, 36% strict, 13/25 explicit rejections.
- `recall`: 52% tool selection, 44% strict, 11/25 explicit rejections.
- `search`, `image`, and `ask_claude`: tool choice was usually right, but exact payloads were lost or rewritten.
- Of 92 strict failures, 37 were associated with upstream verbatim loss, 29 with wrong final tools, 14 with wrong arguments, 6 with argument rewrites, 4 with Qwen domain errors, and 2 with malformed Hammer output.

### Successor decision

The next experiments separated the two dominant problems instead of rerunning 450 cases:

1. Hammer/schema semantic rejection for `see` and `recall`.
2. Exact payload transport for `search`, `image`, and `ask_claude`.

## Phase 5A — focused Hammer schema repair

### Why this phase existed

The exact frozen Qwen delegations were reused. Only dispatcher-facing descriptions for `see`, `image`, `recall`, and `remember` changed; schemas, parameters, model weights, gold, and Yuki implementations did not.

### Results

| Tool / metric | Frozen baseline | A1 schema descriptions | A2 schema + rejection text |
|---|---:|---:|---:|
| `see` tool selection | 40% | **68%** | 68% |
| `see` strict exact | 36% | **56%** | 56% |
| `see` schema valid | 48% | **76%** | 72% |
| `see` rejections | 13 | **6** | 7 |
| `recall` tool selection | 52% | **96%** | 96% |
| `recall` strict exact | 44% | **76%** | 76% |
| `recall` schema valid | 56% | **100%** | 100% |
| `recall` rejections | 11 | **0** | 0 |

Regression checks did not degrade competing tools: `image` strict improved from 48% to 56%, while `remember` remained 76% strict and 96% preservation-aware.

### Conclusion and successor

A1 descriptions were adopted for research. They define `see` as inspecting an existing visual source, `image` as creating a new image, `recall` as retrieving persistent memory, and `remember` as storing a new fact. A2's generic rejection clarification had no independent benefit and was rejected.

`recall` was largely an ontology-description problem. `see` remained partly a model-capacity or interface problem.

## Phase 5B — Qwen lossless-verbatim prompt attempt

### Why this phase existed

The hypothesis was that stronger instructions could make Qwen copy downstream payload strings exactly. The same 75 frozen `search`, `image`, and `ask_claude` cases were used.

### Qwen result

| Metric | Old contract | New lossless prompt | Change |
|---|---:|---:|---:|
| Valid JSON | 100% | 100% | 0 pp |
| Domain accuracy | 98.7% | 97.3% | -1.3 pp |
| Exact character preservation | 61.3% | 53.3% | -8.0 pp |
| Verbatim precision | 61.3% | 52.7% | -8.7 pp |

### Downstream Hammer result

| Tool | Old Qwen strict | New Qwen strict |
|---|---:|---:|
| Search | 60% | 64% |
| Image | **56%** | 48% |
| Ask Claude | **60%** | 52% |
| Combined | **58.7%** | 54.7% |

### Conclusion and successor

The prompt change was rejected. It proved that asking a semantic model more emphatically to copy text is not a dependable lossless-transport mechanism. It also exposed disagreement between literal source preservation and official gold that sometimes removed articles or punctuation.

The successor architecture moved exact copying out of both models and into deterministic code.

## Phase 5C — focused Hammer 3B `see` capacity probe

### Why this phase existed

After schema wording had already improved `see`, the exact same 25 frozen cases tested whether remaining failures were a real capacity gap. Qwen was not called. Hammer 3B was loaded only for this probe.

| Metric | Hammer 1.5B A1 | Hammer 3B A1 | Change |
|---|---:|---:|---:|
| Tool selection | 68% | **96%** | +28 pp |
| Strict exact | 56% | 56% | 0 pp |
| Schema valid | 76% | **100%** | +24 pp |
| Rejections | 6 | **0** | -6 |
| Avg latency | **1.755 s** | 3.726 s | 2.12× |
| Peak MLX | **3.459 GB** | 6.526 GB | 1.89× |

### Conclusion

There is a real 3B capacity advantage for understanding/selecting `see`, but strict accuracy did not move because the surviving errors were target-string boundary problems. This directly motivated deterministic source binding.

## Phase 5D — deterministic source-span binder

### Why this phase existed

Qwen should decide semantics/domain; Hammer should select the tool; neither should be trusted to reproduce exact-sensitive strings. The binder therefore recovered final argument values by slicing the immutable raw request. Frozen annotations were hidden from it until predictions were sealed.

Primary set: 100 cases, 25 each for `search`, `image`, `ask_claude`, and `see`. A separate 25-case `recall` set was diagnostic only. No model was loaded.

### Hammer 1.5B frozen-output result

| Metric | Qwen verbatim → Hammer baseline | Deterministic binder |
|---|---:|---:|
| Tool selection | 89% | 89% |
| Official strict exact | 58% | **64%** |
| Literal-source exact | 51% | **87%** |
| Schema valid | **93%** | 90% |

Per-tool official exact changed as follows:

- Search: 60% → **80%**.
- Image: 56% → 52%, while literal-source exact reached **100%**.
- Ask Claude: 60% → **96%**.
- See: 56% → 28%, while literal-source exact improved from 36% to 60%.
- Recall diagnostic: official 76% → 36%, while literal-source exact improved from 48% to **96%**.

Every emitted bound string was a character-for-character source slice. The apparent regressions show that the official benchmark often expects semantic forms such as `keys` where the literal source is `my keys`.

### Conclusion and successor

Deterministic binding removes most string-copy failures conditional on correct tool selection. It must be applied by explicit field contract, not uniformly:

```text
literal_source     → source span required
semantic_argument → normalized semantic value required
```

## Phase 5E — Hammer 1.5B versus 3B with the same binder

### Why this phase existed

Once string regeneration was removed, this measured whether 3B still offered enough semantic selection gain to justify roughly twice the local cost.

| Metric | Hammer 1.5B | Hammer 3B | 3B change |
|---|---:|---:|---:|
| Tool selection | 89% | **95%** | +6 pp |
| Official strict exact | **64%** | 63% | -1 pp |
| Literal-source exact | 87% | **92%** | +5 pp |
| Schema valid after binding | 90% | **95%** | +5 pp |
| Rejections | 7 | **3** | -4 |
| Avg latency | **2.229 s** | 4.398 s | 1.97× |
| Peak MLX | **3.470 GB** | 6.537 GB | 1.88× |

The entire net capacity gain came from `see`; search and image tied, and 3B was worse on one `ask_claude` case. Hammer 1.5B remained the default experimental dispatcher, with 3B retained as a capacity reference.

## Phase 6A — GPT-OSS-20B as a Qwen replacement

### Why this phase existed

GPT-OSS-20B was first tested in the same Stage A semantic-delegation role as Qwen3.8-27B. The initial default-provider smoke failed materially, so the controlled run used the verified DeepInfra provider and preserved the failed smoke as evidence.

### Full-450 Stage A comparison

| Metric | Qwen3.8-27B | GPT-OSS-20B |
|---|---:|---:|
| Valid delegation | **100.0%** | 99.8% |
| Domain accuracy | **99.1%** | 86.2% |
| Deterministic semantic score | 69.3% | 47.3% |
| Verbatim recall | **90.6%** | 83.4% |
| Verbatim precision | **75.9%** | 55.3% |
| Avg latency | 2.145 s | **1.032 s** |
| Generation rate | 21.25 tok/s | **74.48 tok/s** |
| Observed cost | about $0.1331 | **about $0.0121** |

### Conclusion and successor

GPT-OSS was much faster and cheaper but did not replace Qwen as semantic main brain because its 86.2% domain accuracy would reintroduce the exact bottleneck the Qwen architecture had solved.

Its native tool-calling ability looked promising, so it was next tested directly against all 18 schemas, bypassing both semantic delegation and Hammer.

## Phase 6B — GPT-OSS-20B direct native full 450

### Results

| Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg latency | tok/s | Cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **94.9% (427/450)** | 76.2% (343/450) | 77.8% | **98.7%** | 1.1% | **0.741 s** | 55.10 | $0.0259 |

### Conclusion and successor

Direct GPT-OSS selected tools better than Qwen → Hammer and was 6.25× faster in observed routing latency, but its exact arguments were worse: 76.2% versus 79.6%. It solved most `see` and `recall` selection failures yet still rewrote image prompts, paths, and external-agent questions.

This result suggested that a larger direct function-calling model could be competitive if it retained exact arguments. Hammer 2.0 7B was therefore tested next as a local direct caller.

## Phase 7A — Hammer 2.0 7B direct native full 450

### Why this phase existed

The goal was to test whether increased local function-calling capacity could beat both the two-model pipeline and remote direct GPT-OSS. The model was converted to MLX 8-bit to fit the M1 Air safely. Static prompt-prefix caching was verified on a smoke set before the one full run; it reduced average smoke latency from 34.83 to 5.68 seconds without changing calls.

### Frozen result

| Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg latency | tok/s | Peak MLX |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **91.6% (412/450)** | **81.8% (368/450)** | 81.8% | 95.6% | **0.0%** | 6.236 s cached | 7.53 | 8.91 GB |

Frozen result SHA-256: `97980e3a53d7e105046dda2a4bd6c982682c047c5dace89a55f5d77fd891cb32`.

### Same full-450 architecture comparison

| Architecture | Tool selection | Strict exact | Schema valid | Malformed | Routing latency | Peak local MLX | API cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hammer 2.0 7B direct | 91.6% | **81.8%** | 95.6% | **0.0%** | 6.236 s | 8.91 GB | $0 |
| GPT-OSS-20B direct | **94.9%** | 76.2% | **98.7%** | 1.1% | **0.741 s** | remote | $0.0259 |
| Qwen3.8-27B → Hammer 1.5B | 90.7% | 79.6% | 92.4% | 0.4% | 4.631 s sequential | 3.47 GB Hammer | $0.1331 Qwen |
| Phase 2 auto-selector → Hammer 3B | 66.7% | 60.0% | 67.6% | 0.0% | 4.466 s | 6.53 GB | $0 |
| Phase 2 auto-selector → Arch-Agent 3B | 70.7% | 60.9% | 85.3% | 13.6% | 5.372 s | 6.54 GB | $0 |

The Phase 2 rows use the same source 450 cases but a failed schema-selector architecture, so they diagnose that architecture rather than the underlying models in isolation.

### What Hammer 7B fixed and what remained

Hammer 7B became the strict exact leader and produced no malformed calls. Its 44 correct-tool/wrong-argument cases indicated substantial remaining recoverable payload error. Its largest semantic confusions were:

- `yuki_read → read`: 8 cases.
- `yuki_write → yuki_append`: 6 cases.

This led directly to the current frozen binder and minimal ontology repair.

## Phase 7B — current work: frozen Hammer 7B binder and ontology repair

### Part 1: deterministic binder on frozen outputs

No Hammer case was regenerated. Work began with the 44 cases where the frozen model selected the correct tool but produced the wrong argument.

| Measure | Frozen Hammer | Binder result |
|---|---:|---:|
| Official strict within the 44 | 0/44 | **24/44** |
| Literal-source exact within 31 annotated cases | 7/31 | **31/31** |
| Official calls broken within the 44 | — | **0** |

The 24 official repairs were: search 5, image 3, ask Claude 8, Yuki write 4, and Yuki append 4.

Uniformly applying the binder across all 450 changed official strict from **368/450 (81.8%) to 361/450 (80.2%)**, because it repaired 24 official failures but broke 31 cases whose gold expected normalization. At the same time, literal-source exact across 123 eligible annotations rose from **64/123 (52.0%) to 117/123 (95.1%)**. Every emitted bound value passed the source-copy invariant.

An oracle that applied binding only to the known 44 failures would score **392/450 (87.1%)**, but that is not deployable because production code cannot see gold failure labels. It is an upper bound, not a candidate result.

### Part 2: minimal schema ontology repair

Only the prototype-facing descriptions changed:

- `read`: operating-system filesystem only; never Yuki's internal managed store.
- `yuki_read`: Yuki internal store only; never arbitrary OS paths.
- `yuki_write`: create or replace complete content.
- `yuki_append`: preserve existing content and add after it.

The frozen 49-case target set contained all 14 dominant confusions plus read/write/append controls and all prior explicit rejections.

| Metric | Frozen baseline | Repaired schemas |
|---|---:|---:|
| Tool selection | 15/49 (30.6%) | **28/49 (57.1%)** |
| Strict exact | 9/49 (18.4%) | **22/49 (44.9%)** |
| Schema valid | 29/49 (59.2%) | **36/49 (73.5%)** |
| Rejections | 20 | **13** |
| Malformed | 0 | 0 |
| New strict regressions | — | **0** |
| Avg latency | frozen | 6.47 s |
| Peak MLX | frozen | 8.91 GB |

Confusion repairs:

- `yuki_read → read`: 8 failures → 5, so 3 fixed.
- `yuki_write → yuki_append`: 6 failures → 3, so 3 fixed.
- Combined: 14 → 8, so 6 fixed.

The predeclared full-run gate required at least 7/14 primary confusions fixed. Because only 6 were fixed, the gate was not relaxed and **no repaired 450-case rerun was performed**.

## Phase 8 — Qwen3.8-Flash direct-native 168-case screen

### Why this phase existed

This isolated screen asked whether Qwen3.8-Flash could directly select among all 18 Yuki tools without Qwen semantic delegation, Hammer, schema subsets, retries, or tool execution. It reused the exact frozen broad-72 and targeted-96 development suites and the historical strict live-registry schema profile.

Alibaba rejected the historical `tool_choice=required` value before inference, so the completed run used the provider-compatible `tool_choice=auto` and disclosed that profile difference. Every successful response still produced exactly one native tool call.

### Results

| Suite | Tool selection | Strict exact | Equivalent exact | Schema valid | Malformed | Avg latency | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Broad 72 | 100.0% | 91.7% | 93.1% | 100.0% | 0.0% | 1.124 s | $0.0116 |
| Targeted 96 | 99.0% | 82.3% | 83.3% | 99.0% | 0.0% | 1.592 s | $0.0150 |
| Combined 168 | **99.4% (167/168)** | **86.3% (145/168)** | 87.5% | **99.4%** | **0.0%** | 1.390 s | $0.0267 |

The sole selection/schema failure was an upstream Alibaba 429 with retries disabled, not a model rejection. Flash had zero explicit rejections, zero model-produced no-calls, zero multiple calls, and zero malformed calls. The typed score was 63/64 (98.4%) on independently scoreable cases, but only 64/168 cases had sufficient contract annotation, so the gate passed through historical strict scoring rather than the incomplete typed arm.

### Conclusion

Flash passed every predeclared screening condition and therefore earned consideration for a later sacred-450 direct-native run. That run was not authorized or performed. The remaining weakness was not ontology selection: `search.query` and `ask_claude.question` were each rewritten in 9/20 cases, while filesystem paths, shell commands, URLs, and Yuki file payloads were preserved in all eligible correct calls.

### Zero-inference typed-binder replay

Before any sacred-450 decision, the 168 Flash outputs were frozen and replayed through a typed deterministic binder. Independent raw-request annotations covered all 104 literal-source cases; semantic and no-argument fields passed through unchanged.

| Metric | Raw Flash | Bound Flash |
|---|---:|---:|
| Historical strict exact | 145/168 (86.3%) | **161/168 (95.8%)** |
| Execution-equivalent exact | 147/168 (87.5%) | **163/168 (97.0%)** |
| Fully covered typed exact | 148/168 (88.1%) | **166/168 (98.8%)** |
| Literal annotated typed exact | 85/104 (81.7%) | **103/104 (99.0%)** |
| Schema valid | 167/168 (99.4%) | 167/168 (99.4%) |

The binder repaired all 19 destructive literal rewrites and copied 104/104 emitted literal values from the raw request. It nevertheless caused one new `ask_claude` regression by including the scaffolding `handle this exact question:` in the payload. The predeclared accuracy condition passed, but the zero-regression condition failed. The replay was frozen without tuning that case, and **no sacred-450 Flash run followed**.

### Fresh payload-boundary probe and frozen v2 replay

The v1 result above was frozen before any repair. A separate 64-case, non-sacred development probe then tested literal boundaries: 40 fresh `ask_claude` cases plus 24 controls for search, image, paths, URLs, commands, and Yuki filenames/content. It deliberately included nested delegation-like text, inner colons, `A::B`, JSON, quotes, backticks, and payloads that genuinely begin with phrases such as `handle this exact question:`.

The general v2 parser consumes exactly one outer delegation envelope and never recursively strips prefix-like text from the resulting payload. On that fresh probe it improved from **31/64 under v1 to 64/64**, with **33 repairs and zero literal or schema regressions**. That passed the development gate and authorized exactly one no-inference replay over the already frozen 168 outputs.

| Metric | Raw Flash | Binder v1 | Binder v2 |
|---|---:|---:|---:|
| Historical strict exact | 145/168 (86.3%) | **161/168 (95.8%)** | 159/168 (94.6%) |
| Execution-equivalent exact | 147/168 (87.5%) | **163/168 (97.0%)** | 161/168 (95.8%) |
| Typed-contract exact | 148/168 (88.1%) | **166/168 (98.8%)** | 164/168 (97.6%) |
| Annotated literal typed exact | 85/104 (81.7%) | **103/104 (99.0%)** | 101/104 (97.1%) |
| Schema valid | 167/168 (99.4%) | 167/168 (99.4%) | 167/168 (99.4%) |

V2 repaired the original `target-claude-03` regression, but broke three previously correct `ask_claude` cases: one comma-delimited negative contrast (`Ask Claude, not web search: ...`) and two adverbial envelopes (`Ask Claude exactly: ...`). This produced three historical and three typed regressions despite a 104/104 character-copy invariant. The invariant again proves that output characters came from the request, not that the selected span boundary was correct.

The frozen-168 zero-regression gate therefore **failed**. The v2 result was frozen without adapting the algorithm to those evaluation cases, no second replay was performed, and **Flash is not authorized for the sacred 450** under the stated rule.

### Fresh 200-development / 100-holdout boundary study

Because both binder v1 and v2 had now been replayed on the Flash 168 and their failures inspected, that suite was formally demoted to a historical regression set. A new 300-case corpus was generated and frozen before binder v3 existed. Its 200-case development and 100-case holdout splits used disjoint outer-envelope templates, and the holdout scorer remained locked until the v3 source and passing development result were checksum-frozen.

| Split | Binder v2 exact | Binder v3 exact | V3 schema valid | V3 source-copy |
|---|---:|---:|---:|---:|
| Fresh development 200 | 62/200 (31.0%) | **200/200 (100.0%)** | 200/200 (100.0%) | 200/200 (100.0%) |
| Untouched holdout 100 | 6/100 (6.0%) | **85/100 (85.0%)** | 89/100 (89.0%) | 88/89 emitted (98.9%) |

The untouched holdout rejected v3. Its 15 misses comprised 9 search boundaries, 2 image boundaries, 2 shell boundaries, and 2 quoted Yuki write/append composites. Eleven were explicit abstentions; four selected the wrong source span. V3 caused no literal-exact regression relative to v2, but three v2 schema-valid calls became v3 abstentions, so the schema-regression gate also failed.

This is stronger evidence than another old-168 patch cycle: perfect development performance did not generalize across disjoint envelope wording. Binder v3 is frozen as a failed candidate and will not be tuned on the revealed holdout. Neither the historical 168 nor this now-revealed holdout may authorize the sacred 450 for a future candidate. **Flash remains unauthorized for the sacred 450.**

## Models not carried into later phases

This table prevents missing experiments from being misread as failures.

| Model | Last completed phase | Later phases not run | Reason |
|---|---|---|---|
| Original xLAM 1B | Broad 72 | Targeted 96 onward | 45.8% broad strict was not competitive. |
| xLAM-2 1B | Broad 72 | Targeted 96 onward | Hammer 1.5B was faster and 5.6 points better strict at similar memory. |
| Arch-Function 3B | Targeted 96 | Automatic selector onward | 10.4% targeted malformed rate and no efficiency advantage. |
| Granite 4.1 3B | Targeted 96 | Automatic selector onward | 59.4% targeted strict, poor schema/path fidelity, and 7.44 GB peak. |
| xLAM-2 3B | Automatic subsets | Phase 2 onward | Strong model, but Phase 2 was limited to Hammer 3B and Arch-Agent; not disproven. |
| Arch-Agent 3B | Phase 2 full 450 | Production delegation onward | Phase 2 exposed malformed/conversational risk; the next study changed architecture and focused on Hammer. |
| Hammer 2.1 3B | Phase 2 full 450 | Qwen full-450 pipeline | Explicitly excluded to test the attractive 1.5B efficiency path. Later used only in focused 25/100-case capacity probes. |
| Qwen3-14B surrogate | Production-interface 180 | Intended full 450 | It was temporary by design and was superseded when the exact Qwen3.8-27B endpoint was verified. |
| Qwen3.8-27B | Main-brain full 450 | Direct dispatcher | Its defined role was semantic delegation. It was not evaluated as a direct native all-18 tool caller. |
| GPT-OSS-20B | Stage A and direct full 450 | Binder follow-up | Direct exact score did not beat Hammer 7B; no binder experiment was requested. |
| Hammer 2.0 7B | Direct full 450 | Repaired full 450 | Frozen baseline preserved; targeted ontology repair missed the predeclared rerun gate by one case. |

## What changed in the prototype over time

The experiment remained isolated from production Yuki, but its harness evolved substantially:

1. **Live registry as source of truth.** All tests use the 18 registered Yuki tools and their real `META` definitions rather than a fake duplicate catalog.
2. **Strict schema adapter.** Required fields are corrected and `additionalProperties: false` is applied inside the prototype without rewriting Yuki.
3. **Direct validated argument mapping.** Calls bypass the old `arg` extraction bug and map structured fields directly to each existing tool's expected input shape.
4. **No execution benchmark contract.** Model research stops at parsing, schema validation, and scoring. The known shell bypass is therefore unreachable.
5. **Native output dialects.** xLAM, Hammer, Arch, Granite, and OpenAI-compatible native calls are normalized to one canonical internal call before validation.
6. **Bounded generation.** Maximum generation was raised to 256 tokens for legitimate long arguments, with immediate stop after one complete structured call.
7. **Two argument metrics.** Strict equality was retained, and execution-equivalent/preservation-aware scoring was added per tool.
8. **Failure-focused data.** Work progressed from 72 broad cases to 96 targeted cases and then the frozen 450-case reliability set.
9. **Schema-exposure experiments.** Oracle subsets proved schema competition; automatic subsets then exposed selector generalization failure.
10. **Production-style delegation.** A semantic main brain became responsible for domain and intent; Hammer remained a stateless dedicated dispatcher.
11. **Focused schema repair.** Capability descriptions were separated from persona/chat language for media and memory, then for OS versus Yuki files and replace versus append.
12. **Deterministic exact transport.** Source-span binding moved exact payload reproduction out of the models and exposed the need for per-field literal versus semantic contracts.
13. **Direct-caller alternatives.** GPT-OSS-20B tested remote speed and selection; Hammer 2.0 7B tested larger local function-calling capacity.
14. **Static-prefix caching.** Hammer 7B's repeated all-18 schema prefix was cached after output-equivalence verification, making the local full run practical.
15. **Frozen gates and checksums.** Later work predeclared smoke/regression gates, preserved raw generations, and refused expensive reruns when a gate was missed.
16. **Typed training-data boundary.** The field contract now drives a versioned Hammer 1.5B record format with model-visible prompt/target fields separated from audit metadata, explicit literal spans, semantic targets, and deterministic sacred-450 leakage checks.
17. **Cleaned Pilot v1.1.** The same 150-case composition now has zero leakage flags, more varied ontology language, explicit linguistic-review tags, structured `filename`/`content` targets that deterministically normalize to Yuki's unchanged legacy write/append schema, and a human-reviewed distinction between Yuki's own `process` metric and the system-wide `top` metric.
18. **Approved Pilot v1.2.** Human review kept 145 records, revised 5, rejected 0, and froze the resulting 150-record artifact for staged-scaling design. This approval did not authorize training, inference, tool execution, or a larger dataset in the same task.

## The most important lessons

### 1. Tool selection and argument transport are different problems

GPT-OSS directly selected 94.9% of tools but reached only 76.2% strict exact. Hammer 3B raised `see` selection from 68% to 96% without improving its old official target-string score. Larger models can fix semantics while still rewriting payloads.

### 2. Schema wording matters more than expected

Minimal capability descriptions moved Hammer 1.5B `recall` strict from 44% to 76% and eliminated 11 rejections. Hammer 7B's four-description ontology repair more than doubled targeted strict accuracy with no regressions. Some “model failures” were actually interface failures.

### 3. Schema reduction helps only if the selector generalizes

Oracle subsets improved both Hammer sizes and halved latency. The automatic selector looked excellent on 168 development cases but offered the right tool only 71.3% of the held-out 450. That architecture failed at the selector layer, not necessarily at the dispatcher layer.

### 4. Qwen3.8 is excellent at semantic routing, not lossless copying

Its 99.1% domain accuracy made the production interface viable, but 90.6% verbatim recall was not enough for exact-sensitive tools. Stronger copy instructions made performance worse.

### 5. Deterministic binding works, but only under an explicit contract

The binder achieved 95.1% literal-source accuracy on eligible frozen Hammer 7B cases. Uniform use still lowered official strict accuracy because fields like `image.prompt`, `see.target`, and `recall.topic` can be semantically normalized in the old benchmark. Production needs field-level `literal_source` versus `semantic_argument` behavior.

### 6. More capacity helps selectively

Hammer 3B materially fixed `see` selection but did not justify doubling latency/memory globally. Hammer 7B produced the strongest local strict result, yet still confused OS/Yuki storage and replace/append semantics. Capacity reduces but does not erase ontology/interface errors.

## Current state of the research

The current frozen facts are:

- `Hammer2.0-7b` direct baseline is permanently frozen at **412/450 tool selections** and **368/450 strict exact calls**. It must not be silently replaced by a resampled result.
- Deterministic source binding is validated as a character-preservation mechanism, not yet as a universal argument replacement policy.
- The official 450 gold remains unchanged, even where it conflicts with literal-source annotations.
- The first Hammer 7B ontology repair is promising and non-regressive but did not pass its full-rerun gate.
- No Yuki production tool, runtime loop, or UI has been integrated with this research architecture.
- No full-450 inference is currently pending.
- Qwen3.8-Flash passed the non-sacred 168-case direct-native gate at 86.3% strict exact. No sacred-450 Flash inference was run.
- Its frozen typed-binder v1 replay reached 98.8% typed exact overall and 99.0% on annotated literal cases, but one span-boundary regression failed the no-regression gate. A fresh 64-case probe produced a zero-regression v2 repair, yet the one permitted frozen-168 replay exposed three different Claude-envelope regressions. A subsequent clean 200/100 development-holdout study then rejected v3 at 85% holdout exact despite 100% development exact. Flash therefore remains unauthorized for the sacred 450.
- The typed contract is frozen as `yuki-argument-contract-v1`; it classifies 10 tool fields as literal source, 6 as semantic, and 2 tools as no-argument.
- Immutable Pilot v1.2 now follows that contract. All 150 records pass validation and sacred-450 leakage analysis without a review flag. It is approved as a design/reference artifact for a later 500-record staged-scaling task, not as a trained checkpoint or permission for automatic full-dataset generation.

The architecture supported by the evidence is now:

```text
raw user request
      │
      ├────────────── immutable source text ──────────────┐
      ↓                                                   │
Qwen3.8 semantic delegation                              │
      ↓                                                   │
selected live Yuki schema subset                         │
      ↓                                                   │
Hammer tool selection + semantic arguments               │
      ↓                                                   │
field contract: literal_source or semantic_argument      │
      ↓                                                   │
deterministic binder copies exact fields ◀───────────────┘
      ↓
strict schema validation
      ↓
final tool call
```

This remains a research architecture. The benchmark has never authorized tool execution.

## Present decision point

The next decision is not “which raw score wins?” It is which deployment tradeoff should be optimized:

- **Local strict accuracy:** Hammer 2.0 7B currently leads at 81.8%, but costs 6.24 seconds and 8.91 GB.
- **Local efficiency:** Qwen semantic delegation plus Hammer 1.5B keeps the dispatcher small and reaches 79.6% full-450 strict, but requires a remote main brain and exact-payload binding work.
- **Remote direct speed/selection:** GPT-OSS-20B is dramatically faster and chooses tools best, but reaches only 76.2% strict exact.
- **Hybrid research direction:** Qwen3.8 + Hammer 1.5B + typed deterministic binding has the cleanest separation of semantic reasoning, tool choice, and lossless transport, but the typed binder contract has not yet been evaluated end to end on an independent full set.

Before any new full-450 run, the field-level literal-versus-semantic interface and its training representation need human approval. The contract and 150-record pilot now exist; the next work is review and controlled dataset design, not another benchmark.

The research sequence is therefore:

1. **Completed:** freeze a typed `literal_source` versus `semantic_argument` contract for every Yuki field.
2. **Completed:** keep official gold intact while adding separate contract-aware scoring through checksum-bound sidecar annotations.
3. **Completed as a pilot only:** define the native Hammer record format and generate 150 leakage-checked examples spanning all 18 tools.
4. **Completed as Pilot v1.1:** resolve the remaining leakage flag, diversify ontology wording, annotate noisy-language coverage, and replace the model-visible pipe composite with structured write/append fields.
5. **Completed as Pilot v1.2:** apply five final human-review revisions and freeze the approved 150-record reference artifact.
6. **Next authorized design stage:** separately design and generate one 500-record staged batch while preserving the approved exposure and argument contracts.
7. **Not started:** LoRA/QLoRA Hammer 1.5B and candidate selection on separate validation/hard-dev sets.
8. **Deferred:** run one selected checkpoint on the sacred frozen 450 and compare it with the frozen reference systems.

No Hammer checkpoint was trained during the pilot stage, and no frozen 450-case inference was performed.

No additional random-model benchmark belongs between the interface-contract work and that specialized 1.5B experiment.

## Frozen source reports

- `new-models/FINAL-PHASE1-MODEL-COMPARISON.md`
- `oracle/ORACLE-SUBSET-REPORT.md`
- `automatic/AUTOMATIC-SCHEMA-SELECTION-REPORT.md`
- `phase2/PHASE2-REPORT.md`
- `production-delegation/PRODUCTION-DELEGATION-REPORT.md`
- `production-delegation/QWEN3.8-27B-HAMMER1.5B-FULL450-REPORT.md`
- `focused-repair/FOCUSED-REPAIR-FINAL-REPORT.md`
- `focused-repair/HAMMER15-VS-HAMMER3-SEE-CAPACITY-REPORT.md`
- `deterministic-binding/DETERMINISTIC-EXACT-PAYLOAD-REPORT.md`
- `deterministic-binding-capacity/HAMMER15-VS-HAMMER3-DETERMINISTIC-BINDING-REPORT.md`
- `gpt-oss-20b/GPT-OSS-20B-VS-QWEN38-STAGE-A-REPORT.md`
- `gpt-oss-20b/direct-native/GPT-OSS-20B-DIRECT-NATIVE-FULL450-REPORT.md`
- `hammer20-7b/direct-native/HAMMER20-7B-DIRECT-NATIVE-FULL450-REPORT.md`
- `hammer20-7b/binder-ontology-repair/HAMMER20-7B-BINDER-ONTOLOGY-REPAIR-REPORT.md`
- `qwen38-flash/direct-native/QWEN38-FLASH-DIRECT-NATIVE-SCREEN-168.md`
- `qwen38-flash/direct-native/typed-binder/FLASH168-TYPED-BINDER-REPLAY-REPORT.md`
- `qwen38-flash/direct-native/typed-binder/boundary-probe/LITERAL-BOUNDARY-DEVELOPMENT-PROBE-REPORT.md`
- `qwen38-flash/direct-native/typed-binder/FLASH168-TYPED-BINDER-v2-REPLAY-REPORT.md`
- `qwen38-flash/direct-native/typed-binder/boundary-corpus-v1/LITERAL-BOUNDARY-v3-DEV-HOLDOUT-REPORT.md`

All reported figures were transcribed from these frozen Markdown and machine-readable artifacts. No model generation, resampling, API call, or Yuki tool execution was performed to create this consolidated report.
