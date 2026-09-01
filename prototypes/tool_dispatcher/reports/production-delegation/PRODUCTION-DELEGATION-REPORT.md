# Yuki Production-Interface Delegation Experiment

Date: 2026-08-21  
Status: completed isolated research run  
Main brain: **Qwen3-14B-MLX-4bit temporary main-brain surrogate**

## Executive result

This experiment is useful but not a production verdict. The intended Qwen3.8-27B main brain was not locally available, so every Stage A result below is explicitly a **temporary Qwen3-14B-MLX-4bit surrogate** result.

The surrogate materially outperformed the failed raw-text deterministic selector on the same 180 Phase 2 cases: correct domain/tool coverage rose from 74.4% to 87.8%, and Hammer 3B end-to-end strict exact calls rose from 61.1% to 72.2%. It did not solve routing reliability: 9 delegations were malformed, 13 more valid delegations chose the wrong domain, and exact `verbatim` recall averaged 84.2%.

Hammer 1.5B slightly beat Hammer 3B on the frozen production interface: 73.3% versus 72.2% strict exact calls. The paired difference was only 7 cases versus 5 (`p=0.774`), so there is no evidence of a real accuracy advantage for either size on this set. Operationally, 1.5B was clearly lighter: 2.71 s and 3.46 GB versus 5.21 s and 6.53 GB.

The dominant remaining failures moved upstream. Stage A was the primary attribution for 37/48 strict 1.5B failures and 39/50 strict 3B failures. `verbatim` is essential: removing it collapsed 1.5B strict exact calls from 73.3% to 25.6%.

No Yuki tool was executed. The benchmark had no execution API, both Hammer reports contain the exact same frozen Stage A checksum, and models were loaded one at a time.

## Controls and dataset

- Canonical source: frozen 450-case Phase 2 dataset, SHA-256 `5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466`.
- Derived version: `production-delegation-source-v1`, SHA-256 `ea6711ce0548d3771166f8f2f9a5e59ebd73acb14a9ddf7ecbf2f6589f1744c3`.
- Selected set: deterministic `fallback-stratified-180`, exactly 10 cases per tool, preserving source IDs and original gold calls.
- The 18-case smoke reached only 88.9% strict delegation validity after the prompt adjustment, so the predeclared 180-case fallback was used instead of all 450.
- Stage A: one stateless generation per original request, thinking disabled, 160-token ceiling, complete-JSON stopping, no output repair.
- Stage B: official/native Hammer prompt and fenced tool-call array, one stateless generation per valid delegation, strict normalization and schema validation, no repair.
- The Stage A common-prefix cache reused only the identical rendered system-prefix computation. Previous request tokens and generations were trimmed before every case.
- This tool-centric set contains no no-tool examples and no conversation-context cases; it cannot measure those production decisions.

## Stage A — temporary surrogate delegation

| Metric | Result |
|---|---:|
| Valid delegation | 171/180 (95.0%) |
| Malformed delegation | 9/180 (5.0%) |
| Exact domain, all cases | 158/180 (87.8%) |
| Exact domain, valid delegations | 92.4% |
| Deterministic semantic-request score | 61.7% |
| Average `verbatim` recall | 84.2% |
| Average `verbatim` precision | 61.7% |
| Whole source request reproduced in delegation | 7/180 |
| Final implementation-tool leakage | 0.0% |
| Average / p50 / p95 latency | 5.54 s / 5.35 s / 7.39 s |
| Average generated tokens / tok/s | 23.9 / 6.68 |
| Peak MLX memory | 8.35 GB |

| Expected domain | N | Valid | Exact domain | Verbatim recall | Verbatim precision |
|---|---:|---:|---:|---:|---:|
| computer | 30 | 80.0% | 66.7% | 83.3% | 63.3% |
| external_agent | 10 | 100.0% | 100.0% | 80.0% | 80.0% |
| information | 50 | 98.0% | 94.0% | 94.0% | 50.0% |
| media | 20 | 100.0% | 85.0% | 80.0% | 30.0% |
| memory | 20 | 100.0% | 100.0% | 70.0% | 70.0% |
| yuki_files | 50 | 96.0% | 88.0% | 83.0% | 78.0% |

The nine format failures were four scalar `verbatim` values instead of arrays, four extra closing braces, and one invalidly escaped shell string. They were rejected unchanged. In seven valid cases, the surrogate copied the whole source request into `request` or `verbatim`; Hammer therefore saw text identical to the raw request through Stage A output even though the Stage B harness never injected the raw request separately. The strict semantic-request cue scorer is intentionally transparent but conservative: 42 cases that it marked false still became strict-correct Hammer 1.5B calls, so its 61.7% score should not be read as a calibrated semantic-quality estimate.

The sharpest Stage A weaknesses were preservation cases (59.4% domain accuracy), the computer domain (66.7%), normal `read` formatting, Yuki write/append payload splitting, and recall/question span selection. Zero generated semantic requests leaked implementation-level tool names.

## Stage B and end-to-end results

All accuracy below is end to end over the same 180 original requests. Invalid or wrong-domain Stage A output remains a failure; conditional columns isolate Hammer after a correct domain was supplied.

| Metric | Hammer 1.5B | Hammer 3B |
|---|---:|---:|
| Tool selection | 83.9% | 84.4% |
| Tool selection, correct domain | 95.6% | 96.2% |
| Strict arguments, correct tool | 87.4% | 85.5% |
| Preservation-aware arguments, correct tool | 88.1% | 86.2% |
| Strict exact call | **73.3%** (132/180) | 72.2% (130/180) |
| Preservation-aware exact call | **73.9%** | 72.8% |
| Strict exact call, correct domain | **83.5%** | 82.3% |
| Schema valid | 90.0% | 90.0% |
| Malformed / conversational Hammer output | 0% / 0% | 0% / 0% |
| Dispatcher average / p50 / p95 | 2.71 s / 2.73 s / 3.38 s | 5.21 s / 5.28 s / 6.59 s |
| Generated tok/s | 18.51 | 9.49 |
| Peak MLX memory | 3.46 GB | 6.53 GB |
| End-to-end average latency, resident models | 8.11 s | 10.49 s |

MLX memory was measured with one model loaded at a time and must not be interpreted as a measured simultaneous-residency total. Model load time is excluded from per-case latency.

### Accuracy by case category

| Category | N | 1.5B strict exact | 3B strict exact |
|---|---:|---:|---:|
| adversarial | 18 | 72.2% | 77.8% |
| confusion | 36 | 77.8% | 75.0% |
| normal | 58 | 86.2% | 81.0% |
| paraphrase | 36 | 69.4% | 72.2% |
| preservation | 32 | 50.0% | 50.0% |

Excluding the 18 tagged adversarial/ambiguous cases does not improve the result: strict exact calls are 73.5% for 1.5B and 71.6% for 3B. Preservation cases are the weakest category for both at 50.0%.

### Primary strict-failure decomposition

| Primary attribution | Hammer 1.5B | Hammer 3B |
|---|---:|---:|
| dispatcher_argument_rewrite | 1 | 1 |
| dispatcher_wrong_argument | 5 | 4 |
| dispatcher_wrong_tool | 5 | 6 |
| main_brain_lost_verbatim | 15 | 17 |
| main_brain_wrong_domain | 13 | 13 |
| malformed_delegation | 9 | 9 |
| Ambiguous/adversarial failure (secondary tag) | 5 | 4 |

This precedence attributes lost exact text to Stage A before assigning a downstream wrong-argument category. It is an auditable primary attribution, not proof that every lost-verbatim case alone caused the final failure.

## Same-set selector comparison

The old deterministic selector can be compared fairly on these exact 180 IDs using the frozen Phase 2 Hammer 3B report.

| Same 180 cases | Old raw-text selector | Temporary surrogate `domain_hint` | Change |
|---|---:|---:|---:|
| Exact domain | 70.6% | 87.8% | +17.2 pp |
| Correct tool available | 74.4% | 87.8% | +13.3 pp |
| Hammer 3B strict exact, full pipeline | 61.1% | 72.2% | +11.1 pp |

Semantic main-brain delegation therefore substantially improves held-out schema selection and complete calls. It does **not** fully solve selector generalization: 12.2% of cases still lacked the correct generated domain, and the current result pays a 5.54 s Stage A inference cost.

## Hammer 1.5B ablations

| Domain source | Dispatcher input | Strict exact | Preservation-aware exact | Avg dispatcher |
|---|---|---:|---:|---:|
| generated | semantic + `verbatim` (primary) | 73.3% | 73.9% | 2.71 s |
| generated | semantic only, no `verbatim` | 25.6% | 26.7% | 2.24 s |
| generated | raw user request | 63.3% | 66.1% | 2.60 s |
| gold/oracle | raw user request | 72.8% | 75.6% | 2.94 s |

- `verbatim` is decisively useful: primary beat no-`verbatim` on 87 discordant strict cases and lost only 1 (`p<1e-23`).
- Semantic + `verbatim` beat raw text under the same generated domains by 10.0 points strict; paired discordance was 26 wins to 8 losses (`p=0.0029`).
- Replacing generated domains with gold domains raised raw-text strict accuracy from 63.3% to 72.8%; all 17 discordant strict cases favored gold.
- Primary semantic delegation and same-set gold-domain/raw-user control were effectively tied (73.3% vs 72.8%, paired `p=1.0`). That does not mean Stage A domains were oracle-perfect; semantic normalization and explicit exact strings recovered different cases while malformed/wrong domains lost others.

## Oracle reference and gap

These historical oracle numbers use the different 96-case targeted suite. They are included because they were requested, but subtracting them from this 180-case result is descriptive—not a paired reliability estimate.

| Model | Production interface, 180 | Production, correct domain | Targeted logical oracle, 96 | Targeted focused oracle, 96 |
|---|---:|---:|---:|---:|
| Hammer 1.5B | 73.3% | 83.5% | 71.9% | 76.0% |
| Hammer 3B | 72.2% | 82.3% | 85.4% | 87.5% |

The same-set 1.5B gold-domain/raw-user control is the stronger local comparison: it scored 72.8%, essentially the same aggregate as the primary interface but with substantially different per-case outcomes. No same-set 3B gold-domain ablation was run.

## Answers and next step

1. **Does semantic main-brain delegation solve selector generalization?** Partly. It improves correct schema coverage by 13.3 points on the same cases, but 12.2% still lack a correct domain.
2. **How accurate is `domain_hint`?** 87.8% overall and 92.4% after valid JSON.
3. **How reliable is `verbatim`?** 84.2% average recall and 61.7% precision. It is not yet reliable enough, but the ablation proves the field itself is essential.
4. **How close is production routing to oracle subsets?** For 1.5B it matches the same-set gold-domain/raw control in aggregate; comparison to the historical 96-case oracle is not apples-to-apples. Stage A errors are now the dominant gap.
5. **Does Hammer 1.5B become viable?** It becomes the stronger efficiency candidate in this isolated interface test: slightly higher measured accuracy than 3B at about half the latency and memory. 73.3% is still not production-grade evidence.
6. **Does Hammer 3B justify its cost here?** No measured accuracy benefit in this run. It remains a valid comparator, but this evidence alone does not justify its roughly 2× dispatch latency/memory.
7. **Can the old raw-text selector be removed?** The main-brain route is much stronger on the same subset, but replacement should wait for the intended 27B rerun, full 450 evaluation, and no-tool/context coverage.
8. **Which component dominates failures?** Stage A: malformed JSON, wrong domains, and lost exact strings account for roughly four-fifths of strict failures under the primary precedence.
9. **What next?** Rerun only Stage A with the intended Qwen3.8-27B using this same derived dataset/version and contract, freeze its new delegations, then reuse the exact Hammer 1.5B/3B evaluation. If its smoke is healthy, run all 450. Add separate no-tool and real conversation-context cases before any production recommendation.

No Yuki production routing was modified, and no integration recommendation is made.

## Artifacts

- `production-delegation-comparison-v1.json` — consolidated machine-readable results, paired analyses, prior-selector comparison, oracle reference, checksums, and failure examples.
- `production-delegation-artifact-manifest-v1.json` — SHA-256 manifest for datasets, reports, harness sources, and failure JSONL files.
- `stage-a/` — frozen temporary-surrogate delegations and raw generations.
- `stage-b/` — primary Hammer reports and failure JSONL.
- `ablations/` — three Hammer 1.5B control reports and failure JSONL.
