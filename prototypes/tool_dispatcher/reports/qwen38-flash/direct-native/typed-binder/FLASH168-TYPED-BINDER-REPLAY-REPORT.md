# Qwen3.8-Flash frozen 168 typed-binder replay

Date: 2026-08-26

## Result

**Zero-inference binder threshold: FAIL.** Applying the typed deterministic binder to the already frozen 168 Flash calls raised historical strict exact from **145/168 (86.3%)** to **161/168 (95.8%)**. Fully covered typed-contract exact rose from **148/168 (88.1%)** to **166/168 (98.8%)**.

On the **104 independently annotated literal-source cases**, raw Flash was **85/104 (81.7%)** contract exact and bound Flash was **103/104 (99.0%)**. Historical strict on the same literal subset moved from **85/104 (81.7%)** to **101/104 (97.1%)**.

The replay repaired **17 historical strict failures** and **19 typed failures**, with **1 historical**, **1 typed**, and **0 schema regressions**.

## Controlled architecture

```text
immutable raw request ──────────────────────────────┐
        ↓                                           │
frozen Qwen3.8-Flash native all-18 tool call        │
        ↓                                           │
typed field contract                               │
        ├─ semantic/no-argument → unchanged         │
        └─ literal_source → deterministic raw span ◀┘
                         ↓
                  schema validation
                         ↓
                    scoring only
```

No model was called or loaded. Flash was not rerun, no output was resampled, the sacred 450 was not read as evaluation data, and no Yuki tool could execute.

## Isolation and annotation policy

- The 104 annotations were frozen before binder predictions.
- The annotation builder reads immutable requests and expected tool labels only; it does not read Flash outputs or old expected argument values.
- The gold-free binder input contains no expected tool, expected arguments, score, or literal annotation fields.
- The binder module has no filesystem, benchmark, annotation, or model access.
- Every emitted literal value must be a raw-request slice. Composite Yuki values may insert only the contract-declared `|` separator between independently copied components.
- Ambiguous or unbound spans cause abstention rather than guessing.

## Overall comparison

| Metric | Raw Flash | Bound Flash |
|---|---:|---:|
| Tool selection | 167/168 (99.4%) | 167/168 (99.4%) |
| Historical strict exact | 145/168 (86.3%) | **161/168 (95.8%)** |
| Execution-equivalent exact | 147/168 (87.5%) | **163/168 (97.0%)** |
| Typed-contract exact | 148/168 (88.1%) | **166/168 (98.8%)** |
| Schema valid | 167/168 (99.4%) | 167/168 (99.4%) |

Typed scoring now covers all 168 cases: 104 independent literal references plus the existing deterministic semantic/no-argument comparators. The two bound typed failures are `target-weather-06`, whose frozen Flash request received an upstream Alibaba 429 before inference, and `target-claude-03`, where the binder copied the broader phrase `handle this exact question: review the schema adapter` instead of the annotated payload `review the schema adapter`.

## Binder behavior

- Literal coverage: **104/104 (100.0%)**
- Typed semantic/no-argument passthrough: **63 cases**
- Binder ambiguities: **0**
- Total final-call absences: **1**; this is the pre-existing provider-error no-call, not a source-span ambiguity
- Character-copy invariant: **104/104 (100.0%)**

## Per-tool comparison

| Tool | Cases | Raw historical | Bound historical | Raw typed | Bound typed |
|---|---:|---:|---:|---:|---:|
| `ask_claude` | 20 | 11/20 (55.0%) | 19/20 (95.0%) | 11/20 (55.0%) | 19/20 (95.0%) |
| `calc` | 4 | 3/4 (75.0%) | 3/4 (75.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `fetch` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `hardware` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `image` | 4 | 3/4 (75.0%) | 4/4 (100.0%) | 3/4 (75.0%) | 4/4 (100.0%) |
| `read` | 28 | 28/28 (100.0%) | 28/28 (100.0%) | 28/28 (100.0%) | 28/28 (100.0%) |
| `recall` | 4 | 3/4 (75.0%) | 3/4 (75.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `remember` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `search` | 20 | 11/20 (55.0%) | 18/20 (90.0%) | 11/20 (55.0%) | 20/20 (100.0%) |
| `see` | 20 | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) |
| `shell` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `time` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `weather` | 20 | 18/20 (90.0%) | 18/20 (90.0%) | 19/20 (95.0%) | 19/20 (95.0%) |
| `yuki_append` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `yuki_delete` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `yuki_list` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |
| `yuki_read` | 12 | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) | 12/12 (100.0%) |
| `yuki_write` | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) | 4/4 (100.0%) |

The binder repaired all 19 destructive literal rewrites: 9 `search.query`, 9 `ask_claude.question`, and 1 `image.prompt`. It also broke the previously correct `target-claude-03` call by treating delegation scaffolding as payload. The 104/104 source-copy invariant therefore proves literal copying, not correct span selection. Semantic fields such as `weather.city`, `calc.expression`, `see.target`, `remember.fact`, and `recall.topic` were unchanged. Paths, commands, URLs, Yuki filenames, and Yuki contents remained exact and non-regressive.

## Decision

The proposed threshold was at least 95% contract exact on independently annotated literal cases with no regressions. Accuracy reached **103/104 (99.0%)**, but the replay caused **1 typed regression**. The accuracy condition passes; the no-regression condition fails. Therefore the frozen v1 replay does **not** pass the complete threshold.

This supports the architecture:

```text
Qwen3.8-Flash native all-18 selection
    → typed field contract
    → deterministic literal binder
    → schema validation
    → Yuki tool
```

Flash has not yet earned the sacred-450 run under the stated zero-regression rule. Freeze this result and repair the general external-agent payload boundary on a separate development probe first. No sacred-450 run was performed here.

## Frozen checksums

- Frozen Flash 168 report: `0c71c299cbd559a3c2a0f0dd67e90236620926cda1859db89072239a601baade`
- Literal annotations: `3a5ee9f97f6771b6977a3652418cc89729cbf4d8341791543261c02ec6f9662e`
- Broad 72: `9c4317eee6b271c50bf364f8f96a26b5904899e8a3cde83387a27de59de4c0ba`
- Targeted 96: `b416a10cec7b7fdac805b0114d8a2be642b4434ccd03d8e4d833df79b8da013d`
- Sacred 450 remains: `5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466`
