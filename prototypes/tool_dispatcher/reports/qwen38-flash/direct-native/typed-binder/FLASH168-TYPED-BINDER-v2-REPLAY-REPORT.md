# Qwen3.8-Flash frozen 168 typed-binder v2 replay

Date: 2026-08-26

## Result

**Authorization gate: FAIL.** The fresh 64-case development probe first improved from **31/64** under frozen binder v1 to **64/64** under the general boundary parser, with zero regressions. Only then was binder v2 replayed once over the immutable Flash 168 calls.

Historical strict exact rose from raw **145/168 (86.3%)** to v1 **161/168 (95.8%)** and v2 **159/168 (94.6%)**. Typed-contract exact rose from raw **148/168 (88.1%)** to v1 **166/168 (98.8%)** and v2 **164/168 (97.6%)**.

On the 104 independently annotated literal fields, v2 reached **101/104 (97.1%)** typed exact, with **104/104** emitted values satisfying the character-for-character source-copy invariant.

## Raw vs frozen binder versions

| Metric | Raw Flash | Binder v1 | Binder v2 |
|---|---:|---:|---:|
| Historical strict exact | 145/168 (86.3%) | 161/168 (95.8%) | **159/168 (94.6%)** |
| Execution-equivalent exact | 147/168 (87.5%) | 163/168 (97.0%) | **161/168 (95.8%)** |
| Typed-contract exact | 148/168 (88.1%) | 166/168 (98.8%) | **164/168 (97.6%)** |
| Annotated literal typed exact | 85/104 (81.7%) | 103/104 (99.0%) | **101/104 (97.1%)** |
| Schema valid | 167/168 (99.4%) | 167/168 (99.4%) | 167/168 (99.4%) |

V2 repaired **17** raw historical failures and **19** raw typed failures. Relative to v1 it repaired **1** historical and **1** typed cases, with **3 historical**, **3 typed**, and **0 schema regressions**.

## General boundary rule

The binder parses one outer delegation envelope and then stops. A structural delimiter or grammatical action-plus-transport phrase identifies the boundary. Text after that boundary is copied from the immutable request; prefix-like words inside it are never recursively stripped. Matching quotes/backticks are removed only when they wrap the entire payload. Ambiguous boundaries cause abstention rather than guessing.

This is why these remain different:

```text
Ask Claude this exact question: why does this fail?
                                └─ payload

Ask Claude: handle this exact question: why does this fail?
            └──────────────────────────────────────────── payload
```

## Gate and safety

- Literal typed exact >=95%: **PASS**
- Raw-to-v2 regressions: **3 historical, 3 typed, 0 schema**
- V1-to-v2 regressions: **3 historical, 3 typed, 0 schema**
- Models/API calls: **0**
- New Flash generations: **0**
- Yuki tool executions: **0**
- Sacred-450 evaluation cases run: **0**

The zero-regression evidence gate fails. Flash is not authorized for the sacred-450 evaluation.

## Frozen source hashes

- Original v1 freeze manifest: `3359e7c9aa4d1b9afa36b671eeb11a5ce700cbed1b4398e515f28f8366c774cf`
- Fresh probe comparison: `3ef6d50a53a991efeb5a43bba7ed44cd9a0ee52c5e15e205a1d8c2ef28aa42d9`
- Binder v2 source: `6b1cc7da143b8d8986d0063ae3c06343719b00a59850498f58340695ba85174c`
- Frozen Flash 168 report: `0c71c299cbd559a3c2a0f0dd67e90236620926cda1859db89072239a601baade`
- Independent literal annotations: `3a5ee9f97f6771b6977a3652418cc89729cbf4d8341791543261c02ec6f9662e`
- Sacred 450 (unchanged): `5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466`
