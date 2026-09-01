# Literal payload binder v3: fresh development and holdout study

Date: 2026-08-26

## Result

**Fresh-corpus gate: FAIL.** Binder v3 was developed against 200 fresh cases, frozen by source checksum, and only then evaluated on the untouched 100-case holdout. The historical Flash 168 was not used as an authorization set and the sacred 450 was not touched.

| Split | Binder v2 exact | Binder v3 exact | Schema valid | Source-copy invariant | V2 success → V3 failure |
|---|---:|---:|---:|---:|---:|
| Development 200 | 62/200 (31.0%) | **200/200 (100.0%)** | 200/200 (100.0%) | 200/200 (100.0%) | 0 |
| Holdout 100 | 6/100 (6.0%) | **85/100 (85.0%)** | 89/100 (89.0%) | 88/89 (98.9%) | 0 |

## Isolation protocol

1. All 300 requests and independent span annotations were generated and frozen before `typed_literal_binder_v3.py` existed.
2. Development and holdout use disjoint outer-envelope templates.
3. The development runner read only the 200 development cases and annotations.
4. Binder v3 and the passing development result were checksum-frozen before the holdout runner could execute.
5. Holdout predictions were frozen before holdout annotations were loaded for scoring.
6. Neither the historical 168 nor the sacred 450 participated in development or gating.

## Gate

- Development >=99% exact: **PASS**
- Holdout >=99% exact: **FAIL**
- Development/holdout schema validity 100%: **FAIL**
- Development/holdout source-copy invariant 100%: **FAIL**
- V2-success-to-v3-failure regressions: **0 dev, 0 holdout**

Passing this study demonstrates generalization across the fresh template-disjoint holdout. It does not itself run or rescore the sacred 450.

## Holdout failure analysis

The 15 misses were concentrated rather than random:

- `search`: **9**
- `image`: **2**
- `shell`: **2**
- Yuki write/append composites: **2**

V3 abstained on **11** cases and selected an incorrect source span on **4**. The unseen envelopes exposed lexical/template dependence in search phrasing, image phrasing, shell scope wording, and quoted composite payloads. This is the precise generalization failure the isolated holdout was intended to detect.

The source-copy invariant was **88/89 (98.9%) among emitted calls**. One incorrectly bounded composite did not satisfy the declared filename/content reconstruction invariant. Therefore even the mechanical copy-safety condition missed its required 100% gate.

Binder v3 is frozen as a failed research candidate. Its holdout misses must not be patched in place, and the 100-case holdout must not be reused to authorize a v4 candidate.

## Safety

- Models/API calls: **0**
- Yuki tool executions: **0**
- Historical Flash-168 replays: **0**
- Sacred-450 cases read for evaluation: **0**
