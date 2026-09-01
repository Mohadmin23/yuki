# Literal payload-boundary development probe

Date: 2026-08-26

## Result

**Probe gate: PASS.** The frozen v1 binder scored **31/64 (48.4%)**. The general v2 boundary parser scored **64/64 (100.0%)**, repairing **33** cases with **0 literal** and **0 schema regressions**.

`ask_claude` improved from **10/40 (25.0%)** to **40/40 (100.0%)**. The 24 non-Claude literal controls improved from **21/24** to **24/24**.

## Probe construction

The 64 fresh, non-sacred cases contain:

- 40 `ask_claude` boundary cases.
- 24 controls across search, image, filesystem paths, URLs, commands, and Yuki filenames/content.
- Direct colon and em-dash envelopes.
- `Ask Claude to ...` and `Have Claude ...` verb payloads without delimiters.
- Nested strings such as `Ask Claude: Ask Claude to retry ...`.
- Payloads intentionally beginning with `handle this exact question:`.
- Inner colons, `A::B`, JSON, quotes, backticks, punctuation, and prefix-like path/content text.

Annotations were independently frozen before either scored probe run and were unavailable to both binders.

## General v2 boundary algorithm

The repair is an outer-envelope parser, not a benchmark phrase replacement:

1. Recognize one top-level Claude delegation marker near the start.
2. Consume either an immediate delimiter (`:` or `—`) or a grammatical transport envelope composed of an action plus a transport noun such as `question`, `message`, `request`, or `task`.
3. If no structural delimiter exists, treat the direct verb phrase after Claude as payload.
4. Stop parsing the envelope after that single boundary. Never recursively strip prefix-like text inside the payload.
5. Remove matching outer quotes/backticks only when they delimit the entire payload; preserve inner quotes and punctuation.
6. Copy the final span from the immutable request and validate the resulting call.
7. Abstain if a unique boundary cannot be established.

This distinguishes:

```text
Ask Claude this exact question: why does this fail?
                                └─ payload

Ask Claude: handle this exact question: why does this fail?
            └──────────────────────────────────────────── payload
```

## Gate

| Check | Result |
|---|---:|
| Literal exact | **64/64** |
| Schema valid | **64/64** |
| Binder coverage | **64/64** |
| Raw-source invariant | **64/64** |
| Literal regressions | **0** |
| Schema regressions | **0** |

The probe therefore authorizes exactly one model-free v2 replay of the already frozen Flash 168 outputs. It does not authorize model inference or the sacred 450.

## Frozen inputs

- Probe SHA-256: `516a1214a68f88be7f4fc2b8fbb2d9711e999cf87a94bf12cf07b1eed2eec560`
- Annotation SHA-256: `81ec7dbcf23fa828cb60bb0d0afa5d35ba892ae9bb2c789bee9ce236ee14d7a6`
- V1 comparison SHA-256: `e3c86fccea2ecd086d6e4d82eb951f9080acd3c4b5799dc07b3f222aab87774f`
- V2 comparison SHA-256: `68656ddd341ba2de0e21b53bddbafb38ab6f54636c6c0d10dd9b2204d2839698`
- Combined comparison SHA-256: `3ef6d50a53a991efeb5a43bba7ed44cd9a0ee52c5e15e205a1d8c2ef28aa42d9`
