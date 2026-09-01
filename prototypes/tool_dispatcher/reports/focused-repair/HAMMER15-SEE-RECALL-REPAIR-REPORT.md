# Hammer 1.5B `see` + `recall` focused repair

Experiment A completed and frozen on 2026-08-21.

## Result

Minimal dispatcher-facing schema-description repair substantially improved both target tools. It fully repaired Hammer's `recall` rejection problem but only partially repaired `see`.

| Tool / metric | A0 frozen baseline | A1 schema descriptions | A2 schema + reject clarification |
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

No variant produced malformed output.

## Diagnosis

Every original target rejection was a valid Hammer-native empty array, not malformed generation. Required information was present in the frozen Qwen delegation, and `see.target` is optional.

The original `see` schema described Yuki's “real eye,” a webcam window, and what to do “when the user asks you.” That combines persona, UI behavior, and assistant policy without stating the core routing distinction: inspect existing visual input versus create a new image. Replacing it with a capability-only description removed 7 of 13 rejections. The remaining 6 A1 rejections show that ontology wording was important but not the whole problem.

The original `recall` schema asked a stateless dispatcher to decide whether information was absent from “the current chat,” then instructed it how to use summaries in an assistant reply. Frozen Qwen requests instead consistently asked to retrieve previously stored user information. Replacing the description with the direct persistent-memory retrieval capability eliminated all 11 rejections. This is strong evidence that description mismatch—not model capacity—was the dominant `recall` failure cause.

## Rejection instruction

A2 clarified that Hammer should select a listed capability whenever it can fulfill the delegated action and reject only when none applies or a genuinely required argument is missing.

This produced no accuracy gain over A1. It increased `see` rejections from 6 to 7 and reduced `see` schema validity from 76% to 72%. `recall` was unchanged. Therefore the rejection-prompt change is not selected.

## Regression checks for selected A1

| Tool / metric | Frozen baseline | A1 repair |
|---|---:|---:|
| `image` tool selection | 88% | **100%** |
| `image` strict exact | 48% | **56%** |
| `image` schema valid | 88% | **100%** |
| `remember` tool selection | 100% | 100% |
| `remember` strict exact | 76% | 76% |
| `remember` preservation-aware exact | 96% | 96% |
| `remember` schema valid | 100% | 100% |

The repair did not degrade either competing tool. It improved `image` rejection behavior and left `remember` exactly unchanged.

## Selected repair

A1—dispatcher-facing schema descriptions only—is frozen as the Experiment A winner. It will be used for Experiment B's downstream Hammer comparison. Production Yuki metadata and implementations remain untouched.

## Safety

- Exact frozen 25 `see` and 25 `recall` cases were used for every target variant.
- Exact frozen 25 `image` and 25 `remember` cases were used only after variant selection.
- One stateless Hammer generation per case.
- No retry voting, self-correction, or hidden gold hints.
- No Yuki tool execution capability or attempt.
- Hammer 3B was not loaded.

