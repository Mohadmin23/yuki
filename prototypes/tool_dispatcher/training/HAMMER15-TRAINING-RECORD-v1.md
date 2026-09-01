# Hammer 1.5B Yuki Training Record v1

Status: frozen pilot format. This document defines a data interface, not a trained model or a production Yuki change.

## Purpose

One JSONL row represents one stateless Hammer dispatcher example:

```text
pre-rendered Hammer 2.1 prompt
  → one native tool-call array (or an empty rejection array)
```

The record has exactly three top-level fields:

```json
{
  "record_version": "hammer15-training-record-v1",
  "model_visible": {
    "prompt": "...",
    "assistant_target": "```\n[{...}]"
  },
  "metadata": {
    "...": "audit-only fields"
  }
}
```

The authoritative structural definition is `hammer15-training-record-schema-v1.json`.

## Model-visible contract

Only `model_visible.prompt` and `model_visible.assistant_target` may be exported to supervised training.

- `prompt` is produced by the existing `build_prompt()` Hammer 2.1 native path. It includes the stateless dispatcher instruction, the selected live Yuki schemas, Hammer's format instruction, the user request, and the assistant-role prefix.
- `assistant_target` is Hammer's native fenced-array form. A valid target starts with the native opening fence and contains exactly one compact call: ```` ```\n[{"name":"weather","arguments":{"city":"Tokyo"}}] ````.
- A rejection target is exactly ```` ```\n[] ````.
- The target deliberately ends as soon as the complete JSON array is emitted. It has no prose and no closing Markdown fence, matching the dispatcher's complete-JSON stop boundary.

The v1 prompt profile is `hammer21-yuki-typed-a1-ontology-v1`. It uses the live 18-tool registry plus only the selected prototype-facing description overlays:

- A1 media/memory descriptions for `see`, `image`, `remember`, and `recall`.
- Explicit OS-filesystem versus Yuki-store descriptions for `read` and `yuki_read`.
- Explicit replace versus append descriptions for `yuki_write` and `yuki_append`.
- Field-level transport text derived from `argument-contract-v1.json`.

No implementation function is called while prompts are rendered.

## Metadata boundary

`metadata` exists only for auditing, validation, splitting, leakage review, and scoring. It must never be concatenated into a prompt or target. `model_visible_only()` is the canonical safe export helper and fails if unexpected model-visible keys appear.

Important metadata includes:

- expected tool and arguments;
- typed argument mode and contract version;
- category, confusion family, and difficulty;
- exact literal spans or a semantic canonical target;
- offered tool subset and its schema checksum;
- deterministic generation provenance;
- sacred-450 leakage result;
- strict validation result.

## Literal-source annotations

Every literal target is independently recoverable from `metadata.raw_user_request`.

For a single-component field, one span records its component name, start offset, end offset, and exact text. The validator slices the immutable raw request and requires character equality with both the annotation and final argument.

For `yuki_write.filename_and_content` and `yuki_append.filename_and_content`, two ordered spans identify `filename` and `content`. Only the contract's `|` separator is inserted between them. A literal annotation that cannot reproduce the final argument is invalid.

Literal annotations are never inferred from old benchmark gold.

## Semantic annotations

Semantic records store a canonical value and explicit accepted alternatives. They are checked by the deterministic comparator declared in the typed contract:

- normalized text for city-like values;
- arithmetic-AST equivalence for calculations;
- exact hardware enums;
- referent normalization for `see.target` and `recall.topic`;
- explicit semantic references for `remember.fact`.

Optional semantic arguments may be omitted only where the live schema and typed contract permit it, such as a scene-wide `see` call.

## Validation and leakage gates

An accepted v1 row must pass all of these checks:

1. JSON Schema validity.
2. Reproducible Hammer prompt, target, tool ordering, and schema checksum.
3. Strict Hammer-native parse.
4. Live Yuki schema validation.
5. Typed argument-mode validation.
6. Literal source-span or semantic-target validation.
7. Reproducible sacred-450 leakage analysis.
8. Immutable Phase 2 checksum verification.

Leakage uses normalized exact matching, character similarity, token Jaccard, character-trigram Jaccard, and same-tool literal-payload similarity. It is deliberately conservative but cannot prove semantic independence; review flags require a human decision.

## Safety and scope

The builder is deterministic and model-free. It does not load Hammer, call Qwen, execute a Yuki tool, or modify the frozen benchmark. The v1 pilot is a review artifact and is not approval to train or scale the catalog.
