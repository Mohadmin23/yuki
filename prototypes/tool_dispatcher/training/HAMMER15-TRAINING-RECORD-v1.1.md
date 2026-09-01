# Hammer 1.5B Yuki Training Record v1.1

Status: cleaned pilot format pending human approval. Pilot v1 remains immutable and separately reproducible.

## What changed from v1

The model-visible/native representation for `yuki_write` and `yuki_append` now uses two fields:

```json
{
  "name": "yuki_write",
  "arguments": {
    "filename": "pipes.note",
    "content": "left|right"
  }
}
```

The training-only adapter validates that shape, forbids `|` inside `filename`, then deterministically maps it to the unchanged live Yuki schema:

```json
{
  "tool": "yuki_write",
  "arguments": {
    "filename_and_content": "pipes.note|left|right"
  }
}
```

The live helper uses `split("|", 1)`, so every pipe after the first remains part of `content`. The structured model interface therefore removes the ambiguous model-generation step while retaining production compatibility. No production tool metadata, function, or executor was changed.

## Record boundary

Only these fields are model-visible:

```text
model_visible.prompt
model_visible.assistant_target
```

Everything under `metadata`, including `runtime_arguments`, span annotations, semantic targets, feature tags, leakage results, and validation results, is audit-only.

Pilot v1.1 adds:

- `runtime_arguments`: the deterministic call shape accepted by the live Yuki registry;
- `training_argument_profile`: `structured-yuki-file-content-v1`;
- `linguistic_features`: human-review tags for controlled wording diversity;
- parent-format provenance;
- a new prompt/schema checksum bound to the structured training schema.

## Dual validation

For write/append records, validation occurs twice without execution:

```text
Hammer native target
    ↓ parse
structured training schema: filename + content
    ↓ deterministic adapter
live Yuki schema: filename_and_content
```

The adapter result must exactly match `metadata.runtime_arguments`, and the normalized call must pass the unchanged live registry. Every filename and content value must independently match its source span character-for-character.

All other tools retain their Pilot v1 typed argument behavior.

The Pilot v1.1 dispatcher-facing hardware schema also defines two otherwise ambiguous enum values using the live implementation as authority:

- `process` means Yuki's own current LLM/runtime process only.
- `top` means the system-wide top-five process lists by CPU and RAM.

This clarification is prototype-facing. It does not modify the live hardware tool.

## Linguistic diversity annotations

Pilot v1.1 records may declare human-review tags such as `typo`, `terse_fragment`, `casual`, `pronoun_context`, `negative_contrast`, and `multi_clause`. These tags are not model-visible and do not change scoring. Their only purpose is to make generator habits measurable before scaling.

## Versioning and safety

- Pilot v1 artifacts are not overwritten.
- The typed tool contract remains `yuki-argument-contract-v1`; only the training-facing composite representation changed.
- The frozen 450 remains read-only and checksum-gated.
- The builder performs no inference and has no tool-execution path.
- The v1.1 format is not permission to train or scale the dataset.
