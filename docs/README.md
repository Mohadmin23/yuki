# Yuki documentation

This index separates documentation that describes the live system from dated
inspection reports and archived design history. Start with the project
[`README`](../README.md) for the public overview.

## Current architecture

| Document | Status | Purpose |
| --- | --- | --- |
| [`RAG-PHASE1.md`](RAG-PHASE1.md) | Current | Typed local/web retrieval, provenance and prompt-injection boundaries, hybrid recall, and deliberate Phase 1 limits |
| [`TOOL-ROUTING-ARCHITECTURES.md`](TOOL-ROUTING-ARCHITECTURES.md) | Current | Direct and dedicated-dispatcher routing, validation boundary, literal-source policy, autonomous routing, and TUI controls |
| [`AGENTS.md`](../AGENTS.md) | Current contributor reference | Repository rules, runtime-state warnings, tool-registry conventions, and environment coupling |
| [`CLAUDE.md`](../CLAUDE.md) | Current concise agent guide | Short operational companion to `AGENTS.md` |

Runtime code remains authoritative when a document and implementation disagree.
The most important implementation entry points are `ms_llama.py`,
`interface/tui.py`, `tool_routing/`, `tools/`, and `episodic.py`.

## Historical handoffs

These reports are detailed snapshots from specific points in the project. They
are useful for understanding why the architecture changed, but they are not
living specifications.

| Document | Snapshot | Notes |
| --- | --- | --- |
| [`ARCHITECTURE-HANDOFF-FOR-CHATGPT.md`](handoffs/ARCHITECTURE-HANDOFF-FOR-CHATGPT.md) | 2026-08-22 | Read-only whole-project inspection before the production dispatcher integration and later continuity fixes |
| [`TUI-UX-HANDOFF-FOR-CHATGPT.md`](handoffs/TUI-UX-HANDOFF-FOR-CHATGPT.md) | 2026-08-22 | Pre-redesign TUI audit; many recommendations were subsequently implemented |
| [`NOTE-FROM-CLAUDE.md`](handoffs/NOTE-FROM-CLAUDE.md) | 2026-06-24 | Personal collaboration handoff from the previous coding agent |

## Archived design history

Archived documents are intentionally preserved, not endorsed as current work.
They should not be used as implementation instructions.

| Document | Why it is archived |
| --- | --- |
| [`TODO-function-calling.md`](archive/TODO-function-calling.md) | The proposed native/canonical function-calling work has been implemented and superseded by the strict shared routing boundary |
| [`TODO-memory-security.md`](archive/TODO-memory-security.md) | Describes a deleted multi-user slot, identity, and unlock architecture; Yuki is intentionally single-user |
| [`system-note-legacy.txt`](archive/system-note-legacy.txt) | Generated inventory of an older prompt/tool loop that has drifted from the current runtime |

## Tool-dispatcher research

The research lab has its own detailed documentation:

- [`prototypes/tool_dispatcher/README.md`](../prototypes/tool_dispatcher/README.md)
- [`YUKI-TOOL-DISPATCHER-ALL-MODELS-PHASES-REPORT.md`](../prototypes/tool_dispatcher/reports/YUKI-TOOL-DISPATCHER-ALL-MODELS-PHASES-REPORT.md)

Raw reports remain versioned because the experiments depend on frozen inputs,
outputs, and checksums. They are research evidence rather than normal product
documentation.

## Documentation rules

- Put current system behavior in a current architecture document or the main
  README.
- Put point-in-time inspection reports under `docs/handoffs/` and date them.
- Put superseded proposals and generated snapshots under `docs/archive/` with a
  visible warning.
- Keep benchmark-specific documentation with the benchmark that owns it.
- Do not duplicate runtime truth when a direct link to the responsible module is
  clearer.
