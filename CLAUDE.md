# CLAUDE.md

Guidance for Claude Code working in this repo. Keep it short and current.

## What this is

A local-first, single-user AI companion for Apple Silicon with swappable MLX,
Transformers, GGUF, OpenRouter, and OpenAI-compatible model backends plus Kokoro TTS.
Its CLI, Textual TUI, and FastAPI web UI share one runtime. The active persona is
**Yuki** — match that warm, informal tone in user-facing copy.

## Dev commands

Everything runs through **`uv`** and the `llama` launcher (symlinked onto PATH, so it
works from any directory — never hardcode `./` paths in messages to the user).

```bash
llama setup        # uv sync — create .venv + install deps
llama cli          # terminal chatbot (ms_llama.py)
llama tui          # Textual TUI workspace (interface/tui.py)
llama web          # FastAPI + Granian web UI on :7860 (interface/server.py)
llama test         # uv run pytest tests/ -v

env -u OPENROUTER_API_KEY uv run pytest tests/ -q  # deterministic mock suite
uvx ruff check .                          # lint — ruff is NOT a project dep, run via uvx
uv add / uv remove <pkg>                  # manage deps; never edit pyproject deps by hand
```

Baseline with the OpenRouter key unset: **153 pass / 4 fail**. The four failures are
pre-existing `test_tool_call_re_*` expectations around quote stripping. With a key exported,
the live catalog makes `test_select_model_no_models` a fifth environment-sensitive failure.

## Layout

- `ms_llama.py` — core chatbot engine (large; the live reply pipeline + ReAct loop).
- `episodic.py` — sqlite-vec episodic memory, embedded via OpenRouter.
- `retrieval/` — typed local/web retrieval, SQLite FTS5, rank fusion, provenance, and
  untrusted-content boundaries.
- `interface/` — `server.py` (FastAPI+Granian), `tui.py` (Textual), `*.html` web UIs.
- `tools/` — one tool per subfolder (see below).
- `personas/` — currently just `yuki.txt`.
- `data/` — runtime state (`memory.json` flat facts, `episodic_v2.db`, chats). Don't commit.
- `yuki/` — the bot's own scratch space. **Not junk** — leave it alone.
- `subprojects/` — side workflows (image_gen, img2img, finetune); kept out of the main tree.
- `docs/` — current architecture, dated handoffs, and clearly labeled archived history.

## Architecture notes

- **Tools** live in `tools/<name>/<name>.py` + `__init__.py` (which exports `tool_fn` +
  a `META` dict). `tools/__init__.py` assembles the registry (`TOOLS`, `REACT_TOOL_MAP`,
  `AUTO_DETECT_REGEX`, `_FACTUAL_TOOLS`, `TOOL_DESCRIPTIONS`); `ms_llama.py` does
  `from tools import (...)`. To add a tool: create the subfolder + `META`, append the name
  to `_TOOL_NAMES` — no `ms_llama.py` edits needed. Shared helpers in `tools/_helpers.py`
  (`YUKI_DIR`, `ALLOWED_COMMANDS`, ...). Tools needing bot state lazy-import `ms_llama`.
- **Single-user**: the multi-user UUID/slot system was removed. Memory is a flat
  `memory.json`. Don't reintroduce per-user partitioning or encryption-at-rest — the
  threat model is one real person on one machine.
- **Memory + retrieval**: flat hash-facts plus tool-gated `/recall` over rank-fused episodic
  summaries and Yuki-note FTS5 passages. `search`/`fetch` use the same source-aware boundary;
  web passages are evidence, never executable instructions. Current chat remains direct context.

## Conventions

- **uv only** — never `pip`, `pipx`, or `python -m venv`. Heavy/optional imports
  (numpy, soundfile, mlx) go **inside the function**, not at module top.
- **Keep linting clean** — fix unused imports / dead code immediately; no leftover warnings.
- **Fix at the chokepoint** — for a recurring bug, move the guard into the one function
  every path flows through, not N scattered patches.
- **No silent destructive cleanup** — never append `rm`/db-wipe to a test command; report
  artifact locations or ask first.
- **Bash cwd persists between calls** — don't bare `cd` into a subdir or vendored repo;
  prefer `git -C <path>` and subshells.
