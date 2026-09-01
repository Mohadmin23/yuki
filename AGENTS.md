# AGENTS.md

Guide for **any** coding agent (Codex, Claude Code, Cursor, …) working in this repo.
Read this first. It's the handoff doc — kept short, blunt, and current.

> There is also a `CLAUDE.md` (Claude Code reads it by convention). The two overlap;
> this file is the fuller one and is the source of truth. If you change a rule here,
> mirror it there.

---

## 1. What this is

A **local, single-user voice assistant for Apple Silicon**: Qwen2.5 (MLX) + Kokoro TTS,
with three front-ends — a CLI, a Textual TUI, and a FastAPI web UI. The default persona is
**Yuki**, a playful anime character. **Match that warm, informal tone in any user-facing
copy** (prompts, errors, help text). Corporate voice is wrong here.

It is *single-user by design*. The threat model is one real person on one machine. Don't
reintroduce per-user accounts, UUID slots, partitioning, or encryption-at-rest — that whole
system was deliberately deleted (see §9).

---

## 2. ⚠️ State of the tree right now (READ THIS BEFORE YOU TOUCH ANYTHING)

The working tree has a **large in-flight refactor that is NOT committed**. As of the handoff:

- `ms_llama.py` — ~1150 net lines changed (a near-rewrite of the reply pipeline + ReAct loop).
- `interface/server.py`, `episodic.py`, `tests/test_ms_llama.py` — substantially modified.
- Untracked new work: `interface/tui.py`, `tools/see/`, three `tests/_smoke_*.py` scripts,
  `docs/TODO-function-calling.md`.
- Deleted: `personas/coder.txt`, `personas/therapist.txt` (intentional — Yuki is the only persona now).

**What this means for you:**
- This diff is the previous owner's unfinished work. **Don't blow it away, don't `git checkout`
  it, don't "clean it up" by reverting.** If you need a clean base, `git stash` (don't discard).
- The suite is green at the documented baseline *with this diff applied* (see §8), so the
  refactor is coherent, not broken.
- Before starting your own feature, decide with the user whether to commit this refactor first.
  Tangling new work into this diff makes it un-reviewable.
- Keep your own edits separable from it. The filesystem/lint cleanup that produced this very
  file was deliberately kept tiny and out of the big modified files for that reason.

---

## 3. Golden rules (non-negotiable)

1. **`uv` only.** Never `pip`, `pipx`, `python -m venv`, or `python -m pip`. Manage deps with
   `uv add` / `uv remove` — never hand-edit `[project.dependencies]` in `pyproject.toml`.
2. **Lazy-import heavy/optional packages** (numpy, soundfile, mlx, torch, sqlite_vec, openai)
   **inside the function**, not at module top. Keeps startup fast and import errors local.
3. **Keep linting clean.** `uvx ruff check .` should stay quiet. Fix unused imports / dead code
   immediately. The owner actively checks for warnings. (But read §6 about the re-export trap
   before you ever run `--fix`.)
4. **Fix at the chokepoint.** For a recurring bug, move the guard into the one function every
   path flows through — don't scatter N patches.
5. **No silent destructive cleanup.** Never append `rm`, a db-wipe, or `git reset --hard` to a
   command "to tidy up." Report artifact locations or ask first. `data/`, `yuki/`, and `output/`
   hold real runtime state.
6. **Bash cwd persists between calls.** Don't bare `cd` into a subdir or a vendored repo. Prefer
   `git -C <path>` and subshells `( cd x && … )`.
7. **`yuki/` is not junk.** It's Yuki's own scratch space (notes, images she saves). Leave it.

---

## 4. Dev commands

Everything runs through **`uv`** and the **`llama`** launcher (a bash script symlinked onto
`PATH`, so `llama` works from any directory — never write `./llama` in messages to the user).

```bash
llama setup     # uv sync — create .venv + install deps
llama cli       # terminal chatbot  → ms_llama.py
llama tui       # Textual TUI workspace → interface/tui.py
llama web       # FastAPI + Granian web UI on http://localhost:7860 → interface/server.py
llama test      # uv run pytest tests/ -v

uv run pytest tests/test_ms_llama.py -q   # the main unit suite (mock-based, no network)
uvx ruff check .                          # lint — ruff is NOT a project dep, run via uvx
uv add / uv remove <pkg>                  # manage deps
```

`llama` sets `PYTHONPYCACHEPREFIX` so `__pycache__/` lands in `~/.cache/`, not the source tree.
If you run `python`/`pytest` directly you'll regenerate stray `__pycache__/` — harmless
(gitignored), but prefer the launcher.

---

## 5. Layout

```
llama-voice-assist/
├── llama                # bash launcher (symlinked to ~/.local/bin)
├── ms_llama.py          # core engine: model load, reply pipeline, ReAct loop, CLI entry (large)
├── episodic.py          # sqlite-vec episodic memory, embedded via OpenRouter
├── interface/
│   ├── server.py        # FastAPI + Granian backend (port 7860)
│   ├── tui.py           # Textual TUI workspace (untracked WIP)
│   └── *.html           # web UIs (desktop + mobile)
├── tools/               # one tool per subfolder + the registry (see §6)
├── personas/yuki.txt    # the only persona
├── data/                # RUNTIME state: memory.json, episodic_v2.db, chats/. gitignored. Don't commit.
├── yuki/                # Yuki's personal scratch space. NOT junk. gitignored.
├── output/              # TTS .wav output. gitignored runtime.
├── tests/               # pytest suite + _smoke_*.py manual scripts
├── docs/                # design notes — some STALE (see §9)
└── subprojects/         # side workflows (image_gen, img2img, finetune, dataset scorer);
                         # independent, kept out of the main tree + lint surface
```

---

## 6. The tool system (and the ONE trap that will bite you)

Tools live in `tools/<name>/<name>.py` + `tools/<name>/__init__.py`. Each `__init__.py`
re-exports the implementation as `tool_fn` and declares a `META` dict:

```python
from .calc import tool_calc as tool_fn      # ← load-bearing re-export
META = { "cli_name": "/calc", "react_name": "calc", "description": "...",
         "param_name": "expression", "auto_detect": [...], "is_factual": True }
```

`tools/__init__.py` walks `_TOOL_NAMES` (explicit order matters for auto-detect regex
precedence) and assembles the registry: `TOOLS`, `REACT_TOOL_MAP`, `AUTO_DETECT_REGEX`,
`_FACTUAL_TOOLS`, `TOOL_DESCRIPTIONS`, and `OPENAI_TOOL_SCHEMAS`. `ms_llama.py` just does
`from tools import (...)`.

**To add a tool:** create the subfolder + `META` + `tool_fn`, append the name to `_TOOL_NAMES`.
No `ms_llama.py` edits. Shared helpers (`YUKI_DIR`, `ALLOWED_COMMANDS`, …) live in
`tools/_helpers.py`. Tools that need bot state lazy-import `ms_llama`.

> ### 🚨 THE TRAP: never `ruff check --fix` the tool packages
> ruff flags every `from .x import y as tool_fn` as **F401 "unused import."** It is **NOT
> unused** — `tools/__init__.py` reads `_mod.tool_fn`. Running `--fix` would delete all 18
> re-exports and silently break every tool. A `[tool.ruff.lint.per-file-ignores]` guard in
> `pyproject.toml` now suppresses F401 for `tools/*/__init__.py` so this can't happen by
> accident — leave it in place.

---

## 7. Environment coupling — what won't work on a fresh machine

This repo grew on the owner's specific Mac. Several pieces assume that environment. Don't treat
their failure as a bug to "fix" — they need the surrounding setup:

- **OpenRouter API key.** Cloud models, episodic embeddings, and session summaries need
  `OPENROUTER_API_KEY` exported (the web UI can also set it at runtime). No `.env` file — it's a
  shell env var. Free-tier models routinely **429** mid-session; friendly-error handlers already
  live in `ms_llama.run()` and `server.py` (`_friendly_error`). Don't paper over 429s, surface them.
- **`tools/see/` (the "eye").** Spawns a sibling **Rust** project at `~/eyeoftruth` in an iTerm
  split pane, talks to a **local Ollama** vision model, reads results back over `/tmp/*.jsonl`
  files. Needs: the eye binary built (`cargo build --release`), Ollama running, iTerm/macOS.
  It's Yuki's animated "face." Will no-op with a friendly message if any of that is missing.
- **`tools/ask_claude/` + `yuki-tmux`.** A tmux bridge: Yuki sends questions to *another* agent
  running in a neighboring tmux pane via `tmux send-keys`. Launch the layout with `./yuki-tmux`.
  **Consequence:** an incoming message in the CLI may be *Yuki* relaying through the bridge, not
  the human — keep replies concise and self-contained. Errors come back prefixed
  `ASK_CLAUDE_ERROR:` and are meant to be read aloud, not swallowed.
- **MLX.** Apple-Silicon only. The model-load path falls back MLX → transformers; on non-Mac the
  MLX path just won't engage.

---

## 8. Tests & baseline

- Unit suite: `tests/test_ms_llama.py`, **mock-based — no network, safe to run freely.**
- **Baseline is 67 pass / 5 fail.** The 5 failures are **pre-existing and expected, not
  regressions**: `test_select_model_no_models` (1) + four `test_tool_call_re_*` that assert the
  regex strips quotes when it doesn't. Don't chase them unless you're deliberately reworking
  that regex.
- `tests/_smoke_*.py` are **manual scripts**, not collected by pytest (the `_` prefix keeps them
  out of discovery). They make real API calls — the owner caps spend per smoke run, so **don't
  re-run or broaden them without asking.**
- ⚠️ **`_smoke_recall.py` and `_smoke_verify_wipe.py` are BROKEN against current code** — they
  still call the deleted `bot.unlocked_user_id`, `episodic.count_for(slot)`, and the old 4-arg
  `_cli_record_episode`. They were written pre single-user-collapse and never updated. Fix or
  delete them; don't trust them as examples of the current API.

---

## 9. Memory architecture (current, real)

Dual memory, single-user:
1. **Flat fact memory** — `data/memory.json`, simple hash-keyed facts (names, prefs). Written by
   the `remember` tool / extraction path.
2. **Episodic memory** — `episodic.py`, sqlite-vec over `data/episodic_v2.db`. Every
   user+assistant exchange is embedded (OpenRouter `text-embedding-3-small`); the `recall` tool
   does semantic search and returns *session summaries* (summarized by `EPISODIC_SUMMARY_MODEL`,
   default `openai/gpt-4.1-nano`). Public API: `record`, `summarize_session(session_uuid)`,
   `recall_sessions(query)`, `count_episodes()` — all **single-arg / no-slot**.

**Do not** reintroduce the multi-user system. `docs/TODO-memory-security.md` describes
`MULTI_USER_MODE`, `verify_and_advance`, slots, 2FA unlock, a "122/127 tests" count — **all of
that was deleted 2026-05-15 and no longer exists.** That doc now carries a STALE banner; it's
kept only for design rationale. Don't go looking for symbols it names.

---

## 10. Open TODOs (with pointers)

- **Native function-calling refactor** — `docs/TODO-function-calling.md`. Replace the
  `[TOOL: name("arg")]` text protocol (regex-parsed, leaks on weak models) with backends' native
  `tool_calls`. Groundwork is done: `OPENAI_TOOL_SCHEMAS` is already built from `META` in
  `tools/__init__.py`. Start with the OpenRouter backend.
- **Stored facts leak into new chats** — known memory-isolation bug; the owner has their own fix
  idea. Don't band-aid it; coordinate first.
- **Stale docs** — `docs/TODO-memory-security.md` is historical-only (banner added). `README.md`
  still says "Qwen2.5-1.5B" specifically; the engine is more general now.

---

## 11. Quick gotchas

- `llama` and the tool registry resolve through symlinks — never hardcode `./` paths in messages.
- Commit/push only when the user asks; if you're on `main`, branch first.
- `subprojects/` are independent (some vendored, e.g. the dataset scorer is a cloned third-party
  repo and is gitignored). They're excluded from the lint surface on purpose.
- `.DS_Store` and `__pycache__` are gitignored; a sweep was done at handoff. Don't commit them.

---

## 12. A note from the last agent

A personal note about working with this owner — what they value, how they collaborate, and how to
not step on their toes — lives in **`NOTE-FROM-CLAUDE.md`** at the repo root. Worth a read before
your first real change; it'll save you friction. Short version: real collaborator, high craft bar,
usually has their own fix in mind (ask first), respects cost, and the playfulness is the point.

— Claude Code, handoff 2026-06-24
