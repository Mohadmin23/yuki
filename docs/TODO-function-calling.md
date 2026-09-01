# Historical TODO: Switch to native function calling

> **Implemented/superseded:** normal user-message routing now supports both
> provider-native calls and canonical JSON through one strict Yuki call boundary,
> plus an optional dedicated dispatcher architecture. See
> `docs/TOOL-ROUTING-ARCHITECTURES.md`. The notes below are retained as the
> historical design rationale. The legacy regex loop remains for compatibility,
> but it is no longer selected by the new normal-chat or autonomous routing modes.

**Original status:** Deferred. The old tool path used `[TOOL: name("arg")]` text emitted by the model and regex-parsed by `react_chat`. It worked, but was fragile — weak models could leak the syntax (or worse, quote tool source code) into their reply text.

**Trigger to revisit:** when a real user-visible failure ties back to `[TOOL: …]` parsing — e.g., the 2026-05-15 incident where deepseek-v4-flash:free wrote `results = list(DDGS().text(query, max_results=3))` (the implementation of `tool_search`) into Yuki's reply instead of summarizing a URL.

---

## Why this fixes things

- Model emits **structured JSON tool calls** in a dedicated channel, not free-form text. No regex extraction, nothing to leak.
- Each backend's runtime *enforces* the format — the model literally can't emit a malformed call.
- Models trained on tool use (Qwen 2.5/3, Llama 3.1+, DeepSeek, Mistral, GPT, Claude) all use this convention natively. Asking them to emit `[TOOL: …]` instead is asking them to do the second-best thing.
- Weak free-tier models suddenly look much smarter at tool use, because they're now using a format they were trained on.

## Backend matrix

| Backend | API | Notes |
|---|---|---|
| **OpenRouter** | `chat.completions.create(tools=[...])` → `message.tool_calls` | Cleanest. Most-used path. Start here. |
| **llama-cpp (GGUF)** | `create_chat_completion(tools=[...])` | Recent llama-cpp-python versions. Same shape as OpenAI. |
| **MLX (mlx_lm)** | No `tools=` API on `generate()` | Use `tokenizer.apply_chat_template(messages, tools=[...])` to inject the model's expected tool-prompt format, then parse the model's native tool-call output (Qwen: `<tool_call>{...}</tool_call>`, Llama: `<\|python_tag\|>...`). Still parsing, but parsing the format the model was trained to emit, not custom syntax. |
| **transformers** | `apply_chat_template(messages, tools=[...])` | Same as MLX route. |

## Scope of the refactor

### `tools/__init__.py`
- Add a function that emits OpenAI-format tool schemas from the existing META dicts.
- Each tool's META already has `react_name`, `description`, and the implicit single-arg signature. Schema can be derived without per-tool edits.

### `ms_llama.py`
- Replace the `[TOOL: …]` parsing in `react_chat` with backend-aware tool-call extraction:
  - OpenRouter → read `response.choices[0].message.tool_calls`
  - llama-cpp → same
  - MLX → format prompt via `apply_chat_template(tools=…)`, then parse model-specific tool-call markup
- Drop or fallback-only-mode `TOOL_CALL_RE` regex.
- Keep `auto_detect_tool` for direct user `/slash` commands — unrelated path.

### `interface/server.py`
- No changes needed — `bot.react_chat` is the chokepoint.

### Per-model fallback policy
- If a model doesn't expose `tools` in its `supported_parameters` (check via OpenRouter `/api/v1/models`), fall back to the current text-parsing path.
- Or refuse to load that model. TBD.

---

## How to resume

1. Re-read this file.
2. Pick a backend to start with (OpenRouter is easiest).
3. Add tool-schema emission to `tools/__init__.py` — keep existing META as source of truth.
4. Refactor `react_chat` to branch on `self.backend` and extract `tool_calls` accordingly.
5. Test with `gpt-4.1-mini` or `deepseek-v3.1` (both support tools well) before trying free-tier models.
6. Once OpenRouter works, port the pattern to llama-cpp (trivial — same API shape) and MLX (a bit more parsing).
7. Drop dead `TOOL_CALL_RE` code once all backends are migrated.
