# llama-voice-assist architecture handoff

> **Historical snapshot:** this report describes the repository as inspected on
> 2026-08-22. The Textual redesign, strict production routing boundary,
> dedicated-dispatcher integration, reasoning controls, and later continuity
> fixes changed several conclusions below. Use the main `README.md` and
> `docs/TOOL-ROUTING-ARCHITECTURES.md` for current behavior.

Inspection date: 2026-08-22

This report is based on a read-only inspection of the current working tree. No model was loaded, no service was started, no Yuki tool was executed, and no existing project file was changed. The working tree contains a large, coherent but uncommitted refactor, so this describes the code that is present now—not necessarily the last Git commit. The only new file produced by the inspection is this report.

## Executive summary

`llama-voice-assist` is a single-user Apple-Silicon voice companion with three front ends:

- a custom terminal CLI;
- a Textual TUI workspace;
- a FastAPI web UI.

All three use `VoiceChatBot` in `ms_llama.py` as the shared model/conversation/tool runtime. The active model can be local MLX, local Transformers, optional llama.cpp/GGUF, OpenRouter, or another OpenAI-compatible server. The abstraction is practical but not formal: `VoiceChatBot` branches on a backend string instead of delegating through a uniform adapter interface.

The production tool loop currently has two implementations:

- OpenRouter and selected GGUF/model families use native OpenAI-style `tool_calls`.
- MLX, Transformers, and unsupported models use the older free-text `[TOOL: name("argument")]` ReAct protocol plus regex auto-detection.

There are 18 real tools, generated from one live registry. Production schema validation and permission handling are weak, and the active native-call argument adapter is inconsistent with the natural parameter names emitted by the schemas.

Memory is dual-layered:

- flat persistent facts in `data/memory.json`;
- embedded conversation episodes and per-session summaries in `data/episodic_v2.db` using sqlite-vec and OpenRouter embeddings.

The proposed Qwen main brain → semantic delegation → Hammer dispatcher → deterministic binder architecture already exists in isolated form under `prototypes/tool_dispatcher/`. It is not connected to production. Its natural production insertion point is inside/replacing the dispatch portion of `VoiceChatBot.react_chat`, while continuing to use the live registry and existing tool implementations.

## 1. Overall project structure

| Path | Role and relationships |
| --- | --- |
| `llama` | Unified shell launcher. Runs setup, CLI, TUI, web, or tests through `uv`. It changes to the repository directory and moves Python bytecode caches outside the tree. |
| `ms_llama.py` | Main runtime. Discovers/loads models, owns `VoiceChatBot`, constructs model context, maintains live history and patience, performs both tool loops, executes tools, records CLI/TUI episodes, runs TTS, and implements the CLI. This is the central production chokepoint. |
| `interface/tui.py` | Textual front end. Reuses `VoiceChatBot`, adds model/settings screens, session browsing, chat search, inline images, TTS controls, and autonomous mode. |
| `interface/server.py` | FastAPI/Granian back end for the web UI. Owns one process-global bot and active chat, exposes configuration/chat/memory/media endpoints, saves web chats, records episodes, and synthesizes audio. |
| `interface/Yuki_Chat_v2-2.html` | Active desktop browser client. Calls the FastAPI model, chat, memory, tool, media, parameter, and autonomous endpoints. |
| `interface/Yuki_Mobile.html` | Mostly a visual React/Babel prototype. It checks server status but uses sample chats/memories and simulated replies rather than the real `/chat` flow. |
| `tools/` | One package per real Yuki tool plus the registry and shared helpers. Each package supplies `META` and a `tool_fn` re-export. |
| `episodic.py` | Embedded episode storage, semantic retrieval, and session summarization. This is separate from the flat fact store in `ms_llama.py`. |
| `personas/yuki.txt` | The only active persona. It defines Yuki's voice, patience behavior, tool expectations, and some outdated multi-user memory behavior. |
| `data/` | Runtime state: flat memory, episodic SQLite database, TUI settings, and web chat JSON files. This is user data, not source. |
| `yuki/` | Yuki's personal writable workspace. The Yuki-file tools operate here; generated images use `yuki/images/`. |
| `output/` | Generated TTS audio, normally `response.wav`. |
| `prototypes/tool_dispatcher/` | Isolated tool-dispatch research lab: strict live-registry adapter, model backends, parsers, validators, benchmarks, semantic delegation, focused repairs, and deterministic source binding. It is not part of the production assistant loop. |
| `subprojects/` | Independent workflows: QLoRA fine-tuning, text/image pipelines, and a vendored dataset-quality scorer. They are excluded from the main lint surface and do not participate in normal Yuki execution. |
| `tests/` | Main mock-based unit tests plus old manual smoke scripts. The latter include stale memory APIs and should not be treated as current examples. |
| `docs/` | Design notes and a generated prompt inventory. Some documents describe architectures that the current code has already replaced or deleted. |

The dependency/runtime policy is `uv`-only. Heavy libraries are generally loaded lazily. Primary dependencies include MLX-LM, Transformers/PyTorch, OpenAI's client, FastAPI/Granian, Textual, Kokoro, sqlite-vec, DDGS, and system-stat libraries.

## 2. TUI and harnesses

### User-facing entry points

```text
llama cli  -> ms_llama.py          -> VoiceChatBot
llama tui  -> interface/tui.py     -> VoiceChatBot
llama web  -> interface/server.py  -> VoiceChatBot
```

The CLI is a fixed-scroll ANSI terminal interface with character-by-character input, direct slash commands, tool progress lines, autonomous chatter, and optional Kokoro speech/playback.

The Textual TUI is the richest local control surface. It opens before model selection and supports:

- local model discovery;
- OpenRouter model discovery;
- arbitrary OpenAI-compatible remote endpoints such as vLLM, llama.cpp servers, or Ollama-compatible gateways;
- mid-chat model and persona switching;
- persisted theme, model, TTS, voice, autonomy, image-worker URL, and API/remote credentials in `data/tui_settings.json`;
- episodic-session browsing, transcript search, inline generated images, tool progress, reply statistics, and an idle/autonomous mode.

TUI model loading happens in a worker thread. A successful mid-chat switch constructs a new `VoiceChatBot`, then copies the old history, episodic `session_uuid`, patience, and TTS object into it. The old and new models are not intentionally kept as a multi-model system, although both objects can exist transiently during a switch.

The web application uses a single process-global `bot` and `current_chat_id`, matching the project's single-user design. `/api/init` creates a fresh bot; unlike the TUI switch, it does not copy existing history into the replacement bot. The desktop web client is real and connected. `/m` serves the mobile mock, which should not be mistaken for a second working client.

### Streaming and rendering

There is no token-by-token response streaming in the production interfaces. Each backend returns a complete generation. The CLI and TUI display a spinner/progress state while waiting; the web client receives one JSON response after model generation, tool work, and optional TTS complete. The mobile prototype's “Stream responses” switch is not wired to production.

Tool progress is also presentation-specific:

- CLI: terminal spinner/progress lines.
- TUI: a callback from `VoiceChatBot` mounts tool-status lines safely from the worker thread.
- Web: the request waits; it returns the final text, stats, audio, and discovered media.

Internal tool-call scratch is normally hidden. Both production ReAct loops replace intermediate call/result messages with only the original user message and final assistant reply before web-chat persistence or later normal context use.

### Conversation, sessions, and context

`VoiceChatBot.history` is the live, backend-neutral list of OpenAI-shaped messages. It retains at most 10 user/assistant pairs. User input is capped at 2,000 characters. A transient patience instruction is appended immediately before generation for recency but is not retained as normal history.

Persistence differs by surface:

- CLI: live history only; completed turns are also written to episodic memory.
- TUI: treats episodic sessions as its chat archive. Resume displays up to 30 stored turns, loads the latest 10 pairs into model history, and reuses the selected episodic UUID.
- Web: saves collapsed messages to `data/chats/<12-char-id>.json`. Loading a web chat restores those messages but rotates to a new episodic UUID, so web chat identity and episodic session identity are different concepts.

Direct slash commands bypass the model and execute immediately. In CLI/TUI they are not recorded as ordinary episodic turns. Web slash calls are added to web chat history but are not sent through the normal episodic-record branch.

### Approvals, permissions, logging, and configuration

The production UI has no general confirmation/permission layer for tools. A direct slash command, regex-detected action, or native tool call normally executes immediately. The only central special gate is an image-intent check that blocks image generation if recent user context does not actually request an image.

Debugging is lightweight rather than centralized:

- model backend, token count, elapsed time, and tokens/sec are printed/stored per reply;
- CLI/TUI show tool progress and friendly failures;
- TUI `/stats` shows reply stats, history, patience, and memory counts;
- the server uses Python logging for exceptions and episodic failures;
- `see` and `ask_claude` use `/tmp` control/report/log files as part of their integrations.

The TUI persists secrets in plain JSON if entered through settings. The web API stores the OpenRouter key only in the process environment. Other important environment variables include `REMOTE_LLM_BASE_URL`, `REMOTE_LLM_API_KEY`, `CF_IMAGE_URL`, `EPISODIC_SUMMARY_MODEL`, `LLAMA_FORCE_REGEX`, and eye/Claude bridge settings.

### Isolated dispatcher debugging harness

`prototypes/tool_dispatcher/` also has a separate local web lab and CLI benchmarks. Unlike the assistant UI, it exposes raw model output, schemas, parsing, validation, latency, token rates, and memory metrics. Routing/validation is separated from execution. Most recent production-interface experiments use an even narrower `ValidationOnlyDispatcher` that has no execution API at all.

The lab supports MLX, Transformers, and OpenAI-compatible dispatcher backends, trained output dialects for xLAM/Hammer/Arch-family checkpoints, early stopping after one complete JSON value, a 256-token ceiling, and stateless one-generation-per-request evaluation.

## 3. Agent/runtime loop

### Normal message flow

All active front ends eventually call the same method:

```text
user input
  -> front-end validation/rendering
  -> VoiceChatBot.react_chat(input)
       -> choose native or regex loop
       -> model generation
       -> optional tool selection/parsing
       -> existing tool function executes
       -> raw tool result enters temporary model context
       -> model generates Yuki's final answer
       -> collapse intermediate tool scratch
  -> record episode (normal turns)
  -> save/render final chat response
  -> optional Kokoro TTS
```

The native loop is available only when both backend and model-family checks pass. It sends `OPENAI_TOOL_SCHEMAS` to OpenRouter or llama.cpp, reads `message.tool_calls`, executes each call, appends OpenAI-style assistant/tool messages, and asks the same model for the final answer. OpenRouter models whose provider rejects tool parameters are cached as non-tool-capable and retried as plain chat.

The regex loop temporarily appends prose tool instructions to the system prompt, asks the model to emit `[TOOL: ...]`, and then tries, in order:

1. explicit tool-call regex parsing;
2. regex auto-detection from the model response;
3. regex auto-detection from the original user request.

It executes a matching tool, injects a synthetic `[TOOL_RESULT: ...]` user message, and calls the same model again. Both loops allow several iterations, though normal guidance prefers one tool.

### System-prompt assembly

Context is assembled in layers:

```text
personas/yuki.txt
  + backend self-awareness (local/cloud and hardware meaning)
  + web-only flat-memory block, if using the web UI
  + per-call embodied-eye/emotion note
  + regex-only prose tool catalog (native calls receive schemas separately)
  + normal history (last 10 pairs)
  + transient current-patience hint
```

Tool-result instructions differ by tool. Factual tools get stricter “use these exact facts” guidance. Failures tell the model to report failure honestly. Non-factual tool results ask for a warm persona-consistent response.

### One core, several loops

There is one shared `VoiceChatBot` core but more than one agent loop:

- `_react_chat_native`: native OpenAI-style tool calls for supported OpenRouter/GGUF models;
- `_react_chat_regex`: universal free-text ReAct fallback;
- `autonomous_tick`: a separate idle-agent loop that still uses regex tool parsing even if the selected model supports native calls;
- direct slash-command execution: bypasses the agent loop and model entirely.

This means tool behavior is centralized at `_run_tool_with_spinner`, but selection, argument extraction, result-role formatting, and persistence are not yet one unified pipeline.

## 4. Memory system

### Flat fact memory

Flat memory lives in `data/memory.json` and is implemented mainly by these `ms_llama.py` functions:

- `_empty_facts`, `_load_facts`, `_save_facts`;
- `_add_fact`, `_normalize_value`, `_classify_fact`;
- web adapters `_load_memory`, `_memory_as_prompt`, and `_inject_memory`;
- `tools/remember/` for model/direct-command writes.

The current store is single-user and category-to-list shaped:

```json
{
  "names": [],
  "loves": [],
  "hates": [],
  "hobbies": [],
  "notes": [],
  "favorite_character": [],
  "favorite_series": [],
  "handle": []
}
```

There is no active user/entity UUID, authentication, encrypted slot, or fact hash. `_load_facts` recognizes the deleted multi-user format only to migrate the first non-empty legacy slot into the flat shape and discard slot metadata.

`remember(fact)` classifies a few structured phrasings—name, handle, favorite character/series, loves, hates—and stores everything else as a note. Duplicate detection is case-insensitive string equality. The web memory API can add/delete sidebar items and rebuild the memory prompt. Its item identifiers are `category:index`, so indices are not stable after deletions.

Only the web lifecycle currently injects the flat fact block automatically. CLI and TUI load the same fact counts and can invoke `remember`, but do not mount `_memory_as_prompt` into `VoiceChatBot.system_prompt`. Also, a `remember` tool call writes the file but the shared tool runner does not call web `_inject_memory`; therefore web prompt refresh occurs on init/new/load or memory-API edits, not immediately through the generic remember tool path.

### Episodic memory

`episodic.py` implements the second layer:

```text
normal user + final Yuki reply
  -> OpenRouter embedding (openai/text-embedding-3-small, 1536 dimensions)
  -> episodes row + aligned sqlite-vec row
  -> grouped by session_uuid
  -> session boundary/exit produces a 2-4 sentence summary
```

SQLite schema:

```text
episodes(
  id INTEGER PRIMARY KEY,
  session_uuid TEXT,
  ts REAL,
  user_msg TEXT,
  assistant_msg TEXT
)

session_summaries(
  session_uuid TEXT PRIMARY KEY,
  ts_start REAL,
  ts_end REAL,
  summary TEXT
)

vec_episodes(embedding float[1536])
```

`episodic.record` embeds every normal completed turn. `summarize_session` calls the configured OpenRouter summary model—default `openai/gpt-4.1-nano`—and falls back to a short extract if summarization fails. TUI new-session, web new/load, and process exit summarize the outgoing session.

`recall_sessions(query)` embeds the topic, searches nearest episode vectors, drops candidates beyond distance 1.28, excludes the active session, keeps the best episode per session, and returns up to three session summaries. Unsummarized matching sessions are lazily summarized. `tools/recall/` formats those summaries with fuzzy timestamps.

Episodic memory is not automatically injected into every turn. It enters context only when the model/direct command selects `recall`; the tool result then follows the ordinary tool-result loop. `_ACTIVE_BOT` is a module-global pointer used to exclude the current session.

### Memory architecture in one diagram

```text
                       +-----------------------------+
                       | VoiceChatBot live history   |
                       | last 10 user/assistant pairs|
                       +--------------+--------------+
                                      |
                         completed normal turn
                                      v
raw conversation  -> embedding API -> episodes + vec_episodes
                                      |
                             session boundary
                                      v
                              session_summaries
                                      |
user asks about past -> recall tool -> vector search -> summaries -> model context

remember tool / web memory editor -> data/memory.json
                                      |
                         web-only prompt injection
                                      v
                              VoiceChatBot system prompt
```

## 5. Tool system

### Registry and schemas

Each `tools/<name>/__init__.py` re-exports its implementation as `tool_fn` and declares a `META` dictionary containing:

- CLI and ReAct names;
- help/usage and model description;
- natural parameter name and description;
- factual/non-factual status;
- optional regex auto-detection patterns.

`tools/__init__.py` walks an explicit ordered list and builds:

- `TOOLS`: slash-command registry;
- `REACT_TOOL_MAP`: model tool name to implementation;
- `AUTO_DETECT_REGEX`: ordered natural-language fallbacks;
- `_FACTUAL_TOOLS`: controls result grounding prompts;
- `TOOL_DESCRIPTIONS`: regex-loop tool instructions;
- `OPENAI_TOOL_SCHEMAS`: native function schemas;
- `TOOL_PARAM_NAMES`: intended natural argument-key mapping.

The 18 tools are grouped conceptually as follows:

```text
information:    time, weather, fetch, search, calc
computer:       hardware, read, shell
media:          see, image
yuki files:     yuki_write, yuki_read, yuki_list, yuki_delete, yuki_append
memory:         remember, recall
external agent: ask_claude
```

The re-export is load-bearing: automated removal of apparently unused `tool_fn` imports would break registry construction.

### Active execution paths

```text
/slash command ---------------------------> TOOLS[name].fn(arg)

regex/native/auto-detected model decision
  -> REACT_TOOL_MAP lookup
  -> image-intent gate
  -> tool_fn(single string argument)
  -> raw result converted to text
  -> temporary model context
```

Tools include network calls (weather, fetch, search), local system inspection, arbitrary local-file reading, an allowlisted shell wrapper, Yuki-folder file mutations, memory writes/search, a Cloudflare image worker, a local camera/Ollama/Rust eye bridge, and a tmux bridge to another Claude session.

### Validation and security findings

Production does not perform strict JSON-schema validation before execution. Current generated schemas have `required: []` even for functions that need input and do not forbid extra properties.

There is also an active native-argument mismatch: schemas emit meaningful keys such as `query`, `city`, `prompt`, or `filepath`, but `_native_tool_arg` currently extracts only `parsed["arg"]`. `TOOL_PARAM_NAMES` is built but not used/exported by the active runtime. A correctly structured native call can therefore reach a tool with an empty argument.

Other important boundaries:

- `shell` verifies only the first whitespace-delimited command name, then runs the entire string with `shell=True`; shell operators can bypass the intended allowlist.
- `read` can read any local path the process can access; it is not restricted to the project.
- Yuki-file names remove `/` and `..`, which contains operations to `yuki/`, but write/delete/append execute without confirmation.
- `fetch` accepts any HTTP(S) URL and has no network/SSRF policy beyond protocol and size/time limits.
- `see`, `image`, `remember`, `recall`, and `ask_claude` can open hardware, contact services, mutate state, incur cost, or cross agent boundaries without a general approval layer.

The isolated dispatcher prototype improves these boundaries without changing Yuki: it derives required fields from real signatures, adds `minLength` and `additionalProperties: false`, validates one canonical call, separates route/validate from execution, requires explicit permission for sensitive tools, and provides dry-run/strict shell modes. Those protections are not integrated into production.

## 6. Model abstraction

### Supported active backends

| Input/model form | Loader and runtime | Tool-call behavior |
| --- | --- | --- |
| `openrouter/<model>` | OpenAI client pointed at OpenRouter | Native calls for allowlisted model families; plain-chat fallback if provider rejects tools. |
| `remote/<model>` | OpenAI client pointed at `REMOTE_LLM_BASE_URL` | Internally labeled `openrouter`, so it shares API/native-tool logic. |
| `*.gguf` | Optional `llama_cpp.Llama`, 4096 context, GPU layers enabled | Native OpenAI-shaped calls for allowlisted model families. `llama-cpp-python` is not declared in the current project dependencies. |
| Hugging Face/local model ID | MLX-LM attempted first | Regex ReAct fallback in production. |
| MLX failure/non-MLX environment | Transformers on MPS then CPU | Regex ReAct fallback in production. |

Local discovery looks first under `/Volumes/madisk/huggingface/hub`, then the internal Hugging Face cache. GGUF files inside cached model directories are listed individually; other directories become organization/model IDs.

The architecture is provider-flexible but not fully provider-agnostic. There is one public `VoiceChatBot` surface, but `chat`, `raw_complete`, and tool-aware generation each contain backend-specific branches and output parsing. Feature parity differs: native tools are not implemented for production MLX/Transformers, remote endpoints are represented as OpenRouter internally, and no backend provides production token streaming.

The prototype has the cleaner abstraction: `DispatcherBackend`/`DelegationBackend` classes expose stateless generation, while model-specific prompt dialects, decoder normalization, parsing, and metrics remain outside the registry and execution code.

## 7. Fit for the newer Yuki architecture

Target architecture:

```text
Qwen main brain
  -> semantic delegation
  -> Hammer dispatcher
  -> deterministic exact-payload binder
  -> validated Yuki tool call
  -> existing Yuki tool
  -> raw result
  -> Qwen final response
```

### Equivalent components that already exist

| Target component | Existing equivalent | Status |
| --- | --- | --- |
| Qwen conversational main brain | `VoiceChatBot` plus active backend generation | Production, but it currently also selects tools. |
| Semantic delegation contract | `delegation_contract.py` and `delegation_mainbrain.py` | Isolated and tested experimentally. Contract fields are `request`, `domain_hint`, and `verbatim`; the main brain is forbidden to name the final tool. |
| Domain-to-schema subset mapping | `delegation_contract.domain_tools`; also `automatic_selector.py` for deterministic experiments | Isolated. Frozen experiments used generated domain hints; automatic selection is research, not production. |
| Hammer-only dispatcher | `ValidationOnlyDispatcher`, `prompting.py`, `backends.py`, `parsing.py` | Isolated, stateless, native Hammer format, one generation, no execution capability. |
| Strict live-schema adapter | `ToolRegistry` in prototype `registry.py` | Isolated. Reuses production `META`, schemas, order, and implementations while repairing required/additional-property rules. |
| Deterministic source binder | `source_span_policy.py` | Isolated focused prototype. It copies raw source spans or explicitly abstains. Current coverage is focused on search, image, ask_claude, see, and recall; its pattern inventory should not yet be treated as a general 18-tool production binder. |
| Validated execution boundary | Prototype `Dispatcher.route_and_validate` and guarded `executor.py` | Isolated. Production has direct execution but lacks the strict boundary. |
| Raw result back to main brain | Existing native/regex tool-result loops | Production behavior already exists, though result-role formats differ. |

### Natural production insertion points

The front ends should continue to call one shared runtime. A new coordinator should live at or immediately below `VoiceChatBot.react_chat`:

```text
CLI / TUI / web
      |
      v
VoiceChatBot.react_chat(raw_user_message)
      |
      +-- main-brain generation decides: answer or delegate
      |
      +-- on delegation:
            validate delegation contract
            select live schema subset from domain_hint
            Hammer one-shot dispatch
            normalize native call
            deterministic binder copies exact spans from raw_user_message
            strict live-schema validation
            permission/approval policy
            existing REACT_TOOL_MAP implementation
            raw result returned to Qwen for the only natural-language answer
```

Specific reuse points:

1. Keep `tools/__init__.py` as the source of truth. Move or reuse the prototype's strict adapter rather than creating parallel tool definitions.
2. Reuse `VoiceChatBot.history`, system/persona construction, final reply cleanup, patience, TTS, and front-end persistence for Qwen only.
3. Give Hammer no persona, history, tool result, or answer-generation role. The existing prototype already enforces this.
4. Preserve the immutable raw user string alongside the semantic delegation until binder completion. Current `react_chat` receives the raw string at the right boundary.
5. Replace `_native_tool_arg`, regex argument extraction, and direct execution with one canonical validated-call object before the existing tool function is reached.
6. Put approvals between validation/binding and `REACT_TOOL_MAP`, not inside individual model adapters.
7. Feed the tool's raw result only to Qwen. Do not append it to Hammer state.

The largest integration risk is not the front end; it is having multiple dispatch paths continue to coexist. Production should ultimately have one validated execution boundary used by main chat, auto-detection if retained, direct slash commands where appropriate, and autonomous mode. Until then, the new coordinator should be added without routing through the known broken native `arg` adapter.

## 8. Old, unused, duplicate, or confusing architecture

- The current tree is a large uncommitted refactor. `ms_llama.py`, the server, episodic memory, tests, and tools differ substantially from the last committed baseline. Any integration should remain separable and must not reset these changes.
- `docs/archive/TODO-function-calling.md` says native function calling is deferred and all tools use regex. That is outdated: OpenRouter and GGUF native loops now exist, while MLX/Transformers still need the documented work.
- `docs/archive/TODO-memory-security.md` has a correct stale banner, but most of the document describes deleted multi-user slots, hashes, 2FA, and unlock behavior.
- `personas/yuki.txt` still contains old locked/multi-user “vibe check” instructions even though runtime memory is single-user.
- Comments above flat memory in `ms_llama.py` still describe UUID-keyed, hashed per-user facts. The implementation directly below them is flat and single-user.
- `_reset_session_lock` in the server is a compatibility no-op. Calls and comments around it can mislead readers into thinking identity locking remains active.
- The web memory API accepts `name`, `age`, and `location`, while the canonical loader only retains its predefined category keys; these extra categories do not survive a normal reload consistently.
- `interface/Yuki_Mobile.html` is a UI mock with sample data and timed fake responses, not a mobile equivalent of the desktop client.
- `_react_chat_native`, `_react_chat_regex`, `autonomous_tick`, and direct slash execution duplicate routing/result behavior. The autonomous loop still uses regex even on native-capable backends.
- The regex loop mutates `user_input` to an empty string after an original-input auto-detect fallback; its final history-collapse logic can then omit the original user message on that path.
- `TOOL_PARAM_NAMES` is documented as the native argument solution but is not used by `_native_tool_arg`.
- `ms_llama.py` retains local `ALLOWED_COMMANDS`, `YUKI_DIR`, and `_split_filename_content` definitions while real tool packages use `tools/_helpers.py`; these appear to be refactor leftovers.
- `raw_complete` and its old `tool_context` path remain as subsystem/legacy hooks, but current fact storage and normal tool loops do not rely on them.
- `tests/_smoke_recall.py` and `tests/_smoke_verify_wipe.py` call deleted multi-user APIs. They are manual scripts, not trustworthy examples of current memory behavior.
- The active test file is mostly mock-oriented and does not provide broad protection for the newer native-call path. The apparent shell-injection test does not establish that shell operators are blocked.
- `prototypes/tool_dispatcher/` is intentionally extensive and report-heavy, but it is a separate research system. Its executor and strict schemas should not be mistaken for production protections.
- `subprojects/finetune`, `image_gen`, `img2img`, and `Fine-tune-Dataset-Quality-Scorer` have their own runtimes/dependency assumptions. They are not alternate production agent loops.
- `docs/archive/system-note-legacy.txt` is a useful prompt inventory, but it is generated documentation and may drift from runtime code.

## Current architecture in one diagram

```text
                         +----------------------+
                         | personas/yuki.txt    |
                         | + backend awareness  |
                         | + web flat memory    |
                         +----------+-----------+
                                    |
+----------+   +-----------+   +----v--------------------+
| CLI      |   | Textual   |   | FastAPI + desktop web  |
| ms_llama |   | TUI       |   | one global bot/chat    |
+-----+----+   +-----+-----+   +-----------+-------------+
      |              |                     |
      +--------------+---------------------+
                     v
              +------+-------+
              | VoiceChatBot |
              | live history |
              +------+-------+
                     |
          +----------+----------+
          |                     |
          v                     v
 native OpenAI tool loop    regex ReAct loop
 OpenRouter / GGUF          MLX / Transformers / fallback
          |                     |
          +----------+----------+
                     v
              tools registry
                     |
              existing tool_fn
                     |
              raw tool result
                     |
             same model final reply
                     |
      chat persistence / episodic record / TTS
```

## Model/backend architecture in one diagram

```text
model identifier
   |
   +-- openrouter/... ------> OpenAI client @ OpenRouter --+
   |
   +-- remote/... ----------> OpenAI-compatible client ----+--> backend="openrouter"
   |
   +-- path/to/*.gguf ------> llama_cpp.Llama -------------> backend="gguf"
   |
   +-- HF/local ID --try---> MLX-LM -----------------------> backend="mlx"
                         |
                         +--fallback--> Transformers/MPS/CPU -> backend="transformers"

VoiceChatBot methods branch on backend string;
there is no single production ModelAdapter protocol.
```

## Likely Yuki integration points

```text
Keep:
  front ends, VoiceChatBot conversation state, persona, memory, TTS,
  live tool registry, existing tool implementations, raw-result-to-Qwen step

Insert/replace inside react_chat:
  Qwen delegation contract
    -> domain/schema subset
    -> stateless Hammer adapter
    -> canonical parser
    -> deterministic raw-source binder
    -> strict schema validator
    -> centralized approval policy
    -> one execution chokepoint

Retire/bypass for the new path:
  [TOOL: ...] generation, regex argument parsing, `_native_tool_arg`,
  unvalidated direct native execution
```

The prototype code is close to a reusable reference implementation, but it should be promoted deliberately: keep benchmark-only pattern sets, scorers, reports, frozen artifacts, and execution experiments out of the production runtime.

## Important files ChatGPT should inspect next

Suggested order:

1. `AGENTS.md` — current operating rules, dirty-tree warning, and project conventions.
2. `ms_llama.py` — especially `setup_llm`, `VoiceChatBot.chat`, `react_chat`, both ReAct loops, `_run_tool_with_spinner`, and `autonomous_tick`.
3. `tools/__init__.py` — live registry, schemas, ordered auto-detection, and the parameter mismatch context.
4. `tools/*/__init__.py` and `tools/_helpers.py` — real metadata and shared file/shell conventions.
5. `interface/tui.py` — model switching, episodic session resume, settings, and worker/thread behavior.
6. `interface/server.py` — web chat persistence, memory prompt injection, session rotation, and API lifecycle.
7. `episodic.py`, `tools/remember/remember.py`, and `tools/recall/recall.py` — both persistent memory layers end to end.
8. `personas/yuki.txt` — actual base system prompt plus stale identity sections.
9. `prototypes/tool_dispatcher/registry.py`, `prompting.py`, `backends.py`, `parsing.py`, and `delegation_dispatcher.py` — strict stateless Hammer path.
10. `prototypes/tool_dispatcher/delegation_contract.py`, `delegation_mainbrain.py`, and `source_span_policy.py` — Qwen semantic delegation and deterministic source binding.
11. `prototypes/tool_dispatcher/README.md` and its focused reports — experimental results and safety invariants, not production code.
12. `docs/archive/TODO-function-calling.md` and `docs/archive/TODO-memory-security.md` — useful history only after accounting for their stale sections.
