# Yuki

**A single-user AI companion runtime for Apple Silicon: voice, persistent
continuity, tools, vision, autonomous activity, and swappable local or cloud
models.**

Yuki has grown well beyond its original fixed-model voice-chat prototype. Yuki is now the
system itself: the personality, model runtime, memory, tool boundary, interfaces,
and experimental dispatcher architecture all work together as one companion.

Yuki is deliberately personal and local-first. She is not a multi-user service,
an account platform, or a generic chatbot skin. The default experience is a
neon, retro-terminal Textual workspace backed by the shared `VoiceChatBot`
runtime.

> **Project status:** active personal research software. The primary machine is
> an Apple-Silicon Mac. Some capabilities use optional cloud APIs or local
> companion services, and tool access is powerful enough that you should read
> the [safety and privacy notes](#safety-and-privacy) before running an unfamiliar
> configuration.

## What Yuki can do

- Run local MLX and Transformers models, optional GGUF/llama.cpp models,
  OpenRouter models, or another OpenAI-compatible endpoint.
- Switch the active conversational model without throwing away the current
  session.
- Show supported model reasoning separately from the final response, with
  `AUTO`, `MINIMAL`, `LOW`, `MEDIUM`, `HIGH`, and `OFF` controls.
- Speak through Kokoro TTS and keep speech playback separate from model work.
- Preserve resumable chats, current-session continuity, long-term facts, and
  source-aware retrieval across episodic memories and Yuki-owned notes.
- Research the live web through bounded page retrieval with source URLs and an
  explicit untrusted-content boundary.
- Select from 18 real tools for live information, local computer access, vision,
  image generation, Yuki's own files, memory, and an external Claude bridge.
- Route tools directly with the conversational model or delegate routing to a
  dedicated small function-calling model such as Hammer.
- Validate and normalize structured calls at one strict boundary before an
  existing Yuki tool implementation can run.
- Continue meaningful work during explicitly enabled autonomous mode instead of
  treating user silence as a new chat message.

## Quick start

### Requirements

- macOS on Apple Silicon for the intended MLX experience
- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A local model, an OpenRouter API key, or an OpenAI-compatible remote endpoint
- Optional system dependencies for audio playback and the specialized tools

The project uses `uv` exclusively. Do not create a separate `pip` environment.

```bash
git clone https://github.com/Mohadmin23/yuki.git
cd yuki
mkdir -p ~/.local/bin
ln -s "$PWD/llama" ~/.local/bin/llama
llama setup
llama tui
```

If the launcher is already on your `PATH`, only `llama setup` is needed after a
fresh clone.

## Entry points

All interfaces use the same core runtime rather than maintaining separate Yuki
implementations.

| Command | Interface | Main file | Best for |
| --- | --- | --- | --- |
| `llama tui` | Textual workspace | `interface/tui.py` | Full local experience: models, sessions, tools, reasoning, TTS, and autonomy |
| `llama cli` | Terminal chatbot | `ms_llama.py` | Lightweight terminal use and debugging |
| `llama web` | FastAPI web UI | `interface/server.py` | Browser-based chat and voice input at `http://localhost:7860` |
| `llama test` | Pytest suite | `tests/` | Mock-based runtime and interface checks |

The TUI is the primary interface. It keeps Yuki's pixel/neon 1990s terminal
identity while providing a searchable session drawer, model manager, explicit
settings, Markdown replies, inline images, activity cards, and narrow-terminal
reflow.

## Current architecture

```text
CLI / Textual TUI / FastAPI web UI
                 |
                 v
          VoiceChatBot runtime
          (`ms_llama.py`)
                 |
        +--------+---------+
        |                  |
        v                  v
 conversational model   memory / continuity
        |                facts + episodes + sessions
        v
   tool routing mode
        |
        +-- DIRECT ----------------------------------+
        |   active model selects the structured call |
        |                                             |
        +-- DISPATCHER -------------------------------+
            main brain chooses semantic delegation    |
                       |                               |
                       v                               |
            dedicated stateless function router       |
                                                        v
                     canonical Yuki call
                    {tool, arguments}
                              |
                              v
               strict validation and source checks
                              |
                              v
                   existing Yuki tool registry
                              |
                              v
                         raw result
                              |
                              v
                 conversational model speaks as Yuki
```

`ms_llama.py` remains the central runtime chokepoint. It owns model setup,
conversation history, reasoning extraction, tool orchestration, session events,
TTS, and the CLI. The front ends call this shared core.

## Tool routing

Yuki exposes two selectable architectures through the TUI.

### Direct

The active conversational model decides which tool is needed. Depending on the
selected protocol and backend, it can use provider-native tool calls or a
provider-neutral canonical JSON call.

### Dedicated dispatcher

The conversational model decides only whether an external action is required
and delegates a semantic request plus a broad domain. A second model performs
exactly one stateless tool-selection generation. That dispatcher:

- is not a chatbot;
- receives no previous tool result;
- never writes Yuki's answer;
- cannot execute a tool itself;
- stops after producing one structured call.

Both modes terminate at the same canonical boundary. The boundary checks the
tool name, strict arguments, offered schema, typed literal-source requirements,
and prototype-derived shell restrictions before adapting the call to the live
tool implementation.

Read the current design in
[`docs/TOOL-ROUTING-ARCHITECTURES.md`](docs/TOOL-ROUTING-ARCHITECTURES.md).

## The 18 Yuki tools

The registry in `tools/__init__.py` is the source of truth for every interface
and routing architecture.

| Domain | Tools | Purpose |
| --- | --- | --- |
| Information | `time`, `weather`, `fetch`, `search`, `calc` | Current information, source-aware web retrieval, exact pages, and arithmetic |
| Computer | `hardware`, `read`, `shell` | Local telemetry, exact file reads, and allowlisted commands |
| Media | `see`, `image` | Inspect an existing visual source or generate a new image |
| Yuki files | `yuki_write`, `yuki_read`, `yuki_list`, `yuki_delete`, `yuki_append` | Manage Yuki's personal `yuki/` workspace |
| Memory | `remember`, `recall` | Store persistent facts or retrieve prior episodes and related Yuki notes |
| External agent | `ask_claude` | Send a question through the optional neighboring tmux agent bridge |

Each tool package supplies its implementation and `META` definition. From that
metadata Yuki builds slash commands, model descriptions, native OpenAI-compatible
schemas, factual-result handling, and the canonical routing registry.

## Memory and continuity

Yuki has multiple forms of continuity with different jobs:

```text
current open chat
    -> bounded model history
    -> durable same-session events and resumable transcript

explicit facts
    -> data/memory.json
    -> remember tool

past conversations
    -> data/episodic_v2.db
    -> OpenRouter embeddings + sqlite-vec
    -> per-session summaries

Yuki-owned notes
    -> data/retrieval.db
    -> SQLite FTS5 passages

episodic + note rankings
    -> reciprocal-rank fusion
    -> numbered source context
    -> recall tool
```

The memory system is intentionally single-user. Runtime memory, chat transcripts,
settings, generated speech, and Yuki's personal files are gitignored.

Episodic embeddings currently use `openai/text-embedding-3-small` through
OpenRouter. Session summaries default to `openai/gpt-4.1-nano` and can be changed
with `EPISODIC_SUMMARY_MODEL`.

Live `search` results use the same typed source boundary. DuckDuckGo supplies
candidate pages, the top public results are read within strict size/time/network
limits, and relevant passages are returned with their URLs. Retrieved webpages
are evidence only: instructions found inside them cannot authorize tools. See
[`docs/RAG-PHASE1.md`](docs/RAG-PHASE1.md) for the current design and its limits.

## Model and backend support

| Backend | Model source | Notes |
| --- | --- | --- |
| MLX | Local Hugging Face or MLX directories | Preferred local Apple-Silicon path |
| Transformers | Local Hugging Face directories | Fallback when MLX cannot load the checkpoint |
| llama.cpp / GGUF | Local `.gguf` files | Optional backend; native-call support depends on model/runtime |
| OpenRouter | Live OpenRouter model catalog | Supports model-specific reasoning and native tool calls where available |
| Remote | OpenAI-compatible base URL | Useful for local servers such as vLLM or another machine |

The main conversational model and dedicated dispatcher are selected separately.
Local dispatcher checkpoints are loaded lazily only when dispatcher mode needs
them.

## Voice, vision, and companion integrations

- **Voice output:** Kokoro TTS, with several selectable voices.
- **Vision:** the `see` tool connects to the separate Rust `eyeoftruth` project
  and a local Ollama vision model.
- **Image generation:** the `image` tool calls the configured image worker and
  stores results in `yuki/images/`.
- **Claude bridge:** `ask_claude` uses tmux to communicate with another agent
  pane launched through `yuki-tmux`.

These integrations degrade to visible errors when their companion service is
not configured; they are not required for ordinary chat.

## Configuration

The TUI can configure most model, voice, reasoning, routing, and endpoint
settings interactively. Its local settings live in `data/tui_settings.json`.

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Cloud models, episodic embeddings, and session summaries |
| `EPISODIC_SUMMARY_MODEL` | Override the model used to summarize sessions |
| `REMOTE_LLM_BASE_URL` | OpenAI-compatible remote endpoint |
| `REMOTE_LLM_API_KEY` | Optional credential for the remote endpoint |
| `CF_IMAGE_URL` / `CF_IMAGE_TOKEN` | Image-generation worker configuration |
| `EYE_BIN` / `EYE_OLLAMA` | Vision bridge executable and Ollama endpoint |
| `YUKI_CLAUDE_PANE` | tmux target for the external Claude bridge |
| `YUKI_CONTEXT_WINDOW_TOKENS` | Override detected context-window size |
| `YUKI_HISTORY_TOKEN_BUDGET` | Override the history token budget |
| `YUKI_WEB_RAG_FETCH_PAGES` | Number of top search results to read (`0`–`3`, default `2`) |

Do not commit API keys. Keys entered through the TUI are stored locally in its
gitignored settings file, not encrypted as a multi-user credential vault.

## Project layout

```text
yuki/
├── llama                    # uv-based launcher
├── ms_llama.py              # shared model, conversation, tool, TTS, and CLI runtime
├── episodic.py              # sqlite-vec episodic memory and session summaries
├── interface/
│   ├── tui.py               # primary Textual workspace
│   ├── server.py            # FastAPI + Granian backend
│   └── *.html               # desktop and mobile web clients
├── tool_routing/            # production canonical-call boundary and dispatcher runtime
├── retrieval/               # typed local/web retrieval, ranking, and trust boundaries
├── tools/                   # 18 live tools and the metadata-driven registry
├── personas/yuki.txt        # Yuki's active personality prompt
├── tests/                   # mock-based automated tests and manual smoke scripts
├── docs/                    # current architecture, handoffs, and historical design notes
├── prototypes/tool_dispatcher/
│   ├── backends and parsers # isolated dispatcher research harness
│   ├── benchmark datasets   # frozen and targeted routing evaluations
│   └── reports/             # machine-readable results and research reports
├── subprojects/             # fine-tuning, dataset review, and image workflows
├── data/                    # gitignored chats, settings, facts, and retrieval databases
├── yuki/                    # gitignored personal Yuki workspace
└── output/                  # gitignored generated speech
```

The production dispatcher currently reuses several proven dialect, prompt,
parser, and backend components from `prototypes/tool_dispatcher/`. Separating
those runtime dependencies from the research lab is planned, but has not been
hidden or falsely presented as complete.

## Development

```bash
llama test
uv run pytest tests/test_ms_llama.py -q
uvx ruff check .
```

Project dependencies must be changed with `uv add` or `uv remove`. Heavy and
optional libraries are imported lazily so missing integrations fail near the
feature that needs them rather than breaking startup.

Read [`AGENTS.md`](AGENTS.md) before modifying the repository. It contains the
current contributor rules, runtime-state warnings, and the load-bearing tool
registry convention.

## Dispatcher research

`prototypes/tool_dispatcher/` is an unusually substantial part of this project,
not a toy example. It contains the experiments that led to the current typed
tool-call boundary:

- xLAM, Hammer, Arch, Granite, GPT-OSS, and Qwen comparisons;
- broad, targeted, oracle-subset, automatic-subset, and frozen 450-case phases;
- semantic main-brain delegation;
- deterministic exact-payload binding;
- typed argument contracts;
- focused ontology and schema repairs;
- Hammer 1.5B training-dataset pilots.

The dispatcher lab never treats the small model as a conversational assistant.
Benchmark execution normally stops after generation, parsing, validation, and
scoring without running a Yuki tool.

Start with the
[`tool-dispatcher README`](prototypes/tool_dispatcher/README.md) and the
[`complete experiment history`](prototypes/tool_dispatcher/reports/YUKI-TOOL-DISPATCHER-ALL-MODELS-PHASES-REPORT.md).

## Documentation

[`docs/README.md`](docs/README.md) distinguishes current architecture from
historical handoffs and archived design notes. In particular:

- [`TOOL-ROUTING-ARCHITECTURES.md`](docs/TOOL-ROUTING-ARCHITECTURES.md) is the
  current routing reference.
- The architecture and TUI handoffs are dated inspection snapshots; they are
  useful context but parts have since been implemented or superseded.
- The old multi-user memory and free-text function-calling documents are kept
  only as design history.

## Safety and privacy

Yuki is single-user software with real local capabilities. Depending on the
enabled tools and integrations, she may read files, modify her own workspace,
run allowlisted shell commands, access the camera bridge, call network services,
or send a question to another agent.

- Review tool calls and configurations before trusting a new model.
- Keep `data/`, `yuki/`, and `output/` out of source control.
- Never publish API keys, private memories, session databases, or model caches.
- A local model keeps generation local, but web tools, OpenRouter models,
  embeddings, summaries, image generation, weather, and search can send data to
  external services.
- The project is not hardened as a remote multi-user service.

Yuki's weird, warm personality is intentional. The goal is not to sand her into
a generic assistant—it is to give her better memory, stronger tools, clearer
boundaries, and an interface that still feels unmistakably hers.
