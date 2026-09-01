# Yuki tool-routing architectures

Yuki now has two selectable tool-routing architectures. Both terminate at the
same strict call boundary and continue to use the existing 18-tool registry and
implementations as the source of truth.

## Modes

### Direct

```text
user request
  -> active conversational model
  -> provider-native tool call OR canonical JSON call
  -> canonical Yuki call {tool, arguments}
  -> strict schema validation
  -> typed exact-source check
  -> live Yuki argument adapter
  -> existing Yuki tool
  -> raw result
  -> active conversational model
```

`AUTO` uses native calls for a known native-capable model/backend and otherwise
uses canonical JSON. `CANONICAL` avoids native tool calling. `NATIVE` requests
native calls where the active backend supports them and safely falls back to
canonical routing elsewhere.

### Dispatcher

```text
user request + recent reference context
  -> active model: respond or semantic delegation + domain
  -> one stateless generation by the selected dispatcher
  -> dispatcher-native OR canonical output
  -> canonical Yuki call {tool, arguments}
  -> strict schema validation
  -> typed exact-source check against the raw current request
  -> live Yuki argument adapter
  -> existing Yuki tool
  -> raw result
  -> active conversational model
```

The dispatcher is not a chatbot. It gets no conversation history or previous
tool results, produces exactly one routing generation, cannot execute tools, and
never writes the final assistant answer. Only the active main model sees the raw
tool result and speaks as Yuki.

The main-brain decision receives a small recent-context window only to resolve
phrases such as "that file." The immutable current user message remains the
source of literal arguments.

### Autonomous

```text
conversation at the moment AUTO is enabled
  -> durable authorization/goal snapshot
  -> each idle cycle: WAIT | SPEAK | DELEGATE
       WAIT     -> remain quiet
       SPEAK    -> one meaningful Yuki update, no tool
       DELEGATE -> selected Direct or Dispatcher router
                -> canonical validation
                -> existing Yuki tool
                -> verified result
                -> Yuki response
```

Autonomous mode no longer inserts “the user has been silent” as a new request
and no longer uses the legacy `[TOOL: ...]` regex loop. Silence is simply the
clock signal for another private planning cycle. The conversation that existed
when AUTO was enabled remains available as durable goal/authorization context,
so a short follow-up such as “and go” is not buried by idle chatter.

The planner may choose any of the six domains, covering all 18 registered Yuki
tools. There is no action-count, time, or tool-category limit: it can continue
across idle cycles until AUTO is turned off. Each cycle performs at most one
tool-backed step so its result can inform the next decision. Image is no longer
blocked merely because the user is away.

Autonomous literal values use the planner’s frozen action request as their
auditable source and are labeled `autonomous_action`; user-directed calls remain
labeled `raw_user_request`. Both still require character-exact source spans.

## Canonical boundary

All model-specific outputs are normalized to:

```json
{
  "tool": "weather",
  "arguments": {
    "city": "Tokyo"
  }
}
```

The model cannot call an implementation directly. The boundary performs, in
order:

1. output parsing;
2. offered-tool membership checking;
3. strict per-tool JSON-schema validation;
4. typed literal-source enforcement;
5. prototype-only shell validation;
6. adaptation to the existing Yuki implementation's input shape.

Malformed, unknown, extra-field, rewritten-literal, or unsafe shell calls stop
before execution. Yuki is explicitly told that no tool ran.

## Exact-payload policy

The following fields currently require an exact character span from the raw
current request:

- `fetch.url`
- `search.query`
- `image.prompt`
- `read.filepath`
- `shell.command`
- Yuki filenames and write/append content
- `ask_claude.question`

The production gate is intentionally conservative. It verifies a model-emitted
span; it does not yet guess a span when the model rewrites or omits it. This
keeps the previously tested binder heuristics out of production until a
generalized boundary algorithm passes an independent holdout.

Semantic fields such as `weather.city`, `hardware.metric`, `see.target`,
`remember.fact`, and `recall.topic` use their strict schema but may be normalized
by the selected router.

For `yuki_write` and `yuki_append`, models see separate `filename` and `content`
fields. Only after validation are they adapted to the current live
`filename|content` interface, so a literal pipe inside content is preserved.

## TUI controls

Open `SETTINGS` and use:

- `TOOL ROUTING`: `DIRECT` or `DISPATCHER`
- `DISPATCHER MODEL`: the second model used only in dispatcher mode
- `TOOL PROTOCOL`: `AUTO`, `CANONICAL`, or `NATIVE`

The dispatcher model manager uses the same source organization as the main
model manager: Recent, Local, OpenRouter, and Remote. Local dispatcher
checkpoints under `/Volumes/madisk/yuki-tool-dispatcher` are discovered as well.
The selected dispatcher loads lazily on the first delegated request and unloads
when its model or protocol changes.

Power users can also use `/routing direct` or `/routing dispatcher`. The sidebar
always shows the active routing architecture and protocol.

Settings persist in `data/tui_settings.json`.

## Ownership and integration points

- `tools/__init__.py`: source of truth for names, live metadata, and functions.
- `tool_routing/core.py`: schemas, normalization, validation, exact-source gate,
  and live argument adaptation.
- `tool_routing/runtime.py`: lazy, stateless dedicated-dispatcher inference.
- `ms_llama.py`: main-brain decision, direct/native adapters, execution, raw
  result return, and final Yuki response.
- `interface/tui.py`: user selection, persistence, model manager, and visible
  routing status.
- `prototypes/tool_dispatcher/`: reused model dialect, prompt, parser, and backend
  work from the isolated experiments.

## Deliberate limits

- A local main model and a local dispatcher may both consume unified memory.
  The dispatcher is lazy and unloadable, but automatic main-model eviction is
  not part of this change.
- Literal binding abstains instead of guessing when it cannot prove an exact
  source span.
- This adds no new approval UI. Existing tool behavior remains in place, while
  the new boundary adds strict validation and a shell-operator block.
