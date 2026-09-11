# Textual TUI UX handoff for ChatGPT

> **Historical pre-redesign audit:** this report records the TUI before the
> retro-workspace UX redesign. Many recommendations below are now implemented,
> including labeled focusable controls, a unified activity state, a model
> manager, session drawer, explicit reasoning controls, and responsive reflow.
> Keep it as design rationale, not as a description of the current screen.

Inspection date: 2026-08-22

## Scope

This is a read-only interaction review of `interface/tui.py`, the TUI-specific state and widgets it directly uses, and its calls into `VoiceChatBot`. No model was loaded, the TUI was not started, no Yuki tool was executed, and no existing file was changed. The report evaluates behavior visible from the current code rather than judging a running screenshot.

The short diagnosis is: the TUI is technically capable and visually ambitious, but it exposes too many overlapping interaction systems. Mouse-only glyph controls, hidden keyboard commands, slash commands, expanding sidebar lists, and modal settings all lead to the same features in slightly different ways. Important state is compacted into symbols or disappears at smaller terminal sizes. The result can feel more like operating a cockpit than having a conversation.

## What the user currently sees

### Wide empty-state layout

```text
┌──────────────────────────┬──────────────────────────────────────────────────────────┐
│ ≡  yuki ⊹                │                    new chat  ⌄       ♥ 100% · ∿ off      │
│                          │                                                          │
│ »  search                │              ⊹ ࣪ ˖ ૮( ˶ᵔ ᵕ ᵔ˶ )っ                       │
│ +  new chat              │                                                          │
│ ❯  chats  ›              │              ██╗   ██╗██╗   ██╗██╗  ██╗██╗              │
│    [collapsed sessions]  │              ...large YUKI block logo...                │
│ λ  tools  ›              │                                                          │
│    [collapsed tools]     │                    yours for the voyage.                 │
│ Ψ  brain                 │               [ neural link: offline — no model ]        │
│ ◈  settings              │              click the ▣ pill below to pick a model      │
│ ▞  theme                 │                        ( ◉ yuki )                         │
│                          │                                                          │
│                          │       ╭────────────────────────────────────────────╮     │
│                          │       │ Message Yuki ...          ▣ pick model ⌄  │     │
│                          │       │ ^   »   >_          auto · chat       ↑    │     │
│                          │       ╰────────────────────────────────────────────╯     │
└──────────────────────────┴──────────────────────────────────────────────────────────┘
```

Meaning of the less obvious symbols:

- `^`: render help into the chat transcript;
- composer `»`: insert `/search ` for web search;
- `>_`: insert `/shell `;
- `auto · chat`: toggle autonomous mode;
- `↑`: send;
- sidebar `» search`: search prior conversation text, not the web;
- `Ψ brain`: show counts for facts, summarized sessions, and episodes;
- `∿ on/off`: TTS state in the top-right status.

The same `»` symbol therefore means two different kinds of search depending on location.

### After conversation begins

```text
┌──────────────────────────┬──────────────────────────────────────────────────────────┐
│ sidebar                  │ first user words ⌄                 ♥ 93% · ∿ on · 18 t/s │
│                          │                                                          │
│                          │                                      ╭───────────────╮   │
│                          │                                      │ user message  │   │
│                          │                                      ╰───────────────╯   │
│                          │ ⊹ yuki                                                   │
│                          │ assistant reply in Markdown                              │
│                          │                                                          │
│                          │ ⠹ thinking                                               │
│                          │ 🔧 Search…                                               │
│                          │ ✅ Search — done                                         │
│                          │ ⊹ yuki                                                   │
│                          │ final reply                                               │
│                          │                                                          │
│                          │       ╭────────────────────────────────────────────╮     │
│                          │       │ Message Yuki ...          ▣ model-name ⌄  │     │
│                          │       │ ^   »   >_          auto · chat       ↑    │     │
│                          │       ╰────────────────────────────────────────────╯     │
└──────────────────────────┴──────────────────────────────────────────────────────────┘
```

The top-center “session title” is not a real title. For a new session it changes to the first 24 characters of the first message. When resuming, it uses the first 24 characters of the last displayed user turn.

### Narrow layouts

At widths below 90 columns, the sidebar is automatically hidden and the top-right status becomes invisible. Below 68 columns, the help/search/shell buttons and autonomous-mode pill also disappear. The features remain active, but the screen no longer exposes their state or entry points. `Ctrl+B` can reopen the sidebar, but that shortcut is hidden from Textual's normal key-binding display.

## 1. Screen and layout structure

The structure is a fixed 28-column sidebar plus one main column containing a one-line top bar, scrollable chat, and centered composer card. The composer is 72% of the main width, with a minimum width of 60 and maximum of 100 cells. Chat content has eight columns of horizontal padding on each side until the narrow breakpoint.

What works:

- Clear separation between navigation, conversation, and composer.
- The empty state has personality and communicates whether a model is loaded.
- User and assistant messages are visually distinct.
- Markdown replies and inline terminal images make the TUI more capable than a plain chat log.
- Responsive classes reduce decoration on short/narrow terminals.

Why it can feel awkward:

- The sidebar consumes a large fixed width even though most of its content is collapsed.
- The decorative hero and floating composer prioritize visual resemblance to a web app over terminal information density.
- The top-center chat pill looks like a dropdown, but merely toggles the session list inside the sidebar.
- “New chat” appears both as a sidebar action and as the initial top pill text, although clicking the pill does not create a new chat.
- The composer contains six controls in two rows, several represented only by punctuation or terminal glyphs.
- The explicit modal-centering/overlay CSS covers `SearchPicker`, `SettingsScreen`, `TextEntry`, and `ChatSearchScreen`, but omits `MenuPicker` and `RemoteSetupScreen`. Different steps of the model flow may therefore receive inconsistent placement/backdrop treatment.
- At responsive breakpoints, controls disappear instead of moving into an accessible overflow menu.
- When the sidebar is auto-hidden, clicking the top session pill toggles a list inside that still-hidden sidebar, so it can appear to do nothing.

The layout is polished in isolation, but its visual hierarchy does not reliably communicate which elements are state, which are buttons, and which are menus.

## 2. Keyboard navigation and shortcuts

Global bindings are:

```text
Ctrl+N  new session
Ctrl+B  toggle sidebar
Ctrl+T  toggle TTS
Ctrl+Q  quit
Escape  focus composer
Enter   submit composer / select focused list item
Up/Down composer history
Tab     complete a slash command when there is one unambiguous match
```

All four Ctrl bindings are declared with `show=False`. They do not appear in the normal Textual footer. The empty hero mentions only `Ctrl+B`; `/help` lists slash commands but not the complete keyboard map.

The largest accessibility issue is that most primary controls are `Clicky`, a subclass of `Static` with an `on_click` handler. They are not defined as normal focusable buttons. The sidebar navigation rows, session pill, model pill, glyph shortcuts, auto toggle, and send control are designed around clicking. A keyboard user instead has to learn the parallel command/shortcut system.

Examples:

- There is a clickable tools expander, but no `/tools` TUI command and no direct shortcut to focus its list.
- The sidebar chat list can be toggled via click or `/sessions`, but neither path explicitly moves focus into the list.
- The model pill is clickable; keyboard users need `/model` because Tab navigation is not presented as a reliable route to it.
- Search/model picker modals implement “Down moves from filter to results,” but moving back to the filter is not explained.
- The model filter accepts Enter while the filter is focused and immediately picks the first result. This is fast, but easy to trigger without inspecting similarly named models.
- Chat search does not have the same Enter-on-filter behavior. Its hint says Enter can pick a hit, but the user must first move focus into the result list.

The TUI technically supports keyboard use, but it has two separate navigation grammars: mouse-driven visible controls and command-driven invisible controls.

## 3. Input flow

The composer is a single-line `Input`, not a multiline text area. Enter submits immediately. There is no Shift+Enter/newline path, draft expansion, attachment affordance, or visible character/context limit.

The input sequence is:

```text
submit
  -> strip text
  -> clear composer immediately
  -> save text in in-memory input history
  -> handle quit/help/TUI command/direct tool
  -> if no model, warn and open model picker
  -> if busy, warn
  -> otherwise render user bubble and call VoiceChatBot.react_chat in a worker
```

Clearing the composer before checking model/busy state is a significant pain point. If a model is loading, replying, speaking, or running an autonomous turn, the user's message disappears and a toast says to wait. It can be recovered with Up because it was added to the composer history, but that recovery mechanism is not communicated.

Other issues:

- Exact messages `quit`, `exit`, or `bye` close the app. A natural “bye” intended for Yuki never reaches her.
- The send control remains visually available while `_busy` is true; it is not disabled or changed to Stop/Cancel.
- There is no queued-send behavior and no visible draft retention.
- Direct slash tools are checked before the normal busy guard. They can start while model work is active and do not participate in the same UI busy state.
- Up/Down history is useful but lives only for the current TUI process; it is separate from stored chat history.
- Tab completion shows up to eight ambiguous matches in a temporary toast rather than a persistent inline command palette.

## 4. Model switching flow

The normal model flow is deliberately multi-stage:

```text
click model pill or type /model
  -> modal 1: choose local / OpenRouter / remote
       -> local/OpenRouter: background model-list fetch
            -> modal 2: filter and select model
                 -> loading state
       -> remote: modal 2: base URL / key / optional model ID
            -> direct load, or background list
                 -> modal 3: filter and select remote model
                      -> loading state
```

The implementation is technically strong: list gathering and model loading happen off the UI thread, failed switches leave the old bot in place, and successful switches preserve conversation history, episodic UUID, patience, and TTS.

The UX cost is high:

- The source-selection step appears every time, even when the user usually switches within one source.
- Missing OpenRouter credentials produce a toast telling the user to open Settings. The user must leave the picker, open Settings, enter the key, close/return, and restart model selection.
- A failed remote model-list request says to enter the model manually, but does not reopen the remote form or provide an inline action.
- Remote configuration combines connection setup and model selection in one three-field form. Pressing Enter from any field submits the entire form, making premature submission easy.
- The current model label is truncated to 18 characters and reduced to the last path component after loading. Local/cloud/remote source, quantization, backend, and memory impact are not visible in normal chat.
- During a switch, the status says loading and a toast appears, but there is no progress detail, elapsed time, cancel button, or explicit “old model remains active if this fails” message.
- A saved model may automatically start loading on launch without a confirmation or an easy way to choose a lighter model first.
- Mid-chat continuity is clever but not obvious. “Same chat, same memory” appears only as a transcript system line after success.

A single model manager with source tabs, recent models, current-model details, credentials inline, and one selection step would reduce much of this friction.

## 5. Session and chat browsing

The TUI does not use web chat JSON files. Its session browser reads `session_summaries` and `episodes` directly from the episodic SQLite database.

Current flow:

```text
new chat
  -> clear VoiceChatBot.history
  -> mint new session_uuid
  -> reset patience to 100
  -> show empty hero
  -> summarize old session in a background worker
  -> refresh summarized-session sidebar

resume
  -> read selected session transcript
  -> replace current chat view without confirmation
  -> display latest 30 stored turns
  -> load latest 10 pairs into VoiceChatBot.history
  -> reuse selected session_uuid
```

Strengths:

- Session list and full-text search are local and inexpensive.
- Resume hydrates both the UI and model history.
- Background summarization avoids freezing the interface.

Pain points:

- Only summarized sessions appear in the recent list. The current active session may not be listed or identifiable there.
- Labels are fuzzy time plus the first 20 characters of an AI-generated summary. There are no user titles, stable names, active marker, turn count, model, or preview beyond that tiny fragment.
- The top session pill uses raw user-message fragments rather than the summary label, so the same session has different identities in different places.
- Selecting a past session immediately replaces the current view. The outgoing session is not summarized at that switch point, and there is no confirmation before the current context is displaced.
- Chat search results show time plus a 60-character match fragment, then immediately resume the selected UUID. There is no preview screen.
- There is no rename, pin, delete, export, duplicate, or “open in new session” flow.
- The recent-session list is capped at 40 and only 12 rows high, nested inside the fixed sidebar.
- On a narrow terminal the session list is hidden with the sidebar. The top session pill does not open a separate drawer.
- The distinction between a “session,” “chat,” episodic UUID, summary, and current visible transcript is not explained to the user.

## 6. Settings flow

Settings is one flat modal list with nine rows:

```text
chat model
persona
voice (TTS engine on/off)
TTS voice
remote endpoint
image backend
OpenRouter key
autonomous interval
theme
```

The row values are compact and useful, but selecting a row can mean very different things:

- model/persona/remote: close Settings and open another modal flow;
- image/OpenRouter key: stack a text-entry modal over Settings;
- TTS: toggle/load immediately;
- TTS voice: advance to the next voice ID immediately;
- autonomous: advance through a preset cycle immediately;
- theme: advance through a seven-theme cycle immediately.

That inconsistency increases cognitive load. There is no stable detail pane, explicit Save/Cancel transaction, or description of what Enter will do to a row.

Specific awkwardness:

- Voice choice cycles through identifiers such as `af_heart` and `bm_george` one activation at a time; there is no named picker or audio preview.
- Autonomous settings cycle `off → 60 → 120 → 300 → 600` rather than opening an explicit choice.
- Theme cycles through seven values, making comparison and reversal tedious.
- OpenRouter and remote keys are persisted to `data/tui_settings.json`, but the UI does not explain that credentials are being stored locally in plain text.
- “Image backend” exposes a raw worker URL, which is an implementation detail rather than a user-level choice.
- Model setup appears both here and in the composer model pill.
- Remote endpoint setup appears both as a settings row and as a model-source option.

Settings should be categorized into Model, Voice, Behavior, Services, and Appearance, with explicit value controls rather than activation cycles.

## 7. Tool progress and status presentation

Model-selected tools and direct slash tools have different presentation paths.

Model-selected tool:

```text
user bubble
animated “thinking” line
“🔧 Tool…” transcript line
“✅ Tool — done” transcript line
final Yuki bubble
optional image
```

Direct slash tool:

```text
user bubble
[no initial busy/progress indicator]
“λ /tool” transcript line
boxed raw tool output
optional image
```

The callback from `VoiceChatBot` into the TUI is technically clever: tool progress emitted from worker threads is safely mounted into the chat. But the UX is noisy and inconsistent.

Problems:

- Slow direct tools provide no initial indication that work started.
- Direct tools do not set `_busy`, so their activity is absent from global status and can overlap other work.
- The model-selected path leaves the generic thinking spinner present while adding separate start/done lines.
- Tool progress permanently occupies transcript space rather than collapsing into one result card.
- Tool errors can look like ordinary raw output; there is no consistent success/warning/error visual grammar.
- Search tools can return `(summary, links)`, but the TUI discards the links value in the direct path, and the shared model tool runner also takes only the first tuple item. Users may see search summaries without the actual URLs.
- The tools sidebar only inserts slash-command text into the composer. Selecting a tool may feel like an action, but it merely stages a command.
- There is no visible permission/confirmation step for potentially sensitive direct tools.

A single collapsible activity card should show: selected tool, arguments, running/succeeded/failed state, duration, and optional raw details.

## 8. TTS controls

TTS can be controlled through `Ctrl+T`, `/tts`, or the Settings TTS row. Current on/off state appears as `∿ on/off` in the top-right status when that status is visible.

Good implementation details:

- TTS loads in a worker when needed.
- The engine object is cached when toggled off, allowing quick re-enable.
- Voice preference is persisted.
- TTS failure becomes a toast rather than crashing chat.

UX issues:

- The `∿` glyph is not self-explanatory.
- There is no visible clickable speaker control in the normal wide composer.
- Voice selection is an eight-step cycle of technical IDs with no preview.
- There is no per-message replay, stop-speaking, volume, speed, or “speak this response” control.
- `VoiceChatBot.speak` is called while the bot worker lock remains held. The reply appears, but `_busy` is not cleared until speech playback returns. A user who sends during playback loses the composer text and gets “still replying.”
- Turning TTS off after playback has started does not visibly provide a way to stop the already-running audio process.
- On narrow layouts the entire status containing TTS state disappears.

The simplest redesign is an explicit speaker button in the composer/header, a small Voice settings panel with friendly voice names and preview, and a Stop control while speaking.

## 9. Autonomous mode controls

Autonomy has three separate control models:

- composer pill: toggle the current interval on/off;
- Settings row: cycle through preset intervals;
- `/auto`: accept custom seconds or `off`.

The scheduler checks every five seconds and runs when the idle duration reaches the configured threshold. The default is normally the saved value or 120 seconds, so autonomous behavior may already be enabled on first launch.

UX problems:

- The composer pill shows only `auto · chat`; it does not show the interval.
- At widths below 68 columns the pill disappears even if autonomy remains active.
- The normal status line does not include autonomous state or the next trigger time.
- An autonomous run sets `_busy` but does not show the standard thinking indicator. The model may be working while the interface looks idle.
- If the user submits during that hidden busy period, the message is cleared and rejected.
- Exceptions and “NO” autonomous outcomes are silent.
- Autonomous responses are not recorded through the same explicit user-turn flow, making their relationship to session history less obvious.
- `/watch` controls the eye and changes what autonomy can see, but that coupling is only explained in a system line after using the command.

Autonomy should be off by default for a new user, visibly show its interval/state, and surface “Yuki is considering an autonomous message” with a cancel option.

## 10. Error and loading states

Current loading/error feedback is mostly temporary Textual notifications plus transcript lines.

What works:

- Expensive model and TTS loads run off the UI thread.
- Chat exceptions are converted through `_cli_friendly_error`.
- Failed model switches keep the previous bot object.
- The top status can show model loading.

What is weak:

- Toasts disappear, so configuration and connection failures leave no durable evidence or retry action.
- Chat/model errors are sometimes rendered in the same assistant style as actual Yuki replies, weakening the distinction between product failure and model output.
- Model loading has no percentage, phase, elapsed time, memory estimate, or cancel control.
- `VoiceChatBot` and backend loaders print some diagnostics directly rather than routing them into a structured TUI loading component; those details may be invisible or visually noisy depending on terminal/stdout handling.
- The composer remains active during loading even though submissions will be rejected after the text is cleared.
- Database-read failures return empty lists/counts. “No sessions” is indistinguishable from “database unavailable.”
- Session summarization and autonomous exceptions are silently swallowed.
- Settings-save errors are silently ignored.
- A remote-list failure recommends manual entry but does not return the user to a manual-entry action.
- Direct tool work has no shared loading state.
- There is no persistent diagnostics/status drawer despite extensive internal state.

Use stable inline banners/cards for important failures, with Retry, Change settings, Copy details, and Dismiss actions.

## 11. Actions requiring menus or context switches

The highest-friction journeys are:

### First OpenRouter model without a key

```text
model pill
  -> source modal
  -> OpenRouter
  -> missing-key toast
  -> open Settings
  -> select OpenRouter key
  -> stacked text modal
  -> close Settings
  -> reopen model pill
  -> source modal again
  -> model filter modal
  -> load
```

### Remote server with model discovery

```text
model pill
  -> source modal
  -> remote form
  -> enter URL/key
  -> wait for model list
  -> model filter modal
  -> load
```

### Choose a specific TTS voice

```text
Settings
  -> TTS on/load
  -> repeatedly activate “TTS voice” until the desired technical ID appears
  -> no preview to verify it
```

### Find and resume an old chat on a narrow terminal

```text
remember hidden Ctrl+B shortcut
  -> open sidebar
  -> use tiny nested session list
or
  -> sidebar search modal
  -> type query
  -> Down into results
  -> Enter to immediately replace current session
```

## 12. Hidden or unclear state

Important state that is absent, compressed, or conditionally hidden:

- whether a generation, model load, direct tool, TTS playback, or autonomous turn owns the app;
- autonomous interval and next trigger;
- current model source/backend and whether it is local, cloud, or remote;
- current session's stable identity and whether it has been summarized;
- why only some sessions appear in the sidebar;
- the fact that model switching preserves history/session/patience;
- TTS voice identity except inside Settings;
- API keys are persisted locally;
- current tool activity for direct slash commands;
- whether session summarization failed;
- active memory contents—the “brain” page shows counts only;
- the current persona outside the empty hero and assistant labels;
- all top-right status at widths below 90;
- all auto/helper controls at widths below 68;
- global shortcuts because every binding is hidden.

## 13. Likely new-user confusion

- “Search” in the sidebar searches chats; the identical `»` near the composer inserts web search.
- “Brain” sounds like memory browsing but displays only counts.
- “New chat ⌄” looks like a create/dropdown control but toggles an existing-session list.
- The composer uses `^`, `»`, `>_`, `∿`, `Ψ`, `▣`, and `λ` without persistent labels.
- A user can type before selecting a model; submission then clears the message and opens model selection.
- A saved model may start loading automatically without explaining where it came from.
- The model may still be speaking after the reply is visible, but the app continues to call that state “replying.”
- “Chats” are actually episodic summaries rather than the web UI's persisted chat objects.
- Selecting a tool from the sidebar only inserts a command; it does not open a form or explain the required argument.
- Autonomous mode may be active by default and can begin hidden background work.

## 14. Technically clever but UX-heavy features

- Mid-chat model replacement preserves live history, session UUID, patience, and TTS. This is valuable, but the multi-modal picker and weak state explanation hide the benefit.
- TUI session browsing reads SQLite directly without loading sqlite-vec or calling an API. Efficient, but the UI exposes summary artifacts rather than recognizable chat titles.
- The tool progress callback safely crosses from model worker threads into Textual. Good engineering, but it creates multiple permanent status lines instead of one coherent activity component.
- Responsive breakpoints mirror a web UI in the terminal. Visually clever, but essential state is removed rather than reflowed.
- Textual-image is imported before the app owns the terminal so protocol detection works. The result is useful inline imagery, though its surrounding tool flow is inconsistent.
- TTS object caching makes toggling fast, but the toggle semantics and persisted `tts_pref` are difficult to understand from the screen.
- Input history and slash completion are useful power-user features, but they form another interaction layer that is not taught.
- Autonomous vision via `/watch` is distinctive, but coordinating eye lifetime, auto mode, idle interval, and hidden busy state is UX-heavy.

## 15. Existing features that are hard to discover

- `Ctrl+N`, `Ctrl+T`, and `Ctrl+Q`.
- Up/Down composer history.
- Tab completion for slash commands.
- `/stats` detailed runtime/memory display.
- `/watch [seconds|off]` and its relationship to autonomy.
- `/auto <custom seconds>` beyond preset intervals.
- `/persona` and `/model` as keyboard alternatives to menus.
- Chat full-text search across both user and assistant messages.
- TUI session resume restores 10 model-context pairs while displaying 30 turns.
- Mid-chat model switching preserves conversation state.
- Remote OpenAI-compatible endpoint support.
- Inline image rendering and its path fallback.
- `Escape` always restores composer focus.

## 16. Duplicated features and controls

| Feature | Duplicate entry points |
| --- | --- |
| New chat | sidebar, `Ctrl+N`, `/new` |
| Session list | sidebar Chats, top session pill, `/sessions` |
| Model picker | composer model pill, Settings row, `/model` |
| Persona | Settings row, `/persona` |
| TTS toggle | `Ctrl+T`, Settings row, `/tts` |
| Autonomous mode | composer pill, Settings cycle, `/auto` |
| Theme | sidebar, Settings cycle, `/theme` |
| Tool discovery | sidebar tool list, `/help`, Tab completion |
| Web search tool | composer glyph, tools sidebar, `/search` |
| Shell tool | composer glyph, tools sidebar, `/shell` |
| Memory/runtime counts | sidebar Brain, `/stats` |

Duplication is not inherently bad when mouse and keyboard affordances mirror one another. Here, however, the mirrors often have different capability: only `/auto` accepts arbitrary intervals, only `/stats` shows last reply details, only Settings selects a voice, and only some paths focus their resulting list.

## 17. What should stay, be simplified, or be redesigned

### Keep

- `VoiceChatBot` reuse rather than a TUI-specific agent loop.
- Worker-thread model generation/loading and the bot lock.
- Searchable episodic sessions and local transcript retrieval.
- Mid-chat model continuity.
- Markdown assistant replies and inline images.
- Input history and slash completion as optional power-user features.
- A visible current model control near the composer.
- Tool progress crossing safely into the UI.
- Responsive layout, but with state reflow rather than state removal.
- Yuki's personality and visual identity.

### Simplify

- Replace glyph-only controls with short labels/icons plus tooltips/help text.
- Keep one visible primary path per feature; leave slash commands as power-user mirrors.
- Combine Brain and Stats into one inspectable Status/Memory panel.
- Replace cycling settings with explicit pickers.
- Make one model picker handle source, credentials, recent models, and selection.
- Keep one persistent session drawer with search rather than an expander plus separate modal.
- Reduce permanent tool transcript lines to one collapsible activity card.
- Do not clear composer text until work is accepted.
- Separate speaking from generating in the busy/status model.

### Redesign

- Introduce an explicit app-state model: `idle`, `loading_model`, `generating`, `running_tool`, `speaking`, `autonomous`, `error`.
- Make controls focusable and keyboard-reachable, using actual buttons/list actions rather than click-only `Static` widgets.
- Add Send → Stop during active generation and preserve/queue drafts.
- Give sessions stable titles, active indicators, previews, and safe switching.
- Make model identity/source and autonomy/TTS state visible at every terminal width.
- Rework Settings into categories with direct values and explanations.
- Add persistent actionable error cards and a diagnostics drawer.
- Treat direct tool commands and model-selected tools through one UI activity/policy path.

## Current TUI flow

```text
launch
  -> load saved settings
  -> render TUI immediately
  -> if saved/CLI-selected model exists: load it in background
     else: show offline hero and temporary “pick a model” notice

normal message
  -> composer clears
  -> busy/model checks
  -> user bubble + thinking spinner
  -> VoiceChatBot.react_chat in worker
  -> optional tool-progress transcript lines
  -> episodic record
  -> final assistant bubble + optional image
  -> TTS playback
  -> clear busy state only after playback finishes

session change
  -> New: clear history, mint UUID, reset patience, summarize previous in background
  -> Resume: immediately replace view/history and reuse selected UUID

model change
  -> source modal
  -> source-specific configuration/list modal
  -> background load
  -> copy old history/UUID/patience/TTS on success

autonomous mode
  -> poll every 5 seconds
  -> after configured idle threshold, set hidden busy state
  -> VoiceChatBot.autonomous_tick
  -> optional reply/TTS or silent completion
```

## Top 10 UX pain points

1. Composer text is cleared before no-model/busy checks, so rejected submissions appear lost.
2. Visible controls are mostly click-oriented `Static` widgets, while keyboard equivalents are hidden commands/shortcuts.
3. Model selection requires cascading modals and sends missing-credential users on a settings detour.
4. Responsive layouts hide sidebar, status, TTS/autonomy state, and helper controls while the features remain active.
5. Session identity is inconsistent: summary fragment in the sidebar, message fragment in the top pill, UUID internally.
6. TTS playback keeps the app busy after the reply appears, with no Stop control and misleading “still replying” feedback.
7. Autonomous work can start with no spinner, block input invisibly, and remain enabled when its control is hidden.
8. The same actions are duplicated across sidebar, composer, Settings, slash commands, and hidden shortcuts with different capabilities.
9. Tool presentation is inconsistent and noisy; direct tools have no start state, model tools create several transcript lines, and search links are discarded.
10. Temporary toasts and silent exception handling make failures hard to understand or recover from.

## What I would simplify first

Priority order:

1. Preserve composer drafts and disable/transform Send while work is active.
2. Replace `Clicky(Static)` primary controls with focusable buttons and publish the keyboard map.
3. Build one model manager with Local/OpenRouter/Remote tabs and inline credential setup.
4. Replace the current session expander/search split with one session drawer that works at every width.
5. Make the status model explicit: model, generation, tool, speaking, TTS, and autonomy.
6. Replace settings cycles with actual selectors, especially voice, autonomy, and theme.
7. Collapse tool activity into one consistent card and route direct tools through the same busy/progress UI.
8. Keep slash commands, but treat them as optional power-user shortcuts rather than the only discoverable route to advanced features.

## Suggested redesigned layout

### Wide terminal

```text
┌──────────────────────┬────────────────────────────────────────────────────────────┐
│ YUKI                 │ Chat: Project architecture                  [New] [⋯]      │
│                      │ Model: Qwen3… · Local MLX   Voice: On   Auto: Off          │
│ [Search chats]       ├────────────────────────────────────────────────────────────┤
│                      │                                                            │
│ ● Current chat       │                                      ┌──────────────────┐ │
│   Project arch...    │                                      │ user message     │ │
│   Yesterday · Notes  │                                      └──────────────────┘ │
│   Last week · Setup  │                                                            │
│                      │ Yuki                                                       │
│ [All chats]          │ response…                                                  │
│                      │                                                            │
│ Tools                │ ┌ Tool: Search ─────────────────────────────── succeeded ┐ │
│ Memory & status      │ │ query · duration · [show raw details]                  │ │
│ Settings             │ └────────────────────────────────────────────────────────┘ │
│                      │                                                            │
│                      ├────────────────────────────────────────────────────────────┤
│                      │ ┌ Message Yuki… multiline                                 │
│                      │ │                                                          │
│                      │ └──────────────────────────────────────────────────────────│
│                      │ [Tools] [Voice] [Auto]             1,240 ctx      [Send]   │
└──────────────────────┴────────────────────────────────────────────────────────────┘
```

### Narrow terminal

```text
┌──────────────────────────────────────────────────────┐
│ ☰  Project architecture          Qwen3…   ● Idle    │
│ Voice On · Auto Off                                  │
├──────────────────────────────────────────────────────┤
│ conversation                                         │
│                                                      │
│ [Search succeeded · 2.4s · details]                  │
├──────────────────────────────────────────────────────┤
│ Message Yuki…                                        │
│ [Tools] [Voice] [Auto]                   [Send/Stop] │
└──────────────────────────────────────────────────────┘
```

The narrow header's menu should open a real drawer containing sessions, search, model, memory/status, and settings. No active state should disappear merely because width decreases.

### Unified model manager

```text
┌ Select model ─────────────────────────────────────────┐
│ [Recent] [Local] [OpenRouter] [Remote]                │
│ Search models…                                        │
│                                                      │
│ ● Hammer2.1-1.5B    Local MLX   FP16    ~3.6 GB      │
│   Qwen3…             Local MLX   4-bit                │
│   qwen3.8-27b        OpenRouter  API                  │
│                                                      │
│ OpenRouter key: Not set                 [Configure]   │
│ Current chat and memory will be preserved.            │
│                                      [Cancel] [Load]  │
└──────────────────────────────────────────────────────┘
```

## Important TUI code sections for ChatGPT to inspect

Suggested order:

1. `interface/tui.py:259` — `Clicky`, `NavItem`, and list-item primitives; central to mouse/keyboard accessibility.
2. `interface/tui.py:323` — source menu, remote form, filterable picker, text entry, Settings, and chat-search modal behavior.
3. `interface/tui.py:586` — single-line Composer history and slash completion.
4. `interface/tui.py:635` — complete embedded Textual CSS, fixed sizing, modal styling, and responsive hiding rules.
5. `interface/tui.py:811` — hidden global bindings and app state fields.
6. `interface/tui.py:874` — actual screen composition and meanings of sidebar/topbar/composer controls.
7. `interface/tui.py:910` — startup, saved-model auto-load, focus, timers, and first-run notice.
8. `interface/tui.py:944` — responsive breakpoints that hide sidebar, status, helper buttons, and auto state.
9. `interface/tui.py:981` — hero, model/status chrome, synthetic session title, and summarized-session sidebar.
10. `interface/tui.py:1050` — all click/navigation actions and duplicated entry points.
11. `interface/tui.py:1131` — Settings row actions and cycle-based controls.
12. `interface/tui.py:1193` — message/tool/image rendering and transcript animation.
13. `interface/tui.py:1254` — thinking indicator lifecycle.
14. `interface/tui.py:1281` — composer clearing, command routing, busy check, and chat submission.
15. `interface/tui.py:1326` — `VoiceChatBot.react_chat`, episodic record, final rendering, TTS, and delayed busy reset.
16. `interface/tui.py:1349` — direct tool worker and inconsistent progress/link handling.
17. `interface/tui.py:1366` — autonomous polling and hidden busy behavior.
18. `interface/tui.py:1388` — help and TUI-only slash commands.
19. `interface/tui.py:1463` — full local/OpenRouter/remote model-selection cascade.
20. `interface/tui.py:1582` — safe mid-chat bot replacement and preserved state.
21. `interface/tui.py:1649` — tools/session selection, session resume, and context hydration.
22. `interface/tui.py:1690` — new-session reset and background summary.
23. `interface/tui.py:1735` — TTS caching/loading/toggle state.
24. `interface/tui.py:1782` — startup model matching and persisted configuration restoration.
25. `ms_llama.py:1096` — `VoiceChatBot` state used by the TUI.
26. `ms_llama.py:1204` — backend generation and history behavior.
27. `ms_llama.py:1380` — tool progress callback and raw tool execution.
28. `ms_llama.py:1504` — native/regex tool-loop selection called by the TUI.
29. `ms_llama.py:1771` — autonomous model/tool loop.
30. `ms_llama.py:1827` — blocking TTS synthesis/playback called before the TUI clears `_busy`.
