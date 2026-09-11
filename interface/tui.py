"""Yuki's Textual workspace: a usable neon terminal, not a second runtime.

The TUI owns presentation, navigation, and explicit UI state. Models, tools,
memory, TTS, and autonomous behavior still use :class:`VoiceChatBot`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

sys.path.append(str(Path(__file__).parent.parent))

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, Input, Label, ListItem, ListView, Markdown, Static

import episodic
from ms_llama import (
    MAX_HISTORY_TURNS,
    REASONING_MODES,
    TOOLS,
    VoiceChatBot,
    _backend_self_awareness,
    _cli_friendly_error,
    _cli_persist_session_events,
    _cli_record_episode,
    _cli_record_local_episode,
    _is_tool_error,
    _load_facts,
    fetch_openrouter_models,
    init_tts,
    list_models,
    list_personas,
    list_tts_engines,
    load_persona_by_name,
    normalize_reasoning_mode,
    openrouter_reasoning_is_mandatory,
    run_tool,
    supports_native_tools,
)
from openrouter_providers import fetch_model_providers, provider_label
from tool_routing import (
    DEFAULT_DISPATCHER_MODEL,
    ROUTING_MODES,
    ROUTING_PROTOCOLS,
    normalize_routing_mode,
    normalize_routing_protocol,
)

TUI_COMMANDS = {
    "/model": "open the model manager",
    "/routing": "choose DIRECT or DISPATCHER tool routing",
    "/reasoning": "choose AUTO, effort level, or OFF",
    "/persona": "choose Yuki's persona",
    "/settings": "open explicit settings",
    "/tts": "toggle voice playback",
    "/new": "start a fresh session",
    "/auto": "/auto off · /auto 300",
    "/watch": "/watch [secs|off] — control the eye",
    "/stats": "show model and memory status",
    "/sessions": "open the session drawer",
    "/theme": "choose a terminal theme",
    "/quit": "close the TUI",
}

CYBER_THEME = Theme(
    name="cyber",
    primary="#ff355e",
    secondary="#00e5ff",
    accent="#ff1744",
    foreground="#c8e6ff",
    background="#050914",
    surface="#0a1426",
    panel="#02050d",
    success="#5cff9d",
    warning="#ffd166",
    error="#ff355e",
    dark=True,
)
ODYSSEUS_THEME = Theme(
    name="odysseus",
    primary="#e06c75",
    secondary="#355a66",
    accent="#e06c75",
    foreground="#9cdef2",
    background="#282c34",
    surface="#21252b",
    panel="#111111",
    success="#50fa7b",
    warning="#f0ad4e",
    error="#e06c75",
    dark=True,
)
THEME_CYCLE = [
    "cyber", "odysseus", "tokyo-night", "nord", "gruvbox",
    "catppuccin-mocha", "dracula",
]
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
YUKI_LOGO = """\
██╗   ██╗██╗   ██╗██╗  ██╗██╗
╚██╗ ██╔╝██║   ██║██║ ██╔╝██║
 ╚████╔╝ ██║   ██║█████╔╝ ██║
  ╚██╔╝  ██║   ██║██╔═██╗ ██║
   ██║   ╚██████╔╝██║  ██╗██║
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝"""
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_sky", "af_nicole", "am_adam",
    "am_michael", "bf_emma", "bm_george",
]
VOICE_LABELS = {
    "af_heart": "Heart · warm feminine",
    "af_bella": "Bella · bright feminine",
    "af_sky": "Sky · soft feminine",
    "af_nicole": "Nicole · calm feminine",
    "am_adam": "Adam · clear masculine",
    "am_michael": "Michael · warm masculine",
    "bf_emma": "Emma · British feminine",
    "bm_george": "George · British masculine",
}
AUTO_STEPS = [0, 60, 120, 300, 600]
MIN_AUTO_INTERVAL = 15
REASONING_CHOICES = [
    ("auto", "AUTO · follow model requirements/default"),
    ("minimal", "MINIMAL · shortest reasoning budget"),
    ("low", "LOW · light reasoning"),
    ("medium", "MEDIUM · balanced reasoning"),
    ("high", "HIGH · deepest available reasoning"),
    ("off", "OFF · disable reasoning where supported"),
]
ROUTING_CHOICES = [
    ("direct", "DIRECT · active conversational model selects tools"),
    ("dispatcher", "DISPATCHER · main brain delegates to a second model"),
]
ROUTING_PROTOCOL_CHOICES = [
    ("auto", "AUTO · use each selected model's recommended format"),
    ("canonical", "CANONICAL JSON · provider-neutral structured output"),
    ("native", "MODEL-NATIVE · native API or trained dispatcher format"),
]
SETTINGS_PATH = Path(__file__).parent.parent / "data" / "tui_settings.json"
IMAGES_DIR = Path(__file__).parent.parent / "yuki" / "images"


@dataclass
class ActivityState:
    """The one foreground operation. Speech deliberately lives elsewhere."""

    kind: str = "idle"
    label: str = "IDLE"
    started_at: float | None = None
    cancellable: bool = False
    cancel_requested: bool = False
    token: int = 0

    @property
    def blocks_input(self) -> bool:
        return self.kind not in {"idle", "error"}


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")
    except OSError:
        pass


def _ro_db() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{episodic.DB_PATH}?mode=ro", uri=True)


def read_recent_sessions(limit: int = 40) -> list[dict]:
    """Return every stored transcript, whether or not it has a summary yet.

    Summary generation is best-effort and may not finish before the TUI exits.
    The transcript itself is the durable source of truth for the session drawer.
    """
    try:
        db = _ro_db()
        try:
            rows = db.execute(
                "WITH activity AS ("
                "SELECT session_uuid, MAX(ts) AS ts_end FROM episodes "
                "GROUP BY session_uuid) "
                "SELECT a.session_uuid, a.ts_end, COALESCE(s.summary, ''), "
                "COALESCE((SELECT e.user_msg FROM episodes e "
                "WHERE e.session_uuid = a.session_uuid ORDER BY e.ts LIMIT 1), '') "
                "FROM activity a LEFT JOIN session_summaries s "
                "ON s.session_uuid = a.session_uuid "
                "ORDER BY a.ts_end DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            db.close()
    except sqlite3.Error:
        return []
    return [
        {"uuid": session_uuid, "ts_end": ts, "summary": summary, "title": title}
        for session_uuid, ts, summary, title in rows
    ]


def read_transcript(session_uuid: str) -> list[tuple[str, str]]:
    try:
        db = _ro_db()
        try:
            rows = db.execute(
                "SELECT user_msg, assistant_msg FROM episodes "
                "WHERE session_uuid = ? ORDER BY ts", (session_uuid,),
            ).fetchall()
        finally:
            db.close()
    except sqlite3.Error:
        return []
    return [(user, assistant) for user, assistant in rows]


def search_episodes(needle: str, limit: int = 20) -> list[dict]:
    pattern = "%" + needle.replace("%", r"\%").replace("_", r"\_") + "%"
    try:
        db = _ro_db()
        try:
            rows = db.execute(
                "SELECT session_uuid, ts, user_msg, assistant_msg FROM episodes "
                r"WHERE user_msg LIKE ? ESCAPE '\' OR assistant_msg LIKE ? ESCAPE '\' "
                "ORDER BY ts DESC LIMIT 200", (pattern, pattern),
            ).fetchall()
        finally:
            db.close()
    except sqlite3.Error:
        return []
    seen: dict[str, dict] = {}
    low = needle.casefold()
    for session_uuid, ts, user_msg, assistant_msg in rows:
        if session_uuid in seen:
            continue
        text = user_msg if low in user_msg.casefold() else assistant_msg
        start = max(0, text.casefold().find(low) - 20)
        seen[session_uuid] = {
            "uuid": session_uuid,
            "ts": ts,
            "snippet": text[start:start + 60].replace("\n", " "),
        }
        if len(seen) >= limit:
            break
    return list(seen.values())


def read_memory_counts() -> tuple[int, int, int]:
    facts = 0
    try:
        data = _load_facts()
        facts = sum(len(value) for value in data.values() if isinstance(value, list))
    except Exception:  # noqa: BLE001, S110
        pass
    sessions = episodes = 0
    try:
        db = _ro_db()
        try:
            sessions = db.execute("SELECT COUNT(*) FROM session_summaries").fetchone()[0]
            episodes = db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        finally:
            db.close()
    except sqlite3.Error:
        pass
    return facts, sessions, episodes


def list_local_models() -> list[dict]:
    return [model for model in list_models() if not model["label"].startswith("☁")]


def list_dispatcher_models() -> list[dict]:
    """Return local dispatcher checkpoints, including the external model disk."""
    models: list[dict] = []
    external_root = Path("/Volumes/madisk/yuki-tool-dispatcher")
    if external_root.is_dir():
        for path in sorted(external_root.iterdir()):
            if not path.is_dir() or not (path / "config.json").exists():
                continue
            models.append({"id": str(path), "label": path.name})
    try:
        models.extend(list_local_models())
    except Exception:  # noqa: BLE001, S110
        pass
    deduped: dict[str, dict] = {}
    for model in models:
        deduped.setdefault(model["id"], model)
    return list(deduped.values())


class RetroButton(Button):
    """A focusable neon control routed to an app-level action."""

    def __init__(self, label: str, action: str | None = None, **kwargs):
        super().__init__(label, **kwargs)
        self.nav_action = action


class SessionItem(ListItem):
    def __init__(self, session_uuid: str, label: str):
        super().__init__(Label(label, markup=False))
        self.session_uuid = session_uuid


class ToolItem(ListItem):
    def __init__(self, command: str, blurb: str):
        super().__init__(Label(f"{command}  {blurb}", markup=False))
        self.command = command


class PickItem(ListItem):
    def __init__(self, value: str, label: str):
        super().__init__(Label(label, markup=False))
        self.value = value


class PickHeader(ListItem):
    def __init__(self, label: str):
        super().__init__(Label(label, markup=False), classes="pick-header", disabled=True)
        self.value = None


class ToolActivityCard(Static):
    """A single collapsible record for one tool lifecycle."""

    can_focus = True
    BINDINGS: ClassVar = [
        Binding("enter", "toggle_details", "details", show=False),
        Binding("space", "toggle_details", "details", show=False),
    ]

    def __init__(self, tool_name: str, arguments: str = ""):
        super().__init__(classes="tool-activity", markup=False)
        self.tool_name = tool_name or "tool"
        self.arguments = arguments
        self.status = "RUNNING"
        self.started_at = time.monotonic()
        self.elapsed: float | None = None
        self.details_open = False
        self._render_card()

    def finish(self, status: str, details: str = "") -> None:
        self.status = status.upper()
        self.elapsed = time.monotonic() - self.started_at
        if details:
            self.arguments = details
        self._render_card()

    def _render_card(self) -> None:
        elapsed = f"{self.elapsed:.2f}s" if self.elapsed is not None else "working"
        marker = "[-]" if self.details_open else "[+]"
        lines = [
            f"TOOL // {self.tool_name.upper()}  ──  {self.status}",
            f"{elapsed} · {marker} details",
        ]
        if self.details_open:
            lines.append(self.arguments or "Arguments are managed by the shared runtime.")
        self.update("\n".join(lines))

    def on_click(self) -> None:
        self.action_toggle_details()

    def action_toggle_details(self) -> None:
        self.details_open = not self.details_open
        self._render_card()


class ReasoningCard(Static):
    """Readable model-supplied reasoning, separate from Yuki's final reply."""

    can_focus = True
    BINDINGS: ClassVar = [
        Binding("enter", "toggle_details", "details", show=False),
        Binding("space", "toggle_details", "details", show=False),
    ]

    def __init__(self, reasoning: str):
        super().__init__(classes="reasoning-card", markup=False)
        self.reasoning = reasoning.strip()
        self.details_open = True
        self._render_card()

    def _render_card(self) -> None:
        marker = "[-]" if self.details_open else "[+]"
        text = f"THINKING // MODEL TRACE  {marker}"
        if self.details_open:
            text += f"\n{self.reasoning}"
        self.update(text)

    def on_click(self) -> None:
        self.action_toggle_details()

    def action_toggle_details(self) -> None:
        self.details_open = not self.details_open
        self._render_card()


class StatusCard(Static):
    def __init__(self, title: str, message: str, *, error: bool = False):
        classes = "status-card error" if error else "status-card"
        super().__init__(f"{title}\n{message}", classes=classes, markup=False)


class Composer(Input):
    """One-line composer with history and slash-command completion."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sent: list[str] = []
        self._history_position: int | None = None

    def remember(self, text: str) -> None:
        if text and (not self.sent or self.sent[-1] != text):
            self.sent.append(text)
        self._history_position = None

    def on_key(self, event) -> None:
        if event.key == "up" and self.sent:
            position = len(self.sent) - 1 if self._history_position is None else max(0, self._history_position - 1)
            self._history_position = position
            self.value = self.sent[position]
            self.cursor_position = len(self.value)
            event.stop()
            event.prevent_default()
        elif event.key == "down" and self._history_position is not None:
            position = self._history_position + 1
            if position >= len(self.sent):
                self._history_position = None
                self.value = ""
            else:
                self._history_position = position
                self.value = self.sent[position]
            self.cursor_position = len(self.value)
            event.stop()
            event.prevent_default()
        elif event.key == "tab" and self.value.startswith("/") and " " not in self.value:
            commands = list(TUI_COMMANDS) + list(TOOLS) + ["/help"]
            matches = sorted(command for command in set(commands) if command.startswith(self.value))
            if len(matches) == 1:
                self.value = matches[0] + " "
                self.cursor_position = len(self.value)
            elif matches:
                self.app.notify("  ".join(matches[:8]), title="commands", timeout=4)
            event.stop()
            event.prevent_default()


class ChoiceScreen(ModalScreen[str | None]):
    """An explicit labelled choice, replacing hidden cycling behavior."""

    BINDINGS: ClassVar = [Binding("escape", "cancel", "cancel")]

    def __init__(self, title: str, options: list[tuple[str, str]]):
        super().__init__()
        self._title = title
        self._options = options

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card"):
            yield Static(self._title, classes="modal-title")
            yield ListView(*(PickItem(value, label) for value, label in self._options), id="choice-list")
            yield Static("↑↓ + enter, or click · esc closes", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#choice-list", ListView).focus()

    @on(ListView.Selected, "#choice-list")
    def _picked(self, event: ListView.Selected) -> None:
        value = getattr(event.item, "value", None)
        if value is not None:
            self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SearchPicker(ModalScreen[str | None]):
    BINDINGS: ClassVar = [Binding("escape", "cancel", "cancel")]

    def __init__(self, title: str, sections: list[tuple[str, list[tuple[str, str]]]]):
        super().__init__()
        self._title = title
        self._sections = sections

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card"):
            yield Static(self._title, classes="modal-title")
            yield Input(placeholder="type to filter…", id="picker-filter")
            yield ListView(id="picker-list")
            yield Static("type to filter · enter picks first · esc closes", classes="hint")

    def on_mount(self) -> None:
        self._rebuild("")
        self.query_one("#picker-filter", Input).focus()

    def _rebuild(self, needle: str) -> None:
        low = needle.casefold()
        list_view = self.query_one("#picker-list", ListView)
        list_view.clear()
        for header, items in self._sections:
            matched = [(value, label) for value, label in items if low in label.casefold() or low in value.casefold()]
            if matched:
                list_view.append(PickHeader(f"── {header} ──"))
                for value, label in matched:
                    list_view.append(PickItem(value, label))

    @on(Input.Changed, "#picker-filter")
    def _filter(self, event: Input.Changed) -> None:
        self._rebuild(event.value)

    @on(Input.Submitted, "#picker-filter")
    def _take_first(self) -> None:
        for item in self.query(PickItem):
            self.dismiss(item.value)
            return

    @on(ListView.Selected, "#picker-list")
    def _picked(self, event: ListView.Selected) -> None:
        value = getattr(event.item, "value", None)
        if value is not None:
            self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextEntry(ModalScreen[str | None]):
    BINDINGS: ClassVar = [Binding("escape", "cancel", "cancel")]

    def __init__(self, title: str, value: str = "", *, password: bool = False):
        super().__init__()
        self._title = title
        self._value = value
        self._password = password

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card"):
            yield Static(self._title, classes="modal-title")
            yield Input(value=self._value, password=self._password, id="entry")
            yield Static("enter saves · esc closes", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#entry", Input).focus()

    @on(Input.Submitted, "#entry")
    def _submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class SettingRow(ListItem):
    def __init__(self, key: str, name: str, value: str):
        super().__init__(Label(f"{name:<18} {value}", markup=False))
        self.key = key


class SettingsScreen(ModalScreen[None]):
    BINDINGS: ClassVar = [Binding("escape", "close", "close")]

    def __init__(self) -> None:
        super().__init__()
        self.open_model_manager_after_close = False
        self.open_dispatcher_manager_after_close = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card", id="settings-card"):
            yield Static("SYSTEM CONFIG // EXPLICIT SELECT", classes="modal-title")
            yield ListView(id="settings-list")
            yield Static("enter/click opens a selector · esc closes", classes="hint")

    def on_mount(self) -> None:
        self.refresh_rows()

    def on_unmount(self) -> None:
        if self.open_dispatcher_manager_after_close:
            self.app.call_later(self.app._open_dispatcher_manager)
        elif self.open_model_manager_after_close:
            # Opening the replacement screen here would still happen inside
            # the pop lifecycle. Queue it for the next app message, when this
            # screen is fully gone and the new result callback can be retained.
            self.app.call_later(self.app._open_model_manager)

    def refresh_rows(self) -> None:
        app = self.app
        list_view = self.query_one("#settings-list", ListView)
        index = list_view.index
        list_view.clear()
        image_url = os.environ.get("CF_IMAGE_URL", "")
        image_label = image_url.split("//")[-1][:28] if image_url else "default worker"
        rows = [
            ("model", "model manager", app.model_label or "select model"),
            (
                "tool_routing", "tool routing",
                "DIRECT · main model" if app.tool_routing_mode == "direct"
                else "DISPATCHER · second model",
            ),
            (
                "dispatcher_model", "dispatcher model",
                app.dispatcher_model_label or "select model",
            ),
            ("tool_protocol", "tool protocol", app.tool_routing_protocol.upper()),
            ("reasoning", "reasoning", app.reasoning_mode.upper()),
            ("persona", "persona", app.persona_name),
            ("tts", "voice engine", app.tts_pref or "off"),
            ("voice", "voice character", VOICE_LABELS.get(app.tts_voice, app.tts_voice)),
            (
                "auto", "autonomous",
                f"FREE after {app.autonomy_interval}s idle"
                if app.autonomous_on else "off",
            ),
            ("theme", "terminal theme", THEME_CYCLE[app._theme_idx]),
            ("image", "image backend", image_label),
        ]
        for key, name, value in rows:
            list_view.append(SettingRow(key, name, value))
        if index is not None:
            list_view.index = index

    @on(ListView.Selected, "#settings-list")
    def _selected(self, event: ListView.Selected) -> None:
        key = getattr(event.item, "key", None)
        if key:
            self.app._setting_action(key, self)

    def action_close(self) -> None:
        self.dismiss(None)


class ModelManagerScreen(ModalScreen[dict | None]):
    """Recent | Local | OpenRouter | Remote in one keyboard-friendly manager."""

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "cancel"),
        Binding("down", "focus_first_result", "results", show=False),
        Binding("up", "focus_last_result", "results", show=False),
    ]

    def __init__(
        self,
        settings: dict,
        *,
        api_only: bool = False,
        purpose: str = "main",
        title: str | None = None,
        active_label: str | None = None,
    ):
        super().__init__()
        self._settings = settings
        self._api_only = api_only
        self.purpose = purpose
        self._title = title or "MODEL MANAGER // SELECT LINK"
        self._active_label = active_label
        self._source = "openrouter" if api_only else "recent"
        self._options: list[tuple[str, str]] = []
        self._visible_options: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card", id="model-manager-card"):
            yield Static(self._title, classes="modal-title")
            yield Static(
                f"ACTIVE // {self._active_label or self._settings.get('model_label') or 'none'}",
                id="manager-active",
                classes="hint",
                markup=False,
            )
            with Horizontal(id="model-tabs"):
                if not self._api_only:
                    yield Button("RECENT", id="model-tab-recent", classes="model-tab")
                    yield Button("LOCAL", id="model-tab-local", classes="model-tab")
                yield Button("OPENROUTER", id="model-tab-openrouter", classes="model-tab")
                if not self._api_only:
                    yield Button("REMOTE", id="model-tab-remote", classes="model-tab")
            with Vertical(id="openrouter-config", classes="model-config"):
                yield Static("OPENROUTER API KEY", classes="field-label")
                yield Input(
                    value=os.environ.get("OPENROUTER_API_KEY", ""), password=True,
                    placeholder="sk-or-…", id="manager-openrouter-key",
                )
                yield Button("REFRESH OPENROUTER", id="manager-refresh-openrouter")
            with Vertical(id="remote-config", classes="model-config"):
                yield Static("OPENAI-COMPATIBLE ENDPOINT", classes="field-label")
                yield Input(
                    value=self._settings.get("remote_base_url", ""),
                    placeholder="http://localhost:8000/v1", id="manager-remote-url",
                )
                yield Input(
                    value=self._settings.get("remote_api_key", ""), password=True,
                    placeholder="API key (optional)", id="manager-remote-key",
                )
                yield Input(
                    value=self._settings.get("remote_model", ""),
                    placeholder="model id (or blank to list)", id="manager-remote-model",
                )
                yield Button("CONNECT / LIST", id="manager-connect-remote")
            yield Input(placeholder="filter models…", id="manager-filter")
            yield Static("Select a source. Credentials stay beside that source.", id="manager-state")
            yield Button("LOAD FILTERED MODEL", id="manager-load-filtered", disabled=True)
            yield ListView(id="manager-list")
            yield Static(
                "type to filter · ↓ results · ↑/↓ move · enter loads · esc closes",
                classes="hint",
            )

    def on_mount(self) -> None:
        self.select_source(self._source)

    def _config_snapshot(self) -> dict:
        return {
            "openrouter_key": self.query_one("#manager-openrouter-key", Input).value.strip(),
            "remote_base_url": self.query_one("#manager-remote-url", Input).value.strip(),
            "remote_api_key": self.query_one("#manager-remote-key", Input).value.strip(),
            "remote_model": self.query_one("#manager-remote-model", Input).value.strip(),
        }

    def select_source(self, source: str) -> None:
        self._source = source
        self.query_one("#openrouter-config").display = source == "openrouter"
        self.query_one("#remote-config").display = source == "remote"
        for button in self.query(".model-tab"):
            button.set_class(button.id == f"model-tab-{source}", "selected")
        if source == "recent":
            recent_key = (
                "recent_dispatcher_models"
                if self.purpose == "dispatcher"
                else "recent_models"
            )
            recent = self._settings.get(recent_key, [])
            self.set_options(
                [(entry["id"], entry.get("label") or entry["id"]) for entry in recent],
                "Recent models" if recent else "No recent models yet.",
            )
        elif source == "local":
            self._request_catalog("local")
        elif source == "openrouter":
            if self.query_one("#manager-openrouter-key", Input).value.strip():
                self._request_catalog("openrouter")
            else:
                self.set_options([], "Enter an OpenRouter key, then refresh.")
                self.query_one("#manager-openrouter-key", Input).focus()
        else:
            self.set_options([], "Enter endpoint details, then connect or supply a model id.")

    def _request_catalog(self, source: str) -> None:
        self.set_options([], f"Scanning {source} models…")
        self.app._load_model_catalog(self, source, self._config_snapshot())

    def set_options(self, options: list[tuple[str, str]], message: str = "") -> None:
        if not self.is_mounted:
            return
        self._options = options
        self.query_one("#manager-state", Static).update(
            message or f"{len(options)} models · type to filter"
        )
        self._rebuild(self.query_one("#manager-filter", Input).value)
        if options:
            model_filter = self.query_one("#manager-filter", Input)
            model_filter.focus()

    def _rebuild(self, needle: str) -> None:
        low = needle.casefold().strip()
        self._visible_options = [
            (model_id, label)
            for model_id, label in self._options
            if not low or low in model_id.casefold() or low in label.casefold()
        ]
        list_view = self.query_one("#manager-list", ListView)
        list_view.clear()
        for model_id, label in self._visible_options:
            list_view.append(PickItem(model_id, label))
        load_button = self.query_one("#manager-load-filtered", Button)
        load_button.disabled = not self._visible_options
        if self._visible_options:
            label = self._visible_options[0][1]
            short_label = label[:34] + ("…" if len(label) > 34 else "")
            load_button.label = f"LOAD // {short_label}"
        else:
            load_button.label = "NO MATCH"

    @on(Input.Changed, "#manager-filter")
    def _filter(self, event: Input.Changed) -> None:
        self._rebuild(event.value)

    @on(Input.Submitted, "#manager-filter")
    def _take_first(self) -> None:
        if self._visible_options:
            self._finish(self._visible_options[0][0])
            return
        self.query_one("#manager-state", Static).update(
            "No matching model. Change the filter and try again."
        )

    @on(ListView.Selected, "#manager-list")
    def _selected(self, event: ListView.Selected) -> None:
        # ListView.clear()/append() settle asynchronously. During a fast
        # filter-then-Enter sequence, ``event.item`` may still be a row from
        # the previous render. Resolve the selection against the synchronous
        # filtered snapshot first so the visible index cannot load a stale
        # model (most noticeably, the currently active recent model).
        index = event.list_view.index
        model_id = (
            self._visible_options[index][0]
            if index is not None and 0 <= index < len(self._visible_options)
            else getattr(event.item, "value", None)
        )
        if model_id:
            self._finish(model_id)

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("model-tab-"):
            self.select_source(button_id.removeprefix("model-tab-"))
        elif button_id == "manager-refresh-openrouter":
            self._request_catalog("openrouter")
        elif button_id == "manager-connect-remote":
            config = self._config_snapshot()
            if not config["remote_base_url"]:
                self.query_one("#manager-state", Static).update("A remote base URL is required.")
            elif config["remote_model"]:
                self.dismiss({
                    "id": f"remote/{config['remote_model']}",
                    "label": config["remote_model"].split("/")[-1],
                    "source": "remote", "config": config,
                })
            else:
                self._request_catalog("remote")
        elif button_id == "manager-load-filtered":
            self._take_first()
        event.stop()

    def _finish(self, model_id: str) -> None:
        if getattr(self, "_provider_loading", False):
            return
        label = next((label for value, label in self._options if value == model_id), model_id)
        result = {
            "id": model_id, "label": label, "source": self._source,
            "config": self._config_snapshot(),
        }
        if model_id.startswith("openrouter/") and self.purpose == "main":
            self._provider_loading = True
            self.query_one("#manager-state", Static).update("Checking available providers…")
            self.run_worker(lambda: self._fetch_providers(result), thread=True)
        else:
            self.dismiss(result)

    def _fetch_providers(self, result: dict) -> None:
        try:
            providers = fetch_model_providers(result["id"])
            error = ""
        except Exception as exc:  # noqa: BLE001
            providers, error = [], str(exc)
        self.app.call_from_thread(self._show_providers, result, providers, error)

    def _show_providers(self, result: dict, providers: list[dict], error: str) -> None:
        self._provider_loading = False
        if not self.is_mounted:
            return
        saved = self._settings.get("openrouter_providers", {}).get(result["id"], "")
        options = [("", "Automatic · OpenRouter chooses the provider")]
        options.extend((row["id"], provider_label(row)) for row in providers)
        if saved and not any(value == saved for value, _ in options):
            options.append((saved, f"{saved} · saved choice, availability unverified"))
        title = f"PROVIDER // {result['label']} · current: {saved or 'Automatic'}"
        if error:
            title += f"\nCouldn't fetch providers: {error}"
        elif not providers:
            title += "\nNo provider details available."

        def selected(provider: str | None) -> None:
            if provider is not None:
                self.dismiss({**result, "provider": provider})

        self.app.push_screen(ChoiceScreen(title, options), selected)

    def _focus_result(self, *, last: bool) -> None:
        """Move keyboard focus from the filter into the visible result list."""
        if not self._visible_options:
            self.query_one("#manager-state", Static).update(
                "No matching model. Change the filter and try again."
            )
            return
        list_view = self.query_one("#manager-list", ListView)
        list_view.index = len(self._visible_options) - 1 if last else 0
        list_view.focus()

    def action_focus_first_result(self) -> None:
        self._focus_result(last=False)

    def action_focus_last_result(self) -> None:
        self._focus_result(last=True)

    def action_cancel(self) -> None:
        self.dismiss(None)


class YukiTUI(App):
    TITLE = "yuki"

    CSS = """
    Screen { background: $background; color: $foreground; }
    #body { height: 1fr; }
    RetroButton, Button {
        height: 1; min-width: 5; border: none; padding: 0 1;
        background: $panel; color: $text-muted;
    }
    RetroButton:hover, RetroButton:focus, Button:hover, Button:focus {
        background: $primary 24%; color: $foreground; text-style: bold;
    }
    #sidebar {
        width: 31; height: 1fr; background: $panel;
        border-right: tall $primary 55%; padding: 1;
    }
    #brand {
        height: 2; color: $primary; text-style: bold;
        border-bottom: solid $primary 45%; margin-bottom: 1;
    }
    .side-label { height: 1; color: $primary; margin-top: 1; text-style: bold; }
    #session-search {
        height: 1; border: none; background: $surface;
        padding: 0 1; margin-bottom: 1;
    }
    #current-session {
        width: 100%; height: 2; content-align: left middle;
        color: $success; background: $surface;
    }
    #session-list {
        height: 1fr; min-height: 4; max-height: 18;
        background: $panel; scrollbar-size: 1 1;
    }
    #session-list ListItem, #tools-list ListItem {
        height: 2; padding: 0 1; color: $text-muted; background: $panel;
    }
    #session-list ListItem:hover, #session-list ListItem.-highlight,
    #tools-list ListItem:hover, #tools-list ListItem.-highlight {
        color: $foreground; background: $primary 18%;
    }
    .side-action { width: 100%; content-align: left middle; margin-top: 1; }
    #tools-list { display: none; height: 10; background: $panel; }
    #sidebar-state {
        height: auto; min-height: 2; margin-top: 1; padding: 0 1;
        color: $secondary; border-top: solid $surface-lighten-2;
    }
    #main { width: 1fr; height: 1fr; }
    #topbar {
        height: 3; padding: 1 1 0 1; background: $panel;
        border-bottom: solid $surface-lighten-1;
    }
    #drawer-toggle { display: none; margin-right: 1; }
    #chat-title {
        width: 1fr; height: 1; color: $foreground;
        text-style: bold; padding: 0 1;
    }
    .top-action { margin-left: 1; color: $foreground; }
    #statusbar {
        height: 3; padding: 1; background: $surface;
        border-bottom: solid $primary 35%;
    }
    #model-status { width: 2fr; content-align: left middle; color: $secondary; }
    #activity-status {
        width: 2fr; height: 1; content-align: center middle;
        color: $primary; text-style: bold;
    }
    #voice-status, #auto-status { width: auto; margin-left: 1; color: $foreground; }
    #chat {
        height: 1fr; padding: 1 5 0 5; scrollbar-size: 1 1;
        scrollbar-color: $primary 45%; scrollbar-background: $background;
    }
    #chat.empty { align: center middle; }
    #hero { width: 100%; height: auto; }
    #hero Static { width: 100%; height: auto; text-align: center; }
    .hero-kao, .hero-title, .hero-big { color: $primary; text-style: bold; }
    .hero-sub { color: $secondary; }
    .hero-tag, .hero-tip { color: $text-muted; }
    .row { width: 100%; height: auto; margin-bottom: 1; }
    .row.user { align-horizontal: right; }
    .user-b {
        width: auto; max-width: 85%; padding: 0 1;
        background: $secondary 9%; border: round $secondary 45%;
    }
    .row.bot { margin-right: 6; }
    .bot-label { height: 1; color: $primary; }
    .bot-md { background: transparent; padding: 0; }
    .thinking { height: 1; color: $secondary; }
    .reasoning-card {
        width: 1fr; height: auto; margin: 0 6 1 0; padding: 0 1;
        color: $text-muted; background: $panel;
        border: tall $secondary 55%;
    }
    .reasoning-card:focus {
        color: $foreground; border: tall $secondary;
    }
    .tool-activity {
        width: 1fr; height: auto; min-height: 3; margin: 0 6 1 0;
        padding: 0 1; color: $foreground; background: $surface;
        border: tall $primary 60%;
    }
    .tool-activity:focus { background: $primary 14%; border: tall $primary; }
    .status-card {
        width: 1fr; height: auto; margin: 0 6 1 0; padding: 0 1;
        color: $secondary; border-left: tall $secondary;
    }
    .status-card.error { color: $error; border-left: tall $error; }
    .tool-result {
        width: 1fr; height: auto; margin: 0 6 1 0; padding: 0 1;
        color: $text-muted; border: round $surface-lighten-1;
    }
    .row.img { height: auto; margin-right: 6; }
    .chat-img { width: 48; height: 24; border: round $primary 45%; }
    .img-caption { height: 1; color: $text-muted; }
    #composer-zone { width: 100%; height: auto; align-horizontal: center; }
    #composer-card {
        width: 86%; min-width: 52; max-width: 112; height: auto;
        margin: 0 0 1 0; padding: 1; background: $surface;
        border: double $primary 65%;
    }
    #composer {
        width: 1fr; height: 1; border: none; background: $panel; padding: 0 1;
    }
    #composer-row { height: 2; padding-top: 1; }
    .utility { width: auto; margin-right: 1; }
    #reasoning-button { min-width: 19; color: $secondary; }
    #composer-spacer { width: 1fr; }
    #send-button {
        width: 13; color: $panel; background: $primary; text-style: bold;
    }
    #send-button:hover, #send-button:focus {
        color: $foreground; background: $primary 65%;
    }
    ChoiceScreen, SearchPicker, TextEntry, SettingsScreen, ModelManagerScreen {
        align: center middle; background: $background 72%;
    }
    .modal-card {
        width: 78%; max-width: 100; max-height: 88%; height: auto;
        padding: 1; background: $surface; border: double $primary;
    }
    .modal-title {
        height: 1; margin-bottom: 1; color: $primary; text-style: bold;
    }
    .modal-card Input {
        height: 1; border: none; background: $panel; padding: 0 1; margin-bottom: 1;
    }
    #choice-list, #picker-list, #settings-list, #manager-list {
        height: auto; min-height: 5; max-height: 23; scrollbar-size: 1 1;
    }
    .pick-header { color: $primary; }
    .field-label, .hint, #manager-state { height: 1; color: $text-muted; }
    #model-tabs { height: 2; margin-bottom: 1; }
    .model-tab { width: 1fr; height: 1; margin-right: 1; }
    .model-tab.selected { color: $panel; background: $primary; text-style: bold; }
    .model-config {
        height: auto; padding: 1; border: tall $surface-lighten-2; margin-bottom: 1;
    }
    .model-config Button { width: auto; }
    .-narrow #sidebar { display: none; width: 31; }
    .-narrow #sidebar.drawer-open { display: block; }
    .-narrow #drawer-toggle { display: block; }
    .-narrow #top-settings { display: none; }
    .-narrow #chat { padding: 1 2 0 2; }
    .-narrow #composer-card { width: 96%; min-width: 36; }
    .-narrow #statusbar { padding: 1 0; }
    .-narrow #activity-status { width: 1fr; }
    .-narrow .chat-img { width: 36; height: 18; }
    .-narrow .power-shortcut { display: none; }
    .-tiny #model-status { width: 1fr; }
    .-tiny #voice-status, .-tiny #auto-status { padding: 0; min-width: 7; }
    .-short #hero .hero-big, .-short #hero .hero-kao,
    .-short #hero .hero-tip { display: none; }
    .-short #hero .hero-title { display: block; }
    """

    BINDINGS: ClassVar = [
        Binding("ctrl+n", "new_session", "new chat", show=False),
        Binding("ctrl+b", "toggle_sidebar", "menu", show=False),
        Binding("ctrl+t", "toggle_tts", "voice", show=False),
        Binding("ctrl+q", "quit", "quit", show=False),
        Binding("escape", "focus_composer", "", show=False),
    ]

    def __init__(
        self, bot=None, model_label: str = "", persona_name: str = "yuki",
        persona_text: str | None = None, autonomy_interval: int = 120,
        api_only: bool = False, preferred_model: tuple[str, str] | None = None,
        tts_pref: str | None = None,
    ):
        super().__init__()
        self.bot = bot
        self.model_label = model_label
        self._pending_model_label: str | None = None
        self._model_source = self._source_for_model(
            getattr(bot, "model_id", "") if bot is not None else ""
        )
        self.persona_name = persona_name
        self.persona_text = persona_text
        self.autonomy_interval = autonomy_interval
        self.autonomous_on = autonomy_interval > 0
        self.api_only = api_only
        self.preferred_model = preferred_model
        self.tts_pref = tts_pref
        self._settings = load_settings()
        self.tool_routing_mode = normalize_routing_mode(
            self._settings.get("tool_routing_mode"),
        )
        self.tool_routing_protocol = normalize_routing_protocol(
            self._settings.get("tool_routing_protocol"),
        )
        self.dispatcher_model_id = self._settings.get(
            "dispatcher_model_id",
            DEFAULT_DISPATCHER_MODEL,
        )
        self.dispatcher_model_label = self._settings.get(
            "dispatcher_model_label",
            Path(self.dispatcher_model_id).name,
        )
        self.reasoning_mode = normalize_reasoning_mode(
            self._settings.get("reasoning_mode"), default="auto",
        )
        if (
            self.reasoning_mode == "off"
            and openrouter_reasoning_is_mandatory(getattr(bot, "model_id", None))
        ):
            self.reasoning_mode = "auto"
        if self.bot is not None:
            self.bot.reasoning_mode = self.reasoning_mode
            self._apply_tool_routing_to_bot()
        self.tts_voice = self._settings.get("tts_voice", KOKORO_VOICES[0])
        self.activity = ActivityState()
        self._activity_counter = 0
        self._cancelled_tokens: set[int] = set()
        self._bot_lock = threading.Lock()
        self._last_interaction = time.time()
        self._thinking: Static | None = None
        self._thinking_label = ""
        self._spin_timer = None
        self._spin_index = 0
        self._hero: Vertical | None = None
        self._tts_cache = None
        self._tts_loading = False
        self._speaking = False
        self._speech_token = 0
        self._theme_idx = 0
        self._is_narrow = False
        self._drawer_open = False
        self._main_screen = None
        self._session_title = "New chat"
        self._session_filter = ""
        self._active_tool_card: ToolActivityCard | None = None
        self._img_seen = time.time()

    @property
    def _busy(self) -> bool:
        return self.activity.blocks_input

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("YUKI // SYSTEM\nNEON LINK 199X", id="brand", markup=False)
                yield Static("SEARCH CHATS", classes="side-label", markup=False)
                yield Input(placeholder="filter sessions…", id="session-search")
                yield Static("● CURRENT", classes="side-label", markup=False)
                yield RetroButton("New chat", "current-session", id="current-session")
                yield Static("PREVIOUS CHATS", classes="side-label", markup=False)
                yield ListView(id="session-list")
                yield RetroButton("TOOLS", "tools", id="tools-button", classes="side-action")
                yield ListView(id="tools-list")
                yield RetroButton("MEMORY / STATUS", "brain", classes="side-action")
                yield RetroButton("MODEL MANAGER", "model", classes="side-action")
                yield RetroButton("SETTINGS", "settings", classes="side-action")
                yield Static("", id="sidebar-state", markup=False)
            with Vertical(id="main"):
                with Horizontal(id="topbar"):
                    yield RetroButton("MENU", "menu", id="drawer-toggle")
                    yield Static("New chat", id="chat-title", markup=False)
                    yield RetroButton("NEW", "new", id="top-new", classes="top-action")
                    yield RetroButton("SETTINGS", "settings", id="top-settings", classes="top-action")
                with Horizontal(id="statusbar"):
                    yield RetroButton("MODEL // SELECT", "model", id="model-status")
                    yield Static("IDLE", id="activity-status", markup=False)
                    yield RetroButton("VOICE // OFF", "voice", id="voice-status")
                    yield RetroButton("AUTO // OFF", "auto", id="auto-status")
                yield VerticalScroll(id="chat", classes="empty")
                with Horizontal(id="composer-zone"), Vertical(id="composer-card"):
                        yield Composer(placeholder="Message Yuki…", id="composer")
                        with Horizontal(id="composer-row"):
                            yield RetroButton("HELP", "help", classes="utility power-shortcut")
                            yield RetroButton("WEB", "insert-search", classes="utility power-shortcut")
                            yield RetroButton("SHELL", "insert-shell", classes="utility power-shortcut")
                            yield RetroButton(
                                "REASONING // AUTO", "reasoning",
                                id="reasoning-button", classes="utility",
                            )
                            yield RetroButton(
                                "TOOLS", "tools", id="composer-tools-button", classes="utility",
                            )
                            yield Static("", id="composer-spacer")
                            yield RetroButton("SEND", "send", id="send-button")

    def on_mount(self) -> None:
        self._main_screen = self.screen
        self.register_theme(CYBER_THEME)
        self.register_theme(ODYSSEUS_THEME)
        saved_theme = self._settings.get("theme2")
        if saved_theme in THEME_CYCLE:
            self._theme_idx = THEME_CYCLE.index(saved_theme)
        self.theme = THEME_CYCLE[self._theme_idx]
        if self.bot is not None:
            self.bot._emit_tool_progress = self._tool_progress_sink
            self.bot._emit_routing_progress = self._routing_progress_sink
            if self.autonomous_on and hasattr(self.bot, "begin_autonomy"):
                self.bot.begin_autonomy()
            self._session_title = self._title_from_history(self.bot.history)
        tools_list = self.query_one("#tools-list", ListView)
        for name, tool in TOOLS.items():
            tools_list.append(ToolItem(name, tool["help"][:24]))
        self._mount_hero()
        self._refresh_sidebar()
        self._refresh_chrome()
        self.query_one("#composer", Composer).focus()
        self.set_interval(5.0, self._autonomy_check)
        self._apply_breakpoints(self.size)
        if self.bot is None and self.preferred_model:
            model_id, label = self.preferred_model
            self._begin_model_load(model_id, label)

    def on_resize(self, event) -> None:
        self._apply_breakpoints(event.size)

    def _apply_breakpoints(self, size) -> None:
        if self._main_screen is None:
            return
        narrow = size.width < 92
        self._main_screen.set_class(narrow, "-narrow")
        self._main_screen.set_class(size.width < 68, "-tiny")
        self._main_screen.set_class(size.height < 27, "-short")
        self._is_narrow = narrow
        sidebar = self.query_one("#sidebar")
        if narrow:
            sidebar.set_class(self._drawer_open, "drawer-open")
            sidebar.display = self._drawer_open
        else:
            self._drawer_open = False
            sidebar.remove_class("drawer-open")
            sidebar.display = True

    def _save_settings(self) -> None:
        self._settings.update({
            "model_id": self.bot.model_id if self.bot else self._settings.get("model_id"),
            "model_label": self.model_label or self._settings.get("model_label"),
            "persona": self.persona_name,
            "tts": self.tts_pref,
            "tts_voice": self.tts_voice,
            "reasoning_mode": self.reasoning_mode,
            "tool_routing_mode": self.tool_routing_mode,
            "tool_routing_protocol": self.tool_routing_protocol,
            "dispatcher_model_id": self.dispatcher_model_id,
            "dispatcher_model_label": self.dispatcher_model_label,
            "auto": self.autonomy_interval if self.autonomous_on else 0,
            "theme2": THEME_CYCLE[self._theme_idx],
        })
        save_settings(self._settings)

    def _remember_model(self, model_id: str, label: str, source: str) -> None:
        recent = [entry for entry in self._settings.get("recent_models", []) if entry.get("id") != model_id]
        recent.insert(0, {"id": model_id, "label": label, "source": source})
        self._settings["recent_models"] = recent[:6]

    @staticmethod
    def _clean_title(text: str) -> str:
        title = " ".join(text.split())
        return title[:36] + ("…" if len(title) > 36 else "")

    @staticmethod
    def _source_for_model(model_id: str) -> str:
        if model_id.startswith("openrouter/"):
            return "OpenRouter"
        if model_id.startswith("remote/"):
            return "Remote"
        return "Local" if model_id else ""

    def _title_from_history(self, history: list[dict]) -> str:
        for message in history:
            if message.get("role") == "user" and message.get("content"):
                return self._clean_title(str(message["content"]))
        return "New chat"

    def _voice_enabled(self) -> bool:
        return bool(self.bot and self.bot.tts is not None) or bool(self.tts_pref)

    def _refresh_chrome(self) -> None:
        if not self.is_mounted:
            return
        model = self.model_label or "SELECT"
        model = model[:21] + "…" if len(model) > 22 else model
        source = f" · {self._model_source}" if self._model_source else ""
        self.query_one("#model-status", RetroButton).label = f"MODEL // {model}{source}"
        for hero_link in self.query("#hero-link"):
            hero_link.update(self._neural_link_status())
        self.query_one("#activity-status", Static).update(self.activity.label)
        if self._tts_loading:
            voice = "LOADING"
        elif self._speaking:
            voice = "SPEAKING · STOP"
        else:
            voice = "ON" if self._voice_enabled() else "OFF"
        self.query_one("#voice-status", RetroButton).label = f"VOICE // {voice}"
        auto = f"FREE {self.autonomy_interval}s" if self.autonomous_on else "OFF"
        self.query_one("#auto-status", RetroButton).label = f"AUTO // {auto}"
        self.query_one("#reasoning-button", RetroButton).label = (
            f"REASONING // {self.reasoning_mode.upper()}"
        )
        send = self.query_one("#send-button", RetroButton)
        if self.activity.blocks_input:
            send.label = "CANCELING" if self.activity.cancel_requested else "STOP" if self.activity.cancellable else "WAIT"
        else:
            send.label = "SEND"
        self.query_one("#chat-title", Static).update(self._session_title)
        self.query_one("#current-session", RetroButton).label = f"● {self._session_title}"
        speech = " · SPEAKING" if self._speaking else ""
        self.query_one("#sidebar-state", Static).update(
            f"STATE // {self.activity.label}{speech}\n"
            f"TOOLS // {self._routing_label()}\nAUTO // {auto}"
        )

    def _routing_label(self) -> str:
        if self.tool_routing_mode == "direct":
            protocol = self.tool_routing_protocol.upper()
            if self.tool_routing_protocol == "auto" and self.bot is not None:
                effective = (
                    "NATIVE"
                    if supports_native_tools(
                        getattr(self.bot, "model_id", None),
                        getattr(self.bot, "backend", None),
                    )
                    else "CANONICAL"
                )
                protocol = f"AUTO→{effective}"
            elif (
                self.tool_routing_protocol == "native"
                and self.bot is not None
                and getattr(self.bot, "backend", None) not in {"openrouter", "gguf"}
            ):
                protocol = "NATIVE→CANONICAL"
            return f"DIRECT · {protocol}"
        label = self.dispatcher_model_label or "SELECT"
        return f"DISPATCHER · {label} · {self.tool_routing_protocol.upper()}"

    def _apply_model_provider(self, bot) -> None:
        if str(getattr(bot, "model_id", "")).startswith("openrouter/"):
            client = getattr(bot, "llm_model", None)
            if client is not None:
                client._yuki_openrouter_provider = self._settings.get(
                    "openrouter_providers", {},
                ).get(bot.model_id, "")

    def _apply_tool_routing_to_bot(self) -> None:
        self._apply_model_provider(self.bot)
        if self.bot is None or not hasattr(self.bot, "configure_tool_routing"):
            return
        self.bot.configure_tool_routing(
            mode=self.tool_routing_mode,
            protocol=self.tool_routing_protocol,
            dispatcher_model_id=self.dispatcher_model_id,
        )

    def _neural_link_status(self) -> str:
        if self._pending_model_label:
            return f"[ NEURAL LINK SWITCHING · {self._pending_model_label} ]"
        if self.bot is not None:
            provider = self._settings.get("openrouter_providers", {}).get(self.bot.model_id, "")
            suffix = f" · provider: {provider or 'Automatic'}" if self.bot.model_id.startswith("openrouter/") else ""
            return f"[ NEURAL LINK ONLINE · {self.model_label}{suffix} ]"
        return "[ NEURAL LINK OFFLINE · SELECT MODEL ]"

    def _start_activity(self, kind: str, label: str, *, cancellable: bool) -> int:
        self._activity_counter += 1
        self.activity = ActivityState(
            kind=kind, label=label, started_at=time.monotonic(),
            cancellable=cancellable, token=self._activity_counter,
        )
        self._refresh_chrome()
        return self.activity.token

    def _transition_activity(self, kind: str, label: str) -> None:
        self.activity.kind = kind
        self.activity.label = label
        self._refresh_chrome()

    def _finish_activity(self, token: int, *, error: str | None = None) -> bool:
        if token != self.activity.token:
            return False
        self._hide_thinking()
        self.activity = ActivityState(
            kind="error" if error else "idle", label="ERROR" if error else "IDLE", token=token,
        )
        if error:
            self._mount_chat(StatusCard("SYSTEM // ERROR", error, error=True))
        self._refresh_chrome()
        self._last_interaction = time.time()
        return True

    def _is_cancelled(self, token: int) -> bool:
        return token in self._cancelled_tokens

    def _request_cancel(self) -> None:
        if not self.activity.blocks_input:
            if self._speaking:
                self._stop_speech()
            return
        if not self.activity.cancellable:
            self.notify("This operation cannot be canceled safely.", severity="warning")
            return
        if self.activity.cancel_requested:
            return
        self.activity.cancel_requested = True
        self._cancelled_tokens.add(self.activity.token)
        self.activity.label = "CANCEL REQUESTED"
        if self._active_tool_card and self._active_tool_card.status == "RUNNING":
            self._active_tool_card.finish("CANCEL REQUESTED")
        self._refresh_chrome()
        self.notify("Cancel requested. The composer draft is untouched.")

    def _mount_hero(self) -> None:
        self._hero = Vertical(
            Static("⊹ ࣪ ˖ ૮( ˶ᵔ ᵕ ᵔ˶ )っ", classes="hero-kao", markup=False),
            Static(YUKI_LOGO, classes="hero-big", markup=False),
            Static("Y U K I", classes="hero-title", markup=False),
            Static("yours for the voyage.", classes="hero-tag", markup=False),
            Static(
                self._neural_link_status(),
                id="hero-link",
                classes="hero-sub",
                markup=False,
            ),
            Static(
                "Type a message, or open MODEL MANAGER. Slash commands are optional.",
                classes="hero-tip", markup=False,
            ),
            id="hero",
        )
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(self._hero)
        chat.add_class("empty")

    def _clear_hero(self) -> None:
        if self._hero is not None and self._hero.is_mounted:
            self._hero.remove()
            self.query_one("#chat", VerticalScroll).remove_class("empty")
        self._hero = None

    def _mount_chat(self, widget, *, animate: bool = True) -> None:
        self._clear_hero()
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(widget)
        if animate:
            try:
                widget.styles.opacity = 0.0
                widget.styles.offset = (0, 1)
                widget.styles.animate("opacity", 1.0, duration=0.2)
                widget.styles.animate("offset", (0, 0), duration=0.2)
            except Exception:  # noqa: BLE001, S110
                pass
        self.call_after_refresh(lambda: chat.scroll_end(animate=animate, duration=0.2))

    def _user_bubble(self, text: str, *, animate: bool = True) -> None:
        self._mount_chat(
            Horizontal(Static(text, markup=False, classes="user-b"), classes="row user"),
            animate=animate,
        )

    def _bot_bubble(self, text: str, *, animate: bool = True) -> None:
        self._mount_chat(
            Vertical(
                Static(f"⊹ {self.persona_name}", classes="bot-label", markup=False),
                Markdown(text, classes="bot-md"), classes="row bot",
            ),
            animate=animate,
        )

    def _reasoning_bubble(self, text: str) -> None:
        if text.strip():
            self._mount_chat(ReasoningCard(text))

    def _show_thinking(self, label: str) -> None:
        self._hide_thinking()
        self._thinking_label = label
        self._thinking = Static(f"⠋ {label}", classes="thinking", markup=False)
        self._mount_chat(self._thinking)
        self._spin_index = 0
        self._spin_timer = self.set_interval(0.1, self._spin)

    def _spin(self) -> None:
        if self._thinking is None or not self._thinking.is_mounted:
            if self._spin_timer is not None:
                self._spin_timer.stop()
                self._spin_timer = None
            return
        self._spin_index += 1
        frame = SPINNER_FRAMES[self._spin_index % len(SPINNER_FRAMES)]
        self._thinking.update(f"{frame} {self._thinking_label}")

    def _hide_thinking(self) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
        if self._thinking is not None and self._thinking.is_mounted:
            self._thinking.remove()
        self._thinking = None

    def _new_tool_card(self, tool_name: str, arguments: str = "") -> ToolActivityCard:
        card = ToolActivityCard(tool_name, arguments)
        self._active_tool_card = card
        self._mount_chat(card)
        return card

    def _tool_progress_sink(self, line: str) -> None:
        self.call_from_thread(self._apply_tool_progress, str(line))

    def _routing_progress_sink(self, kind: str, label: str) -> None:
        self.call_from_thread(self._apply_routing_progress, kind, label)

    def _apply_routing_progress(self, kind: str, label: str) -> None:
        if not self.activity.blocks_input:
            return
        self._transition_activity(kind, label)
        if kind in {"routing", "delegating", "dispatching", "validating"}:
            self._show_thinking(label.title().replace(" // ", " · "))

    def _apply_tool_progress(self, line: str) -> None:
        clean = line.strip()
        failed = "— failed" in clean
        done = "— done" in clean or failed
        clean = clean.replace("🔧", "").replace("✅", "").replace("⚠️", "").strip()
        tool_name = clean.split("—", 1)[0].rstrip("… ").strip() or "tool"
        if done:
            if self._active_tool_card is not None:
                self._active_tool_card.finish("FAILURE" if failed else "SUCCESS")
            self._transition_activity("generating", "GENERATING // FINALIZING")
            self._show_thinking("Yuki is finishing the reply")
        else:
            self._hide_thinking()
            self._new_tool_card(tool_name, self._tool_arguments_for_card(tool_name))
            self._transition_activity("tool", f"TOOL // {tool_name.upper()}")

    def _tool_arguments_for_card(self, tool_name: str) -> str:
        """Best-effort argument view without changing VoiceChatBot's callback API."""
        if self.bot is None:
            return ""
        wanted = tool_name.casefold().replace(" ", "_")
        for message in reversed(self.bot.history):
            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function") or {}
                if str(function.get("name", "")).casefold() != wanted:
                    continue
                raw = function.get("arguments", "")
                try:
                    return json.dumps(json.loads(raw), ensure_ascii=False)
                except (TypeError, ValueError):
                    return str(raw)
            content = str(message.get("content") or "")
            marker = f"[TOOL: {wanted}("
            start = content.casefold().find(marker.casefold())
            if start >= 0:
                argument = content[start + len(marker):].split(")]", 1)[0]
                return argument.strip()
        source = getattr(self.bot, "_last_user_input", "")
        return f"source request: {source}" if source else ""

    def _drain_new_images(self) -> list[Path]:
        try:
            fresh = sorted(
                (path for path in IMAGES_DIR.glob("*.png") if path.stat().st_mtime > self._img_seen),
                key=lambda path: path.stat().st_mtime,
            )
        except OSError:
            return []
        if fresh:
            self._img_seen = max(path.stat().st_mtime for path in fresh)
        return fresh

    def _image_bubble(self, path: Path) -> None:
        try:
            from textual_image.widget import Image
            image = Image(str(path), classes="chat-img")
        except Exception:  # noqa: BLE001
            self._mount_chat(StatusCard("IMAGE // SAVED", str(path)))
            return
        self._mount_chat(Vertical(
            image, Static(f"▞ {path.name}", classes="img-caption", markup=False),
            classes="row img",
        ))

    @on(Button.Pressed)
    async def _button_pressed(self, event: Button.Pressed) -> None:
        action = getattr(event.button, "nav_action", None)
        if action:
            await self._nav(action)
            event.stop()

    async def _nav(self, action: str) -> None:
        if action == "send":
            if self.activity.blocks_input:
                self._request_cancel()
            else:
                self._handle_text(self.query_one("#composer", Composer).value)
        elif action == "menu":
            self.action_toggle_sidebar()
        elif action == "new":
            await self.action_new_session()
        elif action == "current-session":
            self._close_drawer()
            self.action_focus_composer()
        elif action == "tools":
            tools = self.query_one("#tools-list", ListView)
            tools.display = not tools.display
            if tools.display:
                tools.focus()
                self._open_drawer()
        elif action == "brain":
            self._show_brain()
            self._close_drawer()
        elif action == "settings":
            self._open_settings()
        elif action == "model":
            self._open_model_manager()
        elif action == "reasoning":
            self._open_reasoning_picker()
        elif action == "voice":
            self.action_toggle_tts()
        elif action == "auto":
            self._toggle_autonomy()
        elif action == "help":
            self._render_help()
        elif action == "insert-search":
            self._insert_composer("/search ")
        elif action == "insert-shell":
            self._insert_composer("/shell ")

    def _insert_composer(self, text: str) -> None:
        if self.activity.blocks_input:
            self.notify("Yuki is busy; your current draft was preserved.", severity="warning")
            return
        composer = self.query_one("#composer", Composer)
        composer.value = text
        composer.cursor_position = len(text)
        composer.focus()

    @on(Input.Changed, "#session-search")
    def _session_search_changed(self, event: Input.Changed) -> None:
        self._session_filter = event.value.strip()
        self._refresh_sidebar()

    @on(ListView.Selected, "#session-list")
    def _session_selected(self, event: ListView.Selected) -> None:
        session_uuid = getattr(event.item, "session_uuid", None)
        if session_uuid:
            self._resume_session(session_uuid)

    @on(ListView.Selected, "#tools-list")
    def _tool_selected(self, event: ListView.Selected) -> None:
        command = getattr(event.item, "command", None)
        if command:
            self._insert_composer(command + " ")
            self._close_drawer()

    def _refresh_sidebar(self) -> None:
        if not self.is_mounted:
            return
        list_view = self.query_one("#session-list", ListView)
        list_view.clear()
        now = time.time()
        needle = self._session_filter.casefold()
        current_uuid = self.bot.session_uuid if self.bot is not None else None
        sessions = read_recent_sessions()
        known: set[str] = set()
        for session in sessions:
            known.add(session["uuid"])
            if session["uuid"] == current_uuid:
                continue
            title = self._clean_title(session["title"] or session["summary"] or "Untitled chat")
            if needle and needle not in title.casefold() and needle not in session["summary"].casefold():
                continue
            when = episodic.fuzzy_when(now - session["ts_end"])
            list_view.append(SessionItem(session["uuid"], f"{title}\n{when}"))
        if len(needle) >= 2:
            for hit in search_episodes(self._session_filter):
                if hit["uuid"] in known or hit["uuid"] == current_uuid:
                    continue
                when = episodic.fuzzy_when(now - hit["ts"])
                list_view.append(SessionItem(hit["uuid"], f"…{hit['snippet']}…\n{when}"))

    def _open_drawer(self) -> None:
        if not self._is_narrow:
            return
        self._drawer_open = True
        sidebar = self.query_one("#sidebar")
        sidebar.add_class("drawer-open")
        sidebar.display = True

    def _close_drawer(self) -> None:
        if not self._is_narrow:
            return
        self._drawer_open = False
        sidebar = self.query_one("#sidebar")
        sidebar.remove_class("drawer-open")
        sidebar.display = False

    def action_toggle_sidebar(self) -> None:
        if self._is_narrow:
            self._close_drawer() if self._drawer_open else self._open_drawer()
            return
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def _show_brain(self) -> None:
        facts, sessions, episodes = read_memory_counts()
        text = f"{facts} remembered facts · {sessions} summarized chats · {episodes} stored moments"
        if self.bot is not None:
            text += f"\n{len(self.bot.history)} messages in context"
        self._mount_chat(StatusCard("MEMORY // STATUS", text))

    @on(Input.Submitted, "#composer")
    def _submitted(self, event: Input.Submitted) -> None:
        self._handle_text(event.value)

    def _accept_composer(self, text: str, *, title_candidate: bool = False) -> None:
        composer = self.query_one("#composer", Composer)
        composer.value = ""
        composer.remember(text)
        if title_candidate and self._session_title == "New chat":
            self._session_title = self._clean_title(text)
            self._refresh_chrome()
        self._last_interaction = time.time()

    def _handle_text(self, raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        if self.activity.blocks_input:
            self.notify("Yuki is busy. Your draft is still in the composer.", severity="warning")
            return
        if text.startswith("/"):
            command = text.split(maxsplit=1)[0].lower()
            argument = text.split(maxsplit=1)[1] if " " in text else ""
            if command == "/help":
                self._accept_composer(text)
                self._render_help()
                return
            if command in TUI_COMMANDS:
                if command == "/new" and self.bot is None:
                    self.notify("Load a model before starting a model session.", severity="warning")
                    return
                self._accept_composer(text)
                self._run_tui_command(command, argument)
                return
            if command in TOOLS:
                self._accept_composer(text, title_candidate=True)
                self._user_bubble(text)
                token = self._start_activity("tool", f"TOOL // {command[1:].upper()}", cancellable=True)
                self._new_tool_card(command[1:], argument)
                self.run_worker(lambda: self._direct_tool_work(text, token), thread=True)
                return
        if self.bot is None:
            self.notify("Select a model first. Your draft is still here.", severity="warning")
            self._open_model_manager()
            return
        if self._speaking:
            self._stop_speech()
        self._accept_composer(text, title_candidate=True)
        self._user_bubble(text)
        token = self._start_activity("generating", "GENERATING // YUKI", cancellable=True)
        self._show_thinking("Yuki is thinking")
        self.run_worker(lambda: self._chat_work(text, token), thread=True)

    def _chat_work(self, text: str, token: int) -> None:
        old_history = list(self.bot.history)
        old_continuity = [
            dict(event)
            for event in getattr(self.bot, "_session_continuity", [])
        ]
        old_pending_events = [
            dict(event)
            for event in getattr(self.bot, "_pending_session_events", [])
        ]
        error: str | None = None
        with self._bot_lock:
            try:
                response = self.bot.react_chat(text)
            except Exception as exc:  # noqa: BLE001
                response = _cli_friendly_error(exc)
                error = response
            if self._is_cancelled(token):
                self.bot.history = old_history
                completed = [
                    event for event in self.bot._pending_session_events[len(old_pending_events):]
                    if event.get("kind") == "verified_tool_outcome"
                ]
                self.bot._session_continuity = old_continuity + completed
                self.bot._pending_session_events = old_pending_events + completed
                _cli_persist_session_events(self.bot)
                self.call_from_thread(self._finish_cancelled, token)
                return
            if not error:
                if self.autonomous_on and hasattr(self.bot, "begin_autonomy"):
                    # A new human turn becomes the fresh durable goal/context.
                    self.bot.begin_autonomy()
                _cli_record_episode(self.bot.session_uuid, text, response)
            _cli_persist_session_events(self.bot)
        self.call_from_thread(self._finish_reply, response, token, error)

    def _finish_reply(self, response: str, token: int, error: str | None = None) -> None:
        if self._is_cancelled(token):
            self._finish_cancelled(token)
            return
        if self._active_tool_card is not None and self._active_tool_card.status == "RUNNING":
            self._active_tool_card.finish("FAILURE" if error else "SUCCESS")
        if error:
            self._finish_activity(token, error=error)
            return
        self._enforce_reasoning_requirement(notify=True)
        self._reasoning_bubble(getattr(self.bot, "last_reasoning", ""))
        self._bot_bubble(response)
        for path in self._drain_new_images():
            self._image_bubble(path)
        self._finish_activity(token)
        self._refresh_sidebar()
        self._begin_speech(response)

    def _finish_cancelled(self, token: int) -> None:
        if token != self.activity.token:
            return
        if self.activity.kind == "model_loading":
            self._pending_model_label = None
        self._hide_thinking()
        if self._active_tool_card is not None and self._active_tool_card.status == "RUNNING":
            self._active_tool_card.finish("CANCELED")
        warning = "The generated result was discarded."
        if self._active_tool_card is not None:
            warning += " A synchronous tool may already have completed its side effect."
        self._mount_chat(StatusCard("SYSTEM // CANCELED", warning))
        self._finish_activity(token)
        self._cancelled_tokens.discard(token)

    def _direct_tool_work(self, text: str, token: int) -> None:
        tool_error: str | None = None
        try:
            result, name, links = run_tool(text)
        except Exception as exc:  # noqa: BLE001
            result, name, links = f"tool error: {exc}", "tool", []
            tool_error = str(exc)

        name = name or "tool"
        result = result if isinstance(result, str) else "" if result is None else str(result)
        links = links or []
        details = result
        if links:
            details += "\n" + "\n".join(str(link) for link in links)
        if tool_error is None and _is_tool_error(result):
            tool_error = result.splitlines()[0] if result else "Tool failed."

        if self.bot is not None:
            with self._bot_lock:
                if hasattr(self.bot, "_remember_tool_outcome"):
                    self.bot._remember_tool_outcome(
                        source="human_direct_command",
                        request=text,
                        tool=name.lstrip("/"),
                        arguments=text.partition(" ")[2],
                        result=result,
                        succeeded=tool_error is None,
                    )
                    _cli_persist_session_events(self.bot)

        if self.bot is None or self._is_cancelled(token):
            self.call_from_thread(
                self._finish_direct_tool,
                token,
                name,
                details,
                tool_error,
            )
            return

        self.call_from_thread(
            self._prepare_direct_tool_reply,
            token,
            name,
            details,
            tool_error,
        )
        if self._is_cancelled(token):
            self.call_from_thread(self._finish_cancelled, token)
            return

        response_error: str | None = None
        with self._bot_lock:
            old_history = list(self.bot.history)
            old_continuity = [
                dict(event)
                for event in getattr(self.bot, "_session_continuity", [])
            ]
            old_pending_events = [
                dict(event)
                for event in getattr(self.bot, "_pending_session_events", [])
            ]
            try:
                response = self.bot.respond_to_completed_tool(
                    text,
                    name.lstrip("/"),
                    details,
                )
            except Exception as exc:  # noqa: BLE001
                response = _cli_friendly_error(exc)
                response_error = response

            if self._is_cancelled(token):
                self.bot.history = old_history
                self.bot._session_continuity = old_continuity
                self.bot._pending_session_events = old_pending_events
                self.call_from_thread(self._finish_cancelled, token)
                return

            if response_error is None:
                _cli_record_episode(self.bot.session_uuid, text, response)
                _cli_persist_session_events(self.bot)

        self.call_from_thread(
            self._finish_direct_tool_reply,
            token,
            response,
            response_error,
        )

    def _prepare_direct_tool_reply(
        self,
        token: int,
        name: str,
        details: str,
        tool_error: str | None,
    ) -> None:
        if token != self.activity.token or self._is_cancelled(token):
            return
        if self._active_tool_card is not None:
            self._active_tool_card.tool_name = name
            self._active_tool_card.finish(
                "FAILURE" if tool_error else "SUCCESS",
                details,
            )
        self._transition_activity("generating", "GENERATING // TOOL RESPONSE")
        self._show_thinking("Yuki is reading the tool result")

    def _finish_direct_tool_reply(
        self,
        token: int,
        response: str,
        error: str | None,
    ) -> None:
        self._finish_reply(response, token, error)
        self._cancelled_tokens.discard(token)

    def _finish_direct_tool(
        self,
        token: int,
        name: str,
        details: str,
        error: str | None,
    ) -> None:
        canceled = self._is_cancelled(token)
        if self._active_tool_card is not None:
            self._active_tool_card.tool_name = name
            if canceled:
                self._active_tool_card.finish(
                    "FINISHED AFTER CANCEL",
                    "Result discarded; synchronous side effects cannot be reversed.",
                )
            else:
                self._active_tool_card.finish("FAILURE" if error else "SUCCESS", details)
        if canceled:
            self._mount_chat(StatusCard(
                "TOOL // FINISHED AFTER CANCEL",
                "Output was hidden. A synchronous tool may already have completed its side effect.",
            ))
        else:
            for path in self._drain_new_images():
                self._image_bubble(path)
        self._finish_activity(token, error=error)
        self._cancelled_tokens.discard(token)

    def _begin_speech(self, response: str) -> None:
        if self.bot is None or self.bot.tts is None:
            return
        self._speech_token += 1
        token = self._speech_token
        self._speaking = True
        self._refresh_chrome()
        self.run_worker(lambda: self._speech_work(response, token), thread=True)

    def _speech_work(self, response: str, token: int) -> None:
        if token != self._speech_token:
            return
        try:
            self.bot.speak(response, voice=self.tts_voice)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self.notify, f"voice playback failed: {exc}", severity="error")
        self.call_from_thread(self._speech_finished, token)

    def _speech_finished(self, token: int) -> None:
        if token == self._speech_token:
            self._speaking = False
            self._refresh_chrome()

    def _stop_speech(self) -> None:
        if not self._speaking:
            return
        self._speech_token += 1
        self._speaking = False
        self._refresh_chrome()

        def stop_player() -> None:
            import subprocess
            for player in ("afplay", "aplay"):
                subprocess.run(["pkill", "-x", player], capture_output=True, check=False)

        self.run_worker(stop_player, thread=True)

    def action_toggle_tts(self) -> None:
        if self._speaking:
            self._stop_speech()
            return
        if self.bot is None:
            if self.tts_pref:
                self.tts_pref = None
                self.notify("Voice will stay off when the model loads.")
            else:
                engines = list_tts_engines()
                self.tts_pref = engines[0]["key"] if engines else None
                self.notify("Voice will load with the model." if self.tts_pref else "No voice engine found.")
            self._save_settings()
            self._refresh_chrome()
            return
        if self.bot.tts is not None:
            self._tts_cache = self.bot.tts
            self.bot.tts = None
            self.tts_pref = None
            self.notify("Voice playback off.")
            self._save_settings()
            self._refresh_chrome()
        elif self._tts_cache is not None:
            self.bot.tts = self._tts_cache
            self.tts_pref = self._settings.get("tts") or "kokoro"
            self.notify("Voice playback on.")
            self._save_settings()
            self._refresh_chrome()
        else:
            engines = list_tts_engines()
            if not engines:
                self.notify("No TTS engines are available.", severity="warning")
                return
            self._set_tts_engine(engines[0]["key"])

    def _set_tts_engine(self, key: str | None) -> None:
        if not key or key == "off":
            if self.bot is not None and self.bot.tts is not None:
                self._tts_cache = self.bot.tts
                self.bot.tts = None
            self.tts_pref = None
            self._save_settings()
            self._refresh_chrome()
            return
        self.tts_pref = key
        if self.bot is None:
            self._save_settings()
            self._refresh_chrome()
            return
        self._tts_loading = True
        self._refresh_chrome()

        def load_engine() -> None:
            self.call_from_thread(self._tts_loaded, key, init_tts(key))

        self.run_worker(load_engine, thread=True)

    def _tts_loaded(self, key: str, engine) -> None:
        self._tts_loading = False
        if engine is None:
            self.notify("Voice engine setup failed.", severity="error")
        else:
            self.bot.tts = engine
            self.tts_pref = key
            self.notify(f"Voice engine ready: {key}")
        self._save_settings()
        self._refresh_chrome()

    def _toggle_autonomy(self) -> None:
        self.autonomous_on = not self.autonomous_on
        if self.autonomous_on and self.autonomy_interval < MIN_AUTO_INTERVAL:
            self.autonomy_interval = 120
        self._sync_bot_autonomy()
        if (
            not self.autonomous_on
            and self.activity.blocks_input
            and self.activity.label.startswith("AUTO //")
        ):
            self._request_cancel()
        state = (
            f"free · {self.autonomy_interval}s idle"
            if self.autonomous_on else "off"
        )
        self.notify(f"Autonomous mode {state}.")
        self._save_settings()
        self._refresh_chrome()

    def _sync_bot_autonomy(self) -> None:
        if self.bot is None:
            return
        if self.autonomous_on and hasattr(self.bot, "begin_autonomy"):
            self.bot.begin_autonomy()
        elif not self.autonomous_on and hasattr(self.bot, "end_autonomy"):
            self.bot.end_autonomy()

    def _autonomy_check(self) -> None:
        if self.bot is None or self.activity.blocks_input or not self.autonomous_on:
            return
        idle = time.time() - self._last_interaction
        if idle < max(self.autonomy_interval, MIN_AUTO_INTERVAL):
            return
        token = self._start_activity("autonomous", "AUTO // GENERATING", cancellable=True)
        self._show_thinking(f"Autonomous check · {self.autonomy_interval}s interval")
        self.run_worker(lambda: self._auto_work(idle, token), thread=True)

    def _auto_work(self, idle: float, token: int) -> None:
        with self._bot_lock:
            try:
                response = self.bot.autonomous_tick(idle)
            except Exception as exc:  # noqa: BLE001
                _cli_persist_session_events(self.bot)
                self.call_from_thread(self._finish_activity, token, error=str(exc))
                return
            if response:
                transcript = self.bot.last_autonomous_transcript or {}
                _cli_record_local_episode(
                    self.bot.session_uuid,
                    transcript.get("user", "[AUTONOMOUS MOMENT]"),
                    response,
                )
            _cli_persist_session_events(self.bot)
        if self._is_cancelled(token):
            self.call_from_thread(self._finish_cancelled, token)
        elif response:
            self.call_from_thread(self._finish_reply, response, token)
        else:
            self.call_from_thread(self._finish_activity, token)

    def _render_help(self) -> None:
        lines = ["COMMAND DECK // shortcuts are optional"]
        for command, description in TUI_COMMANDS.items():
            lines.append(f"  {command:12s} {description}")
        lines.append("\nTOOLS // buttons insert these commands")
        for name, tool in TOOLS.items():
            lines.append(f"  {name:12s} {tool['help']}")
        self._mount_chat(Static("\n".join(lines), markup=False, classes="tool-result"))

    def _run_tui_command(self, command: str, argument: str) -> None:
        if command == "/quit":
            self.exit()
        elif command == "/new":
            self.action_new_session()
        elif command == "/sessions":
            self._open_drawer()
            self.query_one("#session-search", Input).focus()
        elif command == "/settings":
            self._open_settings()
        elif command == "/model":
            self._open_model_manager()
        elif command == "/routing":
            value = argument.strip().casefold()
            if not value:
                def routing_picked(selected: str | None) -> None:
                    if selected in ROUTING_MODES:
                        self.tool_routing_mode = normalize_routing_mode(selected)
                        self._apply_tool_routing_to_bot()
                        self._save_settings()
                        self._refresh_chrome()

                self.push_screen(
                    ChoiceScreen(
                        "TOOL ROUTING // SELECT ARCHITECTURE",
                        ROUTING_CHOICES,
                    ),
                    routing_picked,
                )
            elif value in ROUTING_MODES:
                self.tool_routing_mode = normalize_routing_mode(value)
                self._apply_tool_routing_to_bot()
                self._save_settings()
                self._refresh_chrome()
            else:
                self.notify(
                    "Use /routing direct or /routing dispatcher.",
                    severity="warning",
                )
        elif command == "/reasoning":
            mode = argument.strip().casefold()
            if not mode:
                self._open_reasoning_picker()
            elif mode in REASONING_MODES:
                self._set_reasoning_mode(mode)
            else:
                self.notify(
                    "Use /reasoning auto|minimal|low|medium|high|off.",
                    severity="warning",
                )
        elif command == "/persona":
            self._open_persona_picker()
        elif command == "/tts":
            self.action_toggle_tts()
        elif command == "/theme":
            self._open_theme_picker()
        elif command == "/stats":
            self._show_brain()
        elif command == "/auto":
            self._set_auto_from_command(argument)
        elif command == "/watch":
            self._begin_watch(argument)

    def _set_auto_from_command(self, argument: str) -> None:
        if argument.strip().casefold() in {"off", "0"}:
            self.autonomous_on = False
        else:
            try:
                wanted = int(argument) if argument.strip() else self.autonomy_interval
            except ValueError:
                self.notify("Use /auto off or /auto 300.", severity="warning")
                return
            self.autonomy_interval = max(MIN_AUTO_INTERVAL, wanted)
            self.autonomous_on = True
        self._sync_bot_autonomy()
        self._save_settings()
        self._refresh_chrome()

    def _begin_watch(self, argument: str) -> None:
        token = self._start_activity("tool", "TOOL // EYE CONTROL", cancellable=True)
        card = self._new_tool_card("eye control", argument)

        def watch_work() -> None:
            subcommand = argument.strip().casefold()
            try:
                if subcommand in {"off", "stop"}:
                    import subprocess
                    subprocess.run(["pkill", "-f", "eyeoftruth"], capture_output=True, check=False)
                    result = "Eye closed."
                else:
                    from tools.see.see import open_eye
                    seconds = int(subcommand) if subcommand else 300
                    result = open_eye(seconds) or f"Eye open for {seconds}s."
                self.call_from_thread(self._finish_watch, token, card, result, None)
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._finish_watch, token, card, str(exc), str(exc))

        self.run_worker(watch_work, thread=True)

    def _finish_watch(
        self, token: int, card: ToolActivityCard, result: str, error: str | None,
    ) -> None:
        card.finish("FAILURE" if error else "SUCCESS", result)
        self._finish_activity(token, error=error)

    def _open_settings(self) -> None:
        if self.activity.blocks_input:
            self.notify("Settings can open after the active operation finishes.", severity="warning")
            return
        self._close_drawer()
        self.push_screen(SettingsScreen())

    def _setting_action(self, key: str, screen: SettingsScreen) -> None:
        def refresh(_value=None) -> None:
            if screen.is_mounted:
                screen.refresh_rows()

        if key == "model":
            # The manager must be opened by SettingsScreen.on_unmount. Pushing
            # it in this same ListView.Selected event displays the screen but
            # loses its result callback, so model clicks silently return None.
            screen.open_model_manager_after_close = True
            screen.dismiss(None)
        elif key == "tool_routing":
            def routing_picked(value: str | None) -> None:
                if value in ROUTING_MODES:
                    self.tool_routing_mode = normalize_routing_mode(value)
                    self._apply_tool_routing_to_bot()
                    self._save_settings()
                    self._refresh_chrome()
                    self.notify(
                        f"Tool routing set to {self.tool_routing_mode.upper()}.",
                    )
                refresh()

            self.push_screen(
                ChoiceScreen(
                    "TOOL ROUTING // SELECT ARCHITECTURE",
                    ROUTING_CHOICES,
                ),
                routing_picked,
            )
        elif key == "dispatcher_model":
            screen.open_dispatcher_manager_after_close = True
            screen.dismiss(None)
        elif key == "tool_protocol":
            def protocol_picked(value: str | None) -> None:
                if value in ROUTING_PROTOCOLS:
                    self.tool_routing_protocol = normalize_routing_protocol(value)
                    self._apply_tool_routing_to_bot()
                    self._save_settings()
                    self._refresh_chrome()
                    self.notify(
                        f"Tool protocol set to {self.tool_routing_protocol.upper()}.",
                    )
                refresh()

            self.push_screen(
                ChoiceScreen("TOOL PROTOCOL // SELECT", ROUTING_PROTOCOL_CHOICES),
                protocol_picked,
            )
        elif key == "reasoning":
            self._open_reasoning_picker(refresh)
        elif key == "persona":
            self._open_persona_picker(refresh)
        elif key == "tts":
            options = [("off", "OFF · no speech")]
            options.extend(
                (engine["key"], engine.get("label", engine["key"]))
                for engine in list_tts_engines()
            )

            def engine_picked(value: str | None) -> None:
                if value is not None:
                    self._set_tts_engine(value)
                refresh()

            self.push_screen(ChoiceScreen("VOICE ENGINE // SELECT", options), engine_picked)
        elif key == "voice":
            def voice_picked(value: str | None) -> None:
                if value:
                    self.tts_voice = value
                    self._save_settings()
                    self._refresh_chrome()
                refresh()

            self.push_screen(
                ChoiceScreen("VOICE CHARACTER // SELECT", list(VOICE_LABELS.items())),
                voice_picked,
            )
        elif key == "auto":
            options = [
                (
                    str(seconds),
                    "OFF" if seconds == 0 else f"FREE after {seconds} seconds idle",
                )
                for seconds in AUTO_STEPS
            ]

            def auto_picked(value: str | None) -> None:
                if value is not None:
                    seconds = int(value)
                    self.autonomous_on = seconds > 0
                    if seconds:
                        self.autonomy_interval = seconds
                    self._sync_bot_autonomy()
                    self._save_settings()
                    self._refresh_chrome()
                refresh()

            self.push_screen(ChoiceScreen("AUTONOMOUS INTERVAL // SELECT", options), auto_picked)
        elif key == "theme":
            self._open_theme_picker(refresh)
        elif key == "image":
            current = os.environ.get("CF_IMAGE_URL", "")

            def image_picked(value: str | None) -> None:
                if value is not None:
                    if value:
                        os.environ["CF_IMAGE_URL"] = value
                        self._settings["cf_image_url"] = value
                    else:
                        os.environ.pop("CF_IMAGE_URL", None)
                        self._settings.pop("cf_image_url", None)
                    save_settings(self._settings)
                refresh()

            self.push_screen(
                TextEntry("IMAGE WORKER URL // empty uses default", current), image_picked,
            )

    def _open_reasoning_picker(self, callback=None) -> None:
        if self.activity.blocks_input:
            self.notify(
                "Reasoning can change after the active operation finishes.",
                severity="warning",
            )
            return
        options = list(REASONING_CHOICES)
        model_id = getattr(self.bot, "model_id", None)
        if openrouter_reasoning_is_mandatory(model_id):
            options[-1] = (
                "off",
                "OFF · unavailable — this model requires reasoning",
            )

        def picked(value: str | None) -> None:
            if value is not None:
                self._set_reasoning_mode(value)
            if callback:
                callback(value)

        self.push_screen(
            ChoiceScreen("REASONING // SELECT", options),
            picked,
        )

    def _set_reasoning_mode(self, value: str) -> bool:
        mode = normalize_reasoning_mode(value, default="auto")
        model_id = getattr(self.bot, "model_id", None)
        if mode == "off" and openrouter_reasoning_is_mandatory(model_id):
            self.notify(
                "This model requires reasoning. AUTO follows its required default.",
                severity="warning",
            )
            return False
        self.reasoning_mode = mode
        if self.bot is not None:
            self.bot.reasoning_mode = mode
        self._save_settings()
        self._refresh_chrome()
        self.notify(f"Reasoning set to {mode.upper()}.")
        return True

    def _enforce_reasoning_requirement(self, *, notify: bool) -> bool:
        """Keep the TUI honest after metadata or a provider retry learns OFF is impossible."""
        model_id = getattr(self.bot, "model_id", None)
        if self.reasoning_mode != "off" or not openrouter_reasoning_is_mandatory(model_id):
            return False
        self.reasoning_mode = "auto"
        if self.bot is not None:
            self.bot.reasoning_mode = "auto"
        self._save_settings()
        self._refresh_chrome()
        if notify:
            self.notify(
                "This model requires reasoning, so Yuki switched the control to AUTO.",
                severity="warning",
            )
        return True

    def _open_theme_picker(self, callback=None) -> None:
        def picked(value: str | None) -> None:
            if value in THEME_CYCLE:
                self._theme_idx = THEME_CYCLE.index(value)
                self.theme = value
                self._save_settings()
                self._refresh_chrome()
            if callback:
                callback(value)

        self.push_screen(
            ChoiceScreen("TERMINAL THEME // SELECT", [(name, name) for name in THEME_CYCLE]),
            picked,
        )

    def _open_persona_picker(self, callback=None) -> None:
        options = [(persona["name"], persona["name"]) for persona in list_personas()]
        if not options:
            self.notify("No personas found.", severity="warning")
            return

        def picked(name: str | None) -> None:
            self._persona_picked(name)
            if callback:
                callback(name)

        self.push_screen(SearchPicker("PERSONA // SELECT", [("available", options)]), picked)

    def _persona_picked(self, name: str | None) -> None:
        if not name:
            return
        text = load_persona_by_name(name)
        if not text:
            self.notify(f"Persona '{name}' was not found.", severity="error")
            return
        self.persona_name = name
        self.persona_text = text
        if self.bot is not None:
            self.bot.system_prompt = text.rstrip() + "\n\n" + _backend_self_awareness(
                self.bot.backend, self.bot.model_id,
            )
        self._mount_chat(StatusCard("PERSONA // CHANGED", name))
        self._save_settings()

    def _open_model_manager(self) -> None:
        if self.activity.blocks_input:
            self.notify("Stop or finish the active operation first.", severity="warning")
            return
        self._close_drawer()
        self.push_screen(
            ModelManagerScreen(
                self._settings,
                api_only=self.api_only,
                purpose="main",
                active_label=self.model_label,
            ),
            self._model_manager_result,
        )

    def _open_dispatcher_manager(self) -> None:
        if self.activity.blocks_input:
            self.notify("Stop or finish the active operation first.", severity="warning")
            return
        self._close_drawer()
        self.push_screen(
            ModelManagerScreen(
                self._settings,
                purpose="dispatcher",
                title="DISPATCHER MODEL // SELECT ROUTER",
                active_label=self.dispatcher_model_label,
            ),
            self._dispatcher_manager_result,
        )

    def _load_model_catalog(
        self, screen: ModelManagerScreen, source: str, config: dict,
    ) -> None:
        self.run_worker(lambda: self._model_catalog_work(screen, source, config), thread=True)

    def _model_catalog_work(
        self, screen: ModelManagerScreen, source: str, config: dict,
    ) -> None:
        try:
            if source == "local":
                catalog = (
                    list_dispatcher_models()
                    if screen.purpose == "dispatcher"
                    else list_local_models()
                )
                options = [(model["id"], model["label"]) for model in catalog]
            elif source == "openrouter":
                key = config.get("openrouter_key", "")
                if not key:
                    raise ValueError("Enter an OpenRouter key first.")
                os.environ["OPENROUTER_API_KEY"] = key
                self._settings["openrouter_key"] = key
                save_settings(self._settings)
                options = [(model["id"], model["label"]) for model in fetch_openrouter_models()]
            elif source == "remote":
                base_url = config.get("remote_base_url", "")
                api_key = config.get("remote_api_key", "")
                if not base_url:
                    raise ValueError("A remote base URL is required.")
                from openai import OpenAI
                client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
                options = [
                    (f"remote/{model.id}", model.id) for model in client.models.list().data
                ]
            else:
                options = []
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(screen.set_options, [], f"Catalog error: {exc}")
            return
        message = f"{len(options)} {source} models" if options else f"No {source} models found."
        self.call_from_thread(screen.set_options, options, message)

    def _dispatcher_manager_result(self, result: dict | None) -> None:
        if not result:
            return
        config = result.get("config", {})
        if config.get("openrouter_key"):
            os.environ["OPENROUTER_API_KEY"] = config["openrouter_key"]
            self._settings["openrouter_key"] = config["openrouter_key"]
        if config.get("remote_base_url"):
            os.environ["REMOTE_LLM_BASE_URL"] = config["remote_base_url"]
            self._settings["remote_base_url"] = config["remote_base_url"]
            if config.get("remote_api_key"):
                os.environ["REMOTE_LLM_API_KEY"] = config["remote_api_key"]
                self._settings["remote_api_key"] = config["remote_api_key"]
        self.dispatcher_model_id = result["id"]
        self.dispatcher_model_label = result.get("label") or Path(result["id"]).name
        recent = [
            entry
            for entry in self._settings.get("recent_dispatcher_models", [])
            if entry.get("id") != self.dispatcher_model_id
        ]
        recent.insert(0, {
            "id": self.dispatcher_model_id,
            "label": self.dispatcher_model_label,
            "source": result.get("source", "recent"),
        })
        self._settings["recent_dispatcher_models"] = recent[:6]
        self._apply_tool_routing_to_bot()
        self._save_settings()
        self._refresh_chrome()
        self._mount_chat(StatusCard(
            "DISPATCHER // SELECTED",
            f"{self.dispatcher_model_label} · loads lazily on the next delegated tool request.",
        ))
        self.notify(f"Dispatcher set to {self.dispatcher_model_label}.")

    def _model_manager_result(self, result: dict | None) -> None:
        if not result:
            return
        config = result.get("config", {})
        source = result.get("source", "recent")
        if config.get("openrouter_key"):
            os.environ["OPENROUTER_API_KEY"] = config["openrouter_key"]
            self._settings["openrouter_key"] = config["openrouter_key"]
        if config.get("remote_base_url"):
            os.environ["REMOTE_LLM_BASE_URL"] = config["remote_base_url"]
            self._settings["remote_base_url"] = config["remote_base_url"]
            self._settings["remote_model"] = config.get("remote_model", "")
            if config.get("remote_api_key"):
                os.environ["REMOTE_LLM_API_KEY"] = config["remote_api_key"]
                self._settings["remote_api_key"] = config["remote_api_key"]
            else:
                os.environ.pop("REMOTE_LLM_API_KEY", None)
                self._settings.pop("remote_api_key", None)
        if "provider" in result and result["id"].startswith("openrouter/"):
            self._settings.setdefault("openrouter_providers", {})[result["id"]] = result["provider"]
        save_settings(self._settings)
        requested_label = result.get("label") or result["id"]
        if self.bot is not None and self.bot.model_id == result["id"]:
            self._apply_model_provider(self.bot)
            self._refresh_chrome()
            self.notify(f"{requested_label} · provider: {result.get('provider') or 'Automatic'}")
            self._mount_chat(StatusCard(
                "MODEL // ALREADY ACTIVE",
                requested_label,
            ))
            return
        self.notify(f"Switching neural link to {requested_label}…")
        self._begin_model_load(
            result["id"], requested_label, source,
        )

    def _begin_model_load(
        self, model_id: str, label: str, source: str = "recent",
    ) -> None:
        if self.activity.blocks_input:
            self.notify("Another operation is already active.", severity="warning")
            return
        self._pending_model_label = label or model_id.split("/")[-1]
        token = self._start_activity(
            "model_loading", f"LOADING // {label}", cancellable=True,
        )
        self._mount_chat(StatusCard("MODEL // SWITCHING", label or model_id))
        self.run_worker(
            lambda: self._load_model_work(model_id, label, source, token), thread=True,
        )

    def _load_model_work(
        self, model_id: str, label: str, source: str, token: int,
    ) -> None:
        old = self.bot
        with self._bot_lock:
            try:
                new = VoiceChatBot(
                    model_id, system_prompt=self.persona_text,
                    tts=old.tts if old else None,
                )
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._model_load_failed, token, exc)
                return
            self._apply_model_provider(new)
            new.reasoning_mode = self.reasoning_mode
            if hasattr(new, "configure_tool_routing"):
                new.configure_tool_routing(
                    mode=self.tool_routing_mode,
                    protocol=self.tool_routing_protocol,
                    dispatcher_model_id=self.dispatcher_model_id,
                )
            if self._is_cancelled(token):
                try:
                    import ms_llama
                    ms_llama._ACTIVE_BOT = old
                except Exception:  # noqa: BLE001, S110
                    pass
                self.call_from_thread(self._finish_cancelled, token)
                return
            if old is not None:
                new.history = old.history
                new.session_uuid = old.session_uuid
                if hasattr(new, "copy_session_context_from"):
                    new.copy_session_context_from(old)
            elif self.tts_pref:
                new.tts = init_tts(self.tts_pref)
            new._emit_tool_progress = self._tool_progress_sink
            new._emit_routing_progress = self._routing_progress_sink
            if self.autonomous_on and hasattr(new, "begin_autonomy"):
                new.begin_autonomy()
        self.call_from_thread(
            self._model_load_done, new, model_id, label, source, token, old is not None,
        )

    def _model_load_failed(self, token: int, exc: Exception) -> None:
        self._pending_model_label = None
        active = self.model_label or "none"
        self._finish_activity(
            token,
            error=f"Model switch failed; still using {active}. {exc}",
        )

    def _model_load_done(
        self, new, model_id: str, label: str, source: str,
        token: int, was_switch: bool,
    ) -> None:
        if self._is_cancelled(token):
            self._finish_cancelled(token)
            return
        self.bot = new
        self.model_label = label or model_id.split("/")[-1]
        self._pending_model_label = None
        self._model_source = self._source_for_model(model_id)
        self._enforce_reasoning_requirement(notify=True)
        self._remember_model(model_id, self.model_label, source)
        message = "Same chat and memory retained." if was_switch else "Neural link online. Say hi."
        self._mount_chat(StatusCard(f"MODEL // {self.model_label}", message))
        self._finish_activity(token)
        self._save_settings()
        self.query_one("#composer", Composer).focus()

    def _resume_session(self, session_uuid: str) -> None:
        if self.bot is None:
            self.notify("Load a model before resuming a session.", severity="warning")
            return
        if self.activity.blocks_input:
            self.notify("Finish or stop the active operation first.", severity="warning")
            return
        rows = read_transcript(session_uuid)
        if not rows:
            self.notify("That session has no stored turns.", severity="warning")
            return
        self._clear_hero()
        chat = self.query_one("#chat", VerticalScroll)
        chat.remove_children()
        self._mount_chat(
            StatusCard("SESSION // RESUMED", "Recent turns restored into model context."),
            animate=False,
        )
        for user_message, assistant_message in rows[-30:]:
            if not user_message.startswith("[AUTONOMOUS "):
                self._user_bubble(user_message, animate=False)
            self._bot_bubble(assistant_message, animate=False)
        history: list[dict] = []
        for user_message, assistant_message in rows[-MAX_HISTORY_TURNS:]:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})
        self.bot.history = history
        self.bot.session_uuid = session_uuid
        if hasattr(self.bot, "load_session_events"):
            try:
                events = episodic.read_session_events(session_uuid)
            except Exception:  # noqa: BLE001
                events = []
            self.bot.load_session_events(events)
        if hasattr(self.bot, "_trim_history"):
            self.bot._trim_history()
        if self.autonomous_on and hasattr(self.bot, "begin_autonomy"):
            self.bot.begin_autonomy()
        self._session_title = self._clean_title(rows[0][0])
        self._refresh_chrome()
        self._refresh_sidebar()
        self._close_drawer()
        self.query_one("#composer", Composer).focus()

    async def action_new_session(self) -> None:
        if self.bot is None:
            self.notify("Load a model before creating a model session.", severity="warning")
            return
        if self.activity.blocks_input:
            self.notify("Finish or stop the active operation first.", severity="warning")
            return
        old_session = self.bot.session_uuid
        if hasattr(self.bot, "reset_session_context"):
            self.bot.reset_session_context()
        else:
            self.bot.history = []
        self.bot.session_uuid = uuid.uuid4().hex
        if self.autonomous_on and hasattr(self.bot, "begin_autonomy"):
            self.bot.begin_autonomy()
        self._session_title = "New chat"
        chat = self.query_one("#chat", VerticalScroll)
        self._hero = None
        await chat.remove_children()
        self._mount_hero()
        self._refresh_chrome()
        # The old transcript is already durable. Show it immediately; its
        # optional background summary must not gate session navigation.
        self._refresh_sidebar()
        self._close_drawer()
        self.query_one("#composer", Composer).focus()

        def summarize_and_refresh() -> None:
            try:
                episodic.summarize_session(old_session)
            except Exception:  # noqa: BLE001, S110
                pass
            self.call_from_thread(self._refresh_sidebar)

        self.run_worker(summarize_and_refresh, thread=True)

    def action_focus_composer(self) -> None:
        self.query_one("#composer", Composer).focus()


def _match_model_quietly(needle: str, api_only: bool) -> tuple[str, str] | None:
    needle = needle.casefold()
    candidates: list[dict] = []
    if not api_only:
        try:
            candidates += list_local_models()
        except Exception:  # noqa: BLE001, S110
            pass
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            candidates += fetch_openrouter_models()
        except Exception:  # noqa: BLE001, S110
            pass
    matches = [
        model for model in candidates
        if needle in model["label"].casefold() or needle in model["id"].casefold()
    ]
    if len(matches) == 1:
        return matches[0]["id"], matches[0]["label"]
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="llama-voice-assist neon Textual workspace")
    parser.add_argument("-m", "--model", help="model name substring to load")
    parser.add_argument("-p", "--persona", help="persona name")
    parser.add_argument("--api", action="store_true", help="show only OpenRouter models")
    parser.add_argument("--no-tts", action="store_true", help="disable TTS this run")
    parser.add_argument(
        "--auto", type=int, default=None, metavar="SECS",
        help="autonomous idle interval, 0 disables",
    )
    args = parser.parse_args()
    settings = load_settings()
    if settings.get("openrouter_key") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = settings["openrouter_key"]
    if settings.get("cf_image_url"):
        os.environ.setdefault("CF_IMAGE_URL", settings["cf_image_url"])
    if settings.get("remote_base_url"):
        os.environ.setdefault("REMOTE_LLM_BASE_URL", settings["remote_base_url"])
    if settings.get("remote_api_key"):
        os.environ.setdefault("REMOTE_LLM_API_KEY", settings["remote_api_key"])

    persona_name = args.persona or settings.get("persona") or "yuki"
    persona_text = load_persona_by_name(persona_name)
    preferred: tuple[str, str] | None = None
    if args.model:
        preferred = _match_model_quietly(args.model, args.api)
    elif settings.get("model_id"):
        preferred = (settings["model_id"], settings.get("model_label") or settings["model_id"])
    auto = args.auto if args.auto is not None else settings.get("auto", 120)
    tts_pref = None if args.no_tts else settings.get("tts")

    try:
        import textual_image.widget  # noqa: F401
    except Exception:  # noqa: BLE001, S110
        pass

    YukiTUI(
        persona_name=persona_name,
        persona_text=persona_text,
        autonomy_interval=max(0, auto),
        api_only=args.api,
        preferred_model=preferred,
        tts_pref=tts_pref,
    ).run()
