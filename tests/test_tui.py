"""Headless TUI checks: no model loading, tool execution, TTS, or services."""
import asyncio
import sqlite3

from interface import tui


def test_recent_sessions_include_saved_transcripts_without_summaries(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "episodic-test.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE episodes ("
            "id INTEGER PRIMARY KEY, session_uuid TEXT, ts REAL, "
            "user_msg TEXT, assistant_msg TEXT)"
        )
        db.execute(
            "CREATE TABLE session_summaries ("
            "session_uuid TEXT PRIMARY KEY, ts_start REAL, ts_end REAL, "
            "summary TEXT)"
        )
        db.execute(
            "INSERT INTO episodes VALUES (1, 'summarized', 10, 'Older chat', 'Hi')"
        )
        db.execute(
            "INSERT INTO session_summaries VALUES "
            "('summarized', 10, 10, 'An existing summary')"
        )
        db.execute(
            "INSERT INTO episodes VALUES "
            "(2, 'not-yet-summarized', 20, 'Newest saved chat', 'Still here')"
        )
        db.commit()
    monkeypatch.setattr(tui.episodic, "DB_PATH", db_path)

    sessions = tui.read_recent_sessions()

    assert [session["uuid"] for session in sessions] == [
        "not-yet-summarized",
        "summarized",
    ]
    assert sessions[0] == {
        "uuid": "not-yet-summarized",
        "ts_end": 20,
        "summary": "",
        "title": "Newest saved chat",
    }
    assert sessions[1]["summary"] == "An existing summary"


class StubBot:
    model_id = "stub/model"
    backend = "stub"

    def __init__(self):
        self.history = []
        self.patience = 100
        self.last_stats = None
        self.last_reasoning = ""
        self.session_uuid = "stub-session"
        self.tts = None
        self.system_prompt = "stub"
        self._emit_tool_progress = None

    def react_chat(self, text):
        self.history.extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": "stub reply"},
        ])
        return "stub reply"


class AutonomousStubBot(StubBot):
    def __init__(self):
        super().__init__()
        self.autonomy_events = []

    def begin_autonomy(self):
        self.autonomy_events.append("begin")

    def end_autonomy(self):
        self.autonomy_events.append("end")


def test_no_model_rejection_preserves_draft(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)

    async def scenario():
        app = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
        app._settings["recent_models"] = []
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", tui.Composer)
            composer.value = "please keep this draft"
            app._handle_text(composer.value)
            await pilot.pause()
            assert composer.value == "please keep this draft"
            assert isinstance(app.screen_stack[-1], tui.ModelManagerScreen)

    asyncio.run(scenario())


def test_busy_state_preserves_draft_and_exposes_stop(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)

    async def scenario():
        app = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", tui.Composer)
            composer.value = "draft survives"
            token = app._start_activity(
                "generating", "GENERATING // TEST", cancellable=True,
            )

            app._handle_text(composer.value)
            assert composer.value == "draft survives"
            assert str(app.query_one("#send-button", tui.RetroButton).label) == "STOP"

            app._request_cancel()
            assert composer.value == "draft survives"
            assert str(app.query_one("#send-button", tui.RetroButton).label) == "CANCELING"
            app._finish_activity(token)

    asyncio.run(scenario())


def test_reasoning_card_precedes_final_answer_and_is_collapsible(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)

    async def scenario():
        bot = StubBot()
        bot.last_reasoning = "Checked the request, then selected the safest answer."
        app = tui.YukiTUI(
            bot, model_label="reasoning-model", persona_text="stub",
            autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            token = app._start_activity(
                "generating", "GENERATING // TEST", cancellable=True,
            )
            app._finish_reply("Final answer.", token)
            await pilot.pause()

            card = app.query_one(tui.ReasoningCard)
            assert "THINKING // MODEL TRACE" in str(card.render())
            assert "Checked the request" in str(card.render())
            bot_markdown = app.query_one(".bot-md", tui.Markdown)
            assert bot_markdown.source == "Final answer."
            chat_children = list(app.query_one("#chat").children)
            assert chat_children.index(card) < chat_children.index(bot_markdown.parent)
            card.action_toggle_details()
            assert "Checked the request" not in str(card.render())

    asyncio.run(scenario())


def test_reasoning_button_picker_slash_command_and_persistence(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "load_settings", lambda: {"reasoning_mode": "auto"})
    saved = []
    monkeypatch.setattr(tui, "save_settings", lambda settings: saved.append(dict(settings)))

    async def scenario():
        bot = StubBot()
        app = tui.YukiTUI(
            bot, model_label="reasoning-model", persona_text="stub",
            autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            button = app.query_one("#reasoning-button", tui.RetroButton)
            assert str(button.label) == "REASONING // AUTO"

            await pilot.click("#reasoning-button")
            await pilot.pause()
            picker = app.screen_stack[-1]
            assert isinstance(picker, tui.ChoiceScreen)
            assert [item.value for item in picker.query(tui.PickItem)] == [
                "auto", "minimal", "low", "medium", "high", "off",
            ]
            choices = picker.query_one("#choice-list", tui.ListView)
            choices.index = 2
            choices.action_select_cursor()
            await pilot.pause()

            assert app.reasoning_mode == "low"
            assert bot.reasoning_mode == "low"
            assert str(button.label) == "REASONING // LOW"
            assert saved[-1]["reasoning_mode"] == "low"

            composer = app.query_one("#composer", tui.Composer)
            composer.value = "/reasoning"
            composer.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], tui.ChoiceScreen)
            assert composer.value == ""

    asyncio.run(scenario())


def test_mandatory_reasoning_model_refuses_off(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "load_settings", lambda: {"reasoning_mode": "auto"})
    monkeypatch.setattr(tui, "save_settings", lambda _settings: None)
    monkeypatch.setattr(tui, "openrouter_reasoning_is_mandatory", lambda _model_id: True)

    async def scenario():
        bot = StubBot()
        bot.model_id = "openrouter/z-ai/glm-5"
        bot.backend = "openrouter"
        app = tui.YukiTUI(
            bot, model_label="GLM 5", persona_text="stub", autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            assert not app._set_reasoning_mode("off")
            assert app.reasoning_mode == "auto"
            assert bot.reasoning_mode == "auto"
            assert str(app.query_one("#reasoning-button", tui.RetroButton).label) == (
                "REASONING // AUTO"
            )

    asyncio.run(scenario())


def test_tool_routing_settings_switch_architecture_protocol_and_dispatcher(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "load_settings", lambda: {
        "tool_routing_mode": "direct",
        "tool_routing_protocol": "auto",
        "dispatcher_model_id": "/models/Hammer2.1-1.5b",
        "dispatcher_model_label": "Hammer 1.5B",
    })
    saved = []
    monkeypatch.setattr(tui, "save_settings", lambda value: saved.append(dict(value)))

    class RoutingBot(StubBot):
        def configure_tool_routing(self, **settings):
            self.routing_settings = settings

    async def scenario():
        bot = RoutingBot()
        app = tui.YukiTUI(
            bot,
            model_label="main brain",
            persona_text="stub",
            autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            app._open_settings()
            await pilot.pause()
            screen = app.screen_stack[-1]
            assert isinstance(screen, tui.SettingsScreen)
            keys = [row.key for row in screen.query(tui.SettingRow)]
            assert "tool_routing" in keys
            assert "dispatcher_model" in keys
            assert "tool_protocol" in keys

            app.tool_routing_mode = "dispatcher"
            app.tool_routing_protocol = "canonical"
            app.dispatcher_model_id = "/models/another-hammer"
            app.dispatcher_model_label = "Another Hammer"
            app._apply_tool_routing_to_bot()
            app._save_settings()

            assert bot.routing_settings == {
                "mode": "dispatcher",
                "protocol": "canonical",
                "dispatcher_model_id": "/models/another-hammer",
            }
            assert saved[-1]["tool_routing_mode"] == "dispatcher"
            assert saved[-1]["tool_routing_protocol"] == "canonical"
            assert saved[-1]["dispatcher_model_label"] == "Another Hammer"

    asyncio.run(scenario())


def test_new_session_waits_for_chat_removal_before_remounting_hero(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui.episodic, "summarize_session", lambda _uuid: "")

    async def scenario():
        bot = StubBot()
        app = tui.YukiTUI(
            bot, model_label="stub-model", persona_text="stub",
            autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            old_session = bot.session_uuid
            bot.history = [{"role": "user", "content": "old turn"}]
            app._user_bubble("old turn", animate=False)
            await pilot.pause()

            await pilot.click("#top-new")
            await pilot.pause()
            assert bot.session_uuid != old_session
            assert bot.history == []
            assert len(app.query("#hero")) == 1

            # Repeating the keyboard action must also keep exactly one hero.
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert len(app.query("#hero")) == 1

    asyncio.run(scenario())


def test_narrow_layout_keeps_operational_state_visible(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)

    async def scenario():
        app = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
        async with app.run_test(size=(66, 28)) as pilot:
            await pilot.pause()
            assert app._main_screen.has_class("-narrow")
            assert not app.query_one("#sidebar").display
            assert app.query_one("#statusbar").display
            assert app.query_one("#model-status").display
            assert app.query_one("#activity-status").display
            assert app.query_one("#voice-status").display
            assert app.query_one("#auto-status").display
            assert app.query_one("#reasoning-button").display
            assert app.query_one("#composer-tools-button").display

            app.action_toggle_sidebar()
            assert app.query_one("#sidebar").display
            assert app.query_one("#sidebar").has_class("drawer-open")

    asyncio.run(scenario())


def test_primary_controls_are_focusable_and_model_manager_is_unified(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)

    async def scenario():
        app = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
        app._settings["recent_models"] = []
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            controls = list(app.query(tui.RetroButton))
            assert controls
            assert all(control.can_focus for control in controls)

            app._open_model_manager()
            await pilot.pause()
            manager = app.screen_stack[-1]
            tab_ids = {button.id for button in manager.query(".model-tab")}
            assert tab_ids == {
                "model-tab-recent",
                "model-tab-local",
                "model-tab-openrouter",
                "model-tab-remote",
            }
            assert manager.query_one("#manager-openrouter-key", tui.Input)
            assert manager.query_one("#manager-remote-url", tui.Input)
            assert manager.query_one("#manager-remote-key", tui.Input)
            assert manager.query_one("#manager-remote-model", tui.Input)
            assert manager.query_one("#manager-active", tui.Static)
            assert manager.query_one("#manager-load-filtered", tui.Button)

            manager.set_options(
                [("openrouter/z-ai/glm-5", "GLM 5")],
                "1 OpenRouter model",
            )
            await pilot.pause()
            assert manager.query_one("#manager-filter", tui.Input).has_focus

    asyncio.run(scenario())


def test_settings_mouse_flow_keeps_model_manager_result_callback(monkeypatch):
    """The Settings pop must finish before Model Manager is pushed."""
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "save_settings", lambda _settings: None)

    async def scenario():
        requested = []
        app = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
        app._settings["recent_models"] = []
        app._begin_model_load = (
            lambda model_id, label, source="recent": requested.append(
                (model_id, label, source)
            )
        )
        async with app.run_test(size=(143, 64)) as pilot:
            await pilot.pause()
            app._open_settings()
            await pilot.pause()

            settings = app.screen_stack[-1]
            rows = settings.query_one("#settings-list", tui.ListView)
            rows.index = 0
            rows.action_select_cursor()
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen_stack[-1], tui.ModelManagerScreen):
                    break

            manager = app.screen_stack[-1]
            assert isinstance(manager, tui.ModelManagerScreen)
            manager._source = "openrouter"
            manager.set_options([
                ("openrouter/nvidia/nemotron-nano", "Nemotron Nano"),
                ("openrouter/z-ai/glm-5", "GLM 5"),
            ])
            await pilot.pause()
            manager.query_one("#manager-filter", tui.Input).value = "glm"
            await pilot.pause()
            await pilot.click("#manager-load-filtered")
            await pilot.pause()

            assert requested == [
                ("openrouter/z-ai/glm-5", "GLM 5", "openrouter")
            ]

    asyncio.run(scenario())


def test_neural_link_hero_tracks_pending_and_current_model(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)

    async def scenario():
        app = tui.YukiTUI(
            StubBot(), model_label="old-model", persona_text="stub",
            autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            hero_link = app.query_one("#hero-link", tui.Static)
            assert "NEURAL LINK ONLINE · old-model" in str(hero_link.render())

            app._pending_model_label = "new-openrouter-model"
            token = app._start_activity(
                "model_loading",
                "LOADING // new-openrouter-model",
                cancellable=True,
            )
            assert "NEURAL LINK SWITCHING · new-openrouter-model" in str(
                hero_link.render()
            )

            app.model_label = "new-openrouter-model"
            app._model_source = "OpenRouter"
            app._pending_model_label = None
            app._finish_activity(token)
            assert "NEURAL LINK ONLINE · new-openrouter-model" in str(
                hero_link.render()
            )
            assert "new-openrouter-model" in str(
                app.query_one("#model-status", tui.RetroButton).label
            )

    asyncio.run(scenario())


def test_openrouter_manager_result_switches_runtime_and_saved_identity(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "save_settings", lambda _settings: None)

    class SwitchedBot(StubBot):
        def __init__(self, model_id, system_prompt=None, tts=None):
            super().__init__()
            self.model_id = model_id
            self.backend = "openrouter"
            self.system_prompt = system_prompt
            self.tts = tts

    monkeypatch.setattr(tui, "VoiceChatBot", SwitchedBot)

    async def scenario():
        app = tui.YukiTUI(
            StubBot(), model_label="old-model", persona_text="stub",
            autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            app._model_manager_result({
                "id": "openrouter/z-ai/glm-5",
                "label": "GLM 5",
                "source": "openrouter",
                "config": {},
            })
            for _ in range(50):
                await pilot.pause(0.02)
                if not app.activity.blocks_input:
                    break

            assert app.bot.model_id == "openrouter/z-ai/glm-5"
            assert app.bot.backend == "openrouter"
            assert app.model_label == "GLM 5"
            assert app._model_source == "OpenRouter"
            assert app._settings["model_id"] == "openrouter/z-ai/glm-5"
            assert app._settings["model_label"] == "GLM 5"
            assert app._settings["recent_models"][0]["id"] == (
                "openrouter/z-ai/glm-5"
            )
            assert "GLM 5" in str(
                app.query_one("#model-status", tui.RetroButton).label
            )

    asyncio.run(scenario())


def test_model_filter_enter_uses_new_filter_before_list_render_finishes(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "save_settings", lambda _settings: None)

    async def scenario():
        requested = []
        app = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
        app._settings["recent_models"] = []
        app._begin_model_load = (
            lambda model_id, label, source="recent": requested.append(
                (model_id, label, source)
            )
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            app._open_model_manager()
            await pilot.pause()
            manager = app.screen_stack[-1]
            manager._source = "openrouter"
            manager.set_options([
                ("openrouter/nvidia/nemotron-nano", "Nemotron Nano"),
                ("openrouter/z-ai/glm-5", "GLM 5"),
            ])
            await pilot.pause()

            manager._rebuild("glm")
            load_button = manager.query_one("#manager-load-filtered", tui.Button)
            assert "GLM 5" in str(load_button.label)
            await pilot.click("#manager-load-filtered")
            await pilot.pause()

            assert requested == [
                ("openrouter/z-ai/glm-5", "GLM 5", "openrouter")
            ]

    asyncio.run(scenario())


def test_model_list_enter_uses_filtered_snapshot_not_stale_row(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "save_settings", lambda _settings: None)

    async def scenario():
        requested = []
        app = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
        app._settings["recent_models"] = []
        app._begin_model_load = (
            lambda model_id, label, source="recent": requested.append(
                (model_id, label, source)
            )
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            app._open_model_manager()
            await pilot.pause()
            manager = app.screen_stack[-1]
            manager._source = "openrouter"
            manager.set_options([
                ("openrouter/nvidia/nemotron-nano", "Nemotron Nano"),
                ("openrouter/z-ai/glm-5", "GLM 5"),
            ])
            await pilot.pause()

            # Do not pause after rebuilding: the rendered first row can still
            # be Nemotron even though the current filtered snapshot is GLM.
            manager._rebuild("glm")
            results = manager.query_one("#manager-list", tui.ListView)
            results.index = 0
            results.action_select_cursor()
            await pilot.pause()

            assert requested == [
                ("openrouter/z-ai/glm-5", "GLM 5", "openrouter")
            ]

    asyncio.run(scenario())


def test_model_manager_keyboard_arrows_focus_results_and_enter_loads(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "save_settings", lambda _settings: None)

    async def scenario():
        requested = []
        app = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
        app._settings["recent_models"] = []
        app._begin_model_load = (
            lambda model_id, label, source="recent": requested.append(
                (model_id, label, source)
            )
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            app._open_model_manager()
            await pilot.pause()
            manager = app.screen_stack[-1]
            manager._source = "openrouter"
            manager.set_options([
                ("openrouter/nvidia/nemotron-nano", "Nemotron Nano"),
                ("openrouter/z-ai/glm-5", "GLM 5"),
            ])
            await pilot.pause()

            model_filter = manager.query_one("#manager-filter", tui.Input)
            results = manager.query_one("#manager-list", tui.ListView)
            assert model_filter.has_focus

            await pilot.press("down")
            assert results.has_focus
            assert results.index == 0

            await pilot.press("down", "enter")
            await pilot.pause()
            assert requested == [
                ("openrouter/z-ai/glm-5", "GLM 5", "openrouter")
            ]

    asyncio.run(scenario())


def test_accepted_message_clears_only_after_runtime_accepts(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "_cli_record_episode", lambda *_args, **_kwargs: None)

    async def scenario():
        app = tui.YukiTUI(
            StubBot(), model_label="stub-model", persona_text="stub",
            autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", tui.Composer)
            composer.value = "hello yuki"
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause(0.02)
                if not app.activity.blocks_input:
                    break
            assert composer.value == ""
            assert len(app.query(".row.user")) == 1
            assert len(app.query(".bot-md")) == 1
            assert app._session_title == "hello yuki"

    asyncio.run(scenario())


def test_activity_state_does_not_treat_error_as_busy():
    assert not tui.ActivityState().blocks_input
    assert not tui.ActivityState(kind="error").blocks_input
    assert tui.ActivityState(kind="tool").blocks_input
    assert tui.ActivityState(kind="model_loading").blocks_input


def test_autonomy_toggle_starts_and_stops_durable_bot_state(monkeypatch):
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "save_settings", lambda _settings: None)

    async def scenario():
        bot = AutonomousStubBot()
        app = tui.YukiTUI(
            bot, model_label="stub-model", persona_text="stub",
            autonomy_interval=0,
        )
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            app._toggle_autonomy()
            assert app.autonomous_on
            assert bot.autonomy_events[-1] == "begin"
            assert "AUTO // FREE" in str(
                app.query_one("#auto-status", tui.RetroButton).label
            )

            app._toggle_autonomy()
            assert not app.autonomous_on
            assert bot.autonomy_events[-1] == "end"

    asyncio.run(scenario())
