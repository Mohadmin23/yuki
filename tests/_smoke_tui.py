"""Manual headless TUI smoke: no model, tool, TTS, network, or service calls.

Run with: uv run python tests/_smoke_tui.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from interface import tui

tui.read_recent_sessions = list
tui.search_episodes = lambda _needle: []
tui._cli_record_episode = lambda *_args, **_kwargs: None


class StubBot:
    model_id = "stub/model"
    backend = "stub"

    def __init__(self):
        self.history = []
        self.last_stats = None
        self.session_uuid = "smoke-session"
        self.tts = None
        self.system_prompt = "stub"
        self._emit_tool_progress = None

    def react_chat(self, text):
        self.history.extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": "stub reply"},
        ])
        return "stub reply"


async def smoke() -> None:
    cold = tui.YukiTUI(autonomy_interval=0, preferred_model=None)
    cold._settings["recent_models"] = []
    async with cold.run_test(size=(66, 28)) as pilot:
        await pilot.pause()
        composer = cold.query_one("#composer", tui.Composer)
        composer.value = "keep this draft"
        cold._handle_text(composer.value)
        await pilot.pause()
        assert composer.value == "keep this draft"
        assert isinstance(cold.screen_stack[-1], tui.ModelManagerScreen)
        await pilot.press("escape")

        token = cold._start_activity("generating", "GENERATING // TEST", cancellable=True)
        cold._handle_text(composer.value)
        assert composer.value == "keep this draft"
        assert str(cold.query_one("#send-button", tui.RetroButton).label) == "STOP"
        cold._request_cancel()
        assert str(cold.query_one("#send-button", tui.RetroButton).label) == "CANCELING"
        cold._finish_activity(token)

        assert not cold.query_one("#sidebar").display
        assert cold.query_one("#model-status").display
        assert cold.query_one("#activity-status").display
        assert cold.query_one("#voice-status").display
        assert cold.query_one("#auto-status").display
        cold.action_toggle_sidebar()
        assert cold.query_one("#sidebar").display

    app = tui.YukiTUI(
        StubBot(), model_label="stub-model", persona_text="stub", autonomy_interval=0,
    )
    async with app.run_test(size=(120, 38)) as pilot:
        await pilot.pause()
        composer = app.query_one("#composer", tui.Composer)
        composer.value = "hello yuki"
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause(0.02)
            if not app.activity.blocks_input:
                break
        assert composer.value == ""
        assert len(app.query(".row.user")) == 1
        assert len(app.query(".bot-md")) == 1
        assert all(control.can_focus for control in app.query(tui.RetroButton))


if __name__ == "__main__":
    asyncio.run(smoke())
    print("TUI headless smoke: PASS")
