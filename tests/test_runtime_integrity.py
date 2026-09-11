"""Regression checks for durable turns, completed actions, and tool boundaries."""

import asyncio
import json
import sqlite3
import struct
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import episodic
import ms_llama
from interface import tui
from tools._helpers import ToolResult
from tools.shell.shell import tool_shell


@pytest.fixture(autouse=True)
def isolated_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(episodic, "DB_PATH", tmp_path / "episodes.db")
    monkeypatch.setattr(ms_llama, "_ACTIVE_BOT", None)
    monkeypatch.setattr(episodic, "_get_client", MagicMock(side_effect=RuntimeError("offline")))


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.setattr(ms_llama, "setup_llm", lambda *_args: (MagicMock(), None, "gguf"))
    return ms_llama.VoiceChatBot("test-model", system_prompt="Test persona")


def test_embedding_failure_keeps_resumable_turn_and_reports_degradation(caplog):
    episodic.record("offline-session", "Keep this conversation", "Still here")

    assert tui.read_transcript("offline-session") == [("Keep this conversation", "Still here")]
    assert episodic.count_episodes() == 1
    assert "Chat saved locally; semantic indexing unavailable" in caplog.text


def test_vector_initialization_failure_does_not_duplicate_or_lose_turn(monkeypatch):
    monkeypatch.setattr(episodic, "_embed", lambda _: b"embedding")
    monkeypatch.setattr(episodic, "_get_db", MagicMock(side_effect=RuntimeError("no extension")))
    episodic.record("session", "Question", "Answer")
    assert tui.read_transcript("session") == [("Question", "Answer")]


def test_episodic_record_recall_and_summary_across_threads(monkeypatch):
    vector = struct.pack(f"{episodic.EMBED_DIM}f", 1.0, *([0.0] * (episodic.EMBED_DIM - 1)))
    monkeypatch.setattr(episodic, "_embed", lambda _: vector)
    # Initialize on this thread, then exercise connections on other threads.
    episodic.record("initial", "First", "Reply")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(episodic.record, f"session-{i}", f"Question {i}", "Answer") for i in range(6)]
        for future in futures:
            future.result()
        summary = pool.submit(episodic.summarize_session, "initial").result()
        recalled = pool.submit(episodic.recall_sessions, "First").result()
        assert pool.submit(episodic.count_episodes).result() == 7
    assert summary
    assert recalled
    with sqlite3.connect(episodic.DB_PATH) as db:
        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 7
    # All turns were indexed, not just locally saved after swallowed thread errors.
    db = episodic._get_db()
    try:
        assert db.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0] == 7
    finally:
        db.close()


def _worker_app(bot):
    return SimpleNamespace(
        bot=bot,
        _bot_lock=threading.Lock(),
        _is_cancelled=lambda _token: True,
        call_from_thread=lambda *_args, **_kwargs: None,
        _finish_cancelled=lambda *_args: None,
        _finish_direct_tool=lambda *_args: None,
    )


def test_cancel_keeps_completed_tool_evidence(bot, monkeypatch, tmp_path):
    target = tmp_path / "completed.txt"
    old_history = [{"role": "user", "content": "Earlier conversation"}]
    bot.history = list(old_history)

    def action(_text):
        target.write_text("finished")
        bot._remember_tool_outcome(
            source="human", request="write the note", tool="yuki_write",
            arguments="completed.txt|finished", result="Wrote the note", succeeded=True,
        )
        bot.history.append({"role": "assistant", "content": "Discard this reply"})
        return "Done"

    monkeypatch.setattr(bot, "react_chat", action)
    tui.YukiTUI._chat_work(_worker_app(bot), "write the note", 1)

    assert target.read_text() == "finished"
    assert bot.history == old_history
    assert bot._session_continuity[-1]["succeeded"] is True
    assert episodic.read_session_events(bot.session_uuid)[-1]["tool"] == "yuki_write"
    assert bot.drain_session_events() == []


def test_cancel_direct_command_still_records_result(bot, monkeypatch):
    monkeypatch.setattr(tui, "run_tool", lambda _: (ToolResult("finished", succeeded=True), "/shell", ""))
    tui.YukiTUI._direct_tool_work(_worker_app(bot), "/shell echo hello", 1)
    assert episodic.read_session_events(bot.session_uuid)[-1]["succeeded"] is True


def test_autonomous_followup_failure_retains_action_and_returns_truth(bot, monkeypatch):
    monkeypatch.setattr(bot, "_run_tool_with_spinner", lambda *_args, **_kwargs: ("Wrote the note", False))

    def failed_reply(*_args, **_kwargs):
        assert bot._session_continuity[-1]["succeeded"] is True
        assert bot._autonomous_actions[-1]["tool"] == "yuki_write"
        raise RuntimeError("response model unavailable")

    monkeypatch.setattr(bot, "chat", failed_reply)
    route = {"passed": True, "prepared": {
        "model_call": {"tool": "yuki_write", "arguments": {"filename": "note.txt", "content": "hello"}},
        "runtime_argument": "note.txt|hello",
    }}
    response = bot._execute_autonomous_route("write note", route)
    assert "Wrote the note" in response
    assert "follow-up model response failed" in response
    assert len(bot._session_continuity) == 1
    assert bot._suspend_history_trim == 0


@pytest.mark.parametrize("operation", ["new", "load", "delete"])
def test_web_session_boundary_clears_old_ledger(bot, monkeypatch, tmp_path, operation):
    from interface import server

    monkeypatch.setattr(server, "bot", bot)
    monkeypatch.setattr(server, "current_chat_id", "old-chat")
    monkeypatch.setattr(server, "CHATS_DIR", tmp_path / "chats")
    monkeypatch.setattr(server, "_inject_memory", lambda _bot: None)
    bot._remember_session_event({"kind": "verified_tool_outcome", "result": "OLD ACTION"})
    bot.history = [{"role": "user", "content": "OLD GOAL"}]
    bot.begin_autonomy()
    server._save_chat("old-chat", "Old", [])
    server._save_chat("loaded-chat", "Loaded", [{"role": "user", "content": "Loaded message"}])
    if operation == "new":
        asyncio.run(server.api_chats_new())
    elif operation == "load":
        asyncio.run(server.api_chats_load("loaded-chat"))
        assert bot.history == [{"role": "user", "content": "Loaded message"}]
    else:
        asyncio.run(server.api_chats_delete("old-chat"))
    assert bot._session_continuity == []
    assert bot._autonomous_context == []
    assert bot._autonomous_active is False
    assert "OLD ACTION" not in bot._system_prompt_with_continuity("Persona")


@pytest.mark.parametrize("command", [
    "echo hi; printf bypassed", "echo hi | cat", "echo $(whoami)",
    "echo `whoami`", "echo hi\nwhoami", "echo hi > output.txt", "rm file", "",
])
def test_direct_shell_rejects_unsafe_syntax_without_execution(monkeypatch, command):
    runner = MagicMock()
    monkeypatch.setattr("tools.shell.shell.subprocess.run", runner)
    result, _, _ = ms_llama.run_tool("/shell " + command)
    assert ms_llama._is_tool_error(result)
    runner.assert_not_called()


def test_shell_uses_arguments_and_preserves_explicit_success(bot, monkeypatch):
    runner = MagicMock(return_value=subprocess.CompletedProcess(["echo"], 0, "Error: is just text", ""))
    monkeypatch.setattr("tools.shell.shell.subprocess.run", runner)
    bot._emit_tool_progress = lambda _message: None
    result, blocked = bot._run_tool_with_spinner("shell", "echo Error:")
    assert runner.call_args.args[0] == ["echo", "Error:"]
    assert runner.call_args.kwargs["shell"] is False
    assert not blocked
    assert not ms_llama._is_tool_error(result)

    def reply(*_args, **_kwargs):
        assert "The tool FAILED" not in bot.history[-1]["content"]
        return "The command printed its text."

    monkeypatch.setattr(bot, "chat", reply)
    assert bot.respond_to_completed_tool("/shell echo Error:", "shell", result)


@pytest.mark.parametrize("failure", ["timeout", "exit", "spawn"])
def test_shell_failure_is_explicit_and_survives_serialization(monkeypatch, failure):
    runner = MagicMock(return_value=subprocess.CompletedProcess(["ls"], 2, "", "missing"))
    if failure == "timeout":
        runner.side_effect = subprocess.TimeoutExpired("ls", 10)
    elif failure == "spawn":
        runner.side_effect = OSError("could not start")
    monkeypatch.setattr("tools.shell.shell.subprocess.run", runner)
    result = tool_shell("ls")
    assert ms_llama._is_tool_error(result)
    assert ms_llama._is_tool_error(json.loads(json.dumps(result)))


@pytest.mark.parametrize("with_progress", [True, False])
def test_tool_exceptions_are_failures_in_every_interface(bot, monkeypatch, with_progress):
    progress = []
    bot._emit_tool_progress = progress.append if with_progress else None
    monkeypatch.setitem(ms_llama.REACT_TOOL_MAP, "shell", MagicMock(side_effect=OSError("failed")))
    result, blocked = bot._run_tool_with_spinner("shell", "ls")
    assert not blocked
    assert ms_llama._is_tool_error(result)
    assert "The tool FAILED" in ms_llama._tool_result_injection("shell", result)
    if with_progress:
        assert "— failed" in progress[-1]
        card = MagicMock()
        app = SimpleNamespace(
            _active_tool_card=card,
            _transition_activity=lambda *_args: None,
            _show_thinking=lambda *_args: None,
        )
        tui.YukiTUI._apply_tool_progress(app, progress[-1])
        card.finish.assert_called_once_with("FAILURE")


@pytest.mark.parametrize("mode,protocol", [
    ("direct", "canonical"), ("direct", "native"), ("dispatcher", "auto"),
])
def test_next_reply_after_canceled_write_reports_actual_latest_action(
    bot, monkeypatch, tmp_path, mode, protocol,
):
    """Exercise the real write/ledger/cancellation path, then ask the next question."""
    target = tmp_path / "interrupted.txt"
    monkeypatch.setattr("tools.yuki_write.yuki_write.YUKI_DIR", tmp_path)
    bot.tool_routing_protocol = "canonical"
    bot._emit_tool_progress = lambda _message: None
    bot.history = [
        {"role": "user", "content": "Check my files"},
        {"role": "assistant", "content": "The most recent action was yuki_list."},
    ]
    bot._remember_tool_outcome(
        source="human", request="Check my files", tool="yuki_list", arguments={},
        result="Files listed", succeeded=True,
    )
    ms_llama._cli_persist_session_events(bot)
    request = "Create interrupted.txt containing ACTION_SURVIVED."
    canonical = {"tool": "yuki_write", "arguments": {
        "filename": "interrupted.txt", "content": "ACTION_SURVIVED",
    }}
    prepared = bot._routing_registry.prepare_call(
        canonical, raw_request=request, available_names=bot._routing_registry.names,
    )
    assert prepared["passed"]
    monkeypatch.setattr(bot, "_direct_route_request", lambda _: {
        "passed": True, "prepared": prepared, "canonical_call": canonical,
    })
    app = _worker_app(bot)
    canceled = []
    app._is_cancelled = lambda _token: bool(canceled)
    app.autonomous_on = False

    def interrupted_reply(*_args, **_kwargs):
        assert target.read_text() == "ACTION_SURVIVED"
        canceled.append(True)
        return "This generated explanation is discarded."

    monkeypatch.setattr(bot, "chat", interrupted_reply)
    tui.YukiTUI._chat_work(app, request, 1)
    assert "yuki_list" in bot.history[-1]["content"]
    assert "interrupted.txt" not in str(bot.history)
    assert episodic.read_session_events(bot.session_uuid)[-1]["tool"] == "yuki_write"

    # Neither an older model reply nor routing mode may override the runtime ledger.
    bot.tool_routing_mode = mode
    bot.tool_routing_protocol = protocol
    model = MagicMock(side_effect=AssertionError("status needs no model or tool execution"))
    monkeypatch.setattr(bot, "chat", model)
    monkeypatch.setattr(bot, "_react_chat_native", model)
    monkeypatch.setattr(bot, "_react_chat_dispatcher", model)
    monkeypatch.setattr(bot, "_react_chat_canonical", model)
    response = bot.react_chat("What was the most recent completed tool action?")
    assert "yuki_write" in response
    assert "interrupted.txt" in response
    assert "It succeeded." in response
    assert "yuki_list" not in response
    assert bot.history[-1]["content"] == response
    model.assert_not_called()

    # Restoring the persisted ledger after a restart yields the same factual reply.
    events = episodic.read_session_events(bot.session_uuid)
    bot.reset_session_context()
    bot.load_session_events(events)
    assert bot.react_chat("What was the most recent completed tool action?") == response


def test_action_status_distinguishes_latest_attempt_from_latest_success(bot):
    bot._remember_tool_outcome(
        source="human", request="write", tool="yuki_write", arguments={},
        result="Wrote note.txt", succeeded=True,
    )
    bot._remember_tool_outcome(
        source="human", request="read missing", tool="yuki_read", arguments={},
        result="File not found: missing.txt", succeeded=False,
    )
    latest = bot.react_chat("What action did you just perform, and was it successfully completed?")
    assert "yuki_read" in latest
    assert "It did not succeed." in latest
    success = bot.react_chat("What was the last tool action you successfully completed?")
    assert "yuki_write" in success
    assert "It succeeded." in success
    bot.reset_session_context()
    empty = bot.react_chat("What was the latest tool action?")
    assert "don't have" in empty
    assert "yuki_write" not in empty


@pytest.mark.parametrize("question", [
    "What was the most recent completed tool action? Then delete the file.",
    "What was the latest action in that movie?",
    "Write 'What was the most recent completed tool action?' into a file.",
    "What was the last tool action yesterday?",
    "/shell echo what was the last action",
])
def test_action_status_does_not_swallow_other_requests(bot, monkeypatch, question):
    bot.tool_routing_protocol = "canonical"
    normal = MagicMock(return_value="normal routing")
    monkeypatch.setattr(bot, "_react_chat_canonical", normal)
    assert bot.react_chat(question) == "normal routing"
    normal.assert_called_once()
