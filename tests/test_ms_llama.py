import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from ms_llama import (
    VoiceChatBot,
    setup_llm,
    select_model,
    select_persona,
    select_tts,
    _STAGE_CUE_RE,
    MAX_HISTORY_TURNS,
    TTS_MAX_CHARS,
    MAX_INPUT_CHARS,
    TOOL_CALL_RE,
    auto_detect_tool,
    run_tool,
    _tool_result_injection,
    YUKI_DIR,
    ALLOWED_COMMANDS,
    _normalize_value,
    MEMORY_LIST_CATEGORIES,
    MEMORY_NAME_CATEGORIES,
    _openrouter_chat_completion,
    _openrouter_extra_body,
    _OR_MODELS_REQUIRE_REASONING,
    _extract_openrouter_reasoning,
)
# Tool implementations now live in the tools/ package (moved out of ms_llama).
from tools.calc.calc import tool_calc
from tools.time.time import tool_time
from tools.shell.shell import tool_shell
from tools.yuki_write.yuki_write import tool_yuki_write
from tools.yuki_read.yuki_read import tool_yuki_read
from tools.yuki_list.yuki_list import tool_yuki_list


def _patch_yuki_dir(monkeypatch, path):
    """Point the yuki tools at `path`. Each tool module imported YUKI_DIR by
    name from tools._helpers, so the bound name must be patched per-module."""
    for mod in ("tools.yuki_write.yuki_write",
                "tools.yuki_read.yuki_read",
                "tools.yuki_list.yuki_list"):
        monkeypatch.setattr(f"{mod}.YUKI_DIR", path)


# ============= Stage cue regex =============

def test_stage_cue_removal():
    assert _STAGE_CUE_RE.sub("", "[laughs] Hello!") == " Hello!"
    assert _STAGE_CUE_RE.sub("", "(sighs) Ok") == " Ok"
    assert _STAGE_CUE_RE.sub("", "No cues here") == "No cues here"
    assert _STAGE_CUE_RE.sub("", "[a](b)[c]") == ""


# ============= setup_llm =============

def test_setup_llm_gguf(tmp_path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"fake")

    mock_llama = MagicMock()
    with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama)}):
        model, tokenizer, backend = setup_llm(str(gguf_file))

    assert backend == "gguf"
    assert tokenizer is None
    mock_llama.assert_called_once()


def test_setup_llm_gguf_missing():
    with pytest.raises(FileNotFoundError):
        setup_llm("/nonexistent/model.gguf")


def test_setup_llm_mlx():
    mock_load = MagicMock(return_value=("mock_model", "mock_tokenizer"))
    mock_mlx = MagicMock(load=mock_load)

    with patch.dict("sys.modules", {"mlx_lm": mock_mlx}):
        model, tokenizer, backend = setup_llm("some/mlx-model")

    assert backend == "mlx"
    assert model == "mock_model"
    assert tokenizer == "mock_tokenizer"


def test_setup_llm_transformers_fallback():
    """When MLX import fails, should fall back to transformers."""
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    mock_tokenizer_cls = MagicMock()
    mock_model_cls = MagicMock()
    mock_model_cls.from_pretrained.return_value = MagicMock(device="cpu")

    with patch.dict("sys.modules", {
        "mlx_lm": None,  # simulate ImportError
        "torch": mock_torch,
        "transformers": MagicMock(
            AutoTokenizer=mock_tokenizer_cls,
            AutoModelForCausalLM=mock_model_cls,
        ),
    }):
        model, tokenizer, backend = setup_llm("some/transformers-model")

    assert backend == "transformers"
    mock_model_cls.from_pretrained.assert_called_once()
    mock_tokenizer_cls.from_pretrained.assert_called_once()


def test_openrouter_retries_and_caches_mandatory_reasoning():
    BadRequestError = type("BadRequestError", (Exception,), {})
    client = MagicMock()
    response = MagicMock()
    client.chat.completions.create.side_effect = [
        BadRequestError("Reasoning is mandatory for this endpoint and cannot be disabled."),
        response,
        response,
    ]
    model_id = "z-ai/reasoning-required-test"
    _OR_MODELS_REQUIRE_REASONING.discard(model_id)

    try:
        result = _openrouter_chat_completion(
            client,
            model=model_id,
            messages=[],
            extra_body={"reasoning": {"enabled": False, "exclude": True}, "top_k": 50},
        )
        assert result is response
        first, retry = client.chat.completions.create.call_args_list[:2]
        assert first.kwargs["extra_body"]["reasoning"]["enabled"] is False
        assert retry.kwargs["extra_body"]["reasoning"] == {
            "enabled": True, "exclude": False,
        }
        assert retry.kwargs["extra_body"]["top_k"] == 50

        _openrouter_chat_completion(
            client, model=model_id, messages=[],
            extra_body={"reasoning": {"enabled": False, "exclude": True}},
        )
        cached = client.chat.completions.create.call_args_list[2]
        assert cached.kwargs["extra_body"]["reasoning"]["enabled"] is True
        assert cached.kwargs["extra_body"]["reasoning"]["exclude"] is False
    finally:
        _OR_MODELS_REQUIRE_REASONING.discard(model_id)


def test_openrouter_reasoning_modes_preserve_auto_effort_and_off_contracts():
    assert _openrouter_extra_body("auto", {"top_k": 50}) == {"top_k": 50}
    assert _openrouter_extra_body("minimal") == {
        "reasoning": {"effort": "minimal", "exclude": False},
    }
    assert _openrouter_extra_body("high") == {
        "reasoning": {"effort": "high", "exclude": False},
    }
    assert _openrouter_extra_body("off") == {
        "reasoning": {"effort": "none", "exclude": True},
    }

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock()
    _openrouter_chat_completion(
        client,
        model="reasoning-effort-test",
        messages=[],
        extra_body=_openrouter_extra_body("medium", {"top_k": 40}),
    )
    sent = client.chat.completions.create.call_args.kwargs["extra_body"]
    assert sent == {
        "top_k": 40,
        "reasoning": {"effort": "medium", "exclude": False},
    }


def test_extract_openrouter_reasoning_prefers_readable_fields():
    message = MagicMock()
    message.reasoning = None
    message.reasoning_content = None
    message.reasoning_details = [
        {"type": "reasoning.summary", "summary": "Checked the constraints."},
        {"type": "reasoning.encrypted", "data": "do-not-render"},
        {"type": "reasoning.text", "text": "Compared the two options."},
    ]
    assert _extract_openrouter_reasoning(message) == (
        "Checked the constraints.\n\nCompared the two options."
    )


# ============= select_model =============

def test_select_model_valid(tmp_path):
    model_dir = tmp_path / "models--org--mymodel" / "snapshots" / "abc"
    model_dir.mkdir(parents=True)
    (model_dir / "model.gguf").write_bytes(b"fake")

    with patch("builtins.input", return_value="1"):
        result = select_model(str(tmp_path))

    assert result.endswith("model.gguf")


def test_select_model_no_models(tmp_path):
    with pytest.raises(RuntimeError):
        select_model(str(tmp_path))


def test_select_model_bad_dir():
    with pytest.raises(FileNotFoundError):
        select_model("/nonexistent/path")


# ============= select_persona =============

def test_select_persona_none(tmp_path):
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    (persona_dir / "friendly.txt").write_text("Be friendly!")

    with patch("builtins.input", return_value="0"):
        result = select_persona(str(persona_dir))

    assert result is None


def test_select_persona_load(tmp_path):
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    (persona_dir / "friendly.txt").write_text("Be friendly!")

    with patch("builtins.input", return_value="1"):
        result = select_persona(str(persona_dir))

    assert result == "Be friendly!"


def test_select_persona_no_dir():
    result = select_persona("/nonexistent/path")
    assert result is None


def test_select_persona_empty_dir(tmp_path):
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    result = select_persona(str(persona_dir))
    assert result is None


# ============= VoiceChatBot =============

@pytest.fixture
def mock_bot():
    with patch("ms_llama.setup_llm") as mock_llm:
        mock_llm.return_value = (MagicMock(), MagicMock(), "mlx")
        bot = VoiceChatBot(model_id="test-model")
    return bot


def test_bot_init_without_tts(mock_bot):
    assert mock_bot.tts is None
    assert mock_bot.history == []


def test_bot_speak_no_tts(mock_bot):
    # Should return immediately without error
    mock_bot.speak("Hello world")


def test_bot_history_trimming(mock_bot):
    mock_bot.backend = "gguf"
    mock_bot.llm_model = MagicMock()
    mock_bot.llm_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "reply"}}]
    }

    for i in range(MAX_HISTORY_TURNS + 5):
        mock_bot.chat(f"message {i}")

    assert len(mock_bot.history) <= MAX_HISTORY_TURNS * 2
    assert len(mock_bot.history) > 20


def test_token_budget_evicts_old_turns_into_same_session_continuity(
    mock_bot,
    monkeypatch,
):
    mock_bot.backend = "gguf"
    mock_bot.llm_tokenizer = None
    mock_bot.llm_model = MagicMock()
    mock_bot.llm_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "short reply"}}]
    }
    monkeypatch.setenv("YUKI_HISTORY_TOKEN_BUDGET", "1024")

    for i in range(12):
        mock_bot.chat(f"continuity-marker-{i} " + ("x" * 600))

    assert len(mock_bot.history) < 24
    earlier = [
        event for event in mock_bot._session_continuity
        if event.get("kind") == "earlier_exchange"
    ]
    assert earlier
    assert earlier[0]["user"].startswith("continuity-marker-0")


def test_current_session_continuity_is_injected_as_trusted_context(mock_bot):
    mock_bot.backend = "gguf"
    mock_bot.llm_tokenizer = None
    mock_bot.llm_model = MagicMock()
    mock_bot.llm_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "I remember that."}}]
    }
    mock_bot._remember_session_event({
        "kind": "verified_tool_outcome",
        "tool": "yuki_read",
        "succeeded": True,
        "result": "The housewarming note says the folder is home.",
    })

    mock_bot.chat("What did that note say?")

    messages = mock_bot.llm_model.create_chat_completion.call_args.kwargs["messages"]
    assert "CURRENT-SESSION CONTINUITY" in messages[0]["content"]
    assert "folder is home" in messages[0]["content"]


def test_session_context_can_move_with_model_and_reset_cleanly(mock_bot):
    mock_bot._remember_session_event({"kind": "marker", "value": "same chat"})
    with patch("ms_llama.setup_llm") as mock_llm:
        mock_llm.return_value = (MagicMock(), MagicMock(), "mlx")
        replacement = VoiceChatBot(model_id="replacement")

    replacement.copy_session_context_from(mock_bot)
    assert replacement._session_continuity == mock_bot._session_continuity

    replacement.reset_session_context()
    assert replacement.history == []
    assert replacement._session_continuity == []
    assert replacement.drain_session_events() == []


def test_autonomous_transcript_and_continuity_persist_without_embeddings(
    monkeypatch,
    tmp_path,
):
    import episodic

    db_path = tmp_path / "episodic-test.db"
    monkeypatch.setattr(episodic, "DB_PATH", db_path)
    embed = MagicMock(side_effect=AssertionError("must not embed autonomous turns"))
    monkeypatch.setattr(episodic, "_embed", embed)

    episodic.record_transcript(
        "session-1",
        "[AUTONOMOUS ACTION: read the note]",
        "The note is still here.",
    )
    episodic.record_session_event(
        "session-1",
        {"kind": "verified_tool_outcome", "tool": "yuki_read"},
    )

    import sqlite3

    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT user_msg, assistant_msg FROM episodes",
        ).fetchone()
    assert row == (
        "[AUTONOMOUS ACTION: read the note]",
        "The note is still here.",
    )
    assert episodic.read_session_events("session-1") == [{
        "kind": "verified_tool_outcome",
        "tool": "yuki_read",
    }]
    embed.assert_not_called()


def test_bot_input_truncation(mock_bot):
    mock_bot.backend = "gguf"
    mock_bot.llm_model = MagicMock()
    mock_bot.llm_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }

    long_input = "x" * (MAX_INPUT_CHARS + 500)
    mock_bot.chat(long_input)

    stored = mock_bot.history[0]["content"]
    assert len(stored) == MAX_INPUT_CHARS


def test_bot_chat_gguf_with_system_prompt():
    with patch("ms_llama.setup_llm") as mock_llm:
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_llm.return_value = (mock_model, None, "gguf")
        bot = VoiceChatBot(model_id="test", system_prompt="You are helpful.")

    response = bot.chat("Hi")
    assert response == "Hello!"

    # Verify system prompt was injected into first user message
    call_args = mock_model.create_chat_completion.call_args
    messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
    assert "You are helpful." in messages[0]["content"]


def test_bot_chat_gguf_fallback_response():
    with patch("ms_llama.setup_llm") as mock_llm:
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {"choices": []}
        mock_llm.return_value = (mock_model, None, "gguf")
        bot = VoiceChatBot(model_id="test")

    response = bot.chat("Hi")
    assert response == "Sorry, I could not generate a response."


# ============= TOOL_CALL_RE (ReAct regex) =============

def test_tool_call_re_quoted_arg():
    m = TOOL_CALL_RE.search('[TOOL: search("anime")]')
    assert m.group(1) == "search"
    assert m.group(2) == "anime"


def test_tool_call_re_unquoted_arg():
    m = TOOL_CALL_RE.search('[TOOL: weather(Tokyo)]')
    assert m.group(1) == "weather"
    assert m.group(2) == "Tokyo"


def test_tool_call_re_empty_parens():
    m = TOOL_CALL_RE.search('[TOOL: time("")]')
    assert m.group(1) == "time"
    assert m.group(2) == ""


def test_tool_call_re_no_parens():
    m = TOOL_CALL_RE.search('[TOOL: time]')
    assert m.group(1) == "time"
    assert m.group(2) is None


def test_tool_call_re_with_surrounding_text():
    text = 'Let me check! [TOOL: weather("Paris")] Hope it helps~'
    m = TOOL_CALL_RE.search(text)
    assert m.group(1) == "weather"
    assert m.group(2) == "Paris"


def test_tool_call_re_no_match():
    assert TOOL_CALL_RE.search("no tool here") is None
    assert TOOL_CALL_RE.search("[NOTOOL: foo]") is None


def test_tool_call_re_spaces():
    m = TOOL_CALL_RE.search('[TOOL:  calc( "2+2" )]')
    assert m.group(1) == "calc"
    assert m.group(2) == "2+2"


# ============= auto_detect_tool =============

def test_auto_detect_weather():
    cmd, arg = auto_detect_tool("what's the weather in Tokyo?")
    assert cmd == "/weather"
    assert arg == "Tokyo"


def test_auto_detect_time():
    cmd, arg = auto_detect_tool("what time is it?")
    assert cmd == "/time"
    assert arg == ""


def test_auto_detect_calc():
    cmd, arg = auto_detect_tool("calculate 2 + 2")
    assert cmd == "/calc"
    assert arg == "2 + 2"


def test_auto_detect_search():
    cmd, arg = auto_detect_tool("search for best anime 2024")
    assert cmd == "/search"
    assert arg == "best anime 2024"


def test_auto_detect_ram():
    cmd, arg = auto_detect_tool("how much ram is being used?")
    assert cmd == "/hardware"
    assert arg == "ram"


def test_auto_detect_top_processes():
    cmd, arg = auto_detect_tool("what program is using so much memory?")
    assert cmd == "/hardware"
    assert arg == "top"


def test_auto_detect_no_match():
    cmd, arg = auto_detect_tool("tell me a joke")
    assert cmd is None
    assert arg is None


# ============= tool_calc =============

def test_calc_basic():
    assert tool_calc("2 + 2") == "4"
    assert tool_calc("10 * 3") == "30"
    assert tool_calc("100 / 4") == "25.0"


def test_calc_rejects_letters():
    result = tool_calc("__import__('os')")
    assert "Invalid" in result


def test_calc_rejects_builtins():
    result = tool_calc("abs(-1)")
    assert "Invalid" in result


def test_calc_division_by_zero():
    result = tool_calc("1/0")
    assert "error" in result.lower()


# ============= tool_shell =============

def test_shell_allowed_command():
    result = tool_shell("echo hello")
    assert "hello" in result


def test_shell_blocked_command():
    result = tool_shell("rm -rf /")
    assert "not allowed" in result


def test_shell_empty():
    result = tool_shell("")
    assert "not allowed" in result


# ============= tool_time =============

def test_time_returns_string():
    result = tool_time()
    assert len(result) > 0
    # Should contain a day name
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert any(d in result for d in days)


# ============= tool_yuki_write / read / list =============

def test_yuki_write_and_read(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    result = tool_yuki_write("test.txt|hello world")
    assert "Wrote" in result
    assert (tmp_path / "test.txt").read_text() == "hello world"

    content = tool_yuki_read("test.txt")
    assert content == "hello world"


def test_yuki_write_missing_pipe(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    result = tool_yuki_write("just-a-filename")
    assert "Error" in result


def test_yuki_write_strips_slashes(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    tool_yuki_write("../evil|pwn")
    # Slashes and .. are stripped, so filename becomes "..evil"
    assert not (tmp_path.parent / "evil").exists()


def test_yuki_write_empty_filename(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    result = tool_yuki_write("/|content")
    assert "Error" in result


def test_yuki_read_missing(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    result = tool_yuki_read("nope.txt")
    assert "not found" in result.lower()


def test_yuki_list_empty(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    result = tool_yuki_list()
    assert "empty" in result.lower()


def test_yuki_list_with_files(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    result = tool_yuki_list()
    assert "a.txt" in result
    assert "b.txt" in result


# ============= run_tool (slash commands + auto-detect) =============

def test_run_tool_explicit_time():
    result, name, links = run_tool("/time")
    assert name == "/time"
    assert len(result) > 0


def test_run_tool_explicit_calc():
    result, name, links = run_tool("/calc 3+3")
    assert name == "/calc"
    assert result == "6"


def test_run_tool_help():
    result, name, links = run_tool("/help")
    assert name == "/help"
    assert "Available commands" in result


def test_run_tool_unknown_slash():
    result, name, links = run_tool("/nonexistent blah")
    assert result is None
    assert name is None


def test_run_tool_auto_detect():
    result, name, links = run_tool("what time is it?")
    assert name == "/time"
    assert len(result) > 0


def test_run_tool_no_match():
    result, name, links = run_tool("tell me a story")
    assert result is None
    assert name is None


# ============= _tool_result_injection =============

def test_tool_result_injection_format():
    msg = _tool_result_injection("weather", "Tokyo: Sunny 25°C")
    assert "[TOOL_RESULT: weather]" in msg
    assert "Tokyo: Sunny 25°C" in msg
    assert "RULES" in msg
    assert "no guessing" in msg.lower()


def test_completed_tool_response_uses_result_without_executing_again(monkeypatch):
    bot = VoiceChatBot.__new__(VoiceChatBot)
    bot.history = [{"role": "assistant", "content": "Earlier context."}]
    bot.last_reasoning = "stale trace"
    bot._last_user_input = ""
    bot._suspend_history_trim = 0
    bot._trim_history = lambda: None
    bot._run_tool_with_spinner = MagicMock(
        side_effect=AssertionError("completed-tool response must not execute tools"),
    )
    seen_messages = []

    def fake_chat(user_input, max_tokens=None):
        assert user_input == ""
        assert max_tokens == 512
        seen_messages.extend(dict(message) for message in bot.history)
        bot.history.append({
            "role": "assistant",
            "content": "I found it—the lighthouse code is 7319~",
        })
        return "I found it—the lighthouse code is 7319~"

    bot.chat = fake_chat
    monkeypatch.setattr("ms_llama._drive_eye_emotion", lambda *_args: None)

    response = bot.respond_to_completed_tool(
        "/recall cobalt lighthouse",
        "/recall",
        "[SOURCE 1]\nThe cobalt lighthouse code is 7319.\n[/SOURCE 1]",
        max_tokens=512,
    )

    assert response == "I found it—the lighthouse code is 7319~"
    assert bot.last_reasoning == ""
    assert bot._last_user_input == "/recall cobalt lighthouse"
    assert bot._run_tool_with_spinner.call_count == 0
    assert any(
        "[TOOL_RESULT: recall]" in str(message.get("content", ""))
        and "retrieved memory candidates" in str(message.get("content", ""))
        for message in seen_messages
    )
    assert bot.history == [
        {"role": "assistant", "content": "Earlier context."},
        {"role": "user", "content": "/recall cobalt lighthouse"},
        {
            "role": "assistant",
            "content": "I found it—the lighthouse code is 7319~",
        },
    ]


# ============= Stage cue regex — broader =============

def test_stage_cue_multiple_consecutive():
    assert _STAGE_CUE_RE.sub("", "[laughs][sighs][chuckles] hi") == " hi"


def test_stage_cue_with_inner_whitespace():
    assert _STAGE_CUE_RE.sub("", "[long   pause] done") == " done"


def test_stage_cue_unclosed_ignored():
    # No closing bracket / paren → not a cue, leave it alone.
    assert _STAGE_CUE_RE.sub("", "[unclosed hello") == "[unclosed hello"
    assert _STAGE_CUE_RE.sub("", "(also unclosed") == "(also unclosed"


def test_stage_cue_mixed_parens_and_brackets():
    assert _STAGE_CUE_RE.sub("", "[a] text (b) more [c]").strip() == "text  more"


# ============= auto_detect_tool — broader =============

def test_auto_detect_weather_london():
    cmd, arg = auto_detect_tool("what's the weather in London")
    assert cmd == "/weather"
    assert "London" in arg or "london" in arg.lower()


def test_auto_detect_case_insensitive():
    cmd, arg = auto_detect_tool("WHAT TIME IS IT")
    assert cmd == "/time"


# ============= tool_calc — broader =============

def test_calc_float_math():
    assert tool_calc("1.5 + 2.5") == "4.0"


def test_calc_parentheses():
    assert tool_calc("(2 + 3) * 4") == "20"


def test_calc_negative_numbers():
    assert tool_calc("-5 + 3") == "-2"


def test_calc_power_operator():
    assert tool_calc("2 ** 10") == "1024"


# ============= tool_shell — broader =============

def test_shell_allowed_pwd():
    # pwd is in ALLOWED_COMMANDS by default; if not, skip
    if "pwd" in ALLOWED_COMMANDS:
        result = tool_shell("pwd")
        assert "/" in result


def test_shell_blocks_shell_injection_via_pipe():
    with patch("tools.shell.shell.subprocess.run") as runner:
        result = tool_shell("echo hi | cat")
    assert "not allowed" in result
    runner.assert_not_called()


# ============= tool_yuki — broader =============

def test_yuki_write_overwrites(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    tool_yuki_write("a.txt|first")
    tool_yuki_write("a.txt|second")
    assert (tmp_path / "a.txt").read_text() == "second"


def test_yuki_read_binary_safe(tmp_path, monkeypatch):
    _patch_yuki_dir(monkeypatch, tmp_path)
    (tmp_path / "b.bin").write_bytes(b"\xff\xfe\xfd")
    # Should not crash — either returns a lossy string or an error message
    result = tool_yuki_read("b.bin")
    assert isinstance(result, str)


# ============= _normalize_value =============

def test_normalize_value_lowercases():
    assert _normalize_value("Superman") == "superman"


def test_normalize_value_collapses_whitespace():
    assert _normalize_value("  black   cats  ") == "black cats"


def test_normalize_value_empty():
    assert _normalize_value("") == ""
    assert _normalize_value(None) == ""


def test_normalize_value_preserves_unicode():
    assert _normalize_value("Pokémon") == "pokémon"
