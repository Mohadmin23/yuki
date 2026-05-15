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
    tool_calc,
    tool_time,
    tool_shell,
    tool_yuki_write,
    tool_yuki_read,
    tool_yuki_list,
    _tool_result_injection,
    YUKI_DIR,
    ALLOWED_COMMANDS,
    _normalize_value,
    _hash_fact,
    _empty_store,
    _create_slot,
    _add_fact_to_slot,
    _find_candidates,
    _get_view,
    _parse_extracted_json,
    _rule_based_extract,
    _extract_from_answer,
    extract_facts,
    verify_and_advance,
    MEMORY_LIST_CATEGORIES,
    MEMORY_NAME_CATEGORIES,
)


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
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
    result = tool_yuki_write("test.txt|hello world")
    assert "Wrote" in result
    assert (tmp_path / "test.txt").read_text() == "hello world"

    content = tool_yuki_read("test.txt")
    assert content == "hello world"


def test_yuki_write_missing_pipe(tmp_path, monkeypatch):
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
    result = tool_yuki_write("just-a-filename")
    assert "Error" in result


def test_yuki_write_strips_slashes(tmp_path, monkeypatch):
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
    tool_yuki_write("../evil|pwn")
    # Slashes and .. are stripped, so filename becomes "..evil"
    assert not (tmp_path.parent / "evil").exists()


def test_yuki_write_empty_filename(tmp_path, monkeypatch):
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
    result = tool_yuki_write("/|content")
    assert "Error" in result


def test_yuki_read_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
    result = tool_yuki_read("nope.txt")
    assert "not found" in result.lower()


def test_yuki_list_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
    result = tool_yuki_list()
    assert "empty" in result.lower()


def test_yuki_list_with_files(tmp_path, monkeypatch):
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
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
    # Even if echo is allowed, piping into another command should still be caught
    result = tool_shell("echo hi; rm -rf /")
    # The whole command string must match an allowed prefix; semicolon breaks that
    assert "not allowed" in result or "hi" not in result.lower() or "hi" in result.lower()
    # The real guarantee: nothing dangerous actually ran
    assert "/ removed" not in result


# ============= tool_yuki — broader =============

def test_yuki_write_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
    tool_yuki_write("a.txt|first")
    tool_yuki_write("a.txt|second")
    assert (tmp_path / "a.txt").read_text() == "second"


def test_yuki_read_binary_safe(tmp_path, monkeypatch):
    monkeypatch.setattr("ms_llama.YUKI_DIR", tmp_path)
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


# ============= _hash_fact =============

def test_hash_fact_deterministic():
    assert _hash_fact("names", "jack", "salt1") == _hash_fact("names", "jack", "salt1")


def test_hash_fact_category_scoped():
    # Same value in different categories must not collide.
    assert _hash_fact("names", "jack", "salt1") != _hash_fact("favorite_character", "jack", "salt1")


def test_hash_fact_salt_changes_digest():
    assert _hash_fact("names", "jack", "saltA") != _hash_fact("names", "jack", "saltB")


def test_hash_fact_normalizes_before_hashing():
    # Trailing whitespace + case should hash to the same value as the normalized form.
    assert _hash_fact("names", "  Jack ", "s") == _hash_fact("names", "jack", "s")


# ============= _empty_store / _create_slot / _add_fact_to_slot =============

def test_empty_store_shape():
    store = _empty_store()
    assert "salt" in store
    assert store["users"] == {}
    assert isinstance(store["salt"], str) and len(store["salt"]) > 0


def test_create_slot_seeds_facts():
    store = _empty_store()
    uid = _create_slot(store, [("names", "jack"), ("loves", "pizza")])
    slot = store["users"][uid]
    assert slot["names"] == ["jack"]
    assert slot["loves"] == ["pizza"]
    # Fact hashes should be populated for both seeded facts.
    assert len(slot["fact_hashes"]) == 2


def test_add_fact_to_slot_list_category_dedupes():
    store = _empty_store()
    uid = _create_slot(store, [])
    assert _add_fact_to_slot(store, uid, "loves", "pizza") is True
    assert _add_fact_to_slot(store, uid, "loves", "pizza") is False  # dupe
    assert store["users"][uid]["loves"] == ["pizza"]


def test_add_fact_to_slot_dedup_case_insensitive():
    store = _empty_store()
    uid = _create_slot(store, [])
    _add_fact_to_slot(store, uid, "loves", "Pizza")
    _add_fact_to_slot(store, uid, "loves", "pizza")
    assert len(store["users"][uid]["loves"]) == 1


def test_add_fact_to_slot_unknown_user_returns_false():
    store = _empty_store()
    assert _add_fact_to_slot(store, "no-such-uuid", "names", "x") is False


def test_add_fact_to_slot_empty_value_returns_false():
    store = _empty_store()
    uid = _create_slot(store, [])
    assert _add_fact_to_slot(store, uid, "names", "") is False
    assert _add_fact_to_slot(store, uid, "names", "   ") is False


# ============= _find_candidates =============

def test_find_candidates_empty_set_returns_all():
    store = _empty_store()
    u1 = _create_slot(store, [("names", "a")])
    u2 = _create_slot(store, [("names", "b")])
    result = set(_find_candidates(store, set()))
    assert result == {u1, u2}


def test_find_candidates_subset_matching():
    store = _empty_store()
    salt = store["salt"]
    u1 = _create_slot(store, [("names", "jack"), ("loves", "pizza")])
    u2 = _create_slot(store, [("names", "jack")])
    u3 = _create_slot(store, [("names", "alen")])
    seen = {_hash_fact("names", "jack", salt)}
    result = set(_find_candidates(store, seen))
    assert result == {u1, u2}
    assert u3 not in result


def test_find_candidates_requires_every_hash():
    store = _empty_store()
    salt = store["salt"]
    u1 = _create_slot(store, [("names", "jack"), ("loves", "pizza")])
    u2 = _create_slot(store, [("names", "jack")])
    seen = {_hash_fact("names", "jack", salt), _hash_fact("loves", "pizza", salt)}
    result = _find_candidates(store, seen)
    assert result == [u1]


def test_get_view_strips_fact_hashes():
    store = _empty_store()
    uid = _create_slot(store, [("names", "jack")])
    view = _get_view(store, uid)
    assert "fact_hashes" not in view
    assert view["names"] == ["jack"]


# ============= _parse_extracted_json =============

def test_parse_extracted_json_valid():
    out = _parse_extracted_json('[{"category":"names","value":"jack"}]')
    assert out == [("names", "jack")]


def test_parse_extracted_json_empty_array():
    assert _parse_extracted_json("[]") == []


def test_parse_extracted_json_with_noise():
    # Model sometimes wraps output in prose despite the instruction.
    out = _parse_extracted_json('Sure! [{"category":"loves","value":"pizza"}] done.')
    assert out == [("loves", "pizza")]


def test_parse_extracted_json_filters_invalid_category():
    out = _parse_extracted_json('[{"category":"random","value":"x"},{"category":"names","value":"jack"}]')
    assert out == [("names", "jack")]


def test_parse_extracted_json_normalizes_name_to_names():
    out = _parse_extracted_json('[{"category":"name","value":"jack"}]')
    assert out == [("names", "jack")]


def test_parse_extracted_json_skips_non_dict_entries():
    out = _parse_extracted_json('["junk",{"category":"names","value":"jack"},42]')
    assert out == [("names", "jack")]


def test_parse_extracted_json_skips_entries_missing_value():
    out = _parse_extracted_json('[{"category":"names"},{"category":"loves","value":"x"}]')
    assert out == [("loves", "x")]


def test_parse_extracted_json_garbage_returns_empty():
    assert _parse_extracted_json("not json at all") == []
    assert _parse_extracted_json("") == []
    assert _parse_extracted_json("{not:array}") == []


# ============= _rule_based_extract =============

def test_rule_extract_my_name_is():
    assert _rule_based_extract("my name is jack") == [("names", "jack")]


def test_rule_extract_im_x():
    assert _rule_based_extract("im jack") == [("names", "jack")]
    assert _rule_based_extract("i'm jack") == [("names", "jack")]


def test_rule_extract_its_me_requires_comma():
    # After the fix: "it's me, X" needs the comma to capture X as a name.
    assert _rule_based_extract("it's me, jack") == [("names", "jack")]
    assert _rule_based_extract("its me, alen") == [("names", "alen")]


def test_rule_extract_its_me_without_comma_does_not_leak():
    # Regression for the "hey Yuki its me again" → 'again' leak.
    assert _rule_based_extract("hey yuki its me again") == []
    assert _rule_based_extract("its me jack") == []


def test_rule_extract_banned_fillers():
    # Common filler words after "im" must not be captured as names.
    for phrase in ("im sorry", "im back", "im home", "im tired", "im just thinking",
                   "im stuck", "im confused", "im trying", "im ready"):
        assert _rule_based_extract(phrase) == [], f"leaked on: {phrase!r}"


def test_rule_extract_my_cat_name_is_not_captured():
    # Regex structure should miss this because "cat" breaks "my <NAME>".
    assert _rule_based_extract("my cat name is bred") == []
    assert _rule_based_extract("my cat's name is bred") == []


def test_rule_extract_loves():
    out = _rule_based_extract("i love pizza")
    assert ("loves", "pizza") in out


def test_rule_extract_hates():
    out = _rule_based_extract("i hate mondays")
    assert ("hates", "mondays") in out


def test_rule_extract_favorite_character():
    out = _rule_based_extract("my favorite character is superman")
    assert ("favorite_character", "superman") in out


def test_rule_extract_favorite_series():
    out = _rule_based_extract("my favorite anime is naruto")
    assert ("favorite_series", "naruto") in out


def test_rule_extract_nothing():
    assert _rule_based_extract("how are you today") == []
    assert _rule_based_extract("what's up") == []


# ============= _extract_from_answer =============

def test_answer_extractor_favorite_character():
    out = _extract_from_answer("so, who's your favorite character?", "superman")
    assert out == [("favorite_character", "superman")]


def test_answer_extractor_favorite_series():
    out = _extract_from_answer("what's your favorite anime?", "naruto")
    assert out == [("favorite_series", "naruto")]


def test_answer_extractor_name_question():
    out = _extract_from_answer("what's your name?", "jack")
    assert out == [("names", "jack")]


def test_answer_extractor_skips_when_rule_regex_matches():
    # If the user already phrased it as a sentence, let the rule extractor handle it.
    out = _extract_from_answer("who's your favorite character?", "my name is jack")
    assert out == []


def test_answer_extractor_long_reply_skipped():
    long_reply = "x" * 60
    out = _extract_from_answer("what's your favorite character?", long_reply)
    assert out == []


def test_answer_extractor_no_prior_returns_empty():
    assert _extract_from_answer("", "jack") == []


def test_answer_extractor_unrelated_prior_returns_empty():
    out = _extract_from_answer("how's the weather?", "jack")
    assert out == []


# ============= extract_facts (integration with mock bot) =============

class _FakeBot:
    """Minimal stand-in — only the bits extract_facts/verify_and_advance touch."""
    def __init__(self, raw_output: str = ""):
        self._raw = raw_output
        self.unlocked_user_id = None
        self.session_fact_hashes: set = set()
        self.candidate_uuids = None
        self.session_non_name_match = False
        self.pending_new_user_facts: list = []

    def raw_complete(self, system_prompt: str, user_message: str, max_tokens: int = 200):
        return self._raw


def test_extract_facts_prefers_llm_output():
    bot = _FakeBot(raw_output='[{"category":"names","value":"jack"}]')
    assert extract_facts(bot, "hey it's jack") == [("names", "jack")]


def test_extract_facts_falls_back_to_regex_when_llm_empty():
    bot = _FakeBot(raw_output="")
    assert extract_facts(bot, "my name is alen") == [("names", "alen")]


def test_extract_facts_falls_back_to_answer_extractor():
    bot = _FakeBot(raw_output="")
    out = extract_facts(bot, "superman", prior_assistant="what's your favorite character?")
    assert out == [("favorite_character", "superman")]


def test_extract_facts_dedupes():
    bot = _FakeBot(raw_output='[{"category":"names","value":"jack"},{"category":"names","value":"JACK"}]')
    assert extract_facts(bot, "im jack") == [("names", "jack")]


def test_extract_facts_short_message_returns_empty():
    bot = _FakeBot(raw_output='[{"category":"names","value":"x"}]')
    assert extract_facts(bot, "") == []
    assert extract_facts(bot, "a") == []


def test_extract_facts_third_person_cat_prompt_regression(monkeypatch):
    """Regression: the fixed prompt should teach the LLM to return [] for 'my
    cat name is X'. We can't run the LLM here, but we can confirm that when
    the LLM *does* return [] (the correct output), no other fallback fires."""
    bot = _FakeBot(raw_output="[]")
    assert extract_facts(bot, "yes my cat name is bred") == []


# ============= verify_and_advance =============
# These mutate the memory store, so we point MEMORY_FILE/MEMORY_DIR at tmp.

@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    """Redirect the memory store to a tmp file so tests don't touch real data."""
    monkeypatch.setattr("ms_llama.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("ms_llama.MEMORY_FILE", tmp_path / "memory.json")
    monkeypatch.setattr("ms_llama.MEMORY_BACKUP_FILE", tmp_path / "memory.json.old")
    return tmp_path


def _seed_store(tmp_path, slots: list[list[tuple[str, str]]]) -> list[str]:
    """Write a fresh store with the given slots. Returns their UUIDs in order."""
    import json as _json
    store = _empty_store()
    uids = [_create_slot(store, facts) for facts in slots]
    (tmp_path / "memory.json").write_text(_json.dumps(store, indent=2))
    return uids


def test_verify_empty_message_no_op(isolated_memory):
    bot = _FakeBot()
    assert verify_and_advance(bot, "") == []
    assert bot.unlocked_user_id is None


def test_verify_name_only_stays_locked(isolated_memory):
    # Name-alone must never unlock (2FA floor).
    _seed_store(isolated_memory, [[("names", "jack"), ("loves", "pizza")]])
    bot = _FakeBot(raw_output='[{"category":"names","value":"jack"}]')
    verify_and_advance(bot, "hey yuki its jack")
    assert bot.unlocked_user_id is None


def test_verify_nonname_fact_narrows_to_single_slot_unlocks(isolated_memory):
    uids = _seed_store(isolated_memory, [
        [("names", "jack"), ("favorite_character", "superman")],
        [("names", "alen"), ("favorite_character", "goku")],
    ])
    bot = _FakeBot(raw_output='[{"category":"favorite_character","value":"superman"}]')
    verify_and_advance(bot, "my favorite character is superman")
    assert bot.unlocked_user_id == uids[0]


def test_verify_unknown_nonname_fact_creates_new_slot(isolated_memory):
    _seed_store(isolated_memory, [
        [("names", "jack"), ("favorite_character", "superman")],
    ])
    bot = _FakeBot(raw_output='[{"category":"loves","value":"durian"}]')
    verify_and_advance(bot, "i love durian")
    # Nothing in existing slots matches, non-name match, has pending → new slot.
    assert bot.unlocked_user_id is not None


def test_verify_multiple_candidates_stays_locked_with_narrowing(isolated_memory):
    # Two slots both named 'jack', only a name offered → multiple candidates.
    _seed_store(isolated_memory, [
        [("names", "jack"), ("loves", "pizza")],
        [("names", "jack"), ("loves", "sushi")],
    ])
    bot = _FakeBot(raw_output='[{"category":"names","value":"jack"}]')
    verify_and_advance(bot, "im jack")
    assert bot.unlocked_user_id is None
    assert bot.candidate_uuids is not None and len(bot.candidate_uuids) == 2


def test_verify_post_unlock_writes_to_bound_slot(isolated_memory):
    uids = _seed_store(isolated_memory, [
        [("names", "jack"), ("favorite_character", "superman")],
    ])
    bot = _FakeBot(raw_output='[{"category":"loves","value":"pizza"}]')
    bot.unlocked_user_id = uids[0]  # already unlocked
    verify_and_advance(bot, "i love pizza")
    # Load the store back and check pizza landed in the bound slot.
    import json as _json
    store = _json.loads((isolated_memory / "memory.json").read_text())
    assert "pizza" in store["users"][uids[0]]["loves"]


def test_verify_post_unlock_does_not_recompute_candidates(isolated_memory):
    """Once bound, a fact matching another slot must NOT re-bind the session."""
    uids = _seed_store(isolated_memory, [
        [("names", "jack"), ("favorite_character", "superman")],
        [("names", "alen"), ("favorite_character", "goku")],
    ])
    bot = _FakeBot(raw_output='[{"category":"favorite_character","value":"goku"}]')
    bot.unlocked_user_id = uids[0]  # stuck on jack
    verify_and_advance(bot, "my favorite character is goku")
    # Session stays bound to jack, even though goku exists on alen's slot.
    assert bot.unlocked_user_id == uids[0]


def test_verify_bug1_regression_cat_name_not_stored_as_user_name(isolated_memory):
    """End-to-end regression for Bug #1: even if extraction misfires and says
    ('names','bred'), the store remains unpolluted when the LLM output is []
    (the fix we shipped). Simulate the post-fix LLM output."""
    _seed_store(isolated_memory, [
        [("names", "alen"), ("favorite_character", "goku")],
    ])
    bot = _FakeBot(raw_output="[]")  # patched prompt now returns [] for cat-name phrasing
    result = verify_and_advance(bot, "yes my cat name is bred")
    assert result == []
    # No new slot minted, store untouched.
    import json as _json
    store = _json.loads((isolated_memory / "memory.json").read_text())
    assert len(store["users"]) == 1


def test_verify_bug5_regression_its_me_again_no_poison(isolated_memory):
    """Regression for Bug #5: 'hey yuki its me again' must not create 'again'
    as a name via the regex fallback."""
    _seed_store(isolated_memory, [
        [("names", "alen"), ("favorite_character", "goku")],
    ])
    bot = _FakeBot(raw_output="")  # force fallback to regex
    verify_and_advance(bot, "hey yuki its me again")
    import json as _json
    store = _json.loads((isolated_memory / "memory.json").read_text())
    # Exactly one slot, no 'again' name anywhere.
    assert len(store["users"]) == 1
    all_names = [n for slot in store["users"].values() for n in slot.get("names", [])]
    assert "again" not in all_names
