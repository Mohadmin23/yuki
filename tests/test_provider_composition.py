"""Provider pinning and explicitly authorized file authoring."""
import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ms_llama
import openrouter_providers as providers
from interface import tui
from tool_routing import RoutingRegistry
from tool_routing.composition import parse_file_content_intent

REQUEST = "create a file called yuki_paycheck.txt and write the amount of money you want to have"


def intent(mode="compose", filename="yuki_paycheck.txt", content="$1,000,000"):
    return {"mode": mode, "tool": "yuki_write", "filename": filename, "content": content}


@pytest.mark.parametrize("filename,content,prompt,passed", [
    ("yuki_paycheck.txt", "$1,000,000", REQUEST, True),
    ("invented.txt", "$1,000,000", REQUEST, False),
    ("yuki_paycheck.txt", "$2,000,000", REQUEST, False),
    ("yuki_paycheck.txt", "$1,000,000", "Write hello in yuki_paycheck.txt", False),
])
def test_draft_authorizes_only_bound_content(filename, content, prompt, passed):
    registry = RoutingRegistry()
    result = registry.prepare_call(
        {"tool": "yuki_write", "arguments": {"filename": filename, "content": content}},
        raw_request=prompt, available_names=registry.names,
        literal_sources=[{**intent(), "request": REQUEST, "source_kind": "file_content_intent"}],
    )
    assert result["passed"] is passed


def test_provider_pin_preserves_reasoning_and_original_body():
    body = {"reasoning": {"effort": "low"}, "provider": {"sort": "price"}}
    merged = providers.provider_request_body(body, "example/us")
    assert merged["provider"]["only"] == ["example/us"]
    assert merged["provider"]["allow_fallbacks"] is False
    assert merged["reasoning"] == body["reasoning"]
    assert body["provider"] == {"sort": "price"}
    assert providers.provider_request_body(None, "") is None


def test_provider_pin_reaches_completion():
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="reply"))])
    client = SimpleNamespace(_yuki_openrouter_provider="example/us", chat=SimpleNamespace(
        completions=SimpleNamespace(create=MagicMock(return_value=reply)),
    ))
    assert ms_llama._openrouter_chat_completion(client, model="author/model", messages=[]) is reply
    assert client.chat.completions.create.call_args.kwargs["extra_body"]["provider"]["only"] == ["example/us"]


def test_endpoint_tags_and_prices(monkeypatch):
    payload = {"data": {"endpoints": [{"tag": "example/us", "provider_name": "Example", "context_length": 128000,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "supported_parameters": ["tools"]}]}}
    fetch = MagicMock(return_value=io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(providers, "urlopen", fetch)
    rows = providers.fetch_model_providers("openrouter/author/model")
    assert rows[0]["id"] == "example/us"
    assert "$1 in / $2 out" in providers.provider_label(rows[0])
    assert fetch.call_args.args[0].full_url.endswith("/author/model/endpoints")


def test_provider_picker_and_same_model_selection(monkeypatch):
    from test_tui import StubBot
    monkeypatch.setattr(tui, "read_recent_sessions", list)
    monkeypatch.setattr(tui, "load_settings", dict)
    saved = []
    monkeypatch.setattr(tui, "save_settings", lambda x: saved.append(dict(x)))
    row = {"id": "example/us", "name": "Example", "pricing": {}, "supported_parameters": []}
    monkeypatch.setattr(tui, "fetch_model_providers", lambda _: [row])
    async def scenario():
        bot = StubBot()
        bot.model_id = "openrouter/author/model"
        bot.llm_model = SimpleNamespace()
        app = tui.YukiTUI(bot, model_label="model", persona_text="stub", autonomy_interval=0)
        async with app.run_test(size=(120, 38)) as pilot:
            manager = tui.ModelManagerScreen({})
            app.push_screen(manager, app._model_manager_result)
            await pilot.pause()
            manager._finish(bot.model_id)
            for _ in range(20):
                await pilot.pause()
                if isinstance(app.screen, tui.ChoiceScreen):
                    break
            assert isinstance(app.screen, tui.ChoiceScreen)
            choices = app.screen.query_one("#choice-list", tui.ListView)
            assert [item.value for item in app.screen.query(tui.PickItem)] == ["", "example/us"]
            choices.index = 1
            choices.action_select_cursor()
            await pilot.pause()
            assert bot.llm_model._yuki_openrouter_provider == "example/us"
            assert saved[-1]["openrouter_providers"][bot.model_id] == "example/us"
            app._model_manager_result({"id": bot.model_id, "provider": ""})
            assert bot.llm_model._yuki_openrouter_provider == ""
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["direct", "dispatcher"])
def test_creative_write_executes_through_shared_gate(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(ms_llama, "setup_llm", lambda *_: (MagicMock(), None, "gguf"))
    bot = ms_llama.VoiceChatBot("test-model", system_prompt="Yuki")
    bot.configure_tool_routing(mode=mode, protocol="canonical")
    draft = "$1,000,000"
    call = {"tool": "yuki_write", "arguments": {"filename": "yuki_paycheck.txt", "content": draft}}
    def generate(_system, _user, _schema, **kwargs):
        if kwargs["schema_name"] == "yuki_file_content_intent":
            return json.dumps(intent(content=draft))
        if kwargs["schema_name"] == "yuki_semantic_delegation":
            return json.dumps({"action": "delegate", "request": "Write the authorized draft to yuki_paycheck.txt", "domain_hint": "yuki_files"})
        return json.dumps(call)
    monkeypatch.setattr(bot, "raw_structured_complete", generate)
    if mode == "dispatcher":
        def dispatch(**kwargs):
            prepared = bot._routing_registry.prepare_call(call, raw_request=kwargs["raw_request"],
                available_names=bot._routing_registry.names, literal_sources=kwargs.get("literal_sources", []))
            return {"passed": prepared["passed"], "prepared": prepared, "canonical_call": call}
        bot._dispatcher_client = SimpleNamespace(route=dispatch)
    def write(name, argument, **_kwargs):
        assert name == "yuki_write"
        filename, content = argument.split("|", 1)
        (tmp_path / filename).write_text(content)
        return "Wrote the file", False
    monkeypatch.setattr(bot, "_run_tool_with_spinner", write)
    monkeypatch.setattr(bot, "chat", lambda *_args, **_kwargs: "Done~")
    bot.react_chat(REQUEST)
    assert (tmp_path / "yuki_paycheck.txt").read_text() == draft
    assert bot._file_content_intent is None


@pytest.mark.parametrize("choices", [None, [], [SimpleNamespace(message=None)]])
def test_empty_provider_reply_reports_failure_without_retry(choices):
    response = SimpleNamespace(choices=choices, error={"message": "Upstream unavailable"})
    create = MagicMock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match="Upstream unavailable"):
        ms_llama._openrouter_chat_completion(client, model="author/model", messages=[])
    create.assert_called_once()


def test_exact_wish_request_composes_and_writes_via_native_route(monkeypatch, tmp_path):
    prompt = "hey yuki create a yuki_money.txt file and write any amount you wish you had"
    monkeypatch.setattr(ms_llama, "setup_llm", lambda *_: (MagicMock(), None, "openrouter"))
    bot = ms_llama.VoiceChatBot("openrouter/test/model", system_prompt="Yuki")
    bot.configure_tool_routing(mode="direct", protocol="native")
    monkeypatch.setattr(bot, "raw_structured_complete", lambda *_args, **_kwargs: json.dumps(intent(filename="yuki_money.txt")))
    replies = iter([
        {"content": "", "tool_calls": [{"id": "write1", "name": "yuki_write", "arg_raw": '{"filename":"yuki_money.txt","content":"$1,000,000"}'}]},
        {"content": "Done", "tool_calls": []},
    ])
    monkeypatch.setattr(bot, "_chat_with_tools", lambda *_: next(replies))
    def write(name, argument):
        assert name == "yuki_write"
        filename, content = argument.split("|", 1)
        (tmp_path / filename).write_text(content)
        return "Wrote file", False
    monkeypatch.setattr(bot, "_run_tool_with_spinner", write)
    bot.react_chat(prompt)
    assert (tmp_path / "yuki_money.txt").read_text() == "$1,000,000"


def test_wish_request_with_empty_provider_reply_never_executes(monkeypatch):
    client = SimpleNamespace(_or_model="test/model", chat=SimpleNamespace(completions=SimpleNamespace(
        create=MagicMock(return_value=SimpleNamespace(choices=None, error={"message": "Upstream unavailable"})),
    )))
    monkeypatch.setattr(ms_llama, "setup_llm", lambda *_: (client, None, "openrouter"))
    bot = ms_llama.VoiceChatBot("openrouter/test/model", system_prompt="Yuki")
    execute = MagicMock()
    monkeypatch.setattr(bot, "_run_tool_with_spinner", execute)
    reply = bot.react_chat("create yuki_money.txt and write any amount you wish you had")
    assert "Upstream unavailable" in reply
    execute.assert_not_called()
    assert bot._file_content_intent is None


@pytest.mark.parametrize("prompt,decision,expected", [
    ("Surprise me with something comforting in note.txt", intent(filename="note.txt", content="You can rest now."), "You can rest now."),
    ("mets un petit poème dans note.txt", intent(filename="note.txt", content="La lune sourit."), "La lune sourit."),
    ("note.txt could use a tiny bedtime story; make one up", intent(filename="note.txt", content="Once a star fell asleep."), "Once a star fell asleep."),
    ('Do not compose. Put "Keep THIS!" into note.txt unchanged.', intent("literal", "note.txt", "Keep THIS!"), "Keep THIS!"),
    ("Don't write a file; just discuss note.txt", {"mode": "no_write", "tool": "none", "filename": "", "content": ""}, None),
    ('Copy "invent anything" into note.txt', intent("literal", "note.txt", "rewritten"), None),
])
def test_structured_intent_controls_write_without_phrase_gate(monkeypatch, prompt, decision, expected):
    monkeypatch.setattr(ms_llama, "setup_llm", lambda *_: (MagicMock(), None, "gguf"))
    bot = ms_llama.VoiceChatBot("test", system_prompt="Yuki")
    complete = MagicMock(return_value=json.dumps(decision))
    monkeypatch.setattr(bot, "raw_structured_complete", complete)
    call = {"tool": "yuki_write", "arguments": {"filename": "note.txt", "content": "untrusted router draft"}}
    prepared = bot._prepare_human_call(call, prompt, [])
    assert complete.call_args.kwargs["schema_name"] == "yuki_file_content_intent"
    assert prompt in complete.call_args.args[1]
    if expected is None:
        assert not prepared["passed"]
    else:
        assert prepared["passed"]
        assert prepared["runtime_argument"] == "note.txt|" + expected
    # No request is rewritten to manufacture literal provenance.
    assert bot._file_content_intent["request"] == prompt


@pytest.mark.parametrize("raw", ['null', '{}', '{"mode":"compose"}', '{"mode":"no_write","tool":"yuki_write","filename":"x","content":"x"}'])
def test_invalid_intent_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_file_content_intent(raw)


def test_failed_intent_blocks_even_literal_proposed_write(monkeypatch):
    monkeypatch.setattr(ms_llama, "setup_llm", lambda *_: (MagicMock(), None, "gguf"))
    bot = ms_llama.VoiceChatBot("test", system_prompt="Yuki")
    monkeypatch.setattr(bot, "raw_structured_complete", lambda *_args, **_kwargs: "invalid")
    result = bot._prepare_human_call({"tool": "yuki_write", "arguments": {"filename": "note.txt", "content": "hello"}}, "write hello in note.txt", [])
    assert not result["passed"]
