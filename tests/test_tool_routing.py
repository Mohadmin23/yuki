"""Tool-routing checks with no model loads and no Yuki tool execution."""

import json
from unittest.mock import MagicMock, patch

from ms_llama import VoiceChatBot
from prototypes.tool_dispatcher.backends import FixtureBackend
from tool_routing import DedicatedDispatcherClient, RoutingRegistry
from tool_routing.core import (
    build_main_brain_decision_prompt,
    normalize_openai_tool_call,
    parse_autonomous_decision,
    parse_main_brain_decision,
    referenced_literal_sources,
)


def test_registry_remains_live_and_repairs_dispatcher_facing_boundaries():
    registry = RoutingRegistry()

    assert len(registry.names) == 18
    assert "operating-system filesystem" in registry.description_for("read")
    assert "managed personal file store" in registry.description_for("yuki_read")
    assert "does not generate" in registry.description_for("see")
    assert "persistent memory" in registry.description_for("recall")
    assert "busiest processes" in registry.description_for("hardware")

    write_schema = registry.argument_schema_for("yuki_write")
    assert write_schema["required"] == ["filename", "content"]
    assert write_schema["additionalProperties"] is False
    assert "filename_and_content" not in write_schema["properties"]


def test_conservative_binder_accepts_source_spans_and_rejects_rewrites():
    registry = RoutingRegistry()
    raw = 'Search for "RTX 3060 12GB used Columbus"'

    accepted = registry.prepare_call(
        {
            "tool": "search",
            "arguments": {"query": "RTX 3060 12GB used Columbus"},
        },
        raw_request=raw,
        available_names=registry.names,
    )
    rejected = registry.prepare_call(
        {
            "tool": "search",
            "arguments": {"query": "used RTX 3060 graphics cards in Columbus"},
        },
        raw_request=raw,
        available_names=registry.names,
    )

    assert accepted["passed"]
    assert accepted["binding"]["source_copy_valid"] is True
    assert not rejected["passed"]
    assert "No tool ran" in rejected["errors"][0]


def test_structured_yuki_content_can_contain_pipe_before_runtime_adaptation():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {
            "tool": "yuki_write",
            "arguments": {"filename": "notes.txt", "content": "left|right"},
        },
        raw_request="Put left|right in notes.txt in your own store",
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "notes.txt|left|right"
    assert prepared["model_call"]["arguments"] == {
        "filename": "notes.txt",
        "content": "left|right",
    }


def test_explicit_cross_turn_reference_can_bind_previous_reply_payload():
    registry = RoutingRegistry()
    previous = "Right now I feel warm, awake, and happy you showed up."
    history = [
        {"role": "user", "content": "How do you feel?"},
        {"role": "assistant", "content": previous},
    ]
    raw = "Put that in feelings.txt"
    sources = referenced_literal_sources(raw, history)

    prepared = registry.prepare_call(
        {
            "tool": "yuki_write",
            "arguments": {
                "filename": "feelings.txt",
                "content": previous,
            },
        },
        raw_request=raw,
        available_names=registry.names,
        literal_sources=sources,
    )

    assert sources == [
        {
            "source_kind": "referenced_previous_user",
            "content": "How do you feel?",
        },
        {
            "source_kind": "referenced_previous_assistant",
            "content": previous,
        },
    ]
    assert prepared["passed"]
    assert prepared["runtime_argument"] == f"feelings.txt|{previous}"
    assert prepared["binding"]["field_provenance"] == {
        "filename": "raw_user_request",
        "content": "referenced_previous_assistant",
    }


def test_cross_turn_source_never_authorizes_missing_filename_or_path():
    registry = RoutingRegistry()
    history = [{
        "role": "assistant",
        "content": "I would save it as invented.txt under /tmp/invented.txt.",
    }]
    sources = referenced_literal_sources("Write that down", history)

    missing_filename = registry.prepare_call(
        {
            "tool": "yuki_write",
            "arguments": {
                "filename": "invented.txt",
                "content": "I would save it as invented.txt under /tmp/invented.txt.",
            },
        },
        raw_request="Write that down",
        available_names=registry.names,
        literal_sources=sources,
    )
    assistant_path = registry.prepare_call(
        {"tool": "read", "arguments": {"filepath": "/tmp/invented.txt"}},
        raw_request="Read that file",
        available_names=registry.names,
        literal_sources=sources,
    )

    assert not missing_filename["passed"]
    assert "filename" in missing_filename["errors"][0]
    assert not assistant_path["passed"]
    assert "filepath" in assistant_path["errors"][0]


def test_explicit_check_can_reuse_user_supplied_yuki_filename():
    registry = RoutingRegistry()
    history = [
        {
            "role": "user",
            "content": "Create feelings.txt with exactly: warm and happy.",
        },
        {"role": "assistant", "content": "Done~ feelings.txt is saved."},
    ]
    raw = "Okay, can you check it if it got modified?"
    sources = referenced_literal_sources(raw, history)
    prepared = registry.prepare_call(
        {"tool": "yuki_read", "arguments": {"filename": "feelings.txt"}},
        raw_request=raw,
        available_names=registry.names,
        literal_sources=sources,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "feelings.txt"
    assert prepared["binding"]["field_provenance"] == {
        "filename": "referenced_previous_user",
    }


def test_current_user_message_is_not_mislabeled_as_previous_source():
    current = "Read it"
    history = [
        {"role": "user", "content": "Read notes.txt"},
        {"role": "assistant", "content": "Which file?"},
        {"role": "user", "content": current},
    ]

    sources = referenced_literal_sources(current, history)

    assert sources[0] == {
        "source_kind": "referenced_previous_user",
        "content": "Read notes.txt",
    }


def test_retry_reference_reaches_past_failed_turn_to_user_supplied_filename():
    registry = RoutingRegistry()
    history = [
        {
            "role": "user",
            "content": "Create feelings.txt with exactly: warm and happy.",
        },
        {"role": "assistant", "content": "Done~ feelings.txt is saved."},
        {
            "role": "user",
            "content": "Can you check it if it got modified by someone else?",
        },
        {"role": "assistant", "content": "The validator blocked the read."},
    ]
    raw = "Wanna try again?"
    sources = referenced_literal_sources(raw, history)
    prepared = registry.prepare_call(
        {"tool": "yuki_read", "arguments": {"filename": "feelings.txt"}},
        raw_request=raw,
        available_names=registry.names,
        literal_sources=sources,
    )

    assert [source["content"] for source in sources[:2]] == [
        "Can you check it if it got modified by someone else?",
        "Create feelings.txt with exactly: warm and happy.",
    ]
    assert prepared["passed"]
    assert prepared["runtime_argument"] == "feelings.txt"


def test_previous_reply_is_not_a_source_without_explicit_action_reference():
    history = [{"role": "assistant", "content": "A sentence worth saving."}]

    assert referenced_literal_sources("That sounds good", history) == []


def test_shell_operators_are_rejected_before_execution():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "shell", "arguments": {"command": "ls; whoami"}},
        raw_request="Run ls; whoami",
        available_names=registry.names,
    )

    assert not prepared["passed"]
    assert any("operators" in error for error in prepared["errors"])


def test_native_call_normalizes_named_arguments_not_generic_arg():
    call = normalize_openai_tool_call("weather", '{"city":"Tokyo"}')
    assert call == {"tool": "weather", "arguments": {"city": "Tokyo"}}

    with patch("ms_llama.setup_llm", return_value=(MagicMock(), MagicMock(), "mlx")):
        bot = VoiceChatBot("fixture")
    bot._last_user_input = "What's the weather in Tokyo?"
    assert bot._native_tool_arg("weather", '{"city":"Tokyo"}') == "Tokyo"


def test_main_brain_decision_is_semantic_and_context_is_reference_only():
    prompt = build_main_brain_decision_prompt(
        "read that one",
        [
            {"role": "user", "content": "The path is /tmp/Exact.JSON"},
            {"role": "assistant", "content": "Got it."},
        ],
        session_context=[{
            "kind": "verified_tool_outcome",
            "tool": "search",
            "result": "Earlier result marker",
        }],
    )
    parsed = parse_main_brain_decision(json.dumps({
        "action": "delegate",
        "request": "Read the referenced operating-system file.",
        "domain_hint": "computer",
    }))

    assert "/tmp/Exact.JSON" in prompt
    assert "Earlier result marker" in prompt
    assert parsed["passed"]
    assert parsed["decision"]["domain_hint"] == "computer"


def test_main_brain_prompt_marks_retry_only_after_unresolved_tool_failure():
    retry_prompt = build_main_brain_decision_prompt(
        "Wanna try again?",
        [
            {"role": "user", "content": "Read feelings.txt"},
            {
                "role": "assistant",
                "content": "The validator blocked the read, so nothing actually ran.",
            },
        ],
    )
    ordinary_prompt = build_main_brain_decision_prompt(
        "Wanna try again?",
        [
            {"role": "user", "content": "That puzzle was difficult."},
            {"role": "assistant", "content": "Yeah, one more round could be fun."},
        ],
    )

    assert "TRUSTED RUNTIME RETRY STATE" in retry_prompt
    assert "TRUSTED RUNTIME RETRY STATE" not in ordinary_prompt


def test_autonomous_decision_contract_distinguishes_wait_speak_and_delegate():
    wait = parse_autonomous_decision(json.dumps({
        "action": "wait", "request": "", "domain_hint": "none",
    }))
    delegated = parse_autonomous_decision(json.dumps({
        "action": "delegate",
        "request": "Search for bioluminescent deep-sea animals.",
        "domain_hint": "information",
    }))
    invalid = parse_autonomous_decision(json.dumps({
        "action": "speak", "request": "say hi", "domain_hint": "media",
    }))

    assert wait["passed"]
    assert delegated["passed"]
    assert not invalid["passed"]


def test_dedicated_dispatcher_is_one_shot_validation_only():
    generation = '[{"name":"search","arguments":{"query":"Qwen 3.8"}}]'

    def backend_factory(_model_id, _output_mode):
        return FixtureBackend(generation, "MadeAgents/Hammer2.1-1.5b")

    client = DedicatedDispatcherClient(
        "MadeAgents/Hammer2.1-1.5b",
        backend_factory=backend_factory,
    )
    result = client.route(
        semantic_request="Search the web using the supplied query.",
        domain_hint="information",
        raw_request="Search for Qwen 3.8",
    )

    assert result["passed"]
    assert result["prepared"]["runtime_argument"] == "Qwen 3.8"
    assert result["generation"]["extra"]["generation_index_after_load"] == 1
    assert not hasattr(client, "execute")


def test_dedicated_dispatcher_can_bind_authorized_previous_reply_payload():
    previous = "Right now I feel warm, awake, and happy you showed up."
    generation = json.dumps([{
        "name": "yuki_write",
        "arguments": {
            "filename": "feelings.txt",
            "content": previous,
        },
    }])

    def backend_factory(_model_id, _output_mode):
        return FixtureBackend(generation, "MadeAgents/Hammer2.1-1.5b")

    client = DedicatedDispatcherClient(
        "MadeAgents/Hammer2.1-1.5b",
        backend_factory=backend_factory,
    )
    result = client.route(
        semantic_request="Create a Yuki-store file using the referenced content.",
        domain_hint="yuki_files",
        raw_request="Put that in feelings.txt",
        literal_sources=[{
            "source_kind": "referenced_previous_assistant",
            "content": previous,
        }],
    )

    assert result["passed"]
    assert result["prepared"]["runtime_argument"] == f"feelings.txt|{previous}"
    assert result["prepared"]["binding"]["source_kind"] == "mixed_authorized_sources"


def _fixture_bot(**routing):
    with patch("ms_llama.setup_llm", return_value=(MagicMock(), MagicMock(), "mlx")):
        return VoiceChatBot("fixture-main", **routing)


def _install_final_chat(bot, text="confirmed final"):
    def fake_chat(user_input, max_tokens=None, tool_context=None):
        del max_tokens, tool_context
        if user_input:
            bot.history.append({"role": "user", "content": user_input})
        bot.history.append({"role": "assistant", "content": text})
        return text

    bot.chat = fake_chat


def test_direct_canonical_executes_only_after_binding_and_validation():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="canonical",
    )
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "tool": "search",
        "arguments": {"query": "Exact Qwen Query"},
    }))
    bot._run_tool_with_spinner = MagicMock(return_value=("raw search result", False))
    _install_final_chat(bot)

    response = bot.react_chat("Search for Exact Qwen Query")

    assert response == "confirmed final"
    bot._run_tool_with_spinner.assert_called_once_with("search", "Exact Qwen Query")
    assert bot.last_tool_routing["passed"]
    assert bot.history == [
        {"role": "user", "content": "Search for Exact Qwen Query"},
        {"role": "assistant", "content": "confirmed final"},
    ]
    event = bot._session_continuity[-1]
    assert event["kind"] == "verified_tool_outcome"
    assert event["tool"] == "search"
    assert event["result"] == "raw search result"


def test_direct_canonical_can_write_exact_previous_reply_with_current_filename():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="canonical",
    )
    previous = "Right now I feel warm, awake, and happy you showed up."
    bot.history = [
        {"role": "user", "content": "How do you feel?"},
        {"role": "assistant", "content": previous},
    ]
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "tool": "yuki_write",
        "arguments": {
            "filename": "feelings.txt",
            "content": previous,
        },
    }))
    bot._run_tool_with_spinner = MagicMock(return_value=("file written", False))
    _install_final_chat(bot, "Saved it~")

    response = bot.react_chat("Put that in feelings.txt")

    assert response == "Saved it~"
    bot._run_tool_with_spinner.assert_called_once_with(
        "yuki_write",
        f"feelings.txt|{previous}",
    )
    router_prompt = bot.raw_structured_complete.call_args.args[1]
    assert "AUTHORIZED REFERENCED LITERAL SOURCES" in router_prompt
    assert previous in router_prompt


def test_missing_cross_turn_filename_rejects_call_and_blocks_fake_call_text():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="canonical",
    )
    previous = "Right now I feel warm and happy."
    bot.history = [
        {"role": "user", "content": "How do you feel?"},
        {"role": "assistant", "content": previous},
    ]
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "tool": "yuki_write",
        "arguments": {
            "filename": "invented.txt",
            "content": previous,
        },
    }))
    bot._run_tool_with_spinner = MagicMock()

    def leaking_chat(user_input, max_tokens=None, tool_context=None):
        del user_input, max_tokens, tool_context
        return (
            "[TOOL_CALL: yuki_write with path=yuki/invented.txt "
            f"and content={previous}]"
        )

    bot.chat = leaking_chat

    response = bot.react_chat("Create a file and write that")

    assert "nothing actually ran" in response
    assert "TOOL_CALL" not in response
    bot._run_tool_with_spinner.assert_not_called()
    assert bot.last_tool_routing["passed"] is False


def test_direct_canonical_malformed_route_safely_becomes_no_tool_chat():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="canonical",
    )
    bot.raw_structured_complete = MagicMock(return_value="Sure, I'll do that")
    bot._run_tool_with_spinner = MagicMock()
    _install_final_chat(bot, "ordinary reply")

    response = bot.react_chat("hello there")

    assert response == "ordinary reply"
    bot._run_tool_with_spinner.assert_not_called()
    assert bot.last_tool_routing["respond"] is True
    assert not bot.last_tool_routing["parse"]["passed"]


def test_rewritten_literal_is_blocked_and_never_uses_success_fallback():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="canonical",
    )
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "tool": "search",
        "arguments": {"query": "normalized qwen query"},
    }))
    bot._run_tool_with_spinner = MagicMock()
    bot._chat_with_tool_fallback = MagicMock(
        side_effect=AssertionError("successful-tool fallback must not run"),
    )

    def explain_rejection(user_input, max_tokens=None, tool_context=None):
        del user_input, max_tokens, tool_context
        assert "No tool ran" in bot.history[-1]["content"]
        return "I couldn't validate that exact request."

    bot.chat = explain_rejection
    response = bot.react_chat("Search for QWEN Query")

    assert response == "I couldn't validate that exact request."
    bot._run_tool_with_spinner.assert_not_called()
    assert bot.last_tool_routing["passed"] is False


def test_dispatcher_mode_uses_frozen_raw_request_for_final_binding():
    bot = _fixture_bot(
        tool_routing_mode="dispatcher",
        dispatcher_model_id="fixture-dispatcher",
    )
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "action": "delegate",
        "request": "Search the web with the supplied query.",
        "domain_hint": "information",
    }))
    prepared = bot._routing_registry.prepare_call(
        {"tool": "search", "arguments": {"query": "MiXeD Case"}},
        raw_request="Search MiXeD Case",
        available_names=bot._routing_registry.resolve_names(group="information"),
    )
    dispatcher = MagicMock()
    dispatcher.route.return_value = {
        "passed": True,
        "errors": [],
        "canonical_call": {"tool": "search", "arguments": {"query": "MiXeD Case"}},
        "prepared": prepared,
    }
    bot._dispatcher_client = dispatcher
    bot._run_tool_with_spinner = MagicMock(return_value=("raw result", False))
    _install_final_chat(bot)

    bot.react_chat("Search MiXeD Case")

    dispatcher.route.assert_called_once_with(
        semantic_request="Search the web with the supplied query.",
        domain_hint="information",
        raw_request="Search MiXeD Case",
    )
    bot._run_tool_with_spinner.assert_called_once_with("search", "MiXeD Case")


def test_dispatcher_followup_can_read_user_named_yuki_file():
    bot = _fixture_bot(
        tool_routing_mode="dispatcher",
        dispatcher_model_id="fixture-dispatcher",
    )
    bot.history = [
        {
            "role": "user",
            "content": "Create feelings.txt with exactly: warm and happy.",
        },
        {"role": "assistant", "content": "Done~ feelings.txt is saved."},
        {
            "role": "user",
            "content": "Can you check it if it got modified by someone else?",
        },
        {"role": "assistant", "content": "The validator blocked the read."},
    ]
    current = "Wanna try again?"
    sources = referenced_literal_sources(current, bot.history)
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "action": "delegate",
        "request": "Inspect the referenced file in Yuki's own file store.",
        "domain_hint": "yuki_files",
    }))
    prepared = bot._routing_registry.prepare_call(
        {"tool": "yuki_read", "arguments": {"filename": "feelings.txt"}},
        raw_request=current,
        available_names=bot._routing_registry.resolve_names(group="yuki-files"),
        literal_sources=sources,
    )
    dispatcher = MagicMock()
    dispatcher.route.return_value = {
        "passed": True,
        "errors": [],
        "canonical_call": {
            "tool": "yuki_read",
            "arguments": {"filename": "feelings.txt"},
        },
        "prepared": prepared,
    }
    bot._dispatcher_client = dispatcher
    bot._run_tool_with_spinner = MagicMock(return_value=("warm and happy.", False))
    _install_final_chat(bot, "It still matches what you wrote.")

    response = bot.react_chat(current)

    assert response == "It still matches what you wrote."
    planner_prompt = bot.raw_structured_complete.call_args.args[1]
    assert "TRUSTED RUNTIME RETRY STATE" in planner_prompt
    dispatcher.route.assert_called_once_with(
        semantic_request="Inspect the referenced file in Yuki's own file store.",
        domain_hint="yuki_files",
        raw_request=current,
        literal_sources=sources,
    )
    bot._run_tool_with_spinner.assert_called_once_with(
        "yuki_read", "feelings.txt",
    )


def test_native_direct_path_normalizes_named_call_and_never_runs_regex_fallback():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="native",
    )
    bot.backend = "openrouter"
    bot._chat_with_tools = MagicMock(side_effect=[
        {
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "name": "weather",
                "arg_raw": '{"city":"Tokyo"}',
            }],
            "reasoning": "",
        },
        {"content": "Tokyo result explained", "tool_calls": [], "reasoning": ""},
    ])
    bot._run_tool_with_spinner = MagicMock(return_value=("raw weather", False))

    response = bot.react_chat("What's the weather in Tokyo?")

    assert response == "Tokyo result explained"
    bot._run_tool_with_spinner.assert_called_once_with("weather", "Tokyo")
    assert bot.last_tool_routing["canonical_call"] == {
        "tool": "weather",
        "arguments": {"city": "Tokyo"},
    }


def test_native_path_can_bind_previous_reply_but_not_invent_filename():
    previous = "Right now I feel warm, awake, and happy you showed up."
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="native",
    )
    bot.backend = "openrouter"
    bot.history = [
        {"role": "user", "content": "How do you feel?"},
        {"role": "assistant", "content": previous},
    ]
    bot._chat_with_tools = MagicMock(side_effect=[
        {
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "name": "yuki_write",
                "arg_raw": json.dumps({
                    "filename": "feelings.txt",
                    "content": previous,
                }),
            }],
            "reasoning": "",
        },
        {"content": "Saved it~", "tool_calls": [], "reasoning": ""},
    ])
    bot._run_tool_with_spinner = MagicMock(return_value=("file written", False))

    response = bot.react_chat("Put that in feelings.txt")

    assert response == "Saved it~"
    bot._run_tool_with_spinner.assert_called_once_with(
        "yuki_write",
        f"feelings.txt|{previous}",
    )


def test_native_rejected_call_cannot_render_fake_tool_syntax():
    previous = "Right now I feel warm and happy."
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="native",
    )
    bot.backend = "openrouter"
    bot.history = [{"role": "assistant", "content": previous}]
    bot._chat_with_tools = MagicMock(side_effect=[
        {
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "name": "yuki_write",
                "arg_raw": json.dumps({
                    "filename": "invented.txt",
                    "content": previous,
                }),
            }],
            "reasoning": "",
        },
        {
            "content": "[TOOL_CALL: yuki_write with path=yuki/invented.txt]",
            "tool_calls": [],
            "reasoning": "",
        },
    ])
    bot._run_tool_with_spinner = MagicMock()

    response = bot.react_chat("Create a file and write that")

    assert "nothing actually ran" in response
    assert "TOOL_CALL" not in response
    bot._run_tool_with_spinner.assert_not_called()


def test_autonomy_freezes_goal_context_and_wait_does_not_create_idle_chatter():
    bot = _fixture_bot(tool_routing_mode="direct", tool_routing_protocol="canonical")
    bot.history = [
        {"role": "user", "content": "Explore the new tool architecture while I'm away."},
        {"role": "assistant", "content": "Got it."},
    ]
    bot.begin_autonomy()
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "action": "wait", "request": "", "domain_hint": "none",
    }))
    bot.chat = MagicMock(side_effect=AssertionError("WAIT must not generate chatter"))
    bot._run_tool_with_spinner = MagicMock()

    assert bot.autonomous_tick(60) is None
    assert bot._autonomous_context[0]["content"].startswith("Explore")
    assert bot.history[-1] == {"role": "assistant", "content": "Got it."}
    planner_prompt = bot.raw_structured_complete.call_args.args[1]
    assert '"ask_claude"' in planner_prompt
    assert '"image"' in planner_prompt
    bot.chat.assert_not_called()
    bot._run_tool_with_spinner.assert_not_called()


def test_autonomous_direct_action_uses_canonical_boundary_and_allows_tools():
    bot = _fixture_bot(tool_routing_mode="direct", tool_routing_protocol="canonical")
    bot.history = [{"role": "user", "content": "Go explore while I'm away."}]
    bot.begin_autonomy()
    bot.raw_structured_complete = MagicMock(side_effect=[
        json.dumps({
            "action": "delegate",
            "request": "Search for bioluminescent deep-sea animals.",
            "domain_hint": "information",
        }),
        json.dumps({
            "tool": "search",
            "arguments": {"query": "bioluminescent deep-sea animals"},
        }),
    ])
    bot._run_tool_with_spinner = MagicMock(return_value=("raw search results", False))
    _install_final_chat(bot, "I found something genuinely weird in the deep sea.")

    response = bot.autonomous_tick(60)

    assert response == "I found something genuinely weird in the deep sea."
    bot._run_tool_with_spinner.assert_called_once_with(
        "search", "bioluminescent deep-sea animals", autonomous=True,
    )
    assert bot.last_tool_routing["autonomous"] is True
    assert bot.last_tool_routing["prepared"]["binding"]["source_kind"] == (
        "autonomous_action"
    )
    assert bot._autonomous_actions[-1]["tool"] == "search"
    event = bot._session_continuity[-1]
    assert event["kind"] == "verified_tool_outcome"
    assert event["source"] == "autonomous"
    assert event["result"] == "raw search results"
    assert bot.last_autonomous_transcript["user"].startswith(
        "[AUTONOMOUS ACTION:"
    )
    assert "raw search results" in bot._autonomous_planner_prompt(120)


def test_autonomous_speak_is_meaningful_and_preserves_last_human_input():
    bot = _fixture_bot(tool_routing_mode="direct")
    bot.history = [{"role": "user", "content": "Take your time while I'm gone."}]
    bot._last_user_input = "Take your time while I'm gone."
    bot.begin_autonomy()
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "action": "speak",
        "request": "Share that the architecture inspection has one useful finding.",
        "domain_hint": "none",
    }))
    _install_final_chat(bot, "I found one useful edge case in the routing boundary.")
    bot._run_tool_with_spinner = MagicMock()

    response = bot.autonomous_tick(60)

    assert response == "I found one useful edge case in the routing boundary."
    assert bot._last_user_input == "Take your time while I'm gone."
    bot._run_tool_with_spinner.assert_not_called()
    assert bot._autonomous_actions[-1]["action"] == "spoke"


def test_autonomous_dispatcher_uses_selected_client_and_autonomous_source():
    bot = _fixture_bot(
        tool_routing_mode="dispatcher",
        dispatcher_model_id="fixture-dispatcher",
    )
    bot.begin_autonomy()
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "action": "delegate",
        "request": "Read field-notes.md from Yuki's own file store.",
        "domain_hint": "yuki_files",
    }))
    prepared = bot._routing_registry.prepare_call(
        {"tool": "yuki_read", "arguments": {"filename": "field-notes.md"}},
        raw_request="Read field-notes.md from Yuki's own file store.",
        available_names=bot._routing_registry.resolve_names(group="yuki-files"),
        source_kind="autonomous_action",
    )
    dispatcher = MagicMock()
    dispatcher.route.return_value = {
        "passed": True,
        "errors": [],
        "canonical_call": {
            "tool": "yuki_read", "arguments": {"filename": "field-notes.md"},
        },
        "prepared": prepared,
    }
    bot._dispatcher_client = dispatcher
    bot._run_tool_with_spinner = MagicMock(return_value=("stored notes", False))
    _install_final_chat(bot, "My field notes are still intact.")

    bot.autonomous_tick(120)

    dispatcher.route.assert_called_once_with(
        semantic_request="Read field-notes.md from Yuki's own file store.",
        domain_hint="yuki_files",
        raw_request="Read field-notes.md from Yuki's own file store.",
        source_kind="autonomous_action",
    )
    bot._run_tool_with_spinner.assert_called_once_with(
        "yuki_read", "field-notes.md", autonomous=True,
    )


def test_autonomous_image_is_not_blocked_by_present_user_intent_gate():
    bot = _fixture_bot(tool_routing_mode="direct", tool_routing_protocol="canonical")
    bot.begin_autonomy()
    bot.raw_structured_complete = MagicMock(side_effect=[
        json.dumps({
            "action": "delegate",
            "request": "Create an image of a neon lighthouse in a violet storm.",
            "domain_hint": "media",
        }),
        json.dumps({
            "tool": "image",
            "arguments": {"prompt": "a neon lighthouse in a violet storm"},
        }),
    ])
    bot._run_tool_with_spinner = MagicMock(return_value=("saved image", False))
    _install_final_chat(bot, "I made the stormy lighthouse.")

    bot.autonomous_tick(60)

    bot._run_tool_with_spinner.assert_called_once_with(
        "image", "a neon lighthouse in a violet storm", autonomous=True,
    )


def test_image_execution_chokepoint_accepts_explicit_autonomous_authorization():
    bot = _fixture_bot()
    bot._last_user_input = "I'm away now"
    bot._emit_tool_progress = lambda _line: None
    image_tool = MagicMock(return_value="saved autonomous image")

    with patch.dict("ms_llama.REACT_TOOL_MAP", {"image": image_tool}, clear=True):
        result, blocked = bot._run_tool_with_spinner(
            "image", "neon lighthouse", autonomous=True,
        )

    assert not blocked
    assert result == "saved autonomous image"
    image_tool.assert_called_once_with("neon lighthouse")


def test_no_tool_fallback_rejects_legacy_syntax_instead_of_faking_success():
    bot = _fixture_bot(tool_routing_mode="direct", tool_routing_protocol="canonical")
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "tool": "no_tool", "arguments": {},
    }))

    def leaking_chat(user_input, max_tokens=None, tool_context=None):
        del max_tokens, tool_context
        bot.history.extend([
            {"role": "user", "content": user_input},
            {
                "role": "assistant",
                "content": '[TOOL: weather("Reykjavik")] It worked; it is 2°C.',
            },
        ])
        return bot.history[-1]["content"]

    bot.chat = leaking_chat
    bot._run_tool_with_spinner = MagicMock()

    response = bot.react_chat("and go")

    assert "nothing actually ran" in response
    assert "2°C" not in response
    assert "[TOOL:" not in response
    bot._run_tool_with_spinner.assert_not_called()


def test_no_tool_fallback_rejects_nonlegacy_fake_call_formats():
    fake_outputs = (
        '[TOOL_CALL: yuki_write with path=yuki/feelings.txt and content=hello]',
        '<tool_call>{"name":"yuki_write"}</tool_call>',
        '{"tool":"yuki_write","arguments":{"filename":"feelings.txt"}}',
    )

    for fake_output in fake_outputs:
        bot = _fixture_bot(
            tool_routing_mode="direct",
            tool_routing_protocol="canonical",
        )
        bot.raw_structured_complete = MagicMock(return_value=json.dumps({
            "tool": "no_tool", "arguments": {},
        }))

        def leaking_chat(
            user_input,
            max_tokens=None,
            tool_context=None,
            *,
            _bot=bot,
            _fake_output=fake_output,
        ):
            del max_tokens, tool_context
            _bot.history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": _fake_output},
            ])
            return _fake_output

        bot.chat = leaking_chat
        bot._run_tool_with_spinner = MagicMock()

        response = bot.react_chat("do that")

        assert "nothing actually ran" in response
        assert "tool_call" not in response.casefold()
        assert '"tool"' not in response
        bot._run_tool_with_spinner.assert_not_called()
