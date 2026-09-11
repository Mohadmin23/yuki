"""Tool-routing checks with no model loads and no Yuki tool execution."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ms_llama import (
    VoiceChatBot,
    fetch_openrouter_models,
    openrouter_supported_parameters,
    supports_native_tools,
)
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


def test_search_contract_accepts_ordinary_semantic_rewrite():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {
            "tool": "search",
            "arguments": {"query": "recent sqlite-vec changes"},
        },
        raw_request="What changed in sqlite-vec recently?",
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "recent sqlite-vec changes"
    assert prepared["model_call"]["arguments"]["query"] == (
        "recent sqlite-vec changes"
    )
    assert prepared["binding"]["mode"] == "semantic_argument"


def test_search_contract_accepts_conversational_nvlink_rewrite():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {
            "tool": "search",
            "arguments": {"query": "latest NVLink news"},
        },
        raw_request="can you also look for some news about NVlink",
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "latest NVLink news"
    assert prepared["binding"]["mode"] == "semantic_argument"


def test_search_contract_restores_explicit_exact_operator_query_from_raw():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {
            "tool": "search",
            "arguments": {"query": "normalized qwen stable release"},
        },
        raw_request=(
            'Search exactly for: site:example.com "Qwen 3.8" -beta OR stable?'
        ),
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == (
        'site:example.com "Qwen 3.8" -beta OR stable?'
    )
    assert prepared["binding"]["mode"] == "deterministic_source_span"
    assert prepared["binding"]["source_copy_valid"] is True
    assert prepared["proposed_call"]["arguments"]["query"] == (
        "normalized qwen stable release"
    )


def test_search_contract_rejects_exact_intent_without_bindable_payload():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "invented query"}},
        raw_request="Search exactly for:",
        available_names=registry.names,
    )

    assert not prepared["passed"]
    assert prepared["runtime_argument"] is None
    assert "No tool ran" in prepared["errors"][0]


def test_search_pronoun_repeat_uses_latest_trusted_successful_search():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "it"}},
        raw_request="Search it",
        available_names=registry.names,
        trusted_context=[
            {
                "kind": "verified_tool_outcome",
                "tool": "search",
                "arguments": {"query": "older SQLite query"},
                "succeeded": True,
            },
            {
                "kind": "verified_tool_outcome",
                "tool": "search",
                "arguments": {"query": "failed search query"},
                "succeeded": False,
            },
            {
                "kind": "verified_tool_outcome",
                "tool": "search",
                "arguments": {"query": "latest NVLink query"},
                "succeeded": True,
            },
        ],
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "latest NVLink query"
    assert prepared["binding"]["mode"] == "contextual_semantic_resolution"
    assert prepared["binding"]["binder"] == "trusted_recent_search_v1"


def test_search_pronoun_without_trusted_successful_search_rejects():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "it"}},
        raw_request="Search it",
        available_names=registry.names,
        trusted_context=[
            {
                "kind": "verified_tool_outcome",
                "tool": "search",
                "arguments": {"query": "failed search query"},
                "succeeded": False,
            },
            {
                "kind": "verified_tool_outcome",
                "tool": "recall",
                "arguments": {"topic": "NVLink"},
                "succeeded": True,
            },
        ],
    )

    assert not prepared["passed"]
    assert prepared["runtime_argument"] is None
    assert "No tool ran" in prepared["errors"][0]


def test_search_pronoun_cannot_choose_between_untrusted_history_topics():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "that"}},
        raw_request="Search that",
        available_names=registry.names,
        literal_sources=[
            {
                "source_kind": "referenced_previous_user",
                "content": "We discussed SQLite releases.",
            },
            {
                "source_kind": "referenced_previous_user",
                "content": "We also discussed NVLink releases.",
            },
        ],
        trusted_context=[],
    )

    assert not prepared["passed"]
    assert prepared["runtime_argument"] is None


def test_resolved_current_topic_is_not_overwritten_by_older_search_ledger():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "latest NVLink news"}},
        raw_request="Search it",
        available_names=registry.names,
        literal_sources=[{
            "source_kind": "referenced_previous_user",
            "content": "I want to know what changed with NVLink.",
        }],
        trusted_context=[{
            "kind": "verified_tool_outcome",
            "tool": "search",
            "arguments": {"query": "SQLite release notes"},
            "succeeded": True,
        }],
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "latest NVLink news"
    assert prepared["binding"]["mode"] == "semantic_argument"


def test_unresolved_repeat_prefers_nearest_user_search_over_old_ledger():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "it"}},
        raw_request="Search it",
        available_names=registry.names,
        literal_sources=[
            {
                "source_kind": "referenced_previous_user",
                "content": "Search the web for the newest NVLink announcements.",
            },
            {
                "source_kind": "referenced_previous_user",
                "content": "Search the web for SQLite release notes.",
            },
        ],
        trusted_context=[{
            "kind": "verified_tool_outcome",
            "tool": "search",
            "arguments": {"query": "SQLite release notes"},
            "succeeded": True,
        }],
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "the newest NVLink announcements"
    assert prepared["binding"]["binder"] == (
        "referenced_previous_user_search_v1"
    )


def test_unresolved_newer_topic_never_falls_back_to_old_search_ledger():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "it"}},
        raw_request="Search it",
        available_names=registry.names,
        literal_sources=[{
            "source_kind": "referenced_previous_user",
            "content": "I want to know what changed with NVLink.",
        }],
        trusted_context=[{
            "kind": "verified_tool_outcome",
            "tool": "search",
            "arguments": {"query": "SQLite release notes"},
            "succeeded": True,
        }],
    )

    assert not prepared["passed"]
    assert prepared["runtime_argument"] is None


def test_modified_search_reference_keeps_modifier_and_resolves_referent():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "recent news on that"}},
        raw_request="Could you search for more news about it?",
        available_names=registry.names,
        trusted_context=[{
            "kind": "verified_tool_outcome",
            "tool": "search",
            "arguments": {"query": "NVLink"},
            "succeeded": True,
        }],
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "recent news on NVLink"
    assert prepared["binding"]["mode"] == "contextual_semantic_resolution"


def test_modified_search_reference_without_context_rejects():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "updates for them"}},
        raw_request="Search for updates about them",
        available_names=registry.names,
    )

    assert not prepared["passed"]
    assert prepared["runtime_argument"] is None


@pytest.mark.parametrize(
    "raw_request",
    [
        "Can you look at the image?",
        "Can you look for my phone with the camera?",
    ],
)
def test_visual_request_cannot_be_salvaged_as_search(raw_request):
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "my phone"}},
        raw_request=raw_request,
        available_names=registry.names,
    )

    assert not prepared["passed"]
    assert prepared["runtime_argument"] is None


def test_search_schema_describes_hybrid_query_boundaries():
    registry = RoutingRegistry()
    description = registry.description_for("search").casefold()
    parameter = (
        registry.argument_schema_for("search")["properties"]["query"]
        ["description"].casefold()
    )
    combined = f"{description}\n{parameter}"

    assert "self-contained" in combined
    assert "recent" in combined and "context" in combined
    assert "exact" in combined and "quoted" in combined
    assert "operator" in combined
    assert "camera" in combined or "see" in combined


def test_search_exact_word_in_topic_does_not_trigger_literal_transport():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "meaning of exact"}},
        raw_request="Search what the word exact means",
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "meaning of exact"
    assert prepared["binding"]["mode"] == "semantic_argument"


def test_natural_question_with_quoted_name_remains_semantic():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {
            "tool": "search",
            "arguments": {"query": 'recent changes to "sqlite-vec"'},
        },
        raw_request='What changed in "sqlite-vec" recently?',
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == 'recent changes to "sqlite-vec"'
    assert prepared["binding"]["mode"] == "semantic_argument"


def test_web_image_search_is_not_mistaken_for_visual_inspection():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "Apollo mission photos"}},
        raw_request="Search the web for images from the Apollo missions",
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == "Apollo mission photos"


def test_look_for_images_can_remain_an_ordinary_web_search():
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": "cat images"}},
        raw_request="Look for images of cats",
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["binding"]["mode"] == "semantic_argument"


@pytest.mark.parametrize(
    ("raw_request", "query"),
    [
        ("Check image compression benchmarks online", "image compression benchmarks"),
        ("Check the latest camera news", "latest camera news"),
        ("Describe image-generation model trends", "image generation model trends"),
    ],
)
def test_media_words_in_web_research_do_not_trigger_visual_guard(
    raw_request, query,
):
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": query}},
        raw_request=raw_request,
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == query


@pytest.mark.parametrize(
    ("raw_request", "query"),
    [
        ("Search for ways to keep the query fast", "fast search query design"),
        ("Which is better, AMD OR Intel?", "AMD versus Intel comparison"),
    ],
)
def test_search_topic_words_do_not_false_trigger_exact_mode(raw_request, query):
    registry = RoutingRegistry()
    prepared = registry.prepare_call(
        {"tool": "search", "arguments": {"query": query}},
        raw_request=raw_request,
        available_names=registry.names,
    )

    assert prepared["passed"]
    assert prepared["runtime_argument"] == query
    assert prepared["binding"]["mode"] == "semantic_argument"


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


def test_dedicated_dispatcher_threads_trusted_search_context():
    generation = '[{"name":"search","arguments":{"query":"it"}}]'

    def backend_factory(_model_id, _output_mode):
        return FixtureBackend(generation, "MadeAgents/Hammer2.1-1.5b")

    client = DedicatedDispatcherClient(
        "MadeAgents/Hammer2.1-1.5b",
        backend_factory=backend_factory,
    )
    result = client.route(
        semantic_request="Repeat the referenced web search.",
        domain_hint="information",
        raw_request="Search it",
        trusted_context=[{
            "kind": "verified_tool_outcome",
            "tool": "search",
            "arguments": {"query": "latest NVLink news"},
            "succeeded": True,
        }],
    )

    assert result["passed"]
    assert result["prepared"]["runtime_argument"] == "latest NVLink news"
    assert result["prepared"]["binding"]["binder"] == (
        "trusted_recent_search_v1"
    )


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
    bot.raw_structured_complete = MagicMock(side_effect=[json.dumps({
        "tool": "yuki_write",
        "arguments": {
            "filename": "feelings.txt",
            "content": previous,
        },
    }), json.dumps({"mode": "literal", "tool": "yuki_write", "filename": "feelings.txt", "content": previous})])
    bot._run_tool_with_spinner = MagicMock(return_value=("file written", False))
    _install_final_chat(bot, "Saved it~")

    response = bot.react_chat("Put that in feelings.txt")

    assert response == "Saved it~"
    bot._run_tool_with_spinner.assert_called_once_with(
        "yuki_write",
        f"feelings.txt|{previous}",
    )
    router_prompt = bot.raw_structured_complete.call_args_list[0].args[1]
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


def test_direct_canonical_malformed_route_is_visible_routing_failure():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="canonical",
    )
    bot.raw_structured_complete = MagicMock(return_value="Sure, I'll do that")
    bot._run_tool_with_spinner = MagicMock()
    def explain_failure(user_input, max_tokens=None, tool_context=None):
        del user_input, max_tokens, tool_context
        assert "No tool ran" in bot.history[-1]["content"]
        return "The internal tool route failed validation."

    bot.chat = explain_failure

    response = bot.react_chat("hello there")

    assert response == "The internal tool route failed validation."
    bot._run_tool_with_spinner.assert_not_called()
    assert bot.last_tool_routing["respond"] is False
    assert not bot.last_tool_routing["parse"]["passed"]


def test_dispatcher_stage_a_malformed_route_is_visible_routing_failure():
    bot = _fixture_bot(
        tool_routing_mode="dispatcher",
        tool_routing_protocol="native",
        dispatcher_model_id="fixture-dispatcher",
    )
    bot.raw_structured_complete = MagicMock(return_value="not json")
    bot._last_structured_error = "BadRequestError: provider rejected schema"
    bot._dispatcher_client = MagicMock()
    bot._run_tool_with_spinner = MagicMock()

    def explain_failure(user_input, max_tokens=None, tool_context=None):
        del user_input, max_tokens, tool_context
        assert "No tool ran" in bot.history[-1]["content"]
        return "My semantic delegation failed, so nothing ran."

    bot.chat = explain_failure
    response = bot.react_chat("Search for sqlite-vec changes")

    assert response == "My semantic delegation failed, so nothing ran."
    assert bot.last_tool_routing["passed"] is False
    assert bot.last_tool_routing["respond"] is False
    assert "provider rejected schema" in bot.last_tool_routing["errors"][0]
    bot._dispatcher_client.route.assert_not_called()
    bot._run_tool_with_spinner.assert_not_called()


def test_openrouter_catalog_capabilities_override_name_guessing():
    with patch.dict(
        "ms_llama._OR_MODEL_SUPPORTED_PARAMETERS",
        {
            "z-ai/glm-5.3-flash": frozenset({"tools", "tool_choice"}),
            "qwen/qwen3.8-27b": frozenset({"tools", "response_format"}),
            "qwen/qwen3-no-tools": frozenset({"response_format"}),
        },
        clear=True,
    ):
        assert supports_native_tools(
            "openrouter/z-ai/glm-5.3-flash",
            "openrouter",
        )
        assert supports_native_tools(
            "openrouter/qwen/qwen3.8-27b",
            "openrouter",
        )
        assert not supports_native_tools(
            "openrouter/qwen/qwen3-no-tools",
            "openrouter",
        )
        assert supports_native_tools(
            "openrouter/z-ai/glm-5.3-flash:free",
            "openrouter",
        )


def test_openrouter_catalog_records_supported_parameters():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({
        "data": [{
            "id": "z-ai/glm-5.3-flash",
            "name": "GLM 5.3 Flash",
            "supported_parameters": ["tools", "tool_choice", "response_format"],
            "pricing": {"prompt": "0.15"},
        }],
    }).encode()

    with (
        patch("ms_llama.urlopen", return_value=response),
        patch.dict(
            "ms_llama._or_cache",
            {"models": None, "ts": 0, "failure_ts": 0},
            clear=True,
        ),
        patch.dict(
            "ms_llama._OR_MODEL_SUPPORTED_PARAMETERS",
            {"stale/model": frozenset({"tools"})},
            clear=True,
        ),
    ):
        models = fetch_openrouter_models()
        assert models[0]["supported_parameters"] == [
            "tools",
            "tool_choice",
            "response_format",
        ]
        assert supports_native_tools(
            "openrouter/z-ai/glm-5.3-flash",
            "openrouter",
        )
        assert openrouter_supported_parameters("stale/model") is None


def test_openrouter_catalog_failure_has_short_retry_backoff():
    with (
        patch("ms_llama.urlopen", side_effect=OSError("offline")) as request,
        patch.dict(
            "ms_llama._or_cache",
            {"models": None, "ts": 0, "failure_ts": 0},
            clear=True,
        ),
    ):
        first = fetch_openrouter_models()
        second = fetch_openrouter_models()

    assert first == second
    assert request.call_count == 1


def test_direct_auto_uses_native_tools_for_catalog_capable_glm():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="auto",
    )
    bot.backend = "openrouter"
    bot.model_id = "openrouter/z-ai/glm-5.3-flash"
    bot._react_chat_native = MagicMock(return_value="native reply")
    bot._react_chat_canonical = MagicMock(
        side_effect=AssertionError("catalog-capable GLM must not use canonical routing"),
    )

    with patch.dict(
        "ms_llama._OR_MODEL_SUPPORTED_PARAMETERS",
        {"z-ai/glm-5.3-flash": frozenset({"tools", "tool_choice"})},
        clear=True,
    ):
        response = bot.react_chat("Search the web for sqlite-vec changes")

    assert response == "native reply"
    bot._react_chat_native.assert_called_once_with(
        "Search the web for sqlite-vec changes",
        max_tokens=1024,
        max_steps=3,
    )
    bot._react_chat_canonical.assert_not_called()


def test_openrouter_native_request_keeps_tools_and_omits_redundant_tool_choice():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="native",
    )
    bot.backend = "openrouter"
    bot.llm_model = MagicMock()
    bot.llm_model._or_model = "qwen/qwen3.8-27b"
    message = SimpleNamespace(
        content="ordinary answer",
        tool_calls=[],
        reasoning="",
        reasoning_content="",
        reasoning_details=[],
    )
    bot.llm_model.chat.completions.create.return_value = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(message=message)],
    )
    schemas = bot._routing_registry.strict_openai_schemas(
        bot._routing_registry.names,
    )

    response = bot._chat_with_tools([], schemas, 256, 0.2)

    assert response["content"] == "ordinary answer"
    call = bot.llm_model.chat.completions.create.call_args.kwargs
    assert call["tools"] is schemas
    assert "tool_choice" not in call


def test_openrouter_native_failure_never_retries_as_tool_free_chat():
    class NotFoundError(Exception):
        pass

    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="native",
    )
    bot.backend = "openrouter"
    bot.llm_model = MagicMock()
    bot.llm_model._or_model = "z-ai/glm-5.3-flash"
    error = NotFoundError(
        "No endpoints found that support the provided 'tools' value."
    )
    bot.llm_model.chat.completions.create.side_effect = error
    schemas = bot._routing_registry.strict_openai_schemas(
        bot._routing_registry.names,
    )

    with pytest.raises(NotFoundError):
        bot._chat_with_tools([], schemas, 256, 0.2)

    assert bot.llm_model.chat.completions.create.call_count == 1
    call = bot.llm_model.chat.completions.create.call_args.kwargs
    assert call["tools"] is schemas
    assert "tool_choice" not in call


def test_native_provider_tool_failure_is_visible_and_records_failed_route():
    class NotFoundError(Exception):
        pass

    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="native",
    )
    bot.backend = "openrouter"
    bot._chat_with_tools = MagicMock(side_effect=NotFoundError(
        "No endpoints found that support the provided 'tools' value."
    ))
    bot._run_tool_with_spinner = MagicMock()

    response = bot.react_chat("Search the web for sqlite-vec changes")

    assert "no tool ran" in response.lower()
    assert "Dispatcher routing" in response
    assert bot.last_tool_routing["passed"] is False
    assert bot.last_tool_routing["protocol"] == "native"
    bot._run_tool_with_spinner.assert_not_called()


def test_native_followup_failure_never_denies_completed_tool_execution():
    class APIConnectionError(Exception):
        pass

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
                "name": "search",
                "arg_raw": '{"query":"sqlite-vec changes"}',
            }],
            "reasoning": "",
        },
        APIConnectionError("connection dropped after tool result"),
    ])
    bot._run_tool_with_spinner = MagicMock(return_value=(
        "Search completed with 3 sources.",
        False,
    ))
    bot._remember_tool_outcome = MagicMock()

    response = bot.react_chat("Search for sqlite-vec changes")

    assert "Search completed with 3 sources." in response
    assert "tool already ran" in response
    assert "no tool ran" not in response.lower()
    bot._run_tool_with_spinner.assert_called_once_with(
        "search",
        "sqlite-vec changes",
    )
    assert bot.last_tool_routing["passed"] is True
    assert bot.last_tool_routing["execution_completed"] is True
    assert "APIConnectionError" in bot.last_tool_routing["final_response_error"]


def test_native_ordinary_search_uses_self_contained_semantic_rewrite():
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
                "name": "search",
                "arg_raw": '{"query":"latest changes to sqlite-vec"}',
            }],
            "reasoning": "",
        },
        {
            "content": "Here are the fresh sqlite-vec changes.",
            "tool_calls": [],
            "reasoning": "",
        },
    ])
    bot._run_tool_with_spinner = MagicMock(return_value=(
        "Search completed with 3 sources.",
        False,
    ))
    bot._remember_tool_outcome = MagicMock()

    response = bot.react_chat(
        "Search the web for the latest sqlite-vec changes."
    )

    assert response == "Here are the fresh sqlite-vec changes."
    bot._run_tool_with_spinner.assert_called_once_with(
        "search",
        "latest changes to sqlite-vec",
    )
    prepared = bot.last_tool_routing["prepared"]
    assert prepared["binding"]["mode"] == "semantic_argument"
    assert prepared["model_call"]["arguments"]["query"] == (
        "latest changes to sqlite-vec"
    )
    assert bot._remember_tool_outcome.call_args.kwargs["arguments"] == {
        "query": "latest changes to sqlite-vec",
    }


def test_native_pronoun_search_uses_persisted_verified_query():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="native",
    )
    bot.backend = "openrouter"
    bot.load_session_events([{
        "kind": "verified_tool_outcome",
        "tool": "search",
        "arguments": {"query": "latest NVLink news"},
        "succeeded": True,
        "result": "Earlier source-backed results.",
    }])
    bot._chat_with_tools = MagicMock(side_effect=[
        {
            "content": "",
            "tool_calls": [{
                "id": "call-2",
                "name": "search",
                "arg_raw": '{"query":"it"}',
            }],
            "reasoning": "",
        },
        {
            "content": "I checked that search again.",
            "tool_calls": [],
            "reasoning": "",
        },
    ])
    bot._run_tool_with_spinner = MagicMock(return_value=(
        "Fresh source-backed results.",
        False,
    ))
    bot._remember_tool_outcome = MagicMock()

    response = bot.react_chat("Search it")

    assert response == "I checked that search again."
    bot._run_tool_with_spinner.assert_called_once_with(
        "search",
        "latest NVLink news",
    )
    prepared = bot.last_tool_routing["prepared"]
    assert prepared["binding"]["mode"] == "contextual_semantic_resolution"
    assert prepared["proposed_call"]["arguments"]["query"] == "it"
    assert prepared["model_call"]["arguments"]["query"] == "latest NVLink news"


def test_canonical_followup_failure_never_hides_completed_tool_execution():
    class BadRequestError(Exception):
        pass

    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="canonical",
    )
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "tool": "search",
        "arguments": {"query": "sqlite-vec changes"},
    }))
    bot._run_tool_with_spinner = MagicMock(return_value=(
        "Search completed with 3 sources.",
        False,
    ))
    bot.chat = MagicMock(side_effect=BadRequestError("follow-up rejected"))

    response = bot.react_chat("Search for sqlite-vec changes")

    assert "Search completed with 3 sources." in response
    assert "tool already ran" in response
    assert "no tool ran" not in response.lower()
    bot._run_tool_with_spinner.assert_called_once_with(
        "search",
        "sqlite-vec changes",
    )


def test_failed_exact_search_binding_never_uses_success_fallback():
    bot = _fixture_bot(
        tool_routing_mode="direct",
        tool_routing_protocol="canonical",
    )
    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "tool": "search",
        "arguments": {"query": "invented exact query"},
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
    response = bot.react_chat("Search exactly for:")

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

    bot.raw_structured_complete = MagicMock(return_value=json.dumps({
        "mode": "literal", "tool": "yuki_write", "filename": "feelings.txt", "content": previous,
    }))
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
    assert bot.last_tool_routing["prepared"]["binding"]["mode"] == (
        "semantic_argument"
    )
    assert bot.last_tool_routing["prepared"]["binding"]["source_kind"] is None
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
        trusted_context=[],
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
