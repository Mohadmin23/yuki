from __future__ import annotations

import inspect
import json
from collections import Counter
from copy import deepcopy

import pytest

from prototypes.tool_dispatcher.__main__ import _parser
from prototypes.tool_dispatcher.argument_scoring import score_arguments
from prototypes.tool_dispatcher.automatic_benchmark import (
    AutomaticSubsetBenchmarkRunner,
    _primary_failure_stage,
    write_automatic_failure_jsonl,
)
from prototypes.tool_dispatcher.automatic_selector import AutomaticDomainSelector
from prototypes.tool_dispatcher.backends import (
    MAX_GENERATION_TOKENS,
    FixtureBackend,
    normalize_arch_decoder_text,
    normalize_hammer_decoder_text,
    normalize_xlam_decoder_text,
)
from prototypes.tool_dispatcher.benchmark import (
    HAMMER_TARGETED_CASES_PATH,
    BenchmarkCase,
    BenchmarkRunner,
    load_cases,
    write_fine_tuning_jsonl,
    write_report,
)
from prototypes.tool_dispatcher.build_phase2_dataset import (
    build_cases as build_phase2_cases,
)
from prototypes.tool_dispatcher.comparison import compare_reports, write_comparison
from prototypes.tool_dispatcher.dialects import (
    ARCH_XML_OBJECT,
    HAMMER_FENCED_ARRAY,
    OPENAI_NATIVE_TOOL_CALLS,
    TOOL_CALL_ARRAY,
    XLAM_V1_ENVELOPE,
    native_dialect_for_model,
)
from prototypes.tool_dispatcher.dispatcher import Dispatcher, parse_model_output
from prototypes.tool_dispatcher.executor import validate_strict_shell
from prototypes.tool_dispatcher.json_boundary import (
    is_complete_json_object,
    is_complete_json_value,
)
from prototypes.tool_dispatcher.oracle_subsets import (
    LOGICAL_DOMAINS,
    REMEMBER_VS_RECALL,
    TARGETED_CONFUSIONS,
)
from prototypes.tool_dispatcher.phase2_analysis import (
    build_paired_analysis,
    exact_mcnemar,
)
from prototypes.tool_dispatcher.phase2_benchmark import (
    Phase2ReliabilityRunner,
    classify_failure,
    dataset_sha256,
    wilson_interval,
)
from prototypes.tool_dispatcher.prompting import build_prompt
from prototypes.tool_dispatcher.registry import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


def test_registry_is_the_live_18_tool_registry(registry):
    assert registry.names == tuple(__import__("tools")._TOOL_NAMES)
    assert len(registry.names) == 18


def test_strict_required_fields_come_from_real_signatures(registry):
    assert registry.get("weather").argument_schema["required"] == ["city"]
    assert registry.get("weather").argument_schema["properties"]["city"]["minLength"] == 1
    assert registry.get("hardware").argument_schema["required"] == []
    assert registry.get("see").argument_schema["required"] == []
    assert registry.get("time").argument_schema["properties"] == {}
    assert all(
        registry.get(name).argument_schema["additionalProperties"] is False
        for name in registry.names
    )


def test_xlam_native_and_canonical_prompt_modes(registry):
    native = build_prompt(registry, "Weather in Tokyo", ("weather",), "native")
    canonical = build_prompt(registry, "Weather in Tokyo", ("weather",), "canonical")
    assert '"tool_calls"' in native.content
    assert '"tool":"tool_name"' in canonical.content
    assert native.schemas_sent[0]["name"] == "weather"
    assert native.schemas_sent[0]["required"] == ["city"]


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Salesforce/xLAM-2-1b-fc-r", TOOL_CALL_ARRAY),
        ("/models/xLAM-2-1b-fc-r-fp16", TOOL_CALL_ARRAY),
        ("MadeAgents/Hammer2.1-1.5b", HAMMER_FENCED_ARRAY),
        ("/models/Hammer2.1-1.5b-fp16", HAMMER_FENCED_ARRAY),
        ("MadeAgents/Hammer2.0-7b", HAMMER_FENCED_ARRAY),
        ("/models/Hammer2.0-7b-8bit", HAMMER_FENCED_ARRAY),
        ("katanemo/Arch-Function-3B", ARCH_XML_OBJECT),
        ("/models/Arch-Function-3B-fp16", ARCH_XML_OBJECT),
        ("katanemo/Arch-Agent-3B", ARCH_XML_OBJECT),
        ("/models/Arch-Agent-3B-fp16", ARCH_XML_OBJECT),
        ("ibm-granite/granite-4.1-3b", ARCH_XML_OBJECT),
        ("/models/granite-4.1-3b-fp16", ARCH_XML_OBJECT),
        ("openai/gpt-oss-20b", OPENAI_NATIVE_TOOL_CALLS),
    ],
)
def test_array_native_dialect_detection(model_id, expected):
    assert native_dialect_for_model(model_id) == expected
    assert (
        native_dialect_for_model("Salesforce/xLAM-1b-fc-r")
        == XLAM_V1_ENVELOPE
    )


def test_array_native_prompt_uses_model_chat_template_tools(registry):
    package = build_prompt(
        registry,
        "Weather in Tokyo",
        ("weather",),
        "native",
        model_id="Salesforce/xLAM-2-1b-fc-r",
    )
    assert package.native_dialect == TOOL_CALL_ARRAY
    assert package.json_root == "array"
    assert package.content == "Weather in Tokyo"
    assert package.chat_template_tools == package.schemas_sent
    assert package.schemas_sent[0]["type"] == "function"
    assert package.output_schema["type"] == "array"
    assert "Never produce multiple calls" in package.system_content


def test_gpt_oss_prompt_uses_openai_native_tool_schemas(registry):
    package = build_prompt(
        registry,
        "Weather in Tokyo",
        registry.names,
        "native",
        model_id="openai/gpt-oss-20b",
    )
    assert package.native_dialect == OPENAI_NATIVE_TOOL_CALLS
    assert package.json_root == "array"
    assert package.content == "Weather in Tokyo"
    assert package.chat_template_tools == package.schemas_sent
    assert len(package.schemas_sent) == 18
    assert all(schema["type"] == "function" for schema in package.schemas_sent)
    assert package.output_schema["type"] == "array"


def test_hammer_prompt_uses_official_client_style_json_serialization(registry):
    package = build_prompt(
        registry,
        "Weather in Tokyo",
        ("weather",),
        "native",
        model_id="MadeAgents/Hammer2.1-3b",
    )
    assert package.native_dialect == HAMMER_FENCED_ARRAY
    assert package.pre_rendered is True
    assert package.chat_template_tools is None
    assert package.system_content is None
    assert package.schemas_sent[0]["name"] == "weather"
    assert package.schemas_sent[0]["parameters"]["city"]["required"] is True
    assert '"name": "weather"' in package.content
    assert "{'name': 'weather'" not in package.content
    assert package.content.endswith("<|im_start|>assistant\n")


def test_hammer20_prompt_uses_official_single_user_message_contract(registry):
    package = build_prompt(
        registry,
        "Weather in Tokyo",
        ("weather",),
        "native",
        model_id="MadeAgents/Hammer2.0-7b",
    )
    assert package.native_dialect == HAMMER_FENCED_ARRAY
    assert package.pre_rendered is False
    assert package.chat_template_tools is None
    assert package.system_content is None
    assert package.schemas_sent[0]["name"] == "weather"
    assert package.schemas_sent[0]["parameters"]["city"]["required"] is True
    assert "[BEGIN OF QUERY]\nWeather in Tokyo\n[END OF QUERY]" in package.content
    assert "<|im_start|>" not in package.content


def test_arch_native_prompt_uses_official_xml_tool_contract(registry):
    package = build_prompt(
        registry,
        "Weather in Tokyo",
        ("weather", "search"),
        "native",
        model_id="katanemo/Arch-Function-3B",
    )
    assert package.native_dialect == ARCH_XML_OBJECT
    assert package.content == "Weather in Tokyo"
    assert package.pre_rendered is False
    assert package.chat_template_tools is None
    assert "<tools>" in package.system_content
    assert "</tools>" in package.system_content
    assert "<tool_call>" in package.system_content
    assert '"name": "weather"' in package.system_content
    assert package.output_schema["oneOf"][0]["properties"]["name"] == {
        "const": "weather"
    }


def test_arch_agent_native_prompt_uses_official_xml_tool_contract(registry):
    package = build_prompt(
        registry,
        "Weather in Tokyo",
        ("weather", "search"),
        "native",
        model_id="katanemo/Arch-Agent-3B",
    )
    assert package.native_dialect == ARCH_XML_OBJECT
    assert package.content == "Weather in Tokyo"
    assert package.pre_rendered is False
    assert package.chat_template_tools is None
    assert "<tools>" in package.system_content
    assert "<tool_call>" in package.system_content


def test_granite_native_prompt_uses_official_xml_tool_contract(registry):
    package = build_prompt(
        registry,
        "Weather in Tokyo",
        ("weather", "search"),
        "native",
        model_id="ibm-granite/granite-4.1-3b",
    )
    assert package.native_dialect == ARCH_XML_OBJECT
    assert package.content == "Weather in Tokyo"
    assert package.pre_rendered is False
    assert package.chat_template_tools is None
    assert "<tools>" in package.system_content
    assert "<tool_call>" in package.system_content


def test_generation_ceiling_is_256():
    assert MAX_GENERATION_TOKENS == 256


@pytest.mark.parametrize(
    ("tool", "expected", "actual"),
    [
        ("weather", {"city": "New York"}, {"city": "new   york"}),
        (
            "search",
            {"query": "latest qwen release"},
            {"query": "latest Qwen release"},
        ),
        (
            "ask_claude",
            {"question": "check this error"},
            {"question": "Check this error"},
        ),
        (
            "calc",
            {"expression": "(144 / 12) + 7"},
            {"expression": "144 / 12 + 7"},
        ),
    ],
)
def test_execution_equivalent_argument_rules_accept_safe_normalization(
    tool, expected, actual
):
    score = score_arguments(tool, expected, actual)
    assert score["strict"] is False
    assert score["execution_equivalent"] is True


@pytest.mark.parametrize(
    ("tool", "expected", "actual"),
    [
        ("read", {"filepath": "/Users/me/foo.txt"}, {"filepath": "/Users/me/Foo.txt"}),
        ("yuki_read", {"filename": "Notes.md"}, {"filename": "notes.md"}),
        ("shell", {"command": "ls -la"}, {"command": "ls  -la"}),
        (
            "yuki_write",
            {"filename_and_content": "todo.txt|Call Sam"},
            {"filename_and_content": "todo.txt|call Sam"},
        ),
        (
            "search",
            {"query": "latest MLX documentation"},
            {"query": "MLX documentation"},
        ),
    ],
)
def test_execution_equivalent_argument_rules_preserve_significant_changes(
    tool, expected, actual
):
    score = score_arguments(tool, expected, actual)
    assert score["strict"] is False
    assert score["execution_equivalent"] is False


def test_backend_marks_first_and_warm_runs(registry):
    backend = FixtureBackend(
        ['{"tool_calls":[{"name":"time","arguments":{}}]}'] * 2
    )
    dispatcher = Dispatcher(registry, backend)
    first = dispatcher.dispatch("What time is it?")
    second = dispatcher.dispatch("What time is it?")
    assert first["generation"]["extra"]["run_kind"] == "first_after_load"
    assert second["generation"]["extra"]["run_kind"] == "warm"


def test_xlam_newline_marker_normalization_is_narrow():
    marked = 'ĊĊ{"tool_calls":[{"name":"search",Ġ"arguments":{"query":"weatherĠAPIs"}}]}'
    normalized = '\n\n{"tool_calls":[{"name":"search", "arguments":{"query":"weather APIs"}}]}'
    assert normalize_xlam_decoder_text("Salesforce/xLAM-1b-fc-r", marked) == normalized
    assert normalize_xlam_decoder_text("another-model", marked) == marked


def test_hammer_fence_normalization_is_exact_and_model_specific():
    fenced = '```\n[{"name":"time","arguments":{}}]\n```'
    expected = '[{"name":"time","arguments":{}}]'
    assert normalize_hammer_decoder_text("MadeAgents/Hammer2.1-1.5b", fenced) == expected
    assert normalize_hammer_decoder_text("another-model", fenced) == fenced
    assert (
        normalize_hammer_decoder_text(
            "MadeAgents/Hammer2.1-1.5b", f"Sure\n{fenced}"
        )
        == f"Sure\n{fenced}"
    )


def test_hammer_stream_can_stop_after_array_before_closing_fence():
    partial_native = '```\n[{"name":"time","arguments":{}}]'
    normalized = normalize_hammer_decoder_text(
        "MadeAgents/Hammer2.1-1.5b", partial_native
    )
    assert is_complete_json_value(normalized, "array")


def test_arch_normalization_removes_only_native_xml_wrapper():
    wrapped = (
        '<tool_call>\n{"name":"weather","arguments":{"city":"Tokyo"}}'
        "\n</tool_call>"
    )
    expected = '{"name":"weather","arguments":{"city":"Tokyo"}}'
    assert normalize_arch_decoder_text(
        "katanemo/Arch-Function-3B", wrapped
    ) == expected
    prose = f"Sure\n{wrapped}"
    assert normalize_arch_decoder_text("katanemo/Arch-Function-3B", prose) == prose


def test_arch_agent_normalization_uses_same_native_xml_wrapper():
    wrapped = (
        '<tool_call>\n{"name":"weather","arguments":{"city":"Tokyo"}}'
        "\n</tool_call>"
    )
    expected = '{"name":"weather","arguments":{"city":"Tokyo"}}'
    assert normalize_arch_decoder_text("katanemo/Arch-Agent-3B", wrapped) == expected


def test_granite_normalization_uses_same_native_xml_wrapper():
    wrapped = (
        '<tool_call>\n{"name":"weather","arguments":{"city":"Tokyo"}}'
        "\n</tool_call>"
    )
    expected = '{"name":"weather","arguments":{"city":"Tokyo"}}'
    assert normalize_arch_decoder_text("ibm-granite/granite-4.1-3b", wrapped) == expected


def test_arch_stream_can_stop_after_wrapped_object():
    partial_native = '<tool_call>\n{"name":"time","arguments":{}}'
    normalized = normalize_arch_decoder_text(
        "katanemo/Arch-Function-3B", partial_native
    )
    assert is_complete_json_value(normalized, "object")


def test_arch_native_call_parses_to_canonical_call():
    result = parse_model_output(
        '{"name":"weather","arguments":{"city":"Tokyo"}}',
        "native",
        ARCH_XML_OBJECT,
    )
    assert result["passed"] is True
    assert result["canonical_call"] == {
        "tool": "weather",
        "arguments": {"city": "Tokyo"},
    }


def test_native_no_argument_call_can_omit_arguments(registry):
    parsed = parse_model_output(
        '{"tool_calls":[{"name":"yuki_list"}]}', "native"
    )
    assert parsed["canonical_call"] == {"tool": "yuki_list", "arguments": {}}
    assert registry.validate_call(parsed["canonical_call"], registry.names)["passed"]


def test_missing_native_arguments_still_fail_when_tool_requires_them(registry):
    parsed = parse_model_output('{"tool_calls":[{"name":"weather"}]}', "native")
    validation = registry.validate_call(parsed["canonical_call"], registry.names)
    assert validation["passed"] is False
    assert validation["errors"] == ["Missing required argument: city"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"tool_calls":[]}', True),
        ('  {"tool":"time","arguments":{}}\n', True),
        ('Sure {"tool_calls":[]}', False),
        ('{"tool_calls":[]} trailing', False),
        ('{"tool_calls":[', False),
    ],
)
def test_complete_json_boundary_is_strict(text, expected):
    assert is_complete_json_object(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('[{"name":"time","arguments":{}}]', True),
        ("[]", True),
        ('```json\n[{"name":"time","arguments":{}}]\n```', False),
        ('[{"name":"time","arguments":{}}] trailing', False),
        ('{"tool_calls":[]}', False),
    ],
)
def test_complete_json_array_boundary_is_strict(text, expected):
    assert is_complete_json_value(text, "array") is expected


def test_native_output_normalizes_to_canonical_call():
    parsed = parse_model_output(
        '{"tool_calls":[{"name":"weather","arguments":{"city":"Tokyo"}}]}',
        "native",
    )
    assert parsed["passed"] is True
    assert parsed["canonical_call"] == {
        "tool": "weather",
        "arguments": {"city": "Tokyo"},
    }


def test_array_native_output_normalizes_to_canonical_call():
    parsed = parse_model_output(
        '[{"name":"weather","arguments":{"city":"Tokyo"}}]',
        "native",
        TOOL_CALL_ARRAY,
    )
    assert parsed["passed"] is True
    assert parsed["canonical_call"] == {
        "tool": "weather",
        "arguments": {"city": "Tokyo"},
    }


def test_openai_native_output_normalizes_to_canonical_call():
    parsed = parse_model_output(
        '[{"name":"weather","arguments":{"city":"Tokyo"}}]',
        "native",
        OPENAI_NATIVE_TOOL_CALLS,
    )
    assert parsed["passed"] is True
    assert parsed["canonical_call"] == {
        "tool": "weather",
        "arguments": {"city": "Tokyo"},
    }


def test_array_native_reject_and_fences_are_distinct():
    rejected = parse_model_output("[]", "native", TOOL_CALL_ARRAY)
    fenced = parse_model_output(
        '```\n[{"name":"time","arguments":{}}]\n```',
        "native",
        TOOL_CALL_ARRAY,
    )
    assert rejected["rejected"] is True
    assert rejected["malformed"] is False
    assert fenced["malformed"] is True


def test_dispatcher_selects_array_dialect_from_backend_model(registry):
    backend = FixtureBackend(
        '[{"name":"time","arguments":{}}]',
        model_id="MadeAgents/Hammer2.1-1.5b",
    )
    result = Dispatcher(registry, backend).dispatch("What time is it?")
    assert result["native_dialect"] == HAMMER_FENCED_ARRAY
    assert result["generation"]["extra"]["json_root"] == "array"
    assert result["canonical_call"] == {"tool": "time", "arguments": {}}


def test_native_reject_and_multiple_calls_are_distinct():
    rejected = parse_model_output('{"tool_calls":[]}', "native")
    multiple = parse_model_output(
        '{"tool_calls":[{"name":"time","arguments":{}},{"name":"time","arguments":{}}]}',
        "native",
    )
    assert rejected["rejected"] is True
    assert rejected["malformed"] is False
    assert multiple["malformed"] is True


def test_filler_text_is_malformed():
    parsed = parse_model_output(
        'Sure! {"tool_calls":[{"name":"time","arguments":{}}]}', "native"
    )
    assert parsed["passed"] is False
    assert parsed["malformed"] is True


def test_validator_rejects_missing_extra_and_wrong_type(registry):
    available = registry.names
    assert not registry.validate_call(
        {"tool": "weather", "arguments": {}}, available
    )["passed"]
    assert not registry.validate_call(
        {"tool": "weather", "arguments": {"city": "Tokyo", "unit": "C"}},
        available,
    )["passed"]
    assert not registry.validate_call(
        {"tool": "weather", "arguments": {"city": 12}}, available
    )["passed"]


def test_dispatcher_does_one_generation_and_does_not_execute_by_default(registry):
    backend = FixtureBackend(
        '{"tool_calls":[{"name":"calc","arguments":{"expression":"2+2"}}]}'
    )
    result = Dispatcher(registry, backend).dispatch("Calculate 2+2")
    assert result["generation_count"] == 1
    assert result["stateless"] is True
    assert result["validation"]["passed"] is True
    assert result["execution"]["status"] == "not_requested"


def test_direct_execution_bypasses_generic_arg_path(registry):
    backend = FixtureBackend(
        '{"tool_calls":[{"name":"calc","arguments":{"expression":"2+2"}}]}'
    )
    result = Dispatcher(registry, backend).dispatch("Calculate 2+2", execute=True)
    assert result["execution"]["passed"] is True
    assert result["execution"]["raw_output"] == "4"


@pytest.mark.parametrize(
    "command",
    [
        "ls; whoami",
        "ls | cat",
        "echo $(whoami)",
        "echo `whoami`",
        "cat /tmp/*",
        "ls && pwd",
        "python -c pass",
        "echo 'hello'",
        "ls\npwd",
    ],
)
def test_strict_shell_rejects_bypass_shapes(command):
    assert validate_strict_shell(command)


def test_shell_dry_run_never_calls_real_shell(registry):
    backend = FixtureBackend(
        '{"tool_calls":[{"name":"shell","arguments":{"command":"pwd"}}]}'
    )
    result = Dispatcher(registry, backend).dispatch(
        "Run pwd", execute=True, allow_side_effects=False, shell_mode="dry_run"
    )
    assert result["execution"]["status"] == "dry_run"
    assert result["execution"]["executed"] is False


def test_phase_one_dataset_has_four_cases_for_every_live_tool(registry):
    cases = load_cases()
    counts = Counter(case.expected_tool for case in cases)
    assert len(cases) == 72
    assert set(counts) == set(registry.names)
    assert set(counts.values()) == {4}


def test_hammer_targeted_dataset_has_six_balanced_failure_families():
    cases = load_cases(HAMMER_TARGETED_CASES_PATH)
    families = Counter(case.tags[1] for case in cases)
    assert len(cases) == 96
    assert len({case.case_id for case in cases}) == 96
    assert set(families.values()) == {16}
    assert set(families) == {
        "weather-routing",
        "see-camera-routing",
        "read-vs-yuki-read",
        "search-preservation",
        "filepath-preservation",
        "ask-claude-preservation",
    }


def test_automatic_selector_preserves_expected_tool_for_existing_phase_one_cases(
    registry,
):
    selector = AutomaticDomainSelector(registry)
    cases = [*load_cases(), *load_cases(HAMMER_TARGETED_CASES_PATH)]

    assert all(
        case.expected_tool in selector.select_domains(case.request).tools
        for case in cases
    )


@pytest.mark.parametrize(
    ("input_text", "expected_domains"),
    [
        ("Search the web for read vs yuki_read routing.", ("information",)),
        ("Consult Claude: inspect this parser error.", ("external_agent",)),
        ("Use vision to check the plant.", ("media",)),
        ("Read /tmp/notes.txt.", ("computer",)),
        ("Read notes.md from Yuki's folder.", ("yuki_files",)),
        ("Append one line to notes.md.", ("computer", "yuki_files")),
    ],
)
def test_automatic_selector_domain_rules(registry, input_text, expected_domains):
    selection = AutomaticDomainSelector(registry).select_domains(input_text)
    assert selection.domains == expected_domains


def test_automatic_benchmark_has_no_execution_controls_or_execution_path(
    registry, monkeypatch
):
    import prototypes.tool_dispatcher.dispatcher as dispatcher_module

    def execution_is_forbidden(*args, **kwargs):
        raise AssertionError("automatic benchmark reached Yuki tool execution")

    monkeypatch.setattr(dispatcher_module, "execute_call", execution_is_forbidden)
    monkeypatch.setattr(Dispatcher, "execute_validated", execution_is_forbidden)
    signature = inspect.signature(AutomaticSubsetBenchmarkRunner.run)
    assert "execute" not in signature.parameters
    assert "allow_side_effects" not in signature.parameters
    assert "shell_mode" not in signature.parameters

    backend = FixtureBackend(
        '[{"name":"weather","arguments":{"city":"Tokyo"}}]',
        model_id="MadeAgents/Hammer2.1-1.5b",
    )
    report = AutomaticSubsetBenchmarkRunner(
        Dispatcher(registry, backend)
    ).run(
        [BenchmarkCase("What's the weather in Tokyo?", "weather", {"city": "Tokyo"})],
        dataset="safety-test",
    )

    assert report["configuration"]["execution_capability"] is False
    assert report["configuration"]["execute"] is False
    assert report["case_results"][0]["execution"]["executed"] is False
    assert report["case_results"][0]["execution"]["status"] == "not_requested"


def test_automatic_benchmark_reports_selector_dispatcher_and_end_to_end_metrics(
    registry,
):
    backend = FixtureBackend(
        [
            '[{"name":"weather","arguments":{"city":"Tokyo"}}]',
            '[{"name":"yuki_read","arguments":{"filename":"notes.md"}}]',
        ],
        model_id="MadeAgents/Hammer2.1-1.5b",
    )
    cases = [
        BenchmarkCase("Weather in Tokyo", "weather", {"city": "Tokyo"}),
        BenchmarkCase(
            "Read notes.md from Yuki's folder",
            "yuki_read",
            {"filename": "notes.md"},
        ),
    ]
    report = AutomaticSubsetBenchmarkRunner(
        Dispatcher(registry, backend)
    ).run(cases, dataset="fixture")

    assert report["selector_metrics"]["correct_tool_in_selected_subset_rate"] == 1.0
    assert report["dispatcher_metrics"]["strict_exact_call_accuracy"] == 1.0
    assert report["end_to_end_metrics"]["strict_exact_call_accuracy"] == 1.0
    assert report["end_to_end_metrics"]["schema_valid_successful_call_rate"] == 1.0
    assert all(case["failure_stage"] is None for case in report["case_results"])


def test_automatic_command_rejects_execution_flag():
    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["automatic-benchmark", "--dataset", "fixture", "--execute"]
        )


@pytest.mark.parametrize(
    ("overrides", "expected_stage"),
    [
        ({"correct_tool_available": False}, "tool_unavailable"),
        ({"malformed": True}, "malformed_output"),
        ({"expected_tool_selected": False}, "tool_selection"),
        ({"schema_valid": False}, "schema_validation"),
        ({"equivalent_arguments_correct": False}, "argument_generation"),
        ({"exact_domain": False}, "domain_selection"),
    ],
)
def test_automatic_failure_stage_taxonomy(overrides, expected_stage):
    values = {
        "correct_tool_available": True,
        "malformed": False,
        "expected_tool_selected": True,
        "schema_valid": True,
        "equivalent_arguments_correct": True,
        "exact_domain": True,
    }
    values.update(overrides)
    assert _primary_failure_stage(**values) == expected_stage


def test_automatic_failure_writer_preserves_routing_diagnostics(tmp_path):
    report = {"failure_records": [{"failure_stage": "tool_unavailable"}]}
    path = tmp_path / "automatic" / "failures.jsonl"
    write_automatic_failure_jsonl(report, path)
    assert json.loads(path.read_text()) == report["failure_records"][0]


def test_frozen_phase2_dataset_contract(registry):
    cases = build_phase2_cases()
    tool_counts = Counter(case["expected_tool"] for case in cases)
    category_counts = Counter(case["tags"][1] for case in cases)

    assert len(cases) == 450
    assert set(tool_counts) == set(registry.names)
    assert set(tool_counts.values()) == {25}
    assert category_counts == {
        "normal": 202,
        "paraphrase": 110,
        "confusion": 72,
        "preservation": 48,
        "adversarial": 18,
    }
    assert len({case["case_id"] for case in cases}) == 450
    assert len({case["request"] for case in cases}) == 450


def test_frozen_phase2_checksum_and_selector_baseline(registry):
    path = "prototypes/tool_dispatcher/phase2-cases.json"
    assert dataset_sha256(path) == (
        "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466"
    )
    selector = AutomaticDomainSelector(registry)
    cases = load_cases(path)
    coverage = sum(
        case.expected_tool in selector.select_domains(case.request).tools
        for case in cases
    )
    exact_domains = sum(
        selector.select_domains(case.request).domains
        == (selector.expected_domain_for_tool(case.expected_tool),)
        for case in cases
    )
    assert coverage == 321
    assert exact_domains == 291


def test_phase2_runner_is_checksum_locked_and_cannot_execute(
    registry, monkeypatch, tmp_path
):
    import prototypes.tool_dispatcher.dispatcher as dispatcher_module

    def execution_is_forbidden(*args, **kwargs):
        raise AssertionError("Phase 2 reached Yuki tool execution")

    monkeypatch.setattr(dispatcher_module, "execute_call", execution_is_forbidden)
    monkeypatch.setattr(Dispatcher, "execute_validated", execution_is_forbidden)
    signature = inspect.signature(Phase2ReliabilityRunner.run)
    assert "execute" not in signature.parameters
    assert "allow_side_effects" not in signature.parameters
    assert "shell_mode" not in signature.parameters

    data = [
        {
            "case_id": "p2-fixture-time",
            "request": "What time is it?",
            "expected_tool": "time",
            "expected_arguments": {},
            "tags": ["phase2", "normal", "time"],
        }
    ]
    path = tmp_path / "phase2.json"
    path.write_text(json.dumps(data) + "\n")
    checksum = dataset_sha256(path)
    # Fixture output uses Hammer's real fenced-array normalization contract.
    backend = FixtureBackend(
        '```\n[{"name":"time","arguments":{}}]\n```',
        model_id="MadeAgents/Hammer2.1-3b",
    )
    report = Phase2ReliabilityRunner(Dispatcher(registry, backend)).run(
        [BenchmarkCase.from_dict(data[0])],
        dataset_path=path,
        expected_checksum=checksum,
    )
    assert report["configuration"]["execution_capability"] is False
    assert report["configuration"]["execute"] is False
    assert report["case_results"][0]["execution"]["executed"] is False
    assert report["core_metrics"]["strict_exact_call_accuracy"] == 1.0

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        Phase2ReliabilityRunner(
            Dispatcher(
                registry,
                FixtureBackend(
                    '```\n[{"name":"time","arguments":{}}]\n```',
                    model_id="MadeAgents/Hammer2.1-3b",
                ),
            )
        ).run(
            [BenchmarkCase.from_dict(data[0])],
            dataset_path=path,
            expected_checksum="0" * 64,
        )


def test_phase2_command_rejects_execution_flag():
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "phase2-benchmark",
                "--cases",
                "phase2-cases.json",
                "--dataset-sha256",
                "0" * 64,
                "--model",
                "fixture",
                "--output",
                "report.json",
                "--failures-output",
                "failures.jsonl",
                "--execute",
            ]
        )


def _phase2_taxonomy_case():
    return {
        "malformed": False,
        "generation": {"raw_text": '[{"name":"weather","arguments":{}}]'},
        "validation": {"passed": True, "errors": []},
        "parse": {"errors": []},
        "actual_tool": "weather",
        "expected_tool": "weather",
        "tool_correct": True,
        "correct_tool_in_selected_subset": True,
        "schema_valid": True,
        "strict_arguments_correct": True,
        "execution_equivalent_arguments_correct": True,
        "exact_domain_selection": True,
    }


@pytest.mark.parametrize(
    ("changes", "primary"),
    [
        ({"correct_tool_in_selected_subset": False}, "selector_tool_unavailable"),
        (
            {
                "malformed": True,
                "parse": {"errors": ["Native output must contain exactly one tool call."]},
                "actual_tool": None,
                "tool_correct": False,
            },
            "multiple_calls",
        ),
        (
            {
                "malformed": True,
                "generation": {"raw_text": "Sure, I can help."},
                "actual_tool": None,
                "tool_correct": False,
            },
            "conversational_output",
        ),
        (
            {
                "actual_tool": "invented_tool",
                "tool_correct": False,
                "schema_valid": False,
                "validation": {"passed": False, "errors": ["Unknown tool"]},
            },
            "nonexistent_tool",
        ),
        ({"actual_tool": "search", "tool_correct": False}, "wrong_tool"),
        (
            {
                "schema_valid": False,
                "validation": {
                    "passed": False,
                    "errors": ["Missing required argument: city"],
                },
            },
            "missing_argument",
        ),
        (
            {
                "schema_valid": False,
                "validation": {
                    "passed": False,
                    "errors": ["Unexpected arguments: ['unit']"],
                },
            },
            "extra_argument",
        ),
        (
            {
                "schema_valid": False,
                "validation": {"passed": False, "errors": ["Wrong type"]},
            },
            "schema_invalid",
        ),
        (
            {
                "strict_arguments_correct": False,
                "execution_equivalent_arguments_correct": True,
            },
            "argument_rewrite",
        ),
        (
            {
                "strict_arguments_correct": False,
                "execution_equivalent_arguments_correct": False,
            },
            "wrong_argument",
        ),
    ],
)
def test_phase2_failure_taxonomy(changes, primary):
    case = deepcopy(_phase2_taxonomy_case())
    case.update(changes)
    assert classify_failure(case, ("weather", "search"))["primary"] == primary


def test_wilson_interval_and_exact_mcnemar_are_deterministic():
    interval = wilson_interval(50, 100)
    assert interval["lower"] == pytest.approx(0.4038315)
    assert interval["upper"] == pytest.approx(0.5961685)
    assert exact_mcnemar(8, 2)["p_value"] == pytest.approx(0.109375)


def test_phase2_paired_analysis_uses_identical_cases_and_selector():
    def case(case_id, strict, equivalent, tool, actual_tool):
        return {
            "case_id": case_id,
            "request": f"request {case_id}",
            "expected_tool": "time",
            "expected_arguments": {},
            "selected_domains": ["information"],
            "offered_tools": ["time", "weather", "fetch", "search", "calc"],
            "end_to_end_strict_exact_call": strict,
            "end_to_end_execution_equivalent_exact_call": equivalent,
            "tool_correct": tool,
            "actual_tool": actual_tool,
            "normalized_call": {"tool": actual_tool, "arguments": {}},
            "malformed": False,
            "parse": {"rejected": False},
            "validation": {"passed": True, "errors": []},
        }

    hammer = {
        "configuration": {"dataset_sha256": "abc", "model_id": "hammer"},
        "case_results": [
            case("1", True, True, True, "time"),
            case("2", True, True, True, "time"),
            case("3", False, False, False, "weather"),
            case("4", False, False, False, "weather"),
        ],
    }
    arch = {
        "configuration": {"dataset_sha256": "abc", "model_id": "arch"},
        "case_results": [
            case("1", True, True, True, "time"),
            case("2", False, False, False, "weather"),
            case("3", True, True, True, "time"),
            case("4", False, False, False, "search"),
        ],
    }
    analysis = build_paired_analysis(hammer, arch)
    counts = analysis["dimensions"]["strict_exact_call"]["counts"]
    assert counts == {
        "both_correct": 1,
        "hammer_only_correct": 1,
        "arch_agent_only_correct": 1,
        "both_wrong_same_way": 0,
        "both_wrong_differently": 1,
    }


def test_logical_oracle_offers_expected_domain_per_case(registry):
    generations = [
        '{"tool_calls":[{"name":"weather","arguments":{"city":"Tokyo"}}]}',
        '{"tool_calls":[{"name":"see","arguments":{"target":"camera"}}]}',
    ]
    cases = [
        BenchmarkCase("Weather Tokyo", "weather", {"city": "Tokyo"}),
        BenchmarkCase("Use the camera", "see", {"target": "camera"}),
    ]
    report = BenchmarkRunner(
        Dispatcher(registry, FixtureBackend(generations))
    ).run(cases, oracle_profile=LOGICAL_DOMAINS)

    assert report["configuration"]["oracle_profile"] == LOGICAL_DOMAINS
    assert report["case_results"][0]["oracle_subset"] == "information"
    assert report["case_results"][0]["offered_tools"] == [
        "time",
        "weather",
        "fetch",
        "search",
        "calc",
    ]
    assert report["case_results"][1]["offered_tools"] == ["see", "image"]


def test_confusion_oracle_uses_existing_targeted_tags(registry):
    cases = [
        BenchmarkCase(
            "Read my Yuki note",
            "yuki_read",
            {"filename": "note.txt"},
            tags=("targeted", "read-vs-yuki-read"),
        ),
        BenchmarkCase(
            "Unrelated memory case",
            "remember",
            {"fact": "blue"},
            tags=("targeted", "memory-other"),
        ),
    ]
    backend = FixtureBackend(
        '{"tool_calls":[{"name":"yuki_read","arguments":{"filename":"note.txt"}}]}'
    )
    report = BenchmarkRunner(Dispatcher(registry, backend)).run(
        cases, oracle_profile=TARGETED_CONFUSIONS
    )

    assert report["metrics"]["total_cases"] == 1
    assert report["case_results"][0]["offered_tools"] == [
        "read",
        "yuki_read",
    ]


def test_memory_confusion_oracle_filters_existing_broad_cases(registry):
    cases = [
        BenchmarkCase("Remember blue", "remember", {"fact": "blue"}),
        BenchmarkCase("What time?", "time", {}),
    ]
    backend = FixtureBackend(
        '{"tool_calls":[{"name":"remember","arguments":{"fact":"blue"}}]}'
    )
    report = BenchmarkRunner(Dispatcher(registry, backend)).run(
        cases, oracle_profile=REMEMBER_VS_RECALL
    )

    assert report["metrics"]["total_cases"] == 1
    assert report["case_results"][0]["offered_tools"] == [
        "remember",
        "recall",
    ]


def test_benchmark_metrics_and_execution_separation(registry):
    backend = FixtureBackend(
        [
            '{"tool_calls":[{"name":"weather","arguments":{"city":"Tokyo"}}]}',
            '{"tool_calls":[{"name":"search","arguments":{"query":"Paris"}}]}',
        ]
    )
    runner = BenchmarkRunner(Dispatcher(registry, backend))
    report = runner.run(
        [
            BenchmarkCase("Weather Tokyo", "weather", {"city": "Tokyo"}),
            BenchmarkCase("Weather Paris", "weather", {"city": "Paris"}),
        ]
    )
    metrics = report["metrics"]
    assert metrics["tool_selection_accuracy"] == 0.5
    assert metrics["argument_accuracy"] == 0.5
    assert metrics["strict_argument_accuracy"] == 0.5
    assert metrics["execution_equivalent_argument_accuracy"] == 0.5
    assert metrics["exact_call_accuracy"] == 0.5
    assert metrics["p50_inference_latency_ms"] == 1.0
    assert metrics["p95_inference_latency_ms"] == 1.0
    assert metrics["execution_attempts"] == 0
    assert report["fine_tuning_examples"][0]["failure_types"] == [
        "wrong_tool",
        "wrong_arguments",
        "execution_non_equivalent_arguments",
    ]


def test_phase_two_classification_has_no_small_dataset_ceiling(registry):
    generation = '{"tool_calls":[{"name":"time","arguments":{}}]}'
    backend = FixtureBackend([generation] * 300)
    cases = [BenchmarkCase(f"Time {index}", "time", {}) for index in range(300)]
    report = BenchmarkRunner(Dispatcher(registry, backend)).run(cases)
    assert report["metrics"]["total_cases"] == 300
    assert report["configuration"]["phase"] == "phase_2_reliability"


def test_report_writers_create_missing_parent_directories(tmp_path):
    report = {
        "fine_tuning_examples": [{"request": "Weather in Tokyo"}],
    }
    report_path = tmp_path / "new" / "reports" / "benchmark.json"
    failures_path = tmp_path / "new" / "failures" / "cases.jsonl"

    write_report(report, report_path)
    write_fine_tuning_jsonl(report, failures_path)

    assert json.loads(report_path.read_text()) == report
    assert failures_path.read_text().splitlines() == [
        json.dumps(report["fine_tuning_examples"][0])
    ]

    comparison_path = tmp_path / "new" / "comparisons" / "models.json"
    write_comparison({"models": []}, comparison_path)
    assert json.loads(comparison_path.read_text()) == {"models": []}


def test_report_comparison_verifies_cases_and_uses_robust_throughput(tmp_path):
    def report(model_id, exact, rate):
        return {
            "configuration": {"model_id": model_id},
            "metrics": {
                "total_cases": 1,
                "tool_selection_accuracy": exact,
                "argument_accuracy": exact,
                "exact_call_accuracy": exact,
                "schema_valid_output_rate": 1.0,
                "malformed_output_rate": 0.0,
                "average_inference_latency_ms": 10.0,
                "end_to_end_generated_tokens_per_second": 2.0,
            },
            "case_results": [
                {
                    "case_id": "time-01",
                    "request": "What time is it?",
                    "expected_tool": "time",
                    "expected_arguments": {},
                    "tags": ["clear"],
                    "actual_tool": "time" if exact else None,
                    "actual_arguments": {} if exact else None,
                    "schema_valid": bool(exact),
                    "exact_call": bool(exact),
                    "generation": {
                        "model_id": model_id,
                        "tokens_per_second": rate,
                        "generated_tokens": 10,
                        "finish_reason": "json_complete",
                        "peak_memory_gb": 3.0,
                        "process_rss_mb": 2000.0,
                        "normalized_text": None,
                    },
                }
            ],
        }

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(report("Salesforce/xLAM-1b-fc-r", 0.0, 20.0)))
    second.write_text(
        json.dumps(report("MadeAgents/Hammer2.1-1.5b", 1.0, 18.0))
    )
    comparison = compare_reports([first, second])
    assert comparison["comparison"]["same_cases_verified"] is True
    assert comparison["ranking_by_exact_call_accuracy"] == [
        "Hammer2.1-1.5B",
        "xLAM-1b-fc-r",
    ]
    assert comparison["models"][0][
        "aggregate_generation_tokens_per_second"
    ] == pytest.approx(20.0)
    assert comparison["models"][0]["metrics_by_tag"]["clear"][
        "total_cases"
    ] == 1
