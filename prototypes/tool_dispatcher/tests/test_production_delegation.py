from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from prototypes.tool_dispatcher.__main__ import _parser
from prototypes.tool_dispatcher.backends import FixtureBackend
from prototypes.tool_dispatcher.build_production_delegation_dataset import (
    DATASET_VERSION,
    SOURCE_SHA256,
    build_dataset,
)
from prototypes.tool_dispatcher.build_production_delegation_report import (
    build_comparison,
)
from prototypes.tool_dispatcher.delegation_contract import (
    DELEGATION_DOMAINS,
    domain_tools,
    expected_domain_for_tool,
    final_tool_leakage,
    parse_delegation,
    score_delegation,
)
from prototypes.tool_dispatcher.delegation_dispatcher import (
    DelegationDispatcherRunner,
    ValidationOnlyDispatcher,
    _preservation_aware_arguments,
)
from prototypes.tool_dispatcher.delegation_mainbrain import (
    FixtureDelegationBackend,
    MainBrainDelegationRunner,
    prepare_dataset_cases,
)
from prototypes.tool_dispatcher.registry import ToolRegistry

ROOT = Path(__file__).parents[1]
PHASE2 = ROOT / "phase2-cases.json"
DERIVED = ROOT / "production-delegation-source-v1.json"


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_canonical_phase2_and_derived_dataset_are_frozen(registry):
    assert hashlib.sha256(PHASE2.read_bytes()).hexdigest() == SOURCE_SHA256
    payload = build_dataset(PHASE2)
    saved = json.loads(DERIVED.read_text(encoding="utf-8"))
    assert saved == payload
    assert payload["dataset_version"] == DATASET_VERSION
    assert len(payload["cases"]) == 450
    assert len(payload["selection_sets"]["smoke-18"]) == 18
    assert len(payload["selection_sets"]["fallback-stratified-180"]) == 180
    assert len(payload["selection_sets"]["full-450"]) == 450
    assert len(set(payload["selection_sets"]["full-450"])) == 450
    assert {case["expected_tool"] for case in payload["cases"]} == set(registry.names)


def test_derived_cases_preserve_ids_gold_and_source_exact_verbatim():
    source = {
        case["case_id"]: case for case in json.loads(PHASE2.read_text(encoding="utf-8"))
    }
    derived = json.loads(DERIVED.read_text(encoding="utf-8"))["cases"]
    for case in derived:
        original = source[case["case_id"]]
        assert case["request"] == original["request"]
        assert case["expected_tool"] == original["expected_tool"]
        assert case["expected_arguments"] == original["expected_arguments"]
        assert all(value in case["request"] for value in case["required_verbatim"])


def test_canonical_domain_mapping_uses_live_registry(registry):
    mapping = domain_tools(registry)
    assert tuple(mapping) == DELEGATION_DOMAINS
    assert mapping == {
        "information": ("time", "weather", "fetch", "search", "calc"),
        "computer": ("hardware", "read", "shell"),
        "media": ("see", "image"),
        "yuki_files": (
            "yuki_write",
            "yuki_read",
            "yuki_list",
            "yuki_delete",
            "yuki_append",
        ),
        "memory": ("remember", "recall"),
        "external_agent": ("ask_claude",),
    }
    for domain, tools in mapping.items():
        assert all(expected_domain_for_tool(tool, registry) == domain for tool in tools)


@pytest.mark.parametrize(
    ("raw", "passed"),
    [
        (
            (
                '{"request":"Check current weather conditions.",'
                '"domain_hint":"information","verbatim":["Tokyo"]}'
            ),
            True,
        ),
        ('{"request":"x","domain_hint":"information"}', False),
        ('{"request":"x","domain_hint":"wrong","verbatim":[]}', False),
        ('{"request":"x","domain_hint":"computer","verbatim":"x"}', False),
        ('[{"request":"x","domain_hint":"computer","verbatim":[]}]', False),
        ("Certainly!", False),
    ],
)
def test_delegation_parser_is_strict(raw, passed):
    parsed = parse_delegation(raw)
    assert parsed["passed"] is passed
    assert parsed["malformed"] is (not passed)


def test_leakage_detector_distinguishes_semantics_from_tool_directives(registry):
    assert (
        final_tool_leakage("Read the specified normal filesystem file.", registry.names)
        == []
    )
    assert (
        final_tool_leakage(
            "Retrieve previously stored information about the trip.", registry.names
        )
        == []
    )
    assert final_tool_leakage("Use recall to retrieve the trip.", registry.names) == [
        "recall"
    ]
    assert final_tool_leakage("Call yuki_read for notes.md.", registry.names) == [
        "yuki_read"
    ]


def test_delegation_scoring_separates_domain_semantics_and_verbatim(registry):
    parsed = parse_delegation(
        '{"request":"Retrieve previously stored information about a wedding.",'
        '"domain_hint":"memory","verbatim":["my sister\'s wedding"]}'
    )
    score = score_delegation(
        parsed,
        expected_tool="recall",
        expected_domain="memory",
        required_verbatim=["my sister's wedding"],
        live_tool_names=registry.names,
    )
    assert score["delegation_emitted"]
    assert score["domain_hint_exact"]
    assert score["semantic_request_quality"]
    assert score["verbatim_recall"] == 1.0
    assert score["verbatim_precision"] == 1.0
    assert not score["final_tool_leakage"]


def test_preservation_aware_argument_allows_explicit_source_alternative():
    result = _preservation_aware_arguments(
        "remember",
        {"fact": "user's cat is named Nori"},
        {"fact": "my cat's name is Nori"},
        {"fact": ["my cat's name is Nori"]},
    )
    assert not result["strict"]
    assert not result["execution_equivalent"]
    assert result["preservation_aware"]


def _one_case_dataset(tmp_path: Path) -> tuple[dict, Path, str]:
    payload = prepare_dataset_cases(json.loads(DERIVED.read_text(encoding="utf-8")))
    case_id = "p2-weather-01"
    payload["selection_sets"]["test-one"] = [case_id]
    path = tmp_path / "derived.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload, path, checksum


def test_mainbrain_and_dispatcher_pipeline_has_no_execution_path(
    tmp_path, registry, monkeypatch
):
    from prototypes.tool_dispatcher import executor

    def forbidden(*args, **kwargs):
        raise AssertionError("Yuki execution path was reached")

    monkeypatch.setattr(executor, "execute_call", forbidden)
    dataset, dataset_path, dataset_sha = _one_case_dataset(tmp_path)
    stage_a = MainBrainDelegationRunner(
        FixtureDelegationBackend(
            [
                (
                    '{"request":"Check current weather conditions for the supplied city.",'
                    '"domain_hint":"information","verbatim":["Tokyo"]}'
                )
            ]
        ),
        registry,
    ).run(
        dataset,
        derived_dataset_path=dataset_path,
        expected_dataset_sha256=dataset_sha,
        selection="test-one",
    )
    stage_a["configuration"].update(
        {
            "main_brain_role": "intended_main_brain",
            "surrogate": False,
            "model_label": "intended-main-brain-fixture",
        }
    )
    stage_a_path = tmp_path / "stage-a.json"
    stage_a_path.write_text(json.dumps(stage_a), encoding="utf-8")
    stage_a_sha = hashlib.sha256(stage_a_path.read_bytes()).hexdigest()

    stage_b = DelegationDispatcherRunner(
        ValidationOnlyDispatcher(
            registry,
            FixtureBackend(
                '```\n[{"name":"weather","arguments":{"city":"Tokyo"}}]',
                model_id="MadeAgents/Hammer2.1-1.5b",
            ),
        )
    ).run(
        stage_a,
        delegation_path=stage_a_path,
        expected_delegation_sha256=stage_a_sha,
    )
    case = stage_b["case_results"][0]
    assert case["strict_exact_call"]
    assert case["execution"] == {
        "capability": False,
        "executed": False,
        "status": "not_available",
    }
    assert case["request"] not in case["dispatcher"]["dispatcher_input"]
    assert stage_b["configuration"]["tool_execution_capability"] is False
    assert stage_b["configuration"]["stage_a_surrogate"] is False
    assert (
        stage_b["configuration"]["stage_a_main_brain_role"]
        == "intended_main_brain"
    )


def test_validation_only_dispatcher_exposes_no_execution_controls():
    signature = inspect.signature(ValidationOnlyDispatcher.route)
    assert tuple(signature.parameters) == (
        "self",
        "delegation",
        "raw_request",
        "gold_domain",
    )
    source = inspect.getsource(
        __import__("prototypes.tool_dispatcher.delegation_dispatcher", fromlist=["*"])
    )
    assert "from .executor" not in source
    assert "execute_call" not in source


def test_ablation_input_modes_remain_validation_only(registry):
    dispatcher = ValidationOnlyDispatcher(
        registry,
        FixtureBackend(
            '```\n[{"name":"fetch","arguments":{"url":"https://example.com"}}]',
            model_id="MadeAgents/Hammer2.1-1.5b",
        ),
        input_mode="raw_user",
        domain_source="gold",
    )
    result = dispatcher.route(
        {
            "request": "Retrieve the specified URL.",
            "domain_hint": "memory",
            "verbatim": ["https://example.com"],
        },
        raw_request="Fetch https://example.com exactly.",
        gold_domain="information",
    )
    assert result["dispatcher_input"] == "Fetch https://example.com exactly."
    assert result["domain_hint"] == "information"
    assert result["offered_tools"] == [
        "time",
        "weather",
        "fetch",
        "search",
        "calc",
    ]
    assert result["validation"]["passed"]
    assert result["execution_capability"] is False


def test_completed_production_comparison_is_checksum_locked_and_execution_free():
    comparison = build_comparison()
    assert comparison["status"] == ("completed_with_temporary_main_brain_surrogate")
    assert comparison["main_brain_is_surrogate"] is True
    assert comparison["dataset"]["selection_cases"] == 180
    assert comparison["safety"]["yuki_tool_execution_capability"] is False
    assert comparison["safety"]["yuki_tool_execution_attempts"] == 0
    assert comparison["primary"]["hammer_1_5b"]["strict_exact_call_successes"] == 132
    assert comparison["primary"]["hammer_3b"]["strict_exact_call_successes"] == 130


@pytest.mark.parametrize(
    "command",
    [
        ["delegation-mainbrain", "--execute"],
        ["delegation-dispatch", "--execute"],
    ],
)
def test_production_delegation_cli_rejects_execute_flag(command):
    with pytest.raises(SystemExit):
        _parser().parse_args(command)
