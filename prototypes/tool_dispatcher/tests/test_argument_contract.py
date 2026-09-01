from __future__ import annotations

import hashlib
import json

import pytest

from prototypes.tool_dispatcher.argument_contract import (
    CONTRACT_VERSION,
    LITERAL_SOURCE,
    NO_ARGUMENT,
    SEMANTIC_ARGUMENT,
    contract_manifest,
    score_contract_arguments,
    tool_argument_contract,
    validate_contract_against_registry,
)
from prototypes.tool_dispatcher.backends import FixtureBackend
from prototypes.tool_dispatcher.benchmark import (
    BenchmarkCase,
    BenchmarkRunner,
    load_contract_annotations,
)
from prototypes.tool_dispatcher.dispatcher import Dispatcher
from prototypes.tool_dispatcher.registry import ToolRegistry


def test_contract_covers_the_live_registry_and_real_argument_names() -> None:
    registry = ToolRegistry()
    validate_contract_against_registry(registry)
    assert set(contract_manifest()["tools"]) == set(registry.names)


@pytest.mark.parametrize(
    ("tool", "mode"),
    [
        ("time", NO_ARGUMENT),
        ("yuki_list", NO_ARGUMENT),
        ("fetch", LITERAL_SOURCE),
        ("search", LITERAL_SOURCE),
        ("image", LITERAL_SOURCE),
        ("read", LITERAL_SOURCE),
        ("shell", LITERAL_SOURCE),
        ("yuki_write", LITERAL_SOURCE),
        ("yuki_read", LITERAL_SOURCE),
        ("yuki_delete", LITERAL_SOURCE),
        ("yuki_append", LITERAL_SOURCE),
        ("ask_claude", LITERAL_SOURCE),
        ("weather", SEMANTIC_ARGUMENT),
        ("calc", SEMANTIC_ARGUMENT),
        ("hardware", SEMANTIC_ARGUMENT),
        ("see", SEMANTIC_ARGUMENT),
        ("remember", SEMANTIC_ARGUMENT),
        ("recall", SEMANTIC_ARGUMENT),
    ],
)
def test_every_tool_has_the_expected_transport_mode(tool: str, mode: str) -> None:
    assert tool_argument_contract(tool)["mode"] == mode


def test_literal_field_is_unscoreable_without_independent_reference() -> None:
    score = score_contract_arguments(
        "search",
        {"query": "latest MLX documentation"},
        {"query": "latest MLX documentation"},
    )
    assert score["scoreable"] is False
    assert score["contract_correct"] is None
    assert "official semantic gold" in score["reason"]


def test_literal_field_requires_character_exact_reference() -> None:
    correct = score_contract_arguments(
        "image",
        {"prompt": "red fox"},
        {"prompt": "a red fox"},
        literal_references={"prompt": ["a red fox"]},
    )
    rewritten = score_contract_arguments(
        "image",
        {"prompt": "red fox"},
        {"prompt": "red fox"},
        literal_references={"prompt": ["a red fox"]},
    )
    assert correct["contract_correct"] is True
    assert rewritten["contract_correct"] is False


@pytest.mark.parametrize(
    ("tool", "expected", "actual"),
    [
        ("see", {"target": "keys"}, {"target": "my keys"}),
        ("recall", {"topic": "garden project"}, {"topic": "the garden project"}),
        ("weather", {"city": "New York"}, {"city": "new   york"}),
        (
            "calc",
            {"expression": "(144 / 12) + 7"},
            {"expression": "144 / 12 + 7"},
        ),
    ],
)
def test_semantic_fields_allow_only_their_declared_normalization(
    tool: str,
    expected: dict,
    actual: dict,
) -> None:
    score = score_contract_arguments(tool, expected, actual)
    assert score["scoreable"] is True
    assert score["contract_correct"] is True


def test_semantic_alternative_must_be_explicit_for_nontrivial_rewrite() -> None:
    without = score_contract_arguments(
        "remember",
        {"fact": "user prefers dark mode"},
        {"fact": "dark mode is the user's preference"},
    )
    with_alternative = score_contract_arguments(
        "remember",
        {"fact": "user prefers dark mode"},
        {"fact": "dark mode is the user's preference"},
        semantic_alternatives={
            "fact": ["dark mode is the user's preference"]
        },
    )
    assert without["contract_correct"] is False
    assert with_alternative["contract_correct"] is True


def test_optional_hardware_default_and_argument_free_tools() -> None:
    omitted = score_contract_arguments("hardware", {}, {})
    explicit = score_contract_arguments("hardware", {}, {"metric": "all"})
    time = score_contract_arguments("time", {}, {})
    assert omitted["contract_correct"] is True
    assert explicit["contract_correct"] is True
    assert time["contract_correct"] is True


def test_benchmark_reports_official_and_contract_scores_separately() -> None:
    cases = [
        BenchmarkCase(
            "Search exactly for: exact Query",
            "search",
            {"query": "exact Query"},
            case_id="search-literal",
            argument_contract_version=CONTRACT_VERSION,
            literal_references={"query": ["exact Query"]},
        ),
        BenchmarkCase(
            "Create an image of a red fox.",
            "image",
            {"prompt": "red fox"},
            case_id="image-literal",
            argument_contract_version=CONTRACT_VERSION,
            literal_references={"prompt": ["a red fox"]},
        ),
        BenchmarkCase(
            "Use the camera to find my keys.",
            "see",
            {"target": "keys"},
            case_id="see-semantic",
        ),
    ]
    backend = FixtureBackend(
        [
            '{"tool_calls":[{"name":"search","arguments":{"query":"exact Query"}}]}',
            '{"tool_calls":[{"name":"image","arguments":{"prompt":"a red fox"}}]}',
            '{"tool_calls":[{"name":"see","arguments":{"target":"my keys"}}]}',
        ]
    )
    report = BenchmarkRunner(Dispatcher(ToolRegistry(), backend)).run(cases)
    metrics = report["metrics"]
    assert metrics["exact_call_accuracy"] == pytest.approx(1 / 3)
    assert metrics["contract_scoring_coverage_rate"] == 1.0
    assert metrics["contract_argument_accuracy"] == 1.0
    assert metrics["contract_exact_call_accuracy"] == 1.0
    assert report["case_results"][1]["strict_arguments_correct"] is False
    assert report["case_results"][1]["contract_arguments_correct"] is True


def test_annotation_sidecar_is_bound_to_dataset_checksum(tmp_path) -> None:
    dataset = tmp_path / "cases.json"
    dataset.write_text("[]\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "version": "fixture-annotations-v1",
                "argument_contract_version": CONTRACT_VERSION,
                "dataset_sha256": digest,
                "annotations": {},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_contract_annotations(annotations, dataset_path=dataset)
    assert loaded["dataset_sha256"] == digest

    dataset.write_text("[ ]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not bound"):
        load_contract_annotations(annotations, dataset_path=dataset)
