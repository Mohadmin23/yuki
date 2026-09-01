from __future__ import annotations

import json
from pathlib import Path

import pytest

from prototypes.tool_dispatcher.focused_repair import FocusedRepairRegistry
from prototypes.tool_dispatcher.hammer_binding_capacity import (
    MODELS,
    PAYLOAD_75,
    PRIMARY_100,
    SEE_25,
    _parser,
    arm_paths,
    build_primary_subset,
)
from prototypes.tool_dispatcher.prompting import build_prompt


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_primary_subset_is_exact_concatenation_of_frozen_sources() -> None:
    payload = _load(PAYLOAD_75)
    see = _load(SEE_25)
    subset = build_primary_subset(payload, see)
    assert subset["case_results"] == payload["case_results"] + see["case_results"]
    assert [case["case_id"] for case in subset["case_results"]] == [
        f"p2-{tool}-{index:02d}"
        for tool in ("search", "image", "ask_claude", "see")
        for index in range(1, 26)
    ]
    assert subset["provenance"]["qwen_called"] is False
    assert subset["provenance"]["delegations_regenerated"] is False
    assert subset["configuration"]["tool_execution_capability"] is False


def test_both_arms_use_identical_a1_native_prompt_packages() -> None:
    source = _load(PAYLOAD_75)["case_results"][0]
    delegation = source["delegation"]
    dispatcher_input = (
        "Delegated semantic action:\n"
        f"{delegation['request']}\n\n"
        "Exact user strings to preserve unchanged in arguments:\n"
        f"{json.dumps(delegation['verbatim'], ensure_ascii=False)}"
    )
    registry = FocusedRepairRegistry("schema")
    offered_tools = ("time", "weather", "fetch", "search", "calc")
    packages = [
        build_prompt(
            registry,
            dispatcher_input,
            offered_tools,
            output_mode="native",
            model_id=str(model),
        )
        for model in MODELS.values()
    ]
    assert packages[0] == packages[1]
    assert packages[0].native_dialect == "hammer_fenced_tool_call_array"


def test_capacity_cli_has_no_execute_option() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["run-hammer", "--arm", "hammer15", "--execute"])


def test_capacity_harness_has_no_tool_executor_import() -> None:
    source = Path(__file__).parents[1].joinpath("hammer_binding_capacity.py").read_text()
    assert "from .executor" not in source
    assert "ToolExecutor" not in source


@pytest.mark.parametrize("arm", tuple(MODELS))
def test_capacity_binder_inputs_are_gold_and_annotation_free(arm: str) -> None:
    payload = _load(arm_paths(arm)["binder_input"])
    encoded = json.dumps(payload)
    assert payload["status"] == "gold_and_literal_annotation_free"
    assert len(payload["cases"]) == 100
    forbidden = (
        "expected_tool",
        "expected_arguments",
        "argument_alternatives",
        "strict_exact_call",
        "literal-source-annotations",
    )
    assert all(token not in encoded for token in forbidden)


@pytest.mark.parametrize("arm", tuple(MODELS))
def test_capacity_predictions_are_frozen_without_execution(arm: str) -> None:
    predictions = _load(arm_paths(arm)["binder_predictions"])
    assert predictions["status"] == "predictions_frozen_before_scoring"
    assert predictions["configuration"]["tool_execution_capability"] is False
    assert predictions["configuration"]["tool_execution_attempts"] == 0
    assert len(predictions["case_results"]) == 100
    for case in predictions["case_results"]:
        assert case["execution"] == {
            "capability": False,
            "executed": False,
            "status": "not_available",
        }
        binding = case["binding"]
        if binding["status"] != "bound":
            continue
        span = binding["source_span"]
        value = binding["final_call"]["arguments"][binding["argument_name"]]
        assert value == span["text"]
        assert value == case["raw_request"][span["start"] : span["end"]]


def test_capacity_outputs_use_the_exact_frozen_primary_set() -> None:
    expected_ids = _load(PRIMARY_100)["provenance"]["case_ids"]
    for arm in MODELS:
        dispatch = _load(arm_paths(arm)["dispatch"])
        predictions = _load(arm_paths(arm)["binder_predictions"])
        assert [case["case_id"] for case in dispatch["case_results"]] == expected_ids
        assert [case["case_id"] for case in predictions["case_results"]] == expected_ids
        assert dispatch["configuration"]["qwen_called"] is False
        assert dispatch["configuration"]["yuki_tool_execution_attempts"] == 0
