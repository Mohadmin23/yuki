from __future__ import annotations

import json
from pathlib import Path

import pytest

from prototypes.tool_dispatcher.focused_repair import FocusedRepairRegistry
from prototypes.tool_dispatcher.prompting import build_prompt
from prototypes.tool_dispatcher.see_capacity_probe import (
    A_FROZEN_DELEGATIONS,
    HAMMER3_MODEL,
    HAMMER15_A1,
    _parser,
    build_see_subset,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_see_subset_copies_the_exact_25_frozen_case_objects() -> None:
    source = _load(A_FROZEN_DELEGATIONS)
    subset = build_see_subset(source)
    expected = [
        case for case in source["case_results"] if case["expected_tool"] == "see"
    ]
    assert subset["case_results"] == expected
    assert [case["case_id"] for case in subset["case_results"]] == [
        f"p2-see-{index:02d}" for index in range(1, 26)
    ]
    assert subset["provenance"]["delegations_regenerated"] is False
    assert subset["configuration"]["tool_execution_capability"] is False


def test_hammer3_uses_the_same_a1_native_prompt_package() -> None:
    baseline = _load(HAMMER15_A1)
    old_case = next(
        case for case in baseline["case_results"] if case["expected_tool"] == "see"
    )
    dispatch = old_case["dispatcher"]
    registry = FocusedRepairRegistry("schema")
    old_package = build_prompt(
        registry,
        dispatch["dispatcher_input"],
        dispatch["offered_tools"],
        output_mode="native",
        model_id=baseline["configuration"]["dispatcher_model_id"],
    )
    new_package = build_prompt(
        registry,
        dispatch["dispatcher_input"],
        dispatch["offered_tools"],
        output_mode="native",
        model_id=str(HAMMER3_MODEL),
    )
    assert new_package == old_package
    assert new_package.schemas_sent == dispatch["schemas_sent"]
    assert new_package.native_dialect == "hammer_fenced_tool_call_array"


def test_capacity_probe_cli_has_no_execute_option() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--execute"])


def test_capacity_probe_has_no_tool_executor_import() -> None:
    source = Path(__file__).parents[1].joinpath("see_capacity_probe.py").read_text()
    assert "from .executor" not in source
    assert "ToolExecutor" not in source
