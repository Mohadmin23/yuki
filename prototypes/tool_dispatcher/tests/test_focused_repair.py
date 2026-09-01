from __future__ import annotations

import inspect
import json

import pytest

from prototypes.tool_dispatcher.backends import FixtureBackend
from prototypes.tool_dispatcher.delegation_dispatcher import ValidationOnlyDispatcher
from prototypes.tool_dispatcher.delegation_mainbrain import DELEGATION_SYSTEM_PROMPT
from prototypes.tool_dispatcher.focused_repair import (
    A_REGRESSION_TOOLS,
    A_TARGET_TOOLS,
    B_TARGET_TOOLS,
    FULL_STAGE_A,
    FULL_STAGE_B,
    LOSSLESS_VERBATIM_SYSTEM_PROMPT,
    REPAIRED_TASK_INSTRUCTION,
    FocusedRepairRegistry,
    _diagnosis_report,
    _extract_baseline,
    _old_verbatim_analysis,
    _parser,
    _subset_stage_a,
    verify_frozen_artifacts,
)
from prototypes.tool_dispatcher.prompting import build_prompt
from prototypes.tool_dispatcher.registry import ToolRegistry


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_focused_repair_frozen_inputs_match_checksums():
    observed = verify_frozen_artifacts()
    assert len(observed) == 6


def test_focused_subsets_use_exact_frozen_case_counts():
    stage_a = _load(FULL_STAGE_A)
    assert len(_subset_stage_a(stage_a, A_TARGET_TOOLS, "a")["case_results"]) == 50
    assert (
        len(_subset_stage_a(stage_a, A_REGRESSION_TOOLS, "reg")["case_results"])
        == 50
    )
    assert len(_subset_stage_a(stage_a, B_TARGET_TOOLS, "b")["case_results"]) == 75


def test_description_repairs_change_only_dispatcher_facing_text():
    live = ToolRegistry()
    repaired = FocusedRepairRegistry("schema")
    names = ("see", "image", "remember", "recall")
    before = {item["name"]: item for item in live.hammer_schemas(names)}
    after = {item["name"]: item for item in repaired.hammer_schemas(names)}
    for name in names:
        assert before[name]["description"] != after[name]["description"]
        assert before[name]["parameters"].keys() == after[name]["parameters"].keys()
        for parameter in before[name]["parameters"]:
            old = dict(before[name]["parameters"][parameter])
            new = dict(after[name]["parameters"][parameter])
            old.pop("description")
            new.pop("description")
            assert old == new
        assert live.get(name).description == ToolRegistry().get(name).description


def test_baseline_prompt_rendering_is_unchanged_by_optional_hooks():
    registry = FocusedRepairRegistry("baseline")
    request = "Delegated semantic action:\ninspect the current view"
    default = build_prompt(registry, request, ("see", "image"), model_id="Hammer2.1")
    explicit = build_prompt(
        registry,
        request,
        ("see", "image"),
        model_id="Hammer2.1",
        task_instruction=None,
        hammer_format_instruction=None,
    )
    assert default.content == explicit.content
    assert default.schemas_sent == explicit.schemas_sent


def test_rejection_repair_preserves_reject_path_but_clarifies_selection():
    registry = FocusedRepairRegistry("schema_rejection")
    package = build_prompt(
        registry,
        "Delegated semantic action:\nretrieve stored memory about X",
        ("remember", "recall"),
        model_id="Hammer2.1",
        task_instruction=REPAIRED_TASK_INSTRUCTION,
    )
    assert "Select a listed tool whenever" in package.content
    assert "Reject only when none" in package.content
    assert "output an empty array" in package.content


def test_lossless_qwen_prompt_changes_only_verbatim_instruction_area():
    assert "lossless transport channel" in LOSSLESS_VERBATIM_SYSTEM_PROMPT
    assert "HTML-escape" in LOSSLESS_VERBATIM_SYSTEM_PROMPT
    assert (
        LOSSLESS_VERBATIM_SYSTEM_PROMPT.split("Domain meanings:", 1)[1]
        == DELEGATION_SYSTEM_PROMPT.split("Domain meanings:", 1)[1]
    )
    assert LOSSLESS_VERBATIM_SYSTEM_PROMPT != DELEGATION_SYSTEM_PROMPT


def test_frozen_baseline_and_diagnosis_reproduce_reported_rejections():
    stage_b = _load(FULL_STAGE_B)
    baseline = _extract_baseline(stage_b)
    assert baseline["focused_metrics"]["see"]["tool_selection_accuracy"] == 0.4
    assert baseline["focused_metrics"]["see"]["strict_exact_call_accuracy"] == 0.36
    assert baseline["focused_metrics"]["see"]["rejection_count"] == 13
    assert baseline["focused_metrics"]["recall"]["tool_selection_accuracy"] == 0.52
    assert baseline["focused_metrics"]["recall"]["strict_exact_call_accuracy"] == 0.44
    assert baseline["focused_metrics"]["recall"]["rejection_count"] == 11
    diagnosis = _diagnosis_report(stage_b)
    assert len(diagnosis["rejected_cases"]) == 24
    assert diagnosis["all_rejections_were_valid_empty_arrays"]


def test_old_verbatim_analysis_uses_exact_75_cases():
    analysis = _old_verbatim_analysis(_load(FULL_STAGE_A))
    assert analysis["metrics"]["cases"] == 75
    assert set(analysis["by_tool"]) == set(B_TARGET_TOOLS)
    assert len(analysis["failures"]) == 29


def test_focused_dispatcher_remains_validation_only():
    dispatcher = ValidationOnlyDispatcher(
        FocusedRepairRegistry("schema_rejection"),
        FixtureBackend(
            '```\n[{"name":"recall","arguments":{"topic":"X"}}]',
            model_id="Hammer2.1-fixture",
        ),
        task_instruction=REPAIRED_TASK_INSTRUCTION,
    )
    result = dispatcher.route(
        {"request": "retrieve stored memory", "domain_hint": "memory", "verbatim": ["X"]},
        gold_domain="memory",
    )
    assert result["validation"]["passed"]
    assert result["execution_capability"] is False


def test_focused_harness_has_no_yuki_execution_import_or_cli_flag():
    source = inspect.getsource(
        __import__("prototypes.tool_dispatcher.focused_repair", fromlist=["*"])
    )
    assert "from .executor" not in source
    assert "execute_call" not in source
    with pytest.raises(SystemExit):
        _parser().parse_args(["run-hammer", "--execute"])
