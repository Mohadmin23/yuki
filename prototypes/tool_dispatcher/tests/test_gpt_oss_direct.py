from __future__ import annotations

import inspect

from prototypes.tool_dispatcher.backends import FixtureBackend
from prototypes.tool_dispatcher.gpt_oss_direct import (
    run_experiment,
    selected_cases,
)


def test_frozen_direct_smoke_has_one_case_per_live_tool():
    cases = selected_cases("smoke-18")
    assert len(cases) == 18
    assert len({case.expected_tool for case in cases}) == 18


def test_direct_experiment_is_execution_locked(tmp_path, monkeypatch):
    import prototypes.tool_dispatcher.dispatcher as dispatcher_module

    def execution_is_forbidden(*args, **kwargs):
        raise AssertionError("direct GPT-OSS benchmark reached Yuki execution")

    monkeypatch.setattr(dispatcher_module, "execute_call", execution_is_forbidden)
    cases = selected_cases("smoke-18")
    generations = [
        f'{{"name":"{case.expected_tool}","arguments":'
        f'{__import__("json").dumps(case.expected_arguments)}}}'
        for case in cases
    ]
    generations = [f"[{generation}]" for generation in generations]
    backend = FixtureBackend(generations, model_id="openai/gpt-oss-20b")
    output = tmp_path / "report.json"
    failures = tmp_path / "failures.jsonl"
    report = run_experiment(
        backend,
        selection="smoke-18",
        output=output,
        failures_output=failures,
    )

    assert report["metrics"]["tool_selection_accuracy"] == 1.0
    assert report["metrics"]["exact_call_accuracy"] == 1.0
    assert report["configuration"]["exposure_condition"] == (
        "all_18_real_yuki_schemas"
    )
    assert report["configuration"]["execution_capability"] is False
    assert report["configuration"]["yuki_tool_execution_attempts"] == 0
    assert all(
        case["execution"]["status"] == "not_requested"
        for case in report["case_results"]
    )
    assert output.with_suffix(".json.sha256").is_file()
    assert failures.with_suffix(".jsonl.sha256").is_file()
    assert "execute" not in inspect.signature(run_experiment).parameters
