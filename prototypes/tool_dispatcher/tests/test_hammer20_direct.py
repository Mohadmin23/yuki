from __future__ import annotations

import inspect
import json

from prototypes.tool_dispatcher.backends import FixtureBackend
from prototypes.tool_dispatcher.gpt_oss_direct import selected_cases
from prototypes.tool_dispatcher.hammer20_direct import run_experiment


def test_hammer20_direct_experiment_is_execution_locked(tmp_path, monkeypatch):
    import prototypes.tool_dispatcher.dispatcher as dispatcher_module

    def execution_is_forbidden(*args, **kwargs):
        raise AssertionError("Hammer 2.0 benchmark reached Yuki execution")

    monkeypatch.setattr(dispatcher_module, "execute_call", execution_is_forbidden)
    cases = selected_cases("smoke-18")
    generations = [
        json.dumps(
            [{"name": case.expected_tool, "arguments": case.expected_arguments}]
        )
        for case in cases
    ]
    backend = FixtureBackend(
        generations, model_id="/models/Hammer2.0-7b-8bit"
    )
    output = tmp_path / "report.json"
    failures = tmp_path / "failures.jsonl"
    checkpoint = tmp_path / "checkpoint.jsonl"
    report = run_experiment(
        backend,
        selection="smoke-18",
        output=output,
        failures_output=failures,
        checkpoint=checkpoint,
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
    assert checkpoint.with_suffix(".jsonl.sha256").is_file()
    assert "execute" not in inspect.signature(run_experiment).parameters

    resumed = run_experiment(
        FixtureBackend([], model_id="/models/Hammer2.0-7b-8bit"),
        selection="smoke-18",
        output=tmp_path / "resumed.json",
        failures_output=tmp_path / "resumed-failures.jsonl",
        checkpoint=checkpoint,
    )
    assert resumed["metrics"] == report["metrics"]
    assert resumed["case_results"] == report["case_results"]
