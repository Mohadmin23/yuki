from __future__ import annotations

import json

from prototypes.tool_dispatcher.backends import FixtureBackend
from prototypes.tool_dispatcher.hammer20_ontology_repair import (
    ONTOLOGY_DESCRIPTION_REPAIRS,
    OntologyRepairRegistry,
    _target_ids,
    run_cases,
)
from prototypes.tool_dispatcher.registry import ToolRegistry


def test_ontology_repair_changes_only_four_hammer_descriptions() -> None:
    live = ToolRegistry().hammer_schemas(ToolRegistry().names)
    repaired_registry = OntologyRepairRegistry()
    repaired = repaired_registry.hammer_schemas(repaired_registry.names)
    before = {schema["name"]: schema for schema in live}
    after = {schema["name"]: schema for schema in repaired}
    changed = {
        name
        for name in before
        if before[name]["description"] != after[name]["description"]
    }
    assert changed == set(ONTOLOGY_DESCRIPTION_REPAIRS)
    for name, schema in before.items():
        old = dict(schema)
        new = dict(after[name])
        old.pop("description")
        new.pop("description")
        assert old == new


def test_targeted_regression_set_is_frozen_to_49_unique_cases() -> None:
    ids = _target_ids()
    assert len(ids) == len(set(ids)) == 49


def test_ontology_runner_has_no_execution_path(tmp_path) -> None:
    case_ids = ["p2-read-01", "p2-yuki_read-01"]
    generations = [
        json.dumps([{"name": "read", "arguments": {"filepath": "/tmp/report.txt"}}]),
        json.dumps([{"name": "yuki_read", "arguments": {"filename": "todo.txt"}}]),
    ]
    target_set = tmp_path / "target-set.json"
    target_set.write_text("{}", encoding="utf-8")
    report = run_cases(
        FixtureBackend(generations, model_id="/models/Hammer2.0-7b-8bit"),
        target_ids=case_ids,
        output=tmp_path / "report.json",
        failures_output=tmp_path / "failures.jsonl",
        checkpoint=tmp_path / "checkpoint.jsonl",
        experiment="test",
        target_set_path=target_set,
    )
    assert report["metrics"]["exact_call_accuracy"] == 1.0
    assert report["configuration"]["execution_capability"] is False
    assert report["configuration"]["yuki_tool_execution_attempts"] == 0
    assert all(case["execution"]["executed"] is False for case in report["case_results"])
