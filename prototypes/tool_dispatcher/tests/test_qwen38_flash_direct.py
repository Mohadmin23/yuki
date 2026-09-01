from __future__ import annotations

import json

from prototypes.tool_dispatcher.backends import FixtureBackend
from prototypes.tool_dispatcher.dialects import (
    OPENAI_NATIVE_TOOL_CALLS,
    native_dialect_for_model,
)
from prototypes.tool_dispatcher.qwen38_flash_direct import (
    BROAD_SHA256,
    PHASE2_SHA256,
    PROVIDER,
    TARGETED_SHA256,
    combine_reports,
    run_suite,
    suite_cases,
    verify_sources,
)


def _fixture_calls(suite: str) -> list[str]:
    return [
        json.dumps(
            [
                {
                    "name": case.expected_tool,
                    "arguments": case.expected_arguments,
                }
            ]
        )
        for case in suite_cases(suite)
    ]


def _run_fixture(tmp_path, suite: str):
    backend = FixtureBackend(
        _fixture_calls(suite),
        model_id="qwen/qwen3.8-flash",
    )
    return run_suite(
        backend,
        suite=suite,
        output=tmp_path / f"{suite}.json",
        failures_output=tmp_path / f"{suite}-failures.jsonl",
        raw_output=tmp_path / f"{suite}-raw.jsonl",
        checkpoint=tmp_path / f"{suite}-checkpoint.jsonl",
    )


def test_qwen38_flash_uses_openai_native_dialect():
    assert native_dialect_for_model("qwen/qwen3.8-flash") == OPENAI_NATIVE_TOOL_CALLS


def test_qwen38_flash_source_locks():
    checksums = verify_sources()
    assert PHASE2_SHA256 in checksums.values()
    assert BROAD_SHA256 in checksums.values()
    assert TARGETED_SHA256 in checksums.values()


def test_qwen38_flash_screen_is_execution_locked(tmp_path, monkeypatch):
    import prototypes.tool_dispatcher.dispatcher as dispatcher_module

    def execution_is_forbidden(*args, **kwargs):
        raise AssertionError("Qwen3.8-Flash screen reached Yuki execution")

    monkeypatch.setattr(dispatcher_module, "execute_call", execution_is_forbidden)
    report = _run_fixture(tmp_path, "broad-72")

    assert report["metrics"]["total_cases"] == 72
    assert report["metrics"]["tool_selection_accuracy"] == 1.0
    assert report["metrics"]["strict_exact_call_accuracy"] == 1.0
    assert report["configuration"]["provider"] == PROVIDER
    assert report["configuration"]["all_18_schemas_every_case"] is True
    assert report["configuration"]["execution_capability"] is False
    assert all(len(case["offered_tools"]) == 18 for case in report["case_results"])
    assert all(not case["execution"]["executed"] for case in report["case_results"])
    for path in tmp_path.iterdir():
        if path.suffix in {".json", ".jsonl"}:
            assert path.with_suffix(path.suffix + ".sha256").is_file()


def test_combined_report_applies_predeclared_gate(tmp_path):
    broad = _run_fixture(tmp_path / "broad", "broad-72")
    targeted = _run_fixture(tmp_path / "targeted", "targeted-96")
    combined = combine_reports(broad, targeted)

    assert combined["metrics"]["total_cases"] == 168
    assert combined["screening_gate"]["passed"] is True
    assert combined["configuration"]["suite_sha256"] == {
        "broad-72": BROAD_SHA256,
        "targeted-96": TARGETED_SHA256,
    }
