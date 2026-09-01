from __future__ import annotations

import json

from prototypes.tool_dispatcher.build_flash168_literal_annotations import (
    build_annotations,
)
from prototypes.tool_dispatcher.flash168_typed_binder_replay import COMPARISON
from prototypes.tool_dispatcher.flash168_typed_binder_replay_v2 import (
    COMPARISON as V2_COMPARISON,
)
from prototypes.tool_dispatcher.literal_boundary_corpus_eval import (
    DEV_COMPARISON as BOUNDARY_DEV_COMPARISON,
)
from prototypes.tool_dispatcher.literal_boundary_corpus_eval import (
    FINAL_COMPARISON as BOUNDARY_FINAL_COMPARISON,
)
from prototypes.tool_dispatcher.literal_boundary_corpus_eval import (
    HOLDOUT_COMPARISON as BOUNDARY_HOLDOUT_COMPARISON,
)
from prototypes.tool_dispatcher.literal_boundary_corpus_eval import (
    V3_FREEZE as BOUNDARY_V3_FREEZE,
)
from prototypes.tool_dispatcher.literal_boundary_probe import (
    COMPARISON as BOUNDARY_COMPARISON,
)
from prototypes.tool_dispatcher.typed_literal_binder import bind_typed_call
from prototypes.tool_dispatcher.typed_literal_binder_v2 import (
    bind_typed_call as bind_typed_call_v2,
)


def _bind(raw: str, tool: str, field: str, value: str):
    return bind_typed_call(
        raw_request=raw,
        selected_call={"tool": tool, "arguments": {field: value}},
    )


def test_flash168_annotations_cover_all_literal_cases_without_gold_arguments():
    payload = build_annotations()
    assert payload["summary"] == {
        "total_source_cases": 168,
        "literal_source_cases": 104,
        "reference": 104,
        "ambiguous": 0,
    }
    assert payload["model_outputs_accessed"] is False
    assert payload["official_gold_arguments_used_as_literal_references"] is False


def test_typed_binder_repairs_rewritten_payloads_from_raw_source():
    search = _bind(
        "Search the web for latest MLX documentation.",
        "search",
        "query",
        "MLX docs Apple framework latest",
    )
    question = _bind(
        "Ask Claude to inspect this error: TypeError in server.py",
        "ask_claude",
        "question",
        "Please inspect the TypeError.",
    )
    image = _bind(
        "Create a picture using exactly: tiny robot watering sunflowers",
        "image",
        "prompt",
        "A detailed tiny robot in a garden",
    )

    assert search["final_call"]["arguments"]["query"] == (
        "latest MLX documentation"
    )
    assert question["final_call"]["arguments"]["question"] == (
        "inspect this error: TypeError in server.py"
    )
    assert image["final_call"]["arguments"]["prompt"] == (
        "tiny robot watering sunflowers"
    )
    assert all(item["source_copy_valid"] for item in (search, question, image))


def test_typed_binder_handles_paths_commands_urls_and_composites():
    path = _bind(
        "Read /Users/me/Folder With Spaces/File Name.md.",
        "read",
        "filepath",
        "/Users/me/folder",
    )
    command = _bind(
        "Run uname -a, do not search for the answer.",
        "shell",
        "command",
        "uname",
    )
    url = _bind(
        "Fetch this exact URL: https://example.com/docs.",
        "fetch",
        "url",
        "https://example.com",
    )
    composite = _bind(
        "Make a text file in Yuki's folder using exactly "
        "draft.txt|chapter one, not an image.",
        "yuki_write",
        "filename_and_content",
        "draft.txt|rewritten",
    )

    assert path["final_call"]["arguments"]["filepath"] == (
        "/Users/me/Folder With Spaces/File Name.md"
    )
    assert command["final_call"]["arguments"]["command"] == "uname -a"
    assert url["final_call"]["arguments"]["url"] == "https://example.com/docs"
    assert composite["final_call"]["arguments"]["filename_and_content"] == (
        "draft.txt|chapter one"
    )
    assert composite["status"] == "bound_composite"
    assert composite["source_copy_valid"] is True


def test_typed_binder_passes_semantic_fields_and_abstains_on_ambiguity():
    semantic = _bind("Weather in Tokyo", "weather", "city", "Tokyo")
    ambiguous = _bind(
        "Compare https://one.example with https://two.example",
        "fetch",
        "url",
        "https://one.example",
    )

    assert semantic["status"] == "typed_passthrough"
    assert semantic["final_call"] == {
        "tool": "weather",
        "arguments": {"city": "Tokyo"},
    }
    assert ambiguous["status"] == "ambiguous"
    assert ambiguous["final_call"] is None


def test_frozen_flash168_replay_is_zero_inference_and_execution_locked():
    report = json.loads(COMPARISON.read_text(encoding="utf-8"))
    configuration = report["configuration"]

    assert configuration["models_called"] is False
    assert configuration["flash_rerun"] is False
    assert configuration["new_generation"] is False
    assert configuration["sacred_450_used"] is False
    assert configuration["tool_execution_attempts"] == 0
    assert len(report["case_results"]) == 168
    assert all(not case["execution"]["executed"] for case in report["case_results"])
    assert report["overall"]["literal_annotated_cases"] == 104
    assert report["overall"]["literal_binder_coverage_successes"] == 104
    assert report["overall"]["source_copy_invariant_rate"] == 1.0
    assert report["overall"]["bound_typed_contract_successes"] == 166
    assert report["overall"]["typed_broken_count"] == 1
    assert report["suggested_decision_threshold"]["passed"] is False


def test_v2_claude_parser_consumes_one_outer_envelope_only():
    simple = bind_typed_call_v2(
        raw_request="Ask Claude this exact question: why does this fail?",
        selected_call={
            "tool": "ask_claude",
            "arguments": {"question": "rewritten"},
        },
    )
    nested = bind_typed_call_v2(
        raw_request=(
            "Ask Claude: handle this exact question: why does this fail?"
        ),
        selected_call={
            "tool": "ask_claude",
            "arguments": {"question": "rewritten"},
        },
    )
    em_dash = bind_typed_call_v2(
        raw_request="Claude — check this for me: why does this fail?",
        selected_call={
            "tool": "ask_claude",
            "arguments": {"question": "rewritten"},
        },
    )
    quoted = bind_typed_call_v2(
        raw_request='Send this to Claude: "Why did A::B return null?"',
        selected_call={
            "tool": "ask_claude",
            "arguments": {"question": "rewritten"},
        },
    )

    assert simple["final_call"]["arguments"]["question"] == (
        "why does this fail?"
    )
    assert nested["final_call"]["arguments"]["question"] == (
        "handle this exact question: why does this fail?"
    )
    assert em_dash["final_call"]["arguments"]["question"] == (
        "check this for me: why does this fail?"
    )
    assert quoted["final_call"]["arguments"]["question"] == (
        "Why did A::B return null?"
    )
    assert all(
        result["source_copy_valid"]
        for result in (simple, nested, em_dash, quoted)
    )


def test_fresh_boundary_probe_passes_before_flash168_v2_replay():
    comparison = json.loads(BOUNDARY_COMPARISON.read_text(encoding="utf-8"))

    assert comparison["v1"]["exact_successes"] == 31
    assert comparison["v2"]["exact_successes"] == 64
    assert len(comparison["case_flips"]["failure_to_success"]) == 33
    assert comparison["case_flips"]["success_to_failure"] == []
    assert comparison["case_flips"]["schema_success_to_failure"] == []
    assert comparison["gate"]["passed"] is True


def test_flash168_v2_replay_is_frozen_and_fails_zero_regression_gate():
    report = json.loads(V2_COMPARISON.read_text(encoding="utf-8"))
    configuration = report["configuration"]
    metrics = report["raw_to_v2"]

    assert configuration["models_called"] is False
    assert configuration["flash_rerun"] is False
    assert configuration["new_generation"] is False
    assert configuration["sacred_450_used"] is False
    assert configuration["tool_execution_attempts"] == 0
    assert len(report["case_results"]) == 168
    assert all(not case["execution"]["executed"] for case in report["case_results"])
    assert metrics["literal_bound_typed_successes"] == 101
    assert metrics["source_copy_invariant_rate"] == 1.0
    assert metrics["historical_broken_count"] == 3
    assert metrics["typed_broken_count"] == 3
    assert report["v1_to_v2"]["v1_to_v2_typed_repaired"] == 1
    assert report["v1_to_v2"]["v1_to_v2_typed_broken"] == 3
    assert report["authorization_gate"]["passed"] is False


def test_v3_is_checksum_frozen_before_holdout_unlock():
    freeze = json.loads(BOUNDARY_V3_FREEZE.read_text(encoding="utf-8"))
    dev = json.loads(BOUNDARY_DEV_COMPARISON.read_text(encoding="utf-8"))

    assert freeze["status"] == "binder_v3_frozen_before_holdout_unlock"
    assert freeze["development_comparison"]["gate_passed"] is True
    assert freeze["sealed_holdout"]["read_by_dev_runner"] is False
    assert freeze["sealed_holdout"]["scored_before_binder_freeze"] is False
    assert dev["overall"]["v3_exact_successes"] == 200
    assert dev["overall"]["v3_schema_successes"] == 200
    assert dev["overall"]["v3_source_copy_successes"] == 200
    assert dev["overall"]["v2_success_to_v3_failure"] == 0
    assert dev["gate"]["passed"] is True


def test_untouched_holdout_rejects_v3_without_benchmark_replay():
    holdout = json.loads(
        BOUNDARY_HOLDOUT_COMPARISON.read_text(encoding="utf-8")
    )
    final = json.loads(BOUNDARY_FINAL_COMPARISON.read_text(encoding="utf-8"))

    assert holdout["overall"]["v3_exact_successes"] == 85
    assert holdout["overall"]["v3_schema_successes"] == 89
    assert holdout["overall"]["v3_source_copy_checked"] == 89
    assert holdout["overall"]["v3_source_copy_successes"] == 88
    assert holdout["overall"]["v2_success_to_v3_failure"] == 0
    assert holdout["overall"]["schema_regressions"] == 3
    assert holdout["gate"]["passed"] is False
    assert final["final_gate"]["passed"] is False
    assert final["safety"] == {
        "models_called": False,
        "sacred_450_used": False,
        "historical_168_used": False,
        "yuki_tools_executed": False,
    }
    assert all(
        not case["execution"]["executed"]
        for case in holdout["case_results"]
    )
