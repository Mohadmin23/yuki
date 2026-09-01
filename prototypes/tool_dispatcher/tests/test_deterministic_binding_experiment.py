from __future__ import annotations

import json
from pathlib import Path

from prototypes.tool_dispatcher.prepare_binding_input import build_input

ROOT = Path(__file__).parents[1]
REPORT_ROOT = ROOT / "reports" / "deterministic-binding"


def _load(name: str) -> dict:
    return json.loads(REPORT_ROOT.joinpath(name).read_text(encoding="utf-8"))


def test_prepared_binder_input_contains_no_gold_or_literal_references() -> None:
    payload = build_input()
    encoded = json.dumps(payload)
    assert payload["status"] == "gold_and_literal_annotation_free"
    assert len(payload["cases"]) == 125
    forbidden = (
        "expected_tool",
        "expected_arguments",
        "argument_alternatives",
        "strict_exact_call",
        "literal-source-annotations",
    )
    assert all(token not in encoded for token in forbidden)


def test_binder_runtime_source_cannot_access_annotations_or_gold() -> None:
    sources = [
        ROOT.joinpath("source_span_policy.py").read_text(encoding="utf-8"),
        ROOT.joinpath("deterministic_binding_experiment.py").read_text(
            encoding="utf-8"
        ),
    ]
    forbidden = (
        "literal-source-annotations",
        "expected_arguments",
        "official_gold",
        "score_deterministic_binding",
        "build_source_span_annotations",
    )
    assert all(token not in source for source in sources for token in forbidden)


def test_predictions_were_frozen_before_scoring_without_execution() -> None:
    predictions = _load("deterministic-binding-targeted100-recall25-v1.json")
    assert predictions["status"] == "predictions_frozen_before_scoring"
    assert predictions["configuration"]["models_called"] is False
    assert predictions["configuration"]["hammer_loaded"] is False
    assert predictions["configuration"]["tool_execution_capability"] is False
    assert predictions["configuration"]["tool_execution_attempts"] == 0
    assert len(predictions["case_results"]) == 125
    assert all(
        not case["execution"]["executed"]
        and not case["execution"]["capability"]
        for case in predictions["case_results"]
    )


def test_every_emitted_bound_string_is_its_recorded_raw_slice() -> None:
    predictions = _load("deterministic-binding-targeted100-recall25-v1.json")
    bound = [
        case
        for case in predictions["case_results"]
        if case["binding"]["status"] == "bound"
    ]
    assert len(bound) == 110
    for case in bound:
        binding = case["binding"]
        span = binding["source_span"]
        value = binding["final_call"]["arguments"][binding["argument_name"]]
        assert value == span["text"]
        assert value == case["raw_request"][span["start"] : span["end"]]
        assert binding["source_copy_valid"] is True


def test_literal_annotations_are_independently_well_formed() -> None:
    annotations = _load("literal-source-annotations-targeted100-recall25-v1.json")
    predictions = _load("deterministic-binding-targeted100-recall25-v1.json")
    raw_by_id = {
        case["case_id"]: case["raw_request"] for case in predictions["case_results"]
    }
    assert annotations["binder_access"] is False
    assert annotations["official_gold_included"] is False
    assert annotations["summary"] == {
        "total": 125,
        "reference": 119,
        "ambiguous": 2,
        "no_argument": 4,
    }
    for case_id, annotation in annotations["annotations"].items():
        raw = raw_by_id[case_id]
        for span in annotation["spans"]:
            assert span["text"] == raw[span["start"] : span["end"]]


def test_comparison_keeps_official_and_literal_metrics_separate() -> None:
    comparison = _load("DETERMINISTIC-BINDING-COMPARISON-v1.json")
    primary = comparison["primary_targeted_100"]
    assert comparison["safety"]["official_gold_mutated"] is False
    assert comparison["safety"]["binder_had_literal_annotation_access"] is False
    assert primary["baseline_official_strict_exact_accuracy"] == 0.58
    assert primary["binder_official_strict_exact_accuracy"] == 0.64
    assert primary["baseline_literal_source_exact_conservative"] == 0.51
    assert primary["binder_literal_source_exact_conservative"] == 0.87
