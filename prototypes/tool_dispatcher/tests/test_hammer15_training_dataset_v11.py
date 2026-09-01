from __future__ import annotations

import json
import random
from collections import Counter
from copy import deepcopy

from prototypes.tool_dispatcher.argument_contract import (
    LITERAL_SOURCE,
    SEMANTIC_ARGUMENT,
)
from prototypes.tool_dispatcher.training.build_hammer15_pilot_v11 import (
    DEFAULT_SEED_V11,
    pilot_v11_catalog,
)
from prototypes.tool_dispatcher.training.check_sacred450_leakage import (
    REJECT_THRESHOLDS,
    REVIEW_THRESHOLDS,
)
from prototypes.tool_dispatcher.training.common import (
    PILOT_PATH,
    SACRED_DATASET,
    load_jsonl,
    sha256_path,
    verify_sacred_dataset,
)
from prototypes.tool_dispatcher.training.common_v11 import (
    COMPOSITE_TOOLS,
    PILOT_LEAKAGE_V11,
    PILOT_PATH_V11,
    Hammer15TrainingRegistryV11,
)
from prototypes.tool_dispatcher.training.validate_hammer15_dataset_v11 import (
    CONFUSION_FAMILIES,
    record_errors_v11,
    validate_records_v11,
)
from tools._helpers import split_filename_content


def _pilot_v11() -> list[dict]:
    return load_jsonl(PILOT_PATH_V11)


def _sacred() -> list[dict]:
    return json.loads(SACRED_DATASET.read_text(encoding="utf-8"))


def test_pilot_v11_is_150_records_and_strictly_valid() -> None:
    report = validate_records_v11(_pilot_v11())
    assert report["passed"] is True
    assert report["record_count"] == 150
    assert report["failed_record_count"] == 0


def test_v11_has_no_accepted_leakage_flags_or_rejects() -> None:
    report = json.loads(PILOT_LEAKAGE_V11.read_text(encoding="utf-8"))
    assert report["counts"] == {"pass": 150, "review": 0, "reject": 0}
    assert report["flagged"] == {}
    assert REJECT_THRESHOLDS == {
        "character_similarity": 0.88,
        "token_jaccard": 0.72,
        "trigram_jaccard": 0.72,
        "literal_payload_similarity": 0.92,
    }
    assert REVIEW_THRESHOLDS["character_similarity"] == 0.80


def test_v1_artifact_remains_unchanged_and_separate() -> None:
    assert len(load_jsonl(PILOT_PATH)) == 150
    assert sha256_path(PILOT_PATH) == (
        "563f4cc6f159f497836adaa585f8b3fd437a20126310541a8427f19f878a3c82"
    )
    assert all(row["record_version"] == "hammer15-training-record-v1" for row in load_jsonl(PILOT_PATH))
    assert all(row["record_version"] == "hammer15-training-record-v1.1" for row in _pilot_v11())


def test_all_six_ontology_families_keep_ten_diverse_controls() -> None:
    ontology = [
        row["metadata"]
        for row in _pilot_v11()
        if row["metadata"]["category"] == "ontology_confusion"
    ]
    counts = Counter(row["confusion_family"] for row in ontology)
    assert set(counts) == CONFUSION_FAMILIES
    assert set(counts.values()) == {10}
    openings = {
        " ".join(row["raw_user_request"].casefold().split()[:2]) for row in ontology
    }
    assert len(openings) >= 45


def test_v11_has_a_meaningful_annotated_noisy_language_sample() -> None:
    features = Counter(
        feature
        for row in _pilot_v11()
        for feature in row["metadata"]["linguistic_features"]
    )
    assert features["typo"] >= 8
    assert features["terse_fragment"] >= 10
    assert features["casual"] >= 8
    assert features["negative_contrast"] >= 10
    assert features["pronoun_context"] >= 8


def test_hardware_process_and_top_have_defined_nonoverlapping_meanings() -> None:
    rows = {row["metadata"]["record_id"]: row for row in _pilot_v11()}
    own_process = rows["h15-pilot-v1.1-044"]
    system_top = rows["h15-pilot-v1.1-054"]
    assert "your own LLM process" in own_process["metadata"]["raw_user_request"]
    assert own_process["metadata"]["expected_arguments"] == {"metric": "process"}
    assert "busiest processes" in system_top["metadata"]["raw_user_request"]
    assert system_top["metadata"]["expected_arguments"] == {"metric": "top"}

    schemas = Hammer15TrainingRegistryV11().hammer_schemas(["hardware"])
    description = schemas[0]["parameters"]["metric"]["description"]
    assert "Yuki's own current LLM/runtime process" in description
    assert "top five processes system-wide" in description


def test_write_and_append_use_structured_model_arguments_only() -> None:
    composite = [
        row for row in _pilot_v11() if row["metadata"]["expected_tool"] in COMPOSITE_TOOLS
    ]
    assert composite
    for row in composite:
        expected = row["metadata"]["expected_arguments"]
        assert set(expected) == {"filename", "content"}
        assert "filename_and_content" not in row["model_visible"]["assistant_target"]
        assert set(row["metadata"]["runtime_arguments"]) == {
            "filename_and_content"
        }


def test_composite_adapter_preserves_pipes_inside_content() -> None:
    row = next(
        row
        for row in _pilot_v11()
        if row["metadata"]["expected_tool"] == "yuki_write"
        and "|" in row["metadata"]["expected_arguments"]["content"]
    )
    expected = row["metadata"]["expected_arguments"]
    runtime = row["metadata"]["runtime_arguments"]["filename_and_content"]
    assert split_filename_content(runtime) == (expected["filename"], expected["content"])


def test_composite_adapter_rejects_pipe_in_filename() -> None:
    registry = Hammer15TrainingRegistryV11()
    validation = registry.validate_training_call(
        {
            "tool": "yuki_write",
            "arguments": {"filename": "bad|name.txt", "content": "safe"},
        },
        ["yuki_write"],
    )
    assert validation["passed"] is False
    assert any("filename" in error and "'|'" in error for error in validation["errors"])


def test_v11_literal_spans_and_runtime_reconstruction_are_exact() -> None:
    literal = [
        row
        for row in _pilot_v11()
        if row["metadata"]["argument_mode"] == LITERAL_SOURCE
    ]
    assert len(literal) == 86
    for row in literal:
        metadata = row["metadata"]
        annotation = metadata["literal_source"]
        request = metadata["raw_user_request"]
        pieces = [
            request[span["start"] : span["end"]] for span in annotation["spans"]
        ]
        separator = annotation["deterministic_separator"] or ""
        assert separator.join(pieces) == annotation["reconstructed_value"]


def test_v11_semantic_records_remain_span_free() -> None:
    semantic = [
        row
        for row in _pilot_v11()
        if row["metadata"]["argument_mode"] == SEMANTIC_ARGUMENT
    ]
    assert len(semantic) == 54
    assert all(row["metadata"]["literal_source"] is None for row in semantic)


def test_invalid_structured_target_and_span_are_rejected() -> None:
    record = deepcopy(
        next(
            row
            for row in _pilot_v11()
            if row["metadata"]["expected_tool"] == "yuki_append"
        )
    )
    record["model_visible"]["assistant_target"] = "```\nnot-json"
    record["metadata"]["literal_source"]["spans"][0]["text"] += "!"
    errors = record_errors_v11(
        record,
        registry=Hammer15TrainingRegistryV11(),
        sacred_cases=_sacred(),
        check_stored_validation=False,
    )
    assert any(error.startswith("assistant target:") for error in errors)
    assert any("span 0 text" in error for error in errors)


def test_v11_model_visible_boundary_excludes_runtime_and_audit_metadata() -> None:
    for row in _pilot_v11():
        assert set(row["model_visible"]) == {"prompt", "assistant_target"}
        for forbidden in (
            "runtime_arguments",
            "literal_source",
            "semantic_target",
            "leakage_check",
            "validation",
        ):
            assert forbidden not in row["model_visible"]


def test_v11_seeded_catalog_order_is_reproducible() -> None:
    first = pilot_v11_catalog()
    second = pilot_v11_catalog()
    random.Random(DEFAULT_SEED_V11).shuffle(first)
    random.Random(DEFAULT_SEED_V11).shuffle(second)
    assert [candidate.source_id for candidate, _ in first] == [
        candidate.source_id for candidate, _ in second
    ]


def test_sacred_checksum_is_still_immutable() -> None:
    assert verify_sacred_dataset() == (
        "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466"
    )
