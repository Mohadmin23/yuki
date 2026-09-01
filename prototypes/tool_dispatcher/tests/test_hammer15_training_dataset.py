from __future__ import annotations

import json
import random
from copy import deepcopy

from prototypes.tool_dispatcher.argument_contract import (
    LITERAL_SOURCE,
    NO_ARGUMENT,
    SEMANTIC_ARGUMENT,
    score_contract_arguments,
)
from prototypes.tool_dispatcher.registry import ToolRegistry
from prototypes.tool_dispatcher.training.build_hammer15_pilot import (
    DEFAULT_SEED,
    candidate_catalog,
)
from prototypes.tool_dispatcher.training.check_sacred450_leakage import (
    check_record,
)
from prototypes.tool_dispatcher.training.common import (
    PILOT_PATH,
    SACRED_DATASET,
    Hammer15TrainingRegistry,
    load_jsonl,
    verify_sacred_dataset,
)
from prototypes.tool_dispatcher.training.validate_hammer15_dataset import (
    EXPECTED_CATEGORY_COUNTS,
    record_errors,
    validate_records,
)


def _pilot() -> list[dict]:
    return load_jsonl(PILOT_PATH)


def _sacred() -> list[dict]:
    return json.loads(SACRED_DATASET.read_text(encoding="utf-8"))


def test_frozen_pilot_passes_schema_and_dataset_validation() -> None:
    report = validate_records(_pilot())
    assert report["passed"] is True
    assert report["record_count"] == 150
    assert report["failed_record_count"] == 0
    assert report["category_counts"] == EXPECTED_CATEGORY_COUNTS


def test_all_live_tools_are_represented() -> None:
    records = _pilot()
    represented = {
        row["metadata"]["expected_tool"]
        for row in records
        if row["metadata"]["expected_tool"] is not None
    }
    assert represented == set(ToolRegistry().names)


def test_model_visible_data_is_separated_from_audit_metadata() -> None:
    forbidden = {
        "record_id",
        "expected_tool",
        "argument_mode",
        "literal_source",
        "semantic_target",
        "leakage_check",
        "validation",
    }
    for record in _pilot():
        assert set(record["model_visible"]) == {"prompt", "assistant_target"}
        assert forbidden.isdisjoint(record["model_visible"])
        assert record["metadata"]["raw_user_request"] in record["model_visible"]["prompt"]


def test_every_literal_source_is_character_exact() -> None:
    literal_records = [
        record
        for record in _pilot()
        if record["metadata"]["argument_mode"] == LITERAL_SOURCE
    ]
    assert literal_records
    for record in literal_records:
        metadata = record["metadata"]
        annotation = metadata["literal_source"]
        request = metadata["raw_user_request"]
        pieces = [request[span["start"] : span["end"]] for span in annotation["spans"]]
        separator = annotation["deterministic_separator"] or ""
        reconstructed = separator.join(pieces)
        field = annotation["field"]
        assert reconstructed == annotation["reconstructed_value"]
        assert reconstructed == metadata["expected_arguments"][field]


def test_semantic_records_are_not_forced_to_literal_spans() -> None:
    semantic_records = [
        row
        for row in _pilot()
        if row["metadata"]["argument_mode"] == SEMANTIC_ARGUMENT
    ]
    assert semantic_records
    assert all(row["metadata"]["literal_source"] is None for row in semantic_records)
    assert any(
        row["metadata"]["expected_tool"] == "remember"
        and "I " in row["metadata"]["raw_user_request"]
        and "user" in next(iter(row["metadata"]["expected_arguments"].values()))
        for row in semantic_records
    )


def test_no_argument_tools_and_rejects_use_empty_argument_objects() -> None:
    records = _pilot()
    noarg_records = [
        row for row in records if row["metadata"]["argument_mode"] == NO_ARGUMENT
    ]
    assert {row["metadata"]["expected_tool"] for row in noarg_records} == {
        "time",
        "yuki_list",
    }
    assert all(row["metadata"]["expected_arguments"] == {} for row in noarg_records)
    reject_records = [
        row for row in records if row["metadata"]["argument_mode"] == "reject"
    ]
    assert len(reject_records) == 6
    assert all(row["model_visible"]["assistant_target"] == "```\n[]" for row in reject_records)


def test_calc_semantic_targets_use_ast_equivalence() -> None:
    score = score_contract_arguments(
        "calc",
        {"expression": "(80-14)/3"},
        {"expression": "(80 - 14) / 3"},
    )
    assert score["contract_correct"] is True


def test_invalid_literal_span_is_rejected() -> None:
    record = deepcopy(
        next(
            row
            for row in _pilot()
            if row["metadata"]["argument_mode"] == LITERAL_SOURCE
        )
    )
    record["metadata"]["literal_source"]["spans"][0]["text"] += "!"
    errors = record_errors(
        record,
        registry=Hammer15TrainingRegistry(),
        sacred_cases=_sacred(),
        check_stored_validation=False,
    )
    assert any("span 0 text" in error for error in errors)


def test_sacred_checksum_verification_remains_exact() -> None:
    assert verify_sacred_dataset() == (
        "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466"
    )


def test_leakage_checker_rejects_exact_and_obvious_near_duplicates() -> None:
    sacred = _sacred()
    base = deepcopy(_pilot()[0])
    base["metadata"]["expected_tool"] = sacred[0]["expected_tool"]
    base["metadata"]["expected_arguments"] = sacred[0]["expected_arguments"]
    base["metadata"]["raw_user_request"] = sacred[0]["request"]
    assert check_record(base, sacred)["status"] == "reject"

    base["metadata"]["raw_user_request"] = sacred[0]["request"] + " please"
    assert check_record(base, sacred)["status"] == "reject"


def test_malformed_target_and_live_schema_violation_are_rejected() -> None:
    record = deepcopy(_pilot()[0])
    record["model_visible"]["assistant_target"] = "```\nnot-json"
    errors = record_errors(
        record,
        registry=Hammer15TrainingRegistry(),
        sacred_cases=_sacred(),
        check_stored_validation=False,
    )
    assert any(error.startswith("assistant target:") for error in errors)

    validation = Hammer15TrainingRegistry().validate_call(
        {"tool": "weather", "arguments": {"city": "Tokyo", "extra": "bad"}},
        ["weather"],
    )
    assert validation["passed"] is False
    assert any("Unexpected arguments" in error for error in validation["errors"])


def test_seeded_catalog_order_is_reproducible() -> None:
    first = candidate_catalog()
    second = candidate_catalog()
    random.Random(DEFAULT_SEED).shuffle(first)
    random.Random(DEFAULT_SEED).shuffle(second)
    assert [candidate.source_id for candidate in first] == [
        candidate.source_id for candidate in second
    ]
