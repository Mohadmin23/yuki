from __future__ import annotations

import json

from prototypes.tool_dispatcher.training.build_hammer15_pilot_v12 import (
    REVISED_NUMBERS,
    create_unchecked_records_v12,
    semantic_content_change_numbers,
    verify_source_pilot_v11,
)
from prototypes.tool_dispatcher.training.common import (
    SACRED_DATASET,
    load_jsonl,
    verify_sacred_dataset,
)
from prototypes.tool_dispatcher.training.common_v11 import PILOT_PATH_V11
from prototypes.tool_dispatcher.training.common_v12 import (
    PILOT_DECISION_V12,
    PILOT_LEAKAGE_V12,
    PILOT_PATH_V12,
    SOURCE_PILOT_V11_SHA256,
)
from prototypes.tool_dispatcher.training.validate_hammer15_dataset_v12 import (
    validate_records_v12,
)


def _v11() -> list[dict]:
    return load_jsonl(PILOT_PATH_V11)


def _v12() -> list[dict]:
    return load_jsonl(PILOT_PATH_V12)


def _by_number(records: list[dict]) -> dict[int, dict]:
    return {
        int(record["metadata"]["record_id"].rsplit("-", 1)[1]): record
        for record in records
    }


def test_v12_source_is_the_exact_corrected_v11_artifact() -> None:
    assert verify_source_pilot_v11() == SOURCE_PILOT_V11_SHA256
    assert SOURCE_PILOT_V11_SHA256 == (
        "9bab33fa2e0c1df0701bd51b4da6d8a08940d05d3f88892501ea3ddafa3001ff"
    )


def test_v12_changes_exactly_the_five_human_reviewed_records() -> None:
    assert REVISED_NUMBERS == (30, 31, 36, 94, 117)
    unchecked = create_unchecked_records_v12()
    assert semantic_content_change_numbers(_v11(), unchecked) == list(REVISED_NUMBERS)


def test_record_030_ask_claude_question_has_one_terminal_mark() -> None:
    record = _by_number(_v12())[30]
    metadata = record["metadata"]
    payload = "Can `map[key]` mutate when key == 'A/B'?"
    assert metadata["raw_user_request"] == f"Ask Claude this exact question: {payload}"
    assert metadata["expected_arguments"] == {"question": payload}
    assert metadata["literal_source"]["spans"][0]["text"] == payload
    assert not metadata["raw_user_request"].endswith("?.")


def test_record_031_image_prompt_has_no_wrapper_period() -> None:
    record = _by_number(_v12())[31]
    metadata = record["metadata"]
    payload = "tiny robot holding a sign: DON'T PANIC"
    assert metadata["raw_user_request"] == (
        f"Create new artwork from this exact prompt: {payload}"
    )
    assert metadata["expected_arguments"] == {"prompt": payload}
    assert metadata["literal_source"]["reconstructed_value"] == payload
    assert not metadata["raw_user_request"].endswith(".")


def test_record_036_remember_preserves_explicit_negative_meaning() -> None:
    record = _by_number(_v12())[36]
    metadata = record["metadata"]
    fact = "user does not want meetings before ten"
    assert metadata["raw_user_request"] == (
        "dont forget this abt me: no meetings before ten"
    )
    assert metadata["argument_mode"] == "semantic_argument"
    assert metadata["literal_source"] is None
    assert metadata["expected_arguments"] == {"fact": fact}
    assert metadata["semantic_target"]["canonical_value"] == fact


def test_record_094_see_referent_is_unambiguous() -> None:
    record = _by_number(_v12())[94]
    metadata = record["metadata"]
    assert metadata["raw_user_request"] == (
        "check the blue toolbox in the current view"
    )
    assert metadata["expected_tool"] == "see"
    assert metadata["argument_mode"] == "semantic_argument"
    assert metadata["expected_arguments"] == {"target": "blue toolbox"}


def test_record_117_shell_command_is_literal_and_natural() -> None:
    record = _by_number(_v12())[117]
    metadata = record["metadata"]
    command = "wc -l changelog.txt"
    assert metadata["raw_user_request"] == (
        "Need the line count, not system stats—run exactly: wc -l changelog.txt"
    )
    assert metadata["expected_arguments"] == {"command": command}
    span = metadata["literal_source"]["spans"][0]
    assert metadata["raw_user_request"][span["start"] : span["end"]] == command


def test_v12_is_exactly_150_and_passes_every_validator() -> None:
    records = _v12()
    report = validate_records_v12(records)
    assert len(records) == 150
    assert report["passed"] is True
    assert report["failed_record_count"] == 0
    assert report["category_counts"] == {
        "adversarial_edge": 15,
        "literal_preservation": 35,
        "ontology_confusion": 60,
        "ordinary_easy": 15,
        "semantic_argument": 25,
    }


def test_v12_leakage_is_150_pass_and_no_flags() -> None:
    report = json.loads(PILOT_LEAKAGE_V12.read_text(encoding="utf-8"))
    assert report["counts"] == {"pass": 150, "review": 0, "reject": 0}
    assert report["flagged"] == {}


def test_v12_preserves_rejections_tools_and_schema_exposure() -> None:
    records = _v12()
    metadata = [record["metadata"] for record in records]
    assert sum(row["argument_mode"] == "reject" for row in metadata) == 6
    assert len({row["expected_tool"] for row in metadata if row["expected_tool"]}) == 18
    assert sum(len(row["offered_tools"]) == 18 for row in metadata) == 17
    assert sum(len(row["offered_tools"]) == 2 for row in metadata) == 86


def test_v12_decision_and_safety_invariants_are_explicit() -> None:
    manifest = json.loads(
        PILOT_PATH_V12.with_name("hammer15-pilot-v1.2-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["decision"] == PILOT_DECISION_V12
    assert PILOT_DECISION_V12 == "APPROVE FOR STAGED SCALING"
    assert manifest["human_review_history"] == {
        "keep": 145,
        "revise": 5,
        "reject": 0,
    }
    assert manifest["models_loaded"] is False
    assert manifest["yuki_tools_executed"] is False
    assert manifest["frozen_450_rerun"] is False


def test_frozen_450_checksum_is_unchanged() -> None:
    assert SACRED_DATASET.name == "phase2-cases.json"
    assert verify_sacred_dataset() == (
        "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466"
    )
