"""Apply exactly five approved human revisions to checksum-locked Pilot v1.1."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .check_sacred450_leakage import check_record, check_records
from .common import (
    PILOT_PATH,
    SACRED_DATASET,
    load_jsonl,
    sha256_path,
    write_json,
    write_jsonl,
)
from .common_v11 import (
    DEFAULT_SEED_V11,
    PILOT_PATH_V11,
    PROMPT_PROFILE_V11,
    Hammer15TrainingRegistryV11,
    prompt_schema_sha256_v11,
    render_model_visible_v11,
    verify_v11_safety_constants,
)
from .common_v12 import (
    GENERATOR_VERSION_V12,
    PILOT_DECISION_V12,
    PILOT_LEAKAGE_V12,
    PILOT_MANIFEST_V12,
    PILOT_PATH_V12,
    PILOT_REJECTED_V12,
    PILOT_VALIDATION_V12,
    RECORD_VERSION_V12,
    SOURCE_PILOT_V11_SHA256,
    VALIDATOR_VERSION_V12,
    write_record_schema_v12,
)
from .validate_hammer15_dataset_v12 import (
    record_errors_v12,
    validate_records_v12,
)

REVISED_NUMBERS = (30, 31, 36, 94, 117)


def verify_source_pilot_v11() -> str:
    observed = sha256_path(PILOT_PATH_V11)
    if observed != SOURCE_PILOT_V11_SHA256:
        raise RuntimeError(
            f"Corrected Pilot v1.1 changed: {observed} != {SOURCE_PILOT_V11_SHA256}"
        )
    return observed


def _literal_annotation(
    *, request: str, field: str, payload: str, component: str
) -> dict[str, Any]:
    start = request.index(payload)
    return {
        "field": field,
        "spans": [
            {
                "component": component,
                "start": start,
                "end": start + len(payload),
                "text": payload,
            }
        ],
        "deterministic_separator": None,
        "reconstructed_value": payload,
    }


def _apply_human_revision(record: dict[str, Any], number: int) -> None:
    metadata = record["metadata"]
    if number == 30:
        request = (
            "Ask Claude this exact question: "
            "Can `map[key]` mutate when key == 'A/B'?"
        )
        payload = "Can `map[key]` mutate when key == 'A/B'?"
        metadata["raw_user_request"] = request
        metadata["expected_arguments"] = {"question": payload}
        metadata["runtime_arguments"] = {"question": payload}
        metadata["literal_source"] = _literal_annotation(
            request=request,
            field="question",
            payload=payload,
            component="question",
        )
    elif number == 31:
        request = (
            "Create new artwork from this exact prompt: "
            "tiny robot holding a sign: DON'T PANIC"
        )
        payload = "tiny robot holding a sign: DON'T PANIC"
        metadata["raw_user_request"] = request
        metadata["expected_arguments"] = {"prompt": payload}
        metadata["runtime_arguments"] = {"prompt": payload}
        metadata["literal_source"] = _literal_annotation(
            request=request,
            field="prompt",
            payload=payload,
            component="prompt",
        )
    elif number == 36:
        fact = "user does not want meetings before ten"
        metadata["expected_arguments"] = {"fact": fact}
        metadata["runtime_arguments"] = {"fact": fact}
        metadata["semantic_target"] = {
            "field": "fact",
            "canonical_value": fact,
            "accepted_alternatives": [],
        }
    elif number == 94:
        metadata["raw_user_request"] = "check the blue toolbox in the current view"
        metadata["linguistic_features"] = []
    elif number == 117:
        request = (
            "Need the line count, not system stats—run exactly: "
            "wc -l changelog.txt"
        )
        command = "wc -l changelog.txt"
        metadata["raw_user_request"] = request
        metadata["expected_arguments"] = {"command": command}
        metadata["runtime_arguments"] = {"command": command}
        metadata["literal_source"] = _literal_annotation(
            request=request,
            field="command",
            payload=command,
            component="command",
        )
        metadata["linguistic_features"] = [
            "negative_contrast",
            "punctuation_variation",
        ]
    else:
        raise ValueError(f"No approved human revision for record {number}")


def create_unchecked_records_v12() -> list[dict[str, Any]]:
    """Version v1.1 and change semantic content for exactly five rows."""

    verify_v11_safety_constants()
    verify_source_pilot_v11()
    registry = Hammer15TrainingRegistryV11()
    source = load_jsonl(PILOT_PATH_V11)
    if len(source) != 150:
        raise RuntimeError(f"Expected 150 v1.1 source records, found {len(source)}")
    records: list[dict[str, Any]] = []
    for number, source_record in enumerate(source, 1):
        record = deepcopy(source_record)
        record["record_version"] = RECORD_VERSION_V12
        metadata = record["metadata"]
        metadata["record_id"] = f"h15-pilot-v1.2-{number:03d}"
        metadata["generation_provenance"]["generator"] = GENERATOR_VERSION_V12
        metadata["generation_provenance"]["parent_record_version"] = (
            "hammer15-training-record-v1.1"
        )
        metadata["leakage_check"] = {}
        metadata["validation"] = {
            "validator_version": VALIDATOR_VERSION_V12,
            "passed": False,
            "errors": [],
        }
        if number in REVISED_NUMBERS:
            _apply_human_revision(record, number)
            offered = metadata["offered_tools"]
            record["model_visible"] = render_model_visible_v11(
                registry,
                request=metadata["raw_user_request"],
                offered_tools=offered,
                expected_tool=metadata["expected_tool"],
                expected_arguments=metadata["expected_arguments"],
            )
            metadata["prompt_schema_sha256"] = prompt_schema_sha256_v11(
                registry, offered
            )
        records.append(record)
    return records


def _content_view(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        "model_visible": record["model_visible"],
        "raw_user_request": metadata["raw_user_request"],
        "expected_tool": metadata["expected_tool"],
        "expected_arguments": metadata["expected_arguments"],
        "runtime_arguments": metadata["runtime_arguments"],
        "argument_mode": metadata["argument_mode"],
        "category": metadata["category"],
        "confusion_family": metadata["confusion_family"],
        "linguistic_features": metadata["linguistic_features"],
        "offered_tools": metadata["offered_tools"],
        "literal_source": metadata["literal_source"],
        "semantic_target": metadata["semantic_target"],
        "difficulty": metadata["difficulty"],
        "prompt_profile": metadata["prompt_profile"],
        "prompt_schema_sha256": metadata["prompt_schema_sha256"],
    }


def semantic_content_change_numbers(
    source: list[dict[str, Any]], revised: list[dict[str, Any]]
) -> list[int]:
    return [
        number
        for number, (old, new) in enumerate(zip(source, revised), 1)
        if _content_view(old) != _content_view(new)
    ]


def build_records_v12() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    write_record_schema_v12()
    sacred_cases = json.loads(SACRED_DATASET.read_text(encoding="utf-8"))
    registry = Hammer15TrainingRegistryV11()
    records = create_unchecked_records_v12()
    changes = semantic_content_change_numbers(load_jsonl(PILOT_PATH_V11), records)
    if changes != list(REVISED_NUMBERS):
        raise RuntimeError(f"Expected exactly five content changes, observed {changes}")

    rejected: list[dict[str, Any]] = []
    for record in records:
        leakage = check_record(record, sacred_cases)
        record["metadata"]["leakage_check"] = leakage
        errors = record_errors_v12(
            record,
            registry=registry,
            sacred_cases=sacred_cases,
            check_stored_validation=False,
        )
        record["metadata"]["validation"] = {
            "validator_version": VALIDATOR_VERSION_V12,
            "passed": not errors,
            "errors": errors,
        }
        if leakage["status"] == "reject" or errors:
            rejected.append(record)
    return records, rejected


def _manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = [record["metadata"] for record in records]
    return {
        "version": "hammer15-pilot-manifest-v1.2",
        "record_format": RECORD_VERSION_V12,
        "decision": PILOT_DECISION_V12,
        "source_pilot": str(PILOT_PATH_V11),
        "source_pilot_sha256": verify_source_pilot_v11(),
        "original_pilot_v1_sha256": sha256_path(PILOT_PATH),
        "seed_inherited_from_v1.1": DEFAULT_SEED_V11,
        "record_count": len(records),
        "human_review_history": {"keep": 145, "revise": 5, "reject": 0},
        "revised_record_numbers": list(REVISED_NUMBERS),
        "category_counts": dict(sorted(Counter(row["category"] for row in metadata).items())),
        "argument_mode_counts": dict(sorted(Counter(row["argument_mode"] for row in metadata).items())),
        "tool_counts": {
            str(tool): count
            for tool, count in sorted(
                Counter(row["expected_tool"] for row in metadata).items(),
                key=lambda item: str(item[0]),
            )
        },
        "ontology_family_counts": dict(
            sorted(
                Counter(
                    row["confusion_family"]
                    for row in metadata
                    if row["category"] == "ontology_confusion"
                ).items()
            )
        ),
        "offered_tool_count_distribution": {
            str(count): total
            for count, total in sorted(
                Counter(len(row["offered_tools"]) for row in metadata).items()
            )
        },
        "prompt_profile": PROMPT_PROFILE_V11,
        "structured_write_append_preserved": True,
        "metadata_excluded_from_training": True,
        "inference_used": False,
        "models_loaded": False,
        "yuki_tools_executed": False,
        "frozen_450_rerun": False,
        "sacred_dataset_sha256": verify_v11_safety_constants(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build approved Hammer Pilot v1.2")
    parser.add_argument("--output", type=Path, default=PILOT_PATH_V12)
    args = parser.parse_args()
    records, rejected = build_records_v12()
    write_jsonl(PILOT_REJECTED_V12, rejected)
    if rejected:
        raise RuntimeError(
            f"Pilot v1.2 gate rejected {len(rejected)} records; "
            f"inspect {PILOT_REJECTED_V12}"
        )

    write_jsonl(args.output, records)
    leakage = check_records(records)
    validation = validate_records_v12(records)
    write_json(PILOT_LEAKAGE_V12, leakage)
    write_json(PILOT_VALIDATION_V12, validation)
    write_json(PILOT_MANIFEST_V12, _manifest(records))
    if not validation["passed"]:
        raise RuntimeError("Generated Pilot v1.2 failed its independent validator")

    checksums = {
        path.name: sha256_path(path)
        for path in (
            args.output,
            PILOT_MANIFEST_V12,
            PILOT_VALIDATION_V12,
            PILOT_LEAKAGE_V12,
        )
    }
    write_json(args.output.with_suffix(".checksums.json"), checksums)
    print(
        json.dumps(
            {
                "records": len(records),
                "changes": list(REVISED_NUMBERS),
                "leakage": leakage["counts"],
                "validation_passed": validation["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
