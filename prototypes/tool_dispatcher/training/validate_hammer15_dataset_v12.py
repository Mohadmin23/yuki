"""Validate Pilot v1.2 without model inference or Yuki tool execution."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import SACRED_DATASET, load_jsonl, write_json
from .common_v11 import Hammer15TrainingRegistryV11
from .common_v12 import (
    PILOT_PATH_V12,
    PILOT_VALIDATION_V12,
    RECORD_SCHEMA_V12,
    RECORD_VERSION_V12,
    VALIDATOR_VERSION_V12,
    downgrade_record_to_v11,
    record_schema_v12,
)
from .validate_hammer15_dataset_v11 import (
    record_errors_v11,
    validate_records_v11,
)


def _json_schema_errors_v12(record: dict[str, Any]) -> list[str]:
    import jsonschema

    schema = (
        json.loads(RECORD_SCHEMA_V12.read_text(encoding="utf-8"))
        if RECORD_SCHEMA_V12.exists()
        else record_schema_v12()
    )
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(record), key=lambda item: str(list(item.path)))
    return [
        f"schema:{'/'.join(str(part) for part in error.path) or '<root>'}: "
        f"{error.message}"
        for error in errors
    ]


def record_errors_v12(
    record: dict[str, Any],
    *,
    registry: Hammer15TrainingRegistryV11,
    sacred_cases: list[dict[str, Any]],
    check_stored_validation: bool = True,
) -> list[str]:
    errors = _json_schema_errors_v12(record)
    if record.get("record_version") != RECORD_VERSION_V12:
        errors.append("record_version does not match v1.2")

    downgraded = downgrade_record_to_v11(record)
    errors.extend(
        record_errors_v11(
            downgraded,
            registry=registry,
            sacred_cases=sacred_cases,
            check_stored_validation=False,
        )
    )

    metadata = record.get("metadata", {})
    if check_stored_validation and isinstance(metadata, dict):
        stored = metadata.get("validation")
        stable_errors = [
            error
            for error in errors
            if not error.startswith("schema:metadata/validation")
        ]
        expected = {
            "validator_version": VALIDATOR_VERSION_V12,
            "passed": not stable_errors,
            "errors": stable_errors,
        }
        if stored != expected:
            errors.append("stored v1.2 validation result is not reproducible")
    return errors


def validate_records_v12(records: list[dict[str, Any]]) -> dict[str, Any]:
    sacred_cases = json.loads(SACRED_DATASET.read_text(encoding="utf-8"))
    registry = Hammer15TrainingRegistryV11()
    per_record: dict[str, dict[str, Any]] = {}
    seen: Counter[str] = Counter()
    for index, record in enumerate(records, 1):
        record_id = record.get("metadata", {}).get("record_id", f"line-{index}")
        seen[record_id] += 1
        errors = record_errors_v12(
            record,
            registry=registry,
            sacred_cases=sacred_cases,
        )
        per_record[record_id] = {"passed": not errors, "errors": errors}

    base_report = validate_records_v11(
        [downgrade_record_to_v11(record) for record in records]
    )
    dataset_errors = list(base_report["dataset_errors"])
    duplicates = sorted(record_id for record_id, count in seen.items() if count > 1)
    if duplicates:
        dataset_errors.append(f"duplicate v1.2 record IDs: {duplicates}")
    failed_records = sum(not result["passed"] for result in per_record.values())
    return {
        "version": VALIDATOR_VERSION_V12,
        "dataset": str(PILOT_PATH_V12),
        "record_count": len(records),
        "passed": not dataset_errors and failed_records == 0 and base_report["passed"],
        "failed_record_count": failed_records,
        "dataset_errors": dataset_errors,
        "category_counts": base_report["category_counts"],
        "tool_counts": base_report["tool_counts"],
        "linguistic_feature_counts": base_report["linguistic_feature_counts"],
        "ontology_family_counts": base_report["ontology_family_counts"],
        "leakage_status_counts": base_report["leakage_status_counts"],
        "sacred_dataset_sha256": base_report["sacred_dataset_sha256"],
        "records": per_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate approved Hammer Pilot v1.2")
    parser.add_argument("--input", type=Path, default=PILOT_PATH_V12)
    parser.add_argument("--output", type=Path, default=PILOT_VALIDATION_V12)
    args = parser.parse_args()
    report = validate_records_v12(load_jsonl(args.input))
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "records": report["record_count"],
                "failed": report["failed_record_count"],
            },
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
