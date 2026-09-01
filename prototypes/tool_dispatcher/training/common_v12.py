"""Version and schema primitives for the human-approved Pilot v1.2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import TRAINING_ROOT, write_json
from .common_v11 import RECORD_SCHEMA_V11

SOURCE_PILOT_V11_SHA256 = (
    "9bab33fa2e0c1df0701bd51b4da6d8a08940d05d3f88892501ea3ddafa3001ff"
)
RECORD_VERSION_V12 = "hammer15-training-record-v1.2"
GENERATOR_VERSION_V12 = "hammer15-pilot-builder-v1.2"
VALIDATOR_VERSION_V12 = "hammer15-training-validator-v1.2"
PILOT_DECISION_V12 = "APPROVE FOR STAGED SCALING"

RECORD_SCHEMA_V12 = TRAINING_ROOT / "hammer15-training-record-schema-v1.2.json"
PILOT_PATH_V12 = TRAINING_ROOT / "hammer15-pilot-v1.2.jsonl"
PILOT_MANIFEST_V12 = TRAINING_ROOT / "hammer15-pilot-v1.2-manifest.json"
PILOT_VALIDATION_V12 = TRAINING_ROOT / "hammer15-pilot-v1.2-validation.json"
PILOT_LEAKAGE_V12 = TRAINING_ROOT / "hammer15-pilot-v1.2-leakage.json"
PILOT_REJECTED_V12 = TRAINING_ROOT / "hammer15-pilot-v1.2-rejected.jsonl"


def record_schema_v12() -> dict[str, Any]:
    """Derive v1.2's unchanged structure with versioned const fields."""

    import json

    schema = json.loads(RECORD_SCHEMA_V11.read_text(encoding="utf-8"))
    schema["$id"] = "https://local.yuki/hammer15-training-record-schema-v1.2.json"
    schema["title"] = "Yuki Hammer 1.5B training record v1.2"
    schema["properties"]["record_version"]["const"] = RECORD_VERSION_V12
    metadata = schema["properties"]["metadata"]["properties"]
    metadata["record_id"]["pattern"] = r"^h15-pilot-v1\.2-[0-9]{3}$"
    provenance = schema["$defs"]["provenance"]["properties"]
    provenance["generator"]["const"] = GENERATOR_VERSION_V12
    provenance["parent_record_version"]["const"] = (
        "hammer15-training-record-v1.1"
    )
    schema["$defs"]["validation"]["properties"]["validator_version"][
        "const"
    ] = VALIDATOR_VERSION_V12
    return schema


def write_record_schema_v12(path: Path = RECORD_SCHEMA_V12) -> None:
    write_json(path, record_schema_v12())


def downgrade_record_to_v11(record: dict[str, Any]) -> dict[str, Any]:
    """Create a validation-only v1.1 view; model-visible content is untouched."""

    downgraded = deepcopy(record)
    downgraded["record_version"] = "hammer15-training-record-v1.1"
    metadata = downgraded["metadata"]
    metadata["record_id"] = metadata["record_id"].replace(
        "h15-pilot-v1.2-", "h15-pilot-v1.1-"
    )
    provenance = metadata["generation_provenance"]
    provenance["generator"] = "hammer15-pilot-builder-v1.1"
    provenance["parent_record_version"] = "hammer15-training-record-v1"
    metadata["validation"]["validator_version"] = (
        "hammer15-training-validator-v1.1"
    )
    return downgraded
