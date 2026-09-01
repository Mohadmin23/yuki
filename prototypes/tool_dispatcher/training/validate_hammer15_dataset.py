"""Validate Hammer 1.5B training records without inference or tool execution."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..argument_contract import (
    CONTRACT_VERSION,
    LITERAL_SOURCE,
    NO_ARGUMENT,
    SEMANTIC_ARGUMENT,
    score_contract_arguments,
    tool_argument_contract,
)
from ..backends import normalize_hammer_decoder_text
from ..dialects import HAMMER_FENCED_ARRAY
from ..parsing import parse_model_output
from .check_sacred450_leakage import check_record
from .common import (
    HAMMER_MODEL_ID,
    PILOT_PATH,
    PILOT_VALIDATION,
    RECORD_SCHEMA,
    RECORD_VERSION,
    SACRED_DATASET,
    VALIDATOR_VERSION,
    Hammer15TrainingRegistry,
    load_jsonl,
    prompt_schema_sha256,
    render_model_visible,
    verify_sacred_dataset,
    write_json,
)

EXPECTED_CATEGORY_COUNTS = {
    "ontology_confusion": 60,
    "literal_preservation": 35,
    "semantic_argument": 25,
    "ordinary_easy": 15,
    "adversarial_edge": 15,
}


def _json_schema_errors(record: dict[str, Any]) -> list[str]:
    import jsonschema

    schema = json.loads(RECORD_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"schema:{'/'.join(str(part) for part in error.path) or '<root>'}: "
        f"{error.message}"
        for error in sorted(validator.iter_errors(record), key=lambda item: list(item.path))
    ]


def _target_parse(record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_hammer_decoder_text(
        HAMMER_MODEL_ID,
        record["model_visible"]["assistant_target"],
    )
    return parse_model_output(normalized, "native", HAMMER_FENCED_ARRAY)


def _literal_errors(record: dict[str, Any], field: str) -> list[str]:
    metadata = record["metadata"]
    annotation = metadata["literal_source"]
    errors: list[str] = []
    if not isinstance(annotation, dict):
        return ["literal_source annotation is required"]
    if annotation.get("field") != field:
        errors.append("literal_source field does not match the typed contract")

    request = metadata["raw_user_request"]
    spans = annotation.get("spans", [])
    observed_parts: list[str] = []
    last_end = -1
    for index, span in enumerate(spans):
        start = span.get("start")
        end = span.get("end")
        text = span.get("text")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"literal span {index} has invalid offsets")
            continue
        if start < 0 or end <= start or end > len(request):
            errors.append(f"literal span {index} is outside the raw request")
            continue
        if start < last_end:
            errors.append(f"literal span {index} overlaps or is out of order")
        observed = request[start:end]
        observed_parts.append(observed)
        if observed != text:
            errors.append(f"literal span {index} text does not match its raw slice")
        last_end = end

    contract = tool_argument_contract(metadata["expected_tool"])
    components = contract.get("components")
    if components:
        observed_components = [span.get("component") for span in spans]
        if observed_components != components:
            errors.append("literal component order does not match the typed contract")
        expected_separator = contract["deterministic_separator"]
    else:
        if len(spans) != 1:
            errors.append("single-field literal arguments require exactly one span")
        expected_separator = None
    if annotation.get("deterministic_separator") != expected_separator:
        errors.append("literal separator does not match the typed contract")

    separator = expected_separator or ""
    reconstructed = separator.join(observed_parts)
    if annotation.get("reconstructed_value") != reconstructed:
        errors.append("literal reconstructed_value does not match annotated spans")
    expected_value = metadata["expected_arguments"].get(field)
    if expected_value != reconstructed:
        errors.append("expected literal argument is not the character-exact reconstruction")

    score = score_contract_arguments(
        metadata["expected_tool"],
        metadata["expected_arguments"],
        metadata["expected_arguments"],
        literal_references={field: [reconstructed]},
        declared_version=metadata["argument_contract_version"],
    )
    if not score["contract_correct"]:
        errors.append("literal argument fails the typed-contract scorer")
    return errors


def _semantic_errors(record: dict[str, Any], field: str) -> list[str]:
    metadata = record["metadata"]
    annotation = metadata["semantic_target"]
    contract = tool_argument_contract(metadata["expected_tool"])
    if not metadata["expected_arguments"] and not contract["required"]:
        if annotation is not None:
            return ["omitted optional semantic argument cannot have a target annotation"]
        score = score_contract_arguments(
            metadata["expected_tool"],
            {},
            {},
            declared_version=metadata["argument_contract_version"],
        )
        return [] if score["contract_correct"] else [
            "omitted optional semantic argument fails the typed-contract scorer"
        ]
    if not isinstance(annotation, dict):
        return ["semantic_target annotation is required"]
    errors: list[str] = []
    if annotation.get("field") != field:
        errors.append("semantic_target field does not match the typed contract")
    canonical = annotation.get("canonical_value")
    if metadata["expected_arguments"].get(field) != canonical:
        errors.append("semantic canonical value does not match expected arguments")
    score = score_contract_arguments(
        metadata["expected_tool"],
        metadata["expected_arguments"],
        metadata["expected_arguments"],
        semantic_alternatives={field: annotation.get("accepted_alternatives", [])},
        declared_version=metadata["argument_contract_version"],
    )
    if not score["scoreable"] or not score["contract_correct"]:
        errors.append("semantic argument fails the typed-contract scorer")
    return errors


def record_errors(
    record: dict[str, Any],
    *,
    registry: Hammer15TrainingRegistry,
    sacred_cases: list[dict[str, Any]],
    check_stored_validation: bool = True,
) -> list[str]:
    """Return all deterministic structural and semantic record errors."""

    errors = _json_schema_errors(record)
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return errors + ["metadata is not an object"]
    if record.get("record_version") != RECORD_VERSION:
        errors.append("record_version does not match the v1 format")
    if metadata.get("argument_contract_version") != CONTRACT_VERSION:
        errors.append("argument contract version mismatch")

    offered = metadata.get("offered_tools", [])
    try:
        resolved = list(registry.resolve_names(selected=offered))
    except (TypeError, ValueError) as exc:
        errors.append(f"invalid offered tools: {exc}")
        resolved = []
    if resolved and offered != resolved:
        errors.append("offered tools are not in live-registry order")

    expected_tool = metadata.get("expected_tool")
    expected_arguments = metadata.get("expected_arguments")
    if expected_tool is not None and expected_tool not in offered:
        errors.append("expected tool was not offered to the model")

    if resolved and isinstance(expected_arguments, dict):
        expected_visible = render_model_visible(
            registry,
            request=metadata.get("raw_user_request", ""),
            offered_tools=resolved,
            expected_tool=expected_tool,
            expected_arguments=expected_arguments,
        )
        if record.get("model_visible") != expected_visible:
            errors.append("model-visible prompt or target is not reproducible")
        if metadata.get("prompt_schema_sha256") != prompt_schema_sha256(
            registry, resolved
        ):
            errors.append("prompt schema checksum is not reproducible")

    parsed = _target_parse(record) if "model_visible" in record else None
    if parsed is not None:
        if not parsed["passed"] or parsed["malformed"]:
            errors.extend(f"assistant target: {item}" for item in parsed["errors"])
        elif expected_tool is None:
            if not parsed["rejected"]:
                errors.append("reject record does not emit Hammer's empty-array form")
        else:
            if parsed["rejected"]:
                errors.append("valid call unexpectedly emits a rejection")
            elif parsed["canonical_call"] != {
                "tool": expected_tool,
                "arguments": expected_arguments,
            }:
                errors.append("assistant target differs from the expected call")
            else:
                validation = registry.validate_call(parsed["canonical_call"], offered)
                errors.extend(
                    f"live schema: {item}" for item in validation["errors"]
                )

    mode = metadata.get("argument_mode")
    if expected_tool is None:
        if mode != "reject" or expected_arguments != {}:
            errors.append("reject records require null tool, empty arguments, and reject mode")
        if metadata.get("literal_source") is not None:
            errors.append("reject records cannot have literal_source annotations")
        if metadata.get("semantic_target") is not None:
            errors.append("reject records cannot have semantic_target annotations")
    elif expected_tool in registry.names:
        contract = tool_argument_contract(expected_tool)
        field = contract["argument"]
        if mode != contract["mode"]:
            errors.append("argument mode differs from the typed contract")
        if contract["mode"] == LITERAL_SOURCE:
            errors.extend(_literal_errors(record, field))
            if metadata.get("semantic_target") is not None:
                errors.append("literal record cannot have a semantic_target")
        elif contract["mode"] == SEMANTIC_ARGUMENT:
            errors.extend(_semantic_errors(record, field))
            if metadata.get("literal_source") is not None:
                errors.append("semantic record cannot have a literal_source")
        elif contract["mode"] == NO_ARGUMENT:
            if expected_arguments != {}:
                errors.append("no-argument tool must have empty arguments")
            if metadata.get("literal_source") is not None:
                errors.append("no-argument record cannot have a literal_source")
            if metadata.get("semantic_target") is not None:
                errors.append("no-argument record cannot have a semantic_target")

    if "leakage_check" in metadata:
        observed_leakage = check_record(record, sacred_cases)
        if metadata["leakage_check"] != observed_leakage:
            errors.append("stored leakage result is not reproducible")
        if observed_leakage["status"] == "reject":
            errors.append("record fails the sacred-450 leakage gate")

    if check_stored_validation and "validation" in metadata:
        stored = metadata["validation"]
        expected_validation = {
            "validator_version": VALIDATOR_VERSION,
            "passed": not [error for error in errors if not error.startswith("schema:metadata/validation")],
            "errors": [error for error in errors if not error.startswith("schema:metadata/validation")],
        }
        if stored != expected_validation:
            errors.append("stored validation result is not reproducible")
    return errors


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    sacred_sha = verify_sacred_dataset()
    sacred_cases = json.loads(SACRED_DATASET.read_text(encoding="utf-8"))
    registry = Hammer15TrainingRegistry()
    per_record: dict[str, dict[str, Any]] = {}
    seen: Counter[str] = Counter()

    for index, record in enumerate(records, 1):
        record_id = record.get("metadata", {}).get("record_id", f"line-{index}")
        seen[record_id] += 1
        errors = record_errors(
            record,
            registry=registry,
            sacred_cases=sacred_cases,
        )
        per_record[record_id] = {"passed": not errors, "errors": errors}

    duplicate_ids = sorted(record_id for record_id, count in seen.items() if count > 1)
    categories = Counter(
        record.get("metadata", {}).get("category") for record in records
    )
    tools = Counter(
        record.get("metadata", {}).get("expected_tool") for record in records
    )
    dataset_errors: list[str] = []
    if duplicate_ids:
        dataset_errors.append(f"duplicate record IDs: {duplicate_ids}")
    if len(records) == 150 and dict(categories) != EXPECTED_CATEGORY_COUNTS:
        dataset_errors.append(
            f"pilot category composition differs: {dict(categories)}"
        )
    missing_tools = sorted(set(registry.names) - {tool for tool in tools if tool})
    if missing_tools:
        dataset_errors.append(f"live tools absent from dataset: {missing_tools}")

    failed_records = sum(not result["passed"] for result in per_record.values())
    return {
        "version": VALIDATOR_VERSION,
        "dataset": str(PILOT_PATH),
        "record_count": len(records),
        "passed": not dataset_errors and failed_records == 0,
        "failed_record_count": failed_records,
        "dataset_errors": dataset_errors,
        "category_counts": dict(sorted(categories.items())),
        "tool_counts": {
            str(tool): count for tool, count in sorted(tools.items(), key=lambda item: str(item[0]))
        },
        "sacred_dataset_sha256": sacred_sha,
        "records": per_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Hammer 1.5B dataset")
    parser.add_argument("--input", type=Path, default=PILOT_PATH)
    parser.add_argument("--output", type=Path, default=PILOT_VALIDATION)
    args = parser.parse_args()
    report = validate_records(load_jsonl(args.input))
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
