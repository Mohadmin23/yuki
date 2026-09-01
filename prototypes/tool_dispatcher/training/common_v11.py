"""Pilot v1.1 primitives, including structured Yuki file-write arguments."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from ..dialects import HAMMER_FENCED_ARRAY
from ..prompting import build_prompt
from .common import (
    HAMMER_MODEL_ID,
    SACRED_SHA256,
    TRAINING_ROOT,
    TRAINING_TASK_INSTRUCTION,
    Hammer15TrainingRegistry,
    verify_sacred_dataset,
)

COMPOSITE_TOOLS = frozenset({"yuki_write", "yuki_append"})
STRUCTURED_ARGUMENT_PROFILE = "structured-yuki-file-content-v1"
RECORD_VERSION_V11 = "hammer15-training-record-v1.1"
GENERATOR_VERSION_V11 = "hammer15-pilot-builder-v1.1"
VALIDATOR_VERSION_V11 = "hammer15-training-validator-v1.1"
PROMPT_PROFILE_V11 = "hammer21-yuki-typed-a1-ontology-structured-files-v1.1"
DEFAULT_SEED_V11 = 151_825

HARDWARE_DESCRIPTION_V11 = (
    "Report local machine hardware and runtime resource measurements. "
    "Use metric 'process' only for Yuki's own current LLM/runtime process. "
    "Use metric 'top' for a system-wide ranked list of the busiest processes."
)
HARDWARE_METRIC_DESCRIPTION_V11 = (
    "Canonical metric to report: cpu = whole-system processor usage; "
    "ram = whole-system memory usage; disk = root-volume usage; "
    "process = only Yuki's own current LLM/runtime process (PID, RSS, CPU, "
    "and threads); top = top five processes system-wide by CPU and RAM; "
    "gpu = GPU/ANE availability; temp = CPU temperature; all = complete report. "
    "Extract the canonical semantic value; do not copy surrounding request wording."
)

RECORD_SCHEMA_V11 = TRAINING_ROOT / "hammer15-training-record-schema-v1.1.json"
PILOT_PATH_V11 = TRAINING_ROOT / "hammer15-pilot-v1.1.jsonl"
PILOT_MANIFEST_V11 = TRAINING_ROOT / "hammer15-pilot-v1.1-manifest.json"
PILOT_VALIDATION_V11 = TRAINING_ROOT / "hammer15-pilot-v1.1-validation.json"
PILOT_LEAKAGE_V11 = TRAINING_ROOT / "hammer15-pilot-v1.1-leakage.json"
PILOT_REJECTED_V11 = TRAINING_ROOT / "hammer15-pilot-v1.1-rejected.jsonl"

TRAINING_TASK_INSTRUCTION_V11 = f"""{TRAINING_TASK_INSTRUCTION}
When a schema exposes separate filename and content fields, copy each field from its own literal source span. Never join those fields yourself."""


def _structured_file_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Filename in Yuki's internal store. Copy it character-for-character "
                    "from the query. A vertical bar is not permitted in a filename."
                ),
            },
            "content": {
                "type": "string",
                "description": (
                    "File content. Copy it character-for-character from the query, "
                    "including whitespace and any vertical bars."
                ),
            },
        },
        "required": ["filename", "content"],
        "additionalProperties": False,
    }


class Hammer15TrainingRegistryV11(Hammer15TrainingRegistry):
    """Training-only structured adapter over the unchanged live registry."""

    def hammer_schemas(self, names: Any) -> list[dict[str, Any]]:
        schemas = super().hammer_schemas(names)
        structured = _structured_file_schema()
        for schema in schemas:
            if schema["name"] in COMPOSITE_TOOLS:
                schema["parameters"] = deepcopy(structured["properties"])
                for field in structured["required"]:
                    schema["parameters"][field]["required"] = True
            elif schema["name"] == "hardware":
                schema["description"] = HARDWARE_DESCRIPTION_V11
                schema["parameters"]["metric"]["description"] = (
                    HARDWARE_METRIC_DESCRIPTION_V11
                )
        return schemas

    def output_schema(
        self, names: Any, output_mode: str, native_dialect: str
    ) -> dict[str, Any]:
        schema = super().output_schema(names, output_mode, native_dialect)
        if output_mode != "native" or native_dialect != HAMMER_FENCED_ARRAY:
            return schema
        for variant in schema["items"]["oneOf"]:
            name = variant["properties"]["name"].get("const")
            if name in COMPOSITE_TOOLS:
                variant["properties"]["arguments"] = _structured_file_schema()
        return schema

    def validate_training_call(
        self, call: dict[str, Any], available_names: list[str]
    ) -> dict[str, Any]:
        name = call.get("tool") if isinstance(call, dict) else None
        if name not in COMPOSITE_TOOLS:
            return self.validate_call(call, available_names)
        errors: list[str] = []
        if set(call) != {"tool", "arguments"}:
            errors.append("Call must contain exactly 'tool' and 'arguments'.")
        if name not in available_names:
            errors.append(f"Tool {name!r} was not offered to the model.")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            errors.append("Arguments must be a JSON object.")
            return {"passed": not errors, "errors": errors}
        errors.extend(self._validate_arguments(_structured_file_schema(), arguments))
        filename = arguments.get("filename")
        if isinstance(filename, str) and "|" in filename:
            errors.append("A Yuki filename cannot contain '|'.")
        return {"passed": not errors, "errors": errors}

    def adapt_training_call_to_live(self, call: dict[str, Any]) -> dict[str, Any]:
        """Normalize a validated training call to the unchanged live schema."""

        validation = self.validate_training_call(call, list(self.names))
        if not validation["passed"]:
            raise ValueError("Invalid training call: " + "; ".join(validation["errors"]))
        name = call["tool"]
        if name not in COMPOSITE_TOOLS:
            return deepcopy(call)
        arguments = call["arguments"]
        return {
            "tool": name,
            "arguments": {
                "filename_and_content": (
                    f"{arguments['filename']}|{arguments['content']}"
                )
            },
        }


def hammer_native_target_v11(tool: str | None, arguments: dict[str, Any]) -> str:
    calls: list[dict[str, Any]] = []
    if tool is not None:
        calls.append({"name": tool, "arguments": arguments})
    payload = json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
    return f"```\n{payload}"


def render_model_visible_v11(
    registry: Hammer15TrainingRegistryV11,
    *,
    request: str,
    offered_tools: list[str],
    expected_tool: str | None,
    expected_arguments: dict[str, Any],
) -> dict[str, str]:
    package = build_prompt(
        registry,
        request,
        offered_tools,
        output_mode="native",
        model_id=HAMMER_MODEL_ID,
        task_instruction=TRAINING_TASK_INSTRUCTION_V11,
    )
    if not package.pre_rendered or package.native_dialect != HAMMER_FENCED_ARRAY:
        raise RuntimeError("Hammer 1.5B v1.1 prompt did not use its native path")
    return {
        "prompt": package.content,
        "assistant_target": hammer_native_target_v11(
            expected_tool, expected_arguments
        ),
    }


def prompt_schema_sha256_v11(
    registry: Hammer15TrainingRegistryV11, offered_tools: list[str]
) -> str:
    encoded = json.dumps(
        registry.hammer_schemas(offered_tools),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_v11_safety_constants() -> str:
    observed = verify_sacred_dataset()
    if observed != SACRED_SHA256:
        raise RuntimeError("Pilot v1.1 sacred checksum invariant failed")
    return observed
