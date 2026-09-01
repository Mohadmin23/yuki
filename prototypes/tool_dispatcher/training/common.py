"""Shared v1 training-record primitives with no model or tool execution path."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..argument_contract import (
    LITERAL_SOURCE,
    SEMANTIC_ARGUMENT,
    tool_argument_contract,
    validate_contract_against_registry,
)
from ..focused_repair import REPAIRED_SCHEMA_DESCRIPTIONS
from ..hammer20_ontology_repair import ONTOLOGY_DESCRIPTION_REPAIRS
from ..prompting import TASK_INSTRUCTION, build_prompt
from ..registry import ToolRegistry

ROOT = Path(__file__).parents[1]
TRAINING_ROOT = Path(__file__).parent
SACRED_DATASET = ROOT / "phase2-cases.json"
SACRED_SHA256 = "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466"
RECORD_SCHEMA = TRAINING_ROOT / "hammer15-training-record-schema-v1.json"
PILOT_PATH = TRAINING_ROOT / "hammer15-pilot-v1.jsonl"
PILOT_MANIFEST = TRAINING_ROOT / "hammer15-pilot-v1-manifest.json"
PILOT_VALIDATION = TRAINING_ROOT / "hammer15-pilot-v1-validation.json"
PILOT_LEAKAGE = TRAINING_ROOT / "hammer15-pilot-v1-leakage.json"
PILOT_REJECTED = TRAINING_ROOT / "hammer15-pilot-v1-rejected.jsonl"

RECORD_VERSION = "hammer15-training-record-v1"
GENERATOR_VERSION = "hammer15-pilot-builder-v1"
VALIDATOR_VERSION = "hammer15-training-validator-v1"
LEAKAGE_VERSION = "sacred450-leakage-v1"
PROMPT_PROFILE = "hammer21-yuki-typed-a1-ontology-v1"
HAMMER_MODEL_ID = "MadeAgents/Hammer2.1-1.5b"
HAMMER_NATIVE_DIALECT = "hammer_fenced_tool_call_array"
DEFAULT_SEED = 150_825

TRAINING_TASK_INSTRUCTION = f"""{TASK_INSTRUCTION}
Follow each parameter's transport instruction. For literal-source parameters, copy the smallest complete payload from the query character-for-character. For semantic parameters, emit the canonical meaning described by the schema instead of copying request scaffolding."""


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sacred_dataset() -> str:
    observed = sha256_path(SACRED_DATASET)
    if observed != SACRED_SHA256:
        raise RuntimeError(
            f"Sacred Phase 2 dataset changed: {observed} != {SACRED_SHA256}"
        )
    return observed


class Hammer15TrainingRegistry(ToolRegistry):
    """Live registry with the selected prototype descriptions and typed fields."""

    def __init__(self) -> None:
        super().__init__()
        validate_contract_against_registry(self)

    def hammer_schemas(self, names: Any) -> list[dict[str, Any]]:
        schemas = super().hammer_schemas(names)
        for schema in schemas:
            name = schema["name"]
            focused = REPAIRED_SCHEMA_DESCRIPTIONS.get(name)
            if focused is not None:
                schema["description"] = focused["description"]
                for field, description in focused["parameters"].items():
                    schema["parameters"][field]["description"] = description
            ontology = ONTOLOGY_DESCRIPTION_REPAIRS.get(name)
            if ontology is not None:
                schema["description"] = ontology

            contract = tool_argument_contract(name)
            field = contract["argument"]
            if field is None:
                continue
            current = schema["parameters"][field].get("description", "").rstrip()
            if contract["mode"] == LITERAL_SOURCE:
                transport = (
                    " Copy the minimal argument payload from the query "
                    "character-for-character; do not rewrite or normalize it."
                )
            elif contract["mode"] == SEMANTIC_ARGUMENT:
                transport = (
                    " Extract the canonical semantic value; do not copy surrounding "
                    "request wording."
                )
            else:
                transport = ""
            schema["parameters"][field]["description"] = current + transport
        return schemas


def ordered_tools(registry: ToolRegistry, names: list[str] | tuple[str, ...]) -> list[str]:
    return list(registry.resolve_names(selected=names))


def hammer_native_target(tool: str | None, arguments: dict[str, Any]) -> str:
    calls: list[dict[str, Any]] = []
    if tool is not None:
        calls.append({"name": tool, "arguments": arguments})
    payload = json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
    return f"```\n{payload}"


def render_model_visible(
    registry: ToolRegistry,
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
        task_instruction=TRAINING_TASK_INSTRUCTION,
    )
    if not package.pre_rendered or package.native_dialect != HAMMER_NATIVE_DIALECT:
        raise RuntimeError("Hammer 1.5B training prompt did not use its native path")
    return {
        "prompt": package.content,
        "assistant_target": hammer_native_target(expected_tool, expected_arguments),
    }


def prompt_schema_sha256(registry: ToolRegistry, offered_tools: list[str]) -> str:
    encoded = json.dumps(
        registry.hammer_schemas(offered_tools),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def model_visible_only(record: dict[str, Any]) -> dict[str, str]:
    """Return exactly the fields permitted to enter supervised training."""

    visible = record["model_visible"]
    if set(visible) != {"prompt", "assistant_target"}:
        raise RuntimeError("Unexpected model-visible training fields")
    return deepcopy(visible)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL line {line_number}: {exc}") from exc
    return rows
