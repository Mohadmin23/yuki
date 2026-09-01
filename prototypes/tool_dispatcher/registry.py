"""Read Yuki's live registry and adapt it for strict dispatcher validation."""
from __future__ import annotations

import inspect
import json
from collections.abc import Iterable
from copy import deepcopy
from importlib import import_module
from typing import Any

import tools

from .contracts import ToolSpec
from .dialects import (
    ARCH_XML_OBJECT,
    HAMMER_FENCED_ARRAY,
    OPENAI_NATIVE_TOOL_CALLS,
    TOOL_CALL_ARRAY,
    XLAM_V1_ENVELOPE,
)

TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "information": ("time", "weather", "fetch", "search", "calc"),
    "computer": ("hardware", "read", "shell"),
    "media": ("see", "image"),
    "yuki-files": (
        "yuki_write",
        "yuki_read",
        "yuki_list",
        "yuki_delete",
        "yuki_append",
    ),
    "memory": ("remember", "recall"),
    "external-agent": ("ask_claude",),
}


class ToolRegistry:
    """A read-only view over Yuki's real tool registry."""

    def __init__(self) -> None:
        source_schemas = {
            item["function"]["name"]: item for item in tools.OPENAI_TOOL_SCHEMAS
        }
        self._specs: dict[str, ToolSpec] = {}
        self._order: list[str] = []

        for module_name in tools._TOOL_NAMES:
            module = import_module(f"tools.{module_name}")
            meta = module.META
            name = meta["react_name"]
            fn = module.tool_fn
            source_schema = deepcopy(source_schemas[name])
            argument_schema = self._strict_arguments(source_schema, meta, fn)
            spec = ToolSpec(
                name=name,
                cli_name=meta["cli_name"],
                description=meta["description"],
                param_name=meta.get("param_name", "arg"),
                fn=fn,
                source_schema=source_schema,
                argument_schema=argument_schema,
            )
            if name in self._specs:
                raise RuntimeError(f"Duplicate Yuki tool name: {name}")
            if tools.REACT_TOOL_MAP.get(name) is not fn:
                raise RuntimeError(f"Yuki registry implementation mismatch for {name}")
            self._specs[name] = spec
            self._order.append(name)

        self._validate_groups()

    @staticmethod
    def _strict_arguments(
        source_schema: dict[str, Any], meta: dict[str, Any], fn: Any
    ) -> dict[str, Any]:
        schema = deepcopy(source_schema["function"]["parameters"])
        schema["additionalProperties"] = False
        param_name = meta.get("param_name", "arg")
        if param_name is None:
            schema["required"] = []
            return schema

        parameters = list(inspect.signature(fn).parameters.values())
        takes_required_input = bool(
            parameters and parameters[0].default is inspect.Parameter.empty
        )
        schema["required"] = [param_name] if takes_required_input else []
        if takes_required_input:
            schema["properties"][param_name]["minLength"] = 1
        return schema

    def _validate_groups(self) -> None:
        known = set(self._specs)
        for group, names in TOOL_GROUPS.items():
            missing = set(names) - known
            if missing:
                raise RuntimeError(
                    f"Tool group {group!r} references missing tools: {sorted(missing)}"
                )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._order)

    @property
    def groups(self) -> dict[str, tuple[str, ...]]:
        return {"all": self.names, **TOOL_GROUPS}

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def resolve_names(
        self, group: str = "all", selected: Iterable[str] | None = None
    ) -> tuple[str, ...]:
        if selected is not None:
            requested = set(selected)
            unknown = requested - set(self._specs)
            if unknown:
                raise ValueError(f"Unknown tools: {sorted(unknown)}")
            names = tuple(name for name in self._order if name in requested)
            if not names:
                raise ValueError("At least one tool must be selected")
            return names
        if group == "all":
            return self.names
        if group not in TOOL_GROUPS:
            raise ValueError(f"Unknown tool group: {group}")
        return TOOL_GROUPS[group]

    def strict_openai_schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        result = []
        for name in names:
            spec = self._specs[name]
            schema = deepcopy(spec.source_schema)
            schema["function"]["parameters"] = deepcopy(spec.argument_schema)
            result.append(schema)
        return result

    def xlam_schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """Render strict schemas in xLAM's concise recommended tool format."""

        result = []
        for name in names:
            spec = self._specs[name]
            result.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": deepcopy(spec.argument_schema["properties"]),
                    "required": list(spec.argument_schema["required"]),
                    "additionalProperties": False,
                }
            )
        return result

    def hammer_schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """Render the concise schema shape used by Hammer's official client."""

        result = []
        for name in names:
            spec = self._specs[name]
            parameters = deepcopy(spec.argument_schema["properties"])
            for required_name in spec.argument_schema["required"]:
                parameters[required_name]["required"] = True
            result.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": parameters,
                }
            )
        return result

    def output_schema(
        self, names: Iterable[str], output_mode: str, native_dialect: str
    ) -> dict[str, Any]:
        call_variants = []
        for name in names:
            spec = self._specs[name]
            if output_mode == "native":
                call_variants.append(
                    {
                        "type": "object",
                        "properties": {
                            "name": {"const": name},
                            "arguments": deepcopy(spec.argument_schema),
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                    }
                )
            elif output_mode == "canonical":
                call_variants.append(
                    {
                        "type": "object",
                        "properties": {
                            "tool": {"const": name},
                            "arguments": deepcopy(spec.argument_schema),
                        },
                        "required": ["tool", "arguments"],
                        "additionalProperties": False,
                    }
                )
            else:
                raise ValueError(f"Unknown output mode: {output_mode}")

        if output_mode == "native" and native_dialect in {
            TOOL_CALL_ARRAY,
            OPENAI_NATIVE_TOOL_CALLS,
            HAMMER_FENCED_ARRAY,
        }:
            return {
                "type": "array",
                "minItems": 0,
                "maxItems": 1,
                "items": {"oneOf": call_variants},
            }
        if output_mode == "native" and native_dialect == XLAM_V1_ENVELOPE:
            return {
                "type": "object",
                "properties": {
                    "tool_calls": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 1,
                        "items": {"oneOf": call_variants},
                    }
                },
                "required": ["tool_calls"],
                "additionalProperties": False,
            }
        if output_mode == "native" and native_dialect == ARCH_XML_OBJECT:
            return {"oneOf": call_variants}
        if output_mode == "native":
            raise ValueError(f"Unknown native output dialect: {native_dialect}")
        return {
            "oneOf": call_variants
            + [
                {
                    "type": "object",
                    "properties": {
                        "tool": {"const": "no_tool"},
                        "arguments": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                    "required": ["tool", "arguments"],
                    "additionalProperties": False,
                }
            ]
        }

    def validate_call(
        self, call: dict[str, Any], available_names: Iterable[str]
    ) -> dict[str, Any]:
        errors: list[str] = []
        if not isinstance(call, dict):
            return {"passed": False, "errors": ["Call must be a JSON object."]}
        if set(call) != {"tool", "arguments"}:
            errors.append("Call must contain exactly 'tool' and 'arguments'.")
        name = call.get("tool")
        arguments = call.get("arguments")
        available = set(available_names)
        if not isinstance(name, str):
            errors.append("Tool name must be a string.")
        elif name not in self._specs:
            errors.append(f"Unknown tool: {name}")
        elif name not in available:
            errors.append(f"Tool {name!r} was not offered to the model.")
        if not isinstance(arguments, dict):
            errors.append("Arguments must be a JSON object.")
        elif isinstance(name, str) and name in self._specs:
            errors.extend(self._validate_arguments(self._specs[name].argument_schema, arguments))
        return {"passed": not errors, "errors": errors}

    @staticmethod
    def _validate_arguments(
        schema: dict[str, Any], arguments: dict[str, Any]
    ) -> list[str]:
        errors: list[str] = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in arguments:
                errors.append(f"Missing required argument: {key}")
        if schema.get("additionalProperties") is False:
            extras = set(arguments) - set(properties)
            if extras:
                errors.append(f"Unexpected arguments: {sorted(extras)}")
        for key, value in arguments.items():
            rule = properties.get(key)
            if rule is None:
                continue
            if rule.get("type") == "string" and not isinstance(value, str):
                errors.append(f"Argument {key!r} must be a string.")
                continue
            if isinstance(value, str) and len(value) < rule.get("minLength", 0):
                errors.append(f"Argument {key!r} must not be empty.")
            if "enum" in rule and value not in rule["enum"]:
                errors.append(f"Argument {key!r} must be one of {rule['enum']!r}.")
        return errors

    def regex_call(self, request: str) -> dict[str, Any] | None:
        """Run Yuki's existing regex rules for an optional comparison baseline."""

        cli_to_spec = {spec.cli_name: spec for spec in self._specs.values()}
        for entry in tools.AUTO_DETECT_REGEX:
            match = entry["pattern"].search(request)
            if not match:
                continue
            spec = cli_to_spec.get(entry["tool"])
            if spec is None:
                return None
            if "group" in entry:
                argument = match.group(entry["group"]).strip()
            else:
                argument = entry.get("arg", "")
            arguments = {} if spec.param_name is None else {spec.param_name: argument}
            return {"tool": spec.name, "arguments": arguments}
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "count": len(self._order),
            "names": list(self._order),
            "groups": {key: list(value) for key, value in self.groups.items()},
            "strict_schemas": self.strict_openai_schemas(self._order),
        }

    def dumps_call(self, call: dict[str, Any]) -> str:
        return json.dumps(call, ensure_ascii=False, sort_keys=True)
