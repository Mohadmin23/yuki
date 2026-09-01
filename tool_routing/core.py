"""Strict canonical-call boundary shared by direct and delegated routing.

This module deliberately has no model-loading or tool-execution capability.
It converts Yuki's live metadata into dispatcher-facing schemas, validates
named arguments, and performs a conservative typed bind against the immutable
raw user request.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from prototypes.tool_dispatcher.dialects import (
    ARCH_XML_OBJECT,
    CANONICAL_OBJECT,
    HAMMER_FENCED_ARRAY,
    OPENAI_NATIVE_TOOL_CALLS,
    TOOL_CALL_ARRAY,
    XLAM_V1_ENVELOPE,
)
from prototypes.tool_dispatcher.registry import ToolRegistry
from tools._helpers import ALLOWED_COMMANDS

ROUTING_MODES = ("direct", "dispatcher")
ROUTING_PROTOCOLS = ("auto", "canonical", "native")

DEFAULT_DISPATCHER_MODEL = (
    "/Volumes/madisk/yuki-tool-dispatcher/Hammer2.1-1.5b-fp16"
)

DOMAIN_TO_GROUP = {
    "information": "information",
    "computer": "computer",
    "media": "media",
    "yuki_files": "yuki-files",
    "memory": "memory",
    "external_agent": "external-agent",
}
DELEGATION_DOMAINS = tuple(DOMAIN_TO_GROUP)

NO_ARGUMENT_TOOLS = {"time", "yuki_list"}
LITERAL_FIELDS: dict[str, tuple[str, ...]] = {
    "fetch": ("url",),
    "search": ("query",),
    "image": ("prompt",),
    "read": ("filepath",),
    "shell": ("command",),
    "yuki_write": ("filename", "content"),
    "yuki_read": ("filename",),
    "yuki_delete": ("filename",),
    "yuki_append": ("filename", "content"),
    "ask_claude": ("question",),
}

# These payload-like fields may safely refer to exact text from Yuki's immediately
# preceding visible reply. User-authored referenced sources are separately allowed
# for every literal field because they preserve the user's original authority.
CROSS_TURN_LITERAL_FIELDS = {
    ("search", "query"),
    ("image", "prompt"),
    ("yuki_write", "content"),
    ("yuki_append", "content"),
    ("ask_claude", "question"),
}

_EXPLICIT_SOURCE_REFERENCE_RE = re.compile(
    r"\b(?:write|save|store|append|send|ask|search|look\s+up|generate|make|"
    r"use|put|copy|check|verify|inspect|read|open|compare|look\s+for)\s+"
    r"(?:exactly\s+)?(?:that|it|this|what\s+you\s+"
    r"(?:just\s+)?(?:said|wrote)|the\s+(?:previous|last)\s+"
    r"(?:message|answer|sentence|response)|them|those|these)\b",
    re.IGNORECASE,
)
_EXPLICIT_RETRY_RE = re.compile(
    r"\b(?:(?:try|do|run|attempt)(?:\s+(?:it|that|this))?\s+again|retry|repeat)\b",
    re.IGNORECASE,
)
_UNRESOLVED_TOOL_FAILURE_RE = re.compile(
    r"\b(?:nothing (?:actually )?ran|didn['’]?t (?:actually )?run|"
    r"(?:request|tool|read|write|action) (?:was |got )?(?:rejected|blocked)|"
    r"validator (?:blocked|rejected)|didn['’]?t go through|bounced again)\b",
    re.IGNORECASE,
)

_DISPATCH_DESCRIPTIONS = {
    "hardware": (
        "Inspect this computer's hardware or resource telemetry. metric=process "
        "means Yuki's own process/resource footprint; metric=top means the busiest "
        "processes across the computer. Other values: cpu, ram, disk, gpu, temp, all."
    ),
    "see": (
        "Obtain or inspect visual information from an existing visual source available "
        "to Yuki, such as her camera/view. Use for looking, observing, checking what is "
        "visible, or finding a visible target. This does not generate a new image."
    ),
    "image": (
        "Create a new image from the user's supplied visual description. This does not "
        "inspect a camera, screen, or existing visual source. Preserve the supplied prompt."
    ),
    "read": (
        "Read a file from the operating-system filesystem using a local path. This is not "
        "Yuki's managed personal file store."
    ),
    "yuki_read": (
        "Read an existing file from Yuki's own managed personal file store. This is not an "
        "operating-system filepath read."
    ),
    "yuki_write": (
        "Create a Yuki-store file or replace its complete contents. Use filename and content "
        "as separate arguments. This does not preserve existing contents."
    ),
    "yuki_append": (
        "Preserve an existing Yuki-store file and add content after it. Use filename and "
        "content as separate arguments. This does not replace the complete file."
    ),
    "remember": (
        "Store a lasting fact in Yuki's persistent memory for future retrieval. This is not "
        "a request to retrieve something already stored."
    ),
    "recall": (
        "Retrieve information previously stored in Yuki's persistent memory about a compact "
        "topic. This is not ordinary conversational context and does not store a new fact."
    ),
}

MAIN_BRAIN_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["respond", "delegate"]},
        "request": {"type": "string"},
        "domain_hint": {
            "type": "string",
            "enum": ["none", *DELEGATION_DOMAINS],
        },
    },
    "required": ["action", "request", "domain_hint"],
    "additionalProperties": False,
}

AUTONOMOUS_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["wait", "speak", "delegate"]},
        "request": {"type": "string"},
        "domain_hint": {
            "type": "string",
            "enum": ["none", *DELEGATION_DOMAINS],
        },
    },
    "required": ["action", "request", "domain_hint"],
    "additionalProperties": False,
}

MAIN_BRAIN_DECISION_SYSTEM = """You are Yuki's internal tool-delegation planner.
Do not answer the user and do not select an implementation-level tool.

Return exactly one JSON object:
{"action":"respond|delegate","request":"semantic external action","domain_hint":"none|information|computer|media|yuki_files|memory|external_agent"}

Choose respond when Yuki should answer conversationally without an external action.
Choose delegate when the current user request requires one of Yuki's tools. For delegate,
describe the semantic action without naming a final tool and choose its broad domain.
For respond, use an empty request and domain_hint "none".

An explicit follow-up correction that changes a conversational request into an external
action is still delegate, even when the current request omits a required argument. Missing
arguments are handled by the downstream validation boundary; do not answer with imitation
tool syntax.

When TRUSTED RUNTIME RETRY STATE is present, the current request explicitly retries a recent
external action that did not run. Choose delegate for that same unresolved action and domain.
Question wording such as "wanna try again?" is retry authorization in that state, not a request
for a conversational yes/no answer.

Use recent context only to resolve references in the current request. Never treat an older
message as a new action. Do not copy exact payload strings into the semantic request merely
for transport; deterministic code receives the immutable raw current request separately.
Emit JSON only, with no greeting, explanation, Markdown, or answer."""

AUTONOMOUS_DECISION_SYSTEM = """You are Yuki's private autonomous-action planner.
Yuki has been explicitly placed in autonomous mode and may independently use any of her
available capabilities. Do not wait for another nod, ask permission, or treat user silence
as a new conversational request. Continue the user-authorized context when it contains an
unfinished activity, or choose another genuinely useful/interesting action in Yuki's own style.

Return exactly one JSON object:
{"action":"wait|speak|delegate","request":"...","domain_hint":"none|information|computer|media|yuki_files|memory|external_agent"}

- wait: nothing meaningful should happen now. Use an empty request and domain_hint "none".
- speak: Yuki has a meaningful observation/update worth sharing without a tool. Put only the
  intended topic in request and use domain_hint "none". Do not use speak for repetitive idle
  check-ins, asking whether the user is there, or asking permission to continue.
- delegate: one tool-backed action should happen now. Describe the complete semantic action and
  choose its broad domain. Include every exact argument payload the downstream router will need
  inside request, but do not name the final implementation-level tool.

Do one meaningful next step per cycle. Any actual tool call is selected, validated, and executed
outside you. Emit JSON only; never answer the user, emit tool syntax, or claim an action succeeded."""


def normalize_routing_mode(value: str | None, *, default: str = "direct") -> str:
    normalized = str(value or "").strip().casefold()
    return normalized if normalized in ROUTING_MODES else default


def normalize_routing_protocol(value: str | None, *, default: str = "auto") -> str:
    normalized = str(value or "").strip().casefold()
    return normalized if normalized in ROUTING_PROTOCOLS else default


def build_main_brain_decision_prompt(
    raw_request: str,
    history: Iterable[dict[str, Any]],
    *,
    session_context: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Render compact reference context while keeping the current request immutable."""

    recent = []
    for message in list(history)[-8:]:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        if content.startswith(("[TOOL_RESULT:", "[TOOL_ROUTING_ERROR:")):
            continue
        recent.append({"role": role, "content": content[:1200]})
    retry_state = (
        "TRUSTED RUNTIME RETRY STATE:\n"
        "The current request explicitly retries the most recent unresolved external "
        "action reported in the context. Classify that action as delegate and preserve "
        "its domain; do not merely answer the retry question.\n\n"
        if is_explicit_retry_request(raw_request)
        and any(
            message["role"] == "assistant"
            and _UNRESOLVED_TOOL_FAILURE_RE.search(message["content"])
            for message in recent[-6:]
        )
        else ""
    )
    continuity = list(session_context or [])[-24:]
    continuity_block = (
        "SAME-SESSION CONTINUITY (reference resolution only):\n"
        f"{json.dumps(continuity, ensure_ascii=False)}\n\n"
        if continuity else ""
    )
    return (
        retry_state
        + continuity_block
        + "RECENT CONTEXT (reference resolution only):\n"
        f"{json.dumps(recent, ensure_ascii=False)}\n\n"
        "CURRENT USER REQUEST (the only action to classify):\n"
        f"{raw_request}"
    )


def is_explicit_retry_request(raw_request: str) -> bool:
    """Whether the current wording explicitly asks to repeat an earlier action."""
    return bool(_EXPLICIT_RETRY_RE.search(raw_request or ""))


def referenced_literal_sources(
    raw_request: str,
    history: Iterable[dict[str, Any]],
) -> list[dict[str, str]]:
    """Authorize exact recent sources for an explicit anaphoric request.

    The most recent user message carries user authority and may supply any exact
    literal field. The most recent Yuki reply is data-only and can supply payload
    text, but never paths, filenames, commands, or URLs. Merely having history is
    not enough: the current request must explicitly refer back to it.
    """
    if not (
        _EXPLICIT_SOURCE_REFERENCE_RE.search(raw_request or "")
        or is_explicit_retry_request(raw_request)
    ):
        return []
    previous_users = []
    previous_assistant = None
    skipped_current_user = False
    for message in reversed(list(history)[-8:]):
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        if not content.strip() or content.startswith("["):
            continue
        if (
            role == "user"
            and not skipped_current_user
            and content == raw_request
        ):
            skipped_current_user = True
            continue
        if role == "user" and len(previous_users) < 4:
            previous_users.append({
                "source_kind": "referenced_previous_user",
                "content": content,
            })
        elif role == "assistant" and previous_assistant is None:
            previous_assistant = {
                "source_kind": "referenced_previous_assistant",
                "content": content,
            }
        if len(previous_users) >= 4 and previous_assistant is not None:
            break
    if previous_assistant is not None:
        previous_users.append(previous_assistant)
    return previous_users


def parse_main_brain_decision(raw_text: str) -> dict[str, Any]:
    """Strictly parse a respond/delegate decision; malformed output is non-executable."""

    try:
        decoded = json.loads(raw_text.strip())
    except (TypeError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "decision": None,
            "errors": [f"Invalid delegation JSON: {exc}"],
        }
    errors: list[str] = []
    if not isinstance(decoded, dict):
        errors.append("Delegation decision must be an object.")
    elif set(decoded) != {"action", "request", "domain_hint"}:
        errors.append("Delegation decision has unexpected or missing fields.")
    else:
        action = decoded.get("action")
        request = decoded.get("request")
        domain = decoded.get("domain_hint")
        if action not in {"respond", "delegate"}:
            errors.append("Unknown delegation action.")
        if not isinstance(request, str):
            errors.append("Delegation request must be text.")
        if action == "respond" and (request or domain != "none"):
            errors.append("Respond decisions must use an empty request and domain none.")
        if action == "delegate":
            if not isinstance(request, str) or not request.strip():
                errors.append("Delegated actions require a semantic request.")
            if domain not in DELEGATION_DOMAINS:
                errors.append("Delegated action has an invalid domain.")
    return {
        "passed": not errors,
        "decision": decoded if not errors else None,
        "errors": errors,
    }


def parse_autonomous_decision(raw_text: str) -> dict[str, Any]:
    """Parse a WAIT/SPEAK/DELEGATE decision; malformed output cannot act."""

    try:
        decoded = json.loads(raw_text.strip())
    except (TypeError, json.JSONDecodeError) as exc:
        return {
            "passed": False,
            "decision": None,
            "errors": [f"Invalid autonomous decision JSON: {exc}"],
        }
    errors: list[str] = []
    if not isinstance(decoded, dict):
        errors.append("Autonomous decision must be an object.")
    elif set(decoded) != {"action", "request", "domain_hint"}:
        errors.append("Autonomous decision has unexpected or missing fields.")
    else:
        action = decoded.get("action")
        request = decoded.get("request")
        domain = decoded.get("domain_hint")
        if action not in {"wait", "speak", "delegate"}:
            errors.append("Unknown autonomous action.")
        if not isinstance(request, str):
            errors.append("Autonomous request must be text.")
        if action == "wait" and (request or domain != "none"):
            errors.append("Wait decisions must use an empty request and domain none.")
        if action == "speak":
            if not isinstance(request, str) or not request.strip():
                errors.append("Speak decisions require a meaningful topic.")
            if domain != "none":
                errors.append("Speak decisions must use domain none.")
        if action == "delegate":
            if not isinstance(request, str) or not request.strip():
                errors.append("Delegated autonomous actions require a complete request.")
            if domain not in DELEGATION_DOMAINS:
                errors.append("Delegated autonomous action has an invalid domain.")
    return {
        "passed": not errors,
        "decision": decoded if not errors else None,
        "errors": errors,
    }


class RoutingRegistry(ToolRegistry):
    """Production adapter over the real 18-tool Yuki registry.

    The adapter changes only model-visible descriptions/argument transport.
    Actual implementations and live ``META`` definitions remain untouched.
    """

    @staticmethod
    def _composite_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact filename inside Yuki's managed store.",
                },
                "content": {
                    "type": "string",
                    "description": "Exact content supplied by the user; may contain |.",
                },
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        }

    def argument_schema_for(self, name: str) -> dict[str, Any]:
        if name in {"yuki_write", "yuki_append"}:
            return self._composite_schema()
        spec = self.get(name)
        if spec is None:
            raise KeyError(name)
        return deepcopy(spec.argument_schema)

    def description_for(self, name: str) -> str:
        spec = self.get(name)
        if spec is None:
            raise KeyError(name)
        return _DISPATCH_DESCRIPTIONS.get(name, spec.description)

    def strict_openai_schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        result = []
        for name in names:
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": self.description_for(name),
                    "parameters": self.argument_schema_for(name),
                },
            })
        return result

    def xlam_schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        result = []
        for name in names:
            schema = self.argument_schema_for(name)
            result.append({
                "name": name,
                "description": self.description_for(name),
                "parameters": deepcopy(schema["properties"]),
                "required": list(schema["required"]),
                "additionalProperties": False,
            })
        return result

    def hammer_schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        result = []
        for name in names:
            schema = self.argument_schema_for(name)
            parameters = deepcopy(schema["properties"])
            for required in schema["required"]:
                parameters[required]["required"] = True
            result.append({
                "name": name,
                "description": self.description_for(name),
                "parameters": parameters,
            })
        return result

    def output_schema(
        self,
        names: Iterable[str],
        output_mode: str,
        native_dialect: str,
    ) -> dict[str, Any]:
        variants = []
        for name in names:
            key = "name" if output_mode == "native" else "tool"
            variants.append({
                "type": "object",
                "properties": {
                    key: {"const": name},
                    "arguments": self.argument_schema_for(name),
                },
                "required": [key, "arguments"],
                "additionalProperties": False,
            })
        if output_mode == "canonical":
            variants.append({
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
            })
            return {"oneOf": variants}
        if native_dialect in {
            TOOL_CALL_ARRAY,
            OPENAI_NATIVE_TOOL_CALLS,
            HAMMER_FENCED_ARRAY,
        }:
            return {
                "type": "array",
                "minItems": 0,
                "maxItems": 1,
                "items": {"oneOf": variants},
            }
        if native_dialect == XLAM_V1_ENVELOPE:
            return {
                "type": "object",
                "properties": {
                    "tool_calls": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 1,
                        "items": {"oneOf": variants},
                    },
                },
                "required": ["tool_calls"],
                "additionalProperties": False,
            }
        if native_dialect == ARCH_XML_OBJECT:
            return {"oneOf": variants}
        raise ValueError(f"Unknown native dialect: {native_dialect}")

    def validate_call(
        self,
        call: dict[str, Any],
        available_names: Iterable[str],
    ) -> dict[str, Any]:
        errors: list[str] = []
        if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
            return {
                "passed": False,
                "errors": ["Call must contain exactly tool and arguments."],
            }
        name = call.get("tool")
        arguments = call.get("arguments")
        if not isinstance(name, str) or self.get(name) is None:
            errors.append(f"Unknown tool: {name!r}")
        elif name not in set(available_names):
            errors.append(f"Tool {name!r} was not offered.")
        if not isinstance(arguments, dict):
            errors.append("Arguments must be an object.")
        elif isinstance(name, str) and self.get(name) is not None:
            errors.extend(self._validate_arguments(
                self.argument_schema_for(name), arguments,
            ))
        return {"passed": not errors, "errors": errors}

    def prepare_call(
        self,
        call: dict[str, Any],
        *,
        raw_request: str,
        available_names: Iterable[str],
        source_kind: str = "raw_user_request",
        literal_sources: Iterable[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Validate, bind exact fields, and adapt to the live one-string API."""

        available = tuple(available_names)
        validation = self.validate_call(call, available)
        if not validation["passed"]:
            return {
                "passed": False,
                "errors": validation["errors"],
                "binding": None,
                "canonical_call": None,
                "runtime_argument": None,
            }
        name = call["tool"]
        arguments = call["arguments"]
        fields = LITERAL_FIELDS.get(name, ())
        authorized_sources = [
            source for source in (literal_sources or [])
            if isinstance(source, dict)
            and isinstance(source.get("content"), str)
        ]
        field_provenance = {}
        for field in fields:
            value = arguments.get(field)
            if not isinstance(value, str):
                continue
            if value in raw_request:
                field_provenance[field] = source_kind
                continue
            for source in authorized_sources:
                referenced_user = (
                    source.get("source_kind") == "referenced_previous_user"
                )
                if not referenced_user and (name, field) not in CROSS_TURN_LITERAL_FIELDS:
                    continue
                if value in source["content"]:
                    field_provenance[field] = source.get(
                        "source_kind",
                        "referenced_context",
                    )
                    break
        missing_spans = [
            field
            for field in fields
            if field not in field_provenance
        ]
        if missing_spans:
            source_label = (
                "the autonomous action request"
                if source_kind == "autonomous_action"
                else "the raw user request or an explicitly authorized referenced source"
            )
            return {
                "passed": False,
                "errors": [
                    (
                        "Literal argument field(s) are not character-exact spans of "
                        f"{source_label}: {', '.join(missing_spans)}. No tool ran."
                    )
                ],
                "binding": {
                    "mode": "literal_source",
                    "source_copy_valid": False,
                    "source_kind": source_kind,
                    "fields": list(fields),
                },
                "canonical_call": None,
                "runtime_argument": None,
            }
        if name in {"yuki_write", "yuki_append"}:
            runtime_arguments = {
                "filename_and_content": (
                    f"{arguments['filename']}|{arguments['content']}"
                ),
            }
        else:
            runtime_arguments = deepcopy(arguments)
        live_spec = self.get(name)
        assert live_spec is not None
        live_call = {"tool": name, "arguments": runtime_arguments}
        live_validation = super().validate_call(live_call, available)
        if not live_validation["passed"]:
            return {
                "passed": False,
                "errors": live_validation["errors"],
                "binding": None,
                "canonical_call": None,
                "runtime_argument": None,
            }
        runtime_argument = ""
        if live_spec.param_name is not None:
            runtime_argument = runtime_arguments[live_spec.param_name]
        shell_errors = validate_strict_shell(runtime_argument) if name == "shell" else []
        if shell_errors:
            return {
                "passed": False,
                "errors": shell_errors + ["No shell command ran."],
                "binding": None,
                "canonical_call": None,
                "runtime_argument": None,
            }
        return {
            "passed": True,
            "errors": [],
            "binding": {
                "mode": "literal_source" if fields else "typed_passthrough",
                "source_copy_valid": True if fields else None,
                "source_kind": (
                    source_kind
                    if fields and set(field_provenance.values()) == {source_kind}
                    else "mixed_authorized_sources" if fields else None
                ),
                "fields": list(fields),
                "field_provenance": field_provenance if fields else None,
            },
            "canonical_call": live_call,
            "model_call": deepcopy(call),
            "runtime_argument": runtime_argument,
        }


_STRICT_SHELL_CHARS = re.compile(r"^[A-Za-z0-9_./:=,@%+\-\s]+$")


def validate_strict_shell(command: str) -> list[str]:
    """Prototype-proven shell boundary; reject operators before live execution."""

    errors: list[str] = []
    if any(ord(char) < 32 for char in command):
        errors.append("Control characters and newlines are forbidden in shell calls.")
    if not _STRICT_SHELL_CHARS.fullmatch(command):
        errors.append(
            "Shell operators, substitutions, quotes, escapes, and globs are forbidden."
        )
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        errors.append(f"Invalid shell tokenization: {exc}")
        parts = []
    if not parts:
        errors.append("Shell command must not be empty.")
    elif parts[0] not in ALLOWED_COMMANDS:
        errors.append(
            f"Command {parts[0]!r} is not allowed; expected one of "
            f"{sorted(ALLOWED_COMMANDS)}."
        )
    return errors


def parse_canonical_call(raw_text: str) -> dict[str, Any]:
    """Parse strict canonical JSON through the tested prototype normalizer."""

    from prototypes.tool_dispatcher.parsing import parse_model_output

    return parse_model_output(raw_text, "canonical", CANONICAL_OBJECT)


def normalize_openai_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    """Convert one provider-native call into Yuki's canonical representation."""

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    return {"tool": name, "arguments": arguments}
