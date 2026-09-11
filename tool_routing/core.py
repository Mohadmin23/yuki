"""Strict canonical-call boundary shared by direct and delegated routing.

This module deliberately has no model-loading or tool-execution capability.
It converts Yuki's live metadata into dispatcher-facing schemas, validates
named arguments, and performs a conservative typed bind against the immutable
raw user request.
"""

from __future__ import annotations

import json
import re
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
from prototypes.tool_dispatcher.source_span_policy import bind_source_argument
from tools._helpers import validate_strict_shell

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
    "image": ("prompt",),
    "read": ("filepath",),
    "shell": ("command",),
    "yuki_write": ("filename", "content"),
    "yuki_read": ("filename",),
    "yuki_delete": ("filename",),
    "yuki_append": ("filename", "content"),
    "ask_claude": ("question",),
}

# Search is deliberately different from paths, commands, filenames, content,
# and external-agent messages.  A useful web query is normally a semantic,
# self-contained rewrite of the user's intent.  It becomes literal transport
# only when the user explicitly asks to preserve query syntax/text.
SEARCH_QUERY_MAX_CHARS = 1000

_SEARCH_EXACT_CONTRACT_RE = re.compile(
    r"(?:"
    r"\bsearch\s+exactly(?:\s+for)?\b|"
    r"\b(?:exact|verbatim)\s+(?:search\s+)?query\b|"
    r"\b(?:use|run|submit|copy)\s+(?:this\s+|the\s+)?exact\s+query\b|"
    r"\b(?:keep|preserve)\s+(?:this\s+|the\s+)?"
    r"(?:query|search\s+text|wording)\s+"
    r"(?:exactly|verbatim|unchanged|intact|as\s+written)\b|"
    r"\b(?:keep|preserve)\s+(?:this\s+|the\s+)?"
    r"(?:punctuation|capitalization|casing|operators?|quotes?)\b|"
    r"\b(?:do\s+not|don['’]t)\s+"
    r"(?:change|rewrite|normalize|paraphrase|correct)\b[^.!?\n]{0,64}"
    r"\b(?:query|search\s+text|wording|punctuation|capitalization|casing|"
    r"operators?|quotes?)\b|"
    r"\bcharacter[- ]for[- ]character\b|"
    r"\bpunctuation\s+and\s+all\b"
    r")",
    re.IGNORECASE,
)
_SEARCH_QUOTED_QUERY_RE = re.compile(
    r"\b(?:search(?:\s+(?:the\s+)?(?:web|internet|online))?(?:\s+for)?|"
    r"look\s+up|web\s+query)\b[^\n]{0,120}"
    r'(?:"[^"\n]+"|“[^”\n]+”|`[^`\n]+`)',
    re.IGNORECASE,
)
_SEARCH_FIELD_OPERATOR_RE = re.compile(
    r"(?:\b(?:site|filetype|intitle|inurl|before|after):\S+|"
    r"(?:^|\s)-[^\s-]+)",
    re.IGNORECASE,
)
_SEARCH_BOOLEAN_OPERATOR_RE = re.compile(r"\b(?:AND|OR)\b")
_SEARCH_PAYLOAD_ENVELOPE_RE = re.compile(
    r"^\s*(?:(?:hey|okay|ok)[,!]?\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+)?"
    r"(?:(?:please|also)\s+){0,2}"
    r"(?:search(?:\s+(?:the\s+)?(?:web|internet|online))?(?:\s+for)?|"
    r"look\s+up|web\s+search\s*:|web\s+query\s*:)",
    re.IGNORECASE,
)
_SEARCH_REFERENCE_REQUEST_RE = re.compile(
    r"(?:"
    r"\b(?:search|look\s+up|look\s+for|find|research)\b[^.!?\n]{0,80}"
    r"\b(?:it|that|this|them|those|these|same|previous|last)\b|"
    r"\b(?:more|updates?|news|information|details)\s+"
    r"(?:on|about)\s+(?:it|that|this)\b"
    r")",
    re.IGNORECASE,
)
_UNRESOLVED_SEARCH_QUERIES = {
    "it",
    "that",
    "this",
    "them",
    "those",
    "these",
    "it again",
    "that again",
    "this again",
    "the same",
    "same thing",
    "the same thing",
    "same query",
    "the same query",
    "previous query",
    "the previous query",
    "last query",
    "the last query",
    "that topic",
    "this topic",
    "the topic",
}
_UNRESOLVED_SEARCH_QUERY_RE = re.compile(
    r"^\s*(?:(?:more|recent|latest|new)\s+)*"
    r"(?:(?:news|updates?|details|information|results?)\s+)?"
    r"(?:(?:about|on|for|regarding)\s+)?"
    r"(?:it|that|this|them|those|these)"
    r"(?:\s+again)?\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_EXPLICIT_WEB_SEARCH_RE = re.compile(
    r"\b(?:web\s+search|search\s+(?:the\s+)?(?:web|internet|online)|"
    r"look\s+up\s+online|find\s+(?:online|on\s+the\s+web)|"
    r"internet\s+results?|web\s+results?)\b",
    re.IGNORECASE,
)
_EXISTING_VISUAL_REQUEST_RE = re.compile(
    r"(?:"
    r"\blook\s+at\b[^.!?\n]{0,90}"
    r"\b(?:camera|webcam|screen|display|screenshot|image|photo|picture|scene|"
    r"room)\b|"
    r"\b(?:inspect|check|view|observe|describe|see)\b[^.!?\n]{0,70}"
    r"\b(?:this|that|the|my|your|attached|uploaded|current)\s+"
    r"(?:camera(?:\s+feed)?|webcam(?:\s+feed)?|screen|display|screenshot|"
    r"image|photo|picture|scene|room)\b|"
    r"\b(?:look\s+for|find|spot)\b[^.!?\n]{0,90}"
    r"\b(?:with|using|through|in|on)\s+(?:the\s+)?"
    r"(?:camera|webcam|screen|display|screenshot|existing\s+image|image|"
    r"photo|picture|room)\b|"
    r"\b(?:camera|webcam|screen|display)\s+feed\b[^.!?\n]{0,90}"
    r"\b(?:see|show|find|spot|look|visible)\b"
    r"|\bwhat\s+(?:do|can)\s+you\s+see\b"
    r")",
    re.IGNORECASE,
)


def search_query_requires_literal_source(raw_request: str) -> bool:
    """Return true only when web-query text/syntax is explicitly exact-sensitive."""

    text = raw_request or ""
    return bool(
        _SEARCH_EXACT_CONTRACT_RE.search(text)
        or _SEARCH_QUOTED_QUERY_RE.search(text)
        or _SEARCH_FIELD_OPERATOR_RE.search(text)
        or (
            _SEARCH_PAYLOAD_ENVELOPE_RE.search(text)
            and _SEARCH_BOOLEAN_OPERATOR_RE.search(text)
        )
    )


def _is_unresolved_search_query(query: str) -> bool:
    normalized = re.sub(r"\s+", " ", query.strip(" \t\r\n.,!?;:'\"`“”‘’"))
    return bool(
        normalized.casefold() in _UNRESOLVED_SEARCH_QUERIES
        or _UNRESOLVED_SEARCH_QUERY_RE.fullmatch(query)
    )


def _contextualize_search_query(query: str, referent: str) -> str:
    """Replace one unresolved deictic while retaining useful search modifiers."""

    stripped = query.strip()
    if stripped.strip(".,!?;:'\"`“”‘’").casefold() in _UNRESOLVED_SEARCH_QUERIES:
        return referent
    return re.sub(
        r"\b(?:it|that|this|them|those|these)\b",
        referent,
        stripped,
        count=1,
        flags=re.IGNORECASE,
    )


def _nearest_referenced_user_search_query(
    sources: Iterable[dict[str, str]],
    selected_call: dict[str, Any],
) -> str | None:
    """Recover only the nearest user-authored search; never scan past it."""

    for source in sources:
        if source.get("source_kind") != "referenced_previous_user":
            continue
        candidate = bind_source_argument(
            raw_request=source["content"],
            selected_call=selected_call,
            semantic_request="",
            verbatim=[],
        )
        rebound = candidate.get("final_call")
        rule = candidate.get("source_span", {}).get("rule")
        query = (
            rebound.get("arguments", {}).get("query")
            if isinstance(rebound, dict)
            else None
        )
        if (
            candidate.get("status") == "bound"
            and candidate.get("source_copy_valid") is True
            and rule in {"search_prefix", "search_label", "search_exact_label"}
            and isinstance(query, str)
            and query.strip()
            and not _is_unresolved_search_query(query)
            and len(query) <= SEARCH_QUERY_MAX_CHARS
            and not any(ord(char) < 32 for char in query)
        ):
            return query
        # The first previous-user source is the nearest semantic context.
        # If it is not a recognizable search request, an older one is unsafe.
        return None
    return None


def _latest_verified_search_query(
    trusted_context: Iterable[dict[str, Any]] | None,
) -> str | None:
    """Read only canonical query arguments from successful runtime-ledger events."""

    for event in reversed(list(trusted_context or [])):
        if (
            not isinstance(event, dict)
            or event.get("kind") != "verified_tool_outcome"
            or event.get("tool") != "search"
            or event.get("succeeded") is not True
        ):
            continue
        arguments = event.get("arguments")
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if (
            isinstance(query, str)
            and query.strip()
            and len(query) <= SEARCH_QUERY_MAX_CHARS
            and not any(ord(char) < 32 for char in query)
        ):
            return query
    return None


def _search_selection_conflicts_with_visual_intent(raw_request: str) -> bool:
    """Keep an erroneous search selection from consuming a vision request."""

    return bool(
        _EXISTING_VISUAL_REQUEST_RE.search(raw_request or "")
        and not _EXPLICIT_WEB_SEARCH_RE.search(raw_request or "")
    )

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
                    "description": "File contents. An empty placeholder is allowed; the main model prepares contents before execution.",
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
        schema = deepcopy(spec.argument_schema)
        if name == "search":
            schema["properties"]["query"].update({
                "description": (
                    "A self-contained web-search query grounded in the current request "
                    "or its explicitly referenced recent context. Resolve pronouns before "
                    "calling. Ordinary queries may be concise semantic rewrites. Preserve "
                    "explicitly exact, quoted, or operator-sensitive query text "
                    "character-for-character."
                ),
                "maxLength": SEARCH_QUERY_MAX_CHARS,
            })
        return schema

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
            if name == "search" and isinstance(arguments.get("query"), str):
                query = arguments["query"]
                if len(query) > SEARCH_QUERY_MAX_CHARS:
                    errors.append(
                        f"Argument 'query' must be at most {SEARCH_QUERY_MAX_CHARS} characters."
                    )
                if any(ord(char) < 32 for char in query):
                    errors.append("Argument 'query' must not contain control characters.")
        return {"passed": not errors, "errors": errors}

    def prepare_call(
        self,
        call: dict[str, Any],
        *,
        raw_request: str,
        available_names: Iterable[str],
        source_kind: str = "raw_user_request",
        literal_sources: Iterable[dict[str, str]] | None = None,
        trusted_context: Iterable[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Validate, apply typed transport, and adapt to the live one-string API."""

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
        proposed_call = deepcopy(call)
        name = call["tool"]
        arguments = call["arguments"]
        if name == "search" and _search_selection_conflicts_with_visual_intent(
            raw_request
        ):
            return {
                "passed": False,
                "errors": [
                    (
                        "The selected search tool cannot inspect an existing image, camera, "
                        "screen, or scene. Select the visual-inspection capability instead. "
                        "No tool ran."
                    )
                ],
                "binding": None,
                "canonical_call": None,
                "runtime_argument": None,
            }

        exact_search = (
            name == "search" and search_query_requires_literal_source(raw_request)
        )
        authorized_sources = [
            source for source in (literal_sources or [])
            if isinstance(source, dict)
            and isinstance(source.get("content"), str)
        ]
        contextual_search_query = None
        contextual_search_binder = None
        contextual_search_source_kind = None
        if name == "search" and _is_unresolved_search_query(arguments["query"]):
            referent = None
            if _SEARCH_REFERENCE_REQUEST_RE.search(raw_request or ""):
                referent = _nearest_referenced_user_search_query(
                    authorized_sources,
                    call,
                )
                if referent is not None:
                    contextual_search_binder = "referenced_previous_user_search_v1"
                    contextual_search_source_kind = "referenced_previous_user"
                elif not any(
                    source.get("source_kind") == "referenced_previous_user"
                    for source in authorized_sources
                ):
                    referent = _latest_verified_search_query(trusted_context)
                    if referent is not None:
                        contextual_search_binder = "trusted_recent_search_v1"
                        contextual_search_source_kind = "verified_tool_outcome"
            if referent is None:
                return {
                    "passed": False,
                    "errors": [
                        (
                            "The web-search query contains an unresolved reference and no "
                            "single trusted recent search query can resolve it. No tool ran."
                        )
                    ],
                    "binding": {
                        "mode": "contextual_semantic_resolution",
                        "source_copy_valid": None,
                        "source_kind": None,
                        "fields": ["query"],
                    },
                    "canonical_call": None,
                    "runtime_argument": None,
                }
            contextual_search_query = _contextualize_search_query(
                arguments["query"],
                referent,
            )
            call = {
                "tool": "search",
                "arguments": {"query": contextual_search_query},
            }
            arguments = call["arguments"]

        fields = (
            ("query",)
            if exact_search
            else LITERAL_FIELDS.get(name, ())
        )
        field_provenance = (
            {"query": contextual_search_source_kind}
            if contextual_search_query is not None and exact_search
            else {}
        )
        deterministic_binding = None
        deterministic_binder = None
        # Exact-sensitive web queries keep the proven frozen source binder.
        # Ordinary searches never enter this path: the model may generate a
        # concise, self-contained semantic query from the visible conversation.
        if name == "search" and exact_search and contextual_search_query is None:
            candidate = bind_source_argument(
                raw_request=raw_request,
                selected_call=call,
                semantic_request="",
                verbatim=[],
            )
            rebound_call = candidate.get("final_call")
            rebound_validation = (
                self.validate_call(rebound_call, available)
                if isinstance(rebound_call, dict)
                else {"passed": False}
            )
            if (
                candidate.get("status") == "bound"
                and candidate.get("source_copy_valid") is True
                and rebound_validation["passed"]
            ):
                call = rebound_call
                arguments = call["arguments"]
                deterministic_binding = candidate
                deterministic_binder = "frozen_source_span_v1_search"

        if name in {"yuki_write", "yuki_append"}:
            contracts = [source for source in authorized_sources
                         if source.get("source_kind") == "file_content_intent"]
            for contract in contracts:
                if (contract.get("request") != raw_request
                        or contract.get("mode") not in {"compose", "literal"}
                        or contract.get("tool") != name
                        or contract.get("filename") != arguments.get("filename")
                        or contract.get("content") != arguments.get("content")):
                    return {"passed": False, "errors": [
                        contract.get("error") or "The proposed write conflicts with the current file-content intent.",
                    ], "binding": None, "canonical_call": None, "runtime_argument": None}

        for field in fields:
            value = arguments.get(field)
            if not isinstance(value, str):
                continue
            if value in raw_request:
                field_provenance[field] = source_kind
                continue
            for source in authorized_sources:
                if source.get("source_kind") == "file_content_intent":
                    if (
                        name in {"yuki_write", "yuki_append"} and field == "content"
                        and source.get("mode") == "compose"
                        and source.get("request") == raw_request
                        and source.get("tool") == name
                        and source.get("filename") == arguments.get("filename")
                        and value == source["content"]
                    ):
                        field_provenance[field] = "model_composed_content"
                        break
                    continue
                if source.get("source_kind") not in {"referenced_previous_user", "referenced_previous_assistant"}:
                    continue
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
                "mode": (
                    "deterministic_source_span"
                    if deterministic_binding is not None
                    else "contextual_semantic_resolution"
                    if contextual_search_binder is not None
                    else "literal_source"
                    if fields
                    else "semantic_argument"
                    if name == "search"
                    else "typed_passthrough"
                ),
                "source_copy_valid": (
                    True if fields and contextual_search_binder is None else None
                ),
                "source_kind": (
                    contextual_search_source_kind
                    if contextual_search_binder is not None
                    else source_kind
                    if fields and set(field_provenance.values()) == {source_kind}
                    else "mixed_authorized_sources" if fields else None
                ),
                "fields": ["query"] if name == "search" else list(fields),
                "field_provenance": (
                    {"query": contextual_search_source_kind}
                    if contextual_search_binder is not None
                    else field_provenance if fields
                    else {"query": "model_semantic"} if name == "search"
                    else None
                ),
                "binder": (
                    (contextual_search_binder or deterministic_binder)
                    if deterministic_binding is not None
                    or contextual_search_binder is not None
                    else None
                ),
                "source_span": (
                    deterministic_binding.get("source_span")
                    if deterministic_binding is not None
                    else None
                ),
            },
            "canonical_call": live_call,
            "model_call": deepcopy(call),
            "proposed_call": proposed_call,
            "runtime_argument": runtime_argument,
        }


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
