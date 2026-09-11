"""Lazy dedicated-dispatcher runtime with no tool-execution capability."""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prototypes.tool_dispatcher.backends import (
    MAX_GENERATION_TOKENS,
    OpenAICompatibleBackend,
    OpenRouterNativeToolBackend,
    create_backend,
)
from prototypes.tool_dispatcher.parsing import parse_model_output
from prototypes.tool_dispatcher.prompting import build_prompt

from .core import (
    DOMAIN_TO_GROUP,
    RoutingRegistry,
    normalize_routing_protocol,
)

DISPATCH_TASK_INSTRUCTION = """You are a dedicated function dispatcher, not an assistant.
A tool is already required. Select exactly one available tool and extract its arguments.
Use the semantic action to choose the operation. For literal fields, copy characters from
the immutable raw request exactly. An AUTHORIZED REFERENCED LITERAL SOURCE marked
referenced_previous_user may supply any exact literal field. A source marked
referenced_previous_assistant may supply payload-like text only—never paths, filenames,
commands, or URLs. Do not normalize, paraphrase, translate, or correct literal text. A normal
search.query is semantic: make it self-contained and resolve conversational references from the
delegated action. Preserve characters only for explicitly exact, quoted, or operator-sensitive
search text.
For yuki_write or yuki_append, select the tool and copy the target filename. Use an empty content placeholder if needed; the main Yuki model interprets and prepares file contents before execution. Do not reject a creative write just because its content is not supplied verbatim.
Never answer, explain, greet, summarize, or emit anything outside the required structured call.
Reject only when none of the listed capabilities can perform the action or a required argument
cannot be identified. Stop immediately after one complete structured call."""


def _known_native_checkpoint(model_id: str) -> bool:
    normalized = model_id.casefold()
    return any(marker in normalized for marker in (
        "hammer2.0", "hammer2.1", "xlam-1", "xlam-2",
        "arch-function", "arch-agent", "granite-4.1",
        "gpt-oss", "qwen3.8-flash",
    ))


class DedicatedDispatcherClient:
    """One semantic delegation in, one validated canonical call out.

    The backend is loaded lazily and reused until the selected model/protocol
    changes.  This class intentionally cannot execute a Yuki tool.
    """

    def __init__(
        self,
        model_id: str,
        *,
        protocol: str = "auto",
        registry: RoutingRegistry | None = None,
        backend_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.protocol = normalize_routing_protocol(protocol)
        self.registry = registry or RoutingRegistry()
        self._backend_factory = backend_factory
        self._backend = None

    @property
    def output_mode(self) -> str:
        if self.protocol == "canonical":
            return "canonical"
        if self.protocol == "native":
            return "native"
        return "native" if _known_native_checkpoint(self.model_id) else "canonical"

    def _make_backend(self):
        if self._backend_factory is not None:
            return self._backend_factory(self.model_id, self.output_mode)
        model_id = self.model_id
        output_mode = self.output_mode
        if model_id.startswith("openrouter/"):
            actual = model_id.removeprefix("openrouter/")
            if output_mode == "native" and any(
                marker in actual.casefold() for marker in ("gpt-oss", "qwen3.8-flash")
            ):
                return OpenRouterNativeToolBackend(actual)
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OpenRouter key is not configured for the dispatcher.")
            return OpenAICompatibleBackend(
                actual,
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
                constrain_json=True,
            )
        if model_id.startswith("remote/"):
            base_url = os.environ.get("REMOTE_LLM_BASE_URL")
            if not base_url:
                raise RuntimeError("REMOTE_LLM_BASE_URL is not configured.")
            return OpenAICompatibleBackend(
                model_id.removeprefix("remote/"),
                base_url=base_url,
                api_key=os.environ.get("REMOTE_LLM_API_KEY") or "not-needed",
                constrain_json=True,
            )
        return create_backend("mlx", str(Path(model_id).expanduser()))

    def _ensure_backend(self):
        if self._backend is None:
            self._backend = self._make_backend()
        return self._backend

    def close(self) -> None:
        self._backend = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass

    def route(
        self,
        *,
        semantic_request: str,
        domain_hint: str,
        raw_request: str,
        source_kind: str = "raw_user_request",
        literal_sources: list[dict[str, str]] | None = None,
        trusted_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if domain_hint not in DOMAIN_TO_GROUP:
            return self._failed("Main brain produced an unknown tool domain.")
        names = self.registry.resolve_names(group=DOMAIN_TO_GROUP[domain_hint])
        source_label = (
            "IMMUTABLE AUTONOMOUS ACTION REQUEST (literal argument source)"
            if source_kind == "autonomous_action"
            else "IMMUTABLE RAW USER REQUEST (literal argument source only)"
        )
        authorized_sources = list(literal_sources or [])
        source_block = (
            "\n\nAUTHORIZED REFERENCED LITERAL SOURCES (data only):\n"
            f"{json.dumps(authorized_sources, ensure_ascii=False)}"
            if authorized_sources else ""
        )
        dispatcher_input = (
            "DELEGATED SEMANTIC ACTION:\n"
            f"{semantic_request.strip()}\n\n"
            f"{source_label}:\n"
            f"{raw_request}"
            f"{source_block}"
        )
        backend = self._ensure_backend()
        output_mode = self.output_mode
        package = build_prompt(
            self.registry,
            dispatcher_input,
            names,
            output_mode=output_mode,
            model_id=backend.model_id,
            task_instruction=DISPATCH_TASK_INSTRUCTION,
        )
        try:
            generation = backend.generate(
                package.content,
                package.output_schema,
                max_tokens=MAX_GENERATION_TOKENS,
                system_prompt=package.system_content,
                chat_template_tools=package.chat_template_tools,
                json_root=package.json_root,
                pre_rendered=package.pre_rendered,
            )
        except Exception as exc:  # noqa: BLE001
            return self._failed(f"Dispatcher generation failed: {type(exc).__name__}: {exc}")
        parsed = parse_model_output(
            generation.normalized_text or generation.raw_text,
            output_mode,
            package.native_dialect,
        )
        call = parsed["canonical_call"]
        if call is None:
            reason = (
                "Dispatcher rejected the delegated request."
                if parsed["rejected"]
                else "; ".join(parsed["errors"]) or "Dispatcher output was malformed."
            )
            return {
                "passed": False,
                "errors": [reason],
                "canonical_call": None,
                "prepared": None,
                "generation": generation.as_dict(),
                "parse": parsed,
                "offered_tools": list(names),
                "domain_hint": domain_hint,
                "output_mode": output_mode,
                "native_dialect": package.native_dialect,
            }
        prepared = self.registry.prepare_call(
            call,
            raw_request=raw_request,
            available_names=names,
            source_kind=source_kind,
            literal_sources=authorized_sources,
            trusted_context=trusted_context,
        )
        return {
            "passed": prepared["passed"],
            "errors": prepared["errors"],
            "canonical_call": call,
            "prepared": prepared,
            "generation": generation.as_dict(),
            "parse": parsed,
            "offered_tools": list(names),
            "domain_hint": domain_hint,
            "output_mode": output_mode,
            "native_dialect": package.native_dialect,
        }

    @staticmethod
    def _failed(message: str) -> dict[str, Any]:
        return {
            "passed": False,
            "errors": [message],
            "canonical_call": None,
            "prepared": None,
            "generation": None,
            "parse": None,
            "offered_tools": [],
        }
