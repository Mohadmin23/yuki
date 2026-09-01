"""One-shot tool dispatch, validation, and optional execution."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .backends import MAX_GENERATION_TOKENS, DispatcherBackend
from .executor import execute_call
from .parsing import parse_model_output
from .prompting import build_prompt
from .registry import ToolRegistry


class Dispatcher:
    """Stateless orchestration around a reusable loaded model backend."""

    def __init__(self, registry: ToolRegistry, backend: DispatcherBackend) -> None:
        self.registry = registry
        self.backend = backend

    def route_and_validate(
        self,
        request: str,
        *,
        selected: Iterable[str],
        output_mode: str = "native",
    ) -> dict[str, Any]:
        """Generate, parse, and validate one call without an execution capability.

        Automatic schema-selection benchmarks use this deliberately narrow entry
        point. It exposes no execution or side-effect controls and always stops at
        validation.
        """

        return self.dispatch(
            request,
            selected=selected,
            output_mode=output_mode,
            execute=False,
        )

    def dispatch(
        self,
        request: str,
        *,
        group: str = "all",
        selected: Iterable[str] | None = None,
        output_mode: str = "native",
        execute: bool = False,
        allow_side_effects: bool = False,
        shell_mode: str = "dry_run",
    ) -> dict[str, Any]:
        names = self.registry.resolve_names(group, selected)
        package = build_prompt(
            self.registry,
            request,
            names,
            output_mode,
            model_id=self.backend.model_id,
        )
        base = {
            "request": request,
            "router_mode": "model_only",
            "stateless": True,
            "generation_count": 1,
            "tool_scope": {"group": group, "names": list(names)},
            "output_mode": output_mode,
            "native_dialect": package.native_dialect,
            "schemas_sent": package.schemas_sent,
            "prompt_content": package.content,
            "system_content": package.system_content,
            "output_schema": package.output_schema,
        }
        try:
            generation = self.backend.generate(
                package.content,
                package.output_schema,
                max_tokens=MAX_GENERATION_TOKENS,
                system_prompt=package.system_content,
                chat_template_tools=package.chat_template_tools,
                json_root=package.json_root,
                pre_rendered=package.pre_rendered,
            )
        except Exception as exc:  # noqa: BLE001 - backend failures are report data
            return {
                **base,
                "generation": {
                    "backend": self.backend.backend_name,
                    "model_id": self.backend.model_id,
                    "raw_text": "",
                    "rendered_prompt": package.content,
                    "error": f"{type(exc).__name__}: {exc}",
                    "generated_tokens": None,
                    "tokens_per_second": None,
                    "latency_ms": None,
                },
                "parse": {
                    "passed": False,
                    "malformed": False,
                    "rejected": False,
                    "decoded": None,
                    "canonical_call": None,
                    "errors": ["Model generation failed before parsing."],
                },
                "selected_tool": None,
                "parsed_arguments": None,
                "canonical_call": None,
                "validation": {
                    "passed": False,
                    "errors": ["Generation failed."],
                },
                "execution": self._not_executed("generation_failed"),
            }

        parsed = parse_model_output(
            generation.normalized_text or generation.raw_text,
            output_mode,
            package.native_dialect,
        )
        call = parsed["canonical_call"]
        if call is None:
            errors = parsed["errors"] or (
                ["Model rejected the delegated request."] if parsed["rejected"] else []
            )
            validation = {"passed": False, "errors": errors}
        else:
            validation = self.registry.validate_call(call, names)

        if execute and validation["passed"]:
            execution = execute_call(
                self.registry,
                call,
                names,
                allow_side_effects=allow_side_effects,
                shell_mode=shell_mode,
            )
        else:
            reason = "not_requested" if not execute else "validation_failed"
            execution = self._not_executed(reason)

        return {
            **base,
            "generation": generation.as_dict(),
            "parse": parsed,
            "selected_tool": call.get("tool") if call else None,
            "parsed_arguments": call.get("arguments") if call else None,
            "canonical_call": call,
            "validation": validation,
            "execution": execution,
        }

    def dispatch_regex(
        self,
        request: str,
        *,
        group: str = "all",
        selected: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        names = self.registry.resolve_names(group, selected)
        call = self.registry.regex_call(request)
        if call is None:
            validation = {"passed": False, "errors": ["Regex router found no call."]}
        else:
            validation = self.registry.validate_call(call, names)
        return {
            "request": request,
            "router_mode": "regex_only",
            "stateless": True,
            "generation_count": 0,
            "tool_scope": {"group": group, "names": list(names)},
            "output_mode": "canonical",
            "native_dialect": "canonical_object",
            "schemas_sent": [],
            "prompt_content": None,
            "system_content": None,
            "output_schema": None,
            "generation": None,
            "parse": {
                "passed": call is not None,
                "malformed": False,
                "rejected": call is None,
                "decoded": call,
                "canonical_call": call,
                "errors": [] if call else ["Regex router found no call."],
            },
            "selected_tool": call.get("tool") if call else None,
            "parsed_arguments": call.get("arguments") if call else None,
            "canonical_call": call,
            "validation": validation,
            "execution": self._not_executed("not_requested"),
        }

    def execute_validated(
        self,
        call: dict[str, Any],
        *,
        group: str = "all",
        selected: Iterable[str] | None = None,
        allow_side_effects: bool = False,
        shell_mode: str = "dry_run",
    ) -> dict[str, Any]:
        names = self.registry.resolve_names(group, selected)
        return execute_call(
            self.registry,
            call,
            names,
            allow_side_effects=allow_side_effects,
            shell_mode=shell_mode,
        )

    @staticmethod
    def _not_executed(reason: str) -> dict[str, Any]:
        return {
            "status": reason,
            "passed": False,
            "executed": False,
            "errors": [],
            "raw_output": None,
            "raw_output_repr": None,
            "latency_ms": None,
        }
