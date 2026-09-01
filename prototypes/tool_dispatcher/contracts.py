"""Small shared data structures for the dispatcher prototype."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """One live Yuki tool plus the strict prototype schema derived from it."""

    name: str
    cli_name: str
    description: str
    param_name: str | None
    fn: Callable[[str], Any]
    source_schema: dict[str, Any]
    argument_schema: dict[str, Any]


@dataclass
class GenerationResult:
    """Raw backend generation and observable inference measurements."""

    raw_text: str
    rendered_prompt: str
    backend: str
    model_id: str
    latency_ms: float
    normalized_text: str | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    prompt_tokens_per_second: float | None = None
    tokens_per_second: float | None = None
    peak_memory_gb: float | None = None
    process_rss_mb: float | None = None
    finish_reason: str | None = None
    schema_constrained: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "rendered_prompt": self.rendered_prompt,
            "backend": self.backend,
            "model_id": self.model_id,
            "latency_ms": round(self.latency_ms, 3),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_gb": self.peak_memory_gb,
            "process_rss_mb": self.process_rss_mb,
            "finish_reason": self.finish_reason,
            "schema_constrained": self.schema_constrained,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class PromptPackage:
    """The exact one-shot dispatcher input and its expected output schema."""

    content: str
    schemas_sent: list[dict[str, Any]]
    output_schema: dict[str, Any]
    output_mode: str
    native_dialect: str
    json_root: str
    system_content: str | None = None
    chat_template_tools: list[dict[str, Any]] | None = None
    pre_rendered: bool = False
