"""Swappable, stateless generation backends for one dispatcher call at a time."""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

from .contracts import GenerationResult
from .json_boundary import is_complete_json_value

MAX_GENERATION_TOKENS = 256
MLX_PREFILL_STEP_SIZE = 2048
DEFAULT_XLAM_MODEL = "Salesforce/xLAM-1b-fc-r"


def _process_rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1024**2, 3)
    except (ImportError, OSError):
        return None


def normalize_xlam_decoder_text(model_id: str, text: str) -> str:
    """Map xLAM/DeepSeek byte-level space/newline markers back to whitespace.

    The converted tokenizer can expose these byte-level newline markers as
    literal glyphs under MLX. The untouched decoder output remains visible in
    raw_text; only the parser input receives this model-specific correction.
    """

    if "xlam" not in model_id.lower():
        return text
    return text.replace("Ġ", " ").replace("Ċ", "\n")


def normalize_hammer_decoder_text(model_id: str, text: str) -> str:
    """Remove only Hammer's exact native Markdown wrapper from parser input."""

    if not any(
        version in model_id.lower() for version in ("hammer2.0", "hammer2.1")
    ):
        return text
    stripped = text.lstrip()
    prefix = next(
        (
            candidate
            for candidate in ("```json\r\n", "```json\n", "```\r\n", "```\n")
            if stripped.startswith(candidate)
        ),
        None,
    )
    if prefix is None:
        return text
    payload = stripped[len(prefix) :]
    trimmed = payload.rstrip()
    if trimmed.endswith("```"):
        trimmed = trimmed[:-3].rstrip()
    return trimmed


def normalize_arch_decoder_text(model_id: str, text: str) -> str:
    """Remove only a known model's native XML wrapper from parser input."""

    normalized_model_id = model_id.lower()
    if not any(
        marker in normalized_model_id
        for marker in ("arch-function", "arch-agent", "granite-4.1")
    ):
        return text
    stripped = text.lstrip()
    if not stripped.startswith("<tool_call>"):
        return text
    payload = stripped[len("<tool_call>") :].lstrip("\r\n")
    trimmed = payload.rstrip()
    if trimmed.endswith("</tool_call>"):
        trimmed = trimmed[: -len("</tool_call>")].rstrip()
    return trimmed


def normalize_model_decoder_text(model_id: str, text: str) -> str:
    normalized = normalize_xlam_decoder_text(model_id, text)
    normalized = normalize_hammer_decoder_text(model_id, normalized)
    return normalize_arch_decoder_text(model_id, normalized)


class DispatcherBackend(ABC):
    """One input prompt in, one raw generation out. No history is retained."""

    backend_name = "base"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.load_latency_ms: float | None = None
        self.model_metadata: dict[str, Any] = {}
        self.completed_generations = 0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        max_tokens: int = MAX_GENERATION_TOKENS,
        *,
        system_prompt: str | None = None,
        chat_template_tools: list[dict[str, Any]] | None = None,
        json_root: str = "object",
        pre_rendered: bool = False,
    ) -> GenerationResult:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "model_id": self.model_id,
            "load_latency_ms": self.load_latency_ms,
            "process_rss_mb": _process_rss_mb(),
            "model_metadata": self.model_metadata,
            "max_generation_tokens": MAX_GENERATION_TOKENS,
            "completed_generations": self.completed_generations,
            "stateless": True,
        }

    def _generation_metadata(
        self, max_tokens: int, **extra: Any
    ) -> dict[str, Any]:
        self.completed_generations += 1
        return {
            "generation_ceiling": max_tokens,
            "generation_index_after_load": self.completed_generations,
            "run_kind": (
                "first_after_load" if self.completed_generations == 1 else "warm"
            ),
            **extra,
        }


class MLXBackend(DispatcherBackend):
    """In-process Apple-Silicon backend; intended for the FP16 xLAM baseline."""

    backend_name = "mlx"

    def __init__(
        self,
        model_id: str = DEFAULT_XLAM_MODEL,
        *,
        static_prefix_marker: str | None = None,
    ) -> None:
        if model_id.startswith("~"):
            model_id = str(Path(model_id).expanduser())
        super().__init__(model_id)
        started = time.perf_counter()
        from mlx.utils import tree_flatten
        from mlx_lm import load

        self._model, self._tokenizer, config = load(model_id, return_config=True)
        self._static_prefix_marker = static_prefix_marker
        self._static_prefix_tokens: tuple[int, ...] | None = None
        self._static_prompt_cache: list[Any] | None = None
        flattened_parameters = tree_flatten(self._model.parameters())
        actual_dtype = (
            str(flattened_parameters[0][1].dtype) if flattened_parameters else None
        )
        self.load_latency_ms = (time.perf_counter() - started) * 1000
        self.model_metadata = {
            "actual_weight_dtype": actual_dtype,
            "configured_dtype": config.get("torch_dtype"),
            "model_type": config.get("model_type"),
            "quantization": config.get("quantization"),
            "context_length": config.get("max_position_embeddings"),
            "static_prefix_cache_enabled": static_prefix_marker is not None,
        }

    def _prepare_static_prefix(
        self, rendered: str
    ) -> tuple[str | list[int], list[Any] | None, int]:
        """Cache only an immutable prompt prefix; never cache requests or outputs."""

        marker = self._static_prefix_marker
        if marker is None:
            return rendered, None, 0
        marker_end = rendered.find(marker)
        if marker_end < 0:
            raise RuntimeError(f"Static prefix marker was not found: {marker!r}")
        marker_end += len(marker)
        add_special_tokens = (
            self._tokenizer.bos_token is None
            or not rendered.startswith(self._tokenizer.bos_token)
        )
        full_tokens = self._tokenizer.encode(
            rendered, add_special_tokens=add_special_tokens
        )
        candidate = self._tokenizer.encode(
            rendered[:marker_end], add_special_tokens=add_special_tokens
        )
        prefix_length = 0
        for expected, actual in zip(candidate, full_tokens):
            if expected != actual:
                break
            prefix_length += 1
        if prefix_length < MLX_PREFILL_STEP_SIZE:
            raise RuntimeError(
                "Static prompt prefix is shorter than MLX's prefill boundary"
            )
        prefix_length = MLX_PREFILL_STEP_SIZE
        prefix = tuple(int(token) for token in full_tokens[:prefix_length])

        if self._static_prompt_cache is None:
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache

            started = time.perf_counter()
            prompt_cache = make_prompt_cache(self._model)
            self._model(mx.array(prefix)[None], cache=prompt_cache)
            mx.eval([entry.state for entry in prompt_cache])
            self._static_prefix_tokens = prefix
            self._static_prompt_cache = prompt_cache
            self.model_metadata["static_prefix_tokens"] = len(prefix)
            self.model_metadata["static_prefix_build_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
        elif prefix != self._static_prefix_tokens:
            raise RuntimeError("The supposedly static prompt prefix changed")

        return (
            [int(token) for token in full_tokens[prefix_length:]],
            deepcopy(self._static_prompt_cache),
            prefix_length,
        )

    def _render(
        self,
        prompt: str,
        system_prompt: str | None = None,
        chat_template_tools: list[dict[str, Any]] | None = None,
        pre_rendered: bool = False,
    ) -> str:
        if pre_rendered:
            return prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if chat_template_tools is not None:
            kwargs["tools"] = chat_template_tools
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        max_tokens: int = MAX_GENERATION_TOKENS,
        *,
        system_prompt: str | None = None,
        chat_template_tools: list[dict[str, Any]] | None = None,
        json_root: str = "object",
        pre_rendered: bool = False,
    ) -> GenerationResult:
        del output_schema  # MLX-LM currently has no JSON-schema grammar interface.
        from mlx_lm import stream_generate

        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except (ImportError, AttributeError, RuntimeError):
            mx = None

        rendered = self._render(
            prompt, system_prompt, chat_template_tools, pre_rendered
        )
        generation_prompt, prompt_cache, cached_prompt_tokens = (
            self._prepare_static_prefix(rendered)
        )
        generated_token_ids: list[int] = []
        last = None
        finish_reason = None
        started = time.perf_counter()
        stream = stream_generate(
            self._model,
            self._tokenizer,
            prompt=generation_prompt,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
        )
        try:
            for response in stream:
                last = response
                generated_token_ids.append(int(response.token))
                decoded = self._tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                )
                parser_text = normalize_model_decoder_text(self.model_id, decoded)
                if is_complete_json_value(parser_text, json_root):
                    finish_reason = "json_complete"
                    break
        finally:
            stream.close()
        latency_ms = (time.perf_counter() - started) * 1000
        raw = self._tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        ).strip()
        normalized = normalize_model_decoder_text(self.model_id, raw).strip()
        if last is None:
            raise RuntimeError("MLX returned no generation tokens")
        peak_memory = last.peak_memory
        if mx is not None:
            peak_memory = max(peak_memory, mx.get_peak_memory() / 1e9)
        return GenerationResult(
            raw_text=raw,
            normalized_text=normalized if normalized != raw else None,
            rendered_prompt=rendered,
            backend=self.backend_name,
            model_id=self.model_id,
            latency_ms=latency_ms,
            prompt_tokens=last.prompt_tokens + cached_prompt_tokens,
            generated_tokens=last.generation_tokens,
            prompt_tokens_per_second=round(last.prompt_tps, 3),
            tokens_per_second=round(last.generation_tps, 3),
            peak_memory_gb=round(peak_memory, 4),
            process_rss_mb=_process_rss_mb(),
            finish_reason=finish_reason or last.finish_reason,
            schema_constrained=False,
            extra=self._generation_metadata(
                max_tokens,
                json_root=json_root,
                static_prefix_cache_used=bool(cached_prompt_tokens),
                static_prefix_tokens=cached_prompt_tokens,
            ),
        )


class TransformersBackend(DispatcherBackend):
    """PyTorch/MPS reference backend for comparing the same xLAM checkpoint."""

    backend_name = "transformers"

    def __init__(self, model_id: str = DEFAULT_XLAM_MODEL) -> None:
        if model_id.startswith("~"):
            model_id = str(Path(model_id).expanduser())
        super().__init__(model_id)
        started = time.perf_counter()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self._device == "mps" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=False,
        ).to(self._device)
        self._model.eval()
        self.load_latency_ms = (time.perf_counter() - started) * 1000
        self.model_metadata = {
            "runtime_dtype": str(dtype).replace("torch.", ""),
            "device": self._device,
            "model_type": getattr(self._model.config, "model_type", None),
            "context_length": getattr(
                self._model.config, "max_position_embeddings", None
            ),
        }

    def _render(
        self,
        prompt: str,
        system_prompt: str | None = None,
        chat_template_tools: list[dict[str, Any]] | None = None,
        pre_rendered: bool = False,
    ) -> str:
        if pre_rendered:
            return prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if chat_template_tools is not None:
            kwargs["tools"] = chat_template_tools
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        max_tokens: int = MAX_GENERATION_TOKENS,
        *,
        system_prompt: str | None = None,
        chat_template_tools: list[dict[str, Any]] | None = None,
        json_root: str = "object",
        pre_rendered: bool = False,
    ) -> GenerationResult:
        del output_schema
        torch = self._torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        rendered = self._render(
            prompt, system_prompt, chat_template_tools, pre_rendered
        )
        encoded = self._tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        tokenizer = self._tokenizer
        model_id = self.model_id

        class StopAfterJson(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                del scores, kwargs
                generated = input_ids[:, prompt_length:]
                stopped = []
                for row in generated:
                    text = tokenizer.decode(row, skip_special_tokens=True)
                    parser_text = normalize_model_decoder_text(model_id, text)
                    stopped.append(is_complete_json_value(parser_text, json_root))
                return torch.tensor(stopped, device=input_ids.device).unsqueeze(1)

        if self._device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([StopAfterJson()]),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        if self._device == "mps":
            torch.mps.synchronize()
        latency = time.perf_counter() - started
        generated = output[0, prompt_length:]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        normalized = normalize_model_decoder_text(self.model_id, raw).strip()
        token_count = int(generated.shape[-1])
        peak_memory = None
        if self._device == "mps":
            peak_memory = round(torch.mps.driver_allocated_memory() / 1e9, 4)
        return GenerationResult(
            raw_text=raw,
            normalized_text=normalized if normalized != raw else None,
            rendered_prompt=rendered,
            backend=self.backend_name,
            model_id=self.model_id,
            latency_ms=latency * 1000,
            prompt_tokens=int(prompt_length),
            generated_tokens=token_count,
            tokens_per_second=round(token_count / latency, 3) if latency else None,
            peak_memory_gb=peak_memory,
            process_rss_mb=_process_rss_mb(),
            finish_reason=(
                "json_complete"
                if is_complete_json_value(normalized, json_root)
                else "length_or_eos"
            ),
            schema_constrained=False,
            extra=self._generation_metadata(
                max_tokens, device=self._device, json_root=json_root
            ),
        )


class OpenAICompatibleBackend(DispatcherBackend):
    """Adapter for a local llama.cpp/LM Studio/OpenAI-compatible endpoint."""

    backend_name = "openai-compatible-local"

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str = "local",
        constrain_json: bool = True,
    ) -> None:
        super().__init__(model_id)
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._base_url = base_url
        self._constrain_json = constrain_json
        self.load_latency_ms = 0.0
        self.model_metadata = {
            "base_url": base_url,
            "schema_constraints_requested": constrain_json,
        }

    def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        max_tokens: int = MAX_GENERATION_TOKENS,
        *,
        system_prompt: str | None = None,
        chat_template_tools: list[dict[str, Any]] | None = None,
        json_root: str = "object",
        pre_rendered: bool = False,
    ) -> GenerationResult:
        if pre_rendered:
            started = time.perf_counter()
            response = self._client.completions.create(
                model=self.model_id,
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - started
            raw = response.choices[0].text or ""
            usage = response.usage
            rendered_prompt = prompt
            finish_reason = response.choices[0].finish_reason
            constrained = False
            generated_tokens = getattr(usage, "completion_tokens", None)
            raw = raw.strip()
            normalized = normalize_model_decoder_text(self.model_id, raw).strip()
            return GenerationResult(
                raw_text=raw,
                normalized_text=normalized if normalized != raw else None,
                rendered_prompt=rendered_prompt,
                backend=self.backend_name,
                model_id=self.model_id,
                latency_ms=latency * 1000,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                generated_tokens=generated_tokens,
                tokens_per_second=(
                    round(generated_tokens / latency, 3)
                    if generated_tokens is not None and latency
                    else None
                ),
                process_rss_mb=_process_rss_mb(),
                finish_reason=finish_reason,
                schema_constrained=constrained,
                extra=self._generation_metadata(
                    max_tokens, base_url=self._base_url, json_root=json_root
                ),
            )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if chat_template_tools is not None:
            kwargs["tools"] = chat_template_tools
            kwargs["parallel_tool_calls"] = False
        elif self._constrain_json:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "yuki_dispatch",
                    "strict": True,
                    "schema": output_schema,
                },
            }
        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - started
        message = response.choices[0].message
        if message.tool_calls:
            calls = []
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                calls.append(
                    {"name": tool_call.function.name, "arguments": arguments}
                )
            native_value: Any = calls
            if json_root == "object":
                native_value = {"tool_calls": calls}
            raw = json.dumps(native_value, ensure_ascii=False, separators=(",", ":"))
        else:
            raw = message.content or ""
        raw = raw.strip()
        normalized = normalize_model_decoder_text(self.model_id, raw).strip()
        usage = response.usage
        generated_tokens = getattr(usage, "completion_tokens", None)
        return GenerationResult(
            raw_text=raw,
            normalized_text=normalized if normalized != raw else None,
            rendered_prompt=json.dumps(
                {"messages": messages, "tools": chat_template_tools},
                ensure_ascii=False,
            ),
            backend=self.backend_name,
            model_id=self.model_id,
            latency_ms=latency * 1000,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            generated_tokens=generated_tokens,
            tokens_per_second=(
                round(generated_tokens / latency, 3)
                if generated_tokens is not None and latency
                else None
            ),
            process_rss_mb=_process_rss_mb(),
            finish_reason=response.choices[0].finish_reason,
            schema_constrained=(
                chat_template_tools is not None or self._constrain_json
            ),
            extra=self._generation_metadata(
                max_tokens, base_url=self._base_url, json_root=json_root
            ),
        )


class OpenRouterNativeToolBackend(DispatcherBackend):
    """Stateless OpenRouter backend using native API function calls only."""

    backend_name = "openrouter-native-tools"

    def __init__(
        self,
        model_id: str,
        *,
        provider: str | None = None,
        max_api_cost_usd: float = 0.1,
        max_retries: int = 2,
        reasoning_effort: str = "low",
        reasoning_exclude: bool = True,
        temperature: float = 0,
        seed: int = 0,
        tool_choice: str = "required",
    ) -> None:
        super().__init__(model_id)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=max_retries,
            timeout=90.0,
        )
        self._max_api_cost_usd = max_api_cost_usd
        self._api_cost_usd = 0.0
        self._max_retries = max_retries
        self._reasoning_config = {
            "effort": reasoning_effort,
            "exclude": reasoning_exclude,
        }
        self._temperature = temperature
        self._seed = seed
        self._tool_choice = tool_choice
        self._provider_routing: dict[str, Any] = {"require_parameters": True}
        if provider:
            self._provider_routing.update(
                {"order": [provider], "allow_fallbacks": False}
            )
        else:
            self._provider_routing["sort"] = "price"
        self.load_latency_ms = 0.0
        self.model_metadata = {
            "base_url": "https://openrouter.ai/api/v1",
            "native_tool_calls": True,
            "tool_choice": tool_choice,
            "parallel_tool_calls_parameter_sent": False,
            "reasoning": self._reasoning_config,
            "provider_routing": self._provider_routing,
            "max_api_cost_usd": max_api_cost_usd,
            "max_retries": max_retries,
            "temperature": temperature,
            "seed": seed,
        }

    def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        max_tokens: int = MAX_GENERATION_TOKENS,
        *,
        system_prompt: str | None = None,
        chat_template_tools: list[dict[str, Any]] | None = None,
        json_root: str = "object",
        pre_rendered: bool = False,
    ) -> GenerationResult:
        del output_schema
        if pre_rendered:
            raise ValueError("OpenRouter native tools require chat messages")
        if not chat_template_tools:
            raise ValueError("OpenRouter native tools require function schemas")
        if self._api_cost_usd >= self._max_api_cost_usd:
            raise RuntimeError(
                f"OpenRouter API cost guard reached ${self._api_cost_usd:.6f}"
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=chat_template_tools,
            tool_choice=self._tool_choice,
            temperature=self._temperature,
            seed=self._seed,
            max_tokens=max_tokens,
            extra_body={
                "reasoning": self._reasoning_config,
                "provider": self._provider_routing,
            },
        )
        latency = time.perf_counter() - started
        response_data = response.model_dump()
        usage_data = response.usage.model_dump() if response.usage else {}
        api_cost = usage_data.get("cost")
        if isinstance(api_cost, int | float):
            self._api_cost_usd += float(api_cost)

        message = response.choices[0].message
        calls = []
        for tool_call in message.tool_calls or []:
            arguments: Any = tool_call.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append(
                {"name": tool_call.function.name, "arguments": arguments}
            )
        raw = (
            json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
            if calls
            else (message.content or "")
        ).strip()
        generated_tokens = usage_data.get("completion_tokens")
        completion_details = usage_data.get("completion_tokens_details") or {}
        return GenerationResult(
            raw_text=raw,
            rendered_prompt=json.dumps(
                {"messages": messages, "tools": chat_template_tools},
                ensure_ascii=False,
            ),
            backend=self.backend_name,
            model_id=self.model_id,
            latency_ms=latency * 1000,
            prompt_tokens=usage_data.get("prompt_tokens"),
            generated_tokens=generated_tokens,
            tokens_per_second=(
                round(generated_tokens / latency, 3)
                if generated_tokens is not None and latency
                else None
            ),
            process_rss_mb=_process_rss_mb(),
            finish_reason=response.choices[0].finish_reason,
            schema_constrained=True,
            extra=self._generation_metadata(
                max_tokens,
                base_url="https://openrouter.ai/api/v1",
                json_root=json_root,
                requested_model=self.model_id,
                returned_model=response_data.get("model"),
                provider=response_data.get("provider"),
                native_tool_calls=True,
                tool_choice=self._tool_choice,
                reasoning_tokens=completion_details.get("reasoning_tokens", 0),
                api_cost_usd=api_cost,
                cumulative_api_cost_usd=round(self._api_cost_usd, 10),
                max_api_cost_usd=self._max_api_cost_usd,
                max_retries=self._max_retries,
                temperature=self._temperature,
                seed=self._seed,
                raw_api_response=response_data,
            ),
        )


class FixtureBackend(DispatcherBackend):
    """Predictable backend for unit tests; never exposed as a real model claim."""

    backend_name = "fixture"

    def __init__(
        self, generations: str | Iterable[str], model_id: str = "fixture"
    ) -> None:
        super().__init__(model_id)
        if isinstance(generations, str):
            generations = [generations]
        self._generations = iter(generations)
        self.load_latency_ms = 0.0

    def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        max_tokens: int = MAX_GENERATION_TOKENS,
        *,
        system_prompt: str | None = None,
        chat_template_tools: list[dict[str, Any]] | None = None,
        json_root: str = "object",
        pre_rendered: bool = False,
    ) -> GenerationResult:
        del output_schema, system_prompt, chat_template_tools, pre_rendered
        raw = next(self._generations)
        normalized = normalize_model_decoder_text(self.model_id, raw).strip()
        return GenerationResult(
            raw_text=raw,
            normalized_text=normalized if normalized != raw.strip() else None,
            rendered_prompt=prompt,
            backend=self.backend_name,
            model_id=self.model_id,
            latency_ms=1.0,
            prompt_tokens=100,
            generated_tokens=min(len(raw), max_tokens),
            tokens_per_second=100.0,
            finish_reason=(
                "json_complete"
                if is_complete_json_value(normalized, json_root)
                else "fixture"
            ),
            schema_constrained=False,
            extra=self._generation_metadata(max_tokens, json_root=json_root),
        )


def create_backend(
    kind: str,
    model_id: str = DEFAULT_XLAM_MODEL,
    *,
    base_url: str | None = None,
    api_key: str = "local",
    constrain_json: bool = True,
    provider: str | None = None,
    max_api_cost_usd: float = 0.1,
) -> DispatcherBackend:
    if kind == "mlx":
        return MLXBackend(model_id)
    if kind == "transformers":
        return TransformersBackend(model_id)
    if kind == "openai-compatible-local":
        if not base_url:
            raise ValueError("base_url is required for an OpenAI-compatible backend")
        return OpenAICompatibleBackend(
            model_id,
            base_url=base_url,
            api_key=api_key,
            constrain_json=constrain_json,
        )
    if kind == "openrouter-native-tools":
        return OpenRouterNativeToolBackend(
            model_id,
            provider=provider,
            max_api_cost_usd=max_api_cost_usd,
        )
    raise ValueError(f"Unknown backend: {kind}")
