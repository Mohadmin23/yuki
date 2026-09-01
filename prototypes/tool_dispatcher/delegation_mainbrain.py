"""Stateless Stage A main-brain delegation generation and scoring."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean, median
from typing import Any

from .contracts import GenerationResult
from .delegation_contract import DELEGATION_SCHEMA, parse_delegation, score_delegation
from .json_boundary import is_complete_json_value
from .registry import ToolRegistry

DELEGATION_GENERATION_TOKENS = 160
SURROGATE_LABEL = "Qwen3-14B-MLX-4bit temporary main-brain surrogate"
DELEGATION_CONTRACT_VERSION = "production-delegation-contract-v1.1"

DELEGATION_SYSTEM_PROMPT = """You are Yuki's internal tool-delegation planner.
You are not answering the user and you are not selecting an implementation-level tool.
Every input in this evaluation requires one external action.

Return only one compact JSON object with exactly these fields:
{"request":"semantic external action","domain_hint":"information|computer|media|yuki_files|memory|external_agent","verbatim":["exact user strings"]}

Rules:
- request: a concise semantic description of the required external action. Include enough meaning for a separate dispatcher to choose the implementation. Do not say which implementation-level tool or function to call.
- domain_hint: choose exactly one broad capability family.
- verbatim: copy every exact-sensitive user string character-for-character. Copy the smallest complete argument spans, not the surrounding instruction or the entire user sentence. Use [] only when no exact string must survive. Preserve paths, filenames, URLs, commands, code, quoted text, file contents, image descriptions, search terms, memory facts/topics, and external-agent questions.
- Never answer the request, explain, greet, use Markdown, or emit text outside JSON.

Verbatim examples:
- User says: Please inspect /tmp/Foo.JSON without changing case.
  verbatim must be ["/tmp/Foo.JSON"], not the whole sentence.
- User says: Keep this exact fact: my cat's name is Nori.
  verbatim must be ["my cat's name is Nori"].
- User says: Add `? value_2` to `odd name.txt` in Yuki's files.
  verbatim must be ["? value_2","odd name.txt"].
- User says: Ask the external agent about Tokyo routing; keep this unchanged.
  verbatim must contain only the question payload, ["about Tokyo routing; keep this unchanged"].

Domain meanings:
- information: current time/date, weather, exact URL retrieval, web information finding, or arithmetic.
- computer: local hardware/resource inspection, normal filesystem access, or terminal-style operations.
- media: camera/vision inspection or creating visual artwork.
- yuki_files: create, read, list, delete, or append Yuki's own personal files.
- memory: store a user fact for later or retrieve previously stored user information.
- external_agent: delegate a question or task to Claude or another external reasoning agent.
"""


def _process_rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1024**2, 3)
    except (ImportError, OSError):
        return None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


class DelegationBackend(ABC):
    """One original request in, one raw delegation generation out."""

    backend_name = "base"
    main_brain_role = "temporary_surrogate"
    surrogate = True
    model_label = SURROGATE_LABEL
    external_api_usage = False
    thinking_disabled = True

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.system_prompt = DELEGATION_SYSTEM_PROMPT
        self.generation_ceiling = DELEGATION_GENERATION_TOKENS
        self.load_latency_ms: float | None = None
        self.model_metadata: dict[str, Any] = {}
        self.completed_generations = 0

    @abstractmethod
    def generate(self, request: str) -> GenerationResult:
        raise NotImplementedError


class MLXDelegationBackend(DelegationBackend):
    """Local Qwen Stage A backend with thinking disabled and JSON stopping."""

    backend_name = "mlx"

    def __init__(self, model_id: str) -> None:
        if model_id.startswith("~"):
            model_id = str(Path(model_id).expanduser())
        super().__init__(model_id)
        started = time.perf_counter()
        from mlx.utils import tree_flatten
        from mlx_lm import load

        self._model, self._tokenizer, config = load(model_id, return_config=True)
        # This is a computation cache, not conversational state. Each lookup is
        # trimmed to the exact token prefix shared by the current and previous
        # rendered prompts before the current request is processed.
        self._cached_prompt_tokens: list[int] | None = None
        self._cached_prompt_cache: list[Any] | None = None
        flattened = tree_flatten(self._model.parameters())
        actual_dtype = str(flattened[0][1].dtype) if flattened else None
        self.load_latency_ms = (time.perf_counter() - started) * 1000
        self.model_metadata = {
            "actual_weight_dtype": actual_dtype,
            "configured_dtype": config.get("torch_dtype"),
            "model_type": config.get("model_type"),
            "quantization": config.get("quantization"),
            "context_length": config.get("max_position_embeddings"),
            "thinking_disabled": True,
        }

    def generate(self, request: str) -> GenerationResult:
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except (ImportError, AttributeError, RuntimeError):
            mx = None

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": request},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        add_special_tokens = (
            self._tokenizer.bos_token is None
            or not rendered.startswith(self._tokenizer.bos_token)
        )
        full_prompt_tokens = self._tokenizer.encode(
            rendered, add_special_tokens=add_special_tokens
        )
        if self._cached_prompt_cache is None or self._cached_prompt_tokens is None:
            prompt_cache = make_prompt_cache(self._model)
            remaining_prompt_tokens = full_prompt_tokens
            reused_prefix_tokens = 0
        else:
            common_prefix = 0
            for previous, current in zip(
                self._cached_prompt_tokens, full_prompt_tokens, strict=False
            ):
                if previous != current:
                    break
                common_prefix += 1
            # MLX generation needs at least one uncached prompt token.
            reused_prefix_tokens = min(common_prefix, len(full_prompt_tokens) - 1)
            prompt_cache = copy.deepcopy(self._cached_prompt_cache)
            requested_trim = len(self._cached_prompt_tokens) - reused_prefix_tokens
            actual_trim = trim_prompt_cache(
                prompt_cache,
                requested_trim,
            )
            if actual_trim != requested_trim:
                prompt_cache = make_prompt_cache(self._model)
                remaining_prompt_tokens = full_prompt_tokens
                reused_prefix_tokens = 0
            else:
                remaining_prompt_tokens = full_prompt_tokens[reused_prefix_tokens:]
        generated_ids: list[int] = []
        last = None
        finish_reason = None
        started = time.perf_counter()
        stream = stream_generate(
            self._model,
            self._tokenizer,
            prompt=remaining_prompt_tokens,
            max_tokens=DELEGATION_GENERATION_TOKENS,
            prompt_cache=prompt_cache,
        )
        try:
            for response in stream:
                last = response
                generated_ids.append(int(response.token))
                decoded = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                if is_complete_json_value(decoded, "object"):
                    finish_reason = "json_complete"
                    break
        finally:
            stream.close()

        if last is None:
            raise RuntimeError("MLX returned no delegation tokens")
        trimmed_tokens = trim_prompt_cache(prompt_cache, last.generation_tokens)
        cache_inserted = trimmed_tokens == last.generation_tokens
        if cache_inserted:
            self._cached_prompt_tokens = full_prompt_tokens
            self._cached_prompt_cache = prompt_cache
        latency_ms = (time.perf_counter() - started) * 1000
        raw = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        peak_memory = last.peak_memory
        if mx is not None:
            peak_memory = max(peak_memory, mx.get_peak_memory() / 1e9)
        self.completed_generations += 1
        return GenerationResult(
            raw_text=raw,
            rendered_prompt=rendered,
            backend=self.backend_name,
            model_id=self.model_id,
            latency_ms=latency_ms,
            prompt_tokens=len(full_prompt_tokens),
            generated_tokens=last.generation_tokens,
            prompt_tokens_per_second=round(last.prompt_tps, 3),
            tokens_per_second=round(last.generation_tps, 3),
            peak_memory_gb=round(peak_memory, 4),
            process_rss_mb=_process_rss_mb(),
            finish_reason=finish_reason or last.finish_reason,
            schema_constrained=False,
            extra={
                "generation_ceiling": DELEGATION_GENERATION_TOKENS,
                "generation_index_after_load": self.completed_generations,
                "enable_thinking": False,
                "json_root": "object",
                "prompt_cache_kind": "exact_common_prefix_only",
                "full_prompt_tokens": len(full_prompt_tokens),
                "processed_prompt_tokens": last.prompt_tokens,
                "reused_prefix_tokens": reused_prefix_tokens,
                "prompt_cache_inserted": cache_inserted,
                "prompt_tps_scope": "uncached_suffix_only",
            },
        )


class OpenRouterDelegationBackend(DelegationBackend):
    """Intended main-brain backend using stateless constrained API calls."""

    backend_name = "openrouter"
    main_brain_role = "intended_main_brain"
    surrogate = False
    model_label = "OpenRouter intended main brain"
    external_api_usage = True

    def __init__(
        self,
        model_id: str,
        *,
        max_api_cost_usd: float = 1.0,
        system_prompt: str | None = None,
        generation_ceiling: int = DELEGATION_GENERATION_TOKENS,
        provider: str | None = None,
    ) -> None:
        super().__init__(model_id)
        self.generation_ceiling = generation_ceiling
        self.model_label = f"{model_id} intended main brain via OpenRouter"
        self.thinking_disabled = not model_id.startswith("openai/gpt-oss-")
        self._reasoning_config = {
            "effort": "none" if self.thinking_disabled else "low",
            "exclude": True,
        }
        if system_prompt is not None:
            self.system_prompt = system_prompt
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it before running Stage A."
            )
        from openai import OpenAI

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=2,
            timeout=90.0,
        )
        self._max_api_cost_usd = max_api_cost_usd
        self._api_cost_usd = 0.0
        self._provider_routing: dict[str, Any] = {"require_parameters": True}
        if provider:
            self._provider_routing.update(
                {"order": [provider], "allow_fallbacks": False}
            )
        else:
            self._provider_routing["sort"] = "price"
        self.load_latency_ms = 0.0
        self.model_metadata = {
            "requested_model": model_id,
            "base_url": "https://openrouter.ai/api/v1",
            "schema_constraints_requested": True,
            "thinking_disabled": self.thinking_disabled,
            "reasoning_config": self._reasoning_config,
            "provider_routing": self._provider_routing,
            "generation_ceiling": self.generation_ceiling,
            "max_api_cost_usd": max_api_cost_usd,
        }

    def generate(self, request: str) -> GenerationResult:
        if self._api_cost_usd >= self._max_api_cost_usd:
            raise RuntimeError(
                f"Stage A API cost guard reached ${self._api_cost_usd:.6f}."
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": request},
        ]
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0,
            seed=0,
            max_tokens=self.generation_ceiling,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "yuki_main_brain_delegation",
                    "strict": True,
                    "schema": DELEGATION_SCHEMA,
                },
            },
            extra_body={
                "reasoning": self._reasoning_config,
                "provider": self._provider_routing,
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        response_data = response.model_dump()
        usage_data = response.usage.model_dump() if response.usage else {}
        generated_tokens = usage_data.get("completion_tokens")
        api_cost = usage_data.get("cost")
        if isinstance(api_cost, int | float):
            self._api_cost_usd += float(api_cost)

        choice = response.choices[0]
        raw = (choice.message.content or "").strip()
        provider = response_data.get("provider")
        returned_model = response_data.get("model")
        completion_details = usage_data.get("completion_tokens_details") or {}
        self.completed_generations += 1
        return GenerationResult(
            raw_text=raw,
            rendered_prompt=json.dumps(messages, ensure_ascii=False),
            backend=self.backend_name,
            model_id=self.model_id,
            latency_ms=latency_ms,
            prompt_tokens=usage_data.get("prompt_tokens"),
            generated_tokens=generated_tokens,
            tokens_per_second=(
                round(generated_tokens / (latency_ms / 1000), 3)
                if generated_tokens is not None and latency_ms
                else None
            ),
            process_rss_mb=_process_rss_mb(),
            finish_reason=choice.finish_reason,
            schema_constrained=True,
            extra={
                "generation_ceiling": self.generation_ceiling,
                "generation_index_after_load": self.completed_generations,
                "enable_thinking": not self.thinking_disabled,
                "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
                "json_root": "object",
                "requested_model": self.model_id,
                "returned_model": returned_model,
                "provider": provider,
                "api_cost_usd": api_cost,
                "cumulative_api_cost_usd": round(self._api_cost_usd, 10),
                "max_api_cost_usd": self._max_api_cost_usd,
            },
        )


class FixtureDelegationBackend(DelegationBackend):
    backend_name = "fixture"

    def __init__(self, generations: list[str]) -> None:
        super().__init__("fixture-main-brain")
        self._generations = iter(generations)

    def generate(self, request: str) -> GenerationResult:
        raw = next(self._generations)
        self.completed_generations += 1
        return GenerationResult(
            raw_text=raw,
            rendered_prompt=request,
            backend=self.backend_name,
            model_id=self.model_id,
            latency_ms=1.0,
            generated_tokens=len(raw),
            tokens_per_second=100.0,
            finish_reason="fixture",
            extra={"generation_ceiling": DELEGATION_GENERATION_TOKENS},
        )


def _selected_cases(dataset: dict[str, Any], selection: str) -> list[dict[str, Any]]:
    try:
        ids = dataset["selection_sets"][selection]
    except KeyError as exc:
        raise ValueError(f"Unknown derived selection: {selection}") from exc
    by_id = {case["case_id"]: case for case in dataset["cases"]}
    return [by_id[case_id] for case_id in ids]


def _group_metrics(cases: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({case[field] for case in cases})
    result = {}
    for value in values:
        scoped = [case for case in cases if case[field] == value]
        scores = [case["delegation_scores"] for case in scoped]
        result[value] = {
            "cases": len(scoped),
            "valid_delegation_rate": mean(
                score["delegation_emitted"] for score in scores
            ),
            "domain_hint_accuracy": mean(
                score["domain_hint_exact"] for score in scores
            ),
            "semantic_request_quality_rate": mean(
                score["semantic_request_quality"] for score in scores
            ),
            "average_verbatim_recall": mean(
                score["verbatim_recall"] for score in scores
            ),
            "average_verbatim_precision": mean(
                score["verbatim_precision"] for score in scores
            ),
        }
    return result


class MainBrainDelegationRunner:
    """Generate and score delegations; this class has no Yuki execution API."""

    def __init__(self, backend: DelegationBackend, registry: ToolRegistry) -> None:
        self.backend = backend
        self.registry = registry

    def run(
        self,
        dataset: dict[str, Any],
        *,
        derived_dataset_path: str | Path,
        expected_dataset_sha256: str,
        selection: str,
    ) -> dict[str, Any]:
        dataset_path = Path(derived_dataset_path)
        actual_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if actual_sha != expected_dataset_sha256:
            raise RuntimeError(
                f"Derived dataset checksum mismatch: {actual_sha} != "
                f"{expected_dataset_sha256}"
            )

        results = []
        selected = _selected_cases(dataset, selection)
        for index, case in enumerate(selected):
            print(
                f"Stage A {index + 1}/{len(selected)}: {case['case_id']}",
                flush=True,
            )
            generation = self.backend.generate(case["request"])
            parsed = parse_delegation(generation.raw_text)
            scores = score_delegation(
                parsed,
                expected_tool=case["expected_tool"],
                expected_domain=case["expected_domain"],
                required_verbatim=case["required_verbatim"],
                live_tool_names=self.registry.names,
            )
            generation_data = generation.as_dict()
            generation_data.pop("rendered_prompt", None)
            results.append(
                {
                    "index": index,
                    **case,
                    "context": None,
                    "raw_delegation_generation": generation.raw_text,
                    "generation": generation_data,
                    "parse": parsed,
                    "delegation": parsed["delegation"],
                    "delegation_scores": scores,
                }
            )

        scores = [case["delegation_scores"] for case in results]
        latencies = [case["generation"]["latency_ms"] for case in results]
        token_rates = [
            case["generation"]["tokens_per_second"]
            for case in results
            if case["generation"]["tokens_per_second"] is not None
        ]
        token_counts = [
            case["generation"]["generated_tokens"]
            for case in results
            if case["generation"]["generated_tokens"] is not None
        ]
        peak_memory = [
            case["generation"]["peak_memory_gb"]
            for case in results
            if case["generation"]["peak_memory_gb"] is not None
        ]
        api_costs = [
            case["generation"]["extra"].get("api_cost_usd")
            for case in results
            if isinstance(
                case["generation"]["extra"].get("api_cost_usd"), int | float
            )
        ]
        providers = sorted(
            {
                case["generation"]["extra"].get("provider")
                for case in results
                if case["generation"]["extra"].get("provider")
            }
        )
        return {
            "configuration": {
                "experiment": "production_delegation_stage_a",
                "main_brain_role": self.backend.main_brain_role,
                "surrogate": self.backend.surrogate,
                "model_label": self.backend.model_label,
                "model_id": self.backend.model_id,
                "backend": self.backend.backend_name,
                "generation_ceiling": self.backend.generation_ceiling,
                "delegation_contract_version": DELEGATION_CONTRACT_VERSION,
                "delegation_system_prompt_sha256": hashlib.sha256(
                    self.backend.system_prompt.encode("utf-8")
                ).hexdigest(),
                "thinking_disabled": self.backend.thinking_disabled,
                "stateless": True,
                "prompt_cache_policy": (
                    "exact shared rendered-token prefix only; no prior request "
                    "tokens or generations enter the current context"
                ),
                "generation_count_per_case": 1,
                "tool_execution_capability": False,
                "external_api_usage": self.backend.external_api_usage,
                "selection": selection,
                "derived_dataset_version": dataset["dataset_version"],
                "derived_dataset_sha256": actual_sha,
                "source_dataset_sha256": dataset["source_dataset_sha256"],
                "total_cases": len(results),
                "model_metadata": self.backend.model_metadata,
                "load_latency_ms": self.backend.load_latency_ms,
            },
            "contract": {
                "fields": ["request", "domain_hint", "verbatim"],
                "domain_source": "main-brain-generated domain_hint",
                "tool_name_from_main_brain": False,
                "original_request_sent_to_dispatcher": False,
                "no_tool_case_coverage": 0,
                "conversation_context_case_coverage": 0,
            },
            "summary": {
                "cases": len(results),
                "valid_delegation_rate": mean(
                    score["delegation_emitted"] for score in scores
                ),
                "malformed_delegation_rate": mean(
                    score["malformed_delegation"] for score in scores
                ),
                "domain_hint_accuracy": mean(
                    score["domain_hint_exact"] for score in scores
                ),
                "semantic_request_quality_rate": mean(
                    score["semantic_request_quality"] for score in scores
                ),
                "average_verbatim_recall": mean(
                    score["verbatim_recall"] for score in scores
                ),
                "average_verbatim_precision": mean(
                    score["verbatim_precision"] for score in scores
                ),
                "final_tool_leakage_rate": mean(
                    score["final_tool_leakage"] for score in scores
                ),
                "average_latency_ms": mean(latencies),
                "p50_latency_ms": median(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "average_generated_tokens": mean(token_counts),
                "average_tokens_per_second": mean(token_rates),
                "peak_mlx_memory_gb": max(peak_memory) if peak_memory else None,
                "api_cost_usd": round(sum(api_costs), 10),
                "api_cost_reported_cases": len(api_costs),
                "providers": providers,
            },
            "by_expected_domain": _group_metrics(results, "expected_domain"),
            "by_dataset_category": _group_metrics(results, "tags_category"),
            "case_results": results,
        }


def prepare_dataset_cases(dataset: dict[str, Any]) -> dict[str, Any]:
    """Add a stable category field used by report grouping."""

    prepared = json.loads(json.dumps(dataset))
    for case in prepared["cases"]:
        case["tags_category"] = case["tags"][1]
    return prepared
