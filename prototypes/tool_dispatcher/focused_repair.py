"""Execution-locked focused repairs for Qwen delegation and Hammer routing."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from .backends import create_backend
from .benchmark import write_report
from .delegation_dispatcher import (
    DelegationDispatcherRunner,
    ValidationOnlyDispatcher,
    _scope_metrics,
    failure_records,
)
from .delegation_mainbrain import (
    DELEGATION_SYSTEM_PROMPT,
    MainBrainDelegationRunner,
    OpenRouterDelegationBackend,
    prepare_dataset_cases,
)
from .prompting import HAMMER_FORMAT_INSTRUCTION, TASK_INSTRUCTION, build_prompt
from .registry import ToolRegistry

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "focused-repair"

FULL_STAGE_A = (
    ROOT
    / "reports"
    / "production-delegation"
    / "stage-a"
    / "production-delegation-mainbrain-qwen3.8-27b-openrouter-full450-v1.json"
)
FULL_STAGE_B = (
    ROOT
    / "reports"
    / "production-delegation"
    / "stage-b"
    / "production-delegation-hammer1.5b-qwen3.8-27b-full450-v1.json"
)
FULL_FAILURES = (
    ROOT
    / "reports"
    / "production-delegation"
    / "stage-b"
    / "production-delegation-hammer1.5b-qwen3.8-27b-full450-v1-failures.jsonl"
)
FULL_COMPARISON = (
    ROOT
    / "reports"
    / "production-delegation"
    / "qwen3.8-hammer1.5-full450-comparison-v1.json"
)
DERIVED_DATASET = ROOT / "production-delegation-source-v1.json"
PHASE2_DATASET = ROOT / "phase2-cases.json"

FROZEN_SHA256 = {
    FULL_STAGE_A: "ee68fa403cc3a6d211aa0ffad99a474794378e5bb94f0cbb70b8f94c363be5fa",
    FULL_STAGE_B: "84b619922e3b24c0080b17b91a5963114f024647112e467a6099ae9fab425f8a",
    FULL_FAILURES: "7d82ec2c4f5721ba56db781d903237ac37d84691638f2cecee97387a6e162de1",
    FULL_COMPARISON: "050646d9908bdc5ee03fd42e336d0c1585583512349320bd3ede236821ce40b8",
    DERIVED_DATASET: "ea6711ce0548d3771166f8f2f9a5e59ebd73acb14a9ddf7ecbf2f6589f1744c3",
    PHASE2_DATASET: "5b26b9a493ccc58de7b606a829b14487251df3a068a0bcfd1f69c3ae1f966466",
}

HAMMER_MODEL = "/Volumes/madisk/yuki-tool-dispatcher/Hammer2.1-1.5b-fp16"
QWEN_MODEL = "qwen/qwen3.8-27b"

A_TARGET_TOOLS = ("see", "recall")
A_REGRESSION_TOOLS = ("image", "remember")
B_TARGET_TOOLS = ("search", "image", "ask_claude")
REPAIR_VARIANTS = ("baseline", "schema", "schema_rejection")

A_FROZEN_DELEGATIONS = REPORT_ROOT / "focused-a-see-recall-frozen-delegations-v1.json"
A_REGRESSION_DELEGATIONS = REPORT_ROOT / "focused-a-image-remember-frozen-delegations-v1.json"
B_OLD_DELEGATIONS = REPORT_ROOT / "focused-b-verbatim-old-frozen-delegations-v1.json"
A_BASELINE_REPORT = REPORT_ROOT / "hammer15-see-recall-baseline-v1.json"
A_DIAGNOSIS = REPORT_ROOT / "hammer15-see-recall-rejection-diagnosis-v1.json"
B_OLD_ANALYSIS = REPORT_ROOT / "qwen38-verbatim-repair-old-analysis-v1.json"
B_NEW_DELEGATIONS = REPORT_ROOT / "qwen38-verbatim-repair-delegations-v1.json"
B_QWEN_COMPARISON = REPORT_ROOT / "qwen38-verbatim-repair-old-vs-new-v1.json"
B_QWEN_FAILURES = REPORT_ROOT / "qwen38-verbatim-repair-new-failures-v1.jsonl"

REPAIRED_SCHEMA_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "see": {
        "description": (
            "Obtain or inspect visual information from an existing visual source "
            "available to Yuki, such as the current camera, view, screen, or supplied "
            "image input. Use for looking at, visually inspecting, observing, describing "
            "a scene, or locating an existing object. This does not create a new image."
        ),
        "parameters": {
            "target": (
                "Optional existing object, region, or visual subject to locate or inspect. "
                "Omit it when the action is to describe the whole available scene."
            )
        },
    },
    "image": {
        "description": (
            "Create or generate a new image from a text description. Use for requests to "
            "draw, render, illustrate, paint, or generate new visual artwork. This does "
            "not inspect an existing camera, view, screen, scene, or image input."
        ),
        "parameters": {
            "prompt": "Text description of the new image to create or generate."
        },
    },
    "remember": {
        "description": (
            "Store a fact about the user in Yuki's persistent memory for future retrieval. "
            "This adds new remembered information; it does not retrieve information that "
            "was stored previously."
        ),
        "parameters": {
            "fact": "The user fact to store in persistent memory for future retrieval."
        },
    },
    "recall": {
        "description": (
            "Retrieve information previously stored in Yuki's persistent memory about a "
            "topic. Use when the action asks what the user previously told Yuki, asks to "
            "recall remembered information, or requests stored context concerning a topic. "
            "This retrieves existing memory; it does not store a new fact or rely on the "
            "dispatcher's conversational context."
        ),
        "parameters": {
            "topic": "Topic identifying the previously stored information to retrieve."
        },
    },
}

REPAIRED_TASK_INSTRUCTION = """You are a dedicated function dispatcher, not an assistant.
A tool is already needed. Select exactly one available tool and extract its arguments from the query.
Select a listed tool whenever its described capability can fulfill the delegated action. Reject only when none of the listed capabilities can perform the action or a genuinely required argument is unavailable. Do not reject merely because the operation uses external state; the listed tool provides that capability.
Never answer the query, explain the decision, greet the user, or generate any text outside the required JSON.
Stop immediately after the complete structured call."""

_OLD_VERBATIM_RULE = (
    "- verbatim: copy every exact-sensitive user string character-for-character. Copy "
    "the smallest complete argument spans, not the surrounding instruction or the "
    "entire user sentence. Use [] only when no exact string must survive. Preserve "
    "paths, filenames, URLs, commands, code, quoted text, file contents, image "
    "descriptions, search terms, memory facts/topics, and external-agent questions."
)

_LOSSLESS_VERBATIM_RULE = """- verbatim is a lossless transport channel for user-provided text that will become a downstream tool argument. When wording matters, copy the smallest complete argument span character-for-character.
- Never summarize, normalize, translate, correct spelling, change punctuation or capitalization, remove or alter quotes that belong to the payload, HTML-escape content, or rewrite pronouns inside a copied span.
- Exact-sensitive material includes search queries, image-generation descriptions, external-agent questions/messages, shell commands, URLs, file paths, filenames, quoted text, code, write/append payloads, and memory facts/topics where wording matters.
- Do not copy the whole user message by default. Exclude surrounding routing instructions such as "search for", "ask Claude", or "generate an image of" unless those words are part of the argument itself. Use [] only when no user-provided argument text must survive.
- The semantic request describes how the exact strings are to be used; verbatim carries the strings without rewriting them."""

_LOSSLESS_EXAMPLES = """- User says: Search for \"RTX 3060 12GB used Columbus\"
  request should describe searching with the supplied query and verbatim must be [\"RTX 3060 12GB used Columbus\"].
- User says: Ask Claude: \"Compare foo_bar() and fooBar(). Do not rename them.\"
  request should describe sending the supplied question to the external agent and verbatim must be [\"Compare foo_bar() and fooBar(). Do not rename them.\"].
- User says: Generate an image of \"a red fox under a neon blue moon\"
  request should describe generating an image with the supplied description and verbatim must be [\"a red fox under a neon blue moon\"].
"""

LOSSLESS_VERBATIM_SYSTEM_PROMPT = DELEGATION_SYSTEM_PROMPT.replace(
    _OLD_VERBATIM_RULE, _LOSSLESS_VERBATIM_RULE
).replace("Domain meanings:\n", f"{_LOSSLESS_EXAMPLES}\nDomain meanings:\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_artifacts() -> dict[str, str]:
    observed = {str(path): _sha256(path) for path in FROZEN_SHA256}
    mismatches = {
        str(path): {"expected": expected, "actual": observed[str(path)]}
        for path, expected in FROZEN_SHA256.items()
        if observed[str(path)] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen focused-repair inputs changed: {mismatches}")
    return observed


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n", encoding="utf-8"
    )
    return checksum


def _read_checksum(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    return sidecar.read_text(encoding="utf-8").split()[0]


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


class FocusedRepairRegistry(ToolRegistry):
    """Live Yuki registry with prototype-only Hammer description overrides."""

    def __init__(self, variant: str) -> None:
        if variant not in REPAIR_VARIANTS:
            raise ValueError(f"Unknown repair variant: {variant}")
        self.variant = variant
        super().__init__()

    def hammer_schemas(self, names: Any) -> list[dict[str, Any]]:
        schemas = super().hammer_schemas(names)
        if self.variant == "baseline":
            return schemas
        for schema in schemas:
            repair = REPAIRED_SCHEMA_DESCRIPTIONS.get(schema["name"])
            if repair is None:
                continue
            schema["description"] = repair["description"]
            for parameter, description in repair["parameters"].items():
                schema["parameters"][parameter]["description"] = description
        return schemas


def _variant_task_instruction(variant: str) -> str | None:
    return REPAIRED_TASK_INSTRUCTION if variant == "schema_rejection" else None


def _subset_stage_a(
    full_report: dict[str, Any], tools: tuple[str, ...], label: str
) -> dict[str, Any]:
    cases = [
        deepcopy(case)
        for case in full_report["case_results"]
        if case["expected_tool"] in tools
    ]
    expected_count = len(tools) * 25
    if len(cases) != expected_count:
        raise RuntimeError(f"Expected {expected_count} cases for {tools}, got {len(cases)}")
    configuration = deepcopy(full_report["configuration"])
    configuration.update(
        {
            "experiment": "focused_repair_frozen_stage_a_subset",
            "selection": label,
            "total_cases": len(cases),
            "derived_from_full_stage_a_sha256": FROZEN_SHA256[FULL_STAGE_A],
            "tool_execution_capability": False,
        }
    )
    return {
        "configuration": configuration,
        "contract": deepcopy(full_report["contract"]),
        "provenance": {
            "source": str(FULL_STAGE_A),
            "source_sha256": FROZEN_SHA256[FULL_STAGE_A],
            "tools": list(tools),
            "case_ids": [case["case_id"] for case in cases],
        },
        "case_results": cases,
    }


def _focused_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scope = _scope_metrics(cases)
    generations = [
        case["dispatcher"]["generation"]
        for case in cases
        if case.get("dispatcher") is not None
    ]
    rates = [
        generation.get("tokens_per_second")
        for generation in generations
        if generation.get("tokens_per_second") is not None
    ]
    peaks = [
        generation.get("peak_memory_gb")
        for generation in generations
        if generation.get("peak_memory_gb") is not None
    ]
    return {
        **scope,
        "rejection_count": sum(
            bool(case.get("dispatcher") and case["dispatcher"]["parse"]["rejected"])
            for case in cases
        ),
        "malformed_count": sum(case["malformed_output"] for case in cases),
        "average_generation_tokens_per_second": mean(rates) if rates else None,
        "peak_mlx_memory_gb": max(peaks) if peaks else None,
    }


def _decorate_focused_report(
    report: dict[str, Any], variant: str, source_path: Path
) -> dict[str, Any]:
    report["configuration"].update(
        {
            "experiment": "focused_repair_hammer_dispatch",
            "focused_repair_variant": variant,
            "schema_description_repair": variant != "baseline",
            "rejection_instruction_repair": variant == "schema_rejection",
            "task_instruction_sha256": hashlib.sha256(
                (_variant_task_instruction(variant) or TASK_INSTRUCTION).encode()
            ).hexdigest(),
            "hammer_format_instruction_sha256": hashlib.sha256(
                HAMMER_FORMAT_INSTRUCTION.encode()
            ).hexdigest(),
            "frozen_delegation_subset_sha256": _sha256(source_path),
            "tool_execution_capability": False,
        }
    )
    report["focused_metrics"] = {
        tool: _focused_metrics(
            [case for case in report["case_results"] if case["expected_tool"] == tool]
        )
        for tool in sorted({case["expected_tool"] for case in report["case_results"]})
    }
    return report


def _extract_baseline(full_stage_b: dict[str, Any]) -> dict[str, Any]:
    cases = [
        deepcopy(case)
        for case in full_stage_b["case_results"]
        if case["expected_tool"] in A_TARGET_TOOLS
    ]
    report = {
        "configuration": {
            **deepcopy(full_stage_b["configuration"]),
            "experiment": "focused_repair_hammer_extracted_baseline",
            "focused_repair_variant": "baseline",
            "source_full_stage_b_sha256": FROZEN_SHA256[FULL_STAGE_B],
            "total_cases": len(cases),
            "tool_execution_capability": False,
        },
        "summary": _focused_metrics(cases),
        "focused_metrics": {
            tool: _focused_metrics(
                [case for case in cases if case["expected_tool"] == tool]
            )
            for tool in A_TARGET_TOOLS
        },
        "case_results": cases,
    }
    return report


def _diagnosis_report(full_stage_b: dict[str, Any]) -> dict[str, Any]:
    registry = FocusedRepairRegistry("baseline")
    rejected = []
    for case in full_stage_b["case_results"]:
        if case["expected_tool"] not in A_TARGET_TOOLS:
            continue
        dispatch = case["dispatcher"]
        if not dispatch["parse"]["rejected"]:
            continue
        package = build_prompt(
            registry,
            dispatch["dispatcher_input"],
            dispatch["offered_tools"],
            output_mode="native",
            model_id=HAMMER_MODEL,
        )
        rejected.append(
            {
                "case_id": case["case_id"],
                "expected_tool": case["expected_tool"],
                "raw_user_request": case["request"],
                "expected_arguments": case["expected_arguments"],
                "frozen_qwen_delegation": case["delegation"],
                "schemas_shown": dispatch["schemas_sent"],
                "exact_native_prompt": package.content,
                "exact_native_prompt_sha256": hashlib.sha256(
                    package.content.encode()
                ).hexdigest(),
                "raw_hammer_generation": dispatch["generation"]["raw_text"],
                "normalized_hammer_generation": dispatch["generation"].get(
                    "normalized_text"
                ),
                "parser_result": dispatch["parse"],
                "schema_validator_result": dispatch["validation"],
            }
        )
    return {
        "version": "hammer15-see-recall-rejection-diagnosis-v1",
        "source_full_stage_b_sha256": FROZEN_SHA256[FULL_STAGE_B],
        "current_task_instruction": TASK_INSTRUCTION,
        "current_hammer_format_instruction": HAMMER_FORMAT_INSTRUCTION,
        "current_schemas": {
            "media": registry.hammer_schemas(("see", "image")),
            "memory": registry.hammer_schemas(("remember", "recall")),
        },
        "rejection_counts": dict(Counter(item["expected_tool"] for item in rejected)),
        "all_rejections_were_valid_empty_arrays": all(
            item["parser_result"]["rejected"]
            and not item["parser_result"]["malformed"]
            for item in rejected
        ),
        "diagnosis": {
            "see": (
                "The description uses assistant/personified webcam language and does not "
                "state the existing-visual-input versus generation distinction. Rejections "
                "occur with and without target strings, so missing arguments are not the "
                "dominant cause."
            ),
            "recall": (
                "The description asks a stateless dispatcher to reason about current-chat "
                "absence and future assistant reply behavior. Frozen requests instead use "
                "persistent-memory retrieval language, creating a capability wording mismatch."
            ),
            "prompt": (
                "The empty-array reject path is presented twice and can amplify uncertainty."
            ),
        },
        "rejected_cases": rejected,
    }


def _normal_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _alphanumeric(value: str) -> str:
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).casefold()


def _preservation_flags(case: dict[str, Any]) -> list[str]:
    required = case["required_verbatim"]
    emitted = case["delegation"]["verbatim"]
    if all(value in emitted for value in required):
        flags = []
        if len(emitted) > len(required):
            flags.append("unnecessary_material_in_verbatim")
        return flags

    flags: list[str] = []
    if not emitted:
        flags.append("omitted_from_verbatim")
    joined = " ".join(emitted)
    for expected in required:
        if expected in emitted:
            continue
        if expected in case["delegation"]["request"]:
            flags.append("exact_payload_appears_only_in_request")
        if any(value and (value in expected or expected in value) for value in emitted):
            flags.append("partially_copied")
        elif emitted:
            flags.append("paraphrased")
        if any(value.casefold() == expected.casefold() for value in emitted):
            flags.append("casing_changed")
        if emitted and _alphanumeric(joined) == _alphanumeric(expected):
            flags.append("punctuation_changed")
        if any(mark in expected for mark in ('"', "'", "`")) and not all(
            mark in joined for mark in set(expected) & {'"', "'", "`"}
        ):
            flags.append("quotes_removed_or_changed")
        if (
            html.unescape(joined) == expected
            and joined != expected
            or emitted
            and _normal_text(joined) == _normal_text(expected)
        ):
            flags.append("normalized")
    if len(required) > 1 and len(emitted) < len(required):
        flags.append("multiple_sensitive_strings_incorrectly_merged")
    if len(emitted) > 1 and len(required) == 1:
        flags.append("single_payload_incorrectly_split")
    return list(dict.fromkeys(flags))


def _delegation_subset_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [case["delegation_scores"] for case in cases]
    generations = [case["generation"] for case in cases]
    costs = [
        generation["extra"].get("api_cost_usd")
        for generation in generations
        if isinstance(generation["extra"].get("api_cost_usd"), int | float)
    ]
    return {
        "cases": len(cases),
        "valid_json_rate": mean(score["delegation_emitted"] for score in scores),
        "domain_accuracy": mean(score["domain_hint_exact"] for score in scores),
        "verbatim_recall": mean(score["verbatim_recall"] for score in scores),
        "verbatim_precision": mean(score["verbatim_precision"] for score in scores),
        "exact_character_preservation_rate": mean(
            score["verbatim_recall"] == 1.0 for score in scores
        ),
        "final_tool_leakage_rate": mean(
            score["final_tool_leakage"] for score in scores
        ),
        "average_latency_ms": mean(
            generation["latency_ms"] for generation in generations
        ),
        "average_generated_tokens": mean(
            generation["generated_tokens"] for generation in generations
        ),
        "api_cost_usd": round(sum(costs), 10),
    }


def _old_verbatim_analysis(full_stage_a: dict[str, Any]) -> dict[str, Any]:
    cases = [
        deepcopy(case)
        for case in full_stage_a["case_results"]
        if case["expected_tool"] in B_TARGET_TOOLS
    ]
    failures = []
    for case in cases:
        flags = _preservation_flags(case)
        if flags:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "expected_tool": case["expected_tool"],
                    "raw_user_request": case["request"],
                    "expected_arguments": case["expected_arguments"],
                    "required_verbatim": case["required_verbatim"],
                    "frozen_request": case["delegation"]["request"],
                    "frozen_verbatim": case["delegation"]["verbatim"],
                    "classification_flags": flags,
                }
            )
    return {
        "version": "qwen38-verbatim-repair-old-analysis-v1",
        "source_full_stage_a_sha256": FROZEN_SHA256[FULL_STAGE_A],
        "tools": list(B_TARGET_TOOLS),
        "metrics": _delegation_subset_metrics(cases),
        "by_tool": {
            tool: _delegation_subset_metrics(
                [case for case in cases if case["expected_tool"] == tool]
            )
            for tool in B_TARGET_TOOLS
        },
        "classification_is_nonexclusive": True,
        "classification_counts": dict(
            Counter(flag for failure in failures for flag in failure["classification_flags"])
        ),
        "failures": failures,
    }


def prepare() -> None:
    verify_frozen_artifacts()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stage_a = _load(FULL_STAGE_A)
    stage_b = _load(FULL_STAGE_B)
    outputs = {
        A_FROZEN_DELEGATIONS: _subset_stage_a(
            stage_a, A_TARGET_TOOLS, "focused-a-see-recall-50"
        ),
        A_REGRESSION_DELEGATIONS: _subset_stage_a(
            stage_a, A_REGRESSION_TOOLS, "focused-a-image-remember-regression-50"
        ),
        B_OLD_DELEGATIONS: _subset_stage_a(
            stage_a, B_TARGET_TOOLS, "focused-b-verbatim-old-75"
        ),
        A_BASELINE_REPORT: _extract_baseline(stage_b),
        A_DIAGNOSIS: _diagnosis_report(stage_b),
        B_OLD_ANALYSIS: _old_verbatim_analysis(stage_a),
    }
    for path, payload in outputs.items():
        write_report(payload, path)
        checksum = _write_checksum(path)
        print(f"{checksum}  {path.name}")


def run_hammer(
    delegation_path: Path,
    variant: str,
    output: Path,
    failures_output: Path,
) -> None:
    verify_frozen_artifacts()
    expected = _read_checksum(delegation_path)
    actual = _sha256(delegation_path)
    if actual != expected:
        raise RuntimeError(f"Focused delegation checksum mismatch: {actual} != {expected}")
    registry = FocusedRepairRegistry(variant)
    backend = create_backend("mlx", HAMMER_MODEL)
    dispatcher = ValidationOnlyDispatcher(
        registry,
        backend,
        input_mode="delegated_with_verbatim",
        domain_source="generated",
        task_instruction=_variant_task_instruction(variant),
    )
    report = DelegationDispatcherRunner(dispatcher).run(
        _load(delegation_path),
        delegation_path=delegation_path,
        expected_delegation_sha256=expected,
    )
    report = _decorate_focused_report(report, variant, delegation_path)
    write_report(report, output)
    _write_jsonl(failure_records(report), failures_output)
    print(_write_checksum(output))
    print(_write_checksum(failures_output))


def run_qwen() -> None:
    verify_frozen_artifacts()
    dataset = prepare_dataset_cases(_load(DERIVED_DATASET))
    ids = [
        case["case_id"]
        for case in dataset["cases"]
        if case["expected_tool"] in B_TARGET_TOOLS
    ]
    if len(ids) != 75:
        raise RuntimeError(f"Expected 75 Qwen focused cases, got {len(ids)}")
    dataset["selection_sets"]["focused-verbatim-75"] = ids
    backend = OpenRouterDelegationBackend(
        QWEN_MODEL,
        max_api_cost_usd=0.10,
        system_prompt=LOSSLESS_VERBATIM_SYSTEM_PROMPT,
    )
    report = MainBrainDelegationRunner(backend, ToolRegistry()).run(
        dataset,
        derived_dataset_path=DERIVED_DATASET,
        expected_dataset_sha256=FROZEN_SHA256[DERIVED_DATASET],
        selection="focused-verbatim-75",
    )
    returned_models = {
        case["generation"]["extra"].get("returned_model")
        for case in report["case_results"]
    }
    reasoning_tokens = {
        case["generation"]["extra"].get("reasoning_tokens")
        for case in report["case_results"]
    }
    if returned_models != {QWEN_MODEL} or reasoning_tokens != {0}:
        raise RuntimeError(
            f"OpenRouter behavior changed: models={returned_models}, "
            f"reasoning_tokens={reasoning_tokens}"
        )
    report["configuration"].update(
        {
            "experiment": "focused_repair_lossless_verbatim_stage_a",
            "delegation_contract_version": "focused-lossless-verbatim-v1",
            "tool_execution_capability": False,
            "source_old_full_stage_a_sha256": FROZEN_SHA256[FULL_STAGE_A],
        }
    )
    write_report(report, B_NEW_DELEGATIONS)
    print(_write_checksum(B_NEW_DELEGATIONS))


def compare_qwen() -> None:
    verify_frozen_artifacts()
    expected = _read_checksum(B_NEW_DELEGATIONS)
    if _sha256(B_NEW_DELEGATIONS) != expected:
        raise RuntimeError("New Qwen focused delegations changed after freezing")
    old = _load(B_OLD_DELEGATIONS)
    new = _load(B_NEW_DELEGATIONS)
    old_by_id = {case["case_id"]: case for case in old["case_results"]}
    new_by_id = {case["case_id"]: case for case in new["case_results"]}
    if old_by_id.keys() != new_by_id.keys():
        raise RuntimeError("Old and new Qwen focused case IDs differ")

    def analysis(report: dict[str, Any]) -> dict[str, Any]:
        cases = report["case_results"]
        failures = []
        for case in cases:
            flags = _preservation_flags(case)
            if flags:
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "expected_tool": case["expected_tool"],
                        "raw_user_request": case["request"],
                        "required_verbatim": case["required_verbatim"],
                        "delegation": case["delegation"],
                        "classification_flags": flags,
                    }
                )
        return {
            "metrics": _delegation_subset_metrics(cases),
            "by_tool": {
                tool: _delegation_subset_metrics(
                    [case for case in cases if case["expected_tool"] == tool]
                )
                for tool in B_TARGET_TOOLS
            },
            "failure_count": len(failures),
            "classification_is_nonexclusive": True,
            "classification_counts": dict(
                Counter(
                    flag
                    for failure in failures
                    for flag in failure["classification_flags"]
                )
            ),
            "failures": failures,
        }

    old_analysis = analysis(old)
    new_analysis = analysis(new)
    rate_fields = (
        "valid_json_rate",
        "domain_accuracy",
        "verbatim_recall",
        "verbatim_precision",
        "exact_character_preservation_rate",
        "final_tool_leakage_rate",
    )
    comparison = {
        "version": "qwen38-verbatim-repair-old-vs-new-v1",
        "status": "completed_negative_result",
        "model": QWEN_MODEL,
        "old_delegations_sha256": _sha256(B_OLD_DELEGATIONS),
        "new_delegations_sha256": expected,
        "old": old_analysis,
        "new": new_analysis,
        "change_percentage_points": {
            field: (
                new_analysis["metrics"][field] - old_analysis["metrics"][field]
            )
            * 100
            for field in rate_fields
        },
        "paired_cases": [
            {
                "case_id": case_id,
                "expected_tool": old_by_id[case_id]["expected_tool"],
                "old_domain": old_by_id[case_id]["delegation"]["domain_hint"],
                "new_domain": new_by_id[case_id]["delegation"]["domain_hint"],
                "old_verbatim": old_by_id[case_id]["delegation"]["verbatim"],
                "new_verbatim": new_by_id[case_id]["delegation"]["verbatim"],
                "required_verbatim": old_by_id[case_id]["required_verbatim"],
                "old_exact": old_by_id[case_id]["delegation_scores"][
                    "verbatim_recall"
                ]
                == 1.0,
                "new_exact": new_by_id[case_id]["delegation_scores"][
                    "verbatim_recall"
                ]
                == 1.0,
            }
            for case_id in old_by_id
        ],
        "decision": (
            "Reject the lossless-verbatim prompt candidate. It reduced exact "
            "preservation and domain accuracy; do not iterate on the frozen test set."
        ),
        "safety": {
            "yuki_tool_execution_capability": False,
            "yuki_tool_execution_attempts": 0,
        },
    }
    write_report(comparison, B_QWEN_COMPARISON)
    _write_jsonl(new_analysis["failures"], B_QWEN_FAILURES)
    print(_write_checksum(B_QWEN_COMPARISON))
    print(_write_checksum(B_QWEN_FAILURES))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execution-locked focused repair")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    hammer = commands.add_parser("run-hammer")
    hammer.add_argument("--delegations", required=True)
    hammer.add_argument("--variant", choices=REPAIR_VARIANTS, required=True)
    hammer.add_argument("--output", required=True)
    hammer.add_argument("--failures-output", required=True)
    commands.add_parser("run-qwen")
    commands.add_parser("compare-qwen")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare()
        return
    if args.command == "run-qwen":
        run_qwen()
        return
    if args.command == "compare-qwen":
        compare_qwen()
        return
    run_hammer(
        Path(args.delegations),
        args.variant,
        Path(args.output),
        Path(args.failures_output),
    )


if __name__ == "__main__":
    main()
