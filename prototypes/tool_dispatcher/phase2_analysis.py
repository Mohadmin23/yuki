"""Paired and confusion analysis for completed Phase 2 finalist reports."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

FOCUS_CLUSTERS = {
    "filesystem": ("read", "yuki_read", "shell"),
    "information": ("weather", "search", "fetch"),
    "media": ("see", "image"),
    "memory": ("remember", "recall"),
    "yuki_file_mutation": ("yuki_write", "yuki_append"),
}

DIMENSIONS = {
    "strict_exact_call": "end_to_end_strict_exact_call",
    "execution_equivalent_exact_call": (
        "end_to_end_execution_equivalent_exact_call"
    ),
    "tool_selection": "tool_correct",
}


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _outcome_signature(case: dict[str, Any], dimension: str) -> str:
    if dimension == "tool_selection":
        return str(case["actual_tool"] or "<none>")
    return json.dumps(
        {
            "call": case["normalized_call"],
            "malformed": case["malformed"],
            "rejected": case["parse"].get("rejected", False),
            "validation": case["validation"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def exact_mcnemar(hammer_only: int, arch_only: int) -> dict[str, Any]:
    """Return a two-sided exact binomial McNemar test for discordant pairs."""

    discordant = hammer_only + arch_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail_end = min(hammer_only, arch_only)
        tail = sum(
            math.comb(discordant, value) for value in range(tail_end + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "test": "two-sided exact McNemar (binomial discordant-pair test)",
        "hammer_only": hammer_only,
        "arch_agent_only": arch_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
        "interpretation_guard": (
            "The hand-designed dataset is not a random sample of all future Yuki requests."
        ),
    }


def _verify_pair(
    hammer: dict[str, Any], arch: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    hammer_config = hammer["configuration"]
    arch_config = arch["configuration"]
    if hammer_config["dataset_sha256"] != arch_config["dataset_sha256"]:
        raise ValueError("Finalist reports use different dataset checksums")
    hammer_cases = hammer["case_results"]
    arch_cases = arch["case_results"]
    if len(hammer_cases) != len(arch_cases):
        raise ValueError("Finalist reports have different case counts")
    pairs = list(zip(hammer_cases, arch_cases, strict=True))
    for hammer_case, arch_case in pairs:
        identity = ("case_id", "request", "expected_tool", "expected_arguments")
        if any(hammer_case[key] != arch_case[key] for key in identity):
            raise ValueError(
                "Finalist case mismatch at " f"{hammer_case.get('case_id')}"
            )
        if (
            hammer_case["selected_domains"] != arch_case["selected_domains"]
            or hammer_case["offered_tools"] != arch_case["offered_tools"]
        ):
            raise ValueError(
                "Automatic selector changed between finalist runs at "
                f"{hammer_case['case_id']}"
            )
    return pairs


def build_paired_analysis(
    hammer: dict[str, Any], arch: dict[str, Any]
) -> dict[str, Any]:
    pairs = _verify_pair(hammer, arch)
    dimensions: dict[str, Any] = {}
    for dimension, correctness_key in DIMENSIONS.items():
        buckets: dict[str, list[str]] = {
            "both_correct": [],
            "hammer_only_correct": [],
            "arch_agent_only_correct": [],
            "both_wrong_same_way": [],
            "both_wrong_differently": [],
        }
        for hammer_case, arch_case in pairs:
            hammer_correct = bool(hammer_case[correctness_key])
            arch_correct = bool(arch_case[correctness_key])
            case_id = hammer_case["case_id"]
            if hammer_correct and arch_correct:
                bucket = "both_correct"
            elif hammer_correct:
                bucket = "hammer_only_correct"
            elif arch_correct:
                bucket = "arch_agent_only_correct"
            elif _outcome_signature(
                hammer_case, dimension
            ) == _outcome_signature(arch_case, dimension):
                bucket = "both_wrong_same_way"
            else:
                bucket = "both_wrong_differently"
            buckets[bucket].append(case_id)

        counts = {name: len(case_ids) for name, case_ids in buckets.items()}
        dimensions[dimension] = {
            "counts": counts,
            "case_ids": buckets,
            "mcnemar": exact_mcnemar(
                counts["hammer_only_correct"],
                counts["arch_agent_only_correct"],
            ),
        }

    return {
        "dataset_sha256": hammer["configuration"]["dataset_sha256"],
        "total_paired_cases": len(pairs),
        "hammer_model": hammer["configuration"]["model_id"],
        "arch_agent_model": arch["configuration"]["model_id"],
        "dimensions": dimensions,
    }


def _focus_confusion(
    matrix: dict[str, dict[str, int]], tools: tuple[str, ...]
) -> dict[str, Any]:
    entries = []
    errors = 0
    for expected in tools:
        for actual, count in matrix.get(expected, {}).items():
            if count and actual != expected:
                errors += count
                entries.append(
                    {"expected": expected, "actual": actual, "count": count}
                )
    return {"tools": list(tools), "error_count": errors, "errors": entries}


def build_tool_confusion(
    hammer: dict[str, Any], arch: dict[str, Any]
) -> dict[str, Any]:
    _verify_pair(hammer, arch)
    models = {}
    for label, report in (("hammer_3b", hammer), ("arch_agent_3b", arch)):
        matrix = report["dispatcher_confusion_matrix"]
        models[label] = {
            "model_id": report["configuration"]["model_id"],
            "full_matrix": matrix,
            "focus_clusters": {
                cluster: _focus_confusion(matrix, tools)
                for cluster, tools in FOCUS_CLUSTERS.items()
            },
        }
    return {
        "dataset_sha256": hammer["configuration"]["dataset_sha256"],
        "models": models,
    }


def _category_metrics(report: dict[str, Any]) -> dict[str, Any]:
    categories = sorted({case["tags"][1] for case in report["case_results"]})
    result = {}
    for category in categories:
        cases = [
            case for case in report["case_results"] if case["tags"][1] == category
        ]
        total = len(cases)
        result[category] = {
            "cases": total,
            "correct_tool_in_selected_subset_rate": sum(
                case["correct_tool_in_selected_subset"] for case in cases
            )
            / total,
            "tool_selection_accuracy": sum(case["tool_correct"] for case in cases)
            / total,
            "strict_exact_call_accuracy": sum(
                case["end_to_end_strict_exact_call"] for case in cases
            )
            / total,
            "execution_equivalent_exact_call_accuracy": sum(
                case["end_to_end_execution_equivalent_exact_call"]
                for case in cases
            )
            / total,
        }
    return result


def build_comparison(
    hammer: dict[str, Any], arch: dict[str, Any]
) -> dict[str, Any]:
    _verify_pair(hammer, arch)
    return {
        "experiment": "Yuki Phase 2 dispatcher reliability",
        "dataset": hammer["dataset"],
        "safety": {
            "yuki_tool_execution": False,
            "execution_capability": False,
            "external_service_calls_from_yuki_tools": False,
            "persistent_yuki_state_changes": False,
            "one_stateless_generation_per_case": True,
        },
        "models": {
            "hammer_3b": {
                "model_id": hammer["configuration"]["model_id"],
                "core_metrics": hammer["core_metrics"],
                "performance_metrics": hammer["performance_metrics"],
                "per_tool_results": hammer["per_tool_results"],
                "metrics_by_dataset_category": _category_metrics(hammer),
                "failure_taxonomy": hammer["failure_taxonomy"],
            },
            "arch_agent_3b": {
                "model_id": arch["configuration"]["model_id"],
                "core_metrics": arch["core_metrics"],
                "performance_metrics": arch["performance_metrics"],
                "per_tool_results": arch["per_tool_results"],
                "metrics_by_dataset_category": _category_metrics(arch),
                "failure_taxonomy": arch["failure_taxonomy"],
            },
        },
    }


def write_analysis_artifacts(
    hammer_path: str | Path,
    arch_path: str | Path,
    *,
    comparison_path: str | Path,
    paired_path: str | Path,
    confusion_path: str | Path,
) -> None:
    hammer = _load(hammer_path)
    arch = _load(arch_path)
    artifacts = (
        (comparison_path, build_comparison(hammer, arch)),
        (paired_path, build_paired_analysis(hammer, arch)),
        (confusion_path, build_tool_confusion(hammer, arch)),
    )
    for path, payload in artifacts:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
