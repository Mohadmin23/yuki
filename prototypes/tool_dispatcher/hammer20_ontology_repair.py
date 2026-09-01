"""Prototype-only Hammer 2.0 schema-boundary regression experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from .backends import DispatcherBackend, MLXBackend
from .benchmark import BenchmarkRunner, load_cases, write_report
from .dispatcher import Dispatcher
from .hammer20_direct import (
    _assemble_report,
    _checkpoint_records,
    _group_metrics,
    _sha256,
    _write_checksum,
    _write_jsonl,
)
from .hammer20_frozen_binder import (
    BINDER_COMPARISON,
    BINDER_FREEZE,
    FROZEN_REPORT,
    FROZEN_SHA256,
    PHASE2_DATASET,
)
from .registry import ToolRegistry

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "hammer20-7b" / "binder-ontology-repair"
MODEL = Path("/Volumes/madisk/yuki-tool-dispatcher/Hammer2.0-7b-8bit")

TARGET_SET = REPORT_ROOT / "hammer20-7b-ontology-targeted49-v1.json"
TARGET_REPORT = REPORT_ROOT / "hammer20-7b-ontology-targeted49-repaired-v1.json"
TARGET_FAILURES = (
    REPORT_ROOT / "hammer20-7b-ontology-targeted49-repaired-v1-failures.jsonl"
)
TARGET_CHECKPOINT = (
    REPORT_ROOT / "hammer20-7b-ontology-targeted49-repaired-v1-checkpoint.jsonl"
)
TARGET_COMPARISON = (
    REPORT_ROOT / "HAMMER20-7B-ONTOLOGY-TARGETED49-COMPARISON-v1.json"
)

FULL_REPAIRED_REPORT = (
    REPORT_ROOT / "hammer20-7b-ontology-repaired-full450-v1.json"
)
FULL_REPAIRED_FAILURES = (
    REPORT_ROOT / "hammer20-7b-ontology-repaired-full450-v1-failures.jsonl"
)
FULL_REPAIRED_CHECKPOINT = (
    REPORT_ROOT / "hammer20-7b-ontology-repaired-full450-v1-checkpoint.jsonl"
)

ONTOLOGY_DESCRIPTION_REPAIRS = {
    "read": (
        "Read one file from the operating-system filesystem using its local path. "
        "Use only for normal OS paths such as /tmp/file, ~/Documents/file, ./file, "
        "or ../file. Do not use for files in Yuki's own internal managed file store."
    ),
    "yuki_read": (
        "Read one existing file from Yuki's own internal managed file store (yuki/). "
        "Use for Yuki's personal files, notes, scratch space, or saved internal files. "
        "Do not use for arbitrary operating-system paths."
    ),
    "yuki_write": (
        "Create a file in Yuki's own internal managed file store, or replace that "
        "file's entire contents. Use when the requested content should overwrite or "
        "start fresh. Do not use when existing contents must be preserved."
    ),
    "yuki_append": (
        "Preserve an existing file's contents in Yuki's own internal managed file "
        "store and add the supplied content after them. Use only for append/add/extend/"
        "continue intent. Do not use to create fresh contents or replace a file."
    ),
}

TARGET_FAMILIES = {
    "frozen_yuki_read_to_read_confusions": (
        "p2-yuki_read-02",
        "p2-yuki_read-04",
        "p2-yuki_read-05",
        "p2-yuki_read-11",
        "p2-yuki_read-16",
        "p2-yuki_read-18",
        "p2-yuki_read-24",
        "p2-yuki_read-25",
    ),
    "frozen_yuki_write_to_append_confusions": (
        "p2-yuki_write-02",
        "p2-yuki_write-05",
        "p2-yuki_write-06",
        "p2-yuki_write-09",
        "p2-yuki_write-12",
        "p2-yuki_write-22",
    ),
    "normal_read_controls": (
        "p2-read-18",
        "p2-read-21",
        "p2-read-22",
        "p2-read-23",
        "p2-read-24",
        "p2-read-25",
    ),
    "write_preservation_controls": (
        "p2-yuki_write-01",
        "p2-yuki_write-11",
        "p2-yuki_write-18",
        "p2-yuki_write-23",
        "p2-yuki_write-24",
        "p2-yuki_write-25",
    ),
    "append_preservation_controls": (
        "p2-yuki_append-01",
        "p2-yuki_append-05",
        "p2-yuki_append-18",
        "p2-yuki_append-22",
        "p2-yuki_append-23",
        "p2-yuki_append-24",
    ),
    "frozen_explicit_rejections": (
        "p2-time-07",
        "p2-search-23",
        "p2-search-24",
        "p2-see-06",
        "p2-read-15",
        "p2-read-24",
        "p2-read-25",
        "p2-yuki_read-15",
        "p2-yuki_read-22",
        "p2-yuki_list-16",
        "p2-yuki_list-18",
        "p2-yuki_list-20",
        "p2-yuki_list-24",
        "p2-yuki_delete-13",
        "p2-yuki_delete-17",
        "p2-yuki_append-23",
        "p2-remember-19",
        "p2-remember-21",
        "p2-remember-24",
        "p2-ask_claude-24",
    ),
}


class OntologyRepairRegistry(ToolRegistry):
    """Live Yuki registry with four prototype-facing descriptions replaced."""

    def hammer_schemas(self, names: Any) -> list[dict[str, Any]]:
        schemas = super().hammer_schemas(names)
        for schema in schemas:
            replacement = ONTOLOGY_DESCRIPTION_REPAIRS.get(schema["name"])
            if replacement is not None:
                schema["description"] = replacement
        return schemas


def _verify_prerequisites() -> dict[str, str]:
    expected = {
        FROZEN_REPORT: FROZEN_SHA256[FROZEN_REPORT],
        PHASE2_DATASET: FROZEN_SHA256[PHASE2_DATASET],
    }
    observed = {str(path): _sha256(path) for path in expected}
    mismatches = {
        str(path): {"expected": checksum, "actual": observed[str(path)]}
        for path, checksum in expected.items()
        if observed[str(path)] != checksum
    }
    for path in (BINDER_FREEZE, BINDER_COMPARISON):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        sidecar_sha = sidecar.read_text(encoding="utf-8").split()[0]
        actual = _sha256(path)
        if actual != sidecar_sha:
            mismatches[str(path)] = {"expected": sidecar_sha, "actual": actual}
        observed[str(path)] = actual
    if mismatches:
        raise RuntimeError(f"Ontology repair prerequisites changed: {mismatches}")
    return observed


def _target_ids() -> list[str]:
    requested = {case_id for ids in TARGET_FAMILIES.values() for case_id in ids}
    dataset_order = [case.case_id for case in load_cases(PHASE2_DATASET)]
    result = [case_id for case_id in dataset_order if case_id in requested]
    if len(result) != 49 or len(set(result)) != 49:
        raise RuntimeError(f"Target regression set must contain 49 unique cases: {len(result)}")
    return result


def _case_families(case_id: str) -> list[str]:
    return [name for name, ids in TARGET_FAMILIES.items() if case_id in ids]


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(cases),
        "tool_selection_accuracy": mean(case["tool_correct"] for case in cases),
        "strict_argument_accuracy": mean(
            case["strict_arguments_correct"] for case in cases
        ),
        "strict_exact_accuracy": mean(case["exact_call"] for case in cases),
        "execution_equivalent_exact_accuracy": mean(
            case["execution_equivalent_call"] for case in cases
        ),
        "schema_valid_rate": mean(case["schema_valid"] for case in cases),
        "rejection_count": sum(case["parse"]["rejected"] for case in cases),
        "malformed_count": sum(case["malformed"] for case in cases),
    }


def prepare_targeted() -> None:
    observed = _verify_prerequisites()
    ids = _target_ids()
    frozen = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    baseline_by_id = {case["case_id"]: case for case in frozen["case_results"]}
    missing = sorted(set(ids) - baseline_by_id.keys())
    if missing:
        raise RuntimeError(f"Target IDs missing from frozen Hammer run: {missing}")
    baseline_cases = [deepcopy(baseline_by_id[case_id]) for case_id in ids]
    payload = {
        "version": "hammer20-7b-ontology-targeted49-v1",
        "status": "frozen_before_repaired_inference",
        "case_count": len(ids),
        "case_ids": ids,
        "families": {name: list(family_ids) for name, family_ids in TARGET_FAMILIES.items()},
        "case_family_membership": {
            case_id: _case_families(case_id) for case_id in ids
        },
        "source_checksums": observed,
        "schema_repairs": ONTOLOGY_DESCRIPTION_REPAIRS,
        "change_scope": {
            "tool_descriptions_changed": sorted(ONTOLOGY_DESCRIPTION_REPAIRS),
            "parameter_names_changed": False,
            "parameter_descriptions_changed": False,
            "schema_structure_changed": False,
            "task_instruction_changed": False,
            "format_instruction_changed": False,
            "model_changed": False,
        },
        "full_rerun_gate": {
            "targeted_tool_selection_must_increase": True,
            "targeted_strict_exact_must_increase": True,
            "combined_primary_confusions_must_drop_by_at_least": 7,
            "explicit_rejections_must_not_increase": True,
            "new_strict_regressions_allowed": 1,
        },
        "frozen_baseline_summary": _summary(baseline_cases),
        "frozen_baseline_cases": baseline_cases,
        "safety": {
            "models_called": False,
            "hammer_loaded": False,
            "new_sampling": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
    }
    write_report(payload, TARGET_SET)
    print(_write_checksum(TARGET_SET))


def _schema_sha(registry: ToolRegistry) -> str:
    encoded = json.dumps(
        registry.hammer_schemas(registry.names),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_target_cases(target_ids: list[str]) -> list[Any]:
    by_id = {case.case_id: case for case in load_cases(PHASE2_DATASET)}
    return [by_id[case_id] for case_id in target_ids]


def run_cases(
    backend: DispatcherBackend,
    *,
    target_ids: list[str],
    output: Path,
    failures_output: Path,
    checkpoint: Path,
    experiment: str,
    target_set_path: Path = TARGET_SET,
) -> dict[str, Any]:
    registry = OntologyRepairRegistry()
    schema_sha = _schema_sha(registry)
    target_set_sha = _sha256(target_set_path)
    cases = _selected_target_cases(target_ids)
    order = {case.case_id: index for index, case in enumerate(cases)}
    completed: dict[str, dict[str, Any]] = {}
    for record in _checkpoint_records(checkpoint):
        if record.get("version") != 1:
            raise RuntimeError("Unsupported ontology checkpoint version")
        if record.get("target_set_sha256") != target_set_sha:
            raise RuntimeError("Ontology checkpoint target-set mismatch")
        if record.get("schema_sha256") != schema_sha:
            raise RuntimeError("Ontology checkpoint schema mismatch")
        if record.get("model_id") != backend.model_id:
            raise RuntimeError("Ontology checkpoint model mismatch")
        case_id = record["case_result"]["case_id"]
        if case_id not in order or case_id in completed:
            raise RuntimeError(f"Invalid or duplicate ontology checkpoint case: {case_id}")
        completed[case_id] = record

    pending = [case for case in cases if case.case_id not in completed]

    def save_case(
        case_result: dict[str, Any],
        fine_tuning_example: dict[str, Any] | None,
    ) -> None:
        case_result["index"] = order[case_result["case_id"]]
        record = {
            "version": 1,
            "target_set_sha256": target_set_sha,
            "schema_sha256": schema_sha,
            "model_id": backend.model_id,
            "case_result": case_result,
            "fine_tuning_example": fine_tuning_example,
        }
        completed[case_result["case_id"]] = record
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if pending:
        BenchmarkRunner(Dispatcher(registry, backend)).run(
            pending,
            group="all",
            router_mode="model_only",
            output_mode="native",
            execute=False,
            progress_label=f"Hammer 2.0 7B ontology repair ({len(completed)}/{len(cases)})",
            case_callback=save_case,
        )

    records = [completed[case.case_id] for case in cases]
    case_results = [record["case_result"] for record in records]
    fine_tuning = [
        record["fine_tuning_example"]
        for record in records
        if record.get("fine_tuning_example") is not None
    ]
    report = _assemble_report(
        backend=backend,
        registry=registry,
        selection=experiment,
        schema_sha256=schema_sha,
        case_results=case_results,
        fine_tuning_examples=fine_tuning,
    )
    report["configuration"].update(
        {
            "experiment": experiment,
            "selection": experiment,
            "schema_profile": "prototype_only_ontology_descriptions_v1",
            "target_set_sha256": target_set_sha,
            "task_instruction_changed": False,
            "format_instruction_changed": False,
            "schema_structure_changed": False,
            "model_weights_changed": False,
            "execution_capability": False,
            "yuki_tool_execution_attempts": 0,
            "qwen_used": False,
            "binder_used_during_inference": False,
        }
    )
    report["by_expected_tool"] = _group_metrics(case_results, "expected_tool")
    write_report(report, output)
    _write_jsonl(fine_tuning, failures_output)
    _write_checksum(output)
    _write_checksum(failures_output)
    _write_checksum(checkpoint)
    return report


def run_targeted() -> None:
    _verify_prerequisites()
    target_set_sha = TARGET_SET.with_suffix(TARGET_SET.suffix + ".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    if _sha256(TARGET_SET) != target_set_sha:
        raise RuntimeError("Targeted-set checksum mismatch")
    if not MODEL.is_dir() or not str(MODEL).startswith("/Volumes/"):
        raise RuntimeError(f"External Hammer model is unavailable: {MODEL}")
    target = json.loads(TARGET_SET.read_text(encoding="utf-8"))
    backend = MLXBackend(str(MODEL), static_prefix_marker="[BEGIN OF QUERY]\n")
    run_cases(
        backend,
        target_ids=target["case_ids"],
        output=TARGET_REPORT,
        failures_output=TARGET_FAILURES,
        checkpoint=TARGET_CHECKPOINT,
        experiment="ontology-targeted49-v1",
    )


def _confusion_count(cases: list[dict[str, Any]]) -> int:
    return sum(
        (case["expected_tool"] == "yuki_read" and case["actual_tool"] == "read")
        or (
            case["expected_tool"] == "yuki_write"
            and case["actual_tool"] == "yuki_append"
        )
        for case in cases
    )


def compare_targeted() -> dict[str, Any]:
    _verify_prerequisites()
    target_sha = TARGET_SET.with_suffix(TARGET_SET.suffix + ".sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    repaired_sha = TARGET_REPORT.with_suffix(
        TARGET_REPORT.suffix + ".sha256"
    ).read_text(encoding="utf-8").split()[0]
    if _sha256(TARGET_SET) != target_sha or _sha256(TARGET_REPORT) != repaired_sha:
        raise RuntimeError("Targeted ontology comparison input checksum mismatch")
    target = json.loads(TARGET_SET.read_text(encoding="utf-8"))
    repaired = json.loads(TARGET_REPORT.read_text(encoding="utf-8"))
    baseline_cases = target["frozen_baseline_cases"]
    repaired_cases = repaired["case_results"]
    baseline_by_id = {case["case_id"]: case for case in baseline_cases}
    repaired_by_id = {case["case_id"]: case for case in repaired_cases}
    if baseline_by_id.keys() != repaired_by_id.keys():
        raise RuntimeError("Targeted baseline and repair IDs differ")

    flips = []
    for case_id in target["case_ids"]:
        before = baseline_by_id[case_id]
        after = repaired_by_id[case_id]
        if (
            before["tool_correct"] != after["tool_correct"]
            or before["exact_call"] != after["exact_call"]
            or before["parse"]["rejected"] != after["parse"]["rejected"]
        ):
            flips.append(
                {
                    "case_id": case_id,
                    "families": target["case_family_membership"][case_id],
                    "expected_tool": before["expected_tool"],
                    "baseline_tool": before["actual_tool"],
                    "repaired_tool": after["actual_tool"],
                    "baseline_exact": before["exact_call"],
                    "repaired_exact": after["exact_call"],
                    "baseline_rejected": before["parse"]["rejected"],
                    "repaired_rejected": after["parse"]["rejected"],
                }
            )

    family_metrics = {}
    for family, ids in target["families"].items():
        family_set = set(ids)
        family_metrics[family] = {
            "baseline": _summary(
                [case for case in baseline_cases if case["case_id"] in family_set]
            ),
            "repaired": _summary(
                [case for case in repaired_cases if case["case_id"] in family_set]
            ),
        }
    baseline_summary = _summary(baseline_cases)
    repaired_summary = _summary(repaired_cases)
    baseline_confusions = _confusion_count(baseline_cases)
    repaired_confusions = _confusion_count(repaired_cases)
    new_regressions = [
        item for item in flips if item["baseline_exact"] and not item["repaired_exact"]
    ]
    gate = target["full_rerun_gate"]
    gate_checks = {
        "targeted_tool_selection_increased": repaired_summary[
            "tool_selection_accuracy"
        ]
        > baseline_summary["tool_selection_accuracy"],
        "targeted_strict_exact_increased": repaired_summary["strict_exact_accuracy"]
        > baseline_summary["strict_exact_accuracy"],
        "primary_confusions_dropped_enough": (
            baseline_confusions - repaired_confusions
            >= gate["combined_primary_confusions_must_drop_by_at_least"]
        ),
        "explicit_rejections_did_not_increase": repaired_summary["rejection_count"]
        <= baseline_summary["rejection_count"],
        "new_strict_regressions_within_limit": len(new_regressions)
        <= gate["new_strict_regressions_allowed"],
    }
    gate_passed = all(gate_checks.values())
    comparison = {
        "version": "hammer20-7b-ontology-targeted49-comparison-v1",
        "status": "completed",
        "source_checksums": {
            str(TARGET_SET): target_sha,
            str(TARGET_REPORT): repaired_sha,
        },
        "safety": {
            "qwen_called": False,
            "gold_mutated": False,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "baseline": baseline_summary,
        "repaired": repaired_summary,
        "baseline_primary_confusion_count": baseline_confusions,
        "repaired_primary_confusion_count": repaired_confusions,
        "family_metrics": family_metrics,
        "case_flips": flips,
        "new_strict_regressions": new_regressions,
        "full_450_gate": {
            "criteria": gate,
            "checks": gate_checks,
            "passed": gate_passed,
        },
    }
    write_report(comparison, TARGET_COMPARISON)
    _write_checksum(TARGET_COMPARISON)
    return comparison


def run_full() -> None:
    comparison = compare_targeted()
    if not comparison["full_450_gate"]["passed"]:
        raise RuntimeError("Targeted ontology gate did not pass; full 450 run is blocked")
    if not MODEL.is_dir() or not str(MODEL).startswith("/Volumes/"):
        raise RuntimeError(f"External Hammer model is unavailable: {MODEL}")
    backend = MLXBackend(str(MODEL), static_prefix_marker="[BEGIN OF QUERY]\n")
    run_cases(
        backend,
        target_ids=[case.case_id for case in load_cases(PHASE2_DATASET)],
        output=FULL_REPAIRED_REPORT,
        failures_output=FULL_REPAIRED_FAILURES,
        checkpoint=FULL_REPAIRED_CHECKPOINT,
        experiment="ontology-repaired-full450-v1",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hammer 2.0 7B prototype-only ontology repair"
    )
    parser.add_argument(
        "command", choices=("prepare-targeted", "run-targeted", "compare-targeted", "run-full")
    )
    return parser


def main() -> None:
    command = _parser().parse_args().command
    if command == "prepare-targeted":
        prepare_targeted()
    elif command == "run-targeted":
        run_targeted()
    elif command == "compare-targeted":
        comparison = compare_targeted()
        print(json.dumps(comparison["full_450_gate"], indent=2))
    else:
        run_full()


if __name__ == "__main__":
    main()
