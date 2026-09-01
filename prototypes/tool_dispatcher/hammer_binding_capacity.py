"""Execution-locked Hammer capacity arms for deterministic source binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .backends import create_backend
from .benchmark import write_report
from .delegation_dispatcher import (
    DelegationDispatcherRunner,
    ValidationOnlyDispatcher,
    failure_records,
)
from .focused_repair import (
    FocusedRepairRegistry,
    _decorate_focused_report,
    _read_checksum,
    _sha256,
    _variant_task_instruction,
    _write_checksum,
    _write_jsonl,
    verify_frozen_artifacts,
)

ROOT = Path(__file__).parent
FOCUSED_ROOT = ROOT / "reports" / "focused-repair"
REPORT_ROOT = ROOT / "reports" / "deterministic-binding-capacity"

PAYLOAD_75 = FOCUSED_ROOT / "focused-b-verbatim-old-frozen-delegations-v1.json"
SEE_25 = FOCUSED_ROOT / "focused-see-25-frozen-delegations-v1.json"
PRIMARY_100 = REPORT_ROOT / "primary100-frozen-qwen-delegations-v1.json"

FROZEN_SOURCE_SHA256 = {
    PAYLOAD_75: "2a4771e9adc5c84d94b704b01a320e147dce361b8d9b98eb4487733e2475901d",
    SEE_25: "e34db44b6175fb1b60af4b2dc4eb84ef8024cdcc186b537d00190feac083348c",
}

MODELS = {
    "hammer15": Path("/Volumes/madisk/yuki-tool-dispatcher/Hammer2.1-1.5b-fp16"),
    "hammer3": Path("/Volumes/madisk/yuki-tool-dispatcher/Hammer2.1-3b-fp16"),
}


def arm_paths(arm: str) -> dict[str, Path]:
    if arm not in MODELS:
        raise ValueError(f"Unknown Hammer arm: {arm}")
    return {
        "dispatch": REPORT_ROOT / f"{arm}-primary100-dispatch-v1.json",
        "failures": REPORT_ROOT / f"{arm}-primary100-dispatch-v1-failures.jsonl",
        "binder_input": REPORT_ROOT / f"{arm}-primary100-binder-input-v1.json",
        "binder_predictions": REPORT_ROOT
        / f"{arm}-primary100-binder-predictions-v1.json",
        "binder_ambiguities": REPORT_ROOT
        / f"{arm}-primary100-binder-ambiguities-v1.jsonl",
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_frozen_sources() -> dict[str, str]:
    verify_frozen_artifacts()
    observed = {str(path): _sha256(path) for path in FROZEN_SOURCE_SHA256}
    mismatches = {
        str(path): {"expected": expected, "actual": observed[str(path)]}
        for path, expected in FROZEN_SOURCE_SHA256.items()
        if observed[str(path)] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen capacity inputs changed: {mismatches}")
    return observed


def _verify_sidecar(path: Path) -> str:
    expected = _read_checksum(path)
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen checksum mismatch for {path}: {actual} != {expected}")
    return actual


def build_primary_subset(
    payload: dict[str, Any], see: dict[str, Any]
) -> dict[str, Any]:
    """Combine the exact frozen 75 payload and 25 see case objects."""

    if payload["contract"] != see["contract"]:
        raise RuntimeError("Frozen delegation contracts differ")
    stable_configuration = (
        "surrogate",
        "main_brain_role",
        "model_id",
        "source_dataset_sha256",
        "delegation_contract_version",
    )
    for field in stable_configuration:
        if payload["configuration"].get(field) != see["configuration"].get(field):
            raise RuntimeError(f"Frozen delegation configuration differs: {field}")

    cases = deepcopy(payload["case_results"] + see["case_results"])
    expected_ids = [
        f"p2-{tool}-{index:02d}"
        for tool in ("search", "image", "ask_claude", "see")
        for index in range(1, 26)
    ]
    observed_ids = [case["case_id"] for case in cases]
    if observed_ids != expected_ids:
        raise RuntimeError(f"Unexpected frozen primary IDs: {observed_ids}")

    configuration = deepcopy(payload["configuration"])
    configuration.update(
        {
            "experiment": "hammer_binding_capacity_frozen_stage_a_subset",
            "selection": "primary-targeted-100",
            "total_cases": 100,
            "tool_execution_capability": False,
        }
    )
    return {
        "configuration": configuration,
        "contract": deepcopy(payload["contract"]),
        "provenance": {
            "sources": {
                str(PAYLOAD_75): FROZEN_SOURCE_SHA256[PAYLOAD_75],
                str(SEE_25): FROZEN_SOURCE_SHA256[SEE_25],
            },
            "case_ids": observed_ids,
            "delegations_regenerated": False,
            "qwen_called": False,
            "gold_mutated": False,
        },
        "case_results": cases,
    }


def prepare() -> None:
    _verify_frozen_sources()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    subset = build_primary_subset(_load(PAYLOAD_75), _load(SEE_25))
    write_report(subset, PRIMARY_100)
    print(_write_checksum(PRIMARY_100))


def run_hammer(arm: str) -> None:
    _verify_frozen_sources()
    subset_sha = _verify_sidecar(PRIMARY_100)
    model = MODELS[arm]
    if not model.is_dir() or not str(model).startswith("/Volumes/"):
        raise RuntimeError(f"External Hammer checkpoint unavailable: {model}")

    registry = FocusedRepairRegistry("schema")
    backend = create_backend("mlx", str(model))
    dispatcher = ValidationOnlyDispatcher(
        registry,
        backend,
        input_mode="delegated_with_verbatim",
        domain_source="generated",
        task_instruction=_variant_task_instruction("schema"),
    )
    report = DelegationDispatcherRunner(dispatcher).run(
        _load(PRIMARY_100),
        delegation_path=PRIMARY_100,
        expected_delegation_sha256=subset_sha,
    )
    report = _decorate_focused_report(report, "schema", PRIMARY_100)
    report["configuration"].update(
        {
            "experiment": "hammer_binding_capacity_primary100",
            "capacity_arm": arm,
            "qwen_called": False,
            "delegations_regenerated": False,
            "official_gold_mutated": False,
            "yuki_tool_execution_attempts": 0,
        }
    )
    paths = arm_paths(arm)
    write_report(report, paths["dispatch"])
    _write_jsonl(failure_records(report), paths["failures"])
    print(_write_checksum(paths["dispatch"]))
    print(_write_checksum(paths["failures"]))


def _binder_case(case: dict[str, Any]) -> dict[str, Any]:
    dispatch = case["dispatcher"]
    return {
        "case_id": case["case_id"],
        "split": "primary_targeted_100",
        "raw_request": case["request"],
        "raw_request_sha256": hashlib.sha256(case["request"].encode()).hexdigest(),
        "semantic_delegation": deepcopy(case["delegation"]),
        "selected_call": deepcopy(dispatch["canonical_call"]),
        "hammer_record": {
            "model_id": dispatch["generation"]["model_id"],
            "raw_generation": dispatch["generation"]["raw_text"],
            "parse": deepcopy(dispatch["parse"]),
            "validation": deepcopy(dispatch["validation"]),
            "offered_tools": deepcopy(dispatch["offered_tools"]),
            "native_dialect": dispatch["native_dialect"],
            "stateless": dispatch["stateless"],
            "generation_count": dispatch["generation_count"],
            "execution_capability": dispatch["execution_capability"],
        },
    }


def prepare_binder_input(arm: str) -> None:
    paths = arm_paths(arm)
    subset_sha = _verify_sidecar(PRIMARY_100)
    dispatch_sha = _verify_sidecar(paths["dispatch"])
    report = _load(paths["dispatch"])
    cases = [_binder_case(case) for case in report["case_results"]]
    expected_ids = _load(PRIMARY_100)["provenance"]["case_ids"]
    if [case["case_id"] for case in cases] != expected_ids:
        raise RuntimeError(f"{arm} dispatch IDs differ from the frozen primary set")

    encoded = json.dumps(cases, ensure_ascii=False)
    forbidden = (
        '"expected_tool"',
        '"expected_arguments"',
        '"argument_alternatives"',
        '"strict_exact_call"',
        '"literal-source-annotations"',
    )
    leaked = [token for token in forbidden if token in encoded]
    if leaked:
        raise RuntimeError(f"Scoring data leaked into binder input: {leaked}")
    if any(case["hammer_record"]["execution_capability"] for case in cases):
        raise RuntimeError("Unexpected execution capability in Hammer records")

    payload = {
        "version": f"{arm}-primary100-binder-input-v1",
        "status": "gold_and_literal_annotation_free",
        "configuration": {
            "arm": arm,
            "case_count": 100,
            "models_called": False,
            "delegations_regenerated": False,
            "hammer_generations_regenerated": False,
            "tool_execution_capability": False,
        },
        "source_reports": {
            str(PRIMARY_100): subset_sha,
            str(paths["dispatch"]): dispatch_sha,
        },
        "splits": {"primary_targeted_100": expected_ids},
        "cases": cases,
    }
    write_report(payload, paths["binder_input"])
    print(_write_checksum(paths["binder_input"]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execution-locked Hammer plus source-binding capacity experiment"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    run = commands.add_parser("run-hammer")
    run.add_argument("--arm", choices=tuple(MODELS), required=True)
    bind = commands.add_parser("prepare-binder-input")
    bind.add_argument("--arm", choices=tuple(MODELS), required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run-hammer":
        run_hammer(args.arm)
    else:
        prepare_binder_input(args.arm)


if __name__ == "__main__":
    main()
