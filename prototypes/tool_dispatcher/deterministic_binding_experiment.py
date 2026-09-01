"""Run the pure binder over a frozen gold-free input file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .benchmark import write_report
from .source_span_policy import bind_source_argument

ROOT = Path(__file__).parent
REPORT_ROOT = ROOT / "reports" / "deterministic-binding"
INPUT = REPORT_ROOT / "deterministic-binding-input-targeted100-recall25-v1.json"
OUTPUT = REPORT_ROOT / "deterministic-binding-targeted100-recall25-v1.json"
AMBIGUITIES = REPORT_ROOT / "deterministic-binding-ambiguities-v1.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_checksum(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    return sidecar.read_text(encoding="utf-8").split()[0]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checksum(path: Path) -> str:
    checksum = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n", encoding="utf-8"
    )
    return checksum


def run(
    input_path: Path = INPUT,
    output_path: Path = OUTPUT,
    ambiguities_path: Path = AMBIGUITIES,
    *,
    version: str = "deterministic-binding-targeted100-recall25-v1",
) -> None:
    expected = _read_checksum(input_path)
    actual = _sha256(input_path)
    if actual != expected:
        raise RuntimeError(f"Gold-free binder input changed: {actual} != {expected}")
    payload = _load(input_path)
    if payload["status"] != "gold_and_literal_annotation_free":
        raise RuntimeError("Binder input has not passed the isolation gate")

    results = []
    ambiguities = []
    for case in payload["cases"]:
        delegation = case["semantic_delegation"]
        binding = bind_source_argument(
            raw_request=case["raw_request"],
            selected_call=case["selected_call"],
            semantic_request=delegation["request"],
            verbatim=delegation["verbatim"],
        )
        result = {
            "case_id": case["case_id"],
            "split": case["split"],
            "raw_request": case["raw_request"],
            "raw_request_sha256": case["raw_request_sha256"],
            "semantic_delegation": delegation,
            "baseline_selected_call": case["selected_call"],
            "binding": binding,
            "execution": {
                "capability": False,
                "executed": False,
                "status": "not_available",
            },
        }
        results.append(result)
        if binding["status"] in {"ambiguous", "unbound"}:
            ambiguities.append(result)

    report = {
        "version": version,
        "status": "predictions_frozen_before_scoring",
        "configuration": {
            "input_sha256": actual,
            "case_count": len(results),
            "models_called": False,
            "qwen_called": False,
            "hammer_loaded": False,
            "delegations_regenerated": False,
            "hammer_generations_regenerated": False,
            "stateless": True,
            "tool_execution_capability": False,
            "tool_execution_attempts": 0,
        },
        "splits": payload["splits"],
        "status_counts": {
            status: sum(case["binding"]["status"] == status for case in results)
            for status in sorted({case["binding"]["status"] for case in results})
        },
        "case_results": results,
    }
    write_report(report, output_path)
    ambiguities_path.parent.mkdir(parents=True, exist_ok=True)
    ambiguities_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ambiguities),
        encoding="utf-8",
    )
    print(_write_checksum(output_path))
    print(_write_checksum(ambiguities_path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated source-span binder")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ambiguities-output", type=Path, default=AMBIGUITIES)
    parser.add_argument(
        "--version", default="deterministic-binding-targeted100-recall25-v1"
    )
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    run(
        args.input,
        args.output,
        args.ambiguities_output,
        version=args.version,
    )
