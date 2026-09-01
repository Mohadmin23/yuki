"""Deterministic near-duplicate checks against the immutable Phase 2 set."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..argument_contract import LITERAL_SOURCE, tool_argument_contract
from .common import (
    LEAKAGE_VERSION,
    PILOT_LEAKAGE,
    PILOT_PATH,
    SACRED_DATASET,
    load_jsonl,
    verify_sacred_dataset,
    write_json,
)

TOKEN_RE = re.compile(r"[\w@./:+#~-]+", flags=re.UNICODE)

REJECT_THRESHOLDS = {
    "character_similarity": 0.88,
    "token_jaccard": 0.72,
    "trigram_jaccard": 0.72,
    "literal_payload_similarity": 0.92,
}
REVIEW_THRESHOLDS = {
    "character_similarity": 0.80,
    "token_jaccard": 0.58,
    "trigram_jaccard": 0.58,
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(TOKEN_RE.findall(value))


def _tokens(value: str) -> set[str]:
    return set(normalize_text(value).split())


def _trigrams(value: str) -> set[str]:
    normalized = normalize_text(value)
    return {
        normalized[index : index + 3]
        for index in range(max(0, len(normalized) - 2))
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _similarity(left: str, right: str) -> dict[str, float | bool]:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    return {
        "normalized_exact": normalized_left == normalized_right,
        "character_similarity": SequenceMatcher(
            None, normalized_left, normalized_right
        ).ratio(),
        "token_jaccard": _jaccard(_tokens(left), _tokens(right)),
        "trigram_jaccard": _jaccard(_trigrams(left), _trigrams(right)),
    }


def _literal_payload(record: dict[str, Any]) -> str | None:
    metadata = record["metadata"]
    tool = metadata["expected_tool"]
    if tool is None or tool_argument_contract(tool)["mode"] != LITERAL_SOURCE:
        return None
    field = tool_argument_contract(tool)["argument"]
    value = metadata["expected_arguments"].get(field)
    if value is None:
        value = metadata.get("runtime_arguments", {}).get(field)
    return value if isinstance(value, str) else None


def check_record(
    record: dict[str, Any],
    sacred_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = record["metadata"]
    request = metadata["raw_user_request"]
    expected_tool = metadata["expected_tool"]
    candidate_payload = _literal_payload(record)
    best_case: dict[str, Any] | None = None
    best_signals: dict[str, Any] | None = None
    best_rank = -1.0
    best_payload_similarity = 0.0
    best_payload_exact = False

    for sacred in sacred_cases:
        signals = _similarity(request, sacred["request"])
        sacred_payload = None
        if candidate_payload is not None and sacred["expected_tool"] == expected_tool:
            values = list(sacred.get("expected_arguments", {}).values())
            if len(values) == 1 and isinstance(values[0], str):
                sacred_payload = values[0]
        payload_similarity = (
            SequenceMatcher(
                None,
                normalize_text(candidate_payload),
                normalize_text(sacred_payload),
            ).ratio()
            if candidate_payload and sacred_payload
            else 0.0
        )
        payload_exact = bool(
            candidate_payload
            and sacred_payload
            and normalize_text(candidate_payload) == normalize_text(sacred_payload)
        )
        rank = max(
            float(signals["character_similarity"]),
            float(signals["token_jaccard"]),
            float(signals["trigram_jaccard"]),
            payload_similarity,
        )
        if rank > best_rank:
            best_rank = rank
            best_case = sacred
            best_signals = signals
            best_payload_similarity = payload_similarity
            best_payload_exact = payload_exact

    assert best_signals is not None and best_case is not None
    flags: list[str] = []
    status = "pass"
    if best_signals["normalized_exact"]:
        flags.append("normalized_exact_request")
        status = "reject"
    if (
        best_signals["character_similarity"]
        >= REJECT_THRESHOLDS["character_similarity"]
        and best_signals["token_jaccard"] >= REJECT_THRESHOLDS["token_jaccard"]
    ):
        flags.append("high_character_and_token_similarity")
        status = "reject"
    if (
        best_signals["trigram_jaccard"] >= REJECT_THRESHOLDS["trigram_jaccard"]
        and best_signals["token_jaccard"] >= 0.60
    ):
        flags.append("high_ngram_overlap")
        status = "reject"
    if best_payload_exact and candidate_payload and len(normalize_text(candidate_payload)) >= 5:
        flags.append("same_tool_literal_payload_exact")
        status = "reject"
    elif (
        best_payload_similarity
        >= REJECT_THRESHOLDS["literal_payload_similarity"]
        and candidate_payload
        and len(normalize_text(candidate_payload)) >= 8
    ):
        flags.append("same_tool_literal_payload_near_match")
        status = "reject"

    if status == "pass" and (
        (
            best_signals["character_similarity"]
            >= REVIEW_THRESHOLDS["character_similarity"]
            and best_signals["token_jaccard"]
            >= REVIEW_THRESHOLDS["token_jaccard"]
        )
        or (
            best_signals["trigram_jaccard"]
            >= REVIEW_THRESHOLDS["trigram_jaccard"]
            and best_signals["token_jaccard"] >= 0.50
        )
    ):
        flags.append("moderate_near_duplicate_signal")
        status = "review"

    return {
        "checker_version": LEAKAGE_VERSION,
        "status": status,
        "flags": flags,
        "closest_case_id": best_case.get("case_id"),
        "signals": {
            **best_signals,
            "same_tool_literal_payload_match": best_payload_exact,
            "same_tool_literal_payload_similarity": round(
                best_payload_similarity, 6
            ),
        },
    }


def check_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    sacred_sha = verify_sacred_dataset()
    sacred_cases = json.loads(SACRED_DATASET.read_text(encoding="utf-8"))
    results = {
        record["metadata"]["record_id"]: check_record(record, sacred_cases)
        for record in records
    }
    counts = {
        status: sum(result["status"] == status for result in results.values())
        for status in ("pass", "review", "reject")
    }
    return {
        "version": LEAKAGE_VERSION,
        "sacred_dataset": str(SACRED_DATASET),
        "sacred_dataset_sha256": sacred_sha,
        "thresholds": {
            "reject": REJECT_THRESHOLDS,
            "review": REVIEW_THRESHOLDS,
        },
        "counts": counts,
        "flagged": {
            record_id: result
            for record_id, result in results.items()
            if result["status"] != "pass"
        },
        "records": results,
        "limitations": [
            "Lexical checks cannot prove semantic independence.",
            "Short generic requests have unstable similarity scores.",
            "Semantic fields necessarily reuse small canonical enums or concepts.",
            "The checker flags suspicious overlap but does not rewrite candidates.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Hammer pilot leakage")
    parser.add_argument("--input", type=Path, default=PILOT_PATH)
    parser.add_argument("--output", type=Path, default=PILOT_LEAKAGE)
    args = parser.parse_args()
    report = check_records(load_jsonl(args.input))
    write_json(args.output, report)
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
