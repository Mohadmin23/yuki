"""Build the cleaned, deterministic Hammer 1.5B Pilot v1.1 dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..argument_contract import CONTRACT_VERSION, tool_argument_contract
from .build_hammer15_pilot import Candidate, candidate_catalog
from .check_sacred450_leakage import check_record, check_records
from .common import (
    PILOT_PATH,
    RECORD_VERSION,
    SACRED_DATASET,
    load_jsonl,
    sha256_path,
    write_json,
    write_jsonl,
)
from .common_v11 import (
    COMPOSITE_TOOLS,
    DEFAULT_SEED_V11,
    GENERATOR_VERSION_V11,
    PILOT_LEAKAGE_V11,
    PILOT_MANIFEST_V11,
    PILOT_PATH_V11,
    PILOT_REJECTED_V11,
    PILOT_VALIDATION_V11,
    PROMPT_PROFILE_V11,
    RECORD_VERSION_V11,
    STRUCTURED_ARGUMENT_PROFILE,
    VALIDATOR_VERSION_V11,
    Hammer15TrainingRegistryV11,
    prompt_schema_sha256_v11,
    render_model_visible_v11,
    verify_v11_safety_constants,
)
from .validate_hammer15_dataset import EXPECTED_CATEGORY_COUNTS
from .validate_hammer15_dataset_v11 import (
    record_errors_v11,
    validate_records_v11,
)

FeatureSpec = tuple[str, tuple[str, ...]]

# The ontology group is deliberately hand-authored. Values in braces are copied
# from each v1 source record, so literal annotations remain independently auditable.
DIVERSE_REQUESTS: dict[str, FeatureSpec] = {
    "onto-read-1": ("disk file, exact path: {value} — open that", ("terse_fragment", "punctuation_variation")),
    "onto-read-2": ("Need {value}; regular filesystem, not your notes.", ("terse_fragment", "negative_contrast")),
    "onto-read-3": ("{value} ... read frm the Mac pls", ("typo", "casual", "punctuation_variation")),
    "onto-read-4": ("Correction: use the OS reader on {value}.", ("correction", "implicit_intent")),
    "onto-read-5": ("What's inside {value}? That's on disk.", ("contraction", "clause_reordering")),
    "onto-yread-1": ("that one you keep, {value} — open it", ("pronoun_context", "casual", "punctuation_variation")),
    "onto-yread-2": ("{value}, frm ur own stash. read pls", ("typo", "shorthand", "terse_fragment")),
    "onto-yread-3": ("not the disk copy—the saved {value}", ("negative_contrast", "terse_fragment", "punctuation_variation")),
    "onto-yread-4": ("Yuki store -> {value}; contents?", ("shorthand", "terse_fragment", "punctuation_variation")),
    "onto-yread-5": ("u kept {value} somewhere; pull it up", ("typo", "pronoun_context", "multi_clause")),
    "onto-ywrite-1": ("start {filename} over. new contents: {content}", ("terse_fragment", "correction")),
    "onto-ywrite-2": ("For {filename}, discard the old text and put {content}", ("clause_reordering", "negative_contrast")),
    "onto-ywrite-3": ("{filename} should contain only this now — {content}", ("implicit_intent", "punctuation_variation")),
    "onto-ywrite-4": ("rewrite ur {filename}: {content}", ("typo", "shorthand", "terse_fragment")),
    "onto-ywrite-5": ("No appending this time; replace {filename} with {content}", ("negative_contrast", "multi_clause")),
    "onto-yappend-1": ("don't wipe {filename}; tack this on: {content}", ("contraction", "negative_contrast", "casual")),
    "onto-yappend-2": ("{filename} already has rows—add {content} after them", ("pronoun_context", "implicit_intent")),
    "onto-yappend-3": ("keep what's in {filename}. then add: {content}", ("contraction", "multi_clause", "clause_reordering")),
    "onto-yappend-4": ("{filename} +={content}", ("shorthand", "terse_fragment", "punctuation_variation")),
    "onto-yappend-5": ("one more line for {filename}, dont replace it: {content}", ("typo", "negative_contrast", "multi_clause")),
    "onto-see-1": ("camera check—where's my {value}?", ("contraction", "terse_fragment", "punctuation_variation")),
    "onto-see-2": ("can u spot the {value} in what you're seeing", ("typo", "casual", "contraction")),
    "onto-see-3": ("not a new picture; inspect the current view for {value}", ("negative_contrast", "multi_clause")),
    "onto-see-4": ("{value}—is it visible on the live feed?", ("clause_reordering", "punctuation_variation")),
    "onto-see-5": ("look around it, the {value}; tell me what's there", ("pronoun_context", "multi_clause", "distractor_noun")),
    "onto-image-1": ("make a new image: {value}", ("terse_fragment", "punctuation_variation")),
    "onto-image-2": ("{value} — generate that, don't inspect anything", ("clause_reordering", "negative_contrast")),
    "onto-image-3": ("need artwork of {value}", ("implicit_intent", "terse_fragment")),
    "onto-image-4": ("pls render: {value}", ("shorthand", "casual", "terse_fragment")),
    "onto-image-5": ("Turn this description into a fresh picture—{value}", ("implicit_intent", "punctuation_variation")),
    "onto-search-1": ("web lookup -> {value}", ("shorthand", "terse_fragment", "punctuation_variation")),
    "onto-search-2": ("find this online pls: {value}", ("typo", "casual")),
    "onto-search-3": ("I don't have this stored; search for {value}", ("negative_contrast", "multi_clause")),
    "onto-search-4": ("{value} — current web results, not memory", ("clause_reordering", "negative_contrast")),
    "onto-search-5": ("Could you look up {value}? The cable drawer can wait.", ("distractor_noun", "multi_clause")),
    "onto-recall-1": ("did i ever tell u about {value}?", ("typo", "casual", "pronoun_context")),
    "onto-recall-2": ("{value}—pull up what you already saved", ("clause_reordering", "implicit_intent")),
    "onto-recall-3": ("not a web search. your memory on {value}", ("negative_contrast", "terse_fragment")),
    "onto-recall-4": ("that old {value} thing... what'd I say?", ("pronoun_context", "contraction", "punctuation_variation")),
    "onto-recall-5": ("memory chk: {value}", ("typo", "shorthand", "terse_fragment")),
    "onto-remember-1": ("tea thing—keep this about me: I take it without sugar", ("pronoun_context", "implicit_intent", "punctuation_variation")),
    "onto-remember-2": ("remember, my spare adapter lives in drawer four", ("casual", "punctuation_variation")),
    "onto-remember-3": ("for later: I prefer subtitles on", ("terse_fragment", "implicit_intent")),
    "onto-remember-4": ("Mica—that's my bicycle's name. save the fact", ("clause_reordering", "multi_clause", "punctuation_variation")),
    "onto-remember-5": ("dont forget this abt me: no meetings before ten", ("typo", "shorthand", "contraction")),
    "onto-ywrite-memory-1": ("This is a file, not memory: set {filename} to {content}", ("negative_contrast", "multi_clause")),
    "onto-ywrite-memory-2": ("{filename} <- {content}; replace the stored file", ("shorthand", "clause_reordering", "punctuation_variation")),
    "onto-ywrite-memory-3": ("save literal config text in {filename}: {content}", ("distractor_noun", "implicit_intent")),
    "onto-ywrite-memory-4": ("rewrite {filename}, contents {content}", ("terse_fragment", "punctuation_variation")),
    "onto-ywrite-memory-5": ("not a fact about me—make {filename} contain {content}", ("negative_contrast", "multi_clause")),
    "onto-shell-1": ("terminal, exact: {value}", ("terse_fragment", "punctuation_variation")),
    "onto-shell-2": ("run `{value}`; I want the command output", ("multi_clause", "distractor_noun")),
    "onto-shell-3": ("cmd pls -> {value}", ("typo", "shorthand", "terse_fragment")),
    "onto-shell-4": ("not telemetry—execute {value}", ("negative_contrast", "terse_fragment")),
    "onto-shell-5": ("{value}, exactly; don't summarize hardware", ("clause_reordering", "negative_contrast")),
    "onto-hardware-1": ("RAM looking okay rn? use system stats", ("shorthand", "casual", "implicit_intent")),
    "onto-hardware-2": ("GPU status—not a shell command", ("negative_contrast", "terse_fragment")),
    "onto-hardware-3": ("how much CPU and RAM is your own LLM process using?", ("pronoun_context", "implicit_intent")),
    "onto-hardware-4": ("disk useage pls, from hardware stats", ("typo", "casual", "terse_fragment")),
    "onto-hardware-5": ("Before the benchmark, how busy is the CPU?", ("distractor_noun", "clause_reordering")),
    "easy-time-1": ("local clock, rn??", ("shorthand", "terse_fragment", "punctuation_variation")),
    "easy-ylist-2": ("ur file shelf—list it", ("typo", "pronoun_context", "terse_fragment")),
    "easy-weather": ("Oslo weather rn?", ("shorthand", "terse_fragment")),
    "easy-calc": ("9 + 6, what's that?", ("contraction", "terse_fragment")),
    "easy-hardware": ("RAM stats pls", ("shorthand", "terse_fragment")),
    "easy-see": ("green notebook—can u see it?", ("typo", "pronoun_context", "punctuation_variation")),
    "easy-recall": ("camping stove thing, what'd I say?", ("contraction", "pronoun_context", "casual")),
    "edge-reject-2": ("read that file... uh, path's missing", ("contraction", "pronoun_context", "punctuation_variation")),
    "edge-reject-3": ("search for... idk, nothing specific", ("shorthand", "casual", "punctuation_variation")),
    "edge-reject-6": ("delete the note—name? forgot it", ("terse_fragment", "pronoun_context", "punctuation_variation")),
}


def _template_values(candidate: Candidate) -> dict[str, str]:
    if candidate.literal_source:
        spans = candidate.literal_source["spans"]
        if len(spans) == 2:
            return {span["component"]: span["text"] for span in spans}
        return {"value": spans[0]["text"]}
    if candidate.semantic_target:
        return {"value": candidate.semantic_target["canonical_value"]}
    return {}


def _reannotate(candidate: Candidate, request: str) -> Candidate:
    annotation = candidate.literal_source
    if not annotation:
        return replace(candidate, request=request)
    spans: list[dict[str, Any]] = []
    cursor = 0
    for old_span in annotation["spans"]:
        text = old_span["text"]
        start = request.index(text, cursor)
        end = start + len(text)
        spans.append(
            {
                "component": old_span["component"],
                "start": start,
                "end": end,
                "text": text,
            }
        )
        cursor = end
    new_annotation = {**annotation, "spans": spans}
    if candidate.tool in COMPOSITE_TOOLS:
        arguments = {span["component"]: span["text"] for span in spans}
    else:
        arguments = candidate.arguments
    return replace(
        candidate,
        request=request,
        arguments=arguments,
        literal_source=new_annotation,
    )


def pilot_v11_catalog() -> list[tuple[Candidate, tuple[str, ...]]]:
    rows: list[tuple[Candidate, tuple[str, ...]]] = []
    ontology_ids: set[str] = set()
    for candidate in candidate_catalog():
        feature_spec = DIVERSE_REQUESTS.get(candidate.source_id)
        if feature_spec:
            template, features = feature_spec
            request = template.format(**_template_values(candidate))
            candidate = _reannotate(candidate, request)
        else:
            features = ()
            if candidate.tool in COMPOSITE_TOOLS:
                candidate = _reannotate(candidate, candidate.request)
        if candidate.category == "ontology_confusion":
            ontology_ids.add(candidate.source_id)
            if candidate.source_id not in DIVERSE_REQUESTS:
                raise RuntimeError(f"Ontology record was not diversified: {candidate.source_id}")
        rows.append((candidate, features))
    if len(ontology_ids) != 60:
        raise RuntimeError(f"Expected 60 ontology sources, found {len(ontology_ids)}")
    if Counter(candidate.category for candidate, _ in rows) != EXPECTED_CATEGORY_COUNTS:
        raise RuntimeError("Pilot v1.1 category composition drifted")
    return rows


def _resolve_offered(
    registry: Hammer15TrainingRegistryV11, offered: tuple[str, ...] | str
) -> list[str]:
    if isinstance(offered, str):
        return list(registry.resolve_names(group=offered))
    return list(registry.resolve_names(selected=offered))


def _runtime_arguments(
    registry: Hammer15TrainingRegistryV11, candidate: Candidate
) -> dict[str, Any]:
    if candidate.tool is None:
        return {}
    adapted = registry.adapt_training_call_to_live(
        {"tool": candidate.tool, "arguments": candidate.arguments}
    )
    return adapted["arguments"]


def _record_from_candidate(
    candidate: Candidate,
    features: tuple[str, ...],
    *,
    record_id: str,
    registry: Hammer15TrainingRegistryV11,
    seed: int,
) -> dict[str, Any]:
    offered = _resolve_offered(registry, candidate.offered)
    argument_mode = (
        "reject"
        if candidate.tool is None
        else tool_argument_contract(candidate.tool)["mode"]
    )
    return {
        "record_version": RECORD_VERSION_V11,
        "model_visible": render_model_visible_v11(
            registry,
            request=candidate.request,
            offered_tools=offered,
            expected_tool=candidate.tool,
            expected_arguments=candidate.arguments,
        ),
        "metadata": {
            "record_id": record_id,
            "raw_user_request": candidate.request,
            "expected_tool": candidate.tool,
            "expected_arguments": candidate.arguments,
            "runtime_arguments": _runtime_arguments(registry, candidate),
            "argument_contract_version": CONTRACT_VERSION,
            "argument_mode": argument_mode,
            "training_argument_profile": STRUCTURED_ARGUMENT_PROFILE,
            "category": candidate.category,
            "confusion_family": candidate.confusion_family,
            "linguistic_features": list(features),
            "offered_tools": offered,
            "literal_source": candidate.literal_source,
            "semantic_target": candidate.semantic_target,
            "difficulty": candidate.difficulty,
            "prompt_profile": PROMPT_PROFILE_V11,
            "prompt_schema_sha256": prompt_schema_sha256_v11(registry, offered),
            "generation_provenance": {
                "generator": GENERATOR_VERSION_V11,
                "seed": seed,
                "source_id": candidate.source_id,
                "parent_record_version": RECORD_VERSION,
                "template_family": candidate.template_family,
                "authorship": "hand-authored-scenario-catalog",
            },
            "leakage_check": {},
            "validation": {
                "validator_version": VALIDATOR_VERSION_V11,
                "passed": False,
                "errors": [],
            },
        },
    }


def build_records_v11(
    seed: int = DEFAULT_SEED_V11,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verify_v11_safety_constants()
    sacred_cases = json.loads(SACRED_DATASET.read_text(encoding="utf-8"))
    registry = Hammer15TrainingRegistryV11()
    candidates = pilot_v11_catalog()
    random.Random(seed).shuffle(candidates)
    records = [
        _record_from_candidate(
            candidate,
            features,
            record_id=f"h15-pilot-v1.1-{index:03d}",
            registry=registry,
            seed=seed,
        )
        for index, (candidate, features) in enumerate(candidates, 1)
    ]
    rejected: list[dict[str, Any]] = []
    for record in records:
        leakage = check_record(record, sacred_cases)
        record["metadata"]["leakage_check"] = leakage
        errors = record_errors_v11(
            record,
            registry=registry,
            sacred_cases=sacred_cases,
            check_stored_validation=False,
        )
        record["metadata"]["validation"] = {
            "validator_version": VALIDATOR_VERSION_V11,
            "passed": not errors,
            "errors": errors,
        }
        if leakage["status"] == "reject" or errors:
            rejected.append(record)
    return records, rejected


def _manifest(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    metadata = [record["metadata"] for record in records]
    features = Counter(
        feature for row in metadata for feature in row["linguistic_features"]
    )
    return {
        "version": "hammer15-pilot-manifest-v1.1",
        "record_format": RECORD_VERSION_V11,
        "parent_pilot": str(PILOT_PATH),
        "parent_record_count": len(load_jsonl(PILOT_PATH)),
        "seed": seed,
        "record_count": len(records),
        "category_counts": dict(sorted(Counter(row["category"] for row in metadata).items())),
        "argument_mode_counts": dict(sorted(Counter(row["argument_mode"] for row in metadata).items())),
        "tool_counts": {
            str(tool): count
            for tool, count in sorted(
                Counter(row["expected_tool"] for row in metadata).items(),
                key=lambda item: str(item[0]),
            )
        },
        "confusion_family_counts": dict(
            sorted(
                Counter(
                    row["confusion_family"]
                    for row in metadata
                    if row["confusion_family"]
                ).items()
            )
        ),
        "linguistic_feature_counts": dict(sorted(features.items())),
        "structured_composite_count": sum(
            row["expected_tool"] in COMPOSITE_TOOLS for row in metadata
        ),
        "offered_tool_count_distribution": {
            str(count): records
            for count, records in sorted(
                Counter(len(row["offered_tools"]) for row in metadata).items()
            )
        },
        "model_visible_fields": ["prompt", "assistant_target"],
        "metadata_excluded_from_training": True,
        "inference_used": False,
        "yuki_tools_executed": False,
        "sacred_dataset_sha256": verify_v11_safety_constants(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Hammer Pilot v1.1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED_V11)
    parser.add_argument("--output", type=Path, default=PILOT_PATH_V11)
    args = parser.parse_args()
    records, rejected = build_records_v11(args.seed)
    write_jsonl(PILOT_REJECTED_V11, rejected)
    if rejected:
        raise RuntimeError(
            f"Pilot v1.1 gate rejected {len(rejected)} records; "
            f"inspect {PILOT_REJECTED_V11}"
        )
    write_jsonl(args.output, records)
    leakage = check_records(records)
    validation = validate_records_v11(records)
    write_json(PILOT_LEAKAGE_V11, leakage)
    write_json(PILOT_VALIDATION_V11, validation)
    write_json(PILOT_MANIFEST_V11, _manifest(records, args.seed))
    if not validation["passed"]:
        raise RuntimeError("Generated Pilot v1.1 failed its independent validator")
    checksums = {
        path.name: sha256_path(path)
        for path in (
            args.output,
            PILOT_MANIFEST_V11,
            PILOT_VALIDATION_V11,
            PILOT_LEAKAGE_V11,
        )
    }
    write_json(args.output.with_suffix(".checksums.json"), checksums)
    print(
        json.dumps(
            {
                "records": len(records),
                "leakage": leakage["counts"],
                "validation_passed": validation["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
