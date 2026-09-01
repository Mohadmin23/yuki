"""Build the deterministic, model-free Hammer 1.5B pilot dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..argument_contract import CONTRACT_VERSION, tool_argument_contract
from .check_sacred450_leakage import check_record, check_records
from .common import (
    DEFAULT_SEED,
    GENERATOR_VERSION,
    PILOT_LEAKAGE,
    PILOT_MANIFEST,
    PILOT_PATH,
    PILOT_REJECTED,
    PILOT_VALIDATION,
    PROMPT_PROFILE,
    SACRED_DATASET,
    VALIDATOR_VERSION,
    Hammer15TrainingRegistry,
    prompt_schema_sha256,
    render_model_visible,
    sha256_path,
    verify_sacred_dataset,
    write_json,
    write_jsonl,
)
from .validate_hammer15_dataset import (
    EXPECTED_CATEGORY_COUNTS,
    record_errors,
    validate_records,
)

ALL = "all"
INFO = "information"
COMPUTER = "computer"
MEDIA = "media"
YUKI_FILES = "yuki-files"
MEMORY = "memory"
EXTERNAL = "external-agent"


@dataclass(frozen=True)
class Candidate:
    source_id: str
    request: str
    tool: str | None
    arguments: dict[str, str]
    category: str
    confusion_family: str | None
    offered: tuple[str, ...] | str
    difficulty: str
    template_family: str
    literal_source: dict[str, Any] | None = None
    semantic_target: dict[str, Any] | None = None


def _span(request: str, text: str, component: str, start_at: int = 0) -> dict[str, Any]:
    start = request.index(text, start_at)
    return {
        "component": component,
        "start": start,
        "end": start + len(text),
        "text": text,
    }


def literal(
    source_id: str,
    *,
    prefix: str,
    payload: str,
    suffix: str,
    tool: str,
    category: str,
    offered: tuple[str, ...] | str,
    family: str | None = None,
    difficulty: str = "medium",
) -> Candidate:
    request = f"{prefix}{payload}{suffix}"
    field = tool_argument_contract(tool)["argument"]
    return Candidate(
        source_id,
        request,
        tool,
        {field: payload},
        category,
        family,
        offered,
        difficulty,
        f"literal-{tool}",
        literal_source={
            "field": field,
            "spans": [_span(request, payload, field)],
            "deterministic_separator": None,
            "reconstructed_value": payload,
        },
    )


def composite(
    source_id: str,
    *,
    prefix: str,
    filename: str,
    middle: str,
    content: str,
    suffix: str,
    tool: str,
    category: str,
    offered: tuple[str, ...] | str,
    family: str | None = None,
    difficulty: str = "hard",
) -> Candidate:
    request = f"{prefix}{filename}{middle}{content}{suffix}"
    field = tool_argument_contract(tool)["argument"]
    filename_span = _span(request, filename, "filename")
    content_span = _span(request, content, "content", filename_span["end"])
    value = f"{filename}|{content}"
    return Candidate(
        source_id,
        request,
        tool,
        {field: value},
        category,
        family,
        offered,
        difficulty,
        f"composite-{tool}",
        literal_source={
            "field": field,
            "spans": [filename_span, content_span],
            "deterministic_separator": "|",
            "reconstructed_value": value,
        },
    )


def semantic(
    source_id: str,
    *,
    request: str,
    tool: str,
    value: str,
    category: str,
    offered: tuple[str, ...] | str,
    family: str | None = None,
    alternatives: tuple[str, ...] = (),
    difficulty: str = "medium",
) -> Candidate:
    field = tool_argument_contract(tool)["argument"]
    return Candidate(
        source_id,
        request,
        tool,
        {field: value},
        category,
        family,
        offered,
        difficulty,
        f"semantic-{tool}",
        semantic_target={
            "field": field,
            "canonical_value": value,
            "accepted_alternatives": list(alternatives),
        },
    )


def noarg(
    source_id: str,
    *,
    request: str,
    tool: str,
    category: str,
    offered: tuple[str, ...] | str,
    family: str | None = None,
    difficulty: str = "easy",
) -> Candidate:
    return Candidate(
        source_id,
        request,
        tool,
        {},
        category,
        family,
        offered,
        difficulty,
        f"noarg-{tool}",
    )


def reject(
    source_id: str,
    *,
    request: str,
    offered: tuple[str, ...] | str,
    family: str,
) -> Candidate:
    return Candidate(
        source_id,
        request,
        None,
        {},
        "adversarial_edge",
        family,
        offered,
        "hard",
        "explicit-reject",
    )


def _ontology_candidates() -> list[Candidate]:
    category = "ontology_confusion"
    rows: list[Candidate] = []

    normal_paths = [
        ("/var/tmp/ember-state.toml", "Open the operating-system file at ", "."),
        ("~/Desktop/Quartz Notes.md", "Read this disk path exactly: ", "."),
        ("/opt/local/share/aurora.ini", "Show me the contents of ", " from the Mac filesystem."),
        ("./fixtures/case-sensitive.JSON", "Use the normal file reader on ", "."),
        ("/Users/me/Archive/Map 07.txt", "Inspect the local filesystem file ", "."),
    ]
    yuki_names = [
        ("moon-log.yuki", "Read ", " from your own internal file store."),
        ("tiny rituals.md", "Open your Yuki-store note named ", "."),
        ("Raven_Index.txt", "Get the internal Yuki file ", "."),
        ("palette-v7.json", "Read your managed note ", ", not a disk path."),
        ("weekend.queue", "From Yuki's private file shelf, open ", "."),
    ]
    for index, (payload, prefix, suffix) in enumerate(normal_paths, 1):
        rows.append(literal(f"onto-read-{index}", prefix=prefix, payload=payload, suffix=suffix, tool="read", category=category, offered=("read", "yuki_read"), family="read-vs-yuki_read"))
    for index, (payload, prefix, suffix) in enumerate(yuki_names, 1):
        rows.append(literal(f"onto-yread-{index}", prefix=prefix, payload=payload, suffix=suffix, tool="yuki_read", category=category, offered=("read", "yuki_read"), family="read-vs-yuki_read"))

    replacements = [
        ("inventory.card", "copper=4; zinc=9"),
        ("focus.note", "Only finish the parser tonight."),
        ("garden.plan", "Replace basil with mint."),
        ("boot-sequence.txt", "wake\ncheck sensors\nreport"),
        ("colors.cfg", "accent=#EF3340"),
    ]
    additions = [
        ("field-journal.md", "Add: heard an owl at 02:10."),
        ("snack-count.csv", "2026-08-25,pretzels,2"),
        ("ideas.txt", "A keyboard that purrs when idle."),
        ("repair.log", "Retested connector B; stable."),
        ("watchlist.note", "The Silent Sea — episode 3"),
    ]
    for index, (filename, content) in enumerate(replacements, 1):
        rows.append(composite(f"onto-ywrite-{index}", prefix="Replace everything in your internal file ", filename=filename, middle=" with exactly: ", content=content, suffix="", tool="yuki_write", category=category, offered=("yuki_write", "yuki_append"), family="yuki_write-vs-yuki_append"))
    for index, (filename, content) in enumerate(additions, 1):
        rows.append(composite(f"onto-yappend-{index}", prefix="Keep the existing Yuki file ", filename=filename, middle=" and append this after it: ", content=content, suffix="", tool="yuki_append", category=category, offered=("yuki_write", "yuki_append"), family="yuki_write-vs-yuki_append"))

    see_cases = [
        ("Check the camera view for my soldering iron.", "soldering iron"),
        ("Look through your visual input and locate the striped mug.", "striped mug"),
        ("Inspect what is already on screen for a warning triangle.", "warning triangle"),
        ("Use your vision feed to see whether the window is open.", "window"),
        ("Tell me what you can observe around the blue toolbox.", "blue toolbox"),
    ]
    image_prompts = [
        ("Generate a new picture using this prompt: ", "glass koi swimming through a library", "."),
        ("Create an image of ", "a mossy tram under violet rain", "."),
        ("Render this visual description exactly: ", "isometric bakery on a floating pebble", ""),
        ("Make new artwork showing ", "three paper moons above a red desert", "."),
        ("Produce an image from: ", "pixel-art lighthouse inside a snow globe", ""),
    ]
    for index, (request, value) in enumerate(see_cases, 1):
        rows.append(semantic(f"onto-see-{index}", request=request, tool="see", value=value, alternatives=(f"the {value}",), category=category, offered=("see", "image"), family="see-vs-image"))
    for index, (prefix, payload, suffix) in enumerate(image_prompts, 1):
        rows.append(literal(f"onto-image-{index}", prefix=prefix, payload=payload, suffix=suffix, tool="image", category=category, offered=("see", "image"), family="see-vs-image"))

    searches = [
        "Open-source thermal camera calibration guide 2026",
        "why sqlite wal file remains after checkpoint",
        "quiet 60 percent keyboard foam comparison",
        "Metal shader compile cache location",
        "best way to label drawer cables",
    ]
    recalls = [
        ("What did I previously say about the ceramic keyboard caps?", "ceramic keyboard caps"),
        ("Bring back what you stored concerning the balcony herbs.", "balcony herbs"),
        ("Do you remember our earlier notes on the amber power light?", "amber power light"),
        ("Retrieve my saved history about the bicycle lock code.", "bicycle lock code"),
        ("what'd i tell ya bout the attic router?", "attic router"),
    ]
    for index, payload in enumerate(searches, 1):
        rows.append(literal(f"onto-search-{index}", prefix="Search the public web for: ", payload=payload, suffix="", tool="search", category=category, offered=("search", "recall"), family="search-vs-recall"))
    for index, (request, value) in enumerate(recalls, 1):
        rows.append(semantic(f"onto-recall-{index}", request=request, tool="recall", value=value, alternatives=(f"the {value}",), category=category, offered=("search", "recall"), family="search-vs-recall"))

    memories = [
        ("Store that I take tea without sugar.", "user takes tea without sugar"),
        ("Remember that my spare adapter is in drawer four.", "user's spare adapter is in drawer four"),
        ("Keep this fact for later: I prefer subtitles on.", "user prefers subtitles on"),
        ("Save the fact that my bicycle is called Mica.", "user's bicycle is called Mica"),
        ("Remember: I avoid meetings before ten.", "user avoids meetings before ten"),
    ]
    memory_files = [
        ("preference.card", "tea=no sugar"),
        ("storage.map", "spare adapter=drawer four"),
        ("playback.cfg", "subtitles=on"),
        ("bike.note", "name=Mica"),
        ("schedule.rule", "meetings>=10:00"),
    ]
    for index, (request, value) in enumerate(memories, 1):
        rows.append(semantic(f"onto-remember-{index}", request=request, tool="remember", value=value, category=category, offered=("remember", "yuki_write"), family="remember-vs-yuki_write"))
    for index, (filename, content) in enumerate(memory_files, 1):
        rows.append(composite(f"onto-ywrite-memory-{index}", prefix="Replace your internal file ", filename=filename, middle=" with the text ", content=content, suffix="", tool="yuki_write", category=category, offered=("remember", "yuki_write"), family="remember-vs-yuki_write"))

    shell_commands = ["uptime -p", "df -P", "uname -srm", "pwd", "wc -l changelog.txt"]
    hardware_cases = [
        ("How much system memory is currently in use?", "ram"),
        ("Report the graphics processor status.", "gpu"),
        ("Report the current process-resource summary.", "process"),
        ("Give me the disk utilization summary.", "disk"),
        ("Show the processor load metric.", "cpu"),
    ]
    for index, payload in enumerate(shell_commands, 1):
        rows.append(literal(f"onto-shell-{index}", prefix="Run this allowed terminal command exactly: ", payload=payload, suffix="", tool="shell", category=category, offered=("hardware", "shell"), family="shell-vs-hardware"))
    for index, (request, value) in enumerate(hardware_cases, 1):
        rows.append(semantic(f"onto-hardware-{index}", request=request, tool="hardware", value=value, category=category, offered=("hardware", "shell"), family="shell-vs-hardware"))
    assert len(rows) == 60
    return rows


def _literal_candidates() -> list[Candidate]:
    category = "literal_preservation"
    specs: list[tuple[str, str, str, str, str, tuple[str, ...] | str]] = [
        ("lit-search-1", "Look this up without editing the query: ", "C++ coroutine co_await latency M1", "", "search", INFO),
        ("lit-search-2", "Search for the exact phrase ", '"red team" NOT football', ".", "search", ALL),
        ("lit-search-3", "Web search payload follows: ", "foo_bar() vs fooBar() ABI", "", "search", INFO),
        ("lit-search-4", "Find online results for ", "RTX 4070 Ti SUPER 16GB Columbus", ".", "search", ALL),
        ("lit-search-5", "Search exactly: ", "site:docs.python.org pathlib PurePath", "", "search", INFO),
        ("lit-image-1", "Generate this image: ", "a fox labeled 'NODE_7' beneath cyan fog", "", "image", MEDIA),
        ("lit-image-2", "Use this precise art prompt: ", "35mm photo, rain-soaked arcade, f/1.8", "", "image", ALL),
        ("lit-image-3", "Create new artwork from ", "tiny robot holding a sign: DON'T PANIC", ".", "image", MEDIA),
        ("lit-image-4", "Image prompt—copy it as written: ", "ink wash crane; negative space; #D94A4A", "", "image", MEDIA),
        ("lit-image-5", "Render ", "a room where every shadow points east", " as a new image.", "image", ALL),
        ("lit-claude-1", "Ask the external agent: ", "Compare parse_v2() with parseV2(); preserve both names.", "", "ask_claude", EXTERNAL),
        ("lit-claude-2", "Send Claude this exact question: ", "Why does x != y after NFKC?", "", "ask_claude", ALL),
        ("lit-claude-3", "Pass this message to Claude unchanged: ", "Review ERROR_CODE=0xAF, not 0xaf.", "", "ask_claude", EXTERNAL),
        ("lit-claude-4", "Ask Claude ", "Can `map[key]` mutate when key == 'A/B'?", ".", "ask_claude", ALL),
        ("lit-claude-5", "External-agent payload: ", "Explain why her value stayed 'null'. Do not change the pronoun.", "", "ask_claude", EXTERNAL),
        ("lit-read-1", "Read the disk file ", "/tmp/Case-Sensitive Ω.txt", ".", "read", COMPUTER),
        ("lit-read-2", "Open this exact local path: ", "~/Library/Application Support/Yuki/state.db", "", "read", ALL),
        ("lit-read-3", "Use the filesystem reader for ", "./data/2026-08/notes.final.md", ".", "read", COMPUTER),
        ("lit-read-4", "Read ", "/Volumes/Field Kit/Run #03.log", " from the operating-system filesystem.", "read", ALL),
        ("lit-fetch-1", "Fetch this URL verbatim: ", "https://example.net/a%2Fb?q=X+Y&v=03", "", "fetch", INFO),
        ("lit-fetch-2", "Download the page at ", "https://docs.example.org/Guide#Part-2", ".", "fetch", ALL),
        ("lit-fetch-3", "Retrieve exactly ", "http://127.0.0.1:9999/status?full=true", "", "fetch", INFO),
        ("lit-shell-1", "Run the allowed command ", "head -n 7 CHANGELOG.md", " exactly.", "shell", COMPUTER),
        ("lit-shell-2", "Terminal command, preserve spacing: ", "echo  YUKI_09", "", "shell", ALL),
        ("lit-shell-3", "Execute this permitted command: ", "tail -n 12 /tmp/ember.log", "", "shell", COMPUTER),
        ("lit-shell-4", "Run exactly: ", "which uv", "", "shell", COMPUTER),
        ("lit-yread-1", "Open the Yuki-store filename ", "Exact Name (v2).txt", ".", "yuki_read", YUKI_FILES),
        ("lit-ydelete-1", "Delete only this internal Yuki file: ", "obsolete-DRAFT_09.note", "", "yuki_delete", YUKI_FILES),
        ("lit-ydelete-2", "Remove the Yuki-store entry named ", "temp palette #4.json", ".", "yuki_delete", ALL),
    ]
    rows = [
        literal(source_id, prefix=prefix, payload=payload, suffix=suffix, tool=tool, category=category, offered=offered)
        for source_id, prefix, payload, suffix, tool, offered in specs
    ]
    rows.extend(
        [
            composite("lit-ywrite-1", prefix="Overwrite your internal file ", filename="Release Notes.MD", middle=" with exactly ", content="v2.0 — ship on Friday; owner=me", suffix="", tool="yuki_write", category=category, offered=YUKI_FILES),
            composite("lit-ywrite-2", prefix="Replace the Yuki note ", filename="labels.txt", middle=" using this content: ", content="A/B != A\\B", suffix="", tool="yuki_write", category=category, offered=ALL),
            composite("lit-ywrite-3", prefix="Write fresh contents to your file ", filename="unicode.note", middle=": ", content="café → CAFÉ? no", suffix="", tool="yuki_write", category=category, offered=YUKI_FILES),
            composite("lit-yappend-1", prefix="Append to the existing Yuki file ", filename="measurements.csv", middle=" this row: ", content="07,3.140,μs", suffix="", tool="yuki_append", category=category, offered=YUKI_FILES),
            composite("lit-yappend-2", prefix="Add after the current contents of ", filename="quirks.md", middle=" this exact line: ", content="- keep  TWO spaces", suffix="", tool="yuki_append", category=category, offered=ALL),
            composite("lit-yappend-3", prefix="Preserve and extend your note ", filename="todo.yuki", middle=" by appending ", content="[ ] test_case-β", suffix="", tool="yuki_append", category=category, offered=YUKI_FILES),
        ]
    )
    assert len(rows) == 35
    return rows


def _semantic_candidates() -> list[Candidate]:
    category = "semantic_argument"
    rows: list[Candidate] = []
    weather = [
        ("Could you get the forecast for Reykjavík?", "Reykjavík"),
        ("I need current weather around São Paulo.", "São Paulo"),
        ("What's it like outside in Kyoto today?", "Kyoto"),
        ("wether for Québec City pls?", "Québec City"),
        ("Weather report, please, for Cape Town.", "Cape Town"),
    ]
    calcs = [
        ("Work out twelve times nineteen.", "12*19", ("12 * 19",)),
        ("Calculate one quarter of 360.", "360/4", ("360 / 4",)),
        ("What is seven squared plus three?", "7*7+3", ("(7*7)+3",)),
        ("Evaluate eighty minus 14, then divide by three.", "(80-14)/3", ("(80 - 14) / 3",)),
        ("Compute 15 percent of 240.", "240*0.15", ("240 * 0.15",)),
    ]
    hardware = [
        ("Is the machine running hot? Show temperature stats.", "temp"),
        ("Give me the busiest processes right now.", "top"),
        ("How full are the storage volumes?", "disk"),
        ("Show a complete hardware health overview.", "all"),
    ]
    seeing = [
        ("Can your visual feed find my orange screwdriver?", "orange screwdriver"),
        ("Look at the current scene and focus on the left monitor.", "left monitor"),
        ("Use the camera to check for a loose cable.", "loose cable"),
        ("Describe what your vision input currently sees.", "", ()),
    ]
    memories = [
        ("Remember that I park on level C2.", "user parks on level C2"),
        ("Keep in memory: my dentist is Dr. Vale.", "user's dentist is Dr. Vale"),
        ("Save that I like the desk lamp at 30 percent.", "user likes the desk lamp at 30 percent"),
        ("Remember I promised Lina the blue book.", "user promised Lina the blue book"),
    ]
    recalls = [
        ("What did I tell you earlier about the purple suitcase?", "purple suitcase"),
        ("Retrieve what you remember concerning my tax folder.", "tax folder"),
        ("Bring up our stored history on the north-window leak.", "north-window leak"),
    ]
    for index, (request, value) in enumerate(weather, 1):
        rows.append(semantic(f"sem-weather-{index}", request=request, tool="weather", value=value, category=category, offered=INFO))
    for index, (request, value, alternatives) in enumerate(calcs, 1):
        rows.append(semantic(f"sem-calc-{index}", request=request, tool="calc", value=value, alternatives=alternatives, category=category, offered=INFO))
    for index, (request, value) in enumerate(hardware, 1):
        rows.append(semantic(f"sem-hardware-{index}", request=request, tool="hardware", value=value, category=category, offered=COMPUTER))
    for index, seeing_case in enumerate(seeing, 1):
        request, value, *_ = seeing_case
        alternatives = seeing_case[2] if len(seeing_case) > 2 else (f"the {value}",)
        if value:
            rows.append(semantic(f"sem-see-{index}", request=request, tool="see", value=value, alternatives=alternatives, category=category, offered=MEDIA))
        else:
            rows.append(Candidate(f"sem-see-{index}", request, "see", {}, category, None, MEDIA, "easy", "semantic-see-omitted"))
    for index, (request, value) in enumerate(memories, 1):
        rows.append(semantic(f"sem-remember-{index}", request=request, tool="remember", value=value, category=category, offered=MEMORY))
    for index, (request, value) in enumerate(recalls, 1):
        rows.append(semantic(f"sem-recall-{index}", request=request, tool="recall", value=value, alternatives=(f"the {value}",), category=category, offered=MEMORY))
    assert len(rows) == 25
    return rows


def _ordinary_candidates() -> list[Candidate]:
    category = "ordinary_easy"
    return [
        noarg("easy-time-1", request="Tell me the current local time.", tool="time", category=category, offered=INFO),
        noarg("easy-time-2", request="What date and time is it?", tool="time", category=category, offered=ALL),
        noarg("easy-ylist-1", request="Show the names of every file in your internal store.", tool="yuki_list", category=category, offered=YUKI_FILES),
        noarg("easy-ylist-2", request="List your managed Yuki files.", tool="yuki_list", category=category, offered=ALL),
        semantic("easy-weather", request="Check the weather in Oslo.", tool="weather", value="Oslo", category=category, offered=INFO, difficulty="easy"),
        semantic("easy-calc", request="Calculate 9 plus 6.", tool="calc", value="9+6", alternatives=("9 + 6",), category=category, offered=INFO, difficulty="easy"),
        semantic("easy-hardware", request="Show RAM usage.", tool="hardware", value="ram", category=category, offered=COMPUTER, difficulty="easy"),
        semantic("easy-see", request="Look for the green notebook with your camera.", tool="see", value="green notebook", alternatives=("the green notebook",), category=category, offered=MEDIA, difficulty="easy"),
        semantic("easy-remember", request="Remember that my alarm is set for 6:40.", tool="remember", value="user's alarm is set for 6:40", category=category, offered=MEMORY, difficulty="easy"),
        semantic("easy-recall", request="Recall what I said about the camping stove.", tool="recall", value="camping stove", alternatives=("the camping stove",), category=category, offered=MEMORY, difficulty="easy"),
        literal("easy-fetch", prefix="URL fetch: ", payload="https://sample.example/clock", suffix="", tool="fetch", category=category, offered=INFO, difficulty="easy"),
        literal("easy-search", prefix="Search for ", payload="low-profile keycap sizes", suffix="", tool="search", category=category, offered=INFO, difficulty="easy"),
        literal("easy-read", prefix="Read ", payload="/tmp/yuki-demo.txt", suffix="", tool="read", category=category, offered=COMPUTER, difficulty="easy"),
        literal("easy-image", prefix="Create an image of ", payload="a sleepy comet", suffix="", tool="image", category=category, offered=MEDIA, difficulty="easy"),
        literal("easy-claude", prefix="Ask Claude: ", payload="Check this traceback for the likely cause.", suffix="", tool="ask_claude", category=category, offered=EXTERNAL, difficulty="easy"),
    ]


def _edge_candidates() -> list[Candidate]:
    category = "adversarial_edge"
    return [
        reject("edge-reject-1", request="Use the camera, but I forgot what I wanted you to locate.", offered=("image",), family="unsupported-by-offered-subset"),
        reject("edge-reject-2", request="Read a file for me; I did not provide its path.", offered=("read",), family="missing-required-argument"),
        reject("edge-reject-3", request="Search for something useful, no particular subject.", offered=("search",), family="missing-required-argument"),
        reject("edge-reject-4", request="Please schedule a calendar meeting tomorrow.", offered=("time", "weather"), family="unsupported-capability"),
        reject("edge-reject-5", request="Send a message to the external agent, but there is no message yet.", offered=("ask_claude",), family="missing-required-argument"),
        reject("edge-reject-6", request="Delete the internal note whose name I cannot remember.", offered=("yuki_delete",), family="missing-required-argument"),
        literal("edge-search", prefix="Search; the semicolon belongs to the query: ", payload="alpha; beta | gamma", suffix="", tool="search", category=category, offered=ALL, family="punctuation-boundary", difficulty="hard"),
        literal("edge-image", prefix="Generate an image. The prompt itself is: ", payload="'NO SIGNAL' — white text on black", suffix="", tool="image", category=category, offered=MEDIA, family="quote-boundary", difficulty="hard"),
        literal("edge-claude", prefix="Ask Claude this, including the question mark inside: ", payload="Did variable 'it' mean item_0?", suffix="", tool="ask_claude", category=category, offered=ALL, family="pronoun-and-quote-preservation", difficulty="hard"),
        literal("edge-read", prefix="The period is sentence punctuation; read the path ", payload="/tmp/archive.v1/file", suffix=".", tool="read", category=category, offered=("read", "yuki_read"), family="delimiter-boundary", difficulty="hard"),
        composite("edge-ywrite", prefix="Replace ", filename="pipes.note", middle=" in your Yuki store with this content, whose pipe is payload: ", content="left|right", suffix="", tool="yuki_write", category=category, offered=("yuki_write", "yuki_append"), family="composite-payload-has-separator", difficulty="hard"),
        composite("edge-yappend", prefix="Append to ", filename="blank-lines.md", middle=" this two-line payload: ", content="first\n\nthird", suffix="", tool="yuki_append", category=category, offered=("yuki_write", "yuki_append"), family="multiline-literal", difficulty="hard"),
        semantic("edge-see", request="Don't generate anything—inspect the live view for my 'A/B' adapter.", tool="see", value="A/B adapter", alternatives=("my A/B adapter", "the A/B adapter"), category=category, offered=("see", "image"), family="negated-competing-tool", difficulty="hard"),
        semantic("edge-recall", request="Don't search online; retrieve what I previously told you about Project Moth.", tool="recall", value="Project Moth", alternatives=("the Project Moth",), category=category, offered=("search", "recall"), family="negated-competing-tool", difficulty="hard"),
        semantic("edge-hardware", request="Do not run `df`; use your hardware telemetry to report storage usage.", tool="hardware", value="disk", category=category, offered=("hardware", "shell"), family="negated-competing-tool", difficulty="hard"),
    ]


def candidate_catalog() -> list[Candidate]:
    rows = (
        _ontology_candidates()
        + _literal_candidates()
        + _semantic_candidates()
        + _ordinary_candidates()
        + _edge_candidates()
    )
    counts = Counter(row.category for row in rows)
    if dict(counts) != EXPECTED_CATEGORY_COUNTS:
        raise RuntimeError(f"Pilot composition drifted: {dict(counts)}")
    source_ids = [row.source_id for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Pilot source IDs are not unique")
    return rows


def _resolve_offered(
    registry: Hammer15TrainingRegistry, offered: tuple[str, ...] | str
) -> list[str]:
    if isinstance(offered, str):
        return list(registry.resolve_names(group=offered))
    return list(registry.resolve_names(selected=offered))


def _record_from_candidate(
    candidate: Candidate,
    *,
    record_id: str,
    registry: Hammer15TrainingRegistry,
    seed: int,
) -> dict[str, Any]:
    offered = _resolve_offered(registry, candidate.offered)
    if candidate.tool is None:
        argument_mode = "reject"
    else:
        argument_mode = tool_argument_contract(candidate.tool)["mode"]
    return {
        "record_version": "hammer15-training-record-v1",
        "model_visible": render_model_visible(
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
            "argument_contract_version": CONTRACT_VERSION,
            "argument_mode": argument_mode,
            "category": candidate.category,
            "confusion_family": candidate.confusion_family,
            "offered_tools": offered,
            "literal_source": candidate.literal_source,
            "semantic_target": candidate.semantic_target,
            "difficulty": candidate.difficulty,
            "prompt_profile": PROMPT_PROFILE,
            "prompt_schema_sha256": prompt_schema_sha256(registry, offered),
            "generation_provenance": {
                "generator": GENERATOR_VERSION,
                "seed": seed,
                "source_id": candidate.source_id,
                "template_family": candidate.template_family,
                "authorship": "hand-authored-scenario-catalog",
            },
            "leakage_check": {},
            "validation": {
                "validator_version": VALIDATOR_VERSION,
                "passed": False,
                "errors": [],
            },
        },
    }


def build_records(seed: int = DEFAULT_SEED) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verify_sacred_dataset()
    sacred_cases = json.loads(SACRED_DATASET.read_text(encoding="utf-8"))
    registry = Hammer15TrainingRegistry()
    candidates = candidate_catalog()
    random.Random(seed).shuffle(candidates)
    records = [
        _record_from_candidate(
            candidate,
            record_id=f"h15-pilot-v1-{index:03d}",
            registry=registry,
            seed=seed,
        )
        for index, candidate in enumerate(candidates, 1)
    ]

    rejected: list[dict[str, Any]] = []
    for record in records:
        leakage = check_record(record, sacred_cases)
        record["metadata"]["leakage_check"] = leakage
        errors = record_errors(
            record,
            registry=registry,
            sacred_cases=sacred_cases,
            check_stored_validation=False,
        )
        validation = {
            "validator_version": VALIDATOR_VERSION,
            "passed": not errors,
            "errors": errors,
        }
        record["metadata"]["validation"] = validation
        if leakage["status"] == "reject" or errors:
            rejected.append(record)
    return records, rejected


def _manifest(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    metadata = [record["metadata"] for record in records]
    return {
        "version": "hammer15-pilot-manifest-v1",
        "record_format": "hammer15-training-record-v1",
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
        "confusion_family_counts": {
            str(family): count
            for family, count in sorted(
                Counter(row["confusion_family"] for row in metadata if row["confusion_family"]).items()
            )
        },
        "model_visible_fields": ["prompt", "assistant_target"],
        "metadata_excluded_from_training": True,
        "inference_used": False,
        "yuki_tools_executed": False,
        "sacred_dataset_sha256": verify_sacred_dataset(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Hammer 1.5B pilot")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=PILOT_PATH)
    args = parser.parse_args()

    records, rejected = build_records(args.seed)
    write_jsonl(PILOT_REJECTED, rejected)
    if rejected:
        raise RuntimeError(
            f"Pilot gate rejected {len(rejected)} records; inspect {PILOT_REJECTED}"
        )

    write_jsonl(args.output, records)
    leakage = check_records(records)
    validation = validate_records(records)
    write_json(PILOT_LEAKAGE, leakage)
    write_json(PILOT_VALIDATION, validation)
    write_json(PILOT_MANIFEST, _manifest(records, args.seed))
    if not validation["passed"]:
        raise RuntimeError("Generated pilot failed its independent validator")

    checksums = {
        path.name: sha256_path(path)
        for path in (args.output, PILOT_MANIFEST, PILOT_VALIDATION, PILOT_LEAKAGE)
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
