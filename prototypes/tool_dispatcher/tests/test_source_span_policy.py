from __future__ import annotations

from pathlib import Path

import pytest

from prototypes.tool_dispatcher.source_span_policy import bind_source_argument


def _bind(raw: str, tool: str, arguments: dict | None = None) -> dict:
    return bind_source_argument(
        raw_request=raw,
        selected_call={"tool": tool, "arguments": arguments or {}},
        semantic_request="A semantic hint that must not supply output characters.",
        verbatim=[],
    )


@pytest.mark.parametrize(
    ("raw", "tool", "argument", "expected"),
    [
        (
            "Search the web for the latest runtime documentation.",
            "search",
            "query",
            "the latest runtime documentation",
        ),
        (
            "Create an image of a glass observatory.",
            "image",
            "prompt",
            "a glass observatory",
        ),
        (
            "Ask Claude: is `x-y` safe? Do not execute it.",
            "ask_claude",
            "question",
            "is `x-y` safe? Do not execute it.",
        ),
        (
            "Use the camera to find my green notebook.",
            "see",
            "target",
            "my green notebook",
        ),
        (
            "Recall our previous discussion about the build failure.",
            "recall",
            "topic",
            "the build failure",
        ),
    ],
)
def test_binding_preserves_literal_source_span(
    raw: str, tool: str, argument: str, expected: str
) -> None:
    result = _bind(raw, tool)
    assert result["status"] == "bound"
    assert result["final_call"]["arguments"] == {argument: expected}
    span = result["source_span"]
    assert expected == raw[span["start"] : span["end"]]
    assert result["source_copy_valid"] is True


def test_routing_guard_is_not_part_of_image_payload() -> None:
    result = _bind(
        "Generate a brass fox; do not open the camera.",
        "image",
    )
    assert result["final_call"]["arguments"] == {"prompt": "a brass fox"}


def test_camera_syntax_is_not_part_of_see_target() -> None:
    result = _bind(
        "Find the printed image with the camera; do not generate a new image.",
        "see",
    )
    assert result["status"] == "bound"
    assert result["final_call"]["arguments"] == {"target": "the printed image"}


def test_repeated_see_target_abstains_as_ambiguous() -> None:
    result = _bind(
        "Point the camera toward the window and find the window.",
        "see",
    )
    assert result["status"] == "ambiguous"
    assert result["final_call"] is None
    assert len(result["candidates"]) == 2


def test_whole_scene_see_request_has_no_argument() -> None:
    result = _bind("Look around the room and describe the scene.", "see")
    assert result["status"] == "no_argument"
    assert result["final_call"] == {"tool": "see", "arguments": {}}


def test_unbound_required_argument_abstains() -> None:
    result = _bind("Please do something unrelated.", "search")
    assert result["status"] == "unbound"
    assert result["final_call"] is None


def test_exact_model_hint_is_only_used_when_it_occurs_in_raw() -> None:
    result = bind_source_argument(
        raw_request="Locate `AirPods Pro` and preserve capitalization.",
        selected_call={"tool": "search", "arguments": {"query": "AirPods Pro"}},
        semantic_request="Search for the supplied item.",
        verbatim=["regenerated text that is not in the source"],
    )
    assert result["status"] == "bound"
    assert result["final_call"]["arguments"] == {"query": "AirPods Pro"}


def test_exact_os_path_and_yuki_filename_are_anchored_without_guessing() -> None:
    path = _bind(
        "Preserve case exactly when reading /tmp/FooBar.JSON.",
        "read",
        {"filepath": "/tmp/FooBar.JSON"},
    )
    filename = _bind(
        "Read Yuki filename exactly `My Plan 2.md`.",
        "yuki_read",
        {"filename": "My Plan 2.md"},
    )
    assert path["final_call"]["arguments"] == {"filepath": "/tmp/FooBar.JSON"}
    assert filename["final_call"]["arguments"] == {"filename": "My Plan 2.md"}
    assert path["source_copy_valid"] is True
    assert filename["source_copy_valid"] is True


def test_literal_yuki_pipe_payload_is_one_source_span() -> None:
    raw = 'Write status.json|{"ready": true} in Yuki\'s personal folder.'
    result = _bind(
        raw,
        "yuki_write",
        {"filename_and_content": "status.json|{ready: true}"},
    )
    assert result["status"] == "bound"
    assert result["final_call"]["arguments"] == {
        "filename_and_content": 'status.json|{"ready": true}'
    }
    assert result["source_copy_valid"] is True


def test_noncontiguous_yuki_payload_uses_only_source_slices_and_fixed_separator() -> None:
    raw = "Add exact text to `My Plan 2.md`: café meetup — bring USB-C hub"
    result = _bind(raw, "yuki_append", {"filename_and_content": "rewritten"})
    assert result["status"] == "bound_composite"
    assert result["deterministic_separator"] == "|"
    assert result["final_call"]["arguments"] == {
        "filename_and_content": "My Plan 2.md|café meetup — bring USB-C hub"
    }
    assert all(
        span["text"] == raw[span["start"] : span["end"]]
        for span in result["source_spans"]
    )
    assert result["source_copy_valid"] is True


def test_no_selected_call_and_unsupported_tool_do_not_guess() -> None:
    no_call = bind_source_argument(
        raw_request="Search for x",
        selected_call=None,
        semantic_request="Search.",
        verbatim=[],
    )
    unsupported = _bind("What time is it?", "time")
    assert no_call["status"] == "no_selected_call"
    assert unsupported["status"] == "unsupported_selected_tool"
    assert no_call["final_call"] is None
    assert unsupported["final_call"] is None


def test_binder_source_has_no_gold_annotation_or_filesystem_access() -> None:
    source = Path(__file__).parents[1].joinpath("source_span_policy.py").read_text()
    forbidden = (
        "literal-source-annotations",
        "expected_arguments",
        "official_gold",
        "Path(",
        "open(",
        "read_text(",
        "json.load",
    )
    assert all(token not in source for token in forbidden)
