"""Model-native structured-call dialect selection."""
from __future__ import annotations

XLAM_V1_ENVELOPE = "xlam_v1_tool_calls_envelope"
TOOL_CALL_ARRAY = "tool_call_array"
OPENAI_NATIVE_TOOL_CALLS = "openai_native_tool_calls"
HAMMER_FENCED_ARRAY = "hammer_fenced_tool_call_array"
ARCH_XML_OBJECT = "arch_xml_tool_call_object"
CANONICAL_OBJECT = "canonical_object"


def native_dialect_for_model(model_id: str) -> str:
    """Return the native one-call format recommended by a known checkpoint."""

    normalized = model_id.lower()
    if "gpt-oss" in normalized or "qwen3.8-flash" in normalized:
        return OPENAI_NATIVE_TOOL_CALLS
    if (
        "arch-function" in normalized
        or "arch-agent" in normalized
        or "granite-4.1" in normalized
    ):
        return ARCH_XML_OBJECT
    if "hammer2.0" in normalized or "hammer2.1" in normalized:
        return HAMMER_FENCED_ARRAY
    if "xlam-2" in normalized:
        return TOOL_CALL_ARRAY
    return XLAM_V1_ENVELOPE


def json_root_for_dialect(dialect: str) -> str:
    if dialect in {
        TOOL_CALL_ARRAY,
        OPENAI_NATIVE_TOOL_CALLS,
        HAMMER_FENCED_ARRAY,
    }:
        return "array"
    if dialect in {XLAM_V1_ENVELOPE, ARCH_XML_OBJECT, CANONICAL_OBJECT}:
        return "object"
    raise ValueError(f"Unknown output dialect: {dialect}")
