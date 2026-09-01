"""Build one-shot xLAM dispatcher prompts. There is intentionally no chat state."""
from __future__ import annotations

import json
from collections.abc import Iterable

from .contracts import PromptPackage
from .dialects import (
    ARCH_XML_OBJECT,
    CANONICAL_OBJECT,
    HAMMER_FENCED_ARRAY,
    OPENAI_NATIVE_TOOL_CALLS,
    TOOL_CALL_ARRAY,
    XLAM_V1_ENVELOPE,
    json_root_for_dialect,
    native_dialect_for_model,
)
from .registry import ToolRegistry

TASK_INSTRUCTION = """You are a dedicated function dispatcher, not an assistant.
A tool is already needed. Select exactly one available tool and extract its arguments from the query.
Never answer the query, explain the decision, greet the user, or generate any text outside the required JSON.
If no offered tool can handle the query or a required argument is missing, return the documented reject form.
Stop immediately after the complete structured call."""


NATIVE_FORMAT_INSTRUCTION = """Return exactly one JSON object and no other text.
For a valid dispatch use:
{"tool_calls":[{"name":"tool_name","arguments":{"parameter":"value"}}]}
The tool_calls array must contain exactly one call.
If dispatch is impossible or a required argument is missing, use:
{"tool_calls":[]}
Never produce multiple calls."""


ARRAY_NATIVE_FORMAT_INSTRUCTION = """Return exactly one JSON array and no other text.
For a valid dispatch use:
[{"name":"tool_name","arguments":{"parameter":"value"}}]
The array must contain exactly one call.
If dispatch is impossible or a required argument is missing, use:
[]
Never produce multiple calls or Markdown fences."""


HAMMER_FORMAT_INSTRUCTION = """The output MUST strictly adhere to the following JSON format, and no explanatory text may be included.
Hammer's native Markdown wrapper is allowed:
```
[{"name":"tool_name","arguments":{"parameter":"value"}}]
```
The array must contain exactly one call. If dispatch is impossible or a required argument is missing, output an empty array. Never produce multiple calls."""


CANONICAL_FORMAT_INSTRUCTION = """Return exactly one JSON object and no other text.
For a valid dispatch use:
{"tool":"tool_name","arguments":{"parameter":"value"}}
If dispatch is impossible or a required argument is missing, use:
{"tool":"no_tool","arguments":{}}
Never produce multiple calls."""


ARCH_FORMAT_INSTRUCTION = """For the function call, return exactly one JSON object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name":"tool_name","arguments":{"parameter":"value"}}
</tool_call>
Call exactly one function. Never answer the query or include explanatory text."""


def build_prompt(
    registry: ToolRegistry,
    request: str,
    names: Iterable[str],
    output_mode: str = "native",
    model_id: str = "Salesforce/xLAM-1b-fc-r",
    *,
    task_instruction: str | None = None,
    hammer_format_instruction: str | None = None,
) -> PromptPackage:
    if not request.strip():
        raise ValueError("The delegated request must not be empty")
    names = tuple(names)
    if output_mode == "native":
        dialect = native_dialect_for_model(model_id)
    elif output_mode == "canonical":
        dialect = CANONICAL_OBJECT
    else:
        raise ValueError(f"Unknown output mode: {output_mode}")
    selected_task_instruction = task_instruction or TASK_INSTRUCTION
    selected_hammer_format = hammer_format_instruction or HAMMER_FORMAT_INSTRUCTION

    if dialect == HAMMER_FENCED_ARRAY:
        schemas = registry.hammer_schemas(names)
        tools_json = json.dumps(schemas, ensure_ascii=False)
        if "hammer2.0" in model_id.lower():
            content = (
                "[BEGIN OF TASK INSTRUCTION]\n"
                f"{selected_task_instruction}\n"
                "[END OF TASK INSTRUCTION]\n\n"
                "[BEGIN OF AVAILABLE TOOLS]\n"
                f"{tools_json}\n"
                "[END OF AVAILABLE TOOLS]\n\n"
                "[BEGIN OF FORMAT INSTRUCTION]\n"
                f"{selected_hammer_format}\n"
                "[END OF FORMAT INSTRUCTION]\n\n"
                "[BEGIN OF QUERY]\n"
                f"{request.strip()}\n"
                "[END OF QUERY]"
            )
            return PromptPackage(
                content=content,
                schemas_sent=schemas,
                output_schema=registry.output_schema(names, output_mode, dialect),
                output_mode=output_mode,
                native_dialect=dialect,
                json_root=json_root_for_dialect(dialect),
            )
        rendered = (
            "<|im_start|>system\n"
            "You are a dedicated function dispatcher, not an assistant."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "[BEGIN OF TASK INSTRUCTION]\n"
            f"{selected_task_instruction}\n"
            "[END OF TASK INSTRUCTION]\n\n"
            "[BEGIN OF AVAILABLE TOOLS]\n"
            f"{tools_json}\n"
            "[END OF AVAILABLE TOOLS]\n\n"
            "[BEGIN OF FORMAT INSTRUCTION]\n"
            f"{selected_hammer_format}\n"
            "[END OF FORMAT INSTRUCTION]\n\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{request.strip()}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return PromptPackage(
            content=rendered,
            schemas_sent=schemas,
            output_schema=registry.output_schema(names, output_mode, dialect),
            output_mode=output_mode,
            native_dialect=dialect,
            json_root=json_root_for_dialect(dialect),
            pre_rendered=True,
        )

    if dialect == ARCH_XML_OBJECT:
        schemas = registry.strict_openai_schemas(names)
        tool_text = "\n".join(
            json.dumps(schema, ensure_ascii=False) for schema in schemas
        )
        system_content = (
            "You are a dedicated function dispatcher, not an assistant.\n\n"
            "# Tools\n\n"
            "A tool is already needed. Call exactly one function for the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tool_text}\n"
            "</tools>\n\n"
            f"{ARCH_FORMAT_INSTRUCTION}"
        )
        return PromptPackage(
            content=request.strip(),
            schemas_sent=schemas,
            output_schema=registry.output_schema(names, output_mode, dialect),
            output_mode=output_mode,
            native_dialect=dialect,
            json_root=json_root_for_dialect(dialect),
            system_content=system_content,
        )

    if dialect == OPENAI_NATIVE_TOOL_CALLS:
        schemas = registry.strict_openai_schemas(names)
        return PromptPackage(
            content=request.strip(),
            schemas_sent=schemas,
            output_schema=registry.output_schema(names, output_mode, dialect),
            output_mode=output_mode,
            native_dialect=dialect,
            json_root=json_root_for_dialect(dialect),
            system_content=selected_task_instruction,
            chat_template_tools=schemas,
        )

    if dialect == TOOL_CALL_ARRAY:
        schemas = registry.strict_openai_schemas(names)
        return PromptPackage(
            content=request.strip(),
            schemas_sent=schemas,
            output_schema=registry.output_schema(names, output_mode, dialect),
            output_mode=output_mode,
            native_dialect=dialect,
            json_root=json_root_for_dialect(dialect),
            system_content=(
                f"{selected_task_instruction}\n\n{ARRAY_NATIVE_FORMAT_INSTRUCTION}"
            ),
            chat_template_tools=schemas,
        )

    schemas = registry.xlam_schemas(names)
    format_instruction = (
        NATIVE_FORMAT_INSTRUCTION
        if dialect == XLAM_V1_ENVELOPE
        else CANONICAL_FORMAT_INSTRUCTION
    )

    tools_json = json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    content = (
        "[BEGIN OF TASK INSTRUCTION]\n"
        f"{selected_task_instruction}\n"
        "[END OF TASK INSTRUCTION]\n\n"
        "[BEGIN OF AVAILABLE TOOLS]\n"
        f"{tools_json}\n"
        "[END OF AVAILABLE TOOLS]\n\n"
        "[BEGIN OF FORMAT INSTRUCTION]\n"
        f"{format_instruction}\n"
        "[END OF FORMAT INSTRUCTION]\n\n"
        "[BEGIN OF QUERY]\n"
        f"{request.strip()}\n"
        "[END OF QUERY]"
    )
    return PromptPackage(
        content=content,
        schemas_sent=schemas,
        output_schema=registry.output_schema(names, output_mode, dialect),
        output_mode=output_mode,
        native_dialect=dialect,
        json_root=json_root_for_dialect(dialect),
    )
