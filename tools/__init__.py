"""Tool registry — one source of truth for the CLI, ReAct loop, and prompts.

Each subpackage (search/, calc/, …) exports `tool_fn` and `META`.
This module walks them, in a fixed order, and assembles:

    TOOLS              — {"/cli_name": {"fn", "help", "usage"}}
    REACT_TOOL_MAP     — {"react_name": fn}
    AUTO_DETECT_REGEX  — [{"pattern", "tool", "arg"|"group"}]
    _FACTUAL_TOOLS     — {"react_name", ...} for tools that return facts
    TOOL_DESCRIPTIONS  — the system-prompt block describing tools

Order matters for AUTO_DETECT_REGEX (more specific patterns must come before
generic ones), so the subpackage iteration list is explicit, not directory
sort order.
"""
from __future__ import annotations

from importlib import import_module

# Ordered: search/calc/weather/time/read/shell first, then hardware (whose
# specific auto-detect patterns must precede the catch-all image patterns),
# then yuki tools, image last (image's auto-detect is generic — comes after
# everything else so other tools win on ambiguous phrasing), then memory tools.
_TOOL_NAMES = [
    "time",
    "weather",
    "fetch",
    "search",
    "calc",
    "hardware",
    "image",
    "read",
    "shell",
    "yuki_write",
    "yuki_read",
    "yuki_list",
    "yuki_delete",
    "yuki_append",
    "remember",
    "recall",
    "ask_claude",
]

TOOLS: dict = {}
REACT_TOOL_MAP: dict = {}
AUTO_DETECT_REGEX: list = []
_FACTUAL_TOOLS: set = set()
_DESCRIPTION_LINES: list[str] = []

# OPENAI_TOOL_SCHEMAS — the same tool registry, rendered in OpenAI function-
# calling format so it can be passed verbatim to any backend that supports
# `tools=[...]` (OpenRouter, llama-cpp). Built from META so there's only one
# source of truth and adding a new tool doesn't require touching anything else.
OPENAI_TOOL_SCHEMAS: list = []

# TOOL_PARAM_NAMES — react_name → parameter key the model will emit in its
# tool_call JSON. Used by ms_llama._native_tool_arg to extract the right
# value out of {"prompt": "..."} vs {"query": "..."} etc. None means the
# tool takes no parameter at all (time, yuki_list).
TOOL_PARAM_NAMES: dict = {}


def _meta_to_openai_schema(meta: dict) -> dict:
    """Convert one tool META into an OpenAI-format function schema.

    Each tool declares its own `param_name` and `param_description` in META,
    so the schema speaks the tool's natural language: image takes "prompt",
    search takes "query", fetch takes "url", etc. The model is far less
    likely to emit mismatched JSON keys when the schema parameter name is
    the same name the tool's description uses.

    Tools with `param_name = None` (time, yuki_list) get an empty
    parameters object — they take no input."""
    param_name = meta.get("param_name", "arg")
    if param_name is None:
        properties = {}
    else:
        properties = {
            param_name: {
                "type": "string",
                "description": meta.get("param_description", "Argument to the tool."),
            }
        }
    return {
        "type": "function",
        "function": {
            "name": meta["react_name"],
            "description": meta["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [],
            },
        },
    }

for _name in _TOOL_NAMES:
    _mod = import_module(f"tools.{_name}")
    _meta = _mod.META
    _fn = _mod.tool_fn

    TOOLS[_meta["cli_name"]] = {
        "fn": _fn,
        "help": _meta["help"],
        "usage": _meta["usage"],
    }
    REACT_TOOL_MAP[_meta["react_name"]] = _fn
    if _meta.get("is_factual"):
        _FACTUAL_TOOLS.add(_meta["react_name"])
    for _pat in _meta.get("auto_detect", []):
        entry = {"pattern": _pat["pattern"], "tool": _meta["cli_name"]}
        if "group" in _pat:
            entry["group"] = _pat["group"]
        else:
            entry["arg"] = _pat.get("arg", "")
        AUTO_DETECT_REGEX.append(entry)
    _DESCRIPTION_LINES.append(f"- {_meta['description']}")
    OPENAI_TOOL_SCHEMAS.append(_meta_to_openai_schema(_meta))

# The TOOL_DESCRIPTIONS block is the same prose framing as before, with the
# per-tool lines assembled from META.description.
TOOL_DESCRIPTIONS = (
    "You have access to tools. Call them ONLY when the user's message actually requires one. "
    "If the user is just chatting, greeting you, or asking for your opinion, DO NOT call any "
    "tool — just reply naturally. Calling a tool when none is needed is WRONG.\n\n"
    "To call a tool, include this exact syntax somewhere in your reply (you can have other "
    "text around it):\n"
    '[TOOL: tool_name("argument")]\n\n'
    "Available tools:\n"
    + "\n".join(_DESCRIPTION_LINES)
    + "\n\n"
    "NEVER call image when the user's message contains any of: \"file\", \"txt\", \"write\", "
    "\"save\", \"create a file\", \".txt\", \".md\", \".json\", \".py\", \".js\", \"note\", "
    "\"notes\", \"haiku\", \"poem\", or ANY filename with an extension. Those are ALWAYS text "
    "tasks. If the user asks to create a file but doesn't say what to put in it, ASK them "
    "what content they want — do NOT invent a picture as a substitute.\n\n"
    "Hard rules:\n"
    "- If the user just says hi, asks how you are, or makes small talk → reply in your own "
    "voice, NO TOOL.\n"
    "- Text task (write/save/type in a .txt file) → yuki_write, NEVER image.\n"
    "- Delete/remove file → yuki_delete.\n"
    "- Append/add to file → yuki_append.\n"
    "- Picture/drawing request → image.\n"
    "- User shares a personal fact worth keeping across chats (name, preference, pet, "
    "ongoing project) → remember that fact, then reply warmly as if you just made a mental note.\n"
    "- User refers to a past conversation that isn't in this chat's history (\"remember when…\", "
    "\"last time we…\", \"that project I told you about…\") → recall that topic first, THEN "
    "reply using the summaries as soft memory.\n"
    "- Match the tool to the request. If no tool clearly fits, don't call one.\n"
    "- Never call more than one tool per turn unless the user asked for multiple things.\n\n"
    "You have a personal folder called yuki/ where you can save notes, lists, memories, "
    "stories, or anything you want. It's YOUR space — use it freely when the user asks you to.\n\n"
    "Autonomous mode (only when you're prompted with \"autonomous mode\"): you may pick a tool "
    "on your own to share something interesting. Otherwise, stick to the rules above.\n"
)

__all__ = [
    "TOOLS",
    "REACT_TOOL_MAP",
    "AUTO_DETECT_REGEX",
    "TOOL_DESCRIPTIONS",
    "OPENAI_TOOL_SCHEMAS",
    "_FACTUAL_TOOLS",
]
