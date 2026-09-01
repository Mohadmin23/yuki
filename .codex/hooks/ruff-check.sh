#!/usr/bin/env bash
# PostToolUse hook: after Claude edits a Python file, run ruff on just that
# file and feed any findings back into Claude's context so they get fixed
# immediately (project convention: keep linting clean).
#
# Scoped to Pyflakes (`--select F`): unused imports, unused/undefined names,
# redefinitions, f-string issues — the high-signal stuff. Intentional style
# choices (E402 lazy imports, etc.) are NOT nagged. Widen the --select if you
# want stricter checks. Receives the PostToolUse JSON payload on stdin.

f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' 2>/dev/null)

case "$f" in
  *.py) ;;
  *) exit 0 ;;           # not Python — nothing to do
esac
[ -f "$f" ] || exit 0

out=$(uvx ruff check --select F "$f" 2>&1)
[ $? -eq 0 ] && exit 0   # clean — stay silent

jq -n --arg f "$f" --arg o "$out" \
  '{hookSpecificOutput: {hookEventName: "PostToolUse",
    additionalContext: ("Ruff flagged the file you just edited (" + $f
      + "). Fix these now — project convention is clean linting:\n" + $o)}}'
exit 0
