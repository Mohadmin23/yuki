"""Ask the other Claude (running in a neighboring tmux pane) a question.

Sends the question via tmux's paste-buffer with bracketed-paste enabled
(`paste-buffer -p`), which Ink/React TUIs like Claude Code accept reliably —
plain `send-keys -l` writes raw bytes to the pty that Ink often swallows.

After sending, polls `capture-pane` until output has been stable for
_STABLE_SEC seconds, then returns whatever appeared after the pre-send
snapshot. Yuki's loop sees the reply as a tool result and paraphrases it
back to the user.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

# Hard ceiling on how long to wait for a Claude reply.
_MAX_WAIT_SEC = 120.0
# Pane output must stay unchanged this long before we call it done.
# Generous enough to outlast cursor-blink redraws (~0.5s typical) and
# Claude Code's intermittent status-line refreshes.
_STABLE_SEC = 5.0
_POLL_SEC = 0.5
# Settle time after pasting before pressing Enter — the paste needs a beat
# to land in Ink's input buffer or Enter submits empty.
_PASTE_SETTLE_SEC = 0.3

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
_BUFFER_NAME = "yuki_ask"
_DEBUG_LOG = "/tmp/yuki_ask_claude.log"


def _log(msg: str) -> None:
    try:
        with open(_DEBUG_LOG, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass

# Patterns that, when matched against a pane's current command, identify it
# as a Claude Code session. Claude Code titles its process to its version
# (e.g. "2.1.147") and is typically a node subprocess.
_CLAUDE_CMD_RE = re.compile(r"^(claude|node|\d+\.\d+\.\d+)$", re.IGNORECASE)


def _autodetect_claude_pane(current_pane: str) -> str | None:
    """Find the tmux pane running Claude Code, excluding the caller's pane.
    Returns the %pane_id of the match, or None if zero or multiple matches."""
    r = subprocess.run(
        ["tmux", "list-panes", "-a", "-F",
         "#{pane_id}\t#{pane_current_command}\t#{pane_title}"],
        capture_output=True, text=True,
    )
    _log(f"list-panes rc={r.returncode} stdout={r.stdout!r}")
    if r.returncode != 0:
        return None
    matches = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        pane_id, cmd = parts[0], parts[1]
        title = parts[2] if len(parts) > 2 else ""
        if pane_id == current_pane:
            continue
        if _CLAUDE_CMD_RE.match(cmd) or "claude" in title.lower():
            matches.append(pane_id)
    _log(f"matches={matches}")
    return matches[0] if len(matches) == 1 else None


def _capture(target: str) -> str:
    """Capture the pane including a chunk of scrollback so long replies
    don't get truncated at the visible-area boundary."""
    r = subprocess.run(
        ["tmux", "capture-pane", "-p", "-J", "-S", "-200", "-t", target],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return ""
    return _ANSI_RE.sub("", r.stdout)


def _strip_trailing_blanks(text: str) -> str:
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _send_question(target: str, text: str) -> None:
    """Deliver text to the target pane via send-keys -l (literal). Bracketed
    paste (paste-buffer -p) is rejected by Claude Code's Ink renderer, while
    plain send-keys works."""
    subprocess.run(
        ["tmux", "send-keys", "-t", target, "-l", text],
        check=True, capture_output=True, text=True,
    )
    time.sleep(_PASTE_SETTLE_SEC)
    subprocess.run(
        ["tmux", "send-keys", "-t", target, "Enter"],
        check=True, capture_output=True, text=True,
    )


def tool_ask_claude(question):
    q = (question or "").strip()
    _log(f"--- new call --- question={q!r} TMUX={os.environ.get('TMUX')!r} YUKI_CLAUDE_PANE={os.environ.get('YUKI_CLAUDE_PANE')!r}")
    if not q:
        _log("FAIL: empty question")
        return "ASK_CLAUDE_ERROR: no question provided. Tell the user I need something to ask Claude."

    if not shutil.which("tmux"):
        return "ASK_CLAUDE_ERROR: tmux not installed. Tell the user to brew install tmux."
    if not os.environ.get("TMUX"):
        return "ASK_CLAUDE_ERROR: not inside tmux. Tell the user to launch via ./yuki-tmux."

    # Find Yuki's own pane so we never accidentally target ourselves.
    self_pane = subprocess.run(
        ["tmux", "display", "-p", "#{pane_id}"],
        capture_output=True, text=True,
    ).stdout.strip()

    _log(f"self_pane={self_pane!r}")
    target = os.environ.get("YUKI_CLAUDE_PANE", "").strip()
    if not target:
        detected = _autodetect_claude_pane(self_pane)
        _log(f"autodetect result: {detected!r}")
        if detected:
            target = detected
        else:
            return (
                "ASK_CLAUDE_ERROR: couldn't auto-detect a Claude Code pane "
                "and YUKI_CLAUDE_PANE isn't set. Tell the user to either "
                "(a) start Claude Code in another tmux pane, or (b) export "
                "YUKI_CLAUDE_PANE=%N before launching Yuki."
            )

    panes = subprocess.run(
        ["tmux", "list-panes", "-a", "-F", "#{pane_id} #{window_name}.#{pane_index}"],
        capture_output=True, text=True,
    )
    if panes.returncode != 0:
        return f"ASK_CLAUDE_ERROR: couldn't list tmux panes: {panes.stderr.strip()}"
    pane_lines = panes.stdout.strip().splitlines()
    pane_ids = {ln.split()[0] for ln in pane_lines}
    pane_names = {ln.split()[1] for ln in pane_lines if len(ln.split()) > 1}
    if target.startswith("%"):
        if target not in pane_ids:
            return (
                f"ASK_CLAUDE_ERROR: pane id {target} no longer exists. "
                f"Available: {sorted(pane_ids)}. Tell the user the Claude pane "
                f"probably got closed — restart via ./yuki-tmux."
            )
    else:
        if target not in pane_names:
            return (
                f"ASK_CLAUDE_ERROR: pane name '{target}' not found. "
                f"Available names: {sorted(pane_names)}. Tell the user to set "
                f"YUKI_CLAUDE_PANE to one of these."
            )

    before = _strip_trailing_blanks(_capture(target))
    if not before:
        return (
            f"ASK_CLAUDE_ERROR: target pane {target} returned no content — "
            f"is Claude Code actually running in it? Tell the user to check that pane."
        )

    try:
        _send_question(target, q)
    except subprocess.CalledProcessError as e:
        return f"ASK_CLAUDE_ERROR: failed to send to pane {target}: {e.stderr.strip()}"

    deadline = time.monotonic() + _MAX_WAIT_SEC
    last_snap = before
    last_change = time.monotonic()
    saw_real_change = False
    while time.monotonic() < deadline:
        time.sleep(_POLL_SEC)
        snap = _strip_trailing_blanks(_capture(target))
        if snap != last_snap:
            last_snap = snap
            last_change = time.monotonic()
            if snap != before:
                saw_real_change = True
            continue
        if saw_real_change and time.monotonic() - last_change >= _STABLE_SEC:
            break
    else:
        if not saw_real_change:
            return (
                f"ASK_CLAUDE_ERROR: pane {target} never changed after sending the question. "
                f"Tell the user: the keystrokes didn't reach Claude Code. "
                f"Most likely YUKI_CLAUDE_PANE points at the wrong pane "
                f"(currently {target}). They can run `tmux display -p '#{{pane_id}}'` "
                f"in the Claude pane to see the right id."
            )
        return f"ASK_CLAUDE_ERROR: Claude still streaming after {int(_MAX_WAIT_SEC)}s — gave up waiting."

    after = last_snap
    if after.startswith(before):
        delta = after[len(before):].lstrip("\n")
    else:
        delta = after

    delta = delta.strip()
    if not delta:
        return f"ASK_CLAUDE_ERROR: captured an empty delta from pane {target} — Claude may have erased its own output."

    # Strip the noisy stuff Claude Code's TUI renders that has no semantic
    # value once captured as text: line-number prefixes, box drawing,
    # repeated banner glyphs from animations.
    cleaned_lines = []
    for line in delta.splitlines():
        s = line.rstrip()
        if not s:
            continue
        # Code-viewer line numbers: "   42 ", "  113 ", etc.
        s = re.sub(r"^\s{0,4}\d{1,4}\s{2}", "", s)
        # Drop pure box-drawing / banner rows (no letters or digits at all).
        if not re.search(r"[A-Za-z0-9]", s):
            continue
        cleaned_lines.append(s)
    cleaned = "\n".join(cleaned_lines)
    if len(cleaned) > 800:
        cleaned = cleaned[-800:]

    # Wrap the result with instructions to Yuki's model so she gives her
    # OWN thoughts in her warm voice instead of pasting raw output. Also
    # surface any permission prompt explicitly so she can suggest the
    # right approval keystroke.
    has_permission_prompt = bool(re.search(
        r"\b(do you want to|allow|y/n|\b1\.\s*yes\b)", cleaned, re.IGNORECASE
    ))
    instruction = (
        "TOOL RESULT — what Claude just did/said is below. "
        "Summarize in YOUR OWN warm voice in 1-3 short sentences. "
        "Do NOT paste code, do NOT paste animation frames, do NOT echo this verbatim. "
    )
    if has_permission_prompt:
        instruction += (
            "Claude is asking permission. Tell the user they can approve by saying "
            "'tell claude 1' (yes) or 'tell claude 3' (no), or by clicking into the Claude pane themselves. "
        )
    return f"{instruction}\n---\n{cleaned}"
