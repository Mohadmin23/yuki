"""Yuki's eye — spawns ~/eyeoftruth in a small terminal window and reads
back what it sees.

The eye is a sibling Rust project (an animated TUI eye that face-tracks via
the webcam and describes the scene with a local VLM through Ollama). Flags
added for this bridge: --report <path> appends every oracle result as a JSON
line (the door OUT), --control <path> is the door IN — Yuki appends an emotion
word ("happy") to drive the eye's expression live, or "target:<thing>" /
"describe" to steer what it looks at without re-spawning the window. --target
<thing> starts in hunt mode, --exit-after <secs> self-cleans the window. We
open it in a split pane / small window (the user gets the full eye animation),
tail the report file, and hand Yuki the first real result to phrase.

Because the eye is persistent, it acts as Yuki's living FACE: after every
reply she writes her current mood here (set_emotion) and the eye emotes.

Everything is local: webcam → SeetaFace → Ollama gemma3:4b on localhost.
"""
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

EYE_BIN = Path(os.environ.get("EYE_BIN", Path.home() / "eyeoftruth/target/release/eyeoftruth"))
REPORT_PATH = Path("/tmp/yuki_eye.jsonl")
CONTROL_PATH = Path("/tmp/yuki_emote.txt")  # door IN: moods + look commands
EYE_WINDOW_SECS = 1800  # the eye is Yuki's face now — keep it open for the session,
                        # with a 30-min safety self-clean if she forgets to close it
WAIT_SECS = 45          # first Gemma answer can take ~15-20s on a cold load

# The eye's six expression states (see eyeoftruth/src/eye.rs). Yuki picks one
# per reply via an [EMOTE: word] tag; anything outside this set is ignored.
EMOTIONS = ("neutral", "happy", "sad", "energy", "surprised", "shocked")


def _ollama_ready() -> str | None:
    """Return None if Ollama + a vision model look usable, else a friendly fix."""
    from urllib.request import urlopen

    base = os.environ.get("EYE_OLLAMA", "http://127.0.0.1:11434")
    try:
        with urlopen(f"{base}/api/tags", timeout=2) as r:
            json.load(r)
    except Exception:
        return ("My eye needs Ollama running and it's offline — "
                "tell the user to start it: open -a Ollama")
    return None


def _open_eye_window(cmd: str) -> str | None:
    """Show the eye next to the chat. Returns an error string or None.

    Preferred: a split pane INSIDE the current iTerm window — a separate
    window lands on another macOS Space when the TUI runs fullscreen, which
    looks like the whole TUI going black. The pane closes itself when the
    eye exits (exec + --exit-after) and keyboard focus stays on the chat.
    Fallbacks: small pixel-bounds window at the top-right corner."""
    split = f'''
    tell application "iTerm"
        tell current window
            set origSession to current session
            tell origSession to set eyeSession to (split vertically with default profile)
            tell eyeSession to write text "exec {cmd}"
            tell origSession to select
        end tell
    end tell'''
    corner = '''
    tell application "Finder" to set sb to bounds of window of desktop
    set screenW to item 3 of sb'''
    iterm = f'''{corner}
    tell application "iTerm"
        set w to (create window with default profile)
        set bounds of w to {{screenW - 520, 40, screenW - 20, 400}}
        tell current session of w
            write text "exec {cmd}"
        end tell
    end tell'''
    terminal = f'''{corner}
    tell application "Terminal"
        do script "{cmd}"
        set bounds of front window to {{screenW - 520, 40, screenW - 20, 400}}
    end tell'''
    in_iterm = os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
    scripts = ([split] if in_iterm else []) + [iterm, terminal]
    for script in scripts:
        try:
            done = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=15,
            )
            if done.returncode == 0:
                return None
        except Exception:
            continue
    return "couldn't open a terminal window for the eye"


def eye_is_open() -> bool:
    """True if an eyeoftruth process is currently running (Yuki's face is up)."""
    try:
        r = subprocess.run(["pgrep", "-f", "eyeoftruth"],
                           capture_output=True, timeout=2)
        return r.returncode == 0
    except Exception:
        return False


def _send_control(line: str) -> bool:
    """Append a control line for the live eye. No-op when no eye is running so
    we never leave stale commands lying around for the next session."""
    if not eye_is_open():
        return False
    try:
        with CONTROL_PATH.open("a") as f:
            f.write(line.strip() + "\n")
        return True
    except OSError:
        return False


def set_emotion(emotion: str) -> bool:
    """Drive the eye's expression. Called after every reply with Yuki's mood;
    silently does nothing if the mood is unknown or her eye is closed."""
    word = (emotion or "").strip().lower()
    if word not in EMOTIONS:
        return False
    return _send_control(word)


# Keyword/emoji fallback for when a weak model forgets the [EMOTE:] tag.
# Priority order: punchy reactions first, then the broad moods. A miss returns
# None on purpose — better to hold the last mood than reset to neutral every
# time she says something flat.
_MOOD_HINTS = (
    ("shocked", ("oh my god", "what?!", "no way", "wtf", "horrifying",
                 "terrifying", "can't believe", "😱", "😨", "😰")),
    ("surprised", ("whoa", "woah", "wait what", "didn't expect", "no wayy",
                   "really?!", "huh?!", "?!", "😲", "😳", "🤯")),
    ("energy", ("let's go", "omg yes", "so excited", "so hyped", "can't wait",
                "yesss", "woohoo", "!!!", "🔥", "⚡", "🎉")),
    ("sad", ("i'm sorry", "so sorry", "that sucks", "oh no", "aww", "unfortunately",
             "i miss", "makes me sad", "😢", "😞", "😔", "💔")),
    ("happy", ("love", "glad", "yay", "haha", "hehe", "so cute", "awesome",
               "happy", "♥", "💕", "🥰", "😊", "😄", "~")),
)


def guess_emotion(text: str) -> str | None:
    """Best-effort mood from her reply text when no explicit tag is present."""
    if not text:
        return None
    low = text.lower()
    for emotion, hints in _MOOD_HINTS:
        if any(h in low for h in hints):
            return emotion
    return None


def open_eye(duration: int = EYE_WINDOW_SECS, target: str = "") -> str | None:
    """Run checks and spawn the eye pane. Returns a friendly error or None."""
    if not EYE_BIN.exists():
        return (f"My eye isn't built yet — eyeoftruth binary missing at {EYE_BIN}. "
                "Tell the user: cargo build --release in ~/eyeoftruth")
    err = _ollama_ready()
    if err:
        return err
    REPORT_PATH.unlink(missing_ok=True)
    CONTROL_PATH.unlink(missing_ok=True)  # fresh control channel per session
    cmd = (f"{EYE_BIN} --report {REPORT_PATH} --control {CONTROL_PATH} "
           f"--exit-after {duration}")
    if target:
        cmd += f" --target {shlex.quote(target)}"
    return _open_eye_window(cmd)


def current_sighting(max_age: float = 25.0) -> str | None:
    """Latest thing the eye sees, if an eye session looks alive right now.

    Liveness = report-file freshness: the oracle appends every few seconds
    while running (slow Gemma answers can gap ~15s, hence the loose window).
    Used by autonomous mode so idle Yuki can comment on what she sees."""
    try:
        if time.time() - REPORT_PATH.stat().st_mtime > max_age:
            return None
        for raw in reversed(REPORT_PATH.read_text().splitlines()):
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            if line.get("mode") == "describe" and line.get("text"):
                return line["text"]
            if line.get("mode") == "track":
                if line.get("found"):
                    return f"the {line.get('target', 'target')} you're tracking, still in view"
                return None
    except OSError:
        return None
    return None


def tool_see(target: str = "") -> str:
    """Open (or reuse) the eye; return what it sees (or finds)."""
    target = (target or "").strip().strip('"').strip("'")

    if eye_is_open():
        # Yuki's face is already up — steer it through the control channel
        # rather than spawning a second window. Only count results that land
        # AFTER we ask, so we don't return a stale describe line.
        try:
            seen = len(REPORT_PATH.read_text().splitlines())
        except OSError:
            seen = 0
        _send_control(f"target:{target}" if target else "describe")
    else:
        err = open_eye(EYE_WINDOW_SECS, target)
        if err:
            return err if not err.startswith("couldn't") else f"Eye error: {err}"
        seen = 0

    # Tail the report file for the first meaningful line after `seen`.
    deadline = time.time() + WAIT_SECS
    while time.time() < deadline:
        time.sleep(0.5)
        try:
            lines = REPORT_PATH.read_text().splitlines()
        except OSError:
            continue
        for raw in lines[seen:]:
            seen += 1
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            mode = line.get("mode")
            if mode == "error":
                return f"Your eye opened but hit a problem: {line.get('text', 'unknown')}"
            if mode == "describe":
                # Instruction on its own line — the echo scrubber strips it
                # if a weak model parrots the result back verbatim.
                return (f'Your eye opened and saw: "{line.get("text", "")}".\n'
                        "React to what you see in your own words.")
            if mode == "track":
                if line.get("found"):
                    pct = round(float(line.get("size", 0)) * 100)
                    return (f"Your eye found the {target} and is tracking it "
                            f"(it fills about {pct}% of your view).\n"
                            "React in your own words.")
                return (f"Your eye looked but can't find any {target} right now.\n"
                        "Say so honestly.")
    return ("Your eye is open and looking, but the vision model hasn't answered "
            "yet (it may still be loading). The eye window stays open a minute — "
            "tell the user to hold the thing up and ask again.")
