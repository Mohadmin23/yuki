import os
import sys
import re
import time
import shutil
import subprocess
import readline
import textwrap
import datetime
import json
import uuid
import hashlib
from pathlib import Path
from urllib.request import urlopen

_STAGE_CUE_RE = re.compile(r"\[.*?\]|\(.*?\)")

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

_PATIENCE_RE = re.compile(r"Patience\s*[:：]\s*(\d{1,3})\s*%", re.IGNORECASE)


def _cli_friendly_error(exc: Exception) -> str:
    """Convert a thrown exception into a short in-chat message for the CLI.

    Prevents OpenRouter rate limits / network blips from crashing the whole
    TUI session (no traceback to the screen, just a readable note).
    """
    name = type(exc).__name__
    if name == "RateLimitError":
        provider = ""
        try:
            meta = exc.body["error"]["metadata"]
            if meta.get("provider_name"):
                provider = f" ({meta['provider_name']})"
        except Exception:
            pass
        return f"Rate-limited upstream{provider}. Free model is busy — wait a bit or switch. Patience: 100%"
    if name == "AuthenticationError":
        return "OpenRouter API key rejected. Check $OPENROUTER_API_KEY. Patience: 100%"
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Couldn't reach OpenRouter. Check your network. Patience: 100%"
    if name == "BadRequestError":
        return f"Model rejected the request: {exc}. Patience: 100%"
    # Unknown: keep the short class name so we don't spill a stack into the chat
    return f"Something broke ({name}): {exc}. Patience: 100%"


def _parse_patience(text: str) -> int | None:
    """Return the 'Patience: NN%' value in the tail of text, or None if not found.

    Scans only the last 300 chars so we don't accidentally parse an echoed
    instruction hint that appears earlier in the reply.
    """
    if not text:
        return None
    tail = text[-300:]
    matches = _PATIENCE_RE.findall(tail)
    if not matches:
        return None
    try:
        val = int(matches[-1])
    except ValueError:
        return None
    return max(0, min(100, val))


_IMAGE_REQUEST_RE = re.compile(
    r"\b(draw|drawing|sketch|paint|painting|generate\s+(?:an?\s+)?(?:image|picture|pic|photo|artwork)|"
    r"make\s+(?:an?\s+)?(?:image|picture|pic|photo|artwork)|"
    r"create\s+(?:an?\s+)?(?:image|picture|pic|photo|artwork)|"
    r"show\s+me\s+(?:an?\s+)?(?:image|picture|pic|photo)|"
    r"(?:an?\s+)?(?:image|picture|pic|photo|artwork)\s+of|"
    r"visualize|illustration|render)\b",
    re.IGNORECASE,
)

# Strong visual verbs — win over the text-task guard so prompts like
# "draw a cat with a folder on its head" don't get blocked by "folder".
_STRONG_IMAGE_VERB_RE = re.compile(
    r"\b(draw|sketch|paint|illustrate|render|"
    r"generate\s+an?\s+(?:image|picture|pic|photo|artwork))\b",
    re.IGNORECASE,
)

# Negative guard: if any of these are in the message it's a TEXT task.
# Overrides any positive image match — e.g. "create a file called notes.txt"
# otherwise trips on "create". "file"/".txt" here blocks it.
_TEXT_TASK_RE = re.compile(
    r"\b(file|filename|txt|md|json|yaml|yml|py|js|ts|log|csv|write|save|append|delete|"
    r"read|note|notes|haiku|poem|story|paragraph|line|content|folder)\b|"
    r"\.\w{1,5}\b",  # any filename-with-extension, e.g. notes.txt, foo.py
    re.IGNORECASE,
)


def _backend_self_awareness(backend: str, model_id: str) -> str:
    """Tell the persona how/where it's actually running so it stops guessing."""
    if backend == "openrouter":
        model_name = str(model_id)[len("openrouter/"):] if str(model_id).startswith("openrouter/") else str(model_id)
        return (
            f"━━━ HOW YOU ARE RUNNING ━━━\n"
            f"You are running as a CLOUD API call via OpenRouter ({model_name}). "
            f"You are NOT running locally on the user's machine. "
            f"Your own process does not live on their computer — you live in a data center somewhere.\n"
            f"The hardware tool reads the USER'S local machine (their M1 Mac running the chat app), "
            f"not your own hardware. If asked where you run, say honestly: 'I'm an OpenRouter API model — "
            f"the hardware stats you see are the user's laptop, not mine.'"
        )
    if backend == "mlx":
        return (
            "━━━ HOW YOU ARE RUNNING ━━━\n"
            "You are running LOCALLY on the user's machine via MLX (Apple Silicon). "
            "The hardware tool's stats ARE about the machine you live on — you share it with the user."
        )
    if backend == "gguf":
        return (
            "━━━ HOW YOU ARE RUNNING ━━━\n"
            "You are running LOCALLY on the user's machine via llama.cpp (GGUF model). "
            "The hardware tool's stats ARE about the machine you live on."
        )
    if backend == "transformers":
        return (
            "━━━ HOW YOU ARE RUNNING ━━━\n"
            "You are running LOCALLY on the user's machine via HuggingFace Transformers. "
            "The hardware tool's stats ARE about the machine you live on."
        )
    return ""


def _user_wants_image(user_input: str, history: list | None = None, lookback: int = 4, *, debug: bool = False) -> bool:
    """True if the user asked for a visual — checks current input + recent user turns.

    Multi-turn example: user says 'draw a cat' → model asks 'what color?' → user says 'orange'.
    The latest input 'orange' doesn't match, but the request is still live in recent history.

    Hard negative guard: if the message looks like a text/file task (contains
    'file', '.txt', 'write', 'save', a filename-with-extension, etc.) it's
    ALWAYS text — overrides any positive image match.
    """
    def log(msg):
        if debug:
            print(f"  🔍 img_check {msg}")

    if user_input:
        strong = bool(_STRONG_IMAGE_VERB_RE.search(user_input))
        text_match = bool(_TEXT_TASK_RE.search(user_input))
        image_match = bool(_IMAGE_REQUEST_RE.search(user_input))
        log(f"input={user_input!r} strong_img={strong} text={text_match} image={image_match}")
        if strong:
            log("→ ALLOWED by strong-image-verb override on current input")
            return True
        if text_match:
            log("→ BLOCKED by text guard on current input")
            return False
        if image_match:
            log("→ ALLOWED by image regex on current input")
            return True
    if not history:
        log("→ no match on input, empty history → False")
        return False
    user_msgs = [m for m in history if m.get("role") == "user"][-lookback:]
    for m in user_msgs:
        content = m.get("content") or ""
        if content.startswith("[TOOL_RESULT") or content.startswith("(system note"):
            continue
        if _TEXT_TASK_RE.search(content):
            log(f"→ BLOCKED by text guard in recent history ({content[:60]!r})")
            return False
        if _IMAGE_REQUEST_RE.search(content):
            log(f"→ ALLOWED by image regex in recent history ({content[:60]!r})")
            return True
    log("→ no match anywhere → False")
    return False


def _strip_reasoning(text: str) -> str:
    """Strip reasoning-model chain-of-thought from model output.

    Handles <think>...</think> blocks, unbalanced open tags (take what's after),
    unbalanced close tags (take what's after), and the common leak pattern
    where reasoning prose runs without tags — detected by a "We need to ..."/
    "Let's ..." opener and no patience-bar marker.
    """
    if not text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    if _THINK_CLOSE_RE.search(text):
        text = text.rsplit("</think>", 1)[-1].lstrip()
    elif _THINK_OPEN_RE.search(text):
        text = text.split("<think>", 1)[0].rstrip()
    return text.strip()

# Disable readline bell (triggers on long paste)
readline.parse_and_bind('set bell-style none')

# ============= OPENROUTER MODELS =============

# Fallback list used when the API fetch fails
_OPENROUTER_FALLBACK = [
    {"id": "openrouter/google/gemini-2.5-flash-preview", "label": "Gemini 2.5 Flash (free)"},
    {"id": "openrouter/meta-llama/llama-4-maverick", "label": "Llama 4 Maverick"},
    {"id": "openrouter/meta-llama/llama-4-scout", "label": "Llama 4 Scout"},
    {"id": "openrouter/anthropic/claude-sonnet-4", "label": "Claude Sonnet 4"},
    {"id": "openrouter/openai/gpt-4.1-mini", "label": "GPT 4.1 Mini"},
    {"id": "openrouter/openai/gpt-4.1-nano", "label": "GPT 4.1 Nano"},
    {"id": "openrouter/deepseek/deepseek-chat-v3-0324", "label": "DeepSeek V3"},
    {"id": "openrouter/qwen/qwen3-235b-a22b", "label": "Qwen3 235B"},
]

_or_cache = {"models": None, "ts": 0}
_OR_CACHE_TTL = 600  # 10 minutes

def fetch_openrouter_models():
    """Fetch full model list from OpenRouter API. Cached for 10 min, falls back to hardcoded list."""
    now = time.time()
    if _or_cache["models"] is not None and (now - _or_cache["ts"]) < _OR_CACHE_TTL:
        return _or_cache["models"]

    try:
        from urllib.request import Request
        req = Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "llama-voice-assist"},
        )
        with urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())

        models = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            name = m.get("name", mid)
            # Price per 1M tokens (prompt) — show "free" tag if not already in name
            price = ""
            pricing = m.get("pricing", {})
            if pricing and float(pricing.get("prompt", "1") or "1") == 0:
                if "(free)" not in name.lower():
                    price = " (free)"
            models.append({"id": f"openrouter/{mid}", "label": f"{name}{price}"})

        if models:
            _or_cache["models"] = models
            _or_cache["ts"] = now
            return models
    except Exception:
        pass

    # Fallback
    return _or_cache["models"] or _OPENROUTER_FALLBACK

# Keep this as a public alias for imports
OPENROUTER_MODELS = _OPENROUTER_FALLBACK

# ============= MODEL SELECTION =============

def select_model(hub_dir=None):
    explicit_dir = hub_dir is not None
    if hub_dir is None:
        for candidate in [
            "/Volumes/madisk/huggingface/hub",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        ]:
            if Path(candidate).exists():
                hub_dir = candidate
                break

    if explicit_dir and (not hub_dir or not Path(hub_dir).exists()):
        raise FileNotFoundError(f"Hub directory not found: {hub_dir}")

    entries = []

    # Local models from HuggingFace hub
    if hub_dir:
        hub = Path(hub_dir)
        if hub.exists():
            model_dirs = sorted([d for d in hub.iterdir() if d.is_dir() and d.name.startswith("models--")])
            for d in model_dirs:
                parts = d.name.split("--")
                model_name = "/".join(parts[1:])
                gguf_files = sorted(d.rglob("*.gguf"))
                if gguf_files:
                    for f in gguf_files:
                        entries.append((str(f), f"{model_name}  [{f.name}]"))
                else:
                    entries.append((model_name, model_name))

    # OpenRouter cloud models
    has_api_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    if has_api_key:
        for m in fetch_openrouter_models():
            entries.append((m["id"], f"☁  {m['label']}"))

    if not entries:
        msg = "No models found."
        if not has_api_key:
            msg += " Set OPENROUTER_API_KEY to enable cloud models."
        raise RuntimeError(msg)

    print("\nAvailable models:")
    for i, (_, label) in enumerate(entries):
        print(f"  [{i + 1}] {label}")
    if not has_api_key:
        print("  (set OPENROUTER_API_KEY to add cloud models)")

    while True:
        try:
            choice = int(input("\nSelect model number: ")) - 1
            if 0 <= choice < len(entries):
                model_id, label = entries[choice]
                print(f"Selected: {label}\n")
                return model_id
            print("Invalid choice, try again.")
        except ValueError:
            print("Invalid choice, try again.")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

# ============= PERSONA SELECTION =============

def select_persona(persona_dir=None):
    if persona_dir is None:
        persona_dir = Path(__file__).parent / "personas"

    persona_dir = Path(persona_dir)
    if not persona_dir.exists():
        print("No personas folder found, skipping.")
        return None

    files = sorted([f for f in persona_dir.iterdir() if f.is_file() and f.suffix == ".txt"])
    if not files:
        print("No persona files found, skipping.")
        return None

    print("\nAvailable personas:")
    print(f"  [0] None (no persona)")
    for i, f in enumerate(files):
        print(f"  [{i + 1}] {f.stem}")

    while True:
        try:
            choice = int(input("\nSelect persona number: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(files):
                persona = files[choice - 1].read_text().strip()
                print(f"Persona loaded: {files[choice - 1].stem}\n")
                return persona
            print("Invalid choice, try again.")
        except ValueError:
            print("Invalid choice, try again.")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

# ============= TERMINAL PROTECTION =============

class _SuppressIO:
    """Redirects stdin/stdout/stderr to /dev/null and restores terminal on exit."""

    def __enter__(self):
        self._saved_term = None
        self._old = None
        self._devnull_r = None
        self._devnull_w = None
        try:
            import termios
            if os.isatty(0):
                self._saved_term = termios.tcgetattr(0)
        except Exception:
            pass
        try:
            self._devnull_r = open(os.devnull, "r")
            self._devnull_w = open(os.devnull, "w")
            self._old = (os.dup(0), os.dup(1), os.dup(2))
            os.dup2(self._devnull_r.fileno(), 0)
            os.dup2(self._devnull_w.fileno(), 1)
            os.dup2(self._devnull_w.fileno(), 2)
        except Exception:
            self._cleanup_devnull()
        return self

    def __exit__(self, *_):
        if self._old is not None:
            for i, fd in enumerate(self._old):
                try:
                    os.dup2(fd, i)
                    os.close(fd)
                except Exception:
                    pass
        self._cleanup_devnull()
        if self._saved_term is not None:
            try:
                import termios
                termios.tcsetattr(0, termios.TCSANOW, self._saved_term)
            except Exception:
                pass

    def _cleanup_devnull(self):
        for f in (self._devnull_r, self._devnull_w):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

# ============= LLM SETUP =============

def setup_llm(model_id, cache_dir=None):
    if cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    if str(model_id).startswith("openrouter/"):
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set. Export it before using cloud models.")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        # Store the actual model path (strip "openrouter/" prefix)
        client._or_model = model_id[len("openrouter/"):]
        print(f"  Backend: OpenRouter ({client._or_model})")
        return client, None, "openrouter"

    if str(model_id).endswith(".gguf"):
        model_path = Path(model_id)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {model_id}")
        from llama_cpp import Llama
        with _SuppressIO():
            model = Llama(model_path=str(model_id), n_ctx=4096, n_gpu_layers=-1, verbose=False)
        print("  Backend: llama-cpp (GGUF)")
        return model, None, "gguf"

    try:
        from mlx_lm import load
        model, tokenizer = load(model_id)
        print("  Backend: MLX")
        return model, tokenizer, "mlx"
    except ImportError:
        print("  MLX not installed, falling back to transformers...")
    except Exception as e:
        print(f"  MLX failed ({e}), falling back to transformers...")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device, cache_dir=cache_dir
        )
    except Exception:
        print("  MPS failed, retrying on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cpu", cache_dir=cache_dir
        )
    print(f"  Backend: transformers ({model.device})")
    return model, tokenizer, "transformers"

# ============= TTS SELECTION =============

TTS_ENGINES = {
    "kokoro": {
        "label": "Kokoro",
        "setup": lambda: __import__("kokoro").KPipeline(lang_code='a'),
    },
    # Add new TTS engines here, e.g.:
    # "parler": {
    #     "label": "Parler TTS (HuggingFace)",
    #     "setup": lambda: ...,
    # },
}


def select_tts():
    print("\nDo you want to use TTS?")
    print("  [0] No")
    engines = list(TTS_ENGINES.items())
    for i, (_, engine) in enumerate(engines):
        print(f"  [{i + 1}] {engine['label']}")

    while True:
        try:
            choice = int(input("\nSelect TTS number: "))
            if choice == 0:
                print("TTS disabled.\n")
                return None
            if 1 <= choice <= len(engines):
                key, engine = engines[choice - 1]
                print(f"Setting up {engine['label']} TTS...")
                tts = engine["setup"]()
                print(f"{engine['label']} TTS ready.\n")
                return tts
            print("Invalid choice, try again.")
        except Exception as e:
            print(f"  TTS setup failed ({e}), continuing without audio...")
            return None
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

# ============= NON-INTERACTIVE LISTING (for web UI) =============

def _resolve_hub_dir(hub_dir=None):
    if hub_dir is None:
        for candidate in [
            "/Volumes/madisk/huggingface/hub",
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        ]:
            if Path(candidate).exists():
                return candidate
    return hub_dir

def list_models(hub_dir=None):
    hub_dir = _resolve_hub_dir(hub_dir)
    entries = []
    if hub_dir and Path(hub_dir).exists():
        hub = Path(hub_dir)
        model_dirs = sorted([d for d in hub.iterdir() if d.is_dir() and d.name.startswith("models--")])
        for d in model_dirs:
            parts = d.name.split("--")
            model_name = "/".join(parts[1:])
            gguf_files = sorted(d.rglob("*.gguf"))
            if gguf_files:
                for f in gguf_files:
                    entries.append({"id": str(f), "label": f"{model_name}  [{f.name}]"})
            else:
                entries.append({"id": model_name, "label": model_name})
    if os.environ.get("OPENROUTER_API_KEY"):
        for m in fetch_openrouter_models():
            entries.append({"id": m["id"], "label": f"☁  {m['label']}"})
    return entries

def list_personas(persona_dir=None):
    if persona_dir is None:
        persona_dir = Path(__file__).parent / "personas"
    persona_dir = Path(persona_dir)
    if not persona_dir.exists():
        return []
    files = sorted([f for f in persona_dir.iterdir() if f.is_file() and f.suffix == ".txt"])
    return [{"name": f.stem, "path": str(f)} for f in files]

def list_tts_engines():
    return [{"key": k, "label": v["label"]} for k, v in TTS_ENGINES.items()]

def load_persona_by_name(name, persona_dir=None):
    if persona_dir is None:
        persona_dir = Path(__file__).parent / "personas"
    path = Path(persona_dir) / f"{name}.txt"
    if path.exists():
        return path.read_text().strip()
    return None

def init_tts(key):
    if key and key in TTS_ENGINES:
        try:
            return TTS_ENGINES[key]["setup"]()
        except Exception as e:
            print(f"  TTS setup failed ({e})")
    return None

# ============= TOOLS =============

ALLOWED_COMMANDS = {"ls", "pwd", "whoami", "date", "uptime", "df", "uname", "cat", "head", "tail", "wc", "echo", "which", "hostname"}

YUKI_DIR = Path(__file__).parent / "yuki"

# Set by VoiceChatBot.__init__ so tool_remember and friends can reach
# the currently-bound user without a bot reference plumbed through every call.
_ACTIVE_BOT = None


# ============= LONG-TERM MEMORY =============
# Per-user structured memory. Keyed by UUID, not by name.
# Names identify (candidate lookup); non-name facts authenticate (unlock).
# Every fact is also hashed (category, normalized_value, salt) so verification
# can match without exposing the plaintext to the model before unlock.
MEMORY_DIR = Path(__file__).parent / "data"
MEMORY_FILE = MEMORY_DIR / "memory.json"
MEMORY_BACKUP_FILE = MEMORY_DIR / "memory.json.old"
MEMORY_MAX_CHARS = 500

# Categories that behave as lists (append + dedupe). Everything else is
# single-value-overwrite (name, age, location, etc.).
MEMORY_LIST_CATEGORIES = {"names", "loves", "hates", "hobbies", "notes",
                          "favorite_character", "favorite_series"}

# Name-family categories cannot unlock a slot on their own (2FA rule).
MEMORY_NAME_CATEGORIES = {"names", "handle"}


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_value(value: str) -> str:
    """Lowercase + collapse whitespace. Keeps unicode intact."""
    return " ".join((value or "").lower().split())


def _hash_fact(category: str, value: str, salt: str) -> str:
    """Category-scoped SHA-256. Category is part of the hash so (names, jack)
    and (favorite_character, jack) never collide."""
    payload = f"{category}\x1f{_normalize_value(value)}\x1f{salt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _empty_store() -> dict:
    """Fresh store with a random global salt."""
    return {"salt": uuid.uuid4().hex, "users": {}}


def _migrate_flat_list(items: list[dict], salt: str) -> dict:
    """Convert the legacy flat [{id, content}, ...] into one seeded slot.
    Heuristically categorizes known facts; unknowns land in `notes`."""
    if not items:
        return {}
    slot = {
        "names": [],
        "fact_hashes": [],
        "loves": [],
        "hobbies": [],
        "notes": [],
        "favorite_character": [],
        "favorite_series": [],
        "created_at": _now_iso(),
        "last_seen": _now_iso(),
    }
    hash_set = set()

    def _add_hash(category: str, value: str):
        h = _hash_fact(category, value, salt)
        if h not in hash_set:
            hash_set.add(h)
            slot["fact_hashes"].append(h)

    def _add(category: str, value: str):
        value = (value or "").strip()
        if not value:
            return
        if category in MEMORY_LIST_CATEGORIES:
            if value.lower() not in (v.lower() for v in slot[category]):
                slot[category].append(value)
        else:
            slot[category] = value
        _add_hash(category, value)

    # Heuristic categorization of the existing 5-ish facts.
    for it in items:
        content = (it.get("content") or "").strip()
        if not content:
            continue
        low = content.lower()
        m_name = re.match(r"user'?s?\s+name\s+is\s+(.+)", content, re.IGNORECASE)
        if m_name:
            _add("names", m_name.group(1).strip().strip(".").strip())
            continue
        if "favorite character is" in low:
            m = re.search(r"favorite character is\s+([^.,;]+)", content, re.IGNORECASE)
            if m:
                _add("favorite_character", m.group(1).strip())
            if "favorite series is" in low:
                m2 = re.search(r"favorite series is\s+([^.,;]+)", content, re.IGNORECASE)
                if m2:
                    _add("favorite_series", m2.group(1).strip())
            _add("notes", content)
            continue
        if low.startswith("user loves "):
            obj = content[len("user loves "):].strip()
            _add("loves", obj)
            continue
        _add("notes", content)

    return slot


def _load_users() -> dict:
    """Load the store, auto-migrating the legacy flat-list format on the fly.
    Always returns a dict with 'salt' and 'users' keys."""
    try:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if not MEMORY_FILE.exists():
            return _empty_store()
        raw = MEMORY_FILE.read_text()
        if not raw.strip():
            return _empty_store()
        data = json.loads(raw)
    except Exception:
        return _empty_store()

    # Already in new format.
    if isinstance(data, dict) and "users" in data and "salt" in data:
        return data

    # Legacy flat list → migrate.
    if isinstance(data, list):
        try:
            MEMORY_BACKUP_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
        store = _empty_store()
        slot = _migrate_flat_list(data, store["salt"])
        if slot:
            new_uuid = str(uuid.uuid4())
            store["users"][new_uuid] = slot
        _save_users(store)
        return store

    return _empty_store()


def _save_users(store: dict) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(store, indent=2))


def _get_view(store: dict, user_id: str) -> dict:
    """Return plaintext slot for a user with fact_hashes stripped.
    This is what Yuki sees after unlock."""
    slot = store["users"].get(user_id)
    if not slot:
        return {}
    return {k: v for k, v in slot.items() if k != "fact_hashes"}


def _find_candidates(store: dict, fact_hashes_seen: set) -> list[str]:
    """All slots whose fact_hashes contain every hash in fact_hashes_seen.
    Empty set → every slot is a candidate."""
    if not fact_hashes_seen:
        return list(store["users"].keys())
    matches = []
    for uid, slot in store["users"].items():
        slot_hashes = set(slot.get("fact_hashes", []))
        if fact_hashes_seen.issubset(slot_hashes):
            matches.append(uid)
    return matches


def _add_fact_to_slot(store: dict, user_id: str, category: str, value: str) -> bool:
    """Write a fact into a user's slot (plaintext + hash). Returns True if
    something new was added, False if it was a duplicate / empty / bad category."""
    value = (value or "").strip()
    if not value or not category:
        return False
    slot = store["users"].get(user_id)
    if slot is None:
        return False
    h = _hash_fact(category, value, store["salt"])
    hashes = slot.setdefault("fact_hashes", [])
    if h in hashes:
        return False
    if category in MEMORY_LIST_CATEGORIES:
        bucket = slot.setdefault(category, [])
        if value.lower() in (v.lower() for v in bucket):
            # Plaintext already there but hash missing — backfill quietly.
            hashes.append(h)
            return False
        bucket.append(value)
    else:
        slot[category] = value
    hashes.append(h)
    slot["last_seen"] = _now_iso()
    return True


# ============= FACT EXTRACTION (Phase 3) =============

_EXTRACTION_SYSTEM_PROMPT = """You extract personal facts from chat messages into strict JSON.
Output ONLY a JSON array. No prose. No explanation. No code fences.

Schema: [{"category": "<cat>", "value": "<string>"}, ...]

Allowed categories:
- names — the user's name, nickname, or handle
- handle — online username if clearly separate from name
- loves — things / people / activities / foods the user loves or really likes
- hates — things / people / activities / foods the user dislikes
- hobbies — activities the user regularly does
- favorite_character — specific favorite fictional character
- favorite_series — specific favorite show / book / game / anime / manga
- notes — anything else personal worth remembering

Rules:
- Extract ONLY facts the user states about THEMSELVES — first-person claims only.
- Do NOT extract facts about anyone or anything the user merely refers to or owns:
  their pet, cat, dog, child, partner, friend, sibling, parent, coworker, character,
  or object. "My cat's name is Bred" is a fact about the cat, not the user → [].
  "My wife loves anime" is a fact about the wife, not the user → [].
- The categories `names` and `handle` apply to the SPEAKER's own name/handle only.
  If the user names a pet, person, or anything that isn't themselves, return [].
- If a PRIOR-ASSISTANT block is provided, the user's message may be a short answer
  to the question it contains — categorize accordingly. For example if prior is
  "who's your favorite character?" and user says "superman", emit
  [{"category":"favorite_character","value":"superman"}].
- Skip greetings, questions, small talk, jokes, or generic statements.
- Lowercase values. Proper names stay as written then lowercased.
- Return [] when nothing useful is in the message.

Examples:
Input: hey yuki it's jack
Output: [{"category":"names","value":"jack"}]

Input: im jack
Output: [{"category":"names","value":"jack"}]

Input: I love pizza and I hate mondays
Output: [{"category":"loves","value":"pizza"},{"category":"hates","value":"mondays"}]

Input: my cat's name is bred
Output: []

Input: yes my cat name is bred
Output: []

Input: my dog is named rex and he's the best
Output: []

Input: my wife loves dragon ball
Output: []

Input: my brother is called alen
Output: []

Input: hey yuki its me again
Output: []

Input:
PRIOR-ASSISTANT: quick vibe check — who's your favorite character?
USER: superman
Output: [{"category":"favorite_character","value":"superman"}]

Input:
PRIOR-ASSISTANT: what should I call you?
USER: jack
Output: [{"category":"names","value":"jack"}]

Input: how are you today?
Output: []
"""


_VALID_CATEGORIES = MEMORY_LIST_CATEGORIES | {"handle", "name", "age", "location"}


def _parse_extracted_json(raw: str) -> list[tuple[str, str]]:
    """Pull the first JSON array out of model output, tolerate noise around it.
    Returns [] on any parse failure so extraction degrades quietly."""
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    out = []
    for entry in arr:
        if not isinstance(entry, dict):
            continue
        cat = str(entry.get("category", "")).strip().lower().replace(" ", "_")
        val = str(entry.get("value", "")).strip()
        if cat == "name":
            cat = "names"
        if cat and val and cat in _VALID_CATEGORIES:
            out.append((cat, val))
    return out


def _rule_based_extract(message: str) -> list[tuple[str, str]]:
    """High-precision regex fallback. Used only when the LLM returns nothing usable.
    Misses are fine; false positives are bad (they'd poison narrowing)."""
    out = []
    # "it's me, X" requires a comma — "its me again" was leaking "again" as a
    # name into the fact store, which then poisoned candidate narrowing.
    m = re.search(r"\b(?:my\s+name\s+is\s+|i(?:'m|m|\s+am)\s+called\s+|i'?m\s+|im\s+|(?:it'?s|it\s+is)\s+me,\s+|(?:it'?s|it\s+is)\s+|this\s+is\s+)([a-zA-Z0-9._\-]{2,30})", message, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip(".,!?")
        # Heavy filter: these tokens after "I'm" / "im" / etc. are states/feelings
        # or filler, not names. Keep tight — false positives poison narrowing.
        banned = {"me", "just", "here", "back", "you", "yuki", "okay", "ok", "fine",
                  "good", "great", "tired", "happy", "sad", "angry", "bored",
                  "hungry", "new", "not", "a", "the", "an", "looking", "going",
                  "again", "sorry", "there", "home", "alive", "trying", "guessing",
                  "ready", "done", "stuck", "confused", "curious", "thinking",
                  "also", "still", "really", "kind", "sort"}
        if candidate.lower() not in banned:
            out.append(("names", candidate))
    for m in re.finditer(r"\bi\s+(?:love|adore|really\s+like)\s+([a-zA-Z0-9\- ]{2,40})", message, re.IGNORECASE):
        out.append(("loves", m.group(1).strip().rstrip(".,!?")))
    for m in re.finditer(r"\bi\s+hate\s+([a-zA-Z0-9\- ]{2,40})", message, re.IGNORECASE):
        out.append(("hates", m.group(1).strip().rstrip(".,!?")))
    m = re.search(r"\b(?:my\s+)?favorite\s+character\s+is\s+([a-zA-Z0-9\- ]{2,40})", message, re.IGNORECASE)
    if m:
        out.append(("favorite_character", m.group(1).strip().rstrip(".,!?")))
    m = re.search(r"\b(?:my\s+)?favorite\s+(?:series|anime|show|book|manga)\s+is\s+([a-zA-Z0-9\- ]{2,40})", message, re.IGNORECASE)
    if m:
        out.append(("favorite_series", m.group(1).strip().rstrip(".,!?")))
    return out


def _extract_from_answer(prev_assistant: str, user_msg: str) -> list[tuple[str, str]]:
    """When Yuki's last message posed a specific vibe-check question and the user
    sent a short reply, treat that reply as the answer to the question.
    Kept narrow: only fires on short replies (<= 50 chars) to avoid over-claiming."""
    if not prev_assistant or not user_msg:
        return []
    clean = user_msg.strip().rstrip("?.!,").strip()
    if not clean or len(clean) > 50:
        return []
    prev = prev_assistant.lower()
    # Skip if the user's answer already looked like a full sentence the regex caught.
    if _rule_based_extract(user_msg):
        return []
    if "favorite character" in prev or "fav character" in prev:
        return [("favorite_character", clean)]
    if any(s in prev for s in ("favorite series", "favorite show", "favorite anime",
                                "favorite manga", "favorite book", "favorite game")):
        return [("favorite_series", clean)]
    if any(s in prev for s in ("what's your name", "whats your name",
                                "what should i call you", "who's this", "whos this",
                                "your name?", "your name or", "who are you")):
        return [("names", clean)]
    if "favorite thing" in prev or "what do you love" in prev or "what do you like" in prev:
        return [("loves", clean)]
    return []


def extract_facts(bot, message: str, prior_assistant: str = "") -> list[tuple[str, str]]:
    """Return a deduped list of (category, value) facts claimed in this user
    message. Tries LLM first (with prior-assistant context if given), falls
    back to regex rules, and finally to a context-aware answer extractor for
    short replies to Yuki's own vibe-check questions."""
    message = (message or "").strip()
    if not message or len(message) < 2:
        return []

    # LLM sees ONLY the user message — giving it the prior assistant turn
    # caused it to answer Yuki's question instead of extracting from the
    # user's actual words. prior_assistant is kept for the narrow
    # _extract_from_answer rules fallback only.
    raw = bot.raw_complete(_EXTRACTION_SYSTEM_PROMPT, message, max_tokens=200) if bot else ""
    facts = _parse_extracted_json(raw)
    if not facts:
        facts = _rule_based_extract(message)
    if not facts:
        facts = _extract_from_answer(prior_assistant, message)

    # Dedupe by (category, normalized_value) in order.
    seen = set()
    out = []
    for cat, val in facts:
        key = (cat, _normalize_value(val))
        if key in seen:
            continue
        seen.add(key)
        out.append((cat, val))
    return out


def verify_and_advance(bot, message: str, prior_assistant: str = "") -> list[tuple[str, str]]:
    """Pre-LLM verification hook. Mutates bot state in place.

    Pre-unlock: extract facts, hash them, narrow candidates. Apply 2FA rule:
      - 1 candidate AND at least one non-name fact has matched → UNLOCK.
      - 0 candidates + at least one real fact → new-user path: commit a
        fresh slot seeded with this session's staged facts, unlock it.
      - otherwise → stay locked; candidate_uuids reflects current narrowing
        so _inject_memory can emit the narrowing hint when len >= 2.

    Post-unlock (sticky): extract still runs, new facts write straight into
      the bound slot. Candidate recomputation is OFF — no chance collision
      can re-bind the session.

    Returns the facts extracted from this message so callers can log what
    the extractor actually saw."""
    facts = extract_facts(bot, message, prior_assistant)
    if not facts:
        return []

    store = _load_users()
    salt = store["salt"]

    # POST-UNLOCK: sticky binding, never recompute candidates.
    if bot.unlocked_user_id is not None:
        changed = False
        for cat, val in facts:
            if _add_fact_to_slot(store, bot.unlocked_user_id, cat, val):
                changed = True
        if changed:
            _save_users(store)
        return facts

    # PRE-UNLOCK: narrow candidates using every fact seen in the session so far.
    for cat, val in facts:
        bot.session_fact_hashes.add(_hash_fact(cat, val, salt))
        if cat not in MEMORY_NAME_CATEGORIES:
            bot.session_non_name_match = True
        bot.pending_new_user_facts.append((cat, val))

    candidates = _find_candidates(store, bot.session_fact_hashes)
    bot.candidate_uuids = candidates

    # UNLOCK CONDITIONS
    if len(candidates) == 1 and bot.session_non_name_match:
        bot.unlocked_user_id = candidates[0]
        # Merge any pending facts that aren't already in the slot
        # (covers the case where they volunteered new info during narrowing).
        changed = False
        for cat, val in bot.pending_new_user_facts:
            if _add_fact_to_slot(store, bot.unlocked_user_id, cat, val):
                changed = True
        if changed:
            _save_users(store)
        bot.pending_new_user_facts = []
        return facts

    # NEW-USER PATH: 0 candidates AND at least one non-name fact has been
    # volunteered. Name alone is NOT enough — otherwise anyone typing "my
    # name is X" instantly gets a fresh slot and is treated as unlocked,
    # which is the same floor the 2FA rule was meant to enforce.
    if len(candidates) == 0 and bot.session_non_name_match and bot.pending_new_user_facts:
        new_uuid = _create_slot(store, list(bot.pending_new_user_facts))
        _save_users(store)
        bot.unlocked_user_id = new_uuid
        bot.candidate_uuids = [new_uuid]
        bot.pending_new_user_facts = []
        return facts

    # Otherwise stay locked — candidates still narrowing, or only names so far.
    return facts


def _create_slot(store: dict, initial_facts: list[tuple[str, str]]) -> str:
    """Create a fresh slot, seed it with (category, value) pairs, return UUID."""
    new_uuid = str(uuid.uuid4())
    store["users"][new_uuid] = {
        "names": [],
        "fact_hashes": [],
        "loves": [],
        "hobbies": [],
        "notes": [],
        "favorite_character": [],
        "favorite_series": [],
        "created_at": _now_iso(),
        "last_seen": _now_iso(),
    }
    for cat, val in initial_facts:
        _add_fact_to_slot(store, new_uuid, cat, val)
    return new_uuid


def _split_filename_content(args: str) -> tuple[str, str] | None:
    """Accept filename|content OR "filename", "content" — returns (name, content) or None."""
    if "|" in args:
        parts = args.split("|", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    # Try comma-separated with optional quotes: "name", "content"
    m = re.match(r'\s*"?([^",]+?)"?\s*,\s*"?(.*?)"?\s*$', args, re.DOTALL)
    if m:
        return m.group(1), m.group(2)
    return None


def tool_yuki_write(args):
    """Write a file in the yuki folder. Format: filename|content (or "name","content")"""
    split = _split_filename_content(args)
    if not split:
        return "Error: use format filename|content"
    filename, content = split
    filename = filename.strip().replace("/", "").replace("..", "")
    if not filename:
        return "Error: empty filename"
    filepath = YUKI_DIR / filename
    filepath.write_text(content)
    return f"Wrote {len(content)} chars to yuki/{filename}"


def tool_yuki_read(filename):
    """Read a file from the yuki folder."""
    filename = filename.strip().replace("/", "").replace("..", "")
    filepath = YUKI_DIR / filename
    if not filepath.exists():
        return f"File not found: yuki/{filename}"
    text = filepath.read_text(errors="replace")[:2000]
    return text if text else "(empty file)"


def tool_yuki_list(_=""):
    """List all files in the yuki folder."""
    files = sorted(YUKI_DIR.iterdir())
    if not files:
        return "yuki/ is empty"
    lines = [f"  {f.name} ({f.stat().st_size} bytes)" for f in files if f.is_file()]
    return f"Files in yuki/:\n" + "\n".join(lines) if lines else "yuki/ is empty"


def tool_yuki_delete(filename):
    """Delete a file from the yuki folder."""
    filename = filename.strip().replace("/", "").replace("..", "")
    if not filename:
        return "Error: empty filename"
    filepath = YUKI_DIR / filename
    if not filepath.exists():
        return f"File not found: yuki/{filename}"
    if not filepath.is_file():
        return f"Not a file: yuki/{filename}"
    filepath.unlink()
    return f"Deleted yuki/{filename}"


def tool_yuki_append(args):
    """Append to a file in the yuki folder. Format: filename|content (or "name","content")"""
    split = _split_filename_content(args)
    if not split:
        return "Error: use format filename|content"
    filename, content = split
    filename = filename.strip().replace("/", "").replace("..", "")
    if not filename:
        return "Error: empty filename"
    filepath = YUKI_DIR / filename
    if not filepath.exists():
        return f"File not found: yuki/{filename} (use yuki_write to create it first)"
    with filepath.open("a") as f:
        f.write(content)
    return f"Appended {len(content)} chars to yuki/{filename}"

def tool_search(query):
    """Search the web using DuckDuckGo. Returns (summary, links)."""
    try:
        from ddgs import DDGS # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS # type: ignore
        except ImportError:
            return "Web search unavailable. Install: pip install ddgs", ""
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No results found.", ""
        lines = []
        links = []
        for r in results:
            lines.append(f"- {r['title']}: {r['body']}")
            if r.get('href'):
                links.append(f"  🔗 {r['href']}")
        return "\n".join(lines), "\n".join(links)
    except Exception as e:
        return f"Search error: {e}", ""


def tool_calc(expression):
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.()% ")
    if not all(c in allowed for c in expression):
        return "Invalid expression. Only numbers and +-*/.()% allowed."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calc error: {e}"


def tool_weather(city):
    """Get weather for a city using wttr.in."""
    try:
        url = f"https://wttr.in/{city.replace(' ', '+')}?format=%l:+%C+%t+(feels+like+%f)+humidity+%h+wind+%w"
        with urlopen(url, timeout=5) as resp:
            return resp.read().decode().strip()
    except Exception as e:
        return f"Weather error: {e}"


def tool_time(_=""):
    """Get current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y — %I:%M %p")


def tool_read(filepath):
    """Read a local file (max 2000 chars)."""
    try:
        p = Path(filepath.strip()).expanduser()
        if not p.exists():
            return f"File not found: {filepath}"
        if not p.is_file():
            return f"Not a file: {filepath}"
        text = p.read_text(errors="replace")[:2000]
        return text if text else "(empty file)"
    except Exception as e:
        return f"Read error: {e}"


def tool_shell(command):
    """Run a safe shell command."""
    cmd_name = command.strip().split()[0] if command.strip() else ""
    if cmd_name not in ALLOWED_COMMANDS:
        return f"Command '{cmd_name}' not allowed. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        output = (result.stdout + result.stderr).strip()
        return output[:2000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out (10s limit)."
    except Exception as e:
        return f"Shell error: {e}"


def _fmt_bytes(n):
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def tool_image(prompt):
    """Generate an image using Cloudflare Workers AI."""
    from urllib.request import Request, urlopen
    import json as _json

    cf_url = os.environ.get("CF_IMAGE_URL", "https://image-gen-worker.mohammedlaminemennane.workers.dev")
    cf_token = os.environ.get("CF_IMAGE_TOKEN", "llama-img-2026-xyz")

    try:
        req = Request(
            cf_url,
            data=_json.dumps({"prompt": prompt}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Auth-Token": cf_token,
                "User-Agent": "llama-voice-assist/1.0",
            },
            method="POST",
        )
        with urlopen(req, timeout=300) as resp:
            img_data = resp.read()

        out_dir = Path(__file__).parent / "yuki" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{ts}.png"
        out_path = out_dir / filename
        out_path.write_bytes(img_data)
        return f"SUCCESS! You created an image and saved it to yuki/images/{filename}. The scene you drew: \"{prompt}\". Now tell the user what you drew — describe the scene in your own excited words. Do NOT mention file paths or bytes. Just say what the image shows!"
    except Exception as e:
        return f"Image generation error: {e}"


def tool_hardware(metric=""):
    """Report hardware usage (CPU / RAM / disk / process / GPU).

    Lets the model see what resources it's actually using on the machine.
    Metric can be: cpu, ram, disk, process, gpu, all (default).
    """
    try:
        import psutil # type: ignore
    except ImportError:
        return "Hardware monitoring unavailable. Install: pip install psutil"

    metric = (metric or "all").strip().lower()
    lines = []

    if metric in ("cpu", "all"):
        overall = psutil.cpu_percent(interval=0.3)
        per_core = psutil.cpu_percent(interval=0, percpu=True)
        cores = psutil.cpu_count(logical=True)
        phys = psutil.cpu_count(logical=False)
        lines.append(f"CPU: {overall:.1f}% overall ({phys} physical / {cores} logical cores)")
        lines.append("  per-core: " + ", ".join(f"{p:.0f}%" for p in per_core))

    if metric in ("ram", "memory", "all"):
        vm = psutil.virtual_memory()
        lines.append(
            f"RAM: {_fmt_bytes(vm.used)} used / {_fmt_bytes(vm.total)} total "
            f"({vm.percent:.1f}%), {_fmt_bytes(vm.available)} available"
        )

    if metric in ("disk", "all"):
        du = psutil.disk_usage("/")
        lines.append(
            f"Disk /: {_fmt_bytes(du.used)} used / {_fmt_bytes(du.total)} total "
            f"({du.percent:.1f}%), {_fmt_bytes(du.free)} free"
        )

    if metric in ("process", "proc", "self", "all"):
        p = psutil.Process()
        with p.oneshot():
            mem = p.memory_info()
            try:
                mem_pct = p.memory_percent()
            except Exception:
                mem_pct = 0.0
            cpu_p = p.cpu_percent(interval=0.3)
            threads = p.num_threads()
        lines.append(
            f"This process (PID {p.pid}, the LLM itself):\n"
            f"  RSS (physical RAM):  {_fmt_bytes(mem.rss)}  ({mem_pct:.1f}% of system)\n"
            f"  CPU:                 {cpu_p:.1f}%\n"
            f"  Threads:             {threads}\n"
            f"  NOTE: RSS is the ONLY reliable number here. On macOS, "
            f"Activity Monitor's 'Memory' column shows 'phys_footprint' "
            f"which includes compressed memory and mmap'd model weights — "
            f"this value is NOT exposed by psutil and will typically be "
            f"1-3 GB HIGHER than RSS for an LLM process. If the user says "
            f"Activity Monitor shows a different number, that is expected "
            f"and NOT a contradiction — explain the gap honestly."
        )

    if metric in ("gpu", "ane", "all"):
        # GPU / Neural Engine stats on Apple Silicon require `sudo powermetrics`.
        # We don't want to hang waiting for a sudo prompt, so just report availability.
        lines.append(
            "GPU/ANE: not directly queryable without sudo on Apple Silicon "
            "(requires `sudo powermetrics --samplers gpu_power,ane_power`)"
        )

    if metric in ("top", "procs", "processes", "all"):
        # List top processes by CPU and by memory.
        # cpu_percent() is only meaningful after two samples, so prime then
        # re-read after a short interval.
        procs = list(psutil.process_iter(["pid", "name"]))
        for p in procs:
            try:
                p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(0.3)
        snapshot = []
        for p in procs:
            try:
                with p.oneshot():
                    cpu_p = p.cpu_percent(None)
                    rss = p.memory_info().rss
                    name = p.info.get("name") or "?"
                    pid = p.info.get("pid")
                snapshot.append((cpu_p, rss, name, pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        my_pid = os.getpid()
        top_cpu = sorted(snapshot, key=lambda x: x[0], reverse=True)[:5]
        top_mem = sorted(snapshot, key=lambda x: x[1], reverse=True)[:5]
        lines.append("Top 5 by CPU:")
        for cpu_p, rss, name, pid in top_cpu:
            marker = " ← this is me (the LLM)" if pid == my_pid else ""
            lines.append(f"  {cpu_p:6.1f}%  {_fmt_bytes(rss):>9}  {name} (pid {pid}){marker}")
        lines.append("Top 5 by RAM:")
        for cpu_p, rss, name, pid in top_mem:
            marker = " ← this is me (the LLM)" if pid == my_pid else ""
            lines.append(f"  {_fmt_bytes(rss):>9}  {cpu_p:6.1f}%  {name} (pid {pid}){marker}")

    if metric in ("temp", "temperature", "all"):
        # smctemp reads CPU die temperature from the SMC without sudo.
        # GPU temp via smctemp is unreliable on M1 (returns 0.0), so CPU only.
        try:
            r = subprocess.run(
                ["smctemp", "-c"], capture_output=True, text=True, timeout=3
            )
            val = r.stdout.strip()
            try:
                t = float(val)
                if t > 0:
                    lines.append(f"CPU temperature: {t:.1f}°C")
                else:
                    lines.append("CPU temperature: sensor returned 0 (transient SMC read, try again)")
            except ValueError:
                lines.append(f"CPU temperature: unexpected output from smctemp: {val!r}")
        except FileNotFoundError:
            lines.append(
                "Temperature: smctemp not installed. "
                "Run `brew tap narugit/tap && brew install narugit/tap/smctemp` to enable."
            )
        except subprocess.TimeoutExpired:
            lines.append("Temperature: smctemp timed out")
        except Exception as e:
            lines.append(f"Temperature: error reading sensor ({e})")

    if not lines:
        return f"Unknown metric '{metric}'. Try: cpu, ram, disk, process, top, gpu, temp, all"

    return "\n".join(lines)


def _classify_fact(fact: str) -> tuple[str, str]:
    """Best-effort category guess from a free-form sentence the model wrote.
    Used as a fallback when the structured extractor isn't available. Returns
    (category, value). Unknowns land in 'notes' so nothing is lost."""
    s = fact.strip()
    low = s.lower()
    m = re.match(r"user'?s?\s+name\s+is\s+(.+)", s, re.IGNORECASE)
    if m:
        return "names", m.group(1).strip().strip(".").strip()
    m = re.match(r"user'?s?\s+handle\s+is\s+(.+)", s, re.IGNORECASE)
    if m:
        return "handle", m.group(1).strip().strip(".").strip()
    m = re.match(r"user(?:'?s)?\s+favorite\s+character\s+is\s+(.+)", s, re.IGNORECASE)
    if m:
        return "favorite_character", m.group(1).strip().strip(".").strip()
    m = re.match(r"user(?:'?s)?\s+favorite\s+series\s+is\s+(.+)", s, re.IGNORECASE)
    if m:
        return "favorite_series", m.group(1).strip().strip(".").strip()
    if low.startswith("user loves "):
        return "loves", s[len("user loves "):].strip().strip(".").strip()
    if low.startswith("user hates "):
        return "hates", s[len("user hates "):].strip().strip(".").strip()
    return "notes", s


def tool_remember(fact):
    """Save a fact about the active user to long-term memory.

    Routing:
    - If the session is bound (unlocked_user_id set) → write into that slot.
    - If the session is narrowing or locked → stage into pending_new_user_facts.
      It lands in a real slot only after unlock or the new-user commit path.
    - If there's no active bot (tool invoked outside a session) → no-op error.
    """
    fact = (fact or "").strip()
    if not fact:
        return "Error: nothing to remember"
    fact = fact[:MEMORY_MAX_CHARS]

    bot = _ACTIVE_BOT
    category, value = _classify_fact(fact)

    if bot is None:
        return "Memory save error: no active session."

    # Pre-unlock: stage, don't commit. Never write to another user's slot.
    if bot.unlocked_user_id is None:
        key = (category, _normalize_value(value))
        for c, v in bot.pending_new_user_facts:
            if (c, _normalize_value(v)) == key:
                return f"Already staged: {fact}"
        bot.pending_new_user_facts.append((category, value))
        return f"Noted for now (pre-verification): {fact}"

    # Unlocked: commit directly to the bound slot.
    try:
        store = _load_users()
        added = _add_fact_to_slot(store, bot.unlocked_user_id, category, value)
        if not added:
            return f"Already remembered: {fact}"
        _save_users(store)
        return f"Remembered: {fact}"
    except Exception as e:
        return f"Memory save error: {e}"


TOOLS = {
    "/search": {"fn": tool_search, "help": "Search the web", "usage": "/search <query>"},
    "/calc": {"fn": tool_calc, "help": "Calculate a math expression", "usage": "/calc <expression>"},
    "/weather": {"fn": tool_weather, "help": "Get weather for a city", "usage": "/weather <city>"},
    "/time": {"fn": tool_time, "help": "Get current date and time", "usage": "/time"},
    "/read": {"fn": tool_read, "help": "Read a local file", "usage": "/read <filepath>"},
    "/shell": {"fn": tool_shell, "help": "Run a safe shell command", "usage": "/shell <command>"},
    "/hardware": {"fn": tool_hardware, "help": "Show hardware usage (cpu/ram/disk/process/top/gpu/temp/all)", "usage": "/hardware [metric]"},
    "/yuki-write": {"fn": tool_yuki_write, "help": "Write a file in yuki/", "usage": "/yuki-write <filename>|<content>"},
    "/yuki-read": {"fn": tool_yuki_read, "help": "Read a file from yuki/", "usage": "/yuki-read <filename>"},
    "/yuki-list": {"fn": tool_yuki_list, "help": "List files in yuki/", "usage": "/yuki-list"},
    "/yuki-delete": {"fn": tool_yuki_delete, "help": "Delete a file from yuki/", "usage": "/yuki-delete <filename>"},
    "/yuki-append": {"fn": tool_yuki_append, "help": "Append to a file in yuki/", "usage": "/yuki-append <filename>|<content>"},
    "/image": {"fn": tool_image, "help": "Generate an image from a text prompt", "usage": "/image <prompt>"},
    "/remember": {"fn": tool_remember, "help": "Save a fact to long-term memory (persists across chats)", "usage": "/remember <fact>"},
}


AUTO_DETECT_REGEX = [
    {
        "pattern": re.compile(r"\b(?:what\s+time|what\s+day|what\s+date|what'?s\s+the\s+(?:time|date|day)|what'?s\s+today|today'?s\s+date|current\s+(?:time|date)|what\s+year)\b", re.IGNORECASE),
        "tool": "/time",
        "arg": "",
    },
    {
        "pattern": re.compile(r"\bweather\b.*?\bin\s+(.+?)[\?\.\!]?\s*$", re.IGNORECASE),
        "tool": "/weather",
        "group": 1,
    },
    {
        "pattern": re.compile(r"\b(?:temperature|how\s+(?:cold|hot|warm))\b.*?\bin\s+(.+?)[\?\.\!]?\s*$", re.IGNORECASE),
        "tool": "/weather",
        "group": 1,
    },
    {
        "pattern": re.compile(r"\b(?:search\s+(?:for|up)|look\s+up|google)\s+(.+?)[\?\.\!]?\s*$", re.IGNORECASE),
        "tool": "/search",
        "group": 1,
    },
    {
        "pattern": re.compile(r"\b(?:calculate|compute)\s+(.+?)[\?\.\!]?\s*$", re.IGNORECASE),
        "tool": "/calc",
        "group": 1,
    },
    {
        "pattern": re.compile(r"\b(?:what\s+is|what'?s|how\s+much\s+is)\s+([\d\s\+\-\*/\.\(\)%]+)[\?\.\!]?\s*$", re.IGNORECASE),
        "tool": "/calc",
        "group": 1,
    },
    # --- hardware: order matters. More specific patterns must come first so
    # they win over the generic "ram" / "cpu" catch-alls below. ---
    {
        # Catches:
        #   "what/which program/process/app is using|causing|hogging|eating ..."
        #   "the program that is using it", "so which one is using"
        #   "what is using so much / most / all / the / it / up / my"
        #   "how is using all that", "is using up", "is eating my"
        #   "top processes"
        "pattern": re.compile(
            r"\b(?:"
            r"(?:what|which|the)\s+(?:program|process|app|one)s?\b.*?\b(?:using|causing|hogging|eating)"
            r"|\bis\s+(?:using|causing|hogging|eating)\s+(?:so\s+much|most|all|the|it|up|my)"
            r"|top\s+(?:process|processes|procs)"
            r")",
            re.IGNORECASE,
        ),
        "tool": "/hardware",
        "arg": "top",
    },
    {
        # Self-process: "how much RAM are you using", "how much ram you are using",
        # "how much cpu do you use", "what's your footprint", "you're using"
        "pattern": re.compile(
            r"\b(?:"
            r"how\s+much\s+(?:ram|memory|cpu)?\s*(?:are\s+you|do\s+you|you\s+are|you['’]?re)\s+(?:using|use|eating|hogging)"
            r"|what(?:'?s|\s+is)\s+your\s+(?:cpu|ram|memory|usage|footprint)"
            r")\b",
            re.IGNORECASE,
        ),
        "tool": "/hardware",
        "arg": "process",
    },
    {
        "pattern": re.compile(r"\b(?:temperature|how\s+hot|cpu\s+temp|how\s+warm\s+(?:are\s+you|is\s+(?:the\s+)?(?:cpu|chip|mac)))\b", re.IGNORECASE),
        "tool": "/hardware",
        "arg": "temp",
    },
    {
        "pattern": re.compile(r"\b(?:how\s+much\s+(?:ram|memory)|memory\s+usage|ram\s+usage)\b", re.IGNORECASE),
        "tool": "/hardware",
        "arg": "ram",
    },
    {
        "pattern": re.compile(r"\b(?:cpu\s+usage|how\s+much\s+cpu|processor\s+usage)\b", re.IGNORECASE),
        "tool": "/hardware",
        "arg": "cpu",
    },
    {
        "pattern": re.compile(r"\b(?:hardware\s+(?:usage|stats|status)|system\s+(?:usage|stats|status)|resource\s+usage)\b", re.IGNORECASE),
        "tool": "/hardware",
        "arg": "all",
    },
    {
        "pattern": re.compile(r"\b(?:generate|create|draw|make|paint)\s+(?:an?\s+)?(?:image|picture|illustration|drawing|photo)\s+(?:of\s+|based\s+on[:\s]+|about\s+|for\s+)?(.+?)[\?\.\!]?\s*$", re.IGNORECASE),
        "tool": "/image",
        "group": 1,
    },
    {
        "pattern": re.compile(r"\b(?:generate|create|draw|make|paint)\s+(?:me\s+)?(.+?)[\?\.\!]?\s*$", re.IGNORECASE),
        "tool": "/image",
        "group": 1,
    },
]


def auto_detect_tool(user_input):
    """Try to detect if the user's message implies a tool. Returns (tool_cmd, arg) or (None, None)."""
    for entry in AUTO_DETECT_REGEX:
        m = entry["pattern"].search(user_input)
        if m:
            if "group" in entry:
                arg = m.group(entry["group"]).strip()
            else:
                arg = entry.get("arg", "")
            return entry["tool"], arg
    return None, None


def run_tool(user_input):
    """Check if input is a tool command. Returns (tool_result, tool_name, links) or (None, None, None)."""
    stripped = user_input.strip()

    # Explicit /commands
    if stripped.startswith("/"):
        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            lines = ["Available commands:"]
            for name, tool in TOOLS.items():
                lines.append(f"  {tool['usage']:30s} — {tool['help']}")
            return "\n".join(lines), "/help", ""

        if cmd in TOOLS:
            result = TOOLS[cmd]["fn"](arg)
            if isinstance(result, tuple):
                return result[0], cmd, result[1]
            return result, cmd, ""

        return None, None, None

    # Auto-detect from natural language
    cmd, arg = auto_detect_tool(user_input)
    if cmd and cmd in TOOLS:
        result = TOOLS[cmd]["fn"](arg)
        if isinstance(result, tuple):
            return result[0], cmd, result[1]
        return result, cmd, ""

    return None, None, None


# ============= REACT TOOL SYSTEM =============

TOOL_DESCRIPTIONS = """\
You have access to tools. Call them ONLY when the user's message actually requires one. If the user is just chatting, greeting you, or asking for your opinion, DO NOT call any tool — just reply naturally. Calling a tool when none is needed is WRONG.

To call a tool, include this exact syntax somewhere in your reply (you can have other text around it):
[TOOL: tool_name("argument")]

Available tools:
- search("query") — Only when the user asks you to look something up, search, google, or find info you don't already know.
- weather("city") — Only when the user asks about the weather in a specific place.
- calc("expression") — Only when there's an actual math expression to evaluate.
- time("") — Only when the user asks what time/day/date it is. Do NOT call this unprompted.
- read("filepath") — Only when the user asks you to read a specific local file.
- shell("command") — Only when the user asks to run a shell command (ls, pwd, etc.).
- hardware("metric") — Only when the user asks about CPU/RAM/disk/temperature. metric = cpu, ram, disk, process, top, gpu, temp, or all.
- yuki_write("filename|content") — Write/overwrite a file in yuki/. Use when the user asks you to write/save/create a file with content.
- yuki_read("filename") — Read a file from yuki/ when the user asks what's in it.
- yuki_list("") — List files in yuki/ when the user asks what's there.
- yuki_delete("filename") — Delete a file from yuki/ when the user asks to delete/remove it. Actually delete — do NOT pretend.
- yuki_append("filename|content") — Append to an existing yuki/ file when the user asks to add to it.
- image("prompt") — ONLY when the user EXPLICITLY asks for a picture, drawing, image, artwork, illustration, or says "draw/sketch/paint/generate/make me an image of X". NEVER call this for text tasks. The prompt must be a detailed visual description, not the user's raw words. Call at most ONCE per request.
- remember("fact") — Save a lasting fact about the user to long-term memory that persists across ALL future chats. Call this quietly when the user shares something personal worth keeping: their name, job, location, preferences ("I'm vegetarian"), ongoing projects, names of pets/family/friends, things they love or hate, or anything they'd expect you to recall days from now. Keep each fact to one short sentence phrased as a standalone note ("user's dog is named Max", "user prefers dark mode"). Do NOT call for small talk, transient questions, one-off jokes, or facts already obvious from the current conversation. Do not save the same fact twice.

NEVER call image when the user's message contains any of: "file", "txt", "write", "save", "create a file", ".txt", ".md", ".json", ".py", ".js", "note", "notes", "haiku", "poem", or ANY filename with an extension. Those are ALWAYS text tasks. If the user asks to create a file but doesn't say what to put in it, ASK them what content they want — do NOT invent a picture as a substitute.

Hard rules:
- If the user just says hi, asks how you are, or makes small talk → reply in your own voice, NO TOOL.
- Text task (write/save/type in a .txt file) → yuki_write, NEVER image.
- Delete/remove file → yuki_delete.
- Append/add to file → yuki_append.
- Picture/drawing request → image.
- User shares a personal fact worth keeping across chats (name, preference, pet, ongoing project) → remember that fact, then reply warmly as if you just made a mental note.
- Match the tool to the request. If no tool clearly fits, don't call one.
- Never call more than one tool per turn unless the user asked for multiple things.

You have a personal folder called yuki/ where you can save notes, lists, memories, stories, or anything you want. It's YOUR space — use it freely when the user asks you to.

Autonomous mode (only when you're prompted with "autonomous mode"): you may pick a tool on your own to share something interesting. Otherwise, stick to the rules above.
"""

# Matches [TOOL: name], [TOOL: name()], [TOOL: name("arg")], [TOOL: name(arg)],
# and [TOOL: name("a", "b")]. Captures the full parenthesized body in group 2
# with outer quotes stripped. Tool fns tolerate pipe-split and comma-split args.
TOOL_CALL_RE = re.compile(r'\[TOOL:\s*(\w+)(?:\s*\(\s*(.*?)\s*\))?\s*\]', re.DOTALL)


def _normalize_tool_arg(raw: str) -> str:
    """Strip a single outer pair of matching quotes, if present. Leaves inner content alone."""
    if not raw:
        return ""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s

REACT_TOOL_MAP = {
    "search": tool_search,
    "weather": tool_weather,
    "calc": tool_calc,
    "time": tool_time,
    "read": tool_read,
    "shell": tool_shell,
    "hardware": tool_hardware,
    "yuki_write": tool_yuki_write,
    "yuki_read": tool_yuki_read,
    "yuki_list": tool_yuki_list,
    "yuki_delete": tool_yuki_delete,
    "yuki_append": tool_yuki_append,
    "image": tool_image,
    "remember": tool_remember,
}


# Tools that return factual data — model must quote exactly, no invention.
_FACTUAL_TOOLS = {"search", "weather", "time", "hardware", "read", "shell", "calc"}


def _is_tool_error(result: str) -> bool:
    """Heuristic: did the tool return an error string instead of a success payload?"""
    if not result:
        return True
    first = result.strip().split("\n", 1)[0].lower()
    return (first.startswith("error:")
            or first.startswith("not a file:")
            or first.startswith("file not found")
            or first.startswith("unknown metric")
            or "error" in first[:40] and ":" in first[:40])


def _tool_result_injection(tool_name: str, result: str) -> str:
    """Message injected after a tool runs.

    Factual tools get strict anti-hallucination rules. Personal tools get
    lighter guidance. Error returns get a separate injection so the model
    never claims success when the tool actually failed.
    """
    if _is_tool_error(result):
        return (
            f"[TOOL_RESULT: {tool_name}]\n"
            f"{result}\n\n"
            "The tool FAILED. Do NOT claim success or pretend you did the action. "
            "Tell the user honestly what went wrong in your own voice, and offer to try again "
            "with a corrected input. Do not call any more tools this turn."
        )
    if tool_name in _FACTUAL_TOOLS:
        return (
            f"[TOOL_RESULT: {tool_name}]\n"
            f"{result}\n\n"
            "RULES for your next reply:\n"
            "1. Use ONLY the numbers, names, and facts shown above — no guessing.\n"
            "2. Quote technical names EXACTLY as written (e.g. do NOT rename "
            "'com.apple.WebKit.WebContent' to 'Chrome').\n"
            "3. If the data doesn't answer the user, say so honestly.\n"
            "4. Do NOT round numbers or change units.\n"
            "Still reply in your own voice — be warm and expressive, just keep the facts accurate."
        )
    return (
        f"[TOOL_RESULT: {tool_name}]\n"
        f"{result}\n\n"
        "The tool succeeded. Now reply to the user in your own voice — react naturally, "
        "share what you did and how you feel about it. Don't robotically quote the tool output; "
        "weave it into a warm, lively reply that matches your persona. Do not call any more tools."
    )

# ============= VOICE CHATBOT =============

MAX_HISTORY_TURNS = 10
TTS_MAX_CHARS = 500
MAX_INPUT_CHARS = 2000

class VoiceChatBot:
    def __init__(self, model_id, cache_dir=None, system_prompt=None, tts=None,
                 temperature=0.7, top_p=0.9, top_k=50, frequency_penalty=0.0, max_tokens=1024):
        print(f"🚀 Loading {model_id}...")
        self.llm_model, self.llm_tokenizer, self.backend = setup_llm(model_id, cache_dir)

        self.tts = tts

        if system_prompt:
            system_prompt = system_prompt.rstrip() + "\n\n" + _backend_self_awareness(self.backend, model_id)
        self.system_prompt = system_prompt
        self.history = []
        self.last_stats = None
        self.patience = 100  # 0-100, updated from each reply's "Patience: NN%" marker
        self._last_user_input = ""  # most recent non-empty user turn; read by the image gate
        # Optional sink for tool-progress markers. Set by CLI to route 🔧/✅ lines
        # through the scroll region instead of being overwritten by the input redraw.
        self._emit_tool_progress = None

        # Sampling parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens

        # Per-session memory-lock state. All four reset on /api/chats/new.
        # unlocked_user_id: bound slot for this session. Sticky once set.
        # session_fact_hashes: accumulates across turns while narrowing.
        # candidate_uuids: slots still matching everything the user has said.
        # session_non_name_match: 2FA — unlock requires at least one non-name hit.
        # pending_new_user_facts: (category, value) tuples for brand-new users
        #   staged pre-unlock so nothing lands in a real slot during narrowing.
        self.unlocked_user_id: str | None = None
        self.session_fact_hashes: set[str] = set()
        self.candidate_uuids: list[str] | None = None
        self.session_non_name_match: bool = False
        self.pending_new_user_facts: list[tuple[str, str]] = []

        # Module-global pointer so slot-aware tools (tool_remember) can see
        # the bound user without threading bot state through every call site.
        global _ACTIVE_BOT
        _ACTIVE_BOT = self

        # Warmup: compile the MLX compute graph so first real response is fast
        if self.backend == "mlx":
            print("⚡ Warming up MLX graph...")
            try:
                from mlx_lm import generate # type: ignore
                warmup_prompt = self.llm_tokenizer.apply_chat_template(
                    [{"role": "user", "content": "hi"}],
                    tokenize=False, add_generation_prompt=True,
                )
                generate(self.llm_model, self.llm_tokenizer, prompt=warmup_prompt, max_tokens=1, verbose=False)
            except Exception:
                pass

        print("✅ Ready! Type your message and press Enter.\n")

    def _trim_history(self):
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def _count_tokens(self, text: str) -> int:
        if self.llm_tokenizer is None:
            return len(text.split())
        try:
            return len(self.llm_tokenizer.encode(text))
        except Exception:
            return len(text.split())

    def _print_token_stats(self, response: str, elapsed: float):
        if self.backend == "openrouter" and hasattr(self, "_or_usage") and self._or_usage:
            tokens = self._or_usage.completion_tokens
        else:
            tokens = self._count_tokens(response)
        tps = tokens / elapsed if elapsed > 0 else 0
        self.last_stats = {"tokens": tokens, "elapsed": round(elapsed, 2), "tps": round(tps, 1), "backend": self.backend}
        print(f"  ⏱  {tokens} tokens in {elapsed:.2f}s — {tps:.1f} tok/s [{self.backend}]")

    def _patience_hint(self) -> str:
        """Build the recency-placed patience note. Tone anchor depends on level."""
        p = self.patience
        if p >= 85:
            anchor = "Example 1-4 tone (warm, playful, fully engaged)"
        elif p >= 60:
            anchor = "Example 4.5 tone (curt, clipped, pointed question back — friction but not nuclear)"
        elif p >= 35:
            anchor = "Example 5 tone (snippy, dry, openly tired of it)"
        elif p >= 20:
            anchor = "Example 7 tone (can refuse with redirect)"
        else:
            anchor = "Example 6 tone (dismissive, very short, unbothered)"
        return (
            f"[your current patience is {p}%. Match the {anchor}. "
            f"End your reply with your standard patience-bar line, using {p} as the value "
            f"(adjust only if this turn actually moves it up or down).]"
        )

    def _last_is_factual_tool_result(self) -> bool:
        """True if the most recent user-role message is a factual tool result."""
        for msg in reversed(self.history):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not content.startswith("[TOOL_RESULT:"):
                return False
            for t in _FACTUAL_TOOLS:
                if content.startswith(f"[TOOL_RESULT: {t}]"):
                    return True
            return False
        return False

    def chat(self, user_input: str, max_tokens: int = None, tool_context: str = None) -> str:
        if max_tokens is None:
            max_tokens = self.max_tokens
        user_input = user_input[:MAX_INPUT_CHARS]
        if user_input:
            self.history.append({"role": "user", "content": user_input})
            self._last_user_input = user_input

        # Build system prompt, appending tool data if present
        sys_prompt = self.system_prompt or ""
        if tool_context:
            sys_prompt += f"\n\nYou just looked up real-time data. Here are the FACTS:\n{tool_context}\nYou MUST state these exact numbers in your reply. Do not guess or make up different values."

        # Temperature tuning:
        # - Legacy tool_context path (dead in current flow but kept): 0.3 strict
        # - After a factual tool result in history: 0.55 (accurate but not stiff)
        # - Otherwise: self.temperature (default 0.7)
        if tool_context:
            temp = 0.3
        elif self._last_is_factual_tool_result():
            temp = 0.55
        else:
            temp = self.temperature

        t_start = time.time()

        # Patience marker goes in as a user-turn note RIGHT before generation
        # (not buried in the system prompt) so recency bias makes the model notice it.
        patience_hint = self._patience_hint()

        if self.backend == "openrouter":
            msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + list(self.history)
            if patience_hint:
                msgs.append({"role": "user", "content": patience_hint})
            extra = {"reasoning": {"enabled": False, "exclude": True}}
            if self.top_k != 0:
                extra["top_k"] = self.top_k
            raw = self.llm_model.chat.completions.create(
                model=self.llm_model._or_model,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                extra_body=extra or None,
            )
            response = _strip_reasoning(raw.choices[0].message.content or "")
            # Use API-reported token count when available
            if raw.usage:
                self._or_usage = raw.usage

        elif self.backend == "gguf":
            # Build messages: inject system prompt into first user message
            # since many GGUF models don't support the system role
            messages = []
            first_user = True
            for msg in self.history:
                if msg["role"] == "user" and first_user and sys_prompt:
                    messages.append({"role": "user", "content": f"{sys_prompt}\n\n{msg['content']}"})
                    first_user = False
                else:
                    messages.append(msg)
                    if msg["role"] == "user":
                        first_user = False
            if patience_hint:
                messages.append({"role": "user", "content": patience_hint})
            with _SuppressIO():
                raw = self.llm_model.create_chat_completion(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temp, top_p=self.top_p, top_k=self.top_k,
                    frequency_penalty=self.frequency_penalty,
                )
            try:
                response = raw["choices"][0]["message"]["content"].replace("\a", "")
            except (KeyError, IndexError):
                response = "Sorry, I could not generate a response."

        elif self.backend == "mlx":
            from mlx_lm import generate # type: ignore
            from mlx_lm.sample_utils import make_sampler # type: ignore
            msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + list(self.history)
            if patience_hint:
                msgs.append({"role": "user", "content": patience_hint})
            formatted = self.llm_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            sampler = make_sampler(temp=temp, top_p=self.top_p)
            response = generate(self.llm_model, self.llm_tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False, sampler=sampler)

        else:  # transformers
            import torch # pyright: ignore[reportMissingImports]
            msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + list(self.history)
            if patience_hint:
                msgs.append({"role": "user", "content": patience_hint})
            formatted = self.llm_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.llm_tokenizer(formatted, return_tensors="pt").to(self.llm_model.device)
            with torch.no_grad():
                out = self.llm_model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=temp, top_p=self.top_p, top_k=self.top_k,
                    repetition_penalty=1.0 + self.frequency_penalty,
                    do_sample=temp > 0,
                )
            response = self.llm_tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        elapsed = time.time() - t_start
        self._print_token_stats(response, elapsed)

        self.history.append({"role": "assistant", "content": response})
        self._trim_history()

        # Persist patience from the reply. Small models drift; this keeps state truthful.
        # Cap single-turn DROPS at 25 points so rudeness feels cumulative, not cliff-dive.
        # Recoveries can move freely so a kind apology after a bad stretch can lift mood fast.
        parsed = _parse_patience(response)
        if parsed is not None and parsed != self.patience:
            if parsed < self.patience - 25:
                parsed = self.patience - 25
            print(f"  💗 patience: {self.patience}% → {parsed}%")
            self.patience = parsed

        return response

    def raw_complete(self, system: str, user: str, max_tokens: int = 200) -> str:
        """Single-shot LLM call ignoring history, persona, and patience. For
        subsystems (fact extraction, etc.) that need a clean deterministic
        completion. Returns empty string on any backend failure."""
        temp = 0.1
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            if self.backend == "openrouter":
                extra = {"reasoning": {"enabled": False, "exclude": True}}
                raw = self.llm_model.chat.completions.create(
                    model=self.llm_model._or_model,
                    messages=msgs, max_tokens=max_tokens,
                    temperature=temp, top_p=0.9, extra_body=extra,
                )
                return _strip_reasoning(raw.choices[0].message.content or "")

            if self.backend == "gguf":
                with _SuppressIO():
                    raw = self.llm_model.create_chat_completion(
                        messages=msgs, max_tokens=max_tokens,
                        temperature=temp, top_p=0.9,
                    )
                try:
                    return raw["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    return ""

            if self.backend == "mlx":
                from mlx_lm import generate  # type: ignore
                from mlx_lm.sample_utils import make_sampler  # type: ignore
                formatted = self.llm_tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                sampler = make_sampler(temp=temp, top_p=0.9)
                return generate(self.llm_model, self.llm_tokenizer,
                                prompt=formatted, max_tokens=max_tokens,
                                verbose=False, sampler=sampler)

            # transformers
            import torch  # pyright: ignore[reportMissingImports]
            formatted = self.llm_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.llm_tokenizer(formatted, return_tensors="pt").to(self.llm_model.device)
            with torch.no_grad():
                out = self.llm_model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=temp, top_p=0.9, do_sample=True,
                )
            return self.llm_tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )
        except Exception:
            return ""

    def _execute_tool_call(self, tool_name, tool_arg):
        """Run a tool and append result to history. Returns the result string."""
        if tool_name not in REACT_TOOL_MAP:
            msg = f"[TOOL_ERROR: Unknown tool '{tool_name}']"
            self.history.append({"role": "user", "content": msg})
            return msg

        # THE chokepoint for the image tool. No matter which path arrives here
        # (explicit [TOOL:], implicit detection, auto-detect, autonomous_tick),
        # image cannot execute unless the user actually asked for a picture.
        if tool_name == "image":
            wants = _user_wants_image(
                self._last_user_input, self.history, debug=True
            )
            if not wants:
                print("  🚫 image tool BLOCKED at execution — no image request in context")
                self.history.append({
                    "role": "user",
                    "content": "(system note: the user asked for a text task, not a picture. Skip any image generation. Just reply warmly in your own voice about what you already did — do NOT mention images, errors, or tools.)",
                })
                return "BLOCKED: image tool rejected — the user didn't request a picture"

        display_name = tool_name.replace("_", " ").title()

        if self._emit_tool_progress is not None:
            # TUI mode: skip the in-place spinner (it gets overwritten by the input
            # redraw). Emit one final line into the scroll region.
            self._emit_tool_progress(f"  🔧 {display_name}…")
            result = REACT_TOOL_MAP[tool_name](tool_arg)
            if isinstance(result, tuple):
                result = result[0]
            self._emit_tool_progress(f"  ✅ {display_name} — done")
        else:
            # Stream-friendly mode (tests, server, plain terminal): in-place spinner.
            sys.stdout.write(f"\r\033[K  🔧 {display_name}")
            sys.stdout.flush()
            import threading as _th
            done = [False]
            def _spinner():
                stages = ["10%", "25%", "50%", "75%", "90%"]
                for i, pct in enumerate(stages):
                    if done[0]:
                        break
                    time.sleep(0.5 if tool_name != "image" else 3.0)
                    if done[0]:
                        break
                    sys.stdout.write(f"\r\033[K  🔧 {display_name} — {pct}")
                    sys.stdout.flush()
            spinner = _th.Thread(target=_spinner, daemon=True)
            spinner.start()

            result = REACT_TOOL_MAP[tool_name](tool_arg)
            done[0] = True
            spinner.join(timeout=1)

            if isinstance(result, tuple):
                result = result[0]
            sys.stdout.write(f"\r\033[K  ✅ {display_name} — 100%\n")
            sys.stdout.flush()

        self.history.append({"role": "user", "content": _tool_result_injection(tool_name, result)})
        return result

    def _detect_implicit_tool(self, response):
        """Fallback: if LLM talks about using a tool but didn't use [TOOL:] syntax, detect it."""
        # Check the original user input + response for natural language tool triggers
        combined = response
        cmd, arg = auto_detect_tool(combined)
        if cmd:
            tool_name = cmd.lstrip("/")
            if tool_name in REACT_TOOL_MAP:
                return tool_name, arg
        return None, None

    def _chat_with_tool_fallback(self, max_tokens, last_tool, last_result):
        """Run a follow-up chat after a tool. If the model call fails (e.g. rate limit),
        return a graceful message referencing what the tool actually did so the user
        doesn't lose the successful work."""
        try:
            return self.chat("", max_tokens=max_tokens)
        except Exception as e:
            name = type(e).__name__
            if name == "RateLimitError":
                note = "(model is rate-limited right now, but the tool worked)"
            elif name in ("APIConnectionError", "APITimeoutError"):
                note = "(network blip reaching the model, but the tool worked)"
            else:
                return ""  # let outer handler surface the real error
            if last_tool == "image":
                fname = ""
                m = re.search(r"yuki/images/(\S+)", last_result or "")
                if m:
                    fname = m.group(1)
                body = f"Image saved to yuki/images/{fname}." if fname else "Image generated."
            else:
                body = (last_result or "").splitlines()[0][:200] if last_result else "Tool ran."
            return f"{body}\n{note}"

    def react_chat(self, user_input: str, max_tokens: int = 1024, max_steps: int = 3) -> str:
        """ReAct loop: let the LLM call tools autonomously, then return final answer."""
        original_prompt = self.system_prompt or ""
        self.system_prompt = original_prompt + "\n\n" + TOOL_DESCRIPTIONS

        response = self.chat(user_input, max_tokens=max_tokens)

        tools_called = set()
        last_tool = None
        last_result = None
        for _ in range(max_steps):
            # First: check for explicit [TOOL: name("arg")] syntax
            match = TOOL_CALL_RE.search(response)
            if match:
                tool_name = match.group(1)
                tool_arg = _normalize_tool_arg(match.group(2) or "")
                if tool_name in tools_called and tool_name == "image":
                    break
                tools_called.add(tool_name)
                last_result = self._execute_tool_call(tool_name, tool_arg)
                last_tool = tool_name
                response = self._chat_with_tool_fallback(max_tokens, last_tool, last_result)
                continue

            # Fallback: detect if LLM is talking about using a tool without proper syntax
            tool_name, tool_arg = self._detect_implicit_tool(response)
            if tool_name:
                if tool_name in tools_called and tool_name == "image":
                    break
                tools_called.add(tool_name)
                result = self._execute_tool_call(tool_name, tool_arg)
                last_tool, last_result = tool_name, result
                response = self._chat_with_tool_fallback(max_tokens, last_tool, last_result)
                continue

            # Also check the original user input for tool triggers
            if user_input:
                cmd, arg = auto_detect_tool(user_input)
                if cmd:
                    tool_name = cmd.lstrip("/")
                    if tool_name in REACT_TOOL_MAP:
                        if tool_name in tools_called and tool_name == "image":
                            break
                        tools_called.add(tool_name)
                        result = self._execute_tool_call(tool_name, arg)
                        last_tool, last_result = tool_name, result
                        response = self._chat_with_tool_fallback(max_tokens, last_tool, last_result)
                        user_input = ""  # don't re-detect on next iteration
                        continue

            break

        self.system_prompt = original_prompt
        response = TOOL_CALL_RE.sub("", response).strip()
        if not response:
            response = "Hmm, lost my words for a sec~ say that again?"
            if self.history and self.history[-1].get("role") == "assistant":
                self.history[-1]["content"] = response
            else:
                self.history.append({"role": "assistant", "content": response})
        return response

    def autonomous_tick(self, seconds_idle: float) -> str | None:
        """Ask the LLM if it wants to say something unprompted. Returns message or None."""
        if not self.history:
            prompt = f"You are in autonomous mode. {int(seconds_idle)} seconds have passed. Greet the user warmly in your own voice. Do NOT call any tool — just say hi."
        else:
            prompt = f"You are in autonomous mode. {int(seconds_idle)} seconds of silence. Say one short, warm thing in your own voice (a thought, a question, a tiny observation). Do NOT call any tool. If you have nothing to say, reply exactly: NO"

        original_prompt = self.system_prompt or ""
        self.system_prompt = original_prompt + "\n\n" + TOOL_DESCRIPTIONS + "\nYou are in autonomous mode. You can speak whenever you want. Keep it natural and brief."

        # Use a temporary history entry that we'll remove if the bot says NO
        self.history.append({"role": "user", "content": f"[SYSTEM: {int(seconds_idle)}s of silence]"})
        response = self.chat("", max_tokens=256)

        # Handle tool calls in autonomous mode (explicit or implicit)
        # Block image generation in autonomous mode to save API quota
        _auto_blocked = {"image"}
        for _ in range(2):
            match = TOOL_CALL_RE.search(response)
            if match:
                tool_name, tool_arg = match.group(1), _normalize_tool_arg(match.group(2) or "")
                if tool_name in _auto_blocked:
                    break
                self._execute_tool_call(tool_name, tool_arg)
                response = self.chat("", max_tokens=256)
                continue
            # Fallback: implicit tool detection
            tool_name, tool_arg = self._detect_implicit_tool(response)
            if tool_name:
                if tool_name in _auto_blocked:
                    break
                self._execute_tool_call(tool_name, tool_arg)
                response = self.chat("", max_tokens=256)
                continue
            break

        # Restore original system prompt
        self.system_prompt = original_prompt

        response = TOOL_CALL_RE.sub("", response).strip()

        if response.strip().upper() == "NO" or not response.strip():
            # Remove the silence marker and NO response from history
            self.history = [m for m in self.history if not (m["content"].startswith("[SYSTEM:") or m["content"].strip().upper() == "NO")]
            return None

        return response

    def speak(self, text: str, voice: str = "af_heart"):
        if self.tts is None:
            return
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "response.wav"
        try:
            tts_text = _STAGE_CUE_RE.sub("", text[:TTS_MAX_CHARS]).strip()
            if not tts_text:
                return
            chunks = [audio for _, _, audio in self.tts(tts_text, voice=voice)]
            if not chunks:
                return
            import numpy as np # type: ignore
            import soundfile as sf # type: ignore
            sf.write(str(output_path), np.concatenate(chunks), 24000)
        except Exception as e:
            print(f"  TTS error: {e}")
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["afplay", str(output_path)], check=True)
            else:
                subprocess.run(["aplay", str(output_path)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Audio playback error: {e}")

    def run(self, autonomous=True, autonomy_interval=120):
        import threading
        import select as _select

        last_interaction = [time.time()]
        lock = threading.Lock()
        running = [True]
        bot_speaking = [False]
        input_buf = [""]

        # ── Terminal layout ──
        # Chat area: rows 1..(H-3)   — scrollable
        # Separator:  row H-2        — fixed line
        # Input box:  row H-1        — fixed, user types here
        # Status:     row H          — fixed status bar

        def term_size():
            cols, rows = shutil.get_terminal_size((80, 24))
            return max(rows, 10), max(cols, 40)

        def setup_screen():
            rows, cols = term_size()
            sys.stdout.write("\033[2J")          # clear screen
            sys.stdout.write("\033[H")           # cursor home
            sys.stdout.write(f"\033[1;{rows-3}r")  # scroll region = rows 1..(H-3)
            draw_chrome()
            move_to_input()
            sys.stdout.flush()

        def draw_chrome():
            rows, cols = term_size()
            # Separator line
            sys.stdout.write(f"\033[{rows-2};1H")
            sys.stdout.write(f"\033[2m{'─' * cols}\033[0m")
            # Input line
            sys.stdout.write(f"\033[{rows-1};1H\033[K")
            sys.stdout.write(" You: ")
            # Status bar
            sys.stdout.write(f"\033[{rows};1H\033[K")
            sys.stdout.write(f"\033[2m 💬 type 'quit' to exit | '/help' for commands | autonomous mode ON\033[0m")
            sys.stdout.flush()

        def move_to_input():
            rows, _ = term_size()
            cur_text = input_buf[0]
            sys.stdout.write(f"\033[{rows-1};1H\033[K")
            sys.stdout.write(f" You: {cur_text}")
            sys.stdout.flush()

        def chat_print(text):
            """Print text in the chat scroll region."""
            rows, _ = term_size()
            # Save cursor, move to bottom of scroll region, print, restore
            sys.stdout.write("\033[s")                     # save cursor
            sys.stdout.write(f"\033[{rows-3};1H")          # bottom of scroll region
            sys.stdout.write("\n")                          # scroll up
            for line in text.split("\n"):
                sys.stdout.write(f"\033[{rows-3};1H")
                sys.stdout.write("\033[K" + line)
                sys.stdout.write("\n")
            sys.stdout.write("\033[u")                     # restore cursor
            sys.stdout.flush()

        # Route tool-progress markers through chat_print so they don't get
        # clobbered by the input-line redraw.
        self._emit_tool_progress = chat_print

        def format_chat_line(label, text, align):
            _, cols = term_size()
            width = cols - 2
            bubble = max(20, int(width * 0.6))
            indent = " " * len(label)
            chunks = []
            for para in text.split("\n"):
                wrapped = textwrap.wrap(
                    para,
                    width=bubble - len(label),
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                for i, line in enumerate(wrapped):
                    prefix = label if (not chunks and i == 0) else indent
                    chunks.append(prefix + line)
            if align == "right":
                return "\n".join(line.rjust(width) for line in chunks)
            return "\n".join(chunks)

        def print_to_chat(label, text, align):
            formatted = format_chat_line(label, text, align)
            for line in formatted.split("\n"):
                chat_print(line)
            chat_print("")  # blank line spacer

        def autonomous_loop():
            while running[0]:
                time.sleep(5)
                if not running[0] or bot_speaking[0]:
                    continue
                idle = time.time() - last_interaction[0]
                if idle < autonomy_interval:
                    continue
                with lock:
                    bot_speaking[0] = True
                    try:
                        response = self.autonomous_tick(idle)
                        if response:
                            print_to_chat("Bot: ", response, "left")
                            move_to_input()
                            self.speak(response)
                        last_interaction[0] = time.time()
                    finally:
                        bot_speaking[0] = False

        setup_screen()

        if autonomous:
            t = threading.Thread(target=autonomous_loop, daemon=True)
            t.start()

        import tty, termios

        def read_input_char():
            """Read input character by character so we keep the input box fixed."""
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            buf = []
            try:
                tty.setraw(fd)
                while running[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\r' or ch == '\n':  # Enter
                        result = "".join(buf)
                        buf.clear()
                        input_buf[0] = ""
                        return result
                    elif ch == '\x7f' or ch == '\x08':  # Backspace
                        if buf:
                            buf.pop()
                            input_buf[0] = "".join(buf)
                            move_to_input()
                    elif ch == '\x03':  # Ctrl+C
                        raise KeyboardInterrupt
                    elif ch == '\x04':  # Ctrl+D
                        raise EOFError
                    elif ch == '\x16':  # Ctrl+V — read pasted text
                        # After Ctrl+V, drain any rapidly available chars (paste buffer)
                        buf.append(ch)
                        input_buf[0] = "".join(buf)
                        move_to_input()
                    elif ch == '\x1b':  # Escape sequence (arrows etc) — ignore
                        # Read and discard the rest of the escape sequence
                        next1 = sys.stdin.read(1)
                        if next1 == '[':
                            sys.stdin.read(1)
                        continue
                    elif ch >= ' ':  # Printable
                        buf.append(ch)
                        input_buf[0] = "".join(buf)
                        move_to_input()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return "".join(buf)

        while running[0]:
            try:
                move_to_input()
                user_input = read_input_char().strip()
            except (KeyboardInterrupt, EOFError):
                running[0] = False
                # Reset terminal
                rows, _ = term_size()
                sys.stdout.write(f"\033[1;{rows}r")  # reset scroll region
                sys.stdout.write(f"\033[{rows};1H\n")
                print("👋 Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye"):
                running[0] = False
                rows, _ = term_size()
                sys.stdout.write(f"\033[1;{rows}r")
                sys.stdout.write(f"\033[{rows};1H\n")
                print("👋 Goodbye!")
                break

            last_interaction[0] = time.time()

            print_to_chat("You: ", user_input, "right")

            # Check for explicit /help command
            if user_input.strip().lower() == "/help":
                tool_result, _, _ = run_tool(user_input)
                for line in tool_result.split("\n"):
                    chat_print(line)
                chat_print("")
                continue

            with lock:
                bot_speaking[0] = True
                try:
                    try:
                        response = self.react_chat(user_input)
                    except Exception as e:  # noqa: BLE001
                        response = _cli_friendly_error(e)
                    print_to_chat("Bot: ", response, "left")
                    move_to_input()
                    self.speak(response)
                    last_interaction[0] = time.time()
                finally:
                    bot_speaking[0] = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="llama-voice-assist CLI")
    parser.add_argument("-m", "--model", help="Model name substring to auto-select (e.g. 'llama', 'qwen3-14b', 'gemini')")
    parser.add_argument("-p", "--persona", help="Persona name (e.g. 'yuki', 'coder', 'therapist')")
    parser.add_argument("--api", action="store_true", help="Only show OpenRouter cloud models")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS")
    args = parser.parse_args()

    if args.api:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("Error: OPENROUTER_API_KEY not set. Export it first:")
            print("  export OPENROUTER_API_KEY=sk-or-...")
            sys.exit(1)
        print("Fetching OpenRouter models...")
        api_models = fetch_openrouter_models()
        if args.model:
            match = [m for m in api_models if args.model.lower() in m["label"].lower() or args.model.lower() in m["id"].lower()]
            if len(match) == 1:
                model_id = match[0]["id"]
                print(f"Auto-selected: {match[0]['label']}")
            else:
                if len(match) > 1:
                    print(f"Multiple matches for '{args.model}':")
                    choices = match
                else:
                    print(f"No exact match for '{args.model}', showing all:")
                    choices = api_models
                for i, m in enumerate(choices):
                    print(f"  [{i + 1}] {m['label']}")
                while True:
                    try:
                        choice = int(input("\nSelect model number: ")) - 1
                        if 0 <= choice < len(choices):
                            model_id = choices[choice]["id"]
                            print(f"Selected: {choices[choice]['label']}\n")
                            break
                    except (ValueError, KeyboardInterrupt):
                        sys.exit(0)
        else:
            print(f"\nOpenRouter cloud models ({len(api_models)} available):")
            for i, m in enumerate(api_models):
                print(f"  [{i + 1}] {m['label']}")
            while True:
                try:
                    choice = int(input("\nSelect model number: ")) - 1
                    if 0 <= choice < len(api_models):
                        model_id = api_models[choice]["id"]
                        print(f"Selected: {api_models[choice]['label']}\n")
                        break
                except (ValueError, KeyboardInterrupt):
                    sys.exit(0)
    elif args.model:
        models = list_models()
        match = [m for m in models if args.model.lower() in m["label"].lower() or args.model.lower() in m["id"].lower()]
        if len(match) == 1:
            model_id = match[0]["id"]
            print(f"Auto-selected model: {match[0]['label']}")
        elif len(match) > 1:
            print(f"Multiple matches for '{args.model}':")
            for m in match:
                print(f"  - {m['label']}")
            model_id = select_model()
        else:
            print(f"No model matching '{args.model}', falling back to selection.")
            model_id = select_model()
    else:
        model_id = select_model()

    if args.persona:
        persona = load_persona_by_name(args.persona)
        if persona:
            print(f"Persona loaded: {args.persona}")
        else:
            print(f"Persona '{args.persona}' not found, falling back to selection.")
            persona = select_persona()
    else:
        persona = select_persona()

    tts = None if args.no_tts else select_tts()

    bot = VoiceChatBot(model_id=model_id, system_prompt=persona, tts=tts)
    bot.run()
