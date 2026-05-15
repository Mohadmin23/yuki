import os
import re
import sys
import io
import time
import json
import uuid
import logging
import base64
import asyncio
import numpy as np
import soundfile as sf
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, field_validator
from granian import Granian

sys.path.append(str(Path(__file__).parent.parent))
from ms_llama import (
    VoiceChatBot, TTS_MAX_CHARS, MAX_INPUT_CHARS, _STAGE_CUE_RE,
    OPENROUTER_MODELS, TOOLS, run_tool,
    select_model, select_persona, select_tts,
    list_models, list_personas, list_tts_engines,
    load_persona_by_name, init_tts,
    _load_users, _save_users, _get_view, _find_candidates,
    _add_fact_to_slot, _create_slot, _hash_fact, _default_user_id,
    MEMORY_NAME_CATEGORIES, MEMORY_LIST_CATEGORIES,
    MULTI_USER_MODE,
    verify_and_advance,
)
import episodic

EPISODIC_MARKER = "\n[EPISODIC MEMORY"

IMAGES_DIR = Path(__file__).parent.parent / "yuki" / "images"
_IMG_PATH_RE = re.compile(r"yuki/images/([\w\-.]+\.(?:png|jpg|jpeg|webp|gif))", re.IGNORECASE)

logger = logging.getLogger(__name__)

bot: VoiceChatBot = None
last_interaction_time: float = 0
autonomous_enabled: bool = False
AUTONOMY_INTERVAL = 15  # seconds
headless: bool = True  # Web UI handles all config by default

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, last_interaction_time, headless
    if not headless:
        # Interactive terminal selection (original behavior)
        model_id = select_model()
        persona = select_persona()
        tts = select_tts()
        bot = VoiceChatBot(model_id=model_id, system_prompt=persona, tts=tts)
        last_interaction_time = time.time()
    yield

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v[:MAX_INPUT_CHARS]

class InitRequest(BaseModel):
    model_id: str
    persona: str = ""
    tts_key: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    max_tokens: int = 1024

class ApiKeyRequest(BaseModel):
    key: str

class ParamsRequest(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    max_tokens: int = 1024

class MemoryItemRequest(BaseModel):
    content: str

class ChatRenameRequest(BaseModel):
    title: str

# ── Data persistence ──

DATA_DIR = Path(__file__).parent.parent / "data"
CHATS_DIR = DATA_DIR / "chats"
MEMORY_FILE = DATA_DIR / "memory.json"

current_chat_id: str | None = None

def _ensure_dirs():
    CHATS_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_MARKER = "\n\n[Stored facts about a previously-known user"


def _flatten_view(view: dict, salt: str) -> list[dict]:
    """Turn a plaintext slot view into [{id, category, content}] for the sidebar.
    IDs are the first 8 hex chars of the fact's hash so delete-by-id can
    reconstruct the (category, value) pair."""
    items = []
    for category, val in view.items():
        if category in ("created_at", "last_seen", "fact_hashes"):
            continue
        values = val if isinstance(val, list) else ([val] if val else [])
        for v in values:
            if not v:
                continue
            h = _hash_fact(category, v, salt)
            items.append({"id": h[:12], "category": category, "content": v})
    return items


def _load_memory() -> list[dict]:
    """Legacy-shape list for the /api/memory endpoints and sidebar: items from
    the currently-unlocked slot only. Returns [] when no session is bound."""
    if bot is None or bot.unlocked_user_id is None:
        return []
    store = _load_users()
    view = _get_view(store, bot.unlocked_user_id)
    if not view:
        return []
    return _flatten_view(view, store["salt"])


def _memory_as_prompt(items: list[dict]) -> str:
    """Render the bound slot as a compact block for the system prompt.
    Header text differs between simple mode and multi-user mode; the body
    is the same list of facts in both.

    Anti-confabulation guardrail is included in BOTH headers — Yuki must only
    reference facts that are actually listed below, never invent shared history
    (past meetings, haikus, inside jokes) that aren't in the list."""
    if not items:
        header = (
            f"{MEMORY_MARKER} — you have no stored facts about this user yet. "
            "Do NOT invent or imply any past shared history, prior conversations, "
            "inside jokes, or things you remember — there is nothing to remember. "
            "Meet them with warm first-time energy.]"
        )
        return header
    grouped: dict[str, list[str]] = {}
    for it in items:
        grouped.setdefault(it["category"], []).append(it["content"])
    lines = []
    for cat, vals in grouped.items():
        if len(vals) == 1:
            lines.append(f"- {cat}: {vals[0]}")
        else:
            lines.append(f"- {cat}: {', '.join(vals)}")
    body = "\n".join(lines)
    if MULTI_USER_MODE:
        prelude = (
            f"{MEMORY_MARKER} — this person has passed the vibe check this session. "
            "The facts below are BACKGROUND KNOWLEDGE about them."
        )
    else:
        prelude = (
            f"{MEMORY_MARKER} — BACKGROUND KNOWLEDGE about the person you're chatting with."
        )
    header = (
        f"{prelude}\n"
        "USER-INITIATION RULE: do NOT introduce these topics into the "
        "conversation yourself. Only reference a fact if the USER mentions it "
        "in their CURRENT message or the IMMEDIATELY PRECEDING message. "
        "References to topics from older turns are NOT user-initiated anymore "
        "— treat them as stored facts again. If they ask a generic question, "
        "answer the generic question — do not steer toward stored topics. "
        "'Tangentially related' does NOT count. Surface associations (color → "
        "black → black cats; writing → manga → Usogui) do NOT count. If "
        "unsure whether the user brought it up in the last two messages, "
        "don't reference it.\n"
        "DECORATION IS STILL INJECTION: answering the user's question cleanly "
        "and THEN adding a stored-fact reference as a flourish, garnish, or "
        "example is STILL a violation. Do not season answers with stored "
        "facts. Do not use stored facts as examples in lists, advice, or "
        "metaphors unless the user invoked them in the last two messages.\n"
        "BAD: User: 'I'm thinking of getting a pet.' You: 'Black cats are the "
        "best!' (user did not mention cats)\n"
        "BAD: User: 'What's your favorite color?' You: 'Purple! Pairs "
        "perfectly with black cats.' (color is not a cat topic; the cat line "
        "is a flourish)\n"
        "BAD: User: 'Any advice for creative writing?' You: '...Remix prompts "
        "from your faves (Usogui mind-games into a heist story?)...' (user "
        "did not mention Usogui in the current or previous message; using it "
        "as example material is still injection)\n"
        "GOOD: User: 'What's your favorite color?' You: 'Purple, the deep "
        "cosmic kind.' (just answers)\n"
        "GOOD: User: 'I'm thinking of getting a pet.' You: 'Ooh exciting! "
        "What kind are you leaning toward?' (no stored facts injected)\n"
        "GOOD: User: 'I might get a black cat.' You: 'Ahh great choice~ I "
        "know you love those.' (user brought up the stored topic in the "
        "current message)\n"
        "Also: do NOT open responses by naming the user or their handle, do "
        "NOT use stored facts as greeting flourishes or pet-names. "
        "Background, not foreground.\n"
        "CONFABULATION GUARD: only reference what is literally listed below "
        "— do NOT invent shared past events, prior conversations, inside "
        "jokes, haikus you wrote together, or things 'you remember together' "
        "that aren't in the list. If the user brings up something that isn't "
        "here, say you don't remember that specifically rather than "
        "inventing.]"
    )
    return f"{header}\n{body}"


def _locked_stub(store: dict) -> str:
    """System-prompt snippet shown when no user is unlocked.
    Never names or quotes any stored user — tells Yuki only that memory exists."""
    n = len(store.get("users", {}))
    if n == 0:
        return (
            f"{MEMORY_MARKER} — no one is stored yet. Treat whoever is talking to you "
            "as a brand-new friend. If they share details you'd normally want to remember, "
            "use the remember tool — it'll be routed to them automatically.]"
        )
    return (
        f"{MEMORY_MARKER} — you have {n} known user(s) on file, but you CANNOT see who, "
        "and you CANNOT see their facts until this session verifies them. Follow the "
        "RECOGNIZING THE USER rules in your persona: warm first-meeting energy, do the "
        "vibe-check yourself, never volunteer details, never confirm on name alone. "
        "If the system injects a NARROW hint, phrase the ask in your own voice.]"
    )


def _narrowing_hint(candidate_count: int) -> str:
    """Injected when candidates > 1. Yuki phrases the ask herself; this is just the signal."""
    return (
        f"{MEMORY_MARKER} NARROW — {candidate_count} possible people match what's been said "
        "so far, which isn't enough to be sure. Ask in your own voice for another detail "
        "(favorite character, favorite thing, anything specific). Do not name candidates, "
        "do not say 'multiple matches' out loud. Just be playfully uncertain.]"
    )

def _save_chat(chat_id: str, title: str, messages: list[dict]):
    _ensure_dirs()
    path = CHATS_DIR / f"{chat_id}.json"
    # Preserve any existing owner; otherwise stamp the currently-unlocked user
    # so chats started before vibe-check become claimed on first post-unlock save.
    existing = _load_chat_raw(chat_id)
    if existing and existing.get("user_id"):
        owner = existing["user_id"]
    elif bot is not None and bot.unlocked_user_id is not None:
        owner = bot.unlocked_user_id
    else:
        owner = None
    data = {
        "id": chat_id,
        "title": title,
        "updated": time.time(),
        "user_id": owner,
        "messages": messages,
    }
    path.write_text(json.dumps(data, indent=2))

def _load_chat_raw(chat_id: str) -> dict | None:
    """Read a chat file with no access checks. Internal use only."""
    path = CHATS_DIR / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None

def _chat_accessible(data: dict) -> bool:
    """A chat is accessible iff: it's owned by the currently-unlocked user, or
    it's unowned AND is the active session chat (the pending pre-vibe-check one).

    In simple mode (MULTI_USER_MODE=False), every chat is accessible — the
    ownership gate is off so all chats show in the sidebar regardless of user_id."""
    if not MULTI_USER_MODE:
        return True
    owner = data.get("user_id")
    if owner is None:
        return data.get("id") == current_chat_id
    return bot is not None and bot.unlocked_user_id == owner

def _load_chat(chat_id: str) -> dict | None:
    """Access-checked read. Returns None for missing *or* inaccessible chats so
    the API can't distinguish 'not yours' from 'doesn't exist'."""
    data = _load_chat_raw(chat_id)
    if data is None or not _chat_accessible(data):
        return None
    return data

def _list_chats() -> list[dict]:
    _ensure_dirs()
    # Simple mode: no ownership filter, every chat is visible.
    # Multi-user mode: locked session sees nothing, unlocked sees only its own chats.
    if MULTI_USER_MODE:
        if bot is None or bot.unlocked_user_id is None:
            return []
        uid = bot.unlocked_user_id
    else:
        uid = None  # sentinel: return everything
    chats = []
    for f in CHATS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if uid is not None and data.get("user_id") != uid:
                continue
            chats.append({"id": data["id"], "title": data["title"], "updated": data["updated"]})
        except Exception:
            pass
    chats.sort(key=lambda c: c["updated"], reverse=True)
    return chats

def _auto_title(messages: list[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            text = msg["content"].strip()
            return text[:50] + ("..." if len(text) > 50 else "")
    return "New Chat"

def _friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "RateLimitError":
        provider = ""
        try:
            meta = exc.body["error"]["metadata"]
            provider = f" ({meta.get('provider_name', '')})" if meta.get("provider_name") else ""
        except Exception:
            pass
        return f"Rate-limited upstream{provider}. This free model is busy — retry in a moment or switch models."
    if name == "AuthenticationError":
        return "OpenRouter API key rejected. Check your key in settings."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Couldn't reach OpenRouter. Check your network and try again."
    if name == "BadRequestError":
        return f"Model rejected the request: {exc}"
    return "An internal error occurred."


def _error_status(exc: Exception) -> int:
    name = type(exc).__name__
    if name == "RateLimitError":
        return 429
    if name == "AuthenticationError":
        return 401
    if name in ("APIConnectionError", "APITimeoutError"):
        return 502
    if name == "BadRequestError":
        return 400
    return 500


def _synthesize_audio(text: str) -> str:
    """Generate base64 WAV audio from text. Returns empty string if no TTS."""
    if bot.tts is None:
        return ""
    tts_text = _STAGE_CUE_RE.sub("", text[:TTS_MAX_CHARS]).strip()
    if not tts_text:
        return ""
    chunks = [audio for _, _, audio in bot.tts(tts_text, voice="af_heart")]
    if not chunks:
        return ""
    audio_data = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, audio_data, 24000, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "Yuki_Chat_v2-2.html").read_text()

@app.get("/m", response_class=HTMLResponse)
async def root_mobile():
    return (Path(__file__).parent / "Yuki_Mobile.html").read_text()

_STATIC_DIR = Path(__file__).parent / "static"

@app.get("/static/{name}")
async def static_asset(name: str):
    # Only serve known asset names; refuse any path traversal.
    safe = Path(name).name
    fp = _STATIC_DIR / safe
    if not fp.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    if safe.endswith(".webmanifest"):
        media = "application/manifest+json"
    elif safe.endswith(".png"):
        media = "image/png"
    elif safe.endswith(".js"):
        media = "application/javascript"
    else:
        media = "application/octet-stream"
    # Service worker must be served from the same scope and with no-cache so
    # iOS picks up updates; other static assets can cache normally.
    headers = {"Cache-Control": "no-cache"} if safe == "sw.js" else {"Cache-Control": "public, max-age=86400"}
    return FileResponse(fp, media_type=media, headers=headers)

@app.get("/sw.js")
async def service_worker():
    """Serve the SW from the root so its scope covers the whole origin."""
    fp = _STATIC_DIR / "sw.js"
    return FileResponse(
        fp, media_type="application/javascript",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    )

@app.get("/manifest.webmanifest")
async def root_manifest():
    fp = _STATIC_DIR / "manifest.webmanifest"
    return FileResponse(fp, media_type="application/manifest+json")

@app.get("/apple-touch-icon.png")
async def apple_touch_icon():
    return FileResponse(_STATIC_DIR / "icon-180.png", media_type="image/png")

@app.get("/apple-touch-icon-precomposed.png")
async def apple_touch_icon_pre():
    return FileResponse(_STATIC_DIR / "icon-180.png", media_type="image/png")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(_STATIC_DIR / "icon-180.png", media_type="image/png")

# ── Config endpoints (for web-based setup) ──

@app.get("/api/status")
async def api_status():
    return JSONResponse({"ready": bot is not None})

@app.get("/api/models")
async def api_models():
    models = await asyncio.to_thread(list_models)
    return JSONResponse({"models": models})

@app.get("/api/personas")
async def api_personas():
    personas = list_personas()
    return JSONResponse({"personas": personas})

@app.get("/api/tts")
async def api_tts():
    engines = list_tts_engines()
    return JSONResponse({"engines": engines})

# ── Tools registry (for the slash-command menu) ──

_TOOL_ICONS = {
    "/search": "search", "/calc": "calculator", "/weather": "cloud",
    "/time": "clock", "/read": "file", "/shell": "terminal",
    "/hardware": "cpu", "/yuki-write": "pencil", "/yuki-read": "doc",
    "/yuki-list": "list", "/yuki-delete": "trash", "/yuki-append": "plus",
    "/image": "image", "/remember": "star",
}

def _tool_template(name: str, usage: str) -> str:
    """Return a prefilled template users can edit in the composer."""
    rest = usage[len(name):].strip()  # e.g. "<query>" or "<filename>|<content>"
    return f"{name} " if not rest else f"{name} "

@app.get("/api/tools")
async def api_tools():
    items = []
    for name, tool in TOOLS.items():
        items.append({
            "name": name,
            "help": tool["help"],
            "usage": tool["usage"],
            "icon": _TOOL_ICONS.get(name, "tool"),
            "template": _tool_template(name, tool["usage"]),
        })
    return JSONResponse({"tools": items})

# ── Media ──

def _safe_image_name(filename: str) -> Path | None:
    """Resolve filename inside IMAGES_DIR, blocking path traversal."""
    name = filename.strip().replace("\\", "/").split("/")[-1]
    if not name or name.startswith("."):
        return None
    p = (IMAGES_DIR / name).resolve()
    try:
        p.relative_to(IMAGES_DIR.resolve())
    except ValueError:
        return None
    return p if p.is_file() else None

@app.get("/media/{filename}")
async def media_file(filename: str):
    p = _safe_image_name(filename)
    if p is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(p)

@app.get("/api/media")
async def api_media():
    if not IMAGES_DIR.exists():
        return JSONResponse({"items": []})
    items = []
    for f in sorted(IMAGES_DIR.iterdir(), reverse=True):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            continue
        stat = f.stat()
        items.append({
            "type": "image",
            "filename": f.name,
            "url": f"/media/{f.name}",
            "size": stat.st_size,
            "created_at": stat.st_mtime,
        })
    return JSONResponse({"items": items})

def _collect_recent_media(history: list[dict], since: int = 0) -> list[dict]:
    """Scan history[since:] for yuki/images/* references; build media list."""
    media = []
    seen = set()
    for msg in history[since:]:
        content = msg.get("content", "") or ""
        for m in _IMG_PATH_RE.finditer(content):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            p = IMAGES_DIR / name
            if not p.is_file():
                continue
            stat = p.stat()
            media.append({
                "type": "image",
                "filename": name,
                "url": f"/media/{name}",
                "size": stat.st_size,
                "created_at": stat.st_mtime,
            })
    return media

@app.post("/api/apikey")
async def api_apikey(req: ApiKeyRequest):
    key = req.key.strip()
    if not key:
        # Clear the key
        os.environ.pop("OPENROUTER_API_KEY", None)
        return JSONResponse({"ok": True, "has_key": False})
    os.environ["OPENROUTER_API_KEY"] = key
    return JSONResponse({"ok": True, "has_key": True})

@app.get("/api/apikey/status")
async def api_apikey_status():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    return JSONResponse({"has_key": bool(key), "preview": f"{key[:8]}..." if len(key) > 8 else ""})

@app.post("/api/init")
async def api_init(req: InitRequest):
    global bot, last_interaction_time, autonomous_enabled
    try:
        persona_text = None
        if req.persona:
            persona_text = await asyncio.to_thread(load_persona_by_name, req.persona)
        tts = await asyncio.to_thread(init_tts, req.tts_key) if req.tts_key else None
        bot = await asyncio.to_thread(
            lambda: VoiceChatBot(
                model_id=req.model_id, system_prompt=persona_text, tts=tts,
                temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                frequency_penalty=req.frequency_penalty, max_tokens=req.max_tokens,
            )
        )
        last_interaction_time = time.time()
        autonomous_enabled = False
        _inject_memory(bot)
        # Auto-create a chat if none active
        global current_chat_id
        if not current_chat_id:
            current_chat_id = uuid.uuid4().hex[:12]
            _save_chat(current_chat_id, "New Chat", [])
        return JSONResponse({"ok": True, "chat_id": current_chat_id})
    except Exception as e:
        logger.exception("Error in /api/init")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/params")
async def api_params(req: ParamsRequest):
    """Update sampling params without reloading the model."""
    if bot is None:
        return JSONResponse({"ok": False, "error": "No model loaded"}, status_code=400)
    bot.temperature = req.temperature
    bot.top_p = req.top_p
    bot.top_k = req.top_k
    bot.frequency_penalty = req.frequency_penalty
    bot.max_tokens = req.max_tokens
    return JSONResponse({"ok": True})

@app.get("/api/params")
async def api_params_get():
    """Get current sampling params."""
    if bot is None:
        return JSONResponse({"temperature": 0.7, "top_p": 0.9, "top_k": 50, "frequency_penalty": 0.0, "max_tokens": 1024})
    return JSONResponse({
        "temperature": bot.temperature, "top_p": bot.top_p, "top_k": bot.top_k,
        "frequency_penalty": bot.frequency_penalty, "max_tokens": bot.max_tokens,
    })


# ── Chat history endpoints ──

@app.get("/api/chats")
async def api_chats_list():
    return JSONResponse({"chats": _list_chats()})

@app.post("/api/chats/new")
async def api_chats_new():
    global current_chat_id
    chat_id = uuid.uuid4().hex[:12]
    current_chat_id = chat_id
    if bot is not None:
        bot.history = []
        _reset_session_lock(bot)
        _inject_memory(bot)
    _save_chat(chat_id, "New Chat", [])
    return JSONResponse({"id": chat_id})

@app.get("/api/chats/{chat_id}")
async def api_chats_get(chat_id: str):
    data = _load_chat(chat_id)
    if data is None:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    return JSONResponse(data)

@app.post("/api/chats/{chat_id}/load")
async def api_chats_load(chat_id: str):
    global current_chat_id
    data = _load_chat(chat_id)
    if data is None:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    current_chat_id = chat_id
    if bot is not None:
        bot.history = data.get("messages", [])
        # Loaded chat history is just text — not proof of identity. Start locked
        # and let the verification hook re-unlock as the user continues talking.
        _reset_session_lock(bot)
        _inject_memory(bot)
    return JSONResponse({"ok": True, "messages": data.get("messages", []), "title": data.get("title", "")})

@app.delete("/api/chats/{chat_id}")
async def api_chats_delete(chat_id: str):
    global current_chat_id
    # Access-checked: missing or not-yours both look like 404.
    if _load_chat(chat_id) is None:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    path = CHATS_DIR / f"{chat_id}.json"
    if path.exists():
        path.unlink()
    if current_chat_id == chat_id:
        current_chat_id = None
        if bot is not None:
            bot.history = []
    return JSONResponse({"ok": True})

@app.post("/api/chats/{chat_id}/rename")
async def api_chats_rename(chat_id: str, req: ChatRenameRequest):
    data = _load_chat(chat_id)
    if data is None:
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    data["title"] = req.title.strip()[:100]
    _save_chat(chat_id, data["title"], data["messages"])
    return JSONResponse({"ok": True})

# ── Memory endpoints ──
# All operations target the currently-unlocked slot. When the session is
# locked, GET returns empty and writes are rejected — the sidebar cannot
# touch memory it hasn't been verified into.

@app.get("/api/memory")
async def api_memory_get():
    locked = (bot is None or bot.unlocked_user_id is None)
    return JSONResponse({"items": _load_memory(), "locked": locked})


@app.post("/api/memory")
async def api_memory_add(req: MemoryItemRequest):
    if bot is None or bot.unlocked_user_id is None:
        return JSONResponse({"ok": False, "error": "Memory is locked — no verified user in this session."}, status_code=403)
    raw = (req.content or "").strip()[:500]
    if not raw:
        return JSONResponse({"ok": False, "error": "Empty content"}, status_code=400)
    # Accept "category: value" to pin category, else drop into notes.
    if ":" in raw:
        cat, _, val = raw.partition(":")
        category = cat.strip().lower().replace(" ", "_")
        value = val.strip()
        if not value or category not in (MEMORY_LIST_CATEGORIES | {"handle", "name", "age", "location"}):
            category, value = "notes", raw
    else:
        category, value = "notes", raw
    store = _load_users()
    added = _add_fact_to_slot(store, bot.unlocked_user_id, category, value)
    if added:
        _save_users(store)
    _inject_memory(bot)
    item_id = _hash_fact(category, value, store["salt"])[:12]
    return JSONResponse({"ok": True, "item": {"id": item_id, "category": category, "content": value}})


@app.delete("/api/memory/{item_id}")
async def api_memory_delete(item_id: str):
    if bot is None or bot.unlocked_user_id is None:
        return JSONResponse({"ok": False, "error": "Memory is locked."}, status_code=403)
    store = _load_users()
    slot = store["users"].get(bot.unlocked_user_id)
    if slot is None:
        return JSONResponse({"ok": True})
    # Find the (category, value) whose hash prefix matches item_id and remove it
    # from both the plaintext list and fact_hashes.
    salt = store["salt"]
    removed = False
    for category, val in list(slot.items()):
        if category in ("created_at", "last_seen", "fact_hashes"):
            continue
        values = val if isinstance(val, list) else ([val] if val else [])
        for v in list(values):
            h = _hash_fact(category, v, salt)
            if h.startswith(item_id):
                if isinstance(val, list):
                    val.remove(v)
                else:
                    slot[category] = ""
                if h in slot.get("fact_hashes", []):
                    slot["fact_hashes"].remove(h)
                removed = True
                break
        if removed:
            break
    if removed:
        _save_users(store)
    _inject_memory(bot)
    return JSONResponse({"ok": True, "removed": removed})


# ── Debug-only manual unlock (Phase 2 testing before auto-verify lands) ──

@app.post("/api/debug/unlock/{uuid_prefix}")
async def api_debug_unlock(uuid_prefix: str):
    """Force-unlock a slot by UUID prefix. Used for Phase 2 testing only.
    Verification hook (Phase 3) will supersede this in normal flow."""
    if bot is None:
        return JSONResponse({"ok": False, "error": "No model loaded"}, status_code=400)
    store = _load_users()
    matches = [uid for uid in store["users"] if uid.startswith(uuid_prefix)]
    if len(matches) == 0:
        return JSONResponse({"ok": False, "error": "No slot with that prefix"}, status_code=404)
    if len(matches) > 1:
        return JSONResponse({"ok": False, "error": f"Ambiguous prefix matches {len(matches)} slots"}, status_code=400)
    bot.unlocked_user_id = matches[0]
    bot.candidate_uuids = [matches[0]]
    bot.session_non_name_match = True
    _inject_memory(bot)
    logger.warning("DEBUG unlock: session bound to %s", matches[0])
    return JSONResponse({"ok": True, "user_id": matches[0]})


@app.post("/api/debug/relock")
async def api_debug_relock():
    """Clear the session binding so you can test another user without restarting."""
    if bot is None:
        return JSONResponse({"ok": False, "error": "No model loaded"}, status_code=400)
    _reset_session_lock(bot)
    _inject_memory(bot)
    return JSONResponse({"ok": True})


def _reset_session_lock(b: VoiceChatBot) -> None:
    """Clear all per-session lock state. Called on /api/chats/new and on relock.
    In simple mode, re-bind to the default user instead of leaving locked."""
    b.session_fact_hashes = set()
    b.candidate_uuids = None
    b.session_non_name_match = False
    b.pending_new_user_facts = []
    if MULTI_USER_MODE:
        b.unlocked_user_id = None
    else:
        store = _load_users()
        b.unlocked_user_id = _default_user_id(store)


def _inject_memory(b: VoiceChatBot):
    """Rebuild the system prompt's memory block based on current lock state.
    Locked + no candidates narrowed → locked stub.
    Locked + narrowing (2+ candidates) → narrowing hint.
    Unlocked → plaintext slot view.
    Always strips any previous block first so it never stacks."""
    base = b.system_prompt or ""
    idx = base.find(MEMORY_MARKER)
    if idx != -1:
        base = base[:idx]

    if b.unlocked_user_id is not None:
        items = _load_memory()
        b.system_prompt = base + _memory_as_prompt(items)
        return

    store = _load_users()
    cand = b.candidate_uuids
    if cand is not None and len(cand) >= 2:
        b.system_prompt = base + _narrowing_hint(len(cand))
    else:
        b.system_prompt = base + _locked_stub(store)


def _attach_episodic(b: VoiceChatBot, query: str) -> None:
    """Strip any prior episodic block and append a fresh one for this turn.

    Vector memory is per-turn (depends on the current query), unlike the
    fact block which only rebuilds on lock-state change."""
    if b.unlocked_user_id is None:
        return
    base = b.system_prompt or ""
    idx = base.find(EPISODIC_MARKER)
    if idx != -1:
        base = base[:idx]
    try:
        block = episodic.recall_block(b.unlocked_user_id, query)
    except Exception as e:  # noqa: BLE001
        logger.warning("episodic.recall failed: %s", e)
        block = ""
    b.system_prompt = base + block


def _record_episode(slot_id: str | None, user_msg: str, assistant_msg: str) -> None:
    if not slot_id:
        return
    try:
        episodic.record(slot_id, user_msg, assistant_msg)
    except Exception as e:  # noqa: BLE001
        logger.warning("episodic.record failed: %s", e)


def _run_slash_command(message: str) -> str | None:
    """If message is a direct slash invocation, run it and return a user-facing string.

    Returns None when the message isn't a slash command (or the command is unknown).
    For /image, we synthesize a friendly reply without going through the LLM."""
    stripped = message.strip()
    if not stripped.startswith("/"):
        return None
    cmd = stripped.split(maxsplit=1)[0].lower()
    if cmd == "/help":
        result, _, _ = run_tool(stripped)
        return result
    if cmd not in TOOLS:
        return None
    result, tool_name, links = run_tool(stripped)
    if result is None:
        return None
    if tool_name == "/image":
        m = _IMG_PATH_RE.search(result)
        if m:
            return "Here's the image I made for you~! ✧"
        return result  # error message from tool_image
    if links:
        return f"{result}\n\n{links}"
    return result


@app.post("/chat")
async def chat(req: ChatRequest):
    global last_interaction_time, current_chat_id
    if bot is None:
        return JSONResponse({"error": "Model is still loading, please wait."}, status_code=503)
    try:
        last_interaction_time = time.time()
        history_start = len(bot.history)

        # Slash commands bypass the LLM and run the tool directly.
        slash_reply = await asyncio.to_thread(_run_slash_command, req.message)
        if slash_reply is not None:
            bot.history.append({"role": "user", "content": req.message})
            bot.history.append({"role": "assistant", "content": slash_reply})
            response = slash_reply
        else:
            # Memory verification hook — runs before the LLM turn so the
            # injected memory block reflects any lock state change this
            # message triggers (unlock, narrowing update, new-user commit).
            prev_unlocked = bot.unlocked_user_id
            prev_candidates = bot.candidate_uuids
            prior_assistant = ""
            for turn in reversed(bot.history):
                if turn.get("role") == "assistant":
                    prior_assistant = turn.get("content", "") or ""
                    break
            extracted = await asyncio.to_thread(
                verify_and_advance, bot, req.message, prior_assistant
            )
            print(
                f"[verify] unlocked={bot.unlocked_user_id} "
                f"candidates={bot.candidate_uuids} "
                f"extracted={extracted}",
                flush=True,
            )
            if bot.unlocked_user_id != prev_unlocked or bot.candidate_uuids != prev_candidates:
                _inject_memory(bot)
            await asyncio.to_thread(_attach_episodic, bot, req.message)
            response = await asyncio.to_thread(bot.react_chat, req.message)
            await asyncio.to_thread(
                _record_episode, bot.unlocked_user_id, req.message, response
            )

        audio_b64 = await asyncio.to_thread(_synthesize_audio, response)
        last_interaction_time = time.time()
        stats = bot.last_stats if bot.last_stats else None
        media = _collect_recent_media(bot.history, since=history_start)

        # Auto-save chat history
        if current_chat_id and bot is not None:
            title = _auto_title(bot.history)
            _save_chat(current_chat_id, title, bot.history)

        return JSONResponse({
            "text": response, "audio": audio_b64, "stats": stats,
            "chat_id": current_chat_id, "media": media,
        })
    except Exception as e:
        logger.exception("Error in /chat")
        return JSONResponse({"error": _friendly_error(e)}, status_code=_error_status(e))

@app.get("/autonomous")
async def autonomous_poll():
    global last_interaction_time
    if bot is None or not autonomous_enabled:
        return JSONResponse({"text": "", "audio": ""})
    try:
        seconds_idle = time.time() - last_interaction_time
        if seconds_idle < AUTONOMY_INTERVAL:
            return JSONResponse({"text": "", "audio": ""})

        response = await asyncio.to_thread(bot.autonomous_tick, seconds_idle)
        if response is None:
            last_interaction_time = time.time()
            return JSONResponse({"text": "", "audio": ""})

        audio_b64 = await asyncio.to_thread(_synthesize_audio, response)
        last_interaction_time = time.time()
        stats = bot.last_stats if bot.last_stats else None
        return JSONResponse({"text": response, "audio": audio_b64, "stats": stats})
    except Exception as e:
        logger.exception("Error in /autonomous")
        return JSONResponse({"text": "", "audio": ""})

@app.post("/autonomous/toggle")
async def autonomous_toggle():
    global autonomous_enabled, last_interaction_time
    autonomous_enabled = not autonomous_enabled
    last_interaction_time = time.time()
    return JSONResponse({"enabled": autonomous_enabled})

if __name__ == "__main__":
    if "--interactive" in sys.argv:
        headless = False
        print("Starting in interactive mode — select model in terminal")
    else:
        print("Starting web UI — configure at http://localhost:7860")
    Granian(
        "interface.server:app",
        address="127.0.0.1",
        port=7860,
        interface="asgi",
    ).serve()
