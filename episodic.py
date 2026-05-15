"""Episodic memory: vector-recall over past chat turns.

Lives alongside the existing hash-based fact memory (semantic memory).
Two systems, two jobs:
  - facts go to memory.json (names, dates, preferences) — exact recall
  - vibes go here (conversations, stories, emotional moments) — fuzzy recall

Embeddings: OpenRouter, openai/text-embedding-3-small (1536 dim)
Storage:    sqlite-vec, data/episodic_v2.db
"""
from __future__ import annotations

import math
import os
import sqlite3
import struct
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "episodic_v2.db"
EMBED_DIM = 1536
EMBED_MODEL = "openai/text-embedding-3-small"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client = None
_db: Optional[sqlite3.Connection] = None


def _get_client():
    """Lazy-init the OpenAI SDK pointed at OpenRouter — mirrors ms_llama.py."""
    global _client
    if _client is None:
        from openai import OpenAI  # type: ignore
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set — episodic memory needs it")
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _client


def _embed(text: str) -> bytes:
    """Embed via OpenRouter. Vectors are L2-normalized so vec0's L2 distance
    behaves like cosine distance (0 = identical, ~1.41 = orthogonal)."""
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=text)
    vec = list(resp.data[0].embedding)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return struct.pack(f"{EMBED_DIM}f", *vec)


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is not None:
        return _db
    import sqlite_vec  # type: ignore
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute(
        "CREATE TABLE IF NOT EXISTS episodes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "slot_id TEXT NOT NULL, "
        "ts REAL NOT NULL, "
        "user_msg TEXT NOT NULL, "
        "assistant_msg TEXT NOT NULL)"
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_episodes_slot ON episodes(slot_id)")
    db.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes "
        f"USING vec0(embedding float[{EMBED_DIM}])"
    )
    _db = db
    return db


def record(slot_id: str, user_msg: str, assistant_msg: str) -> None:
    """Embed and store one user+assistant exchange under a slot."""
    if not slot_id or not user_msg or not assistant_msg:
        return
    text = f"User: {user_msg}\nYuki: {assistant_msg}"
    embedding = _embed(text)
    db = _get_db()
    cur = db.execute(
        "INSERT INTO episodes (slot_id, ts, user_msg, assistant_msg) VALUES (?, ?, ?, ?)",
        (slot_id, time.time(), user_msg, assistant_msg),
    )
    rowid = cur.lastrowid
    db.execute(
        "INSERT INTO vec_episodes (rowid, embedding) VALUES (?, ?)",
        (rowid, embedding),
    )
    db.commit()


def recall(slot_id: str, query: str, k: int = 3, max_distance: float = 1.28) -> list[dict]:
    """Return up to k past episodes most similar to query, for this slot.

    max_distance: L2 distance cutoff on normalized embeddings. Calibrated for
    openai/text-embedding-3-small: strong topic matches land 1.10–1.22, noise
    floor starts at ~1.27, true orthogonal ~1.40. Default 1.28 catches strong
    matches and excludes noise; tighten to 1.20 for very high precision."""
    if not slot_id or not query:
        return []
    embedding = _embed(query)
    db = _get_db()
    rows = db.execute(
        "SELECT e.user_msg, e.assistant_msg, e.ts, v.distance "
        "FROM vec_episodes v "
        "JOIN episodes e ON e.id = v.rowid "
        "WHERE v.embedding MATCH ? AND e.slot_id = ? AND k = ? "
        "ORDER BY v.distance",
        (embedding, slot_id, k * 4),  # over-fetch then filter by slot
    ).fetchall()
    out = []
    for user_msg, assistant_msg, ts, distance in rows:
        if distance > max_distance:
            continue
        out.append({
            "user_msg": user_msg,
            "assistant_msg": assistant_msg,
            "ts": ts,
            "distance": float(distance),
        })
        if len(out) >= k:
            break
    return out


def recall_block(slot_id: str, query: str, k: int = 3, max_distance: float = 1.25) -> str:
    """Render a 'you vaguely remember' block for the system prompt.

    Returns '' if there's nothing to remember — caller can append unconditionally."""
    episodes = recall(slot_id, query, k=k, max_distance=max_distance)
    if not episodes:
        return ""
    lines = []
    now = time.time()
    for ep in episodes:
        age_s = now - ep["ts"]
        when = _fuzzy_when(age_s)
        snippet = ep["user_msg"].strip().replace("\n", " ")
        if len(snippet) > 140:
            snippet = snippet[:140] + "…"
        reply = ep["assistant_msg"].strip().replace("\n", " ")
        if len(reply) > 140:
            reply = reply[:140] + "…"
        lines.append(f"  • {when}, they said: \"{snippet}\" — you replied: \"{reply}\"")
    body = "\n".join(lines)
    return (
        "\n[EPISODIC MEMORY — past moments you faintly recall together. "
        "Treat these like half-remembered vibes, not transcripts. "
        "Only reference one if the user's CURRENT message genuinely echoes it. "
        "Do not dump these into conversation, do not quote them verbatim, do not "
        "list them as a recap. If unsure, stay quiet about them.]\n"
        f"{body}"
    )


def _fuzzy_when(age_seconds: float) -> str:
    if age_seconds < 60:
        return "just now"
    if age_seconds < 3600:
        return f"about {int(age_seconds // 60)} min ago"
    if age_seconds < 86400:
        return f"about {int(age_seconds // 3600)} hr ago"
    days = int(age_seconds // 86400)
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 30:
        return f"about {days // 7} week{'s' if days // 7 != 1 else ''} ago"
    return f"a while ago ({days} days)"


def count_for(slot_id: str) -> int:
    """How many episodes are stored for this slot — handy for smoke tests."""
    if not slot_id:
        return 0
    db = _get_db()
    row = db.execute(
        "SELECT COUNT(*) FROM episodes WHERE slot_id = ?", (slot_id,)
    ).fetchone()
    return int(row[0]) if row else 0
