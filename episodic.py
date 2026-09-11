"""Episodic memory: tool-recalled summaries of past sessions.

Single-user mode — there is no per-user slot. Every episode and summary
belongs to the one logical user of this Yuki instance.

  1. Every user+assistant exchange is embedded and stored under its session_uuid.
  2. When /recall(query) fires, we embed the query, find the closest episodes,
     collapse them by session_uuid, and return the *summary* of each session.
  3. Summaries are written when a session ends. If a session was never closed,
     recall lazy-summarizes it on first hit.

Embeddings: OpenRouter, openai/text-embedding-3-small (1536 dim)
Summarizer: OpenRouter, $EPISODIC_SUMMARY_MODEL (default openai/gpt-4.1-nano)
Storage:    sqlite-vec, data/episodic_v2.db
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import struct
import time
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "episodic_v2.db"
EMBED_DIM = 1536
EMBED_MODEL = "openai/text-embedding-3-small"
SUMMARY_MODEL = os.environ.get("EPISODIC_SUMMARY_MODEL", "openai/gpt-4.1-nano")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client = None
logger = logging.getLogger(__name__)


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # type: ignore
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set — episodic memory needs it")
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _client


def _embed(text: str) -> bytes:
    """Embed via OpenRouter. L2-normalized so vec0's L2 distance ≈ cosine."""
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=text)
    vec = list(resp.data[0].embedding)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return struct.pack(f"{EMBED_DIM}f", *vec)


def _get_db() -> sqlite3.Connection:
    """Open a vector-enabled connection owned and closed by the caller."""
    import sqlite_vec  # type: ignore
    db = _open_local_transcript_db()
    try:
        db.enable_load_extension(True)
        try:
            sqlite_vec.load(db)
        finally:
            db.enable_load_extension(False)
        db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes "
            f"USING vec0(embedding float[{EMBED_DIM}])"
        )
        return db
    except Exception:
        db.close()
        raise


def _open_local_transcript_db() -> sqlite3.Connection:
    """Open a short-lived connection for thread-safe local session writes."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.execute(
        "CREATE TABLE IF NOT EXISTS episodes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_uuid TEXT NOT NULL, "
        "ts REAL NOT NULL, "
        "user_msg TEXT NOT NULL, "
        "assistant_msg TEXT NOT NULL)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_uuid)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS session_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_uuid TEXT NOT NULL, "
        "ts REAL NOT NULL, "
        "payload TEXT NOT NULL)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_events_session "
        "ON session_events(session_uuid)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS session_summaries ("
        "session_uuid TEXT PRIMARY KEY, ts_start REAL NOT NULL, "
        "ts_end REAL NOT NULL, summary TEXT NOT NULL)"
    )
    return db


def record(session_uuid: str, user_msg: str, assistant_msg: str) -> None:
    """Save locally first; optional embedding failure must not lose the turn."""
    if not session_uuid or not user_msg or not assistant_msg:
        return
    rowid = record_transcript(session_uuid, user_msg, assistant_msg)
    try:
        embedding = _embed(f"User: {user_msg}\nYuki: {assistant_msg}")
        with closing(_get_db()) as db:
            db.execute(
                "INSERT INTO vec_episodes (rowid, embedding) VALUES (?, ?)",
                (rowid, embedding),
            )
            db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chat saved locally; semantic indexing unavailable: %s", exc)


def record_transcript(
    session_uuid: str,
    user_msg: str,
    assistant_msg: str,
) -> int | None:
    """Store a resumable turn without paying for an embedding.

    Autonomous turns use this path. They remain part of the session transcript
    and its eventual summary, while user-authored turns continue through
    :func:`record` so semantic recall retains its existing behavior and cost.
    """
    if not session_uuid or not user_msg or not assistant_msg:
        return
    db = _open_local_transcript_db()
    try:
        cur = db.execute(
            "INSERT INTO episodes (session_uuid, ts, user_msg, assistant_msg) "
            "VALUES (?, ?, ?, ?)",
            (session_uuid, time.time(), user_msg, assistant_msg),
        )
        db.commit()
        return cur.lastrowid
    finally:
        db.close()


def record_session_event(session_uuid: str, event: dict) -> None:
    """Persist one compact current-session continuity event locally."""
    if not session_uuid or not isinstance(event, dict) or not event:
        return
    payload = json.dumps(event, ensure_ascii=False, default=str)
    db = _open_local_transcript_db()
    try:
        db.execute(
            "INSERT INTO session_events (session_uuid, ts, payload) VALUES (?, ?, ?)",
            (session_uuid, time.time(), payload),
        )
        db.commit()
    finally:
        db.close()


def read_session_events(session_uuid: str, limit: int = 48) -> list[dict]:
    """Return a session's newest continuity records in chronological order."""
    if not session_uuid:
        return []
    db = _open_local_transcript_db()
    try:
        rows = db.execute(
            "SELECT payload FROM session_events WHERE session_uuid = ? "
            "ORDER BY ts DESC, id DESC LIMIT ?",
            (session_uuid, max(1, int(limit))),
        ).fetchall()
    finally:
        db.close()
    events = []
    for (payload,) in reversed(rows):
        try:
            event = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summarize_session(session_uuid: str) -> str:
    """Generate (or refresh) a summary for one session. Idempotent."""
    if not session_uuid:
        return ""
    with closing(_open_local_transcript_db()) as db:
        rows = db.execute(
            "SELECT ts, user_msg, assistant_msg FROM episodes "
            "WHERE session_uuid = ? ORDER BY ts, id",
            (session_uuid,),
        ).fetchall()
    if not rows:
        return ""

    transcript = "\n".join(
        f"User: {u.strip()}\nYuki: {a.strip()}" for _, u, a in rows
    )
    prompt = (
        "Summarize this conversation between a user and an AI companion (Yuki) "
        "in 2-4 short sentences. Focus on what they discussed, decided, or "
        "what the user was feeling/working on. No headers, no bullets — "
        "just plain prose, as if recalling the chat later.\n\n"
        f"---\n{transcript}\n---"
    )
    try:
        resp = _get_client().chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        summary = (resp.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001
        first_turns = [u for _, u, _ in rows[:3]]
        summary = "Earlier they talked about: " + " / ".join(first_turns)[:300]

    ts_start = rows[0][0]
    ts_end = rows[-1][0]
    with closing(_open_local_transcript_db()) as db:
        db.execute(
            "INSERT INTO session_summaries (session_uuid, ts_start, ts_end, summary) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_uuid) DO UPDATE SET "
            "ts_end=excluded.ts_end, summary=excluded.summary",
            (session_uuid, ts_start, ts_end, summary),
        )
        db.commit()
    return summary


def recall_sessions(
    query: str,
    k: int = 3,
    max_distance: float = 1.28,
    exclude_session: str | None = None,
) -> list[dict]:
    """Find sessions semantically similar to `query`, return their summaries.

    exclude_session: skip episodes from this session (the current one — its
    contents are already in the bot's live history).
    """
    if not query:
        return []
    embedding = _embed(query)
    with closing(_get_db()) as db:
        rows = db.execute(
            "SELECT e.session_uuid, e.ts, v.distance "
            "FROM vec_episodes v "
            "JOIN episodes e ON e.id = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? "
            "ORDER BY v.distance",
            (embedding, k * 8),
        ).fetchall()

    best_by_session: dict[str, tuple[float, float]] = {}  # uuid -> (distance, ts)
    for session_uuid, ts, distance in rows:
        if distance > max_distance:
            continue
        if exclude_session and session_uuid == exclude_session:
            continue
        prev = best_by_session.get(session_uuid)
        if prev is None or distance < prev[0]:
            best_by_session[session_uuid] = (float(distance), float(ts))
        if len(best_by_session) >= k * 4:
            break

    ranked = sorted(best_by_session.items(), key=lambda kv: kv[1][0])[:k]
    if not ranked:
        return []

    out = []
    for session_uuid, (distance, ts) in ranked:
        summary = _get_or_build_summary(session_uuid)
        if not summary:
            continue
        out.append({
            "session_uuid": session_uuid,
            "summary": summary,
            "ts": ts,
            "distance": distance,
        })
    return out


def _get_or_build_summary(session_uuid: str) -> str:
    """Return the stored summary for a session, or build one on the fly."""
    with closing(_open_local_transcript_db()) as db:
        row = db.execute(
            "SELECT summary FROM session_summaries WHERE session_uuid = ?",
            (session_uuid,),
        ).fetchone()
    if row and row[0]:
        return row[0]
    return summarize_session(session_uuid)


def fuzzy_when(age_seconds: float) -> str:
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


def count_episodes() -> int:
    """Total episode count — handy for smoke tests."""
    with closing(_open_local_transcript_db()) as db:
        row = db.execute("SELECT COUNT(*) FROM episodes").fetchone()
    return int(row[0]) if row else 0
