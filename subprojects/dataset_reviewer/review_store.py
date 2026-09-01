"""Transactional SQLite persistence for human review decisions."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .dataset_store import DatasetStore

VERDICTS = ("good", "bad", "fix", "unsure")
REASONS = (
    "factual error",
    "hallucination",
    "bad reasoning",
    "ignored instruction",
    "bad tool behavior",
    "bad personality/style",
    "too verbose",
    "too short",
    "formatting",
    "duplicate",
    "unsafe/unwanted behavior",
    "other",
)


class ReviewStore:
    def __init__(self, path: Path, dataset: DatasetStore):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.dataset_id = dataset.fingerprint
        self._lock = threading.RLock()
        self.started_at = time.time()
        self._initialize()
        self._register_dataset()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("PRAGMA synchronous = FULL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    fingerprint TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    format TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    byte_size INTEGER NOT NULL,
                    registered_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS samples (
                    dataset_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    original_index INTEGER NOT NULL,
                    row_hash TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, sample_id),
                    UNIQUE (dataset_id, original_index),
                    FOREIGN KEY (dataset_id) REFERENCES datasets(fingerprint)
                );
                CREATE INDEX IF NOT EXISTS idx_samples_dataset_index
                    ON samples(dataset_id, original_index);
                CREATE TABLE IF NOT EXISTS reviews (
                    dataset_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    original_index INTEGER NOT NULL,
                    verdict TEXT NOT NULL CHECK(verdict IN ('good','bad','fix','unsure')),
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    note TEXT NOT NULL DEFAULT '',
                    corrected_json TEXT,
                    reviewed_at REAL NOT NULL,
                    review_duration_ms INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (dataset_id, sample_id),
                    FOREIGN KEY (dataset_id, sample_id) REFERENCES samples(dataset_id, sample_id)
                );
                CREATE INDEX IF NOT EXISTS idx_reviews_dataset_verdict
                    ON reviews(dataset_id, verdict);
                CREATE TABLE IF NOT EXISTS drafts (
                    dataset_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    verdict TEXT CHECK(verdict IN ('bad','fix')),
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    note TEXT NOT NULL DEFAULT '',
                    corrected_text TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (dataset_id, sample_id),
                    FOREIGN KEY (dataset_id, sample_id) REFERENCES samples(dataset_id, sample_id)
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    original_index INTEGER NOT NULL,
                    previous_json TEXT,
                    current_json TEXT,
                    created_at REAL NOT NULL,
                    undone INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_events_dataset_time
                    ON review_events(dataset_id, id DESC);
                """
            )
            draft_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(drafts)").fetchall()
            }
            if "verdict" not in draft_columns:
                db.execute(
                    "ALTER TABLE drafts ADD COLUMN verdict TEXT "
                    "CHECK(verdict IN ('bad','fix'))"
                )

    def _register_dataset(self) -> None:
        meta = self.dataset.metadata()
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO datasets
                   (fingerprint, path, name, format, total, byte_size, registered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.dataset_id, meta["path"], meta["name"], meta["format"],
                    meta["total"], meta["bytes"], time.time(),
                ),
            )
            rows = []
            for pointer in self.dataset.pointers:
                rows.append((
                    self.dataset_id,
                    pointer.sample_id,
                    pointer.index,
                    pointer.row_hash,
                    self.dataset.searchable_text(pointer.index).casefold(),
                ))
            db.executemany(
                """INSERT OR IGNORE INTO samples
                   (dataset_id, sample_id, original_index, row_hash, search_text)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )

    @staticmethod
    def _decode_review(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "sample_id": row["sample_id"],
            "index": row["original_index"],
            "verdict": row["verdict"],
            "reasons": json.loads(row["reasons_json"]),
            "note": row["note"],
            "corrected_content": json.loads(row["corrected_json"]) if row["corrected_json"] else None,
            "reviewed_at": row["reviewed_at"],
            "review_duration_ms": row["review_duration_ms"],
            "revision": row["revision"],
        }

    def get_review(self, sample_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM reviews WHERE dataset_id = ? AND sample_id = ?",
                (self.dataset_id, sample_id),
            ).fetchone()
        return self._decode_review(row)

    def get_draft(self, sample_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM drafts WHERE dataset_id = ? AND sample_id = ?",
                (self.dataset_id, sample_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "verdict": row["verdict"],
            "reasons": json.loads(row["reasons_json"]),
            "note": row["note"],
            "corrected_text": row["corrected_text"],
            "updated_at": row["updated_at"],
        }

    def save_draft(
        self,
        sample_id: str,
        verdict: str | None,
        reasons: list[str],
        note: str,
        corrected_text: str,
    ) -> dict[str, Any]:
        if verdict not in {None, "bad", "fix"}:
            raise ValueError("Draft verdict must be bad, fix, or empty")
        self._validate_reasons(reasons)
        with self._lock, self._connect() as db:
            self._assert_sample(db, sample_id)
            now = time.time()
            db.execute(
                """INSERT INTO drafts
                   (dataset_id, sample_id, verdict, reasons_json, note, corrected_text, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dataset_id, sample_id) DO UPDATE SET
                     verdict=excluded.verdict, reasons_json=excluded.reasons_json, note=excluded.note,
                     corrected_text=excluded.corrected_text, updated_at=excluded.updated_at""",
                (
                    self.dataset_id, sample_id, verdict, json.dumps(reasons),
                    note, corrected_text, now,
                ),
            )
        return {"saved": True, "updated_at": now}

    def save_review(
        self,
        sample_id: str,
        verdict: str,
        reasons: list[str],
        note: str,
        corrected_content: dict[str, Any] | None,
        duration_ms: int,
    ) -> dict[str, Any]:
        if verdict not in VERDICTS:
            raise ValueError(f"Unsupported verdict: {verdict}")
        self._validate_reasons(reasons)
        if corrected_content is not None and not isinstance(corrected_content, dict):
            raise ValueError("Corrected content must be a JSON object")
        corrected_json = (
            json.dumps(corrected_content, ensure_ascii=False, sort_keys=True)
            if corrected_content is not None else None
        )
        now = time.time()
        duration_ms = max(0, min(int(duration_ms), 86_400_000))
        with self._lock, self._connect() as db:
            index = self._assert_sample(db, sample_id)
            previous_row = db.execute(
                "SELECT * FROM reviews WHERE dataset_id = ? AND sample_id = ?",
                (self.dataset_id, sample_id),
            ).fetchone()
            previous = self._decode_review(previous_row)
            revision = (previous["revision"] + 1) if previous else 1
            db.execute(
                """INSERT INTO reviews
                   (dataset_id, sample_id, original_index, verdict, reasons_json,
                    note, corrected_json, reviewed_at, review_duration_ms, revision)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(dataset_id, sample_id) DO UPDATE SET
                     verdict=excluded.verdict, reasons_json=excluded.reasons_json,
                     note=excluded.note, corrected_json=excluded.corrected_json,
                     reviewed_at=excluded.reviewed_at,
                     review_duration_ms=excluded.review_duration_ms,
                     revision=excluded.revision""",
                (
                    self.dataset_id, sample_id, index, verdict, json.dumps(reasons),
                    note, corrected_json, now, duration_ms, revision,
                ),
            )
            current_row = db.execute(
                "SELECT * FROM reviews WHERE dataset_id = ? AND sample_id = ?",
                (self.dataset_id, sample_id),
            ).fetchone()
            current = self._decode_review(current_row)
            db.execute(
                """INSERT INTO review_events
                   (dataset_id, sample_id, original_index, previous_json,
                    current_json, created_at, undone)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (
                    self.dataset_id, sample_id, index,
                    json.dumps(previous, ensure_ascii=False) if previous else None,
                    json.dumps(current, ensure_ascii=False), now,
                ),
            )
            db.execute(
                "DELETE FROM drafts WHERE dataset_id = ? AND sample_id = ?",
                (self.dataset_id, sample_id),
            )
        return current

    def undo(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            event = db.execute(
                """SELECT * FROM review_events
                   WHERE dataset_id = ? AND undone = 0
                   ORDER BY id DESC LIMIT 1""",
                (self.dataset_id,),
            ).fetchone()
            if event is None:
                return None
            previous = json.loads(event["previous_json"]) if event["previous_json"] else None
            if previous is None:
                db.execute(
                    "DELETE FROM reviews WHERE dataset_id = ? AND sample_id = ?",
                    (self.dataset_id, event["sample_id"]),
                )
            else:
                corrected_json = (
                    json.dumps(previous["corrected_content"], ensure_ascii=False, sort_keys=True)
                    if previous.get("corrected_content") is not None else None
                )
                db.execute(
                    """INSERT OR REPLACE INTO reviews
                       (dataset_id, sample_id, original_index, verdict, reasons_json,
                        note, corrected_json, reviewed_at, review_duration_ms, revision)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        self.dataset_id, previous["sample_id"], previous["index"],
                        previous["verdict"], json.dumps(previous["reasons"]),
                        previous["note"], corrected_json, previous["reviewed_at"],
                        previous["review_duration_ms"], previous["revision"],
                    ),
                )
            db.execute("UPDATE review_events SET undone = 1 WHERE id = ?", (event["id"],))
            return {
                "sample_id": event["sample_id"],
                "index": event["original_index"],
                "restored_review": previous,
            }

    def _assert_sample(self, db: sqlite3.Connection, sample_id: str) -> int:
        row = db.execute(
            "SELECT original_index FROM samples WHERE dataset_id = ? AND sample_id = ?",
            (self.dataset_id, sample_id),
        ).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return int(row[0])

    @staticmethod
    def _validate_reasons(reasons: list[str]) -> None:
        invalid = sorted(set(reasons) - set(REASONS))
        if invalid:
            raise ValueError(f"Unsupported reason categories: {', '.join(invalid)}")

    def counts(self) -> dict[str, Any]:
        counts = {verdict: 0 for verdict in VERDICTS}
        with self._connect() as db:
            for row in db.execute(
                """SELECT verdict, COUNT(*) AS total FROM reviews
                   WHERE dataset_id = ? GROUP BY verdict""",
                (self.dataset_id,),
            ):
                counts[row["verdict"]] = row["total"]
        reviewed = sum(counts.values())
        total = len(self.dataset)
        return {
            **counts,
            "reviewed": reviewed,
            "remaining": total - reviewed,
            "total": total,
            "percent": round((reviewed / total) * 100, 2),
        }

    def dashboard(self) -> dict[str, Any]:
        counts = self.counts()
        reasons = {reason: 0 for reason in REASONS}
        durations: list[int] = []
        with self._connect() as db:
            rows = db.execute(
                "SELECT reasons_json, review_duration_ms FROM reviews WHERE dataset_id = ?",
                (self.dataset_id,),
            ).fetchall()
            for row in rows:
                durations.append(row["review_duration_ms"])
                for reason in json.loads(row["reasons_json"]):
                    reasons[reason] += 1
            session_row = db.execute(
                """SELECT COUNT(*) AS decisions,
                          COALESCE(SUM(json_extract(current_json, '$.review_duration_ms')), 0) AS duration
                   FROM review_events
                   WHERE dataset_id = ? AND undone = 0 AND created_at >= ?""",
                (self.dataset_id, self.started_at),
            ).fetchone()
        ranked_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
            if count
        ]
        return {
            "counts": counts,
            "reason_counts": ranked_reasons,
            "average_review_ms": round(sum(durations) / len(durations)) if durations else 0,
            "session": {
                "decisions": session_row["decisions"],
                "review_duration_ms": session_row["duration"],
                "started_at": self.started_at,
            },
        }

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, sample_id, original_index, current_json, created_at, undone
                   FROM review_events WHERE dataset_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (self.dataset_id, max(1, min(limit, 500))),
            ).fetchall()
        return [
            {
                "event_id": row["id"],
                "sample_id": row["sample_id"],
                "index": row["original_index"],
                "review": json.loads(row["current_json"]),
                "created_at": row["created_at"],
                "undone": bool(row["undone"]),
            }
            for row in rows
        ]

    def matching_index(
        self, current: int, direction: str, verdict_filter: str, search: str,
    ) -> tuple[int | None, bool]:
        if direction not in {"next", "previous"}:
            raise ValueError("direction must be next or previous")
        operator = ">" if direction == "next" else "<"
        ordering = "ASC" if direction == "next" else "DESC"
        clauses = ["s.dataset_id = ?", f"s.original_index {operator} ?"]
        params: list[Any] = [self.dataset_id, current]
        self._append_filter(clauses, params, verdict_filter, search)
        base = (
            "SELECT s.original_index FROM samples s "
            "LEFT JOIN reviews r ON r.dataset_id=s.dataset_id AND r.sample_id=s.sample_id "
        )
        with self._connect() as db:
            row = db.execute(
                base + "WHERE " + " AND ".join(clauses) + f" ORDER BY s.original_index {ordering} LIMIT 1",
                params,
            ).fetchone()
            if row is not None:
                return int(row[0]), False
            clauses = ["s.dataset_id = ?"]
            params = [self.dataset_id]
            self._append_filter(clauses, params, verdict_filter, search)
            row = db.execute(
                base + "WHERE " + " AND ".join(clauses) + f" ORDER BY s.original_index {ordering} LIMIT 1",
                params,
            ).fetchone()
        return (int(row[0]), True) if row is not None else (None, False)

    @staticmethod
    def _append_filter(
        clauses: list[str], params: list[Any], verdict_filter: str, search: str,
    ) -> None:
        if verdict_filter == "unreviewed":
            clauses.append("r.verdict IS NULL")
        elif verdict_filter in VERDICTS:
            clauses.append("r.verdict = ?")
            params.append(verdict_filter)
        elif verdict_filter not in {"all", "reviewed"}:
            raise ValueError(f"Unsupported filter: {verdict_filter}")
        elif verdict_filter == "reviewed":
            clauses.append("r.verdict IS NOT NULL")
        if search.strip():
            clauses.append("s.search_text LIKE ?")
            params.append(f"%{search.casefold()}%")

    def clear_reviews(self, confirmation: str) -> int:
        if confirmation != self.dataset_id[:8]:
            raise ValueError("Confirmation does not match the dataset fingerprint")
        with self._lock, self._connect() as db:
            count = db.execute(
                "SELECT COUNT(*) FROM reviews WHERE dataset_id = ?", (self.dataset_id,),
            ).fetchone()[0]
            db.execute("DELETE FROM reviews WHERE dataset_id = ?", (self.dataset_id,))
            db.execute("DELETE FROM drafts WHERE dataset_id = ?", (self.dataset_id,))
            db.execute("DELETE FROM review_events WHERE dataset_id = ?", (self.dataset_id,))
        return count

    def all_reviews(self) -> dict[str, dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM reviews WHERE dataset_id = ?", (self.dataset_id,),
            ).fetchall()
        return {row["sample_id"]: self._decode_review(row) for row in rows}
