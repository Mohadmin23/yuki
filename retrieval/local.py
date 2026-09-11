"""Local retrieval across Yuki-owned notes and existing episodic memory."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Protocol

from .core import (
    RetrievalHit,
    chunk_text,
    query_terms,
    reciprocal_rank_fusion,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "retrieval.db"
DEFAULT_YUKI_DIR = PROJECT_ROOT / "yuki"
MAX_FILE_BYTES = 1_000_000
_BINARY_SUFFIXES = {
    ".avif",
    ".bin",
    ".gif",
    ".heic",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".sqlite",
    ".sqlite3",
    ".wav",
    ".webp",
}


class EpisodicProvider(Protocol):
    def recall_sessions(
        self,
        query: str,
        k: int = 3,
        max_distance: float = 1.28,
        exclude_session: str | None = None,
    ) -> list[dict]: ...

    def fuzzy_when(self, age_seconds: float) -> str: ...


class YukiFileIndex:
    """A tiny incremental FTS5 index over text-like files in ``yuki/``."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_DB_PATH,
        root_dir: Path = DEFAULT_YUKI_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.root_dir = Path(root_dir)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(self.db_path))
        db.execute(
            "CREATE TABLE IF NOT EXISTS indexed_files ("
            "source_uri TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, "
            "size INTEGER NOT NULL, sha256 TEXT NOT NULL)"
        )
        db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS yuki_chunks USING fts5("
            "title, passage, source_uri UNINDEXED, chunk_id UNINDEXED, "
            "tokenize='unicode61')"
        )
        return db

    def _read_text_file(self, path: Path) -> str | None:
        try:
            if (
                not path.is_file()
                or path.suffix.casefold() in _BINARY_SUFFIXES
                or path.stat().st_size > MAX_FILE_BYTES
            ):
                return None
            raw = path.read_bytes()
        except OSError:
            return None
        if b"\x00" in raw:
            return None
        text = raw.decode("utf-8", errors="replace").strip()
        return text or None

    def sync(self) -> None:
        """Index changed text files and remove entries for files that disappeared."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        resolved_root = self.root_dir.resolve()
        discovered: dict[str, tuple[Path, int, int]] = {}
        for path in sorted(self.root_dir.rglob("*")):
            try:
                stat = path.stat()
                if not path.resolve().is_relative_to(resolved_root):
                    continue
            except OSError:
                continue
            if not path.is_file() or stat.st_size > MAX_FILE_BYTES:
                continue
            relative = path.relative_to(self.root_dir).as_posix()
            discovered[f"yuki://{relative}"] = (path, stat.st_mtime_ns, stat.st_size)

        db = self._connect()
        try:
            existing = {
                row[0]: (int(row[1]), int(row[2]), str(row[3]))
                for row in db.execute(
                    "SELECT source_uri, mtime_ns, size, sha256 FROM indexed_files"
                )
            }
            for source_uri in set(existing) - set(discovered):
                db.execute(
                    "DELETE FROM yuki_chunks WHERE source_uri = ?", (source_uri,)
                )
                db.execute(
                    "DELETE FROM indexed_files WHERE source_uri = ?", (source_uri,)
                )

            for source_uri, (path, mtime_ns, size) in discovered.items():
                prior = existing.get(source_uri)
                if prior and prior[:2] == (mtime_ns, size):
                    continue
                text = self._read_text_file(path)
                if text is None:
                    db.execute(
                        "DELETE FROM yuki_chunks WHERE source_uri = ?", (source_uri,)
                    )
                    db.execute(
                        "DELETE FROM indexed_files WHERE source_uri = ?", (source_uri,)
                    )
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if prior and prior[2] == digest:
                    db.execute(
                        "UPDATE indexed_files SET mtime_ns = ?, size = ? WHERE source_uri = ?",
                        (mtime_ns, size, source_uri),
                    )
                    continue
                db.execute(
                    "DELETE FROM yuki_chunks WHERE source_uri = ?", (source_uri,)
                )
                title = path.relative_to(self.root_dir).as_posix()
                for chunk_index, passage in enumerate(chunk_text(text)):
                    db.execute(
                        "INSERT INTO yuki_chunks(title, passage, source_uri, chunk_id) "
                        "VALUES (?, ?, ?, ?)",
                        (title, passage, source_uri, str(chunk_index)),
                    )
                db.execute(
                    "INSERT INTO indexed_files(source_uri, mtime_ns, size, sha256) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(source_uri) DO UPDATE SET "
                    "mtime_ns=excluded.mtime_ns, size=excluded.size, sha256=excluded.sha256",
                    (source_uri, mtime_ns, size, digest),
                )
            db.commit()
        finally:
            db.close()

    def search(self, query: str, *, limit: int = 4) -> list[RetrievalHit]:
        terms = query_terms(query)
        if not terms:
            return []
        self.sync()
        fts_query = " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
        )
        db = self._connect()
        try:
            rows = db.execute(
                "SELECT title, passage, source_uri, chunk_id, "
                "bm25(yuki_chunks, 3.0, 1.0) AS relevance "
                "FROM yuki_chunks WHERE yuki_chunks MATCH ? "
                "ORDER BY relevance LIMIT ?",
                (fts_query, max(1, int(limit))),
            ).fetchall()
        finally:
            db.close()
        return [
            RetrievalHit(
                id=f"{source_uri}#chunk-{chunk_id}",
                source_type="yuki_file",
                title=title,
                text=passage,
                uri=source_uri,
                rank=index,
                score=float(relevance),
                metadata={"chunk_index": int(chunk_id)},
            )
            for index, (title, passage, source_uri, chunk_id, relevance) in enumerate(
                rows, start=1
            )
        ]


def retrieve_local_memory(
    query: str,
    *,
    exclude_session: str | None = None,
    limit: int = 5,
    file_index: YukiFileIndex | None = None,
    episodic_provider: EpisodicProvider | None = None,
) -> tuple[list[RetrievalHit], list[str]]:
    """Retrieve past-session summaries and related Yuki notes, then rank-fuse."""
    warnings: list[str] = []
    episodic_hits: list[RetrievalHit] = []
    file_hits: list[RetrievalHit] = []

    if episodic_provider is None:
        import episodic as episodic_provider  # lazy: may require OpenRouter on use
    try:
        sessions = episodic_provider.recall_sessions(
            query,
            k=max(3, limit),
            exclude_session=exclude_session,
        )
        now = time.time()
        episodic_hits = [
            RetrievalHit(
                id=f"episodic://{item['session_uuid']}",
                source_type="past_conversation",
                title="Past conversation",
                text=str(item["summary"]),
                uri=f"episodic://{item['session_uuid']}",
                rank=index,
                score=float(item["distance"]),
                metadata={
                    "timestamp": float(item["ts"]),
                    "when": episodic_provider.fuzzy_when(now - float(item["ts"])),
                },
            )
            for index, item in enumerate(sessions, start=1)
        ]
    except Exception as exc:  # noqa: BLE001 - partial local retrieval is useful
        warnings.append(f"past-conversation search unavailable: {exc}")

    index = file_index or YukiFileIndex()
    try:
        file_hits = index.search(query, limit=max(3, limit))
    except Exception as exc:  # noqa: BLE001 - episodic retrieval may still work
        warnings.append(f"Yuki-note search unavailable: {exc}")

    rankings = [ranking for ranking in (episodic_hits, file_hits) if ranking]
    return reciprocal_rank_fusion(rankings, limit=limit), warnings
