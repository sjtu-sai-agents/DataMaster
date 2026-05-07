"""SQLite + FTS5 memory storage layer.

A single SQLite file stores all user memories, isolated by the user_id field.
The FTS5 index uses jieba tokenization preprocessing to support Chinese search.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path

import jieba

from .types import MemoryEntry

logger = logging.getLogger(__name__)

# Silence jieba's initialization logs
jieba.setLogLevel(logging.WARNING)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    category    TEXT DEFAULT 'other',
    importance  REAL DEFAULT 0.5,
    source      TEXT DEFAULT 'auto',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(user_id, category);
"""

# Standalone FTS table (not linked to a content table, because jieba-segmented text is stored)
_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content);
"""

# Deduplication similarity threshold
_DEDUP_SIMILARITY = 0.90
# Time decay coefficient: reduces the score weight by 1% per day
_DECAY_RATE = 0.01


def _segment(text: str) -> str:
    """Perform jieba tokenization on text and return a space-separated result.

    Example: "用户喜欢吃草莓" -> "用户 喜欢 吃 草莓" (i.e. "user likes eating strawberries" -> segmented tokens)
    """
    return " ".join(jieba.cut(text))


class MemoryStore:
    """SQLite + FTS5 memory storage.

    The FTS5 index stores jieba-segmented text; queries are also segmented before searching.
    Thread-safe: all database operations are protected by a threading.Lock.
    """

    def __init__(self, db_path: str | Path):
        """Initialize the memory store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Connect to the SQLite database."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(_SCHEMA_SQL)

            # Migration: if old trigger-driven FTS table exists, drop and rebuild
            self._migrate_fts(cur)

            self._conn.commit()

    def _migrate_fts(self, cur: sqlite3.Cursor) -> None:
        """Detect and migrate old FTS table (from trigger-based sync to jieba manual sync)."""
        # Check for old triggers
        old_triggers = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN "
            "('memories_ai', 'memories_ad', 'memories_au')"
        ).fetchall()

        needs_rebuild = False

        if old_triggers:
            # Old triggers found -- migration needed
            logger.info("Migrating FTS from trigger-based to jieba-segmented...")
            for trigger in old_triggers:
                cur.execute(f"DROP TRIGGER IF EXISTS {trigger['name']}")
            # Drop old FTS table and rebuild
            cur.execute("DROP TABLE IF EXISTS memories_fts")
            needs_rebuild = True

        # Ensure FTS table exists
        try:
            cur.execute("SELECT * FROM memories_fts LIMIT 0")
        except sqlite3.OperationalError:
            needs_rebuild = True

        # Check if FTS content is already jieba-segmented (segmented text contains spaces)
        if not needs_rebuild:
            sample = cur.execute(
                "SELECT content FROM memories_fts LIMIT 1"
            ).fetchone()
            if sample and sample["content"] and " " not in sample["content"]:
                logger.info("FTS index not jieba-segmented, rebuilding...")
                needs_rebuild = True

        if needs_rebuild:
            cur.execute("DROP TABLE IF EXISTS memories_fts")
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content)"
            )
            self._rebuild_fts_index(cur)

    def _rebuild_fts_index(self, cur: sqlite3.Cursor) -> None:
        """Rebuild the entire FTS index using jieba segmentation."""
        cur.execute("DELETE FROM memories_fts")
        rows = cur.execute("SELECT rowid, content FROM memories").fetchall()
        for row in rows:
            segmented = _segment(row["content"])
            cur.execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                (row["rowid"], segmented),
            )
        logger.info("Rebuilt FTS index with jieba segmentation: %d entries", len(rows))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        user_id: str,
        content: str,
        category: str = "other",
        importance: float = 0.5,
        source: str = "auto",
    ) -> str | None:
        """Insert a memory. Auto-deduplication: if a highly similar memory already exists, updates updated_at and returns None."""
        content = content.strip()
        if not content:
            return None

        # Deduplication check
        existing = self.search(user_id, content, limit=1)
        if existing:
            top = existing[0]
            sim = SequenceMatcher(None, content.lower(), top.content.lower()).ratio()
            if sim >= _DEDUP_SIMILARITY:
                logger.debug(
                    "Duplicate memory (sim=%.2f), updating timestamp: %s",
                    sim, top.id,
                )
                with self._lock:
                    self._conn.execute(
                        "UPDATE memories SET updated_at = ?, access_count = access_count + 1 WHERE id = ?",
                        (time.time(), top.id),
                    )
                    self._conn.commit()
                return None

        now = time.time()
        memory_id = uuid.uuid4().hex[:16]
        segmented = _segment(content)

        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, user_id, content, category, importance, source, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (memory_id, user_id, content, category, importance, source, now, now),
            )
            # Manually sync the FTS index (with jieba-segmented text)
            rowid = self._conn.execute(
                "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()["rowid"]
            self._conn.execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                (rowid, segmented),
            )
            self._conn.commit()
        logger.info("Saved memory %s for user %s: %s", memory_id, user_id, content[:80])
        return memory_id

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """FTS5 search + BM25 scoring + time decay.

        The query is jieba-segmented before building the FTS query.
        Falls back to LIKE search if FTS5 returns empty results or errors.
        """
        query = query.strip()
        if not query:
            return self.get_recent(user_id, limit)

        # Build FTS query with jieba segmentation
        tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
        if not tokens:
            return self.get_recent(user_id, limit)

        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        sql = """
            SELECT m.*, bm25(memories_fts) AS rank
            FROM memories m
            JOIN memories_fts ON memories_fts.rowid = m.rowid
            WHERE memories_fts MATCH ? AND m.user_id = ?
            ORDER BY rank
            LIMIT ?
        """
        rows = []
        with self._lock:
            try:
                rows = self._conn.execute(sql, (fts_query, user_id, limit * 2)).fetchall()
            except sqlite3.OperationalError:
                logger.debug("FTS query failed, falling back to LIKE search")

        # FTS5 returned no results; fall back to LIKE search
        if not rows:
            return self._search_like(user_id, query, limit)

        now = time.time()
        entries = []
        for row in rows:
            entry = self._row_to_entry(row)
            # BM25 returns negative values; smaller means more relevant -- take absolute value
            bm25 = abs(row["rank"])
            days = (now - entry.updated_at) / 86400
            decay = 1.0 / (1.0 + days * _DECAY_RATE)
            entry.score = bm25 * decay
            entries.append(entry)

        # Sort by score descending
        entries.sort(key=lambda e: e.score, reverse=True)
        return entries[:limit]

    def _search_like(self, user_id: str, query: str, limit: int) -> list[MemoryEntry]:
        """LIKE fallback search: uses jieba segmentation and keyword matching."""
        tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
        if not tokens:
            return self.get_recent(user_id, limit)

        # Build OR conditions: any keyword match is sufficient
        conditions = " OR ".join("content LIKE ?" for _ in tokens)
        params: list = [user_id] + [f"%{t}%" for t in tokens] + [limit]
        sql = f"""
            SELECT * FROM memories
            WHERE user_id = ? AND ({conditions})
            ORDER BY updated_at DESC
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_recent(self, user_id: str, limit: int = 10) -> list[MemoryEntry]:
        """Get the most recent memories."""
        sql = "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (user_id, limit)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_by_category(
        self, user_id: str, category: str, limit: int = 10
    ) -> list[MemoryEntry]:
        """Get memories by category."""
        sql = (
            "SELECT * FROM memories WHERE user_id = ? AND category = ? "
            "ORDER BY updated_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (user_id, category, limit)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def delete(self, memory_id: str) -> bool:
        """Delete by ID."""
        with self._lock:
            # Delete the FTS index first
            row = self._conn.execute(
                "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row:
                self._conn.execute(
                    "DELETE FROM memories_fts WHERE rowid = ?", (row["rowid"],)
                )
            # Then delete from the main table
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_by_query(self, user_id: str, query: str) -> int:
        """Search and delete matching memories, returning the number deleted."""
        matches = self.search(user_id, query, limit=3)
        if not matches:
            return 0
        deleted = 0
        for m in matches:
            # Only delete those with high similarity
            sim = SequenceMatcher(None, query.lower(), m.content.lower()).ratio()
            if sim >= 0.5:
                if self.delete(m.id):
                    deleted += 1
        return deleted

    def count(self, user_id: str) -> int:
        """Get the total number of memories for a user."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row["cnt"] if row else 0

    def enforce_limit(self, user_id: str, max_count: int) -> int:
        """Enforce a limit on the number of user memories, deleting the oldest excess entries. Returns the number deleted."""
        current = self.count(user_id)
        if current <= max_count:
            return 0
        to_delete = current - max_count

        with self._lock:
            # Find the rowids of the records to delete
            rows = self._conn.execute(
                "SELECT rowid FROM memories WHERE user_id = ? ORDER BY updated_at ASC LIMIT ?",
                (user_id, to_delete),
            ).fetchall()

            if not rows:
                return 0

            rowids = [r["rowid"] for r in rows]
            placeholders = ",".join("?" for _ in rowids)

            # Delete FTS first
            self._conn.execute(
                f"DELETE FROM memories_fts WHERE rowid IN ({placeholders})", rowids
            )
            # Then delete from the main table
            cur = self._conn.execute(
                f"DELETE FROM memories WHERE rowid IN ({placeholders})", rowids
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            category=row["category"],
            importance=row["importance"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
        )
