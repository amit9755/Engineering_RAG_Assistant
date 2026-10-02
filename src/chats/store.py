"""Saved chat conversations (SQLite, data/chats.db). Only the most recent MAX_CHATS are kept."""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)

MAX_CHATS = 10
MAX_MESSAGES = 200
MAX_CONTENT_CHARS = 20000
CHAT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatStore:
    def __init__(self, db_path: Path = Path("data/chats.db"), max_chats: int = MAX_CHATS):
        self.db_path = Path(db_path)
        self.max_chats = max_chats
        self._lock = Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    chat_id   TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    position  INTEGER NOT NULL,
                    role      TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    sources   TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY (chat_id, position)
                );
            """)
            columns = {r["name"] for r in conn.execute("PRAGMA table_info(chats)")}
            if "owner" not in columns:   # migration: chats saved before accounts existed
                conn.execute("ALTER TABLE chats ADD COLUMN owner TEXT")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _title(messages: List[dict]) -> str:
        first = next((m["content"] for m in messages if m["role"] == "user"), "New chat")
        first = " ".join(first.split())
        return first if len(first) <= 60 else first[:57] + "..."

    def assign_unowned(self, owner: str) -> int:
        with self._lock, self._connect() as conn:
            return conn.execute("UPDATE chats SET owner = ? WHERE owner IS NULL", (owner,)).rowcount

    def save(self, chat_id: str, messages: List[dict], owner: str) -> dict:
        """Replace one of the owner's chats (create it if new), then keep only their newest chats."""
        if not CHAT_ID.match(chat_id or ""):
            raise ValueError("Invalid chat id")
        clean = []
        for m in messages[-MAX_MESSAGES:]:
            if m.get("role") not in ("user", "assistant") or not str(m.get("content", "")).strip():
                continue
            clean.append({"role": m["role"], "content": str(m["content"])[:MAX_CONTENT_CHARS],
                          "sources": [str(s)[:300] for s in (m.get("sources") or [])][:20]})
        if not clean:
            raise ValueError("A chat needs at least one message")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            current = conn.execute("SELECT owner FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if current and current["owner"] != owner:
                raise PermissionError("Chat not found")   # never overwrite another user's chat
            conn.execute(
                "INSERT INTO chats (id, title, created_at, updated_at, owner) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET title = excluded.title, updated_at = excluded.updated_at",
                (chat_id, self._title(clean), now, now, owner))
            conn.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
            conn.executemany(
                "INSERT INTO chat_messages (chat_id, position, role, content, sources) VALUES (?, ?, ?, ?, ?)",
                [(chat_id, i, m["role"], m["content"], json.dumps(m["sources"])) for i, m in enumerate(clean)])
            # Keep only each user's most recently updated chats.
            conn.execute("DELETE FROM chats WHERE owner = ? AND id NOT IN "
                         "(SELECT id FROM chats WHERE owner = ? ORDER BY updated_at DESC LIMIT ?)",
                         (owner, owner, self.max_chats))
        return {"id": chat_id, "title": self._title(clean), "updated_at": now, "message_count": len(clean)}

    def list(self, owner: str) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.title, c.updated_at, COUNT(m.position) AS message_count FROM chats c "
                "LEFT JOIN chat_messages m ON m.chat_id = c.id WHERE c.owner = ? GROUP BY c.id "
                "ORDER BY c.updated_at DESC", (owner,)).fetchall()
        return [dict(r) for r in rows]

    def get(self, chat_id: str, owner: str) -> Optional[dict]:
        with self._connect() as conn:
            chat = conn.execute("SELECT * FROM chats WHERE id = ? AND owner = ?", (chat_id, owner)).fetchone()
            if not chat:
                return None
            rows = conn.execute("SELECT role, content, sources FROM chat_messages WHERE chat_id = ? "
                                "ORDER BY position", (chat_id,)).fetchall()
        return {"id": chat["id"], "title": chat["title"], "updated_at": chat["updated_at"],
                "messages": [{"role": r["role"], "content": r["content"], "sources": json.loads(r["sources"])}
                             for r in rows]}

    def delete(self, chat_id: str, owner: str) -> bool:
        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM chats WHERE id = ? AND owner = ?", (chat_id, owner)).rowcount > 0


chat_store = ChatStore()
