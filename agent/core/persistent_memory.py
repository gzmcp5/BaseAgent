from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from .memory import ConversationMemory
from .message import Message, Role, ToolCall

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    system_prompt TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL DEFAULT '',
    tool_calls   TEXT,
    tool_call_id TEXT,
    name         TEXT,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


class PersistentMemory(ConversationMemory):
    """SQLite-backed memory that survives process restarts.

    Usage — new session:
        mem = PersistentMemory("agent.db", system_prompt="You are helpful.")

    Usage — resume existing session:
        mem = PersistentMemory("agent.db", session_id="<uuid>")

    Usage — as context manager (auto-closes DB connection):
        with PersistentMemory("agent.db") as mem:
            agent = Agent(llm=llm, memory=mem)
            agent.run("hello")

    List all sessions:
        PersistentMemory.list_sessions("agent.db")
    """

    def __init__(
        self,
        db_path: str = "agent_memory.db",
        session_id: Optional[str] = None,
        max_messages: int = 100,
        system_prompt: str = "",
    ) -> None:
        super().__init__(max_messages=max_messages, system_prompt=system_prompt)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # If session setup fails (e.g. unknown session_id), close the
        # already-open connection so we don't leak a SQLite handle.
        try:
            self._init_schema()
            if session_id:
                self.session_id = session_id
                self._load_session()
            else:
                self.session_id = str(uuid.uuid4())
                self._create_session()
        except Exception:
            self._conn.close()
            raise

    # ------------------------------------------------------------------
    # Schema / session lifecycle
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _create_session(self) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, system_prompt, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (self.session_id, self.system_prompt, self._now(), self._now()),
            )

    def _load_session(self) -> None:
        row = self._conn.execute(
            "SELECT system_prompt FROM sessions WHERE id = ?",
            (self.session_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Session '{self.session_id}' not found in '{self.db_path}'"
            )
        if not self.system_prompt:
            self.system_prompt = row["system_prompt"]

        rows = self._conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, name "
            "FROM messages WHERE session_id = ? ORDER BY id",
            (self.session_id,),
        ).fetchall()

        self.messages = [self._row_to_message(r) for r in rows]
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    # ------------------------------------------------------------------
    # ConversationMemory overrides
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        super().add(message)
        tool_calls_json: Optional[str] = None
        if message.tool_calls:
            tool_calls_json = json.dumps(
                [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in message.tool_calls
                ]
            )
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages "
                "(session_id, role, content, tool_calls, tool_call_id, name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    message.role.value,
                    message.content,
                    tool_calls_json,
                    message.tool_call_id,
                    message.name,
                    self._now(),
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (self._now(), self.session_id),
            )

    def clear(self) -> None:
        """Clear the in-memory buffer only; DB records are preserved."""
        super().clear()

    # ------------------------------------------------------------------
    # Session management helpers
    # ------------------------------------------------------------------

    def delete_session(self) -> None:
        """Permanently delete this session and all its messages from the DB."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (self.session_id,)
            )
            self._conn.execute(
                "DELETE FROM sessions WHERE id = ?", (self.session_id,)
            )
        self.messages.clear()

    def close(self) -> None:
        """Close the underlying DB connection."""
        self._conn.close()

    def __enter__(self) -> PersistentMemory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Serialisation helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        tool_calls: Optional[list[ToolCall]] = None
        if row["tool_calls"]:
            raw = json.loads(row["tool_calls"])
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in raw
            ]
        return Message(
            role=Role(row["role"]),
            content=row["content"],
            tool_calls=tool_calls,
            tool_call_id=row["tool_call_id"],
            name=row["name"],
        )

    # ------------------------------------------------------------------
    # Class-level utilities
    # ------------------------------------------------------------------

    @staticmethod
    def list_sessions(db_path: str) -> list[dict]:
        """Return metadata for all sessions in *db_path*, newest-first.

        Each dict has keys: id, system_prompt, created_at, updated_at,
        message_count.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            rows = conn.execute(
                "SELECT s.id, s.system_prompt, s.created_at, s.updated_at, "
                "  COUNT(m.id) AS message_count "
                "FROM sessions s "
                "LEFT JOIN messages m ON m.session_id = s.id "
                "GROUP BY s.id "
                "ORDER BY s.updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
