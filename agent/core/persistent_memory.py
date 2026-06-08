"""대화를 SQLite 파일에 영구 저장해, 프로그램을 껐다 켜도 이어갈 수 있는 메모리.

지금까지의 메모리들은 모두 프로세스가 끝나면 사라진다(인메모리). PersistentMemory는
모든 메시지를 SQLite DB에 기록하므로, 나중에 같은 session_id로 다시 열면 과거 대화가
그대로 복원된다. 여러 대화를 'session' 단위로 구분해 한 DB 파일에 함께 보관한다.
표준 라이브러리 sqlite3만 쓰므로 별도 설치가 필요 없다.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from .memory import ConversationMemory
from .message import Message, Role, ToolCall

# DB 테이블 정의. sessions(대화 단위) 1 : N messages(메시지). IF NOT EXISTS로 매번 안전하게 실행.
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
        # check_same_thread=False: 다른 스레드에서도 이 연결을 쓸 수 있게 허용(AsyncAgent 대비).
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row     # 결과를 컬럼명으로 접근(row["content"]) 가능하게.
        self._conn.execute("PRAGMA journal_mode=WAL")  # 동시 읽기/쓰기 성능 향상 모드.
        self._conn.execute("PRAGMA foreign_keys=ON")   # 외래키 제약(CASCADE 삭제 등)을 켠다.
        # 세션 준비 중 실패하면(예: 없는 session_id), 이미 연 연결을 닫아 핸들 누수를 막는다.
        try:
            self._init_schema()
            if session_id:
                # 기존 세션 이어가기: id를 받고 DB에서 과거 메시지를 불러온다.
                self.session_id = session_id
                self._load_session()
            else:
                # 새 세션 시작: 무작위 UUID를 발급하고 sessions 테이블에 한 줄 만든다.
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
        # 1) 세션이 실제로 존재하는지 확인하면서 저장된 system_prompt를 가져온다.
        row = self._conn.execute(
            "SELECT system_prompt FROM sessions WHERE id = ?",
            (self.session_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Session '{self.session_id}' not found in '{self.db_path}'"
            )
        # 생성자에서 새 프롬프트를 주지 않았다면, DB에 저장돼 있던 프롬프트를 복원한다.
        if not self.system_prompt:
            self.system_prompt = row["system_prompt"]

        # 2) 이 세션의 모든 메시지를 id 순서(=시간 순서)대로 읽어온다.
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
        super().add(message)  # 먼저 인메모리 목록에 추가(부모 동작),
        # tool_calls는 객체 목록이라 DB에 바로 못 넣으므로 JSON 문자열로 직렬화한다.
        tool_calls_json: Optional[str] = None
        if message.tool_calls:
            tool_calls_json = json.dumps(
                [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in message.tool_calls
                ]
            )
        # with self._conn: 블록은 트랜잭션 — 두 INSERT/UPDATE가 모두 성공해야 커밋된다.
        # 값은 ? 자리표시자로 바인딩한다(SQL 인젝션 방지 + 타입 안전).
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
            # 메시지가 추가될 때마다 세션의 '마지막 수정 시각'을 갱신(목록 정렬용).
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
        # DB의 한 행(row)을 다시 Message 객체로 되돌린다(add의 역과정).
        tool_calls: Optional[list[ToolCall]] = None
        if row["tool_calls"]:
            raw = json.loads(row["tool_calls"])  # JSON 문자열 → 객체 목록 복원.
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
        # 인스턴스를 만들지 않고도 DB의 모든 세션 목록을 훑어볼 수 있는 정적 메서드.
        # 자체 연결을 열고 finally에서 반드시 닫는다.
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            # 세션별 메시지 개수를 LEFT JOIN+COUNT로 함께 집계하고, 최근 수정순으로 정렬.
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
