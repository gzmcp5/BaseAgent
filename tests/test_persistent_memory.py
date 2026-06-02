"""Tests for SQLite-backed PersistentMemory."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.persistent_memory import PersistentMemory
from agent.core.message import Message, Role, ToolCall


class TestPersistentMemoryNewSession(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self) -> None:
        os.unlink(self.db)

    def test_creates_session_and_adds_messages(self) -> None:
        mem = PersistentMemory(self.db, system_prompt="Be helpful.")
        mem.add(Message(role=Role.USER, content="Hello"))
        mem.add(Message(role=Role.ASSISTANT, content="Hi there"))
        self.assertEqual(len(mem.messages), 2)
        mem.close()

    def test_system_prompt_prepended_in_get_messages(self) -> None:
        mem = PersistentMemory(self.db, system_prompt="Be helpful.")
        mem.add(Message(role=Role.USER, content="Hi"))
        msgs = mem.get_messages()
        self.assertEqual(msgs[0].role, Role.SYSTEM)
        self.assertEqual(msgs[0].content, "Be helpful.")
        mem.close()

    def test_max_messages_trimming(self) -> None:
        mem = PersistentMemory(self.db, max_messages=3)
        for i in range(5):
            mem.add(Message(role=Role.USER, content=f"msg{i}"))
        self.assertEqual(len(mem.messages), 3)
        mem.close()


class TestPersistentMemoryRestore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self) -> None:
        os.unlink(self.db)

    def test_messages_survive_process_restart(self) -> None:
        mem = PersistentMemory(self.db, system_prompt="Sys")
        mem.add(Message(role=Role.USER, content="Remember this"))
        mem.add(Message(role=Role.ASSISTANT, content="Remembered"))
        session_id = mem.session_id
        mem.close()

        mem2 = PersistentMemory(self.db, session_id=session_id)
        self.assertEqual(len(mem2.messages), 2)
        self.assertEqual(mem2.messages[0].content, "Remember this")
        self.assertEqual(mem2.messages[1].content, "Remembered")
        self.assertEqual(mem2.system_prompt, "Sys")
        mem2.close()

    def test_invalid_session_raises(self) -> None:
        with self.assertRaises(ValueError):
            PersistentMemory(self.db, session_id="nonexistent-id")

    def test_tool_calls_roundtrip(self) -> None:
        mem = PersistentMemory(self.db)
        tc = ToolCall(id="call-1", name="get_weather", arguments={"city": "Seoul"})
        mem.add(Message(role=Role.ASSISTANT, content="", tool_calls=[tc]))
        session_id = mem.session_id
        mem.close()

        mem2 = PersistentMemory(self.db, session_id=session_id)
        restored = mem2.messages[0]
        self.assertIsNotNone(restored.tool_calls)
        self.assertEqual(restored.tool_calls[0].name, "get_weather")
        self.assertEqual(restored.tool_calls[0].arguments["city"], "Seoul")
        mem2.close()


class TestPersistentMemoryUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = self.tmp.name

    def tearDown(self) -> None:
        os.unlink(self.db)

    def test_list_sessions(self) -> None:
        for prompt in ("Session A", "Session B", "Session C"):
            mem = PersistentMemory(self.db, system_prompt=prompt)
            mem.add(Message(role=Role.USER, content="hi"))
            mem.close()

        sessions = PersistentMemory.list_sessions(self.db)
        self.assertEqual(len(sessions), 3)
        self.assertIn("id", sessions[0])
        self.assertIn("message_count", sessions[0])
        self.assertEqual(sessions[0]["message_count"], 1)

    def test_delete_session(self) -> None:
        mem = PersistentMemory(self.db)
        mem.add(Message(role=Role.USER, content="bye"))
        mem.delete_session()
        mem.close()

        sessions = PersistentMemory.list_sessions(self.db)
        self.assertEqual(len(sessions), 0)

    def test_context_manager(self) -> None:
        with PersistentMemory(self.db, system_prompt="ctx") as mem:
            mem.add(Message(role=Role.USER, content="test"))
            sid = mem.session_id

        sessions = PersistentMemory.list_sessions(self.db)
        self.assertEqual(sessions[0]["id"], sid)

    def test_clear_does_not_delete_db_records(self) -> None:
        mem = PersistentMemory(self.db)
        mem.add(Message(role=Role.USER, content="keep me"))
        sid = mem.session_id
        mem.clear()
        self.assertEqual(len(mem.messages), 0)
        mem.close()

        mem2 = PersistentMemory(self.db, session_id=sid)
        self.assertEqual(len(mem2.messages), 1)
        mem2.close()


if __name__ == "__main__":
    unittest.main()
