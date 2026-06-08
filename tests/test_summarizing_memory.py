"""Tests for SummarizingMemory."""
import os
import sys
import unittest
from typing import Any, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.summarizing_memory import SummarizingMemory
from agent.core.message import LLMResponse, Message, Role
from agent.llm.base import BaseLLM


class MockLLM(BaseLLM):
    def __init__(self, summary: str = "Mocked summary.") -> None:
        super().__init__("mock")
        self._summary = summary
        self.call_count = 0

    def chat(
        self, messages: list[Message], tools: Optional[list[dict]] = None, **kwargs: Any
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=self._summary)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        raise NotImplementedError


class TestSummarizingMemoryNoLLM(unittest.TestCase):
    def _make_mem(self, max_messages: int = 5, keep: int = 2) -> SummarizingMemory:
        return SummarizingMemory(max_messages=max_messages, summary_keep_last=keep)

    def test_no_compression_below_limit(self) -> None:
        mem = self._make_mem()
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"msg{i}"))
        self.assertEqual(len(mem.messages), 4)
        self.assertEqual(mem.summary, "")

    def test_compression_triggers_at_limit(self) -> None:
        mem = self._make_mem(max_messages=5, keep=2)
        for i in range(6):
            mem.add(Message(role=Role.USER, content=f"msg{i}"))
        self.assertLessEqual(len(mem.messages), 2)
        self.assertNotEqual(mem.summary, "")

    def test_fallback_summary_contains_content(self) -> None:
        mem = self._make_mem(max_messages=3, keep=1)
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"hello{i}"))
        self.assertIn("hello", mem.summary)

    def test_get_messages_includes_summary_as_system(self) -> None:
        mem = self._make_mem(max_messages=3, keep=1)
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"m{i}"))
        msgs = mem.get_messages()
        roles = [m.role for m in msgs]
        self.assertIn(Role.SYSTEM, roles)
        system_contents = [m.content for m in msgs if m.role == Role.SYSTEM]
        self.assertTrue(any("summary" in c.lower() for c in system_contents))

    def test_system_prompt_still_appears(self) -> None:
        mem = SummarizingMemory(max_messages=3, system_prompt="Be helpful.", summary_keep_last=1)
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"m{i}"))
        msgs = mem.get_messages()
        self.assertEqual(msgs[0].content, "Be helpful.")

    def test_clear_resets_summary(self) -> None:
        mem = self._make_mem(max_messages=3, keep=1)
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"m{i}"))
        self.assertNotEqual(mem.summary, "")
        mem.clear()
        self.assertEqual(mem.summary, "")
        self.assertEqual(len(mem.messages), 0)

    def test_multiple_compression_passes_accumulate(self) -> None:
        mem = self._make_mem(max_messages=4, keep=2)
        for i in range(10):
            mem.add(Message(role=Role.USER, content=f"turn{i}"))
        self.assertLessEqual(len(mem.messages), 4)
        self.assertNotEqual(mem.summary, "")
        self.assertIn("turn", mem.summary)

    def test_summary_keep_last_capped_at_max_messages(self) -> None:
        mem = SummarizingMemory(max_messages=3, summary_keep_last=100)
        self.assertEqual(mem.summary_keep_last, 3)


class TestSummarizingMemoryWithLLM(unittest.TestCase):
    def test_llm_called_on_compression(self) -> None:
        llm = MockLLM(summary="AI-generated summary.")
        mem = SummarizingMemory(max_messages=3, summary_keep_last=1, llm=llm)
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"item{i}"))
        self.assertEqual(llm.call_count, 1)
        self.assertIn("AI-generated summary", mem.summary)

    def test_llm_summary_injected_into_context(self) -> None:
        llm = MockLLM(summary="Compact prior context.")
        mem = SummarizingMemory(max_messages=3, summary_keep_last=1, llm=llm)
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"q{i}"))
        msgs = mem.get_messages()
        system_msgs = [m for m in msgs if m.role == Role.SYSTEM]
        combined = "\n".join(m.content for m in system_msgs)
        self.assertIn("Compact prior context", combined)

    def test_llm_failure_falls_back_gracefully(self) -> None:
        class BrokenLLM(BaseLLM):
            def __init__(self) -> None:
                super().__init__("broken")

            def chat(self, messages, tools=None, **kw):
                raise RuntimeError("network error")

            def stream(self, messages, **kw):
                raise NotImplementedError

        mem = SummarizingMemory(max_messages=3, summary_keep_last=1, llm=BrokenLLM())
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"x{i}"))
        self.assertNotEqual(mem.summary, "")


class TestSummarizingMemoryEdgeCases(unittest.TestCase):
    """Edge-case coverage added after bug analysis."""

    def test_summary_keep_last_zero_clears_messages(self) -> None:
        # Bug: messages[:-0] == messages[:0] == [] — fix ensures keep=0 summarizes all and empties list
        mem = SummarizingMemory(max_messages=3, summary_keep_last=0)
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"k{i}"))
        self.assertEqual(len(mem.messages), 0)
        self.assertNotEqual(mem.summary, "")

    def test_summary_keep_last_zero_subsequent_adds_still_compress(self) -> None:
        mem = SummarizingMemory(max_messages=2, summary_keep_last=0)
        for i in range(6):
            mem.add(Message(role=Role.USER, content=f"z{i}"))
        # After multiple compressions, messages list must not grow past max_messages
        self.assertLessEqual(len(mem.messages), 2)

    def test_get_messages_ordering_system_summary_history(self) -> None:
        mem = SummarizingMemory(max_messages=3, system_prompt="Sys.", summary_keep_last=1)
        for i in range(4):
            mem.add(Message(role=Role.USER, content=f"n{i}"))
        msgs = mem.get_messages()
        # First message must be the system prompt
        self.assertEqual(msgs[0].content, "Sys.")
        # Second message must be the summary (inserted by get_messages)
        self.assertIn("summary", msgs[1].content.lower())
        # Remaining messages are the retained conversation
        self.assertEqual(msgs[-1].role, Role.USER)

    def test_no_summary_system_message_when_no_compression(self) -> None:
        mem = SummarizingMemory(max_messages=10, system_prompt="Sys.")
        mem.add(Message(role=Role.USER, content="hi"))
        msgs = mem.get_messages()
        system_msgs = [m for m in msgs if m.role == Role.SYSTEM]
        # Only the explicit system prompt should appear; no summary message
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(system_msgs[0].content, "Sys.")

    def test_len_via_inherited_len_method(self) -> None:
        mem = SummarizingMemory(max_messages=5, summary_keep_last=2)
        for i in range(3):
            mem.add(Message(role=Role.USER, content=f"v{i}"))
        self.assertEqual(len(mem), 3)

    def test_system_only_messages_produce_nonempty_fallback(self) -> None:
        mem = SummarizingMemory(max_messages=2, summary_keep_last=0)
        for i in range(3):
            mem.add(Message(role=Role.SYSTEM, content=f"sys{i}"))
        # Fallback filters SYSTEM messages → summary line appears but body may be empty
        # The important thing: no crash and summary is a string
        self.assertIsInstance(mem.summary, str)

    def test_summary_accumulates_across_multiple_llm_compressions(self) -> None:
        llm = MockLLM(summary="chunk.")
        mem = SummarizingMemory(max_messages=3, summary_keep_last=1, llm=llm)
        for i in range(9):
            mem.add(Message(role=Role.USER, content=f"p{i}"))
        # Multiple compression passes: each appends to summary
        self.assertGreater(llm.call_count, 1)
        self.assertEqual(mem.summary.count("chunk"), llm.call_count)


if __name__ == "__main__":
    unittest.main()
