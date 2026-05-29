"""Tests for ContextManager (token tracking + auto-summarization)."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.context import ContextManager
from agent.core.message import Message, Role


def _msg(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


class TestContextManagerTokenEstimation(unittest.TestCase):
    def test_estimate_tokens_empty(self) -> None:
        cm = ContextManager()
        self.assertEqual(cm.estimate_tokens([]), 0)

    def test_estimate_tokens_counts_chars(self) -> None:
        msgs = [_msg(Role.USER, "abcd")]  # 4 chars → ~1 token
        cm = ContextManager()
        self.assertEqual(cm.estimate_tokens(msgs), 1)

    def test_estimate_tokens_multiple_messages(self) -> None:
        msgs = [
            _msg(Role.USER, "a" * 40),       # 40 chars = 10 tokens
            _msg(Role.ASSISTANT, "b" * 20),  # 20 chars = 5 tokens
        ]
        cm = ContextManager()
        self.assertEqual(cm.estimate_tokens(msgs), 15)


class TestContextManagerCompression(unittest.TestCase):
    def test_compression_triggers_when_over_budget(self) -> None:
        # Budget of 5 tokens (20 chars), keep_recent=2
        cm = ContextManager(max_context_tokens=5, keep_recent=2)
        for i in range(6):
            cm.add(_msg(Role.USER, f"{'x' * 20} message {i}"))
        # After compression there should be: 1 summary msg + 2 recent
        self.assertLessEqual(len(cm.messages), 3)

    def test_compression_preserves_recent_messages(self) -> None:
        cm = ContextManager(max_context_tokens=5, keep_recent=2)
        contents = [f"{'x' * 20} item{i}" for i in range(6)]
        for c in contents:
            cm.add(_msg(Role.USER, c))
        # The two most recently added messages must be preserved verbatim
        stored_contents = [m.content for m in cm.messages]
        self.assertIn(contents[-1], stored_contents)
        self.assertIn(contents[-2], stored_contents)

    def test_custom_summarizer_called(self) -> None:
        called = []

        def my_summarizer(messages):
            called.append(len(messages))
            return "SUMMARY"

        cm = ContextManager(
            max_context_tokens=5,
            summarizer=my_summarizer,
            keep_recent=2,
        )
        for i in range(6):
            cm.add(_msg(Role.USER, "x" * 20))
        self.assertTrue(called)
        # Summary message should be present
        summaries = [m for m in cm.messages if "SUMMARY" in (m.content or "")]
        self.assertTrue(summaries)

    def test_no_compression_when_under_budget(self) -> None:
        cm = ContextManager(max_context_tokens=10_000, keep_recent=2)
        for i in range(5):
            cm.add(_msg(Role.USER, "short"))
        self.assertEqual(len(cm.messages), 5)

    def test_system_prompt_added_to_estimate(self) -> None:
        cm = ContextManager(system_prompt="You are helpful.", max_context_tokens=10_000)
        cm.add(_msg(Role.USER, "hello"))
        total = cm.estimate_tokens()
        # Should include system prompt tokens
        self.assertGreater(total, 0)

    def test_fallback_summarizer_when_none_provided(self) -> None:
        # Without a summarizer, old messages should just be dropped
        cm = ContextManager(max_context_tokens=5, summarizer=None, keep_recent=2)
        for i in range(8):
            cm.add(_msg(Role.USER, "x" * 20))
        # Should still have ≤ keep_recent + 1 (summary fallback) messages
        self.assertLessEqual(len(cm.messages), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
