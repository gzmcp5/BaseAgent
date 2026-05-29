"""Tests for HookRegistry middleware."""
import sys
import os
import unittest
from typing import Any, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.hooks import HookRegistry
from agent.core.agent import Agent
from agent.core.message import LLMResponse, Message, Role, ToolCall
from agent.llm.base import BaseLLM


class MockLLM(BaseLLM):
    def __init__(self, responses):
        super().__init__("mock")
        self._responses = list(responses)
        self._index = 0

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        r = self._responses[self._index] if self._index < len(self._responses) else LLMResponse(content="end")
        self._index += 1
        return r

    def stream(self, messages, **kwargs) -> Iterator[str]:
        yield "hello"


class TestHookRegistry(unittest.TestCase):
    def test_basic_hook_registration(self) -> None:
        hooks = HookRegistry()
        called = []

        @hooks.on("before_llm_call")
        def spy(messages):
            called.append(len(messages))

        hooks.run("before_llm_call", [])
        self.assertEqual(called, [0])

    def test_hook_transforms_value(self) -> None:
        hooks = HookRegistry()

        @hooks.on("before_llm_call")
        def add_system(messages):
            return [Message(role=Role.SYSTEM, content="injected")] + messages

        result = hooks.run("before_llm_call", [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "injected")

    def test_none_return_passes_through(self) -> None:
        hooks = HookRegistry()

        @hooks.on("before_llm_call")
        def noop(messages):
            return None  # should not change value

        original = [Message(role=Role.USER, content="hi")]
        result = hooks.run("before_llm_call", original)
        self.assertIs(result, original)

    def test_multiple_hooks_chained(self) -> None:
        hooks = HookRegistry()
        order = []

        @hooks.on("after_llm_response")
        def first(response):
            order.append("first")

        @hooks.on("after_llm_response")
        def second(response):
            order.append("second")

        hooks.run("after_llm_response", LLMResponse(content="x"))
        self.assertEqual(order, ["first", "second"])

    def test_invalid_event_raises(self) -> None:
        hooks = HookRegistry()
        with self.assertRaises(ValueError):
            hooks.on("nonexistent_event")

    def test_clear_specific_event(self) -> None:
        hooks = HookRegistry()
        hooks.register("before_llm_call", lambda m: m)
        hooks.register("after_llm_response", lambda r: r)
        hooks.clear("before_llm_call")
        self.assertEqual(len(hooks), 1)

    def test_after_tool_execute_two_args(self) -> None:
        hooks = HookRegistry()
        captured = []

        @hooks.on("after_tool_execute")
        def capture(tool_call, result):
            captured.append((tool_call.name, result))
            return result.upper()

        tc = ToolCall(id="1", name="greet", arguments={})
        out = hooks.run("after_tool_execute", tc, "hello")
        self.assertEqual(out, "HELLO")
        self.assertEqual(captured, [("greet", "hello")])

    def test_hooks_integrated_with_agent(self) -> None:
        """Hook should be called inside Agent.run()."""
        llm = MockLLM([LLMResponse(content="pong")])
        hooks = HookRegistry()
        received = []

        @hooks.on("before_llm_call")
        def capture(messages):
            received.append(messages[-1].content)

        agent = Agent(llm=llm, hooks=hooks)
        agent.run("ping")
        self.assertIn("ping", received)


if __name__ == "__main__":
    unittest.main(verbosity=2)
