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
        # Signature: (result, tool_call) — result is threaded, tool_call is context
        hooks = HookRegistry()
        captured = []

        @hooks.on("after_tool_execute")
        def capture(result, tool_call):
            captured.append((result, tool_call.name))
            return result.upper()

        tc = ToolCall(id="1", name="greet", arguments={})
        out = hooks.run("after_tool_execute", "hello", tc)
        self.assertEqual(out, "HELLO")
        self.assertEqual(captured, [("hello", "greet")])

    def test_after_tool_execute_no_hooks_returns_none(self) -> None:
        # Critical: no hooks → run() returns None, caller falls back to original result
        hooks = HookRegistry()
        tc = ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})
        out = hooks.run("after_tool_execute", 3, tc)
        self.assertIsNone(out)

    def test_after_tool_execute_falsy_result_preserved(self) -> None:
        # 0, "", False are valid tool results — must not be swallowed by `or` pattern
        from agent.core.agent import Agent
        from agent.core.message import LLMResponse, ToolCall

        for falsy_val in (0, "", False, []):
            with self.subTest(result=falsy_val):
                tool_resp = LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="x", name="zero", arguments={})],
                )
                final_resp = LLMResponse(content="done")
                llm = MockLLM([tool_resp, final_resp])

                from agent.core.tool import ToolRegistry
                tools = ToolRegistry()

                @tools.register(description="returns falsy value")
                def zero() -> Any:
                    return falsy_val

                agent = Agent(llm=llm, tools=tools)
                agent.run("go")
                # Tool result message should contain str(falsy_val), not str(ToolCall)
                tool_msgs = [m for m in agent.memory.messages
                             if m.role == Role.TOOL]
                self.assertEqual(len(tool_msgs), 1)
                self.assertEqual(tool_msgs[0].content, str(falsy_val))

    def test_after_tool_execute_multi_hook_chain(self) -> None:
        # Each hook sees the latest result, not the original
        hooks = HookRegistry()

        @hooks.on("after_tool_execute")
        def h1(result, tool_call):
            return result * 2

        @hooks.on("after_tool_execute")
        def h2(result, tool_call):
            # result here should be the output of h1 (6), not the original (3)
            self.assertIsInstance(result, int)
            return result + 1

        tc = ToolCall(id="1", name="calc", arguments={})
        out = hooks.run("after_tool_execute", 3, tc)
        self.assertEqual(out, 7)  # (3*2)+1

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
