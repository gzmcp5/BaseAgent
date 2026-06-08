"""Tests for AsyncAgent (concurrent tool execution, async context manager)."""
import asyncio
import sys
import os
import unittest
from typing import Any, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.async_agent import AsyncAgent
from agent.core.message import LLMResponse, Message, Role, ToolCall
from agent.core.tool import ToolRegistry
from agent.core.tool_selector import ToolSelector
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
        for chunk in ["Hello", ", ", "world", "!"]:
            yield chunk


def _run(coro):
    return asyncio.run(coro)


class TestAsyncAgentRun(unittest.TestCase):
    def test_simple_response(self) -> None:
        llm = MockLLM([LLMResponse(content="async pong")])
        agent = AsyncAgent(llm=llm)
        result = _run(agent.run("ping"))
        self.assertEqual(result, "async pong")

    def test_memory_accumulates(self) -> None:
        llm = MockLLM([
            LLMResponse(content="Hi!"),
            LLMResponse(content="You said hello."),
        ])
        agent = AsyncAgent(llm=llm)
        _run(agent.run("Hello"))
        _run(agent.run("What did I say?"))
        self.assertEqual(len(agent.memory.messages), 4)

    def test_tool_call_concurrent(self) -> None:
        """Two tool calls in one turn should both be executed."""
        two_calls = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="2", name="add", arguments={"a": 3, "b": 4}),
            ],
        )
        final = LLMResponse(content="Results: 3 and 7.")
        llm = MockLLM([two_calls, final])

        tools = ToolRegistry()

        @tools.register(description="Add two integers.")
        def add(a: int, b: int) -> int:
            return a + b

        agent = AsyncAgent(llm=llm, tools=tools)
        result = _run(agent.run("Add some numbers"))
        self.assertEqual(result, "Results: 3 and 7.")

        # Both tool results should be in memory
        tool_messages = [m for m in agent.memory.messages if m.role == Role.TOOL]
        self.assertEqual(len(tool_messages), 2)

    def test_reset_clears_memory(self) -> None:
        llm = MockLLM([LLMResponse(content="Hi")])
        agent = AsyncAgent(llm=llm)
        _run(agent.run("Hello"))
        agent.reset()
        self.assertEqual(len(agent.memory.messages), 0)

    def test_context_manager(self) -> None:
        llm = MockLLM([LLMResponse(content="Hi")])

        async def _test():
            async with AsyncAgent(llm=llm) as agent:
                return await agent.run("Hello")

        result = _run(_test())
        self.assertEqual(result, "Hi")


class TestAsyncAgentStream(unittest.TestCase):
    def test_stream_yields_chunks(self) -> None:
        llm = MockLLM([])

        async def _collect():
            agent = AsyncAgent(llm=llm)
            chunks = []
            async for chunk in agent.stream("Tell me something"):
                chunks.append(chunk)
            return chunks

        chunks = _run(_collect())
        self.assertEqual("".join(chunks), "Hello, world!")

    def test_stream_updates_memory(self) -> None:
        llm = MockLLM([])

        async def _test():
            agent = AsyncAgent(llm=llm)
            async for _ in agent.stream("hi"):
                pass
            return agent

        agent = _run(_test())
        assistant_msgs = [m for m in agent.memory.messages if m.role == Role.ASSISTANT]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0].content, "Hello, world!")


class TestAsyncHumanInTheLoop(unittest.TestCase):
    def _agent(self, approval_callback):
        llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="rm", arguments={"path": "/x"})],
            ),
            LLMResponse(content="done"),
        ])
        tools = ToolRegistry()

        @tools.register(description="Delete a file.", requires_approval=True)
        def rm(path: str) -> str:
            return f"deleted {path}"

        return AsyncAgent(llm=llm, tools=tools, approval_callback=approval_callback)

    def test_approved_executes(self) -> None:
        agent = self._agent(approval_callback=lambda tc: True)
        _run(agent.run("delete"))
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertEqual(tool_msg.content, "deleted /x")

    def test_denied_blocked(self) -> None:
        agent = self._agent(approval_callback=lambda tc: False)
        _run(agent.run("delete"))
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertIn("denied", tool_msg.content.lower())

    def test_fail_closed_without_callback(self) -> None:
        agent = self._agent(approval_callback=None)
        _run(agent.run("delete"))
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertIn("denied", tool_msg.content.lower())


class TestAsyncToolSelector(unittest.TestCase):
    def _registry(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.register(description="Delete a file from disk.")
        def delete_file(path: str) -> str:
            return "deleted"

        @reg.register(description="Add two integers.")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        return reg

    def test_selector_prunes_schemas(self) -> None:
        class _CapturingLLM(BaseLLM):
            def __init__(self):
                super().__init__("mock")
                self.tools_seen = []

            def chat(self, messages, tools=None, **kwargs):
                self.tools_seen.append(tools)
                return LLMResponse(content="done")

            def stream(self, messages, **kwargs):
                raise NotImplementedError

        reg = self._registry()
        llm = _CapturingLLM()
        agent = AsyncAgent(llm=llm, tools=reg, tool_selector=ToolSelector(reg, top_k=1))
        _run(agent.run("delete the file please"))
        self.assertEqual(len(llm.tools_seen[0]), 1)
        self.assertEqual(llm.tools_seen[0][0]["name"], "delete_file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
