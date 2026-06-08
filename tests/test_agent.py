"""Unit tests for base agent components (no network calls)."""
import sys
import os
import unittest
from typing import Any, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.agent import Agent
from agent.core.memory import ConversationMemory
from agent.core.message import LLMResponse, Message, Role, ToolCall
from agent.core.tool import ToolRegistry
from agent.core.tool_selector import ToolSelector
from agent.llm.base import BaseLLM


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__("mock-model")
        self._responses = list(responses)
        self._call_index = 0

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
        else:
            resp = LLMResponse(content="(no more mock responses)")
        self._call_index += 1
        return resp

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class TestConversationMemory(unittest.TestCase):
    def test_system_prompt_prepended(self) -> None:
        mem = ConversationMemory(system_prompt="Be brief.")
        mem.add(Message(role=Role.USER, content="Hello"))
        msgs = mem.get_messages()
        self.assertEqual(msgs[0].role, Role.SYSTEM)
        self.assertEqual(msgs[0].content, "Be brief.")
        self.assertEqual(msgs[1].content, "Hello")

    def test_max_messages_trimming(self) -> None:
        mem = ConversationMemory(max_messages=3)
        for i in range(5):
            mem.add(Message(role=Role.USER, content=f"msg{i}"))
        self.assertEqual(len(mem.messages), 3)
        self.assertEqual(mem.messages[-1].content, "msg4")

    def test_clear(self) -> None:
        mem = ConversationMemory()
        mem.add(Message(role=Role.USER, content="x"))
        mem.clear()
        self.assertEqual(len(mem), 0)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry(unittest.TestCase):
    def test_register_and_execute(self) -> None:
        reg = ToolRegistry()

        @reg.register(description="Multiply two numbers.")
        def multiply(a: int, b: int) -> int:
            return a * b

        tool = reg.get("multiply")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.execute(a=3, b=7), 21)

    def test_schema_generation(self) -> None:
        reg = ToolRegistry()

        @reg.register(description="Greet a user by name.")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        schemas = reg.get_schemas()
        self.assertEqual(len(schemas), 1)
        schema = schemas[0]
        self.assertEqual(schema["name"], "greet")
        self.assertIn("name", schema["parameters"]["properties"])
        self.assertIn("name", schema["parameters"]["required"])

    def test_unknown_tool_returns_none(self) -> None:
        reg = ToolRegistry()
        self.assertIsNone(reg.get("does_not_exist"))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent(unittest.TestCase):
    def test_simple_response(self) -> None:
        llm = MockLLM([LLMResponse(content="Paris.")])
        agent = Agent(llm=llm, system_prompt="Be concise.")
        result = agent.run("Capital of France?")
        self.assertEqual(result, "Paris.")

    def test_memory_accumulates(self) -> None:
        llm = MockLLM([
            LLMResponse(content="Hello!"),
            LLMResponse(content="You said hello."),
        ])
        agent = Agent(llm=llm)
        agent.run("Hello")
        agent.run("What did I say?")
        # 2 user + 2 assistant
        self.assertEqual(len(agent.memory.messages), 4)

    def test_tool_call_cycle(self) -> None:
        llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="add", arguments={"a": 10, "b": 5})],
            ),
            LLMResponse(content="The answer is 15."),
        ])
        tools = ToolRegistry()

        @tools.register(description="Add two integers.")
        def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(llm=llm, tools=tools)
        result = agent.run("What is 10 + 5?")
        self.assertEqual(result, "The answer is 15.")

    def test_unknown_tool_returns_error_string(self) -> None:
        llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="x", name="nonexistent", arguments={})],
            ),
            LLMResponse(content="Sorry, I couldn't do that."),
        ])
        agent = Agent(llm=llm)
        result = agent.run("Call nonexistent tool")
        # Should not raise; error message propagated back to LLM
        self.assertIsInstance(result, str)

    def test_reset_clears_history(self) -> None:
        llm = MockLLM([LLMResponse(content="Hi")])
        agent = Agent(llm=llm)
        agent.run("Hello")
        agent.reset()
        self.assertEqual(len(agent.memory.messages), 0)

    def test_max_tool_iterations_guard(self) -> None:
        # LLM always returns a tool call → should hit the iteration cap
        tool_call_resp = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="loop", arguments={})],
        )
        llm = MockLLM([tool_call_resp] * 20)
        tools = ToolRegistry()

        @tools.register(description="Loops forever.")
        def loop() -> str:
            return "looping"

        agent = Agent(llm=llm, max_tool_iterations=3)
        result = agent.run("Loop please")
        self.assertIn("Max tool iterations", result)


class TestHumanInTheLoop(unittest.TestCase):
    """Approval gating for tools marked requires_approval."""

    def _agent_with_danger(self, approval_callback):
        llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="rm", arguments={"path": "/x"})],
            ),
            LLMResponse(content="all done"),
        ])
        tools = ToolRegistry()

        @tools.register(description="Delete a file.", requires_approval=True)
        def rm(path: str) -> str:
            return f"deleted {path}"

        return Agent(llm=llm, tools=tools, approval_callback=approval_callback), tools

    def test_approved_tool_executes(self) -> None:
        agent, _ = self._agent_with_danger(approval_callback=lambda tc: True)
        agent.run("delete it")
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertEqual(tool_msg.content, "deleted /x")

    def test_denied_tool_is_blocked(self) -> None:
        agent, _ = self._agent_with_danger(approval_callback=lambda tc: False)
        agent.run("delete it")
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertIn("denied", tool_msg.content.lower())
        self.assertNotIn("deleted", tool_msg.content)

    def test_requires_approval_without_callback_is_fail_closed(self) -> None:
        # No approval_callback configured → sensitive tool must NOT run.
        agent, _ = self._agent_with_danger(approval_callback=None)
        agent.run("delete it")
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertIn("denied", tool_msg.content.lower())

    def test_callback_receives_tool_call(self) -> None:
        seen = []
        agent, _ = self._agent_with_danger(
            approval_callback=lambda tc: seen.append(tc) or True
        )
        agent.run("delete it")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].name, "rm")
        self.assertEqual(seen[0].arguments, {"path": "/x"})

    def test_non_sensitive_tool_runs_without_callback(self) -> None:
        llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="safe", arguments={})],
            ),
            LLMResponse(content="ok"),
        ])
        tools = ToolRegistry()

        @tools.register(description="Safe op.")
        def safe() -> str:
            return "safe-result"

        agent = Agent(llm=llm, tools=tools)  # no approval_callback
        agent.run("go")
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertEqual(tool_msg.content, "safe-result")


class TestToolSelectorIntegration(unittest.TestCase):
    """Agent forwards only RAG-selected tool schemas to the LLM."""

    class _CapturingLLM(BaseLLM):
        def __init__(self, responses):
            super().__init__("mock")
            self._responses = list(responses)
            self._i = 0
            self.tools_seen = []

        def chat(self, messages, tools=None, **kwargs):
            self.tools_seen.append(tools)
            r = self._responses[self._i] if self._i < len(self._responses) else LLMResponse(content="end")
            self._i += 1
            return r

        def stream(self, messages, **kwargs):
            raise NotImplementedError

    def _registry(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.register(description="Delete a file from disk.")
        def delete_file(path: str) -> str:
            return "deleted"

        @reg.register(description="Add two integers together.")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        return reg

    def test_only_relevant_schema_sent(self) -> None:
        reg = self._registry()
        llm = self._CapturingLLM([LLMResponse(content="done")])
        agent = Agent(llm=llm, tools=reg, tool_selector=ToolSelector(reg, top_k=1))
        agent.run("please delete the file")
        sent = llm.tools_seen[0]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["name"], "delete_file")

    def test_without_selector_all_schemas_sent(self) -> None:
        reg = self._registry()
        llm = self._CapturingLLM([LLMResponse(content="done")])
        agent = Agent(llm=llm, tools=reg)
        agent.run("please delete the file")
        self.assertEqual(len(llm.tools_seen[0]), 2)

    def test_selected_tool_still_executes(self) -> None:
        reg = self._registry()
        llm = self._CapturingLLM([
            LLMResponse(content="", tool_calls=[ToolCall(id="1", name="delete_file", arguments={"path": "/t"})]),
            LLMResponse(content="done"),
        ])
        agent = Agent(llm=llm, tools=reg, tool_selector=ToolSelector(reg, top_k=1))
        agent.run("delete the file")
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertEqual(tool_msg.content, "deleted")


class TestToolSchema(unittest.TestCase):
    """Verify _hint_to_json_type handles generic and Optional annotations."""

    def test_generic_list_type(self) -> None:
        from typing import List
        reg = ToolRegistry()

        @reg.register(description="Accepts a list of strings.")
        def f(items: List[str]) -> str:
            return ",".join(items)

        schema = reg.get_schemas()[0]
        self.assertEqual(schema["parameters"]["properties"]["items"]["type"], "array")

    def test_optional_type(self) -> None:
        from typing import Optional as Opt
        reg = ToolRegistry()

        @reg.register(description="Optional int param.")
        def f(x: Opt[int] = None) -> int:
            return x or 0

        schema = reg.get_schemas()[0]
        # Optional[int] → "integer"
        self.assertEqual(schema["parameters"]["properties"]["x"]["type"], "integer")
        # Has default → not required
        self.assertNotIn("x", schema["parameters"]["required"])

    def test_bare_list_and_dict(self) -> None:
        reg = ToolRegistry()

        @reg.register(description="Bare list and dict.")
        def f(items: list, mapping: dict) -> None:
            pass

        props = reg.get_schemas()[0]["parameters"]["properties"]
        self.assertEqual(props["items"]["type"], "array")
        self.assertEqual(props["mapping"]["type"], "object")

    def test_tool_call_cycle_correct_result_in_memory(self) -> None:
        """Verify the tool result (not the ToolCall object) is stored in memory."""
        llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="zero", arguments={})],
            ),
            LLMResponse(content="done"),
        ])
        tools = ToolRegistry()

        @tools.register(description="Returns zero.")
        def zero() -> int:
            return 0

        agent = Agent(llm=llm, tools=tools)
        agent.run("go")
        tool_msg = next(m for m in agent.memory.messages if m.role == Role.TOOL)
        self.assertEqual(tool_msg.content, "0")  # not str(ToolCall(...))


if __name__ == "__main__":
    unittest.main(verbosity=2)
