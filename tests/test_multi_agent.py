"""Tests for multi-agent architecture: OrchestratorAgent and Pipeline."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, OrchestratorAgent, Pipeline
from agent.core.message import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Minimal mock helpers
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """Returns pre-set LLMResponse objects in order (cycling on exhaustion)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self._index = 0
        self.calls: list = []

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        resp = self._responses[self._index % len(self._responses)]
        self._index += 1
        self.calls.append(messages)
        return resp

    def stream(self, messages, **kwargs):
        yield "chunk"


class _EchoLLM:
    """Echoes the last user message, optionally with a prefix."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def chat(self, messages, **kwargs) -> LLMResponse:
        user_msg = next(
            (m.content for m in reversed(messages) if m.role.value == "user"), ""
        )
        return LLMResponse(content=f"{self.prefix}{user_msg}" if self.prefix else user_msg)

    def stream(self, messages, **kwargs):
        yield "chunk"


def _echo_agent(prefix: str = "") -> Agent:
    return Agent(llm=_EchoLLM(prefix), system_prompt="")


# ---------------------------------------------------------------------------
# OrchestratorAgent tests
# ---------------------------------------------------------------------------

class TestOrchestratorAgentTools(unittest.TestCase):
    def test_tool_names_match_agent_keys(self):
        orch = OrchestratorAgent(
            llm=_ScriptedLLM([LLMResponse(content="done")]),
            agents={"alpha": _echo_agent(), "beta": _echo_agent()},
        )
        names = {s["name"] for s in orch._inner.tools.get_schemas()}
        self.assertIn("ask_alpha", names)
        self.assertIn("ask_beta", names)

    def test_tool_count_matches_agent_count(self):
        orch = OrchestratorAgent(
            llm=_ScriptedLLM([LLMResponse(content="done")]),
            agents={"a": _echo_agent(), "b": _echo_agent(), "c": _echo_agent()},
        )
        self.assertEqual(len(orch._inner.tools), 3)


class TestOrchestratorAgentDelegation(unittest.TestCase):
    def test_delegates_task_to_sub_agent(self):
        sub = _echo_agent(prefix="[sub] ")
        # Orchestrator LLM: call the sub-agent tool, then synthesise
        scripted = _ScriptedLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="ask_sub", arguments={"task": "hello"})],
            ),
            LLMResponse(content="Orchestrator done: [sub] hello"),
        ])
        orch = OrchestratorAgent(llm=scripted, agents={"sub": sub})
        result = orch.run("say hello via sub")
        self.assertEqual(result, "Orchestrator done: [sub] hello")

    def test_sub_agent_receives_correct_task(self):
        received: list[str] = []

        class _CaptureLLM:
            def chat(self, messages, **kwargs):
                user_msg = next(
                    (m.content for m in reversed(messages) if m.role.value == "user"), ""
                )
                received.append(user_msg)
                return LLMResponse(content=f"done: {user_msg}")
            def stream(self, messages, **kwargs):
                yield ""

        sub = Agent(llm=_CaptureLLM())
        scripted = _ScriptedLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="1", name="ask_worker", arguments={"task": "specific task"})],
            ),
            LLMResponse(content="ok"),
        ])
        orch = OrchestratorAgent(llm=scripted, agents={"worker": sub})
        orch.run("do specific task")
        self.assertIn("specific task", received)

    def test_multiple_tool_calls_in_sequence(self):
        researcher = _echo_agent(prefix="R:")
        writer = _echo_agent(prefix="W:")
        scripted = _ScriptedLLM([
            LLMResponse(content="", tool_calls=[
                ToolCall(id="1", name="ask_researcher", arguments={"task": "facts"}),
            ]),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="2", name="ask_writer", arguments={"task": "R:facts"}),
            ]),
            LLMResponse(content="Final: W:R:facts"),
        ])
        orch = OrchestratorAgent(
            llm=scripted,
            agents={"researcher": researcher, "writer": writer},
        )
        result = orch.run("research and write")
        self.assertEqual(result, "Final: W:R:facts")


class TestOrchestratorAgentReset(unittest.TestCase):
    def test_reset_clears_orchestrator_memory(self):
        orch = OrchestratorAgent(
            llm=_ScriptedLLM([LLMResponse(content="hi")]),
            agents={"sub": _echo_agent()},
        )
        orch.run("hello")
        self.assertGreater(len(orch._inner.memory.messages), 0)
        orch.reset()
        self.assertEqual(len(orch._inner.memory.messages), 0)

    def test_reset_clears_sub_agent_memories(self):
        sub = _echo_agent()
        orch = OrchestratorAgent(
            llm=_ScriptedLLM([
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="1", name="ask_sub", arguments={"task": "hi"}),
                ]),
                LLMResponse(content="done"),
            ]),
            agents={"sub": sub},
        )
        orch.run("trigger sub")
        self.assertGreater(len(sub.memory.messages), 0)
        orch.reset()
        self.assertEqual(len(sub.memory.messages), 0)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipelineExecution(unittest.TestCase):
    def _transform_agent(self, fn) -> Agent:
        class _FnLLM:
            def __init__(self, f):
                self._f = f
            def chat(self, messages, **kwargs):
                user_msg = next(
                    (m.content for m in reversed(messages) if m.role.value == "user"), ""
                )
                return LLMResponse(content=self._f(user_msg))
            def stream(self, messages, **kwargs):
                yield ""
        return Agent(llm=_FnLLM(fn))

    def test_single_stage(self):
        pipeline = Pipeline(agents=[self._transform_agent(str.upper)])
        self.assertEqual(pipeline.run("hello"), "HELLO")

    def test_output_chains_through_stages(self):
        pipeline = Pipeline(agents=[
            self._transform_agent(lambda s: f"[A]{s}"),
            self._transform_agent(lambda s: f"[B]{s}"),
            self._transform_agent(lambda s: f"[C]{s}"),
        ])
        self.assertEqual(pipeline.run("x"), "[C][B][A]x")

    def test_named_stages_stored(self):
        a = self._transform_agent(str.upper)
        pipeline = Pipeline(agents=[("upper", a)])
        self.assertEqual(pipeline.stages[0][0], "upper")

    def test_unnamed_stages_get_default_names(self):
        pipeline = Pipeline(agents=[
            self._transform_agent(str.upper),
            self._transform_agent(str.lower),
        ])
        self.assertEqual(pipeline.stages[0][0], "stage_0")
        self.assertEqual(pipeline.stages[1][0], "stage_1")

    def test_mixed_named_and_unnamed(self):
        a = self._transform_agent(str.upper)
        b = self._transform_agent(str.lower)
        pipeline = Pipeline(agents=[("upper", a), b])
        self.assertEqual(pipeline.stages[0][0], "upper")
        self.assertEqual(pipeline.stages[1][0], "stage_1")


class TestPipelineReset(unittest.TestCase):
    def test_reset_clears_all_stage_memories(self):
        stages = [
            Agent(llm=_EchoLLM()),
            Agent(llm=_EchoLLM()),
        ]
        pipeline = Pipeline(agents=stages)
        pipeline.run("hello")
        for agent in stages:
            self.assertGreater(len(agent.memory.messages), 0)
        pipeline.reset()
        for agent in stages:
            self.assertEqual(len(agent.memory.messages), 0)


class TestOrchestratorSchema(unittest.TestCase):
    """Regression: the bound sub-agent must not leak into the tool schema."""

    def test_delegate_tool_exposes_only_task_param(self):
        orch = OrchestratorAgent(
            llm=_ScriptedLLM([LLMResponse(content="done")]),
            agents={"researcher": _echo_agent()},
        )
        schemas = orch._inner.tools.get_schemas()
        self.assertEqual(len(schemas), 1)
        props = schemas[0]["parameters"]["properties"]
        # Only 'task' must be exposed — no internal '_a'/'bound' binding leak.
        self.assertEqual(list(props.keys()), ["task"])
        self.assertEqual(schemas[0]["parameters"]["required"], ["task"])

    def test_each_delegate_binds_its_own_agent(self):
        # Guards against the late-binding closure bug: each tool must call
        # its own sub-agent, not the last one in the loop.
        orch = OrchestratorAgent(
            llm=_ScriptedLLM([LLMResponse(content="ignored")]),
            agents={"a": _echo_agent(prefix="A:"), "b": _echo_agent(prefix="B:")},
        )
        self.assertEqual(orch._inner.tools.get("ask_a").execute(task="q"), "A:q")
        self.assertEqual(orch._inner.tools.get("ask_b").execute(task="q"), "B:q")


if __name__ == "__main__":
    unittest.main()
