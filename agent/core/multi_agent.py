from __future__ import annotations

from typing import Callable, Optional, Union

from .agent import Agent
from .hooks import HookRegistry
from .tool import ToolRegistry


class OrchestratorAgent:
    """Supervisor-style multi-agent orchestrator.

    Each sub-agent is exposed to the orchestrator LLM as a tool named
    ``ask_{name}``.  The LLM decides which sub-agents to call and in what
    order; sub-agents maintain independent conversation memory.

    Usage::

        researcher = Agent(llm=llm, system_prompt="You research topics.")
        coder = Agent(llm=llm, system_prompt="You write Python code.")

        orch = OrchestratorAgent(
            llm=llm,
            agents={"researcher": researcher, "coder": coder},
            system_prompt=(
                "Coordinate researcher and coder to complete tasks. "
                "First gather facts, then produce code."
            ),
        )
        result = orch.run("Build a web scraper for news headlines.")
    """

    def __init__(
        self,
        llm,
        agents: dict[str, Agent],
        system_prompt: str = "",
        hooks: Optional[HookRegistry] = None,
        max_iterations: int = 20,
    ) -> None:
        self.sub_agents: dict[str, Agent] = dict(agents)
        self._inner = Agent(
            llm=llm,
            system_prompt=system_prompt,
            tools=self._build_tools(self.sub_agents),
            hooks=hooks or HookRegistry(),
            max_tool_iterations=max_iterations,
        )

    @staticmethod
    def _build_tools(agents: dict[str, Agent]) -> ToolRegistry:
        registry = ToolRegistry()

        # Capture each sub-agent via an enclosing factory so the bound agent
        # is NOT exposed as a tool parameter (a default-arg binding would leak
        # into the generated JSON schema) while still avoiding the classic
        # late-binding closure bug.
        def make_delegate(bound: Agent) -> Callable[[str], str]:
            def _delegate(task: str) -> str:
                return bound.run(task)

            return _delegate

        for agent_name, agent in agents.items():
            registry.register(
                make_delegate(agent),
                name=f"ask_{agent_name}",
                description=(
                    f"Delegate a task to the '{agent_name}' sub-agent "
                    f"and return its response."
                ),
            )
        return registry

    def run(self, user_input: str) -> str:
        """Run a single user turn through the orchestrator."""
        return self._inner.run(user_input)

    def reset(self) -> None:
        """Clear orchestrator memory and all sub-agent memories."""
        self._inner.reset()
        for agent in self.sub_agents.values():
            agent.reset()


class Pipeline:
    """Sequential multi-agent pipeline.

    Each agent's output becomes the next agent's input.
    Useful for multi-step workflows: analyse → draft → review.

    Usage::

        pipeline = Pipeline(agents=[analyst, writer, reviewer])
        result = pipeline.run("Summarise the latest AI trends.")
        # analyst.run(input) → writer.run(analyst_output) → reviewer.run(writer_output)

    Named stages (for inspection via ``pipeline.stages``)::

        pipeline = Pipeline(agents=[
            ("analyst", analyst),
            ("writer", writer),
        ])
    """

    def __init__(
        self,
        agents: list[Union[Agent, tuple[str, Agent]]],
    ) -> None:
        self.stages: list[tuple[str, Agent]] = []
        for i, entry in enumerate(agents):
            if isinstance(entry, tuple):
                name, agent = entry
            else:
                name = f"stage_{i}"
                agent = entry
            self.stages.append((name, agent))

    def run(self, user_input: str) -> str:
        """Pass *user_input* through each stage in order."""
        result = user_input
        for _name, agent in self.stages:
            result = agent.run(result)
        return result

    def reset(self) -> None:
        """Clear every stage agent's memory."""
        for _name, agent in self.stages:
            agent.reset()
