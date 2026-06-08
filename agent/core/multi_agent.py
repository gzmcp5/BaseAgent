"""여러 에이전트를 조합하는 두 가지 패턴 — 오케스트레이터와 파이프라인.

복잡한 일은 역할이 다른 여러 에이전트로 나누면 더 잘 풀린다. 이 모듈은 두 방식을 제공한다:
  - OrchestratorAgent(지휘자형): 각 서브에이전트를 'ask_{이름}' 도구로 노출하고, 지휘
    LLM이 상황을 보며 누구를 어떤 순서로 부를지 '동적으로' 결정한다.
  - Pipeline(직렬형): 정해진 순서대로 한 에이전트의 출력이 다음 에이전트의 입력이 된다
    (예: 분석가 → 작성자 → 검토자). 흐름이 고정된 작업에 적합.
"""
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
        self.sub_agents: dict[str, Agent] = dict(agents)  # 이름 → 서브에이전트 매핑.
        # 핵심 트릭: 지휘자도 결국 평범한 Agent다. 단지 도구가 '다른 에이전트 호출'일 뿐.
        # 서브에이전트들을 도구로 변환해(_build_tools) 일반 Agent에 넣으면 지휘가 성립한다.
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

        # 팩토리 함수로 각 서브에이전트를 '가둔다'. 이렇게 하면 두 가지를 동시에 해결한다:
        #  1) 에이전트를 함수 인자로 노출하지 않음(기본인자로 묶으면 JSON 스키마에 새어 나감).
        #  2) 반복문에서 흔한 늦은 바인딩 버그(모든 클로저가 마지막 agent를 가리키는 문제) 회피.
        def make_delegate(bound: Agent) -> Callable[[str], str]:
            def _delegate(task: str) -> str:
                return bound.run(task)  # 위임: 받은 작업을 해당 서브에이전트에게 실행시킴.

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
        # 지휘 LLM이 알아서 서브에이전트들을 도구처럼 호출하며 일을 끝낸다.
        return self._inner.run(user_input)

    def reset(self) -> None:
        """Clear orchestrator memory and all sub-agent memories."""
        # 지휘자 본인뿐 아니라 모든 서브에이전트의 대화 기록도 함께 초기화한다.
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
        # 각 단계를 (이름, 에이전트) 형태로 정규화해 저장한다.
        self.stages: list[tuple[str, Agent]] = []
        for i, entry in enumerate(agents):
            if isinstance(entry, tuple):
                name, agent = entry            # 이름이 함께 주어진 경우.
            else:
                name = f"stage_{i}"            # 이름 없이 에이전트만 주면 자동 이름 부여.
                agent = entry
            self.stages.append((name, agent))

    def run(self, user_input: str) -> str:
        """Pass *user_input* through each stage in order."""
        # 첫 입력에서 시작해, 각 단계의 출력을 다음 단계의 입력으로 계속 넘긴다(릴레이).
        result = user_input
        for _name, agent in self.stages:
            result = agent.run(result)
        return result  # 마지막 단계의 출력이 최종 결과.

    def reset(self) -> None:
        """Clear every stage agent's memory."""
        for _name, agent in self.stages:
            agent.reset()
