#!/usr/bin/env python3
"""Advanced multi-agent demo: OrchestratorAgent + Pipeline combined workflow.

Scenario: AI-assisted code review pipeline
  Stage 1 (Pipeline)  — Analyst agent summarises the requirements
  Stage 2 (Pipeline)  — Coder agent writes Python code
  Stage 3 (Orchestrator) — Review supervisor delegates to a style-checker and a
                            security-checker sub-agent, then produces a final verdict

Usage:
    python examples/multi_agent_advanced_demo.py [provider]

Examples:
    python examples/multi_agent_advanced_demo.py
    python examples/multi_agent_advanced_demo.py openai
    LLM_MODEL=llama3.1:latest python examples/multi_agent_advanced_demo.py ollama
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, OrchestratorAgent, Pipeline, create_llm, load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _llm(provider: str, model: str | None = None):
    return create_llm(provider, model=model)


def separator(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


# ---------------------------------------------------------------------------
# Demo 1: Pipeline  (analyse → code → document)
# ---------------------------------------------------------------------------

def demo_pipeline(provider: str, model: str | None) -> str:
    separator("Demo 1 · Pipeline: analyse → code → document")

    analyst = Agent(
        llm=_llm(provider, model),
        system_prompt=(
            "You are a requirements analyst. "
            "Given a task description, output a concise numbered list of "
            "technical requirements (max 5 items). "
            "Be brief — one short sentence per item."
        ),
    )

    coder = Agent(
        llm=_llm(provider, model),
        system_prompt=(
            "You are a Python developer. "
            "Given a numbered list of requirements, write a clean, minimal "
            "Python function that satisfies them. "
            "Include a brief docstring and return the code only."
        ),
    )

    documenter = Agent(
        llm=_llm(provider, model),
        system_prompt=(
            "You are a technical writer. "
            "Given Python source code, add a detailed docstring (Args, Returns, "
            "Example sections) and inline comments where helpful. "
            "Return the complete annotated code."
        ),
    )

    pipeline = Pipeline(agents=[
        ("analyst",    analyst),
        ("coder",      coder),
        ("documenter", documenter),
    ])

    task = (
        "Write a function that takes a list of integers and returns "
        "a new list with duplicates removed, preserving the original order."
    )
    print(f"Task: {task}\n")
    result = pipeline.run(task)
    print(result)
    return result


# ---------------------------------------------------------------------------
# Demo 2: OrchestratorAgent  (style-checker + security-checker → verdict)
# ---------------------------------------------------------------------------

def demo_orchestrator(provider: str, model: str | None, code: str) -> str:
    separator("Demo 2 · OrchestratorAgent: style-check + security-check → verdict")

    style_checker = Agent(
        llm=_llm(provider, model),
        system_prompt=(
            "You are a Python style reviewer. "
            "Review the code for PEP 8 compliance, naming conventions, and "
            "code clarity. Output: a bullet list of findings (PASS or issues)."
        ),
    )

    security_checker = Agent(
        llm=_llm(provider, model),
        system_prompt=(
            "You are a security-focused code reviewer. "
            "Check for common vulnerabilities: injection risks, unsafe eval/exec, "
            "unbounded loops, resource leaks. "
            "Output: a bullet list of findings (PASS or issues)."
        ),
    )

    supervisor = OrchestratorAgent(
        llm=_llm(provider, model),
        agents={
            "style_checker":    style_checker,
            "security_checker": security_checker,
        },
        system_prompt=(
            "You are a senior code review supervisor. "
            "Given Python code to review:\n"
            "1. Ask the style_checker to review it.\n"
            "2. Ask the security_checker to review it.\n"
            "3. Combine both reports into a final verdict: "
            "APPROVED, APPROVED_WITH_NOTES, or NEEDS_REVISION. "
            "Include a short summary of the key points."
        ),
        max_iterations=10,
    )

    print(f"Reviewing code ({len(code)} chars)...\n")
    verdict = supervisor.run(f"Please review the following Python code:\n\n{code}")
    print(verdict)
    return verdict


# ---------------------------------------------------------------------------
# Demo 3: Pipeline wrapping an OrchestratorAgent as a stage
# ---------------------------------------------------------------------------

def demo_combined(provider: str, model: str | None) -> None:
    separator("Demo 3 · Combined: Pipeline stage feeds into OrchestratorAgent")

    # Stage 1: turn a one-line idea into a spec
    specwriter = Agent(
        llm=_llm(provider, model),
        system_prompt=(
            "You are a spec writer. Turn the user's one-liner into a 3-point "
            "numbered technical spec. Be concise."
        ),
    )

    # Stage 2: generate code from spec
    coder = Agent(
        llm=_llm(provider, model),
        system_prompt=(
            "You are a Python developer. "
            "Write a clean Python function matching the spec. Code only."
        ),
    )

    # Stage 3: orchestrator performs dual review
    style_agent = Agent(
        llm=_llm(provider, model),
        system_prompt="Review Python code for style. List findings briefly.",
    )
    perf_agent = Agent(
        llm=_llm(provider, model),
        system_prompt="Review Python code for performance issues. List findings briefly.",
    )

    review_orchestrator = OrchestratorAgent(
        llm=_llm(provider, model),
        agents={"style": style_agent, "performance": perf_agent},
        system_prompt=(
            "You are a review coordinator. "
            "Ask style and performance agents to review the code, "
            "then give a one-paragraph final summary."
        ),
        max_iterations=8,
    )

    # Wrap the orchestrator as a plain callable stage
    class _OrchestratorStage(Agent):
        """Thin wrapper so OrchestratorAgent fits in a Pipeline."""
        def __init__(self, orch: OrchestratorAgent) -> None:
            self._orch = orch

        def run(self, user_input: str, **_) -> str:  # type: ignore[override]
            return self._orch.run(user_input)

        def reset(self) -> None:
            self._orch.reset()

    pipeline = Pipeline(agents=[
        ("spec",   specwriter),
        ("code",   coder),
        ("review", _OrchestratorStage(review_orchestrator)),
    ])

    idea = "a function that merges two sorted lists into one sorted list"
    print(f"Idea: {idea}\n")
    result = pipeline.run(idea)
    print(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    model = os.environ.get("LLM_MODEL") or None

    # Demo 1: pure Pipeline
    code = demo_pipeline(provider, model)

    # Demo 2: pure OrchestratorAgent (reviews the code from Demo 1)
    demo_orchestrator(provider, model, code)

    # Demo 3: Pipeline with an OrchestratorAgent stage
    demo_combined(provider, model)
