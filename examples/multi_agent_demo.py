#!/usr/bin/env python3
"""Demonstrates multi-agent architecture: OrchestratorAgent and Pipeline.

Two demos:
  orchestrator  — supervisor LLM delegates to researcher + writer sub-agents
  pipeline      — sequential analyst → writer → reviewer chain

Usage:
    python examples/multi_agent_demo.py [orchestrator|pipeline|both] [provider]

Examples:
    python examples/multi_agent_demo.py
    python examples/multi_agent_demo.py pipeline openai
    python examples/multi_agent_demo.py orchestrator claude
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, OrchestratorAgent, Pipeline, create_llm, load_dotenv

load_dotenv()


def demo_orchestrator(provider: str) -> None:
    model = os.environ.get("LLM_MODEL") or None
    print(f"\n=== OrchestratorAgent — {provider} ===\n")

    researcher = Agent(
        llm=create_llm(provider, model=model),
        system_prompt="You are a research assistant. Return 3 concise bullet points of facts.",
    )
    writer = Agent(
        llm=create_llm(provider, model=model),
        system_prompt="You are a technical writer. Turn bullet points into a short paragraph.",
    )

    orchestrator = OrchestratorAgent(
        llm=create_llm(provider, model=model),
        agents={"researcher": researcher, "writer": writer},
        system_prompt=(
            "You coordinate a researcher and writer to answer questions. "
            "Step 1: ask the researcher for facts. "
            "Step 2: ask the writer to turn those facts into a paragraph. "
            "Step 3: return the final paragraph."
        ),
    )

    result = orchestrator.run("What are large language models?")
    print(result)


def demo_pipeline(provider: str) -> None:
    model = os.environ.get("LLM_MODEL") or None
    print(f"\n=== Pipeline — {provider} ===\n")

    analyst = Agent(
        llm=create_llm(provider, model=model),
        system_prompt="Analyse the topic and list exactly 3 key points as bullet points. Be brief.",
    )
    writer = Agent(
        llm=create_llm(provider, model=model),
        system_prompt="Expand the bullet points into a coherent short paragraph (2-3 sentences).",
    )
    reviewer = Agent(
        llm=create_llm(provider, model=model),
        system_prompt="Review the paragraph for clarity and return a polished final version.",
    )

    pipeline = Pipeline(agents=[
        ("analyst", analyst),
        ("writer", writer),
        ("reviewer", reviewer),
    ])

    result = pipeline.run("The impact of AI on software development")
    print(result)


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "both"
    provider = args[1] if len(args) > 1 else "claude"

    if mode in ("orchestrator", "both"):
        demo_orchestrator(provider)
    if mode in ("pipeline", "both"):
        demo_pipeline(provider)
