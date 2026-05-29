#!/usr/bin/env python3
"""Demonstrates tool/function-calling with the base agent.

Usage:
    python examples/tool_use_demo.py                  # defaults to Claude
    python examples/tool_use_demo.py openai
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, ToolRegistry, create_llm, load_dotenv

load_dotenv()


def build_tools() -> ToolRegistry:
    tools = ToolRegistry()

    @tools.register(description="Add two integers and return their sum.")
    def add(a: int, b: int) -> int:
        return a + b

    @tools.register(description="Return the current date and time as an ISO-8601 string.")
    def get_current_time() -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    @tools.register(description="Convert a temperature from Celsius to Fahrenheit.")
    def celsius_to_fahrenheit(celsius: float) -> float:
        return celsius * 9 / 5 + 32

    return tools


def main(provider: str = "claude") -> None:
    model = os.environ.get("LLM_MODEL") or None
    print(f"\n=== Tool Use Demo — provider: {provider}  model: {model or 'default'} ===\n")
    llm = create_llm(provider, model=model)
    agent = Agent(
        llm=llm,
        system_prompt="You are a helpful assistant. Use the available tools when needed.",
        tools=build_tools(),
    )

    queries = [
        "What is 237 + 489?",
        "What time is it right now?",
        "Convert 37 degrees Celsius to Fahrenheit.",
    ]

    for query in queries:
        print(f"User : {query}")
        answer = agent.run(query)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    main(provider)
