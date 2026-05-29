#!/usr/bin/env python3
"""Demonstrates AsyncAgent with concurrent tool execution.

Usage:
    python examples/async_demo.py
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import AsyncAgent, ToolRegistry, create_llm, load_dotenv

load_dotenv()


def build_tools() -> ToolRegistry:
    tools = ToolRegistry()

    @tools.register(description="Fetch weather for a city (simulated 0.5 s latency).")
    def get_weather(city: str) -> str:
        time.sleep(0.5)  # simulate I/O
        return f"Sunny, 22°C in {city}"

    @tools.register(description="Look up the population of a city (simulated 0.5 s latency).")
    def get_population(city: str) -> str:
        time.sleep(0.5)
        populations = {"Tokyo": "13.96M", "Paris": "2.16M", "Seoul": "9.77M"}
        return populations.get(city, "Unknown")

    return tools


async def main(provider: str = "claude") -> None:
    model = os.environ.get("LLM_MODEL") or None
    print(f"\n=== AsyncAgent Demo — provider: {provider}  model: {model or 'default'} ===\n")
    llm = create_llm(provider, model=model)
    tools = build_tools()

    async with AsyncAgent(llm=llm, tools=tools, system_prompt="Use tools for city info.") as agent:
        query = "What's the weather and population of Tokyo?"
        print(f"User : {query}")

        start = time.perf_counter()
        response = await agent.run(query)
        elapsed = time.perf_counter() - start

        print(f"Agent: {response}")
        print(f"\n(Two 0.5 s tool calls executed concurrently in {elapsed:.2f} s)\n")


if __name__ == "__main__":
    _provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    asyncio.run(main(_provider))
