#!/usr/bin/env python3
"""Demonstrates streaming output with the base agent.

Usage:
    python examples/streaming_demo.py                  # defaults to Claude
    python examples/streaming_demo.py openai
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, create_llm, load_dotenv

load_dotenv()


def main(provider: str = "claude") -> None:
    print(f"\n=== Streaming Demo — provider: {provider} ===\n")
    llm = create_llm(provider)
    agent = Agent(llm=llm, system_prompt="You are a helpful assistant.")

    prompt = "Write a three-sentence story about a robot learning to paint."
    print(f"User : {prompt}\nAgent: ", end="", flush=True)

    for chunk in agent.stream(prompt):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    main(provider)
