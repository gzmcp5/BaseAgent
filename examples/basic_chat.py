#!/usr/bin/env python3
"""Demonstrates basic multi-turn conversation with memory.

Usage:
    python examples/basic_chat.py                  # defaults to Claude
    python examples/basic_chat.py openai
    python examples/basic_chat.py google
    python examples/basic_chat.py ollama
    python examples/basic_chat.py openrouter
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, create_llm, load_dotenv

load_dotenv()


def main(provider: str = "claude") -> None:
    print(f"\n=== Basic Chat — provider: {provider} ===\n")
    llm = create_llm(provider)
    agent = Agent(llm=llm, system_prompt="You are a concise, helpful assistant.")

    turns = [
        "What is the capital of France?",
        "What was my previous question?",  # tests conversation memory
    ]

    for question in turns:
        print(f"User : {question}")
        answer = agent.run(question)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    main(provider)
