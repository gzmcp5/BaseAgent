#!/usr/bin/env python3
"""Base Agent — interactive CLI entry point.

Usage:
    python main.py                          # Claude (default)
    LLM_PROVIDER=openai python main.py
    LLM_PROVIDER=ollama LLM_MODEL=llama3.2 python main.py

Commands inside the REPL:
    /reset      Clear conversation history
    /provider   Show current LLM provider and model
    /quit       Exit
"""
import os
import sys
from datetime import datetime

from agent import Agent, ToolRegistry, create_llm, load_dotenv
from agent.utils.config import Config

load_dotenv()


def build_default_tools() -> ToolRegistry:
    tools = ToolRegistry()

    @tools.register(description="Return the current date and time.")
    def get_current_time() -> str:
        return datetime.now().isoformat()

    return tools


def main() -> None:
    config = Config.from_env()
    provider = config.get("llm_provider", "claude")
    model = config.get("llm_model")

    try:
        llm = create_llm(provider, model=model)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    tools = build_default_tools()
    agent = Agent(
        llm=llm,
        system_prompt="You are a helpful AI assistant.",
        tools=tools,
    )

    model_label = model or llm.model
    print(f"Base Agent  |  provider={provider}  model={model_label}")
    print("Commands: /reset  /provider  /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lstrip("/").lower()
            if cmd == "quit":
                print("Goodbye.")
                break
            elif cmd == "reset":
                agent.reset()
                print("Conversation cleared.\n")
            elif cmd == "provider":
                print(f"provider={provider}  model={llm.model}\n")
            else:
                print(f"Unknown command: {user_input}\n")
            continue

        try:
            response = agent.run(user_input)
            print(f"Agent: {response}\n")
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
