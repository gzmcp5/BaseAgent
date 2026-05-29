#!/usr/bin/env python3
"""Demonstrates the hook/middleware system.

Shows how to inject logging, redaction, and audit trails without
touching the core Agent or LLM code.

Usage:
    python examples/hooks_demo.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, HookRegistry, ToolRegistry, create_llm, load_dotenv

load_dotenv()


def build_hooks() -> HookRegistry:
    hooks = HookRegistry()

    @hooks.on("before_llm_call")
    def log_messages(messages):
        print(f"  [hook] before_llm_call — {len(messages)} message(s)")

    @hooks.on("after_llm_response")
    def log_response(response):
        preview = (response.content or "")[:60].replace("\n", " ")
        print(f"  [hook] after_llm_response — '{preview}...'")

    @hooks.on("before_tool_execute")
    def log_tool(tool_call):
        print(f"  [hook] before_tool_execute — {tool_call.name}({tool_call.arguments})")

    @hooks.on("after_tool_execute")
    def log_result(result, tool_call):
        print(f"  [hook] after_tool_execute  — {tool_call.name} → result={result!r}")

    return hooks


def main(provider: str = "claude") -> None:
    model = os.environ.get("LLM_MODEL") or None
    print(f"\n=== Hooks / Middleware Demo — provider: {provider}  model: {model or 'default'} ===\n")
    llm = create_llm(provider, model=model)

    tools = ToolRegistry()

    @tools.register(description="Multiply two numbers.")
    def multiply(a: int, b: int) -> int:
        return a * b

    agent = Agent(
        llm=llm,
        system_prompt="Use tools when asked to compute.",
        tools=tools,
        hooks=build_hooks(),
    )

    print("User : What is 7 × 8?")
    result = agent.run("What is 7 × 8?")
    print(f"Agent: {result}\n")


if __name__ == "__main__":
    import sys
    _provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    main(_provider)
