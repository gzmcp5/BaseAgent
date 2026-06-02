#!/usr/bin/env python3
"""Demonstrates long-term memory with SQLite-backed PersistentMemory.

Two modes:
  new      — start a fresh session, chat a few turns, then print the session ID
  resume   — load a previous session by ID and continue the conversation

Usage:
    python examples/persistent_memory_demo.py new [provider]
    python examples/persistent_memory_demo.py resume <session_id> [provider]
    python examples/persistent_memory_demo.py list

Examples:
    python examples/persistent_memory_demo.py new
    python examples/persistent_memory_demo.py new openai
    python examples/persistent_memory_demo.py list
    python examples/persistent_memory_demo.py resume abc123 claude
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, PersistentMemory, create_llm, load_dotenv

load_dotenv()

DB_PATH = "agent_memory.db"


def demo_new(provider: str) -> None:
    model = os.environ.get("LLM_MODEL") or None
    print(f"\n=== New Session — provider: {provider}  model: {model or 'default'} ===\n")

    with PersistentMemory(DB_PATH, system_prompt="You are a concise, helpful assistant.") as mem:
        agent = Agent(llm=create_llm(provider, model=model), memory=mem)

        turns = [
            "My name is Alex and I'm learning about AI agents.",
            "What did I just tell you about myself?",
        ]
        for question in turns:
            print(f"User : {question}")
            print(f"Agent: {agent.run(question)}\n")

        print(f"Session saved. Resume it with:\n")
        print(f"  python examples/persistent_memory_demo.py resume {mem.session_id}\n")


def demo_resume(session_id: str, provider: str) -> None:
    model = os.environ.get("LLM_MODEL") or None
    print(f"\n=== Resuming Session {session_id[:8]}… — provider: {provider} ===\n")

    with PersistentMemory(DB_PATH, session_id=session_id) as mem:
        print(f"Loaded {len(mem.messages)} messages from previous session.\n")
        agent = Agent(llm=create_llm(provider, model=model), memory=mem)

        question = "Do you still remember what I told you earlier?"
        print(f"User : {question}")
        print(f"Agent: {agent.run(question)}\n")


def demo_list() -> None:
    sessions = PersistentMemory.list_sessions(DB_PATH)
    if not sessions:
        print("No sessions found in", DB_PATH)
        return

    print(f"\n{'ID':<38} {'Messages':>8}  {'Last updated'}")
    print("-" * 70)
    for s in sessions:
        print(f"{s['id']:<38} {s['message_count']:>8}  {s['updated_at'][:19]}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "new"

    if mode == "list":
        demo_list()
    elif mode == "resume":
        if len(args) < 2:
            print("Usage: persistent_memory_demo.py resume <session_id> [provider]")
            sys.exit(1)
        demo_resume(session_id=args[1], provider=args[2] if len(args) > 2 else "claude")
    else:
        demo_new(provider=args[1] if len(args) > 1 else "claude")
