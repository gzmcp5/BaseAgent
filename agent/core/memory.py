from __future__ import annotations
from .message import Message, Role


class ConversationMemory:
    """Stores and manages conversation history."""

    def __init__(self, max_messages: int = 100, system_prompt: str = "") -> None:
        self.messages: list[Message] = []
        self.max_messages = max_messages
        self.system_prompt = system_prompt

    def add(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            # Keep the most recent messages; preserve leading assistant context if needed
            self.messages = self.messages[-self.max_messages :]

    def get_messages(self) -> list[Message]:
        result: list[Message] = []
        if self.system_prompt:
            result.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)
