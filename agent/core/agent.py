from __future__ import annotations
from typing import Any, Optional

from .memory import ConversationMemory
from .message import Message, Role, ToolCall
from .tool import ToolRegistry
from ..llm.base import BaseLLM


class Agent:
    """Core agent that orchestrates LLM calls, tool execution, and memory."""

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str = "",
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        max_tool_iterations: int = 10,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory(system_prompt=system_prompt)
        self.max_tool_iterations = max_tool_iterations

    def run(self, user_input: str, **llm_kwargs: Any) -> str:
        """Run a single user turn; returns the final assistant response."""
        self.memory.add(Message(role=Role.USER, content=user_input))

        for _ in range(self.max_tool_iterations):
            messages = self.memory.get_messages()
            schemas = self.tools.get_schemas() or None

            response = self.llm.chat(messages, tools=schemas, **llm_kwargs)

            if not response.tool_calls:
                self.memory.add(Message(role=Role.ASSISTANT, content=response.content))
                return response.content

            # Assistant message with pending tool calls
            self.memory.add(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                self.memory.add(
                    Message(
                        role=Role.TOOL,
                        content=str(result),
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )

        return "Max tool iterations reached without a final response."

    def _execute_tool(self, tool_call: ToolCall) -> Any:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return f"Error: tool '{tool_call.name}' is not registered."
        try:
            return tool.execute(**tool_call.arguments)
        except Exception as exc:
            return f"Error executing '{tool_call.name}': {exc}"

    def reset(self) -> None:
        """Clear conversation history while keeping the system prompt."""
        self.memory.clear()
