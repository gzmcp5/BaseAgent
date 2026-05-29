from __future__ import annotations
from typing import Any, Iterator, Optional

from .hooks import HookRegistry
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
        hooks: Optional[HookRegistry] = None,
        max_tool_iterations: int = 10,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory(system_prompt=system_prompt)
        self.hooks = hooks or HookRegistry()
        self.max_tool_iterations = max_tool_iterations

    def run(self, user_input: str, **llm_kwargs: Any) -> str:
        """Run a single user turn; returns the final assistant response."""
        self.memory.add(Message(role=Role.USER, content=user_input))

        for _ in range(self.max_tool_iterations):
            messages = self.memory.get_messages()
            schemas = self.tools.get_schemas() or None

            # Middleware: may rewrite or inspect messages before the LLM call
            messages = self.hooks.run("before_llm_call", messages) or messages

            response = self.llm.chat(messages, tools=schemas, **llm_kwargs)

            # Middleware: may rewrite the LLM response
            response = self.hooks.run("after_llm_response", response) or response

            if not response.tool_calls:
                self.memory.add(Message(role=Role.ASSISTANT, content=response.content))
                return response.content

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

    def stream(self, user_input: str, **llm_kwargs: Any) -> Iterator[str]:
        """Stream text chunks for a user turn.

        Tool calls are not supported in streaming mode — use run() for that.
        Memory is updated with the full assembled response when the stream ends.
        """
        self.memory.add(Message(role=Role.USER, content=user_input))
        messages = self.memory.get_messages()
        messages = self.hooks.run("before_llm_call", messages) or messages

        full_response = ""
        for chunk in self.llm.stream(messages, **llm_kwargs):
            full_response += chunk
            yield chunk

        self.memory.add(Message(role=Role.ASSISTANT, content=full_response))

    def _execute_tool(self, tool_call: ToolCall) -> Any:
        # Middleware: may rewrite the tool call before execution
        tool_call = self.hooks.run("before_tool_execute", tool_call) or tool_call

        tool = self.tools.get(tool_call.name)
        if tool is None:
            result: Any = f"Error: tool '{tool_call.name}' is not registered."
        else:
            try:
                result = tool.execute(**tool_call.arguments)
            except Exception as exc:
                result = f"Error executing '{tool_call.name}': {exc}"

        # Middleware: result is threaded first; tool_call is read-only context.
        result = self.hooks.run("after_tool_execute", result, tool_call) or result
        return result

    def reset(self) -> None:
        """Clear conversation history while keeping the system prompt."""
        self.memory.clear()
