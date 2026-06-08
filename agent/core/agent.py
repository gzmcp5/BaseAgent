from __future__ import annotations
from typing import Any, Callable, Iterator, Optional

from .hooks import HookRegistry
from .memory import ConversationMemory
from .message import Message, Role, ToolCall
from .tool import ToolRegistry
from .tool_selector import ToolSelector
from ..llm.base import BaseLLM

# Human-in-the-loop approval callback: receives the pending ToolCall and returns
# True to allow execution, False to deny it.
ApprovalCallback = Callable[[ToolCall], bool]


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
        tool_selector: Optional[ToolSelector] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory(system_prompt=system_prompt)
        self.hooks = hooks or HookRegistry()
        self.max_tool_iterations = max_tool_iterations
        # Optional RAG-based tool pruning: when set, only the tools most
        # relevant to the user query are sent to the LLM.
        self.tool_selector = tool_selector
        # Optional human-in-the-loop gate for tools marked requires_approval.
        self.approval_callback = approval_callback

    def run(self, user_input: str, **llm_kwargs: Any) -> str:
        """Run a single user turn; returns the final assistant response."""
        self.memory.add(Message(role=Role.USER, content=user_input))

        for _ in range(self.max_tool_iterations):
            messages = self.memory.get_messages()
            schemas = self._schemas_for(user_input)

            _m = self.hooks.run("before_llm_call", messages)
            if _m is not None:
                messages = _m

            response = self.llm.chat(messages, tools=schemas, **llm_kwargs)

            _resp = self.hooks.run("after_llm_response", response)
            if _resp is not None:
                response = _resp

            if not response.tool_calls:
                content = response.content or ""
                self.memory.add(Message(role=Role.ASSISTANT, content=content))
                return content

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
        _m = self.hooks.run("before_llm_call", messages)
        if _m is not None:
            messages = _m

        full_response = ""
        for chunk in self.llm.stream(messages, **llm_kwargs):
            full_response += chunk
            yield chunk

        self.memory.add(Message(role=Role.ASSISTANT, content=full_response))

    def _schemas_for(self, user_input: str) -> Optional[list[dict]]:
        """Tool schemas to advertise to the LLM for this turn.

        With a tool_selector, only the tools most relevant to *user_input*
        are sent (RAG pruning); otherwise all registered tools are sent.
        """
        if self.tool_selector is not None:
            return self.tool_selector.select_schemas(user_input) or None
        return self.tools.get_schemas() or None

    def _approved(self, tool: "Any", tool_call: ToolCall) -> bool:
        """Human-in-the-loop gate for tools marked requires_approval.

        Fail-closed: an approval-required tool is denied unless a callback is
        configured and explicitly returns True.
        """
        if not tool.requires_approval:
            return True
        if self.approval_callback is None:
            return False
        return bool(self.approval_callback(tool_call))

    def _execute_tool(self, tool_call: ToolCall) -> Any:
        # Middleware: may rewrite the tool call before execution
        tool_call = self.hooks.run("before_tool_execute", tool_call) or tool_call

        tool = self.tools.get(tool_call.name)
        if tool is None:
            result: Any = f"Error: tool '{tool_call.name}' is not registered."
        elif not self._approved(tool, tool_call):
            result = f"Tool '{tool_call.name}' execution was denied (awaiting user approval)."
        else:
            try:
                result = tool.execute(**tool_call.arguments)
            except Exception as exc:
                result = f"Error executing '{tool_call.name}': {exc}"

        # Use explicit None check — `or` would swallow falsy results like 0 or "".
        _r = self.hooks.run("after_tool_execute", result, tool_call)
        if _r is not None:
            result = _r
        return result

    def reset(self) -> None:
        """Clear conversation history while keeping the system prompt."""
        self.memory.clear()
