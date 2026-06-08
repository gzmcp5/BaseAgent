from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Callable, Optional

from .hooks import HookRegistry
from .memory import ConversationMemory
from .message import Message, Role, ToolCall
from .tool import ToolRegistry
from .tool_selector import ToolSelector
from ..llm.base import BaseLLM

# Human-in-the-loop approval callback: returns True to allow, False to deny.
ApprovalCallback = Callable[[ToolCall], bool]


class AsyncAgent:
    """Async version of Agent — concurrent tool execution via asyncio.

    LLM HTTP calls are dispatched to a thread-pool executor (because the
    underlying providers use stdlib urllib, which is blocking).  Tool calls
    in a single turn are executed *concurrently* via asyncio.gather, which
    improves throughput when tools do I/O work.

    Hook contract: all hooks registered via HookRegistry run on the event loop
    thread, not in the executor.  Hooks must therefore be non-blocking and
    free of thread-unsafe side effects.  Blocking I/O inside a hook will stall
    the event loop.

    Usage::

        async with AsyncAgent(llm=llm, system_prompt="...", tools=tools) as agent:
            response = await agent.run("Hello!")

            # Streaming
            async for chunk in agent.stream("Tell me a story"):
                print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str = "",
        tools: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        hooks: Optional[HookRegistry] = None,
        max_tool_iterations: int = 10,
        max_workers: int = 8,
        tool_selector: Optional[ToolSelector] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory(system_prompt=system_prompt)
        self.hooks = hooks or HookRegistry()
        self.max_tool_iterations = max_tool_iterations
        self._max_workers = max_workers
        # RAG-based tool pruning (optional).
        self.tool_selector = tool_selector
        # Human-in-the-loop gate for tools marked requires_approval (optional).
        self.approval_callback = approval_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_input: str, **llm_kwargs: Any) -> str:
        """Run a single user turn asynchronously; returns the final response."""
        self.memory.add(Message(role=Role.USER, content=user_input))

        for _ in range(self.max_tool_iterations):
            messages = self.memory.get_messages()
            schemas = self._schemas_for(user_input)

            _m = self.hooks.run("before_llm_call", messages)
            if _m is not None:
                messages = _m

            response = await self._run_blocking(
                lambda msgs=messages, sc=schemas: self.llm.chat(msgs, tools=sc, **llm_kwargs),
            )

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

            # Execute all tool calls for this turn concurrently
            results = await asyncio.gather(
                *[self._execute_tool_async(tc) for tc in response.tool_calls]
            )

            for tc, result in zip(response.tool_calls, results):
                self.memory.add(
                    Message(
                        role=Role.TOOL,
                        content=str(result),
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )

        return "Max tool iterations reached without a final response."

    async def stream(self, user_input: str, **llm_kwargs: Any) -> AsyncIterator[str]:
        """Stream text chunks asynchronously.

        Yields each text chunk as it arrives.  Tool calls are not supported
        in streaming mode — use run() for tool-enabled turns.
        Raises any exception thrown by the LLM provider during streaming.
        Memory is only updated on clean completion (no partial saves on error).
        """
        self.memory.add(Message(role=Role.USER, content=user_input))
        messages = self.memory.get_messages()
        _m = self.hooks.run("before_llm_call", messages)
        if _m is not None:
            messages = _m

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        exc_box: list[BaseException] = []

        def _produce() -> None:
            try:
                for chunk in self.llm.stream(messages, **llm_kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                exc_box.append(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        fut = asyncio.create_task(self._run_blocking(_produce))

        full_response = ""
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            full_response += chunk
            yield chunk

        await fut  # ensure thread has exited; propagates executor-level errors

        if exc_box:
            raise exc_box[0]  # surface any LLM stream error

        # Only commit to memory on clean completion
        self.memory.add(Message(role=Role.ASSISTANT, content=full_response))

    def reset(self) -> None:
        self.memory.clear()

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncAgent":
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schemas_for(self, user_input: str) -> Optional[list[dict]]:
        """Tool schemas to advertise to the LLM (RAG-pruned when a selector is set)."""
        if self.tool_selector is not None:
            return self.tool_selector.select_schemas(user_input) or None
        return self.tools.get_schemas() or None

    def _approved(self, tool: Any, tool_call: ToolCall) -> bool:
        """Human-in-the-loop gate (fail-closed) for requires_approval tools."""
        if not tool.requires_approval:
            return True
        if self.approval_callback is None:
            return False
        return bool(self.approval_callback(tool_call))

    async def _execute_tool_async(self, tool_call: ToolCall) -> Any:
        tool_call = self.hooks.run("before_tool_execute", tool_call) or tool_call

        tool = self.tools.get(tool_call.name)
        if tool is None:
            result: Any = f"Error: tool '{tool_call.name}' is not registered."
        elif not self._approved(tool, tool_call):
            result = f"Tool '{tool_call.name}' execution was denied (awaiting user approval)."
        else:
            try:
                result = await self._run_blocking(
                    lambda tc=tool_call: tool.execute(**tc.arguments),
                )
            except Exception as exc:
                result = f"Error executing '{tool_call.name}': {exc}"

        # Use explicit None check — `or` would swallow falsy results like 0 or "".
        _r = self.hooks.run("after_tool_execute", result, tool_call)
        if _r is not None:
            result = _r
        return result

    async def _run_blocking(self, fn: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            return await loop.run_in_executor(executor, fn)
        finally:
            executor.shutdown(wait=True)
