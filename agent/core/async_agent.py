from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Optional

from .hooks import HookRegistry
from .memory import ConversationMemory
from .message import Message, Role, ToolCall
from .tool import ToolRegistry
from ..llm.base import BaseLLM


class AsyncAgent:
    """Async version of Agent — concurrent tool execution via asyncio.

    LLM HTTP calls are dispatched to a thread-pool executor (because the
    underlying providers use stdlib urllib, which is blocking).  Tool calls
    in a single turn are executed *concurrently* via asyncio.gather, which
    improves throughput when tools do I/O work.

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
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory(system_prompt=system_prompt)
        self.hooks = hooks or HookRegistry()
        self.max_tool_iterations = max_tool_iterations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_input: str, **llm_kwargs: Any) -> str:
        """Run a single user turn asynchronously; returns the final response."""
        self.memory.add(Message(role=Role.USER, content=user_input))
        loop = asyncio.get_running_loop()

        for _ in range(self.max_tool_iterations):
            messages = self.memory.get_messages()
            schemas = self.tools.get_schemas() or None

            _m = self.hooks.run("before_llm_call", messages)
            if _m is not None:
                messages = _m

            response = await loop.run_in_executor(
                self._executor,
                lambda msgs=messages, sc=schemas: self.llm.chat(msgs, tools=sc, **llm_kwargs),
            )

            _resp = self.hooks.run("after_llm_response", response)
            if _resp is not None:
                response = _resp

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

        fut = loop.run_in_executor(self._executor, _produce)

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
        self._executor.shutdown(wait=False)

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

    async def _execute_tool_async(self, tool_call: ToolCall) -> Any:
        loop = asyncio.get_running_loop()

        tool_call = self.hooks.run("before_tool_execute", tool_call) or tool_call

        tool = self.tools.get(tool_call.name)
        if tool is None:
            result: Any = f"Error: tool '{tool_call.name}' is not registered."
        else:
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    lambda tc=tool_call: tool.execute(**tc.arguments),
                )
            except Exception as exc:
                result = f"Error executing '{tool_call.name}': {exc}"

        # Use explicit None check — `or` would swallow falsy results like 0 or "".
        _r = self.hooks.run("after_tool_execute", result, tool_call)
        if _r is not None:
            result = _r
        return result
