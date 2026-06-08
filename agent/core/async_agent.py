"""Agent의 비동기 버전 — 한 턴의 여러 도구를 '동시에' 실행한다.

동기 Agent는 도구를 하나씩 순서대로 실행하지만, AsyncAgent는 LLM이 한 번에 여러 도구를
요청했을 때 asyncio.gather로 병렬 실행해 I/O 대기 시간을 겹친다(예: 도구 3개가 각각
네트워크를 기다린다면, 순차로는 3배의 시간이 걸리지만 동시 실행하면 거의 1배).

주의: LLM 호출과 도구 실행은 내부적으로 동기(blocking) 코드(urllib 사용)라서,
스레드풀(executor)에 떠넘겨 이벤트 루프를 막지 않도록 처리한다. 반면 훅은 이벤트 루프
스레드에서 그대로 실행되므로 훅 안에서 블로킹 작업을 하면 루프가 멈춘다(docstring 참고).
"""
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

            # llm.chat은 블로킹이므로 스레드풀에서 돌려 이벤트 루프를 막지 않는다.
            # (람다에 기본 인자로 msgs/sc를 묶는 건 늦은 바인딩 버그를 피하기 위함.)
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

            # 이번 턴에 요청된 모든 도구 호출을 '동시에' 실행한다(동기 Agent와의 핵심 차이).
            # gather는 결과를 '호출 순서대로' 돌려주므로 아래 zip이 안전하게 짝지어진다.
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

        # 블로킹 스트림(별도 스레드)과 비동기 소비자(이 코루틴)를 '큐'로 연결한다.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        exc_box: list[BaseException] = []  # 스레드에서 난 예외를 바깥으로 옮겨 담는 상자.

        # 생산자: 별도 스레드에서 동기 스트림을 읽어 조각을 큐에 넣는다.
        def _produce() -> None:
            try:
                for chunk in self.llm.stream(messages, **llm_kwargs):
                    # 다른 스레드에서 이벤트 루프의 큐를 건드리므로 threadsafe API를 써야 한다.
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                exc_box.append(exc)
            finally:
                # 끝(또는 에러)을 알리는 신호로 None(센티넬)을 넣어 소비 루프를 멈추게 한다.
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        fut = asyncio.create_task(self._run_blocking(_produce))

        # 소비자: 큐에서 조각을 꺼내 호출자에게 흘려보내면서 전체 응답을 모은다.
        full_response = ""
        while True:
            chunk = await queue.get()
            if chunk is None:  # 센티넬을 만나면 스트림 종료.
                break
            full_response += chunk
            yield chunk

        await fut  # 스레드가 완전히 끝났는지 확인(executor 단계 에러도 여기서 드러남).

        if exc_box:
            raise exc_box[0]  # 스트리밍 중 LLM 에러가 있었으면 그대로 올린다.

        # 에러 없이 끝났을 때만 메모리에 저장한다(중간에 끊겼다면 반쪽짜리를 남기지 않음).
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
        # 동기(블로킹) 함수를 스레드풀에서 실행하고 결과를 await로 받는 헬퍼.
        # 덕분에 이벤트 루프(이 코루틴들)는 블로킹 호출 동안에도 멈추지 않는다.
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            return await loop.run_in_executor(executor, fn)
        finally:
            executor.shutdown(wait=True)  # 작업이 끝나면 스레드풀을 정리(자원 누수 방지).
