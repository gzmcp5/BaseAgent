"""프레임워크의 심장 — 에이전트 실행 루프.

에이전트 한 턴의 흐름은 다음과 같다(run 메서드 참고):
  1) 사용자 입력을 메모리에 저장
  2) (메모리 + 도구 목록)을 LLM에 전달해 응답을 받음
  3) 응답에 '도구 호출' 요청이 없으면 → 그 텍스트가 최종 답변이므로 반환
  4) 도구 호출 요청이 있으면 → 도구들을 실행하고 결과를 메모리에 넣은 뒤 2)로 되돌아감
  5) 이 과정을 최대 max_tool_iterations번 반복

이 'LLM ↔ 도구' 왕복 구조가 에이전트가 스스로 일을 처리하는 핵심 원리다.
보안·로깅 등 부가 기능은 코드를 고치지 않고 '훅(hooks)'으로 끼워 넣는다.
"""
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
# (사람-개입 승인 콜백) 위험한 도구를 실행하기 직전 호출되어, True면 실행 허용·False면 거부.
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
        self.llm = llm                                    # 실제 호출할 LLM 제공자.
        self.tools = tools or ToolRegistry()              # 사용할 도구 모음(없으면 빈 레지스트리).
        # 메모리를 직접 주지 않으면 기본 인메모리 대화 기록을 만들어 system_prompt를 심는다.
        self.memory = memory or ConversationMemory(system_prompt=system_prompt)
        self.hooks = hooks or HookRegistry()              # 훅 파이프라인(보안·로깅 확장점).
        # 한 턴에서 도구 호출↔LLM 왕복을 몇 번까지 허용할지(무한 루프 방지 안전장치).
        self.max_tool_iterations = max_tool_iterations
        # Optional RAG-based tool pruning: when set, only the tools most
        # relevant to the user query are sent to the LLM.
        self.tool_selector = tool_selector
        # Optional human-in-the-loop gate for tools marked requires_approval.
        self.approval_callback = approval_callback

    def run(self, user_input: str, **llm_kwargs: Any) -> str:
        """Run a single user turn; returns the final assistant response."""
        # 1) 사용자 입력을 대화 기록에 추가.
        self.memory.add(Message(role=Role.USER, content=user_input))

        # 도구 호출이 끝없이 이어지지 않도록 반복 횟수에 상한을 둔다.
        for _ in range(self.max_tool_iterations):
            messages = self.memory.get_messages()      # 지금까지의 전체 대화.
            schemas = self._schemas_for(user_input)    # 이번에 LLM에 알려줄 도구 목록.

            # [훅] LLM 호출 직전 — 프롬프트 검열/주입 방어/감사 로그 등. 값을 반환하면 교체됨.
            _m = self.hooks.run("before_llm_call", messages)
            if _m is not None:
                messages = _m

            # 2) LLM에 대화와 도구 목록을 보내 응답을 받는다.
            response = self.llm.chat(messages, tools=schemas, **llm_kwargs)

            # [훅] LLM 응답 직후 — 출력 리댁션(API 키 마스킹 등). 값을 반환하면 교체됨.
            _resp = self.hooks.run("after_llm_response", response)
            if _resp is not None:
                response = _resp

            # 3) 도구 호출 요청이 없다 = LLM이 최종 답을 내놓았다 → 저장 후 종료.
            if not response.tool_calls:
                content = response.content or ""
                self.memory.add(Message(role=Role.ASSISTANT, content=content))
                return content

            # 4) 도구 호출 요청이 있으면, 그 요청(어시스턴트 메시지)을 먼저 기록한다.
            #    (LLM이 "무엇을 호출했는지"가 대화 맥락에 남아야 다음 턴이 일관된다.)
            self.memory.add(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
            )

            # 요청된 도구들을 차례로 실행하고, 각 결과를 TOOL 메시지로 기록한다.
            # (결과가 메모리에 들어가면 다음 반복에서 LLM이 그것을 보고 이어서 판단한다.)
            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                self.memory.add(
                    Message(
                        role=Role.TOOL,
                        content=str(result),
                        tool_call_id=tool_call.id,   # 어떤 호출의 결과인지 짝지음.
                        name=tool_call.name,
                    )
                )

        # 5) 상한까지 반복했는데도 최종 답이 안 나온 경우의 안전 메시지.
        return "Max tool iterations reached without a final response."

    def stream(self, user_input: str, **llm_kwargs: Any) -> Iterator[str]:
        """Stream text chunks for a user turn.

        Tool calls are not supported in streaming mode — use run() for that.
        Memory is updated with the full assembled response when the stream ends.
        """
        self.memory.add(Message(role=Role.USER, content=user_input))
        messages = self.memory.get_messages()
        # 스트리밍에도 before_llm_call 훅은 동일하게 적용된다.
        _m = self.hooks.run("before_llm_call", messages)
        if _m is not None:
            messages = _m

        # 조각(chunk)을 받는 즉시 호출자에게 흘려보내면서, 전체 응답도 따로 모은다.
        full_response = ""
        for chunk in self.llm.stream(messages, **llm_kwargs):
            full_response += chunk
            yield chunk

        # 스트림이 끝난 뒤에야 완성된 응답을 메모리에 한 번에 저장한다.
        self.memory.add(Message(role=Role.ASSISTANT, content=full_response))

    def _schemas_for(self, user_input: str) -> Optional[list[dict]]:
        """Tool schemas to advertise to the LLM for this turn.

        With a tool_selector, only the tools most relevant to *user_input*
        are sent (RAG pruning); otherwise all registered tools are sent.
        """
        # tool_selector가 있으면 질의와 관련된 상위 몇 개 도구 스키마만 보낸다(토큰 절약).
        if self.tool_selector is not None:
            return self.tool_selector.select_schemas(user_input) or None
        # 없으면 등록된 모든 도구 스키마를 보낸다. 도구가 없으면 None(=도구 비활성).
        return self.tools.get_schemas() or None

    def _approved(self, tool: "Any", tool_call: ToolCall) -> bool:
        """Human-in-the-loop gate for tools marked requires_approval.

        Fail-closed: an approval-required tool is denied unless a callback is
        configured and explicitly returns True.
        """
        # 승인이 필요 없는 일반 도구는 그냥 통과.
        if not tool.requires_approval:
            return True
        # 승인이 필요한데 승인 콜백 자체가 없으면 → 안전하게 거부(fail-closed).
        if self.approval_callback is None:
            return False
        # 콜백이 명시적으로 True를 줄 때만 실행 허용.
        return bool(self.approval_callback(tool_call))

    def _execute_tool(self, tool_call: ToolCall) -> Any:
        # [훅] 도구 실행 직전 — 인자 변조 검사·경로 탈출 차단 등. 반환값이 있으면 호출 자체를 교체.
        tool_call = self.hooks.run("before_tool_execute", tool_call) or tool_call

        # 이름으로 실제 도구를 찾는다. 아래에서 4가지 경우로 분기:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            # (a) 등록되지 않은 도구를 부른 경우 → 에러 메시지를 결과로(예외 대신).
            result: Any = f"Error: tool '{tool_call.name}' is not registered."
        elif not self._approved(tool, tool_call):
            # (b) 승인이 필요한데 허가되지 않은 경우 → 거부 메시지.
            result = f"Tool '{tool_call.name}' execution was denied (awaiting user approval)."
        else:
            try:
                # (c) 정상 실행. 도구 안에서 난 예외는...
                result = tool.execute(**tool_call.arguments)
            except Exception as exc:
                # (d) ...삼켜서 에러 문자열로 바꾼다. 그래야 LLM이 실패를 '보고' 대처할 수 있다.
                result = f"Error executing '{tool_call.name}': {exc}"

        # [훅] 도구 실행 직후 — 결과 리댁션·감사 로그 등.
        # None 여부를 명시적으로 확인한다. `or`를 쓰면 0이나 "" 같은 정상 결과까지 덮어쓰기 때문.
        _r = self.hooks.run("after_tool_execute", result, tool_call)
        if _r is not None:
            result = _r
        return result

    def reset(self) -> None:
        """Clear conversation history while keeping the system prompt."""
        # 대화만 초기화하고 에이전트 성격(system_prompt)은 유지 → 같은 에이전트로 새 대화 시작.
        self.memory.clear()
