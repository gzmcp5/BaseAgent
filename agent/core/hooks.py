"""훅(Hook) 시스템 — 코어 코드를 고치지 않고 기능을 끼워 넣는 확장점.

이 프레임워크의 보안·로깅 확장은 모두 여기서 출발한다(SECURITY_FEATURE.md).
에이전트 실행 흐름의 정해진 4개 지점(이벤트)에서 등록된 함수들이 자동 호출된다:
  - before_llm_call    : LLM에 보내기 직전의 메시지 (프롬프트 주입 방어 등)
  - after_llm_response : LLM이 응답한 직후 (API 키 마스킹 등)
  - before_tool_execute: 도구 실행 직전 (경로 탈출 차단 등)
  - after_tool_execute : 도구 실행 직후 (결과 검열 등)

규칙은 단순하다: 훅이 None을 반환하면 '통과'(원본 유지), 값을 반환하면 그 값으로 '교체'.
여러 훅을 같은 이벤트에 걸면 값이 체인처럼 순서대로 전달된다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# 훅으로 등록할 수 있는 함수의 타입(임의 인자를 받아 임의 값을 반환).
HookFn = Callable[..., Any]

# 허용된 이벤트 이름 집합. 오타로 엉뚱한 이벤트에 거는 실수를 막기 위해 화이트리스트로 검증한다.
_VALID_EVENTS = frozenset(
    {
        "before_llm_call",    # (messages: list[Message]) -> list[Message] | None
        "after_llm_response", # (response: LLMResponse)  -> LLMResponse | None
        "before_tool_execute",# (tool_call: ToolCall)     -> ToolCall | None
        "after_tool_execute", # (result: Any, tool_call: ToolCall) -> Any | None
    }
)


@dataclass
class HookRegistry:
    """Lightweight middleware pipeline for the Agent loop.

    Each hook event threads its primary value through registered handlers.
    Returning a non-None value from a hook replaces the current value;
    returning None leaves it unchanged.

    Event signatures:
        before_llm_call(messages)          -> messages | None
        after_llm_response(response)       -> response | None
        before_tool_execute(tool_call)     -> tool_call | None
        after_tool_execute(result, tool_call) -> result | None
            - *result* is the primary value threaded through the chain.
            - *tool_call* is read-only context forwarded to every handler.

    Usage::

        hooks = HookRegistry()

        @hooks.on("before_llm_call")
        def log_messages(messages):
            print(f"Sending {len(messages)} messages to LLM")

        @hooks.on("after_tool_execute")
        def redact(result, tool_call):
            if "secret" in str(result):
                return "[REDACTED]"
    """

    # 이벤트 이름 → 그 이벤트에 걸린 훅 함수들의 목록. (init=False: 생성자 인자로 받지 않음)
    _registry: dict[str, list[HookFn]] = field(default_factory=dict, init=False)

    def on(self, event: str) -> Callable[[HookFn], HookFn]:
        """Decorator that registers a hook for *event*."""
        # 모르는 이벤트 이름이면 바로 에러 — 조용히 무시되면 디버깅이 어렵기 때문.
        if event not in _VALID_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Valid events: {sorted(_VALID_EVENTS)}"
            )

        # 데코레이터 본체: 함수를 해당 이벤트 목록에 추가하고 원본을 그대로 돌려준다.
        def decorator(fn: HookFn) -> HookFn:
            self._registry.setdefault(event, []).append(fn)
            return fn

        return decorator

    def register(self, event: str, fn: HookFn) -> None:
        """Register *fn* for *event* (non-decorator form)."""
        # on()의 데코레이터 없이 함수를 직접 등록하는 버전(동작은 동일).
        if event not in _VALID_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Valid events: {sorted(_VALID_EVENTS)}"
            )
        self._registry.setdefault(event, []).append(fn)

    def run(self, event: str, *args: Any) -> Any:
        """Thread args[0] through all handlers for *event*.

        Returns the (possibly transformed) first arg, or None when no
        handlers are registered (callers must supply their own fallback).
        """
        handlers = self._registry.get(event, [])
        # 걸린 훅이 하나도 없으면 None을 반환 → 호출 측이 `결과 or 원본값`으로 원본을 쓴다.
        if not handlers:
            return None  # caller uses `or original_value` as fallback

        # args[0] = 체인을 따라 변형될 '주 값'. 나머지(extra)는 읽기 전용 부가 정보로 전달.
        value = args[0]
        extra = args[1:]
        for fn in handlers:
            candidate = fn(value, *extra)
            # 훅이 값을 반환하면 다음 훅에는 그 새 값이 넘어간다(None이면 현재 값 유지).
            if candidate is not None:
                value = candidate
        return value

    def clear(self, event: Optional[str] = None) -> None:
        """Remove hooks — all events if *event* is None, otherwise that event only."""
        # event를 안 주면 전체 훅 제거, 주면 해당 이벤트의 훅만 제거(없어도 에러 안 남).
        if event is None:
            self._registry.clear()
        else:
            self._registry.pop(event, None)

    def __len__(self) -> int:
        # 이벤트 구분 없이 등록된 훅의 '총' 개수.
        return sum(len(fns) for fns in self._registry.values())
