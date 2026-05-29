from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


HookFn = Callable[..., Any]

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

    _registry: dict[str, list[HookFn]] = field(default_factory=dict, init=False)

    def on(self, event: str) -> Callable[[HookFn], HookFn]:
        """Decorator that registers a hook for *event*."""
        if event not in _VALID_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Valid events: {sorted(_VALID_EVENTS)}"
            )

        def decorator(fn: HookFn) -> HookFn:
            self._registry.setdefault(event, []).append(fn)
            return fn

        return decorator

    def register(self, event: str, fn: HookFn) -> None:
        """Register *fn* for *event* (non-decorator form)."""
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
        if not handlers:
            return None  # caller uses `or original_value` as fallback

        value = args[0]
        extra = args[1:]
        for fn in handlers:
            candidate = fn(value, *extra)
            if candidate is not None:
                value = candidate
        return value

    def clear(self, event: Optional[str] = None) -> None:
        """Remove hooks — all events if *event* is None, otherwise that event only."""
        if event is None:
            self._registry.clear()
        else:
            self._registry.pop(event, None)

    def __len__(self) -> int:
        return sum(len(fns) for fns in self._registry.values())
