from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# Each hook receives the current value and may return a transformed replacement.
# If a hook returns None the value is passed through unchanged.
HookFn = Callable[..., Any]

_VALID_EVENTS = frozenset(
    {
        "before_llm_call",    # (messages: list[Message]) -> list[Message] | None
        "after_llm_response", # (response: LLMResponse)  -> LLMResponse | None
        "before_tool_execute",# (tool_call: ToolCall)     -> ToolCall | None
        "after_tool_execute", # (tool_call: ToolCall, result: Any) -> Any | None
    }
)


@dataclass
class HookRegistry:
    """Lightweight middleware pipeline for the Agent loop.

    Usage::

        hooks = HookRegistry()

        @hooks.on("before_llm_call")
        def log_messages(messages):
            print(f"Sending {len(messages)} messages to LLM")
            # Return None to leave messages unchanged

        @hooks.on("after_tool_execute")
        def audit_tool(tool_call, result):
            print(f"Tool {tool_call.name} returned: {result}")
    """

    _registry: dict[str, list[HookFn]] = field(default_factory=dict, init=False)

    def on(self, event: str) -> Callable[[HookFn], HookFn]:
        """Decorator to register a hook for *event*."""
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
        """Run all hooks for *event* in registration order.

        The first positional arg is threaded through the hook chain.
        Each hook may return a replacement value; returning None keeps the
        current value. Additional args (e.g. result in after_tool_execute)
        are forwarded read-only.
        """
        handlers = self._registry.get(event, [])
        if not handlers:
            return args[0] if args else None

        value = args[0]
        extra = args[1:]
        for fn in handlers:
            result = fn(value, *extra)
            if result is not None:
                value = result
        return value

    def clear(self, event: Optional[str] = None) -> None:
        """Remove hooks — all events if *event* is None, otherwise only that event."""
        if event is None:
            self._registry.clear()
        else:
            self._registry.pop(event, None)

    def __len__(self) -> int:
        return sum(len(fns) for fns in self._registry.values())
