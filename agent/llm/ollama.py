from __future__ import annotations
import json
import os
import uuid
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message, Role, ToolCall
from .base import BaseLLM

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaLLM(BaseLLM):
    """Ollama local LLM via direct HTTP."""

    DEFAULT_MODEL = "llama3.2"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        retry_config: Optional["RetryConfig"] = None,
    ) -> None:
        from .retry import RetryConfig as _RC  # noqa: F401
        super().__init__(model, retry_config=retry_config)
        raw = base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        self.base_url = raw.rstrip("/")

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": False,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        url = f"{self.base_url}/api/chat"
        data = self._request(url, payload, {"Content-Type": "application/json"})
        return self._parse_response(data)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream text chunks from Ollama via NDJSON."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": True,
        }
        url = f"{self.base_url}/api/chat"
        for chunk in self._stream_ndjson(url, payload, {"Content-Type": "application/json"}):
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield text
            if chunk.get("done"):
                break

    # ------------------------------------------------------------------

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.USER, Role.ASSISTANT):
                formatted.append(
                    {"role": msg.role.value, "content": msg.content or ""}
                )
            elif msg.role == Role.TOOL:
                formatted.append({"role": "tool", "content": msg.content})
        return formatted

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        msg = data.get("message", {})
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(
                ToolCall(
                    id=str(uuid.uuid4()),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", self.model),
        )
