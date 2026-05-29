from __future__ import annotations
import json
import os
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message, Role, ToolCall
from .base import BaseLLM

_DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completions via direct HTTP (no SDK)."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or _DEFAULT_BASE_URL
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = self._request(self.base_url, payload, headers)
        return self._parse_response(data)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream text chunks via OpenAI SSE."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for data_str in self._stream_sse(self.base_url, payload, headers):
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content") or ""
            if text:
                yield text

    # ------------------------------------------------------------------

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.USER, Role.ASSISTANT):
                entry: dict[str, Any] = {
                    "role": msg.role.value,
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                formatted.append(entry)
            elif msg.role == Role.TOOL:
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
        return formatted

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        choice = data["choices"][0]["message"]
        tool_calls: list[ToolCall] = []
        for tc in choice.get("tool_calls") or []:
            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                )
            )
        return LLMResponse(
            content=choice.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
        )
