from __future__ import annotations
import json
import os
import uuid
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message, Role, ToolCall
from .base import BaseLLM

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class ClaudeLLM(BaseLLM):
    """Anthropic Claude via direct HTTP (no SDK)."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_prompt, formatted = self._format_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": formatted,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = [self._to_claude_tool(t) for t in tools]

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        data = self._request(_API_URL, payload, headers)
        return self._parse_response(data)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream text chunks via Anthropic SSE.

        Note: streaming does not support tool calls — use chat() for tool-enabled turns.
        """
        system_prompt, formatted = self._format_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": formatted,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        for data_str in self._stream_sse(_API_URL, payload, headers):
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        system_prompt = ""
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            elif msg.role == Role.USER:
                formatted.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    parts: list[dict[str, Any]] = []
                    if msg.content:
                        parts.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        parts.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                    formatted.append({"role": "assistant", "content": parts})
                else:
                    formatted.append({"role": "assistant", "content": msg.content})
            elif msg.role == Role.TOOL:
                # Tool results must be wrapped in a user message
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                if (
                    formatted
                    and formatted[-1]["role"] == "user"
                    and isinstance(formatted[-1]["content"], list)
                ):
                    formatted[-1]["content"].append(tool_result)
                else:
                    formatted.append({"role": "user", "content": [tool_result]})

        return system_prompt, formatted

    @staticmethod
    def _to_claude_tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        content = ""
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content = block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
        )
