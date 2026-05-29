from __future__ import annotations
import os
import uuid
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message, Role, ToolCall
from .base import BaseLLM

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleLLM(BaseLLM):
    """Google Gemini via direct HTTP (no SDK)."""

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        retry_config: Optional["RetryConfig"] = None,
    ) -> None:
        super().__init__(model, retry_config=retry_config)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_prompt, contents = self._format_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens)
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            payload["tools"] = [
                {"function_declarations": [self._to_gemini_tool(t) for t in tools]}
            ]

        url = f"{_BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = self._request(url, payload, headers)
        return self._parse_response(data)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream text chunks via Gemini SSE (streamGenerateContent)."""
        import json as _json

        system_prompt, contents = self._format_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens)
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        url = (
            f"{_BASE_URL}/{self.model}:streamGenerateContent"
            f"?alt=sse&key={self.api_key}"
        )
        headers = {"Content-Type": "application/json"}
        for data_str in self._stream_sse(url, payload, headers):
            if not data_str:
                continue
            try:
                chunk = _json.loads(data_str)
            except _json.JSONDecodeError:
                continue
            parts = (
                chunk.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            for part in parts:
                text = part.get("text", "")
                if text:
                    yield text

    # ------------------------------------------------------------------

    def _format_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        system_prompt = ""
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            elif msg.role == Role.USER:
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == Role.ASSISTANT:
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            {"functionCall": {"name": tc.name, "args": tc.arguments}}
                        )
                contents.append({"role": "model", "parts": parts})
            elif msg.role == Role.TOOL:
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.name,
                                    "response": {"result": msg.content},
                                }
                            }
                        ],
                    }
                )

        return system_prompt, contents

    @staticmethod
    def _to_gemini_tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        candidate = (data.get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        content = ""
        tool_calls: list[ToolCall] = []
        for part in parts:
            if "text" in part:
                content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=str(uuid.uuid4()),
                        name=fc["name"],
                        arguments=fc.get("args", {}),
                    )
                )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=self.model,
        )
