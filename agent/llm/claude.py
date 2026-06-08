"""Anthropic Claude 제공자 — 표준 Message ↔ Claude API 형식을 상호 변환한다.

이 파일이 하는 일은 결국 '번역'이다: 프레임워크 공용 Message 목록을 Claude API가 원하는
JSON으로 바꿔 보내고(_format_messages), 받은 JSON 응답을 다시 공용 LLMResponse로 되돌린다
(_parse_response). HTTP 호출·재시도·스트리밍 같은 공통 작업은 부모 BaseLLM이 처리한다.
"""
from __future__ import annotations
import json
import os
import uuid
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message, Role, ToolCall
from .base import BaseLLM

_API_URL = "https://api.anthropic.com/v1/messages"  # Claude 메시지 API 엔드포인트.
_API_VERSION = "2023-06-01"                          # Anthropic API 버전 헤더 값.


class ClaudeLLM(BaseLLM):
    """Anthropic Claude via direct HTTP (no SDK)."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        retry_config: Optional["RetryConfig"] = None,
    ) -> None:
        super().__init__(model, retry_config=retry_config)
        # 키를 직접 주지 않으면 환경변수 ANTHROPIC_API_KEY에서 읽는다.
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens  # 응답 최대 토큰 수.

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # 공용 메시지를 Claude 형식으로 변환. Claude는 system을 별도 필드로 받는다.
        system_prompt, formatted = self._format_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": formatted,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            # 공용 도구 스키마를 Claude의 도구 형식(input_schema 키 등)으로 변환해 넣는다.
            payload["tools"] = [self._to_claude_tool(t) for t in tools]

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        data = self._request(_API_URL, payload, headers)  # 부모의 재시도 로직을 탄다.
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
                break  # 스트림 종료 신호.
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue  # 깨진 이벤트 줄은 무시하고 계속.
            # 텍스트가 조금씩 추가되는 이벤트(content_block_delta의 text_delta)만 골라 내보낸다.
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
        system_prompt = ""                      # Claude는 system을 messages 밖으로 빼서 반환.
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # 시스템 메시지는 목록에 넣지 않고 따로 모아 둔다(마지막 것이 적용됨).
                system_prompt = msg.content
            elif msg.role == Role.USER:
                formatted.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    # 도구 호출이 있으면 content를 'parts(블록) 배열'로 구성한다.
                    parts: list[dict[str, Any]] = []
                    if msg.content:
                        parts.append({"type": "text", "text": msg.content})  # 텍스트 블록.
                    for tc in msg.tool_calls:
                        parts.append(
                            {
                                "type": "tool_use",   # 도구 사용 블록.
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                    formatted.append({"role": "assistant", "content": parts})
                else:
                    # 평범한 텍스트 응답.
                    formatted.append({"role": "assistant", "content": msg.content})
            elif msg.role == Role.TOOL:
                # Claude에서는 도구 결과를 'user' 메시지 안의 tool_result 블록으로 보내야 한다.
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,  # 어떤 tool_use에 대한 결과인지 연결.
                    "content": msg.content,
                }
                # 직전 메시지가 이미 user의 블록 배열이면 거기에 이어 붙이고(결과 여러 개 합치기),
                if (
                    formatted
                    and formatted[-1]["role"] == "user"
                    and isinstance(formatted[-1]["content"], list)
                ):
                    formatted[-1]["content"].append(tool_result)
                else:
                    # 아니면 새 user 메시지를 만든다.
                    formatted.append({"role": "user", "content": [tool_result]})

        return system_prompt, formatted

    @staticmethod
    def _to_claude_tool(tool: dict[str, Any]) -> dict[str, Any]:
        # 공용 스키마의 "parameters"를 Claude가 쓰는 "input_schema" 키로 바꿔 준다.
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        # Claude 응답의 content는 블록 배열이다. 텍스트 블록과 도구 사용 블록을 나눠 모은다.
        content = ""
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]            # 텍스트 블록은 이어 붙임.
            elif block["type"] == "tool_use":
                tool_calls.append(                  # 도구 사용 블록은 ToolCall로 변환.
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
