"""OpenAI Chat Completions 제공자 — 공용 Message ↔ OpenAI 형식 변환.

Claude와 형식이 다른 점이 핵심: OpenAI는 system도 그냥 messages 배열 안의 한 항목이고,
도구 호출은 어시스턴트 메시지의 'tool_calls' 필드에, 도구 결과는 role="tool" 메시지에 담는다.
이 OpenAI 형식은 사실상 표준처럼 쓰여서, Ollama와 OpenRouter도 거의 같은 형식을 따른다.
"""
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
        retry_config: Optional["RetryConfig"] = None,
    ) -> None:
        super().__init__(model, retry_config=retry_config)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")  # 키 없으면 환경변수에서.
        self.base_url = base_url or _DEFAULT_BASE_URL  # 호환 API라면 base_url만 바꿔 재사용 가능.
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
            # OpenAI는 각 도구를 {"type":"function","function":{...}} 로 감싸서 전달해야 한다.
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        headers = {
            "Authorization": f"Bearer {self.api_key}",  # OpenAI는 Bearer 토큰 방식.
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
                break  # 스트림 종료 신호.
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue  # 깨진 줄 무시.
            # 스트리밍에서는 새 텍스트가 choices[0].delta.content 에 조금씩 담겨 온다.
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content") or ""
            if text:
                yield text

    # ------------------------------------------------------------------

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            # system/user/assistant는 형식이 같아 한꺼번에 처리(Claude와 달리 system도 배열 내부).
            if msg.role in (Role.SYSTEM, Role.USER, Role.ASSISTANT):
                entry: dict[str, Any] = {
                    "role": msg.role.value,
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    # OpenAI는 인자를 JSON '문자열'로 직렬화해 넣는다(딕셔너리가 아님에 주의).
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
                # 도구 결과는 role="tool" 메시지로, 어떤 호출의 결과인지 id로 연결한다.
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
        return formatted

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        # 응답은 choices 배열 — 보통 첫 번째 후보의 message만 사용한다.
        choice = data["choices"][0]["message"]
        tool_calls: list[ToolCall] = []
        for tc in choice.get("tool_calls") or []:
            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    # 보낼 때 문자열이었으니, 받을 때는 다시 딕셔너리로 역직렬화한다.
                    arguments=json.loads(tc["function"]["arguments"]),
                )
            )
        return LLMResponse(
            content=choice.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
        )
