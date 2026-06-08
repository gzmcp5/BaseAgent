"""Google Gemini 제공자 — 공용 Message ↔ Gemini 형식 변환.

Gemini는 용어가 또 다르다: 메시지 목록이 'contents', 어시스턴트 역할은 'model', 도구 결과는
'function'/'functionResponse'로 표현한다. 또 Gemini 응답에는 도구 호출 id가 없어서 여기서
uuid로 직접 만들어 붙인다(프레임워크가 호출과 결과를 짝지으려면 id가 꼭 필요하기 때문).
"""
from __future__ import annotations
import os
import uuid
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message, Role, ToolCall
from .base import BaseLLM

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleLLM(BaseLLM):
    """Google Gemini via direct HTTP (no SDK)."""

    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        retry_config: Optional["RetryConfig"] = None,
    ) -> None:
        super().__init__(model, retry_config=retry_config)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")  # 키 없으면 환경변수에서.
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
            # Gemini는 system을 별도 systemInstruction 필드로 받는다(Claude와 비슷).
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            # 도구들은 function_declarations 목록 하나로 묶어 전달.
            payload["tools"] = [
                {"function_declarations": [self._to_gemini_tool(t) for t in tools]}
            ]

        # 모델 이름과 동작(generateContent)이 URL 경로에 들어간다.
        url = f"{_BASE_URL}/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        data = self._request(url, payload, headers)
        return self._parse_response(data)

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream text chunks via Gemini SSE (streamGenerateContent)."""
        import json as _json  # 스트리밍에서만 쓰므로 함수 내부에서 import.

        system_prompt, contents = self._format_messages(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.max_tokens)
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        # 스트리밍 전용 동작(streamGenerateContent) + alt=sse로 SSE 형식 요청.
        url = f"{_BASE_URL}/{self.model}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        for data_str in self._stream_sse(url, payload, headers):
            if not data_str:
                continue  # 빈 줄 무시.
            try:
                chunk = _json.loads(data_str)
            except _json.JSONDecodeError:
                continue  # 깨진 줄 무시.
            # 각 조각에서 후보의 parts를 꺼내 텍스트만 흘려보낸다.
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
        system_prompt = ""                      # system은 contents 밖으로 분리해 반환.
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
            elif msg.role == Role.USER:
                # Gemini는 텍스트도 'parts' 배열 안에 담는다.
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == Role.ASSISTANT:
                # 어시스턴트 역할은 Gemini 용어로 'model'.
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        # 도구 호출은 functionCall 파트로(인자는 args 키, 딕셔너리 그대로).
                        parts.append(
                            {"functionCall": {"name": tc.name, "args": tc.arguments}}
                        )
                contents.append({"role": "model", "parts": parts})
            elif msg.role == Role.TOOL:
                # 도구 결과는 'function' 역할의 functionResponse 파트로 보낸다.
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
        # 첫 번째 후보(candidate)의 parts를 훑어 텍스트와 함수 호출을 분리한다.
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
                        id=str(uuid.uuid4()),  # Gemini는 호출 id를 안 주므로 직접 생성.
                        name=fc["name"],
                        arguments=fc.get("args", {}),
                    )
                )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=self.model,
        )
