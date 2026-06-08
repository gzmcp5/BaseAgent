"""Ollama 제공자 — 내 컴퓨터에서 로컬로 도는 LLM에 연결한다.

클라우드 API와 달리 Ollama는 보통 localhost에서 실행되므로 API 키가 필요 없다.
메시지/도구 형식은 OpenAI와 비슷하지만, 스트리밍은 SSE가 아니라 NDJSON(줄마다 JSON)을 쓴다.
"""
from __future__ import annotations
import json
import os
import uuid
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message, Role, ToolCall
from .base import BaseLLM

_DEFAULT_BASE_URL = "http://localhost:11434"  # Ollama 기본 주소(로컬).


class OllamaLLM(BaseLLM):
    """Ollama local LLM via direct HTTP."""

    DEFAULT_MODEL = "llama3.2"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        retry_config: Optional["RetryConfig"] = None,
    ) -> None:
        super().__init__(model, retry_config=retry_config)
        # 주소 우선순위: 인자 > 환경변수 OLLAMA_BASE_URL > 기본 localhost.
        raw = base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        self.base_url = raw.rstrip("/")  # 끝의 '/'를 제거해 URL 조립 시 '//' 중복을 막는다.

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
        # Ollama는 NDJSON 스트림: 각 줄이 하나의 JSON 청크다.
        for chunk in self._stream_ndjson(url, payload, {"Content-Type": "application/json"}):
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield text
            if chunk.get("done"):  # done=True 이면 마지막 청크 → 종료.
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
            # 모델/버전에 따라 인자가 문자열로 올 수도 있어, 그럴 때만 딕셔너리로 파싱한다.
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(
                ToolCall(
                    id=str(uuid.uuid4()),  # Ollama도 호출 id를 주지 않으므로 직접 생성.
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", self.model),
        )
