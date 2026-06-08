"""모든 LLM 제공자의 공통 토대(추상 클래스)와 HTTP 통신 헬퍼.

Claude/OpenAI/Gemini/Ollama 등 제공자는 API 모양이 다 다르지만, 에이전트 입장에서는
'메시지를 보내면 응답을 받는다'는 동작만 같으면 된다. BaseLLM은 그 공통 인터페이스
(chat/stream)를 강제하고, 모든 제공자가 똑같이 쓰는 HTTP 호출·재시도·스트리밍 파싱을
한곳에 모아 둔다. 외부 SDK 없이 표준 라이브러리 urllib만 사용하는 것이 이 프로젝트의 원칙.
"""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message
from .retry import RetryConfig, DEFAULT_RETRY


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model: str, retry_config: Optional[RetryConfig] = None) -> None:
        self.model = model                                # 사용할 모델 이름.
        self.retry_config = retry_config or DEFAULT_RETRY  # 재시도 정책(없으면 공용 기본값).

    # 아래 두 메서드는 @abstractmethod라 하위 제공자 클래스가 '반드시' 구현해야 한다.
    # (구현하지 않으면 그 클래스는 인스턴스화 자체가 불가능 → 깜빡한 구현을 일찍 잡아낸다.)

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages and return a complete response."""
        # 한 번 호출해 완성된 응답을 통째로 받는다(도구 호출 지원).

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield response text chunks (streaming)."""
        # 응답을 토막토막 실시간으로 흘려보낸다(도구 호출은 미지원).

    # ------------------------------------------------------------------
    # Shared HTTP helpers
    # ------------------------------------------------------------------

    def _request(self, url: str, payload: dict, headers: dict) -> dict:
        """HTTP POST with automatic retry and exponential backoff."""
        cfg = self.retry_config
        last_exc: Exception = RuntimeError("No attempts made")

        # 최초 1회 + 재시도 max_retries회 = 총 (max_retries + 1)번 시도한다.
        for attempt in range(cfg.max_retries + 1):
            try:
                return self._do_request(url, payload, headers)
            except RuntimeError as exc:
                last_exc = exc
                # 재시도 의미 없는 에러(예: 400)이거나 마지막 시도였다면 그대로 예외를 올린다.
                if not cfg.is_retryable(str(exc)) or attempt >= cfg.max_retries:
                    raise
                # 그 외에는 점점 길어지는 시간만큼 쉬었다가 다음 시도(지수 백오프).
                time.sleep(cfg.delay_for(attempt))

        raise last_exc  # pragma: no cover  (이론상 도달 불가 — 위 루프에서 항상 반환/예외)

    @staticmethod
    def _safe_url(url: str) -> str:
        """Strip query string from URL to avoid leaking secrets in error messages."""
        # 일부 제공자는 API 키를 쿼리스트링(?key=...)에 담는다. 에러 메시지에 그대로 찍히면
        # 키가 로그로 새므로, '?' 앞부분만 남겨 안전하게 만든다.
        return url.split("?")[0]

    @staticmethod
    def _do_request(url: str, payload: dict, headers: dict) -> dict:
        """Single HTTP POST using stdlib urllib (no external deps)."""
        body = json.dumps(payload).encode()  # 파이썬 dict → JSON → 바이트.
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        safe = BaseLLM._safe_url(url)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())  # 응답 JSON을 dict로 파싱해 반환.
        except urllib.error.HTTPError as exc:
            # 서버가 4xx/5xx로 응답한 경우: 본문을 읽어 어떤 에러인지 메시지에 담는다.
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {safe}: {body_text}") from exc
        except urllib.error.URLError as exc:
            # 연결 자체가 실패한 경우(DNS 실패·연결 거부·타임아웃 등).
            raise RuntimeError(f"Network error connecting to {safe}: {exc.reason}") from exc

    @staticmethod
    def _stream_sse(url: str, payload: dict, headers: dict) -> Iterator[str]:
        """Open an SSE stream; yield each `data:` payload as a raw string."""
        # SSE(Server-Sent Events): 서버가 "data: ..." 줄을 연달아 보내는 스트리밍 방식.
        # Claude/OpenAI/Gemini가 사용한다. 여기서는 'data: ' 접두어를 떼고 본문만 흘려보낸다.
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        safe = BaseLLM._safe_url(url)
        try:
            with urllib.request.urlopen(req) as resp:
                for raw_line in resp:  # 응답을 한 줄씩 읽는다.
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if line.startswith("data: "):
                        yield line[6:]  # "data: " (6글자) 이후의 실제 데이터.
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {safe}: {body_text}") from exc

    @staticmethod
    def _stream_ndjson(url: str, payload: dict, headers: dict) -> Iterator[dict]:
        """Open an NDJSON stream; yield parsed dicts (used by Ollama)."""
        # NDJSON: 한 줄에 JSON 하나씩 오는 스트리밍 방식(Ollama가 사용). 줄마다 파싱해 dict로 반환.
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        safe = BaseLLM._safe_url(url)
        try:
            with urllib.request.urlopen(req) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if line:  # 빈 줄은 건너뛴다.
                        yield json.loads(line)
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {safe}: {body_text}") from exc
