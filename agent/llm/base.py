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
        self.model = model
        self.retry_config = retry_config or DEFAULT_RETRY

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages and return a complete response."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield response text chunks (streaming)."""

    # ------------------------------------------------------------------
    # Shared HTTP helpers
    # ------------------------------------------------------------------

    def _request(self, url: str, payload: dict, headers: dict) -> dict:
        """HTTP POST with automatic retry and exponential backoff."""
        cfg = self.retry_config
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(cfg.max_retries + 1):
            try:
                return self._do_request(url, payload, headers)
            except RuntimeError as exc:
                last_exc = exc
                if not cfg.is_retryable(str(exc)) or attempt >= cfg.max_retries:
                    raise
                time.sleep(cfg.delay_for(attempt))

        raise last_exc  # pragma: no cover

    @staticmethod
    def _safe_url(url: str) -> str:
        """Strip query string from URL to avoid leaking secrets in error messages."""
        return url.split("?")[0]

    @staticmethod
    def _do_request(url: str, payload: dict, headers: dict) -> dict:
        """Single HTTP POST using stdlib urllib (no external deps)."""
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        safe = BaseLLM._safe_url(url)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {safe}: {body_text}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error connecting to {safe}: {exc.reason}") from exc

    @staticmethod
    def _stream_sse(url: str, payload: dict, headers: dict) -> Iterator[str]:
        """Open an SSE stream; yield each `data:` payload as a raw string."""
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        safe = BaseLLM._safe_url(url)
        try:
            with urllib.request.urlopen(req) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if line.startswith("data: "):
                        yield line[6:]
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {safe}: {body_text}") from exc

    @staticmethod
    def _stream_ndjson(url: str, payload: dict, headers: dict) -> Iterator[dict]:
        """Open an NDJSON stream; yield parsed dicts (used by Ollama)."""
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        safe = BaseLLM._safe_url(url)
        try:
            with urllib.request.urlopen(req) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {safe}: {body_text}") from exc
