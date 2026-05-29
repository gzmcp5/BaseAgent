from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

from ..core.message import LLMResponse, Message


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, model: str) -> None:
        self.model = model

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

    def _request(self, url: str, payload: dict, headers: dict) -> dict:
        """Shared HTTP POST using stdlib urllib (no external deps)."""
        import json
        import urllib.error
        import urllib.request

        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"HTTP {exc.code} from {url}: {body_text}"
            ) from exc
