"""OpenRouter 제공자 — 하나의 API로 200개 이상 모델을 쓸 수 있는 중계 서비스.

OpenRouter는 OpenAI와 '완전히 호환'되는 API를 제공한다. 그래서 새로 구현할 게 거의 없고,
OpenAILLM을 상속한 뒤 (1) 엔드포인트 주소와 (2) API 키 환경변수만 바꿔주면 끝난다.
이것이 base_url을 갈아끼울 수 있게 설계해 둔 덕을 보는 대표적인 예다.
"""
from __future__ import annotations
import os
from typing import Optional

from .openai_llm import OpenAILLM

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"  # OpenRouter 엔드포인트.


class OpenRouterLLM(OpenAILLM):
    """OpenRouter — OpenAI-compatible API supporting 200+ models."""

    DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"  # "제공자/모델" 형식으로 모델을 지정.

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        retry_config: Optional["RetryConfig"] = None,
    ) -> None:
        # OpenAI와 다른 환경변수에서 키를 읽는다.
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        # 나머지 동작(chat/stream/형식 변환)은 부모 OpenAILLM을 그대로 재사용, base_url만 교체.
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=_BASE_URL,
            max_tokens=max_tokens,
            retry_config=retry_config,
        )
