"""LLM 제공자들의 공개 묶음과 '제공자 이름 → 클래스' 팩토리.

사용자는 ClaudeLLM/OpenAILLM 등을 직접 import 하지 않고, create_llm("claude") 처럼
문자열 이름만으로 알맞은 제공자 인스턴스를 만들 수 있다(설정·환경변수와 잘 어울린다).
"""
from __future__ import annotations
from typing import Any, Optional

from .base import BaseLLM
from .claude import ClaudeLLM
from .google import GoogleLLM
from .ollama import OllamaLLM
from .openai_llm import OpenAILLM
from .openrouter import OpenRouterLLM
from .retry import RetryConfig, DEFAULT_RETRY

# 제공자 이름(소문자) → 해당 LLM 클래스. create_llm이 이 표를 보고 인스턴스를 만든다.
_PROVIDERS: dict[str, type[BaseLLM]] = {
    "claude": ClaudeLLM,
    "openai": OpenAILLM,
    "google": GoogleLLM,
    "ollama": OllamaLLM,
    "openrouter": OpenRouterLLM,
}


def create_llm(
    provider: str,
    model: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLM:
    """Factory function — create an LLM instance by provider name.

    Args:
        provider: One of "claude", "openai", "google", "ollama", "openrouter".
        model: Optional model name; uses the provider's default if omitted.
        **kwargs: Extra arguments forwarded to the provider constructor.
    """
    # 이름으로 제공자 클래스를 찾는다(대소문자 무시).
    cls = _PROVIDERS.get(provider.lower())
    if cls is None:
        # 모르는 이름이면 어떤 값이 가능한지 함께 알려 주는 친절한 에러.
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {available}"
        )
    # 모델을 지정했으면 그 모델로, 아니면 각 제공자의 DEFAULT_MODEL로 생성한다.
    if model is not None:
        return cls(model=model, **kwargs)
    return cls(**kwargs)


__all__ = [
    "BaseLLM",
    "ClaudeLLM",
    "DEFAULT_RETRY",
    "GoogleLLM",
    "OllamaLLM",
    "OpenAILLM",
    "OpenRouterLLM",
    "RetryConfig",
    "create_llm",
]
