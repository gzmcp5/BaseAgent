from __future__ import annotations
from typing import Any, Optional

from .base import BaseLLM
from .claude import ClaudeLLM
from .google import GoogleLLM
from .ollama import OllamaLLM
from .openai_llm import OpenAILLM
from .openrouter import OpenRouterLLM
from .retry import RetryConfig, DEFAULT_RETRY

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
    cls = _PROVIDERS.get(provider.lower())
    if cls is None:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {available}"
        )
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
