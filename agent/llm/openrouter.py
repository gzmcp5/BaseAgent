from __future__ import annotations
import os
from typing import Optional

from .openai_llm import OpenAILLM

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLM(OpenAILLM):
    """OpenRouter — OpenAI-compatible API supporting 200+ models."""

    DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=_BASE_URL,
            max_tokens=max_tokens,
        )
