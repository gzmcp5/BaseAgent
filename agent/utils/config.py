from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional


class Config:
    """Simple configuration container.

    Priority: explicit values > environment variables > defaults.
    """

    def __init__(self, data: Optional[dict[str, Any]] = None) -> None:
        self._data: dict[str, Any] = data or {}

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            {
                "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
                "openai_api_key": os.environ.get("OPENAI_API_KEY"),
                "google_api_key": os.environ.get("GOOGLE_API_KEY"),
                "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY"),
                "ollama_base_url": os.environ.get(
                    "OLLAMA_BASE_URL", "http://localhost:11434"
                ),
                "llm_provider": os.environ.get("LLM_PROVIDER", "claude"),
                "llm_model": os.environ.get("LLM_MODEL"),
            }
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config: pip install pyyaml"
            ) from exc
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            return None

    def __repr__(self) -> str:
        safe = {k: "***" if "key" in k.lower() else v for k, v in self._data.items()}
        return f"Config({safe})"
