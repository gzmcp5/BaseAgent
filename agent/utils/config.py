"""설정값을 한곳에 모아 다루는 가벼운 컨테이너.

API 키·기본 제공자·모델 같은 설정을 환경변수나 YAML 파일에서 읽어와 딕셔너리처럼 보관한다.
값 우선순위는 '직접 지정 > 환경변수 > 기본값'이다.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional


class Config:
    """Simple configuration container.

    Priority: explicit values > environment variables > defaults.
    """

    def __init__(self, data: Optional[dict[str, Any]] = None) -> None:
        self._data: dict[str, Any] = data or {}  # 실제 설정값 저장소.

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Config":
        # 환경변수에서 알려진 설정 키들을 한 번에 읽어 Config를 만든다.
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
        # YAML 설정 파일을 읽는다. PyYAML은 선택적 의존성이라, 없을 때 친절한 안내로 에러를 낸다.
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config: pip install pyyaml"
            ) from exc
        with open(path) as f:
            data = yaml.safe_load(f) or {}  # 빈 파일이면 None이 나오므로 {}로 대체.
        return cls(data)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        # 딕셔너리식 조회. 없으면 default 반환.
        return self._data.get(key, default)

    def __getattr__(self, key: str) -> Any:
        # config.llm_provider 처럼 점(.)으로도 값에 접근할 수 있게 해 준다.
        # 단, __xxx__ 같은 특수 속성은 정상 처리해야 pickle/copy가 깨지지 않는다.
        if key.startswith("__"):
            raise AttributeError(key)
        return self._data.get(key)  # 없는 키는 None.

    def __repr__(self) -> str:
        # 디버깅 출력 시 키 이름에 'key'가 들어간 값(API 키 등)은 ***로 가려 유출을 막는다.
        safe = {k: "***" if "key" in k.lower() else v for k, v in self._data.items()}
        return f"Config({safe})"
