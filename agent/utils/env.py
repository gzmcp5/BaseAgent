"""외부 라이브러리 없이 .env 파일을 읽어 환경변수로 올리는 최소 구현.

python-dotenv 패키지를 설치하지 않고도 API 키 등을 .env 파일에 적어두고 쓰게 해 준다.
표준 라이브러리만으로 KEY=VALUE 형식을 직접 파싱한다.
"""
from __future__ import annotations
import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader — no external libraries required.

    Supports:
    - KEY=VALUE
    - KEY="VALUE" / KEY='VALUE'
    - # comments
    - export KEY=VALUE
    """
    env_path = Path(path)
    if not env_path.exists():
        return  # .env가 없으면 조용히 넘어간다(필수가 아니므로).

    with env_path.open() as f:
        for line in f:
            line = line.strip()
            # 빈 줄과 주석(#)은 건너뛴다.
            if not line or line.startswith("#"):
                continue
            # 셸 습관으로 붙는 'export ' 접두어가 있으면 떼어 낸다.
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue  # KEY=VALUE 형태가 아니면 무시.
            # 첫 '='를 기준으로 키와 값을 나눈다(값 안에 '='가 더 있어도 안전).
            key, _, raw_value = line.partition("=")
            key = key.strip()
            value = raw_value.strip()
            # 값을 감싼 따옴표("..." 또는 '...')가 있으면 제거.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            # setdefault: 이미 설정된 환경변수는 덮어쓰지 않는다(실제 환경값을 우선).
            os.environ.setdefault(key, value)
