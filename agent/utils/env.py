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
        return

    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key = key.strip()
            value = raw_value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)
