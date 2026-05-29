from __future__ import annotations
import random
import time
from dataclasses import dataclass, field


@dataclass
class RetryConfig:
    """Exponential-backoff retry configuration for LLM HTTP calls."""

    max_retries: int = 3
    base_delay: float = 1.0        # seconds before first retry
    max_delay: float = 60.0        # ceiling on backoff delay
    backoff_factor: float = 2.0    # multiplier per attempt
    jitter: bool = True            # add randomness to avoid thundering herd
    # HTTP status codes that warrant a retry
    retryable_codes: tuple[int, ...] = field(
        default_factory=lambda: (429, 500, 502, 503, 504)
    )

    def delay_for(self, attempt: int) -> float:
        """Return sleep duration (seconds) for a given attempt index (0-based)."""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay *= 0.5 + random.random() * 0.5
        return delay

    def is_retryable(self, error_message: str) -> bool:
        return any(f"HTTP {code}" in error_message for code in self.retryable_codes)


# Shared default — providers use this unless overridden
DEFAULT_RETRY = RetryConfig()
