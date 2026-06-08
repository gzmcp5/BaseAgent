"""LLM HTTP 호출 실패 시 '재시도 정책'을 정의하는 모듈.

네트워크 호출은 일시적으로 실패할 수 있다(서버 과부하 429, 일시 장애 503 등). 이럴 때
바로 포기하지 않고, 점점 더 긴 간격으로 다시 시도하면 대개 성공한다. 이 '지수 백오프'
전략의 설정값들을 RetryConfig가 담는다. base.py의 _request가 이 설정을 보고 동작한다.
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field


@dataclass
class RetryConfig:
    """Exponential-backoff retry configuration for LLM HTTP calls."""

    max_retries: int = 3           # 최대 재시도 횟수.
    base_delay: float = 1.0        # seconds before first retry  (첫 재시도 전 기본 대기 초)
    max_delay: float = 60.0        # ceiling on backoff delay    (대기 시간 상한)
    backoff_factor: float = 2.0    # multiplier per attempt      (시도마다 곱해지는 배수)
    jitter: bool = True            # add randomness to avoid thundering herd
                                   # (여러 클라이언트가 동시에 몰려 재요청하는 현상 방지용 무작위성)
    # 재시도할 가치가 있는 HTTP 상태 코드들(과부하·일시 장애 계열).
    retryable_codes: tuple[int, ...] = field(
        default_factory=lambda: (429, 500, 502, 503, 504)
    )

    def delay_for(self, attempt: int) -> float:
        """Return sleep duration (seconds) for a given attempt index (0-based)."""
        # base_delay × (backoff_factor ^ 시도횟수) 로 점점 길어지되, max_delay를 넘지 않게 한다.
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            # 계산된 대기의 50~100% 사이로 흔들어, 동시 재시도가 한 시점에 겹치지 않게 한다.
            delay *= 0.5 + random.random() * 0.5
        return delay

    def is_retryable(self, error_message: str) -> bool:
        # 에러 메시지에 재시도 대상 상태코드가 들어 있으면 재시도 대상.
        if any(f"HTTP {code}" in error_message for code in self.retryable_codes):
            return True
        # 일시적 네트워크 오류(DNS 실패·연결 거부·타임아웃)도 재시도 대상.
        return "Network error" in error_message


# Shared default — providers use this unless overridden
# 공용 기본 정책. 각 제공자는 따로 지정하지 않으면 이것을 쓴다.
DEFAULT_RETRY = RetryConfig()
