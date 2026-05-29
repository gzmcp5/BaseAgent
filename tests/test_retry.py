"""Tests for RetryConfig and retry behaviour in BaseLLM._request()."""
import sys
import os
import unittest
from unittest.mock import patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm.retry import RetryConfig, DEFAULT_RETRY


class TestRetryConfig(unittest.TestCase):
    def test_default_values(self) -> None:
        cfg = RetryConfig()
        self.assertEqual(cfg.max_retries, 3)
        self.assertIn(429, cfg.retryable_codes)

    def test_is_retryable_matches_code(self) -> None:
        cfg = RetryConfig()
        self.assertTrue(cfg.is_retryable("HTTP 429 from https://api.example.com"))
        self.assertTrue(cfg.is_retryable("HTTP 500 from https://api.example.com"))
        self.assertFalse(cfg.is_retryable("HTTP 400 from https://api.example.com"))
        self.assertFalse(cfg.is_retryable("connection refused"))

    def test_delay_increases_with_attempts(self) -> None:
        cfg = RetryConfig(base_delay=1.0, backoff_factor=2.0, jitter=False)
        self.assertAlmostEqual(cfg.delay_for(0), 1.0)
        self.assertAlmostEqual(cfg.delay_for(1), 2.0)
        self.assertAlmostEqual(cfg.delay_for(2), 4.0)

    def test_delay_capped_at_max(self) -> None:
        cfg = RetryConfig(base_delay=10.0, backoff_factor=10.0, max_delay=30.0, jitter=False)
        self.assertLessEqual(cfg.delay_for(5), 30.0)

    def test_jitter_adds_randomness(self) -> None:
        cfg = RetryConfig(base_delay=1.0, backoff_factor=1.0, jitter=True)
        delays = {cfg.delay_for(0) for _ in range(20)}
        self.assertGreater(len(delays), 1)  # should not all be the same


class TestRetryBehaviour(unittest.TestCase):
    def _make_llm(self, retry_config=None):
        """Create a minimal concrete BaseLLM subclass for testing."""
        from agent.llm.base import BaseLLM
        from agent.core.message import LLMResponse, Message

        class StubLLM(BaseLLM):
            def chat(self, messages, tools=None, **kwargs):
                return LLMResponse(content="ok")
            def stream(self, messages, **kwargs):
                return iter([])

        return StubLLM("stub", retry_config=retry_config)

    def test_no_retry_on_success(self) -> None:
        llm = self._make_llm()
        with patch.object(type(llm), "_do_request", return_value={"ok": True}) as mock_req:
            llm._request("http://test", {}, {})
            self.assertEqual(mock_req.call_count, 1)

    def test_retries_on_429(self) -> None:
        llm = self._make_llm(RetryConfig(max_retries=2, base_delay=0, jitter=False))
        attempts = []

        def side_effect(*args, **kwargs):
            attempts.append(1)
            raise RuntimeError("HTTP 429 Too Many Requests")

        with patch.object(type(llm), "_do_request", side_effect=side_effect):
            with patch("agent.llm.base.time.sleep"):
                with self.assertRaises(RuntimeError):
                    llm._request("http://test", {}, {})

        self.assertEqual(len(attempts), 3)  # 1 initial + 2 retries

    def test_no_retry_on_400(self) -> None:
        llm = self._make_llm(RetryConfig(max_retries=3, base_delay=0, jitter=False))
        attempts = []

        def side_effect(*args, **kwargs):
            attempts.append(1)
            raise RuntimeError("HTTP 400 Bad Request")

        with patch.object(type(llm), "_do_request", side_effect=side_effect):
            with self.assertRaises(RuntimeError):
                llm._request("http://test", {}, {})

        self.assertEqual(len(attempts), 1)  # no retry for 400

    def test_succeeds_after_transient_failure(self) -> None:
        llm = self._make_llm(RetryConfig(max_retries=2, base_delay=0, jitter=False))
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("HTTP 503 Service Unavailable")
            return {"result": "ok"}

        with patch.object(type(llm), "_do_request", side_effect=side_effect):
            with patch("agent.llm.base.time.sleep"):
                result = llm._request("http://test", {}, {})

        self.assertEqual(result, {"result": "ok"})
        self.assertEqual(call_count[0], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
