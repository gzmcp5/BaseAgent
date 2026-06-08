"""토큰 예산을 초과하면 자동으로 오래된 대화를 '요약·압축'하는 메모리.

LLM은 한 번에 받을 수 있는 토큰(문맥 창)이 정해져 있다. 대화가 길어지면 그 한도를
넘어 비용이 커지거나 호출 자체가 실패한다. ContextManager는 메시지를 추가할 때마다
대략적인 토큰량을 재서, 예산을 넘으면 오래된 메시지들을 한 줄 요약으로 압축하고
최근 몇 개만 원문 그대로 남긴다. ConversationMemory를 상속하므로 Agent(memory=...)에
그대로 끼워 넣을 수 있는 '드롭인 교체' 메모리다.
"""
from __future__ import annotations
from typing import Callable, Optional

from .memory import ConversationMemory
from .message import Message, Role


class ContextManager(ConversationMemory):
    """ConversationMemory extended with token tracking and auto-summarization.

    Drop-in replacement for ConversationMemory — pass it to Agent(memory=...).

    Token counting uses a simple heuristic (4 chars ≈ 1 token).  When the
    estimated context exceeds *max_context_tokens*, old messages are condensed
    into a plain-text digest so the context window stays within budget.

    Args:
        max_context_tokens: Soft limit in approximate tokens.  When exceeded
            on the *next* message add, a summarization pass is triggered.
        summarizer: Callable that receives a list of Messages and returns a
            summary string.  If omitted, excess messages are condensed into a
            plain-text digest (not dropped).
        keep_recent: Number of recent messages always preserved intact.
        system_prompt: Forwarded to the parent ConversationMemory.
        max_messages: Hard message-count cap (from parent).
    """

    # 토큰 수를 어림하는 휴리스틱: 영어 기준 대략 4글자 ≈ 1토큰. (정확한 토크나이저 대신 근사)
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        max_context_tokens: int = 3_000,
        summarizer: Optional[Callable[[list[Message]], str]] = None,
        keep_recent: int = 6,
        system_prompt: str = "",
        max_messages: int = 100,
    ) -> None:
        super().__init__(max_messages=max_messages, system_prompt=system_prompt)
        self.max_context_tokens = max_context_tokens   # 이 토큰 수를 넘으면 압축을 시작.
        self.summarizer = summarizer                   # 요약 함수(없으면 단순 텍스트 다이제스트).
        self.keep_recent = max(2, keep_recent)         # 항상 원문으로 남길 최근 메시지 수(최소 2).

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        super().add(message)          # 평소처럼 메시지를 추가한 뒤,
        if self._over_budget():       # 예산을 넘었으면
            self._compress()          # 오래된 부분을 요약해 압축한다.

    def estimate_tokens(self, messages: Optional[list[Message]] = None) -> int:
        """Return rough token count for *messages* (default: all stored messages)."""
        # 모든 메시지 내용의 글자 수를 더해 CHARS_PER_TOKEN으로 나눈 근사 토큰량.
        msgs = messages if messages is not None else self.get_messages()
        return sum(len(m.content or "") for m in msgs) // self.CHARS_PER_TOKEN

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _over_budget(self) -> bool:
        # 현재 추정 토큰량이 예산을 넘었는지 여부.
        return self.estimate_tokens() > self.max_context_tokens

    def _compress(self) -> None:
        # 예산 아래로 내려가거나 최소 크기(요약 1개 + 최근 keep_recent개)에 닿을 때까지 반복.
        # 최소 크기에서 더 돌리면 같은 구조를 다시 만들어 무한 루프가 되므로 거기서 멈춘다.
        while self._over_budget() and len(self.messages) > self.keep_recent:
            if len(self.messages) == self.keep_recent + 1:
                # 이미 최소 상태 — 최근 메시지만으로도 예산 초과라 더 줄일 수 없다.
                break

            old = self.messages[: -self.keep_recent]    # 요약할 오래된 부분.
            recent = self.messages[-self.keep_recent :] # 원문으로 보존할 최근 부분.

            summary_text = self._summarize(old)
            # 요약문이 예산보다 커지지 않도록 길이를 제한한다(최소 200자는 보장).
            max_summary_chars = max(self.max_context_tokens * self.CHARS_PER_TOKEN, 200)
            summary_text = summary_text[:max_summary_chars]

            # 요약을 한 개의 메시지로 만들어 최근 메시지 앞에 붙인다 → 전체 길이가 줄어든다.
            summary_msg = Message(
                role=Role.ASSISTANT,
                content=f"[Summary of earlier conversation: {summary_text}]",
            )
            self.messages = [summary_msg] + recent

    def _summarize(self, messages: list[Message]) -> str:
        # 사용자가 요약 함수(LLM 등)를 줬으면 그것을 우선 사용.
        if self.summarizer:
            try:
                return self.summarizer(messages)
            except Exception:
                pass  # 요약 함수가 실패하면 아래의 단순 다이제스트로 넘어간다(견고성).
        # 폴백: LLM 호출 없이 각 메시지를 "역할: 앞 120자" 형태로 이어붙인 텍스트 요약.
        lines = []
        for m in messages:
            tag = m.role.value.upper()
            snippet = (m.content or "")[:120].replace("\n", " ")
            lines.append(f"{tag}: {snippet}")
        return " | ".join(lines)
