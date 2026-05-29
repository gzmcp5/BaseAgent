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
        self.max_context_tokens = max_context_tokens
        self.summarizer = summarizer
        self.keep_recent = max(2, keep_recent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        super().add(message)
        if self._over_budget():
            self._compress()

    def estimate_tokens(self, messages: Optional[list[Message]] = None) -> int:
        """Return rough token count for *messages* (default: all stored messages)."""
        msgs = messages if messages is not None else self.get_messages()
        return sum(len(m.content or "") for m in msgs) // self.CHARS_PER_TOKEN

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _over_budget(self) -> bool:
        return self.estimate_tokens() > self.max_context_tokens

    def _compress(self) -> None:
        # Repeat until under budget or we reach the minimum size (1 summary + keep_recent).
        # At minimum size further iteration would regenerate the same layout → infinite loop.
        while self._over_budget() and len(self.messages) > self.keep_recent:
            if len(self.messages) == self.keep_recent + 1:
                # Already at minimum — recent messages alone exceed budget; can't do more.
                break

            old = self.messages[: -self.keep_recent]
            recent = self.messages[-self.keep_recent :]

            summary_text = self._summarize(old)
            # Cap summary length so it can't grow larger than the token budget
            max_summary_chars = max(self.max_context_tokens * self.CHARS_PER_TOKEN, 200)
            summary_text = summary_text[:max_summary_chars]

            summary_msg = Message(
                role=Role.ASSISTANT,
                content=f"[Summary of earlier conversation: {summary_text}]",
            )
            self.messages = [summary_msg] + recent

    def _summarize(self, messages: list[Message]) -> str:
        if self.summarizer:
            try:
                return self.summarizer(messages)
            except Exception:
                pass  # fall through to plain-text digest
        # Fallback: build a plain-text digest without an LLM call
        lines = []
        for m in messages:
            tag = m.role.value.upper()
            snippet = (m.content or "")[:120].replace("\n", " ")
            lines.append(f"{tag}: {snippet}")
        return " | ".join(lines)
