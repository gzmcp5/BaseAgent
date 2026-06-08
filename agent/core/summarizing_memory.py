from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .memory import ConversationMemory
from .message import Message, Role

if TYPE_CHECKING:
    from ..llm.base import BaseLLM

_SUMMARIZE_PROMPT = (
    "Summarize the following conversation into a concise paragraph. "
    "Preserve names, decisions, and key facts. Be brief."
)


class SummarizingMemory(ConversationMemory):
    """ConversationMemory that summarizes old messages instead of dropping them.

    When the message buffer exceeds *max_messages*, the oldest
    ``max_messages - summary_keep_last`` messages are condensed into a
    summary string.  The summary is injected as a SYSTEM message in
    ``get_messages()`` so the LLM always sees prior context.

    Args:
        max_messages: Trigger threshold.  When exceeded, summarization fires.
        system_prompt: Permanent system instruction.
        llm: Optional LLM used for summarization.  When *None* a simple
            concatenation fallback is used.
        summary_keep_last: How many recent messages to retain verbatim after
            each summarization pass.
    """

    def __init__(
        self,
        max_messages: int = 100,
        system_prompt: str = "",
        llm: Optional[BaseLLM] = None,
        summary_keep_last: int = 20,
    ) -> None:
        super().__init__(max_messages=max_messages, system_prompt=system_prompt)
        self.llm = llm
        self.summary_keep_last = min(summary_keep_last, max_messages)
        self.summary: str = ""

    # ------------------------------------------------------------------
    # ConversationMemory overrides
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self._compress()

    def get_messages(self) -> list[Message]:
        result: list[Message] = []
        if self.system_prompt:
            result.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        if self.summary:
            result.append(
                Message(
                    role=Role.SYSTEM,
                    content=f"[Earlier conversation summary]\n{self.summary}",
                )
            )
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        super().clear()
        self.summary = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress(self) -> None:
        keep = self.summary_keep_last
        if keep:
            to_summarize = self.messages[:-keep]
            self.messages = self.messages[-keep:]
        else:
            to_summarize = list(self.messages)
            self.messages = []
        new_summary = self._summarize(to_summarize)
        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"
        else:
            self.summary = new_summary

    def _summarize(self, messages: list[Message]) -> str:
        if not messages:
            return ""
        if self.llm is None:
            return self._fallback_summary(messages)
        return self._llm_summary(messages)

    def _llm_summary(self, messages: list[Message]) -> str:
        from .message import LLMResponse  # local import avoids circular dep

        transcript = "\n".join(
            f"{m.role.value}: {m.content}"
            for m in messages
            if m.content and m.role != Role.SYSTEM
        )
        prompt: list[Message] = [
            Message(role=Role.SYSTEM, content=_SUMMARIZE_PROMPT),
            Message(role=Role.USER, content=transcript),
        ]
        try:
            response = self.llm.chat(prompt)  # type: ignore[union-attr]
            return response.content.strip()
        except Exception:
            return self._fallback_summary(messages)

    @staticmethod
    def _fallback_summary(messages: list[Message]) -> str:
        lines = [
            f"{m.role.value}: {m.content}"
            for m in messages
            if m.content and m.role != Role.SYSTEM
        ]
        return "Summary of earlier conversation:\n" + "\n".join(lines)
