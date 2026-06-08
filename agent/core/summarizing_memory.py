"""오래된 메시지를 '버리지 않고' 요약해 보존하는 메모리.

ConversationMemory는 한도를 넘으면 오래된 메시지를 그냥 삭제한다. SummarizingMemory는
대신 그것들을 요약문으로 압축해 SYSTEM 메시지로 계속 들고 다닌다. 그래서 대화가 길어져도
"앞에서 무슨 얘기를 했는지"의 맥락이 사라지지 않는다.

ContextManager와의 차이: ContextManager는 '토큰 예산' 기준으로 압축하고, 이쪽은
'메시지 개수(max_messages)' 기준으로 압축한다. 또 요약에 실제 LLM을 쓸 수 있다(llm 인자).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .memory import ConversationMemory
from .message import Message, Role

# 타입 검사용으로만 import(실행 시점 순환 import 방지). 런타임에는 불러오지 않는다.
if TYPE_CHECKING:
    from ..llm.base import BaseLLM

# LLM으로 요약할 때 사용하는 시스템 지시문.
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
        self.llm = llm                                              # 요약에 쓸 LLM(없으면 폴백 사용).
        self.summary_keep_last = min(summary_keep_last, max_messages)  # 압축 후 원문으로 남길 최근 개수.
        self.summary: str = ""                                      # 지금까지 누적된 요약문.

    # ------------------------------------------------------------------
    # ConversationMemory overrides
    # ------------------------------------------------------------------

    def add(self, message: Message) -> None:
        self.messages.append(message)
        # 부모와 달리 한도를 넘으면 '삭제'가 아니라 '요약 압축'을 한다.
        if len(self.messages) > self.max_messages:
            self._compress()

    def get_messages(self) -> list[Message]:
        result: list[Message] = []
        # 1) 시스템 프롬프트(있으면) → 2) 누적 요약(있으면) → 3) 최근 원문 메시지 순으로 조립.
        if self.system_prompt:
            result.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        if self.summary:
            # 요약을 SYSTEM 메시지로 끼워 넣어 LLM이 항상 이전 맥락을 보게 한다.
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
        self.summary = ""  # 대화뿐 아니라 누적 요약도 함께 비운다.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress(self) -> None:
        keep = self.summary_keep_last
        if keep:
            # 최근 keep개는 원문으로 남기고, 그보다 오래된 것들만 요약 대상으로.
            to_summarize = self.messages[:-keep]
            self.messages = self.messages[-keep:]
        else:
            # keep이 0이면 전부 요약하고 원문 메시지는 모두 비운다.
            to_summarize = list(self.messages)
            self.messages = []
        new_summary = self._summarize(to_summarize)
        # 기존 요약이 있으면 이어 붙이고(누적), 없으면 새 요약으로 시작.
        if self.summary:
            self.summary = f"{self.summary}\n{new_summary}"
        else:
            self.summary = new_summary

    def _summarize(self, messages: list[Message]) -> str:
        if not messages:
            return ""
        # LLM이 없으면 단순 연결 폴백, 있으면 LLM으로 진짜 요약을 만든다.
        if self.llm is None:
            return self._fallback_summary(messages)
        return self._llm_summary(messages)

    def _llm_summary(self, messages: list[Message]) -> str:
        from .message import LLMResponse  # 함수 안 import로 순환 의존을 피한다.

        # 메시지들을 "역할: 내용" 한 줄씩의 대화록 텍스트로 만든다(SYSTEM·빈 내용 제외).
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
            # LLM 호출이 실패해도 요약 자체가 멈추면 안 되므로 폴백으로 대체한다.
            return self._fallback_summary(messages)

    @staticmethod
    def _fallback_summary(messages: list[Message]) -> str:
        # LLM 없이 메시지들을 "역할: 내용"으로 단순 나열한 요약(맥락 보존이 목적).
        lines = [
            f"{m.role.value}: {m.content}"
            for m in messages
            if m.content and m.role != Role.SYSTEM
        ]
        return "Summary of earlier conversation:\n" + "\n".join(lines)
