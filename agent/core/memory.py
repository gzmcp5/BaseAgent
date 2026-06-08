"""대화 기록(메모리)의 가장 기본 구현.

에이전트는 매 턴마다 "지금까지의 대화 전체"를 LLM에 다시 보내야 한다(LLM은 상태가
없으므로). 그 기록을 보관하는 곳이 메모리다. 이 ConversationMemory가 기본형이고,
ContextManager·SummarizingMemory·ProfileMemory·PersistentMemory가 모두 이 클래스를
상속해 동작을 바꾼다. Agent(memory=...) 에 무엇을 넣느냐에 따라 메모리 전략이 교체된다.
"""
from __future__ import annotations
from .message import Message, Role


class ConversationMemory:
    """Stores and manages conversation history."""

    def __init__(self, max_messages: int = 100, system_prompt: str = "") -> None:
        # 실제 대화 메시지들(시스템 프롬프트는 여기 넣지 않고 get_messages에서 앞에 붙인다).
        self.messages: list[Message] = []
        # 보관할 최대 메시지 수. 초과하면 가장 오래된 것부터 버려 메모리 폭주를 막는다.
        self.max_messages = max_messages
        # 에이전트의 성격·규칙을 담는 고정 지시문. 매 호출 맨 앞에 항상 따라붙는다.
        self.system_prompt = system_prompt

    def add(self, message: Message) -> None:
        # 새 메시지를 뒤에 추가하고,
        self.messages.append(message)
        # 개수가 한도를 넘으면 뒤쪽(최신) max_messages개만 남긴다 = 오래된 것 폐기.
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_messages(self) -> list[Message]:
        # LLM에 보낼 최종 메시지 목록을 만든다.
        result: list[Message] = []
        # 시스템 프롬프트가 있으면 항상 맨 앞에 SYSTEM 메시지로 끼워 넣는다.
        if self.system_prompt:
            result.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        # 그 뒤에 실제 대화 기록을 이어 붙인다.
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        # 대화 기록만 비운다. system_prompt(에이전트 성격)는 그대로 유지된다.
        self.messages.clear()

    def __len__(self) -> int:
        # len(memory) 로 저장된 메시지 개수를 알 수 있게 한다.
        return len(self.messages)
