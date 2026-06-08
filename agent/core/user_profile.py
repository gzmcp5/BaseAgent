"""사용자 정보를 기억해 매 대화에 자동으로 주입하는 메모리.

같은 사용자와 여러 번 대화할 때 이름·선호도·알게 된 사실(facts)을 기억해 두면, 매번
다시 설명하지 않아도 LLM이 사용자 맥락을 갖고 답한다. UserProfile은 그 정보를 담는
구조체이고, ProfileMemory는 그것을 SYSTEM 메시지로 변환해 대화 앞에 끼워 넣는 메모리다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .memory import ConversationMemory
from .message import Message, Role


@dataclass
class UserProfile:
    """Structured container for user-specific information."""

    name: str = ""                                            # 사용자 이름.
    preferences: dict[str, Any] = field(default_factory=dict)  # 선호도(예: 언어=한국어).
    facts: list[str] = field(default_factory=list)            # 사용자에 대해 알게 된 사실들.

    def update(self, **kwargs: Any) -> None:
        """Set top-level fields (name, preferences) from keyword args."""
        for key, value in kwargs.items():
            # preferences는 통째로 바꾸지 않고 기존 값에 '병합'한다.
            if key == "preferences" and isinstance(value, dict):
                self.preferences.update(value)
            # 그 외에는 실제 존재하는 필드(name 등)만 설정 → 오타로 엉뚱한 속성 생성 방지.
            elif hasattr(self, key):
                setattr(self, key, value)

    def add_fact(self, fact: str) -> None:
        # 비어 있지 않고 중복도 아닐 때만 사실을 추가.
        if fact and fact not in self.facts:
            self.facts.append(fact)

    def remove_fact(self, fact: str) -> None:
        # 해당 사실을 목록에서 제거(일치하는 것만 걸러냄).
        self.facts = [f for f in self.facts if f != fact]

    def to_context_string(self) -> str:
        # 프로필을 LLM이 읽기 좋은 텍스트 블록으로 변환(채워진 항목만 포함).
        parts: list[str] = []
        if self.name:
            parts.append(f"User name: {self.name}")
        if self.preferences:
            prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
            parts.append(f"Preferences: {prefs}")
        if self.facts:
            bullet_facts = "\n".join(f"- {f}" for f in self.facts)
            parts.append(f"Known facts about the user:\n{bullet_facts}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        # 저장(직렬화)용 딕셔너리로 변환.
        # 가변 컨테이너(list/dict)는 복사해서 넣는다 → 반환된 dict를 바꿔도 내부 상태가
        # 변하지 않고, 반대도 마찬가지(의도치 않은 공유 참조 방지).
        return {
            "name": self.name,
            "preferences": dict(self.preferences),
            "facts": list(self.facts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        # 저장된 딕셔너리에서 프로필을 복원. 키가 없을 때를 대비해 기본값을 둔다.
        # 여기서도 list/dict를 복사해 원본 data와 메모리를 공유하지 않게 한다.
        return cls(
            name=data.get("name", ""),
            preferences=dict(data.get("preferences", {})),
            facts=list(data.get("facts", [])),
        )


class ProfileMemory(ConversationMemory):
    """ConversationMemory with an attached UserProfile.

    The profile is injected as a SYSTEM message (after the main system prompt)
    on every ``get_messages()`` call, so the LLM always has user context.

    Usage::

        mem = ProfileMemory(system_prompt="You are a helpful assistant.")
        mem.profile.name = "Alice"
        mem.profile.add_fact("Prefers concise answers")
        mem.profile.preferences["language"] = "Korean"
        agent = Agent(llm=llm, memory=mem)
    """

    def __init__(
        self,
        max_messages: int = 100,
        system_prompt: str = "",
        profile: Optional[UserProfile] = None,
    ) -> None:
        super().__init__(max_messages=max_messages, system_prompt=system_prompt)
        self.profile: UserProfile = profile or UserProfile()

    # ------------------------------------------------------------------
    # ConversationMemory overrides
    # ------------------------------------------------------------------

    def get_messages(self) -> list[Message]:
        result: list[Message] = []
        if self.system_prompt:
            result.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        # 프로필에 내용이 있을 때만(:= 로 한 번에 만들고 검사) SYSTEM 메시지로 주입한다.
        if profile_ctx := self.profile.to_context_string():
            result.append(
                Message(
                    role=Role.SYSTEM,
                    content=f"[User Profile]\n{profile_ctx}",
                )
            )
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        """Clear conversation history; profile is preserved."""
        super().clear()  # 대화만 비우고 프로필(사용자 정보)은 유지한다.
