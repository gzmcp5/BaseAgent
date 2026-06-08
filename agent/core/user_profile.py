from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .memory import ConversationMemory
from .message import Message, Role


@dataclass
class UserProfile:
    """Structured container for user-specific information."""

    name: str = ""
    preferences: dict[str, Any] = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)

    def update(self, **kwargs: Any) -> None:
        """Set top-level fields (name, preferences) from keyword args."""
        for key, value in kwargs.items():
            if key == "preferences" and isinstance(value, dict):
                self.preferences.update(value)
            elif hasattr(self, key):
                setattr(self, key, value)

    def add_fact(self, fact: str) -> None:
        if fact and fact not in self.facts:
            self.facts.append(fact)

    def remove_fact(self, fact: str) -> None:
        self.facts = [f for f in self.facts if f != fact]

    def to_context_string(self) -> str:
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
        # Copy mutable containers so callers cannot mutate internal state
        # through the returned dict (and vice-versa).
        return {
            "name": self.name,
            "preferences": dict(self.preferences),
            "facts": list(self.facts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        # Copy mutable containers so the restored profile does not alias the
        # source dict's lists/dicts.
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
        super().clear()
