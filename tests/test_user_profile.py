"""Tests for UserProfile and ProfileMemory."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.user_profile import UserProfile, ProfileMemory
from agent.core.message import Message, Role


class TestUserProfile(unittest.TestCase):
    def test_default_empty(self) -> None:
        p = UserProfile()
        self.assertEqual(p.name, "")
        self.assertEqual(p.preferences, {})
        self.assertEqual(p.facts, [])

    def test_update_name(self) -> None:
        p = UserProfile()
        p.update(name="Alice")
        self.assertEqual(p.name, "Alice")

    def test_update_merges_preferences(self) -> None:
        p = UserProfile(preferences={"lang": "en"})
        p.update(preferences={"theme": "dark"})
        self.assertEqual(p.preferences["lang"], "en")
        self.assertEqual(p.preferences["theme"], "dark")

    def test_add_fact_deduplication(self) -> None:
        p = UserProfile()
        p.add_fact("Likes brevity")
        p.add_fact("Likes brevity")
        self.assertEqual(len(p.facts), 1)

    def test_remove_fact(self) -> None:
        p = UserProfile(facts=["fact A", "fact B"])
        p.remove_fact("fact A")
        self.assertNotIn("fact A", p.facts)
        self.assertIn("fact B", p.facts)

    def test_to_context_string_empty(self) -> None:
        self.assertEqual(UserProfile().to_context_string(), "")

    def test_to_context_string_all_fields(self) -> None:
        p = UserProfile(name="Bob", preferences={"lang": "ko"}, facts=["Prefers bullets"])
        ctx = p.to_context_string()
        self.assertIn("Bob", ctx)
        self.assertIn("lang=ko", ctx)
        self.assertIn("Prefers bullets", ctx)

    def test_roundtrip_dict(self) -> None:
        p = UserProfile(name="Carol", preferences={"a": 1}, facts=["x"])
        restored = UserProfile.from_dict(p.to_dict())
        self.assertEqual(restored.name, "Carol")
        self.assertEqual(restored.preferences, {"a": 1})
        self.assertEqual(restored.facts, ["x"])


class TestProfileMemory(unittest.TestCase):
    def test_profile_injected_when_populated(self) -> None:
        mem = ProfileMemory(system_prompt="Be helpful.")
        mem.profile.name = "Dave"
        mem.add(Message(role=Role.USER, content="Hi"))
        msgs = mem.get_messages()
        system_contents = [m.content for m in msgs if m.role == Role.SYSTEM]
        self.assertTrue(any("Dave" in c for c in system_contents))

    def test_profile_not_injected_when_empty(self) -> None:
        mem = ProfileMemory(system_prompt="Sys.")
        mem.add(Message(role=Role.USER, content="Hi"))
        msgs = mem.get_messages()
        system_msgs = [m for m in msgs if m.role == Role.SYSTEM]
        self.assertEqual(len(system_msgs), 1)
        self.assertNotIn("User Profile", system_msgs[0].content)

    def test_system_prompt_appears_before_profile(self) -> None:
        mem = ProfileMemory(system_prompt="Sys.")
        mem.profile.name = "Eve"
        msgs = mem.get_messages()
        self.assertEqual(msgs[0].content, "Sys.")
        self.assertIn("Eve", msgs[1].content)

    def test_clear_preserves_profile(self) -> None:
        mem = ProfileMemory()
        mem.profile.name = "Frank"
        mem.add(Message(role=Role.USER, content="hello"))
        mem.clear()
        self.assertEqual(len(mem.messages), 0)
        self.assertEqual(mem.profile.name, "Frank")

    def test_max_messages_still_respected(self) -> None:
        mem = ProfileMemory(max_messages=3)
        for i in range(5):
            mem.add(Message(role=Role.USER, content=f"m{i}"))
        self.assertEqual(len(mem.messages), 3)

    def test_profile_update_reflected_in_next_call(self) -> None:
        mem = ProfileMemory()
        mem.add(Message(role=Role.USER, content="q"))
        mem.profile.add_fact("Speaks Spanish")
        msgs = mem.get_messages()
        combined = "\n".join(m.content for m in msgs if m.role == Role.SYSTEM)
        self.assertIn("Speaks Spanish", combined)


class TestUserProfileEdgeCases(unittest.TestCase):
    def test_add_empty_string_fact_ignored(self) -> None:
        p = UserProfile()
        p.add_fact("")
        self.assertEqual(len(p.facts), 0)

    def test_update_unknown_key_is_silent_noop(self) -> None:
        p = UserProfile(name="Alice")
        p.update(nonexistent_field="value")
        self.assertEqual(p.name, "Alice")
        self.assertFalse(hasattr(p, "nonexistent_field"))

    def test_update_facts_replaces_list_not_appends(self) -> None:
        # update() uses setattr for list fields — replaces, not appends
        p = UserProfile(facts=["old"])
        p.update(facts=["new"])
        self.assertEqual(p.facts, ["new"])

    def test_update_preferences_merges_not_replaces(self) -> None:
        p = UserProfile(preferences={"a": 1, "b": 2})
        p.update(preferences={"b": 99, "c": 3})
        self.assertEqual(p.preferences["a"], 1)   # original preserved
        self.assertEqual(p.preferences["b"], 99)  # overwritten
        self.assertEqual(p.preferences["c"], 3)   # new key added

    def test_from_dict_missing_fields_uses_defaults(self) -> None:
        p = UserProfile.from_dict({})
        self.assertEqual(p.name, "")
        self.assertEqual(p.preferences, {})
        self.assertEqual(p.facts, [])

    def test_to_context_string_name_only(self) -> None:
        p = UserProfile(name="Only")
        ctx = p.to_context_string()
        self.assertIn("Only", ctx)
        self.assertNotIn("Preferences", ctx)
        self.assertNotIn("facts", ctx.lower())

    def test_to_context_string_facts_only(self) -> None:
        p = UserProfile(facts=["fact one"])
        ctx = p.to_context_string()
        self.assertIn("fact one", ctx)
        self.assertNotIn("User name", ctx)

    def test_remove_nonexistent_fact_is_noop(self) -> None:
        p = UserProfile(facts=["a"])
        p.remove_fact("does not exist")
        self.assertEqual(p.facts, ["a"])

    # --- side-effect / aliasing regression tests ---

    def test_to_dict_does_not_leak_internal_facts(self) -> None:
        p = UserProfile(facts=["f1"])
        d = p.to_dict()
        d["facts"].append("INJECTED")
        self.assertNotIn("INJECTED", p.facts)

    def test_to_dict_does_not_leak_internal_preferences(self) -> None:
        p = UserProfile(preferences={"lang": "ko"})
        d = p.to_dict()
        d["preferences"]["hacked"] = True
        self.assertNotIn("hacked", p.preferences)

    def test_from_dict_does_not_alias_source(self) -> None:
        src = {"name": "B", "preferences": {"x": 1}, "facts": ["g1"]}
        p = UserProfile.from_dict(src)
        p.add_fact("NEW")
        p.preferences["y"] = 2
        self.assertNotIn("NEW", src["facts"])
        self.assertNotIn("y", src["preferences"])

    def test_roundtrip_profiles_are_independent(self) -> None:
        orig = UserProfile(name="C", preferences={"k": 1}, facts=["h1"])
        restored = UserProfile.from_dict(orig.to_dict())
        restored.add_fact("SHARED")
        restored.preferences["k"] = 999
        self.assertNotIn("SHARED", orig.facts)
        self.assertEqual(orig.preferences["k"], 1)


class TestProfileMemoryEdgeCases(unittest.TestCase):
    def test_no_system_prompt_with_profile_injects_only_profile(self) -> None:
        mem = ProfileMemory()
        mem.profile.name = "Ghost"
        msgs = mem.get_messages()
        self.assertEqual(len([m for m in msgs if m.role == Role.SYSTEM]), 1)
        self.assertIn("Ghost", msgs[0].content)

    def test_profile_passed_via_constructor_is_used(self) -> None:
        profile = UserProfile(name="Injected")
        mem = ProfileMemory(profile=profile)
        msgs = mem.get_messages()
        combined = "\n".join(m.content for m in msgs if m.role == Role.SYSTEM)
        self.assertIn("Injected", combined)

    def test_get_messages_empty_state_returns_empty_list(self) -> None:
        mem = ProfileMemory()
        msgs = mem.get_messages()
        self.assertEqual(msgs, [])

    def test_profile_mutation_after_add_reflected_immediately(self) -> None:
        mem = ProfileMemory(system_prompt="S.")
        mem.add(Message(role=Role.USER, content="q"))
        mem.profile.preferences["lang"] = "ko"
        msgs = mem.get_messages()
        combined = "\n".join(m.content for m in msgs if m.role == Role.SYSTEM)
        self.assertIn("lang=ko", combined)


if __name__ == "__main__":
    unittest.main()
