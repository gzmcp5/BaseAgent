"""Tests for ToolRegistry JSON Schema generation."""
import os
import sys
import unittest
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.tool import ToolRegistry


class TestToolSchema(unittest.TestCase):
    def test_basic_types_mapped(self) -> None:
        reg = ToolRegistry()

        def f(a: str, b: int, c: float, d: bool) -> str:
            return "ok"

        reg.register(f, description="d")
        props = reg.get_schemas()[0]["parameters"]["properties"]
        self.assertEqual(props["a"]["type"], "string")
        self.assertEqual(props["b"]["type"], "integer")
        self.assertEqual(props["c"]["type"], "number")
        self.assertEqual(props["d"]["type"], "boolean")

    def test_optional_param_not_required(self) -> None:
        reg = ToolRegistry()

        def f(a: str, b: int = 3) -> str:
            return "ok"

        reg.register(f, description="d")
        schema = reg.get_schemas()[0]["parameters"]
        self.assertEqual(schema["required"], ["a"])
        self.assertIn("b", schema["properties"])

    def test_optional_hint_maps_to_inner_type(self) -> None:
        reg = ToolRegistry()

        def f(a: Optional[int] = None) -> str:
            return "ok"

        reg.register(f, description="d")
        props = reg.get_schemas()[0]["parameters"]["properties"]
        self.assertEqual(props["a"]["type"], "integer")

    def test_list_and_dict_hints(self) -> None:
        reg = ToolRegistry()

        def f(items: list[str], mapping: dict[str, int]) -> str:
            return "ok"

        reg.register(f, description="d")
        props = reg.get_schemas()[0]["parameters"]["properties"]
        self.assertEqual(props["items"]["type"], "array")
        self.assertEqual(props["mapping"]["type"], "object")

    # --- regression: *args / **kwargs must not appear as schema properties ---

    def test_var_positional_and_keyword_excluded(self) -> None:
        reg = ToolRegistry()

        def f(x: int, *args, **kwargs) -> str:
            return "ok"

        reg.register(f, description="d")
        schema = reg.get_schemas()[0]["parameters"]
        self.assertEqual(list(schema["properties"].keys()), ["x"])
        self.assertEqual(schema["required"], ["x"])

    def test_only_varargs_yields_empty_schema(self) -> None:
        reg = ToolRegistry()

        def f(*args, **kwargs) -> str:
            return "ok"

        reg.register(f, description="d")
        schema = reg.get_schemas()[0]["parameters"]
        self.assertEqual(schema["properties"], {})
        self.assertEqual(schema["required"], [])

    def test_execute_still_passes_kwargs_through(self) -> None:
        # Schema hides **kwargs, but execution must still forward them.
        reg = ToolRegistry()

        def f(x: int, **kwargs) -> dict:
            return {"x": x, "extra": kwargs}

        reg.register(f, description="d")
        out = reg.get("f").execute(x=1, y=2)
        self.assertEqual(out, {"x": 1, "extra": {"y": 2}})


class TestRequiresApproval(unittest.TestCase):
    def test_defaults_to_false(self) -> None:
        reg = ToolRegistry()

        def f() -> str:
            return "ok"

        reg.register(f, description="d")
        self.assertFalse(reg.get("f").requires_approval)

    def test_flag_persisted_on_tool(self) -> None:
        reg = ToolRegistry()

        def danger() -> str:
            return "boom"

        reg.register(danger, description="Deletes everything.", requires_approval=True)
        self.assertTrue(reg.get("danger").requires_approval)

    def test_decorator_form_with_approval(self) -> None:
        reg = ToolRegistry()

        @reg.register(description="Drops the DB.", requires_approval=True)
        def drop_db() -> str:
            return "dropped"

        self.assertTrue(reg.get("drop_db").requires_approval)
        # Still executable directly (gating happens in the Agent layer).
        self.assertEqual(reg.get("drop_db").execute(), "dropped")


if __name__ == "__main__":
    unittest.main()
