from __future__ import annotations
import functools
import inspect
import typing
from typing import Any, Callable, Optional


_TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _hint_to_json_type(hint: Any) -> str:
    """Convert a Python type annotation to a JSON Schema type string.

    Handles bare builtins, generic aliases (list[str], dict[str, Any]),
    and Optional[X] / Union[X, None].
    """
    # Direct match first
    if hint in _TYPE_MAP:
        return _TYPE_MAP[hint]

    origin = typing.get_origin(hint)
    if origin is not None:
        # Optional[X] == Union[X, None]
        if origin is typing.Union:
            non_none = [a for a in typing.get_args(hint) if a is not type(None)]
            if non_none:
                return _hint_to_json_type(non_none[0])
            return "string"
        # list[X], List[X]
        if origin in (list,):
            return "array"
        # dict[K, V], Dict[K, V]
        if origin in (dict,):
            return "object"

    return "string"  # safe default for unrecognised types


class Tool:
    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self.func = func
        self.name = name
        self.description = description
        self.parameters = parameters

    def execute(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Decorator-based tool registry with automatic JSON Schema generation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Any:
        """Register a function as a callable tool.

        Can be used as @registry.register or
        @registry.register(name="...", description="...").
        """
        if func is None:
            return functools.partial(self.register, name=name, description=description)

        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()
        parameters = self._build_parameters(func)

        self._tools[tool_name] = Tool(func, tool_name, tool_desc, parameters)
        return func

    def _build_parameters(self, func: Callable) -> dict[str, Any]:
        sig = inspect.signature(func)
        hints = typing.get_type_hints(func)  # resolves string annotations
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            hint = hints.get(param_name, str)
            json_type = _hint_to_json_type(hint)
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def all(self) -> dict[str, Tool]:
        return dict(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
