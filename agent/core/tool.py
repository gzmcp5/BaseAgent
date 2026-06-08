"""도구(Tool) 정의와 등록을 담당하는 모듈.

'도구'란 LLM이 글만 쓰는 대신 실제 행동(시간 조회, 파일 읽기, 계산 등)을 하도록
연결해 주는 일반 파이썬 함수다. 핵심 아이디어:
  - 개발자는 그냥 함수를 작성하고 @registry.register 로 표시만 한다.
  - 그러면 이 모듈이 함수의 '타입 힌트'를 읽어 LLM이 이해하는 JSON Schema를 자동 생성한다.
    (예: def add(a: int, b: int) → {"a": integer, "b": integer} 스키마)
  - LLM은 그 스키마를 보고 "이 도구를 이런 인자로 호출하라"고 결정한다.
즉 파이썬 함수 시그니처가 곧 LLM용 도구 설명서가 되도록 자동화한 것이 이 파일의 역할이다.
"""
from __future__ import annotations
import functools
import inspect
import typing
from typing import Any, Callable, Optional


# 파이썬 타입 → JSON Schema 타입 문자열 대응표. LLM API는 JSON Schema 용어를 쓴다.
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
    # 먼저 str/int 같은 단순 타입을 표에서 바로 찾는다.
    if hint in _TYPE_MAP:
        return _TYPE_MAP[hint]

    # list[str], Optional[int] 같은 '제네릭' 타입은 origin(바깥 틀)을 꺼내 분석한다.
    origin = typing.get_origin(hint)
    if origin is not None:
        # Optional[X]는 사실 Union[X, None]이다 → None을 빼고 남은 실제 타입으로 재귀 판단.
        if origin is typing.Union:
            non_none = [a for a in typing.get_args(hint) if a is not type(None)]
            if non_none:
                return _hint_to_json_type(non_none[0])
            return "string"
        # list[X], List[X] → JSON의 배열.
        if origin in (list,):
            return "array"
        # dict[K, V], Dict[K, V] → JSON의 객체.
        if origin in (dict,):
            return "object"

    return "string"  # 알 수 없는 타입은 일단 문자열로 처리(안전한 기본값).


class Tool:
    """등록된 도구 하나를 감싸는 객체 — 실행할 함수와 LLM용 설명서를 함께 들고 있다."""

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: dict[str, Any],
        requires_approval: bool = False,
    ) -> None:
        self.func = func                  # 실제로 실행될 파이썬 함수.
        self.name = name                  # LLM이 부를 때 쓰는 도구 이름.
        self.description = description     # 이 도구가 무엇을 하는지 설명(LLM이 선택 근거로 삼음).
        self.parameters = parameters      # 자동 생성된 JSON Schema(인자 형식).
        # When True, the agent gates execution behind a human-in-the-loop
        # approval callback (see Agent.approval_callback).
        # True면 실행 전에 사람의 승인을 받아야 한다(파일 삭제 등 위험한 도구에 사용).
        self.requires_approval = requires_approval

    def execute(self, **kwargs: Any) -> Any:
        # 실제 함수를 키워드 인자로 호출. (LLM이 준 arguments가 그대로 들어온다.)
        return self.func(**kwargs)

    def to_schema(self) -> dict[str, Any]:
        # LLM API에 보낼 도구 설명서(딕셔너리) 형태로 변환.
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Decorator-based tool registry with automatic JSON Schema generation."""

    def __init__(self) -> None:
        # 도구 이름 → Tool 객체 매핑. 같은 이름으로 다시 등록하면 덮어쓴다.
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        requires_approval: bool = False,
    ) -> Any:
        """Register a function as a callable tool.

        Can be used as @registry.register or
        @registry.register(name="...", description="...").

        Set ``requires_approval=True`` for sensitive tools (file deletion,
        DB writes, …); the agent will then ask a human-in-the-loop callback
        for confirmation before each execution.
        """
        # 데코레이터를 인자와 함께 쓴 경우(@register(name=...))엔 func가 아직 안 넘어온다.
        # 이때는 partial을 돌려줘서 "다음에 함수가 오면 그 인자들로 등록"하도록 한다.
        if func is None:
            return functools.partial(
                self.register,
                name=name,
                description=description,
                requires_approval=requires_approval,
            )

        # 이름을 안 주면 함수 이름을, 설명을 안 주면 함수의 docstring을 그대로 쓴다.
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()
        # 함수 시그니처를 분석해 JSON Schema(인자 형식)를 자동 생성.
        parameters = self._build_parameters(func)

        self._tools[tool_name] = Tool(
            func, tool_name, tool_desc, parameters, requires_approval
        )
        return func  # 원본 함수를 그대로 반환 → 데코레이터로 감싸도 함수는 정상 사용 가능.

    def _build_parameters(self, func: Callable) -> dict[str, Any]:
        # 함수의 매개변수 목록과 타입 힌트를 읽어온다.
        sig = inspect.signature(func)
        hints = typing.get_type_hints(func)  # 문자열 형태의 어노테이션도 실제 타입으로 변환.
        properties: dict[str, Any] = {}      # 각 인자의 타입 정보.
        required: list[str] = []             # 기본값이 없는 = 필수 인자 목록.

        for param_name, param in sig.parameters.items():
            if param_name == "self":  # 메서드의 self는 도구 인자가 아니므로 건너뜀.
                continue
            # *args / **kwargs는 이름 있는 JSON Schema 속성으로 표현할 수 없으니 제외한다
            # (억지로 넣으면 엉뚱한 필수 필드가 생긴다).
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            hint = hints.get(param_name, str)          # 힌트가 없으면 문자열로 가정.
            json_type = _hint_to_json_type(hint)       # 파이썬 타입 → JSON 타입.
            properties[param_name] = {"type": json_type}
            # 기본값이 없으면(=empty) 호출 시 반드시 채워야 하는 필수 인자.
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}

    def get(self, name: str) -> Optional[Tool]:
        # 이름으로 도구를 찾는다. 없으면 None.
        return self._tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        # 등록된 모든 도구의 LLM용 스키마 목록(LLM에 통째로 보낼 때 사용).
        return [tool.to_schema() for tool in self._tools.values()]

    def all(self) -> dict[str, Tool]:
        # 내부 딕셔너리의 '복사본'을 반환 → 외부에서 실수로 원본을 건드리지 못하게 함.
        return dict(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
