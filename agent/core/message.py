"""대화에서 오가는 데이터의 '공통 형식'을 정의하는 모듈.

이 프레임워크는 Claude, OpenAI, Gemini 등 여러 LLM 제공자를 지원하는데,
제공자마다 API가 요구하는 JSON 형식이 제각각이다. 그래서 내부에서는 항상
여기 정의된 표준 형식(Message / ToolCall / LLMResponse)으로만 데이터를 주고받고,
각 제공자 어댑터(agent/llm/*.py)가 마지막에 제공자별 형식으로 변환한다.
즉, 이 파일은 프레임워크 전체의 '공용어' 역할을 한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(str, Enum):
    """메시지를 보낸 주체(역할)를 나타내는 열거형.

    str을 함께 상속하므로 Role.USER == "user" 처럼 문자열과 바로 비교·직렬화된다.
    """

    SYSTEM = "system"        # 시스템 지시문(에이전트의 성격·규칙). 보통 대화 맨 앞 1개.
    USER = "user"            # 사용자가 입력한 메시지.
    ASSISTANT = "assistant"  # LLM(에이전트)이 생성한 응답.
    TOOL = "tool"            # 도구 실행 결과를 LLM에게 되돌려줄 때 쓰는 역할.


@dataclass
class ToolCall:
    """LLM이 "이 도구를 이런 인자로 실행해 달라"고 요청한 한 건의 호출."""

    id: str                      # 호출 식별자. 도구 결과를 이 호출과 짝지을 때 사용.
    name: str                    # 실행할 도구 이름(ToolRegistry에 등록된 이름).
    arguments: dict[str, Any]    # 도구에 넘길 인자들(키워드 인자 형태).


@dataclass
class Message:
    """대화 기록의 한 줄. 메모리에 차곡차곡 쌓이는 기본 단위다."""

    role: Role                                  # 누가 보낸 메시지인지(위 Role 참고).
    content: str                                # 실제 텍스트 내용.
    # 아래 세 필드는 도구 사용과 관련될 때만 채워지고, 평범한 대화에서는 None이다.
    tool_calls: Optional[list[ToolCall]] = None  # ASSISTANT가 요청한 도구 호출 목록.
    tool_call_id: Optional[str] = None           # TOOL 메시지가 어떤 호출의 결과인지.
    name: Optional[str] = None                   # 결과를 만든 도구의 이름.


@dataclass
class LLMResponse:
    """LLM 한 번의 호출에 대한 표준 응답 형태.

    제공자별 응답 JSON을 파싱해 항상 이 형태로 통일한다(agent/llm/*.py 참고).
    """

    content: str                                            # 응답 텍스트.
    tool_calls: list[ToolCall] = field(default_factory=list)  # 도구 호출 요청(없으면 빈 리스트).
    model: str = ""                                          # 실제 응답한 모델명.
    usage: dict[str, int] = field(default_factory=dict)      # 토큰 사용량 등 메타정보.
