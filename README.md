# BaseAgent

Python으로 작성된 멀티 LLM 제공자 지원 AI 에이전트 베이스 프레임워크.  
외부 라이브러리 없이 순수 stdlib만 사용하며, 추후 보안 기능(`SECURITY_FEATURE.md`) 구현을 위한 확장 포인트가 설계되어 있습니다.

---

## 목차

- [특징](#특징)
- [요구사항](#요구사항)
- [빠른 시작](#빠른-시작)
- [프로젝트 구조](#프로젝트-구조)
- [핵심 개념](#핵심-개념)
  - [Agent](#agent)
  - [LLM 제공자](#llm-제공자)
  - [Tool 등록](#tool-등록)
  - [Middleware / Hooks](#middleware--hooks)
  - [Streaming](#streaming)
  - [AsyncAgent](#asyncagent)
  - [Context 관리](#context-관리)
  - [PersistentMemory](#persistentmemory)
  - [Retry / Rate Limit](#retry--rate-limit)
- [설정](#설정)
- [예제 실행](#예제-실행)
- [테스트](#테스트)
- [보안 기능 확장 계획](#보안-기능-확장-계획)

---

## 특징

| 항목 | 내용 |
|------|------|
| **외부 의존성 없음** | HTTP는 stdlib `urllib`만 사용 — `pip install` 불필요 |
| **5개 LLM 제공자** | Claude, OpenAI, Google Gemini, Ollama(로컬), OpenRouter |
| **Tool / Function Calling** | 데코레이터 기반 등록, JSON Schema 자동 생성, 제네릭 타입 지원 |
| **Middleware Hooks** | LLM 호출 전후, 툴 실행 전후 4개 이벤트 인터셉트 |
| **스트리밍** | SSE(Claude·OpenAI·Google·OpenRouter), NDJSON(Ollama) |
| **AsyncAgent** | asyncio + ThreadPoolExecutor, 동일 턴 툴들 동시 실행 |
| **Context 관리** | 토큰 초과 시 자동 요약 압축, 커스텀 요약 함수 지원 |
| **PersistentMemory** | SQLite 기반 영속 메모리 — 프로세스 재시작 후에도 대화 히스토리 유지, 세션 resume 지원 |
| **Retry** | 지수 백오프 + 지터, HTTP 4xx/5xx 및 네트워크 오류 재시도 |
| **Python 3.10+** | `|` union 문법, `typing.get_origin` 등 최신 기능 활용 |

---

## 요구사항

- Python 3.10 이상
- 사용할 LLM 제공자의 API 키 (Ollama는 로컬 실행이므로 불필요)

```bash
# 선택: YAML 설정 파일 로드 시에만 필요
pip install pyyaml
```

---

## 빠른 시작

### 1. 저장소 클론 및 환경 설정

```bash
git clone <repo-url>
cd BaseAgent
cp .env.example .env
# 선택: .env 파일에 실제 API 키 입력
```

### 2. `.env` 파일 편집 (선택)

`.env.example`은 커밋 가능한 템플릿이고, `.env`는 실제 실행용 로컬 설정입니다. CLI에서 provider를 선택할 때 API 키를 직접 입력할 수도 있지만, 반복 사용한다면 `.env`에 저장해 두면 편합니다.

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. 대화형 CLI 실행

```bash
python main.py
```

```
Base Agent  |  provider=claude  model=claude-sonnet-4-6
Commands: /reset  /provider  /quit

> 안녕하세요!
Agent: 안녕하세요! 무엇을 도와드릴까요?
```

`/`를 입력하면 커맨드 후보가 표시됩니다. 글자를 계속 입력해 후보를 필터링하고, 방향키와 Enter로 선택할 수 있습니다.

| 커맨드 | 설명 |
|--------|------|
| `/provider` | provider와 model을 방향키로 선택합니다. API 키가 필요하고 환경변수에 없으면 직접 입력받습니다. |
| `/reset` | 대화 히스토리를 초기화합니다. |
| `/quit` | CLI를 종료합니다. |

`/provider`에서 선택한 provider/model은 `.baseagent.json`에 저장되어 다음 실행에도 유지됩니다. API 키는 저장하지 않습니다. `LLM_PROVIDER` 또는 `LLM_MODEL` 환경변수를 지정하면 저장된 값보다 우선합니다.

Ollama를 선택하면 로컬 Ollama 서버의 `/api/tags`에서 설치된 모델 목록을 가져와 선택할 수 있습니다. Ollama가 실행 중이 아니거나 모델 목록을 가져오지 못하면 모델명을 직접 입력합니다.

### 4. 코드에서 직접 사용

```python
from agent import Agent, create_llm

llm = create_llm("claude")
agent = Agent(llm=llm, system_prompt="You are a helpful assistant.")

response = agent.run("What is the capital of France?")
print(response)  # Paris is the capital of France.
```

---

## 프로젝트 구조

```
BaseAgent/
├── agent/
│   ├── core/
│   │   ├── agent.py          # 메인 에이전트 오케스트레이터
│   │   ├── async_agent.py    # 비동기 에이전트 (asyncio)
│   │   ├── context.py        # 토큰 예산 관리 + 자동 요약
│   │   ├── hooks.py          # 미들웨어 훅 레지스트리
│   │   ├── memory.py         # 대화 히스토리 관리
│   │   ├── message.py        # Message, Role, ToolCall, LLMResponse 데이터클래스
│   │   ├── persistent_memory.py  # SQLite 영속 메모리 (세션 resume 지원)
│   │   └── tool.py           # Tool 등록 및 JSON Schema 생성
│   ├── llm/
│   │   ├── base.py           # 추상 BaseLLM + 공유 HTTP 유틸
│   │   ├── claude.py         # Anthropic Claude
│   │   ├── openai_llm.py     # OpenAI
│   │   ├── google.py         # Google Gemini
│   │   ├── ollama.py         # Ollama (로컬)
│   │   ├── openrouter.py     # OpenRouter
│   │   └── retry.py          # RetryConfig 데이터클래스
│   └── utils/
│       ├── config.py         # Config (환경변수 / YAML)
│       └── env.py            # stdlib 기반 .env 로더
├── examples/
│   ├── basic_chat.py              # 기본 멀티턴 대화
│   ├── tool_use_demo.py           # 툴 호출 예제
│   ├── streaming_demo.py          # 스트리밍 출력
│   ├── hooks_demo.py              # 미들웨어 훅 예제
│   ├── async_demo.py              # 비동기 + 동시 툴 실행
│   └── persistent_memory_demo.py  # SQLite 영속 메모리 (new/resume/list)
├── tests/
│   ├── test_agent.py
│   ├── test_async_agent.py
│   ├── test_context.py
│   ├── test_hooks.py
│   ├── test_persistent_memory.py
│   └── test_retry.py
├── main.py                   # CLI 진입점
├── .env.example
├── .baseagent.json           # CLI provider/model 저장 파일 (gitignore)
└── SECURITY_FEATURE.md       # 추후 구현 예정 보안 기능 명세
```

---

## 핵심 개념

### Agent

`Agent`는 LLM 호출 → 툴 실행 → 메모리 저장의 루프를 관리합니다.

```python
from agent import Agent, create_llm, ToolRegistry

llm = create_llm("openai", model="gpt-4o")

tools = ToolRegistry()

@tools.register(description="두 정수를 더합니다.")
def add(a: int, b: int) -> int:
    return a + b

agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    tools=tools,
    max_tool_iterations=10,   # 툴 호출 최대 반복 횟수
)

# 단일 턴 실행 (메모리에 대화 누적)
response = agent.run("What is 5 + 3?")

# 히스토리 초기화
agent.reset()
```

**`Agent` 생성자 파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `llm` | `BaseLLM` | 필수 | LLM 인스턴스 |
| `system_prompt` | `str` | `""` | 시스템 프롬프트 |
| `tools` | `ToolRegistry` | 빈 레지스트리 | 등록된 툴 목록 |
| `memory` | `ConversationMemory` | 자동 생성 | 대화 메모리 (`ContextManager`로 교체 가능) |
| `hooks` | `HookRegistry` | 빈 레지스트리 | 미들웨어 훅 |
| `max_tool_iterations` | `int` | `10` | 툴 루프 최대 반복 횟수 |

---

### LLM 제공자

`create_llm(provider, model=None, **kwargs)`로 생성합니다.

```python
from agent import create_llm

# Anthropic Claude (기본: claude-sonnet-4-6)
llm = create_llm("claude", api_key="sk-ant-...")

# OpenAI (기본: gpt-4o)
llm = create_llm("openai", model="gpt-4o-mini")

# Google Gemini (기본: gemini-2.5-flash-lite)
llm = create_llm("google")

# Ollama 로컬 (기본: llama3.2)
llm = create_llm("ollama", model="llama3.1:latest")

# OpenRouter — 200+ 모델 지원 (기본: anthropic/claude-sonnet-4-6)
llm = create_llm("openrouter", model="meta-llama/llama-3.1-8b-instruct")
```

**API 키 설정 우선순위:**

1. 생성자 인수: `create_llm("claude", api_key="sk-...")`
2. 환경변수: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`
3. `.env` 파일 (앱 시작 시 `load_dotenv()` 호출)

**각 제공자 기본 모델:**

| 제공자 키 | 클래스 | 기본 모델 |
|----------|--------|----------|
| `claude` | `ClaudeLLM` | `claude-sonnet-4-6` |
| `openai` | `OpenAILLM` | `gpt-4o` |
| `google` | `GoogleLLM` | `gemini-2.5-flash-lite` |
| `ollama` | `OllamaLLM` | `llama3.2` |
| `openrouter` | `OpenRouterLLM` | `anthropic/claude-sonnet-4-6` |

---

### Tool 등록

```python
from agent import ToolRegistry
from typing import Optional

tools = ToolRegistry()

# 기본 타입
@tools.register(description="현재 날짜와 시간을 반환합니다.")
def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().isoformat()

# 제네릭 타입도 올바르게 JSON Schema로 변환됨
@tools.register(description="문자열 목록을 하나로 합칩니다.")
def join_strings(items: list[str], separator: str = ", ") -> str:
    return separator.join(items)

# Optional 파라미터
@tools.register(description="숫자를 반올림합니다.")
def round_number(value: float, decimals: Optional[int] = None) -> float:
    return round(value, decimals)
```

**Python 타입 → JSON Schema 매핑:**

| Python 타입 | JSON Schema 타입 |
|------------|----------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list`, `list[X]`, `List[X]` | `"array"` |
| `dict`, `dict[K,V]` | `"object"` |
| `Optional[X]` | X의 타입 (required 목록에서 제외) |

기본값 없는 파라미터는 자동으로 `required`에 추가됩니다.

---

### Middleware / Hooks

4개 이벤트를 가로채서 로깅, 검열, 감사, 변환 등을 수행할 수 있습니다.  
**`SECURITY_FEATURE.md`의 보안 기능이 이 훅을 통해 구현될 예정입니다.**

```python
from agent import Agent, HookRegistry, create_llm

hooks = HookRegistry()

# LLM 호출 직전 — 메시지 검사/변환
@hooks.on("before_llm_call")
def log_request(messages):
    print(f"[LOG] LLM 요청: {len(messages)}개 메시지")
    # None 반환 → 메시지 그대로 통과
    # list 반환 → 해당 리스트로 교체

# LLM 응답 직후 — 응답 검사/변환
@hooks.on("after_llm_response")
def log_response(response):
    print(f"[LOG] LLM 응답: {response.content[:50]}")

# 툴 실행 직전 — 툴 호출 검사/차단
@hooks.on("before_tool_execute")
def audit_tool(tool_call):
    print(f"[AUDIT] 툴 호출: {tool_call.name}({tool_call.arguments})")

# 툴 실행 직후 — 결과 검사/변환 (result가 첫 번째 인자)
@hooks.on("after_tool_execute")
def redact_secrets(result, tool_call):
    if "password" in str(result).lower():
        return "[REDACTED]"
    # None 반환 → 결과 그대로 통과

agent = Agent(llm=create_llm("claude"), hooks=hooks)
```

**훅 이벤트 시그니처:**

| 이벤트 | 시그니처 | 반환값 |
|--------|---------|--------|
| `before_llm_call` | `(messages: list[Message])` | `list[Message] \| None` |
| `after_llm_response` | `(response: LLMResponse)` | `LLMResponse \| None` |
| `before_tool_execute` | `(tool_call: ToolCall)` | `ToolCall \| None` |
| `after_tool_execute` | `(result: Any, tool_call: ToolCall)` | `Any \| None` |

`None`을 반환하면 현재 값이 변경 없이 통과합니다. 여러 훅이 등록된 경우 등록 순서대로 체인 실행됩니다.

---

### Streaming

```python
from agent import Agent, create_llm

agent = Agent(llm=create_llm("claude"))

# 청크 단위 출력
for chunk in agent.stream("긴 이야기를 써줘."):
    print(chunk, end="", flush=True)
print()
```

스트리밍 모드에서는 툴 호출이 지원되지 않습니다. 툴이 필요한 경우 `agent.run()`을 사용하세요.

---

### AsyncAgent

툴 호출이 여러 개인 경우 `asyncio.gather`로 **동시 실행**합니다.  
I/O 대기가 있는 툴에서 성능 향상을 기대할 수 있습니다.

```python
import asyncio
from agent import AsyncAgent, ToolRegistry, create_llm

tools = ToolRegistry()

@tools.register(description="날씨를 조회합니다.")
def get_weather(city: str) -> str:
    import time; time.sleep(0.5)  # I/O 시뮬레이션
    return f"맑음, 22°C in {city}"

@tools.register(description="인구를 조회합니다.")
def get_population(city: str) -> str:
    import time; time.sleep(0.5)
    return {"Seoul": "9.77M"}.get(city, "Unknown")

async def main():
    # async with로 ThreadPoolExecutor 자원 정리
    async with AsyncAgent(llm=create_llm("claude"), tools=tools) as agent:
        # 두 툴이 동시에 실행 → 약 0.5초 (순차 실행이면 1초+)
        response = await agent.run("서울의 날씨와 인구는?")
        print(response)

        # 비동기 스트리밍
        async for chunk in agent.stream("짧은 이야기를 써줘."):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

---

### Context 관리

대화가 길어져 토큰 한도에 가까워지면 `ContextManager`가 자동으로 오래된 메시지를 요약합니다.  
`ConversationMemory`의 드롭인 교체제로, `Agent(memory=...)`에 그대로 전달할 수 있습니다.

```python
from agent import Agent, ContextManager, create_llm

# LLM으로 요약하는 커스텀 summarizer (선택)
def my_summarizer(messages):
    from agent.core.message import Message, Role
    llm = create_llm("claude")
    text = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
    resp = llm.chat([
        Message(role=Role.USER, content=f"다음 대화를 한 문단으로 요약:\n{text}")
    ])
    return resp.content

memory = ContextManager(
    max_context_tokens=3_000,   # 약 3000 토큰 (4자 ≈ 1토큰 기준)
    summarizer=my_summarizer,   # 생략 시 plain-text 다이제스트 사용
    keep_recent=6,              # 최근 6개 메시지는 항상 원문 보존
    system_prompt="You are a helpful assistant.",
)

agent = Agent(llm=create_llm("claude"), memory=memory)
```

**동작 방식:**

1. 메시지 추가 시마다 토큰 예산 확인 (4자 ≈ 1토큰 휴리스틱)
2. 초과 시 오래된 메시지들을 요약 메시지 1개로 압축
3. 최근 `keep_recent`개 메시지는 항상 원문 유지
4. `summarizer`가 없으면 plain-text 다이제스트로 대체

---

### PersistentMemory

`ConversationMemory`의 드롭인 교체제로, 대화 히스토리를 SQLite에 저장합니다.  
프로세스를 종료한 뒤에도 세션 ID로 이전 대화를 재개할 수 있습니다.

```python
from agent import Agent, PersistentMemory, create_llm

# 새 세션 생성
with PersistentMemory("agent.db", system_prompt="You are helpful.") as mem:
    agent = Agent(llm=create_llm("claude"), memory=mem)
    agent.run("My name is Alex.")
    print("Session ID:", mem.session_id)   # 저장해 두기

# 나중에 — 세션 재개
with PersistentMemory("agent.db", session_id="<저장한 session_id>") as mem:
    print(f"{len(mem.messages)}개 메시지 복원됨")
    agent = Agent(llm=create_llm("claude"), memory=mem)
    agent.run("Do you remember my name?")  # 이전 대화를 기억

# 전체 세션 목록 조회
sessions = PersistentMemory.list_sessions("agent.db")
for s in sessions:
    print(s["id"], s["message_count"], s["updated_at"])
```

**`PersistentMemory` 생성자 파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `db_path` | `str` | `"agent_memory.db"` | SQLite DB 파일 경로 |
| `session_id` | `str \| None` | `None` | 기존 세션 재개 시 지정 (없으면 신규 생성) |
| `max_messages` | `int` | `100` | 인메모리 버퍼 최대 메시지 수 |
| `system_prompt` | `str` | `""` | 신규 세션 시 사용할 시스템 프롬프트 |

**동작 특성:**

- `add()` 호출 시 SQLite에 즉시 기록 (WAL 모드)
- `clear()`는 인메모리 버퍼만 비우며 DB 레코드는 보존
- `delete_session()`으로 DB에서 세션과 메시지를 완전 삭제
- 컨텍스트 매니저(`with` 구문) 사용 시 DB 연결 자동 정리

---

### Retry / Rate Limit

모든 HTTP 요청에 지수 백오프 재시도가 기본 적용됩니다.

```python
from agent import create_llm
from agent.llm.retry import RetryConfig

retry = RetryConfig(
    max_retries=5,          # 최대 재시도 횟수 (기본: 3)
    base_delay=2.0,         # 첫 재시도 대기 시간 초 (기본: 1.0)
    backoff_factor=2.0,     # 대기 시간 배수 (기본: 2.0)
    max_delay=120.0,        # 최대 대기 시간 초 (기본: 60.0)
    jitter=True,            # 랜덤 지터 추가 (기본: True)
)

llm = create_llm("claude", retry_config=retry)
```

**기본 재시도 대상:** HTTP 429(Rate Limit), 500, 502, 503, 504, 네트워크 연결 오류(DNS 실패, 연결 거부)

---

## 설정

### CLI 저장 설정

`python main.py`에서 `/provider`로 선택한 provider/model은 프로젝트 루트의 `.baseagent.json`에 저장됩니다.

```json
{
  "provider": "ollama",
  "model": "llama3.2"
}
```

이 파일은 로컬 실행 상태이므로 `.gitignore`에 포함되어 있습니다. API 키는 저장하지 않습니다.

설정 우선순위:

1. 셸 환경변수 또는 `.env`: `LLM_PROVIDER`, `LLM_MODEL`
2. CLI 저장값: `.baseagent.json`
3. 코드 기본값: `claude`, provider별 기본 모델

### 환경변수

```bash
# .env 파일 또는 셸 환경변수
LLM_PROVIDER=claude                       # 저장된 provider보다 우선
LLM_MODEL=claude-sonnet-4-6              # 저장된 model보다 우선
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
OLLAMA_BASE_URL=http://localhost:11434    # Ollama 서버 주소
```

API 키 설정 우선순위는 코드에서 직접 넘긴 인자, 환경변수 또는 `.env`, CLI에서 직접 입력한 값 순입니다. CLI에서 입력한 API 키는 현재 실행에만 사용됩니다.

`.env.example`에는 실제 키를 넣지 말고 placeholder만 남겨 두세요. 실제 키는 `.env`에만 저장합니다.

### 코드에서 Config 사용

```python
from agent import Config, load_dotenv

load_dotenv()  # .env 파일 로드 (이미 셸 환경변수에 있으면 생략 가능)

config = Config.from_env()
provider = config.get("llm_provider", "claude")   # 없으면 "claude"
model    = config.get("llm_model")                # 없으면 None
```

`Config.from_env()`는 환경변수와 `.env`를 읽기 위한 유틸리티입니다. CLI의 `.baseagent.json` 저장값은 `main.py`에서만 사용합니다.

---

## 예제 실행

```bash
# Ollama 로컬 모델 사용 시
LLM_MODEL=llama3.1:latest python examples/basic_chat.py ollama
LLM_MODEL=llama3.1:latest python examples/tool_use_demo.py ollama
LLM_MODEL=llama3.1:latest python examples/streaming_demo.py ollama
LLM_MODEL=llama3.1:latest python examples/hooks_demo.py ollama
LLM_MODEL=llama3.1:latest python examples/async_demo.py ollama

# Claude / OpenAI 사용 시 (.env에 API 키 설정 후)
python examples/basic_chat.py claude
python examples/basic_chat.py openai
```

```bash
# SQLite 영속 메모리 데모
python examples/persistent_memory_demo.py new             # 새 세션 시작
python examples/persistent_memory_demo.py list            # 저장된 세션 목록
python examples/persistent_memory_demo.py resume <id>    # 세션 재개
```

| 예제 파일 | 설명 |
|----------|------|
| `basic_chat.py` | 멀티턴 대화, 이전 질문 기억 확인 |
| `tool_use_demo.py` | 덧셈, 현재 시간, 온도 변환 툴 호출 |
| `streaming_demo.py` | 청크 단위 스트리밍 텍스트 출력 |
| `hooks_demo.py` | 4개 훅 이벤트 로깅 및 툴 감사 |
| `async_demo.py` | 두 툴 동시 실행 vs 순차 실행 시간 비교 |
| `persistent_memory_demo.py` | SQLite 영속 메모리 new/resume/list 3가지 모드 |

---

## 테스트

외부 라이브러리 없이 stdlib `unittest`로 실행합니다. 모든 테스트는 실제 API 호출 없이 `MockLLM`으로 동작합니다.

```bash
# 개별 실행
python tests/test_agent.py
python tests/test_hooks.py
python tests/test_retry.py
python tests/test_context.py
python tests/test_async_agent.py
python tests/test_persistent_memory.py

# 전체 실행
python -m unittest discover -v
```

| 테스트 파일 | 테스트 수 | 검증 내용 |
|------------|---------|---------|
| `test_agent.py` | 16 | 에이전트 루프, 툴 실행, 메모리, JSON Schema 타입 |
| `test_hooks.py` | 11 | 훅 체인, falsy 결과 보존, 다중 훅 |
| `test_retry.py` | 9 | 지수 백오프, 재시도 조건, 성공 복귀 |
| `test_context.py` | 9 | 토큰 추정, 압축 트리거, 커스텀 요약 |
| `test_async_agent.py` | 7 | 비동기 실행, 동시 툴, 스트리밍 메모리 |
| `test_persistent_memory.py` | 10 | 세션 생성/복원, 툴콜 직렬화, 세션 삭제, clear 격리 |
| **합계** | **62** | |

---

## 보안 기능 확장 계획

`SECURITY_FEATURE.md`에 명세된 보안 기능들은 이 베이스 에이전트 위에 구현될 예정입니다.  
훅 시스템 덕분에 에이전트 코어를 수정하지 않고 보안 레이어를 주입할 수 있습니다.

| 보안 기능 | 구현 예정 위치 |
|----------|--------------|
| 출력 Redaction (API 키 실시간 마스킹) | `after_llm_response` + `after_tool_execute` 훅 |
| 감사 로그 (Append-only Audit Logging) | `before_llm_call` + `after_tool_execute` 훅 |
| 프롬프트 인젝션 방어 | `before_llm_call` 훅 입력 검증 |
| 경로 탐색 공격 차단 (Traversal Guard) | `before_tool_execute` 훅 인수 검증 |
| 자격 증명 스캐너 | `after_tool_execute` 훅 결과 스캔 |
| 쉴드 모드 (Immutable Config) | `Agent` 서브클래스 또는 커스텀 Memory |
| 샌드박스 격리 / 네트워크 제어 | 프로세스/컨테이너 수준 (에이전트 외부) |
