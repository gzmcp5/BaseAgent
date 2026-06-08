# BaseAgent

Python stdlib만 사용하는 멀티 LLM 제공자 AI 에이전트 프레임워크. Python 3.10+.  
보안 기능(`SECURITY_FEATURE.md`)을 훅 주입으로 코어 수정 없이 확장하는 구조.

## 구조

```
agent/core/       — agent.py, async_agent.py, hooks.py, tool.py, tool_selector.py,
                    memory.py, context.py, summarizing_memory.py, user_profile.py,
                    multi_agent.py, persistent_memory.py, message.py
agent/llm/        — base.py, claude.py, openai_llm.py, google.py, ollama.py,
                    openrouter.py, retry.py
agent/utils/      — config.py, env.py
main.py           — 대화형 CLI
examples/         — 실행 예제
tests/            — MockLLM 기반 unittest 164개
```

## 핵심 API

```python
from agent import Agent, create_llm, ToolRegistry, HookRegistry

llm = create_llm("claude")          # openai / google / ollama / openrouter 도 가능
agent = Agent(llm=llm, system_prompt="...", tools=tools, hooks=hooks)
agent.run("질문")
```

## 메모리 종류 (`Agent(memory=...)` 드롭인 교체)

| 클래스 | 용도 |
|--------|------|
| `ConversationMemory` | 기본 인메모리 히스토리 |
| `ContextManager` | 토큰 예산 초과 시 자동 요약 압축 |
| `SummarizingMemory` | max_messages 초과 시 오래된 메시지를 LLM(또는 폴백)으로 요약하여 SYSTEM 메시지로 보존 |
| `ProfileMemory` | `UserProfile`(이름·선호도·facts)을 SYSTEM 메시지로 자동 주입 |
| `PersistentMemory` | SQLite 영속 저장, 프로세스 재시작 후 세션 resume |

## 도구 실행 고도화

- **RAG 도구 선택**: `ToolSelector`(TF-IDF 코사인, 유니코드/한글 토큰화) → `Agent(tool_selector=...)`로 관련 상위 `top_k`개 스키마만 전송해 토큰 절감.
- **동시 도구 실행**: `AsyncAgent.run()`이 한 턴의 `tool_calls`를 `asyncio.gather`로 병렬 실행.
- **사용자 동의(Human-in-the-loop)**: `register(requires_approval=True)`로 민감 도구 표시, `Agent(approval_callback=...)`가 실행 전 동의 확인(fail-closed).

## 훅 (보안 확장 핵심)

| 이벤트 | 용도 |
|--------|------|
| `before_llm_call` | 프롬프트 인젝션 방어, 감사 로그 |
| `after_llm_response` | 출력 리댁션 (API 키 마스킹) |
| `before_tool_execute` | 경로 탐색 차단 |
| `after_tool_execute` | 결과 리댁션, 감사 로그 |

`None` 반환 → 통과 / 값 반환 → 교체.

## 멀티에이전트

- **OrchestratorAgent**: 서브에이전트를 `ask_{name}` 툴로 노출, LLM이 호출 순서 결정
- **Pipeline**: 각 에이전트 출력 → 다음 에이전트 입력

## 환경변수

`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`  
`LLM_PROVIDER`, `LLM_MODEL` 으로 기본값 지정 가능.

## 검증

전체 프로젝트 반복 검증 스킬: `.claude/skills/verify-project/`.  
5축(구문 컴파일·임포트·랜덤순서 반복 실행·경고강제·ToolSelector 퍼즈)을 한 번에 실행하며 종료코드 0=전체 통과(CI 게이트 겸용).

```bash
python .claude/skills/verify-project/scripts/verify.py        # 전체
python -m unittest discover -s tests                          # 테스트만
```

참고: `.claude/` → `.agent/` 심볼릭 링크, `CLAUDE.md` → `AGENTS.md` 심볼릭 링크 (한쪽만 수정하면 양쪽 반영).
