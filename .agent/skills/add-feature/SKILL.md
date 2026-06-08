---
name: add-feature
description: >-
  Add a new feature, module, class, or function to the BaseAgent framework while
  enforcing this project's documentation convention: Korean educational comments
  on top of the existing English docstrings, at a learner-friendly level of
  detail. Use this whenever the user asks to "add a feature", "make a new
  module/class/tool/LLM provider/memory type", "implement X", "extend the agent",
  or otherwise write new source code under `agent/` or `main.py` — so the new
  code is commented the same way as the rest of the tree. After the feature is
  implemented, this skill ALWAYS finishes by invoking the `verify-project` skill
  to validate the change. 새 기능 추가, 새 모듈/클래스/도구/제공자 만들기, 기능 구현,
  코드 확장 시 사용. 기능을 추가할 때 반드시 한국어 설명 주석을 (지금까지의 스타일대로)
  달아야 하며, 구현이 끝나면 자동으로 verify-project 스킬로 검증한다.
---

# Add Feature (한국어 주석 규칙 적용)

이 프로젝트(BaseAgent)에 **새 기능을 추가할 때 따라야 하는 절차와 주석 규칙**을 담은
스킬이다. 핵심 목적은 단 하나: 새로 작성하는 모든 소스 코드가 **기존 코드와 똑같은
방식으로 주석 처리**되도록 보장하는 것이다. 사용자가 "기능을 추가할 때 지금처럼 주석을
달아야 한다"고 요구했기 때문에, 새 코드 작성은 항상 이 규칙을 동반한다.

## 언제 사용하나

"기능 추가", "새 모듈/클래스/도구/메모리/LLM 제공자 만들기", "X 구현", "에이전트 확장"
처럼 `agent/` 또는 `main.py` 아래에 **새 소스 코드를 작성**하는 모든 요청에서 사용한다.

## 주석 규칙 (반드시 지킬 것)

기존 코드베이스에 적용된 컨벤션과 100% 일치시킨다.

1. **언어**: 설명 주석은 **한국어**로 단다. 기존 영어 docstring은 **지우지 말고 그대로
   둔다**. 즉 "영어 docstring 유지 + 한국어 설명 주석 추가" 방식이다.
2. **상세도(교육용)**: 초보자가 따라 읽을 수 있는 수준으로 쓴다. "무엇을 하는가"보다
   **"왜 이렇게 하는가"**와 **개념 설명**에 무게를 둔다.
   - 예: TF-IDF/코사인 유사도, SSE/NDJSON, fail-closed, 지수 백오프, 늦은 바인딩
     클로저 버그 등 비자명한 개념은 한두 줄로 풀어 설명한다.
3. **모듈 헤더**: 각 새 `.py` 파일 맨 위에 한국어 모듈 docstring을 둔다. 이 모듈이
   **프레임워크 전체에서 맡는 역할**과 **다른 부분과의 관계**를 설명한다.
4. **클래스/메서드**: 기존 영어 docstring은 유지하고, 그 아래 또는 옆에 한국어로
   목적·핵심 동작을 보충한다. docstring이 없던 곳에는 한국어 설명을 새로 단다.
5. **속성/필드**: 생성자(`__init__`)의 각 인스턴스 속성과 dataclass 필드에는 끝에
   인라인 한국어 주석(`# ...`)으로 의미를 적는다. 여러 줄이면 세로로 보기 좋게 정렬한다.
6. **까다로운 로직**: 분기·예외 처리·트릭에는 "왜 이렇게 했는지"를 인라인 주석으로
   남긴다. 특히 안전장치(fail-closed, 0/"" 같은 falsy 값 처리, 자원 정리 등)는 의도를
   명시한다.
7. **건드리지 않는 곳**: `examples/`와 `tests/` 폴더에는 **주석을 추가하지 않는다**
   (사용자 지시). 이 스킬은 `agent/`와 `main.py` 같은 제품 소스에만 적용한다.

### 좋은 예 (이 코드베이스의 실제 스타일)

```python
"""<모듈 한 줄 요약>.

<이 모듈이 프레임워크에서 하는 역할과 다른 컴포넌트와의 관계를 2~5줄로 설명.>
"""

class Foo:
    """Existing English docstring stays as-is."""

    def __init__(self, bar: int) -> None:
        self.bar = bar          # bar가 무엇이고 왜 필요한지 한국어로.

    def run(self) -> None:
        # 승인 콜백이 없으면 안전하게 거부한다(fail-closed). 권한이 모호할 땐
        # 막는 쪽이 안전하기 때문.
        ...
```

## 절차

1. **기존 패턴 파악**: 추가할 기능과 가장 비슷한 기존 파일을 먼저 읽는다. 예) 새 LLM
   제공자 → `agent/llm/openai_llm.py`·`claude.py`; 새 메모리 → `agent/core/memory.py`와
   그 파생들; 새 도구 → `agent/core/tool.py`와 `main.py`의 `build_default_tools`.
   - 새 제공자는 `BaseLLM`을 상속하고 `chat`/`stream`을 구현하며, 공용 `Message` ↔
     제공자 형식 변환만 책임진다(HTTP·재시도·스트리밍 파싱은 `BaseLLM` 재사용).
   - 새 메모리는 `ConversationMemory`를 상속해 드롭인 교체가 되도록 한다.
2. **구현 + 주석 동시 진행**: 위 "주석 규칙"을 지키며 코드를 작성한다. 주석은 나중에
   몰아 달지 말고 작성과 함께 단다.
3. **공개 노출**: 외부 API라면 알맞은 `__init__.py`와 `__all__`에 추가하고, 새 제공자는
   `agent/llm/__init__.py`의 `_PROVIDERS` 표에 등록한다.
4. **테스트**: `tests/`에 MockLLM 기반 단위 테스트를 추가한다(테스트 파일에는 한국어
   주석을 달지 않는다).
5. **자동 검증 (필수 · 생략 불가)**: 구현이 끝나면 **자동으로 `verify-project` 스킬을
   호출**해 회귀가 없는지 검증한다. 이 단계는 add-feature 워크플로의 고정된 마지막
   단계다 — 사용자가 따로 요청하지 않아도 항상 실행한다.

   - 진행 방식: Skill 도구로 `verify-project`를 호출한다(셸에서 직접 `verify.py`만
     돌리지 말고, 검증 스킬의 실패→수정→재검증 루프를 그대로 따른다).
   - 통과 기준: `verify-project`가 `OVERALL: ALL PASS`(종료코드 0)를 보고할 때까지
     수정하고 다시 검증한다.
   - 검증을 통과하지 못했거나 아직 돌리지 않았다면, 기능 추가 작업을 "완료"로
     보고하지 않는다.

## 완료 기준 체크리스트

- [ ] 새 `.py` 파일마다 한국어 모듈 docstring이 있다.
- [ ] 기존 영어 docstring을 지우지 않았다.
- [ ] 모든 인스턴스 속성/dataclass 필드에 한국어 인라인 주석이 있다.
- [ ] 비자명한 개념·분기·안전장치에 "왜"를 설명하는 주석이 있다.
- [ ] `examples/`·`tests/`에는 주석을 추가하지 않았다.
- [ ] 공개 API/`_PROVIDERS` 등록을 마쳤다(해당 시).
- [ ] **(필수)** 마지막에 `verify-project` 스킬을 호출했고 `OVERALL: ALL PASS`를 받았다.
