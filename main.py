#!/usr/bin/env python3
"""Base Agent — interactive CLI entry point.

Usage:
    python main.py                          # Claude (default)
    LLM_PROVIDER=openai python main.py
    LLM_PROVIDER=ollama LLM_MODEL=llama3.2 python main.py

Commands inside the REPL:
    /reset      Clear conversation history
    /provider   Choose LLM provider and model
    /quit       Exit
"""
import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

from agent import Agent, ToolRegistry, create_llm, load_dotenv

load_dotenv()  # 가장 먼저 .env를 읽어 API 키 등을 환경변수로 올린다.


# ── 상수 정의 ─────────────────────────────────────────────────────────────
# 지원 제공자 목록: (내부 이름, 표시 이름, 필요한 API 키 환경변수). 키가 필요 없으면 None.
PROVIDERS = [
    ("claude", "Anthropic Claude", "ANTHROPIC_API_KEY"),
    ("openai", "OpenAI", "OPENAI_API_KEY"),
    ("google", "Google Gemini", "GOOGLE_API_KEY"),
    ("ollama", "Ollama local", None),
    ("openrouter", "OpenRouter", "OPENROUTER_API_KEY"),
]
PROVIDER_KEYS = {name for name, _, _ in PROVIDERS}  # 빠른 유효성 검사용 이름 집합.
# REPL 안에서 쓸 수 있는 슬래시 명령어와 설명.
COMMAND_OPTIONS = [
    ("/reset", "Clear conversation history"),
    ("/provider", "Choose LLM provider and model"),
    ("/quit", "Exit"),
]
COMMANDS = [command for command, _ in COMMAND_OPTIONS]   # 자동완성용 명령어 이름만 추출.
PROVIDER_NAMES = [name for name, _, _ in PROVIDERS]       # 자동완성용 제공자 이름만 추출.
STATE_PATH = Path(".baseagent.json")  # 마지막으로 고른 제공자/모델을 저장하는 파일.
COMPLETION_MODE = "command"           # 현재 탭 자동완성 맥락(command/provider/none).
# ANSI 이스케이프 색상 코드. 터미널 출력에 색을 입힌다.
RESET = "\x1b[0m"                # 색 초기화.
PROMPT_COLOR = "\x1b[1;36m"      # 프롬프트("> ") 색.
INPUT_COLOR = "\x1b[38;5;250m"   # 사용자 입력 색.
MENU_COLOR = "\x1b[38;5;244m"    # 메뉴/안내 글자 색.
SELECTED_COLOR = "\x1b[1;32m"    # 메뉴에서 선택된 항목 색.


def _supports_color() -> bool:
    # 출력이 진짜 터미널(tty)이고 NO_COLOR 환경변수가 없을 때만 색을 쓴다
    # (파일로 리다이렉트하면 색 코드가 깨져 보이므로).
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _color(text: str, color: str) -> str:
    # 색을 지원할 때만 색 코드로 감싸고, 아니면 원문 그대로 반환.
    if not _supports_color():
        return text
    return f"{color}{text}{RESET}"


def _prompt_text(prompt: str) -> str:
    return _color(prompt, PROMPT_COLOR)


def _input_text(text: str) -> str:
    return _color(text, INPUT_COLOR)


def set_completion_mode(mode: str) -> None:
    global COMPLETION_MODE
    COMPLETION_MODE = mode


def setup_autocomplete() -> Callable[[], None]:
    """Enable readline tab-completion when available."""
    # readline은 OS에 따라 없을 수 있다. 없으면 '아무것도 안 하는 복원 함수'를 돌려주고 끝.
    try:
        import readline
    except ImportError:
        return lambda: None

    # 기존 자동완성 설정을 백업해 두었다가 끝날 때 되돌린다(다른 코드에 영향 주지 않기 위해).
    previous_completer = readline.get_completer()
    previous_delims = readline.get_completer_delims()
    readline.set_completer_delims(" \t\n")  # 단어 구분 기준.

    # readline이 탭을 누를 때마다 호출하는 콜백. 같은 text로 state=0,1,2... 순으로 물어
    # 후보를 하나씩 받아간다(IndexError가 나면 후보 끝으로 인식).
    def complete(text: str, state: int) -> str | None:
        line = readline.get_line_buffer()
        stripped = line.lstrip()

        # 현재 맥락에 따라 후보 목록을 다르게 만든다.
        if COMPLETION_MODE == "provider":
            matches = [name for name in PROVIDER_NAMES if name.startswith(text)]
        elif stripped.startswith("/provider "):
            matches = [name for name in PROVIDER_NAMES if name.startswith(text)]
        elif stripped.startswith("/"):
            matches = [cmd for cmd in COMMANDS if cmd.startswith(text)]
        else:
            matches = []

        try:
            return matches[state]
        except IndexError:
            return None

    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")  # 탭 키에 자동완성 연결.

    # 호출자가 끝날 때 부르면 원래 설정으로 되돌리는 함수.
    def restore() -> None:
        readline.set_completer(previous_completer)
        readline.set_completer_delims(previous_delims)

    return restore


def _clear_lines(count: int) -> None:
    # 커서를 위로 올리며(\x1b[F) 줄을 지운다(\x1b[2K). 메뉴를 다시 그리기 전에 이전 출력을 지움.
    for _ in range(count):
        sys.stdout.write("\x1b[F\x1b[2K")


def _read_key() -> str:
    # 키 입력 한 번을 '문자 단위'로 읽는다. 방향키 같은 특수키는 여러 바이트(이스케이프 시퀀스)라
    # 단순 read 한 번으로는 부족해서, 아래처럼 경우를 나눠 모아 읽는다.
    first = os.read(sys.stdin.fileno(), 1)
    if not first:
        return ""  # 입력 끝(EOF).

    # ESC(\x1b)로 시작하면 방향키 등 이스케이프 시퀀스일 가능성이 높다.
    if first == b"\x1b":
        import select

        # ESC 다음에 바로 입력이 더 없으면, 사용자가 ESC 키 자체를 누른 것으로 본다.
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not ready:
            return "\x1b"

        char = "\x1b" + os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        # ESC + '['나 'O'가 아니면 Alt+키 조합 등 → 두 글자만으로 확정.
        if len(char) == 2 and char[1] not in {"[", "O"}:
            return char

        # '['/'O'로 이어지는 CSI 시퀀스는 종료 문자(@~)가 올 때까지 마저 읽는다.
        while len(char) < 8:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                break
            char += os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
            if "@" <= char[-1] <= "~":
                break
        return char

    # 여기부터는 일반 문자. UTF-8은 첫 바이트로 전체 길이를 알 수 있다(한글은 보통 3바이트).
    # 첫 바이트의 비트 패턴을 보고 이어 읽어야 할 바이트 수를 정한다.
    lead = first[0]
    if lead < 0x80:
        expected = 1       # ASCII(1바이트).
    elif 0xC0 <= lead < 0xE0:
        expected = 2       # 2바이트 문자.
    elif 0xE0 <= lead < 0xF0:
        expected = 3       # 3바이트 문자(한글 등).
    elif 0xF0 <= lead < 0xF8:
        expected = 4       # 4바이트 문자(이모지 등).
    else:
        expected = 1       # 비정상 선행 바이트는 1바이트로 처리.

    # 한 글자를 이루는 나머지 바이트를 다 모은 뒤 디코딩해 반환.
    data = first
    while len(data) < expected:
        data += os.read(sys.stdin.fileno(), expected - len(data))
    return data.decode(errors="ignore")


def _select_command(prompt: str) -> str:
    # 사용자가 '/'를 입력하면 뜨는 대화형 명령어 선택기.
    # 입력한 글자로 후보를 좁히고 ↑/↓로 고른 뒤 Enter로 확정한다.
    query = "/"          # 현재까지 입력된 명령어 텍스트.
    selected = 0         # 현재 강조된 항목 인덱스.
    rendered_lines = 0   # 화면에 그린 줄 수(지울 때 사용).

    def matches() -> list[tuple[str, str]]:
        # 지금 query로 시작하는 명령어만 추린다.
        return [
            (command, description)
            for command, description in COMMAND_OPTIONS
            if command.startswith(query)
        ]

    def render() -> int:
        current_matches = matches()
        print(f"{_prompt_text(prompt)}{_input_text(query)}")
        print(_color("Up/Down to select, Enter to run, Esc to cancel", MENU_COLOR))
        if not current_matches:
            print(_color("  No matching commands", MENU_COLOR))
            sys.stdout.flush()
            return 3

        for index, (command, description) in enumerate(current_matches):
            marker = ">" if index == selected else " "
            line = f"{marker} {command:<10} {description}"
            print(_color(line, SELECTED_COLOR if index == selected else MENU_COLOR))
        sys.stdout.flush()
        return len(current_matches) + 2

    sys.stdout.write("\r\x1b[2K")
    rendered_lines = render()

    # 키 입력을 받아 처리하는 루프(Enter/Esc로 빠져나간다).
    while True:
        key = _read_key()
        if key in {"\r", "\n"}:                # Enter: 현재 선택을 확정.
            current_matches = matches()
            command = current_matches[selected][0] if current_matches else query
            _clear_lines(rendered_lines)
            print(f"{_prompt_text(prompt)}{_input_text(command)}")
            return command
        if key in {"\x1b", "\x1b["}:            # Esc: 취소(빈 문자열 반환).
            _clear_lines(rendered_lines)
            return ""
        if key in {"\x1b[A", "\x1bOA"}:         # ↑: 위 항목으로(% 로 끝에서 순환).
            current_matches = matches()
            if current_matches:
                selected = (selected - 1) % len(current_matches)
        elif key in {"\x1b[B", "\x1bOB"}:       # ↓: 아래 항목으로.
            current_matches = matches()
            if current_matches:
                selected = (selected + 1) % len(current_matches)
        elif key == "\x03":                     # Ctrl+C.
            raise KeyboardInterrupt
        elif key == "\x04":                     # Ctrl+D.
            raise EOFError
        elif key in {"\x7f", "\b"}:             # 백스페이스.
            if len(query) == 1:                 # '/'만 남았는데 또 지우면 선택 취소.
                _clear_lines(rendered_lines)
                return ""
            query = query[:-1]
            selected = 0
        elif key == "\t":                       # 탭: 현재 강조 항목으로 입력을 채움.
            current_matches = matches()
            if current_matches:
                query = current_matches[selected][0]
                selected = 0
        elif len(key) == 1 and key.isprintable() and not key.isspace():
            query += key                        # 일반 글자: 검색어에 덧붙임.
            selected = 0
        else:
            continue                            # 그 외 키는 무시(화면 다시 그리지 않음).

        # 입력/선택이 바뀌었으니 이전 화면을 지우고 다시 그린다.
        _clear_lines(rendered_lines)
        rendered_lines = render()


def select_option(
    title: str,
    options: list[tuple[str, str]],
    *,
    cancel_label: str = "Cancel",
) -> str | None:
    # 화살표로 고르는 범용 선택 메뉴(제공자·모델 선택 등에 재사용).
    # 입력이 터미널이 아니면(파이프 등) 화살표를 못 쓰므로 번호 입력 방식으로 대체한다.
    if not sys.stdin.isatty():
        print(title)
        for index, (value, label) in enumerate(options, start=1):
            print(f"  {index}. {value} ({label})")
        print("  q. cancel")

        choice = input("> ").strip().lower()
        if choice in {"", "q", "quit", "cancel"}:
            return None

        try:
            return options[int(choice) - 1][0]
        except (ValueError, IndexError):
            match = next((value for value, _ in options if value == choice), None)
            return match

    # 화살표 입력을 받으려면 터미널을 '한 글자씩 즉시 읽는' 모드로 바꿔야 한다(termios/tty).
    # 윈도우 등에서 없을 수 있어 import 실패 시 선택 불가(None)로 처리.
    try:
        import termios
        import tty
    except ImportError:
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)  # 끝나고 복원할 원래 터미널 설정 백업.
    selected = 0
    rendered_lines = 0
    menu_options = options + [("", cancel_label)]  # 맨 끝에 '취소' 항목을 추가.

    def render() -> int:
        print(_color(title, MENU_COLOR))
        print(_color("Up/Down to select, Enter to confirm, Esc to cancel", MENU_COLOR))
        for index, (value, label) in enumerate(menu_options):
            marker = ">" if index == selected else " "
            text = cancel_label if value == "" else f"{value:<16} {label}"
            line = f"{marker} {text}"
            print(_color(line, SELECTED_COLOR if index == selected else MENU_COLOR))
        sys.stdout.flush()
        return len(menu_options) + 2

    try:
        tty.setcbreak(fd)  # cbreak 모드: Enter 없이도 키를 즉시 받는다.
        rendered_lines = render()
        while True:
            key = _read_key()
            if key in {"\r", "\n"}:                # Enter: 선택 확정.
                value = menu_options[selected][0]
                _clear_lines(rendered_lines)
                if value:
                    print(f"{_prompt_text('> ')}{_input_text(value)}")
                return value or None               # 취소 항목은 빈 값 → None.
            if key in {"\x1b", "\x1b["}:
                _clear_lines(rendered_lines)
                return None
            if key in {"\x1b[A", "\x1bOA"}:
                selected = (selected - 1) % len(menu_options)
            elif key in {"\x1b[B", "\x1bOB"}:
                selected = (selected + 1) % len(menu_options)
            elif key == "\x03":
                raise KeyboardInterrupt
            elif key == "\x04":
                raise EOFError
            else:
                continue

            _clear_lines(rendered_lines)
            rendered_lines = render()
    finally:
        # 무슨 일이 있어도 터미널 설정을 원래대로 복원한다(안 하면 셸이 깨진 상태로 남음).
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_repl_input(prompt: str) -> str:
    # REPL의 한 줄 입력을 읽는다. '/'를 맨 앞에 누르면 명령어 선택기로 전환하는 게 핵심.
    # 비터미널 입력이거나 termios가 없으면 평범한 input()으로 대체한다.
    if not sys.stdin.isatty():
        return input(prompt).strip()

    try:
        import termios
        import tty
    except ImportError:
        return input(prompt).strip()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)  # 복원용 백업.
    buffer = ""                            # 지금까지 입력한 글자들.
    print(f"{_prompt_text(prompt)}{INPUT_COLOR}", end="", flush=True)

    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key()
            if key in {"\r", "\n"}:                 # Enter: 입력 완료.
                print(RESET if _supports_color() else "")
                return buffer.strip()
            if key == "\x03":                       # Ctrl+C.
                raise KeyboardInterrupt
            if key == "\x04":                       # Ctrl+D.
                raise EOFError
            if key in {"\x7f", "\b"}:               # 백스페이스: 마지막 글자를 지운다.
                if buffer:
                    buffer = buffer[:-1]
                    sys.stdout.write("\b \b")       # 화면에서도 한 칸 지움(뒤로→공백→뒤로).
                    sys.stdout.flush()
                continue
            if not buffer and key == "/":           # 빈 줄에서 '/' → 명령어 선택기로 진입.
                if _supports_color():
                    sys.stdout.write(RESET)
                return _select_command(prompt).strip()
            if key.startswith("\x1b"):              # 방향키 등 이스케이프 키는 입력에서 무시.
                continue

            buffer += key                            # 일반 글자: 버퍼에 추가하고 화면에 표시.
            print(key, end="", flush=True)
    finally:
        if _supports_color():
            sys.stdout.write(RESET)
            sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # 터미널 설정 복원.


def build_default_tools() -> ToolRegistry:
    # 에이전트에 기본 제공할 도구 모음. 여기서는 예시로 '현재 시각' 도구 하나만 등록한다.
    tools = ToolRegistry()

    @tools.register(description="Return the current date and time.")
    def get_current_time() -> str:
        return datetime.now().isoformat()

    return tools


def load_saved_settings() -> dict[str, str]:
    # 지난번에 저장해 둔 제공자/모델 설정을 불러온다(없거나 깨졌으면 빈 dict).
    if not STATE_PATH.exists():
        return {}
    try:
        with STATE_PATH.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}  # 파일이 손상돼도 앱이 죽지 않도록 조용히 무시.

    # 파일 내용을 그대로 믿지 않고 타입·유효성을 검사한 값만 받아들인다(방어적 로딩).
    settings = {}
    provider = data.get("provider")
    model = data.get("model")
    if isinstance(provider, str) and provider in PROVIDER_KEYS:
        settings["provider"] = provider
    if isinstance(model, str) and model:
        settings["model"] = model
    return settings


def save_settings(provider: str, model: str) -> None:
    # 현재 제공자/모델을 파일에 저장해 다음 실행 때 자동 복원되게 한다.
    payload = {"provider": provider, "model": model}
    try:
        with STATE_PATH.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except OSError as exc:
        # 저장 실패는 치명적이지 않으므로 경고만 하고 계속 진행한다.
        print(f"Warning: failed to save settings: {exc}\n")


def get_ollama_models() -> list[str]:
    # 로컬 Ollama 서버에 설치된 모델 목록을 조회한다(/api/tags).
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # 짧은 타임아웃으로 무한 대기 방지.
            data = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []  # 서버가 꺼져 있거나 응답이 이상하면 빈 목록.

    models = data.get("models", [])
    names = [item.get("name", "") for item in models if item.get("name")]
    return sorted(names)


def choose_model(provider: str) -> str | None:
    # 모델 선택. Ollama만 설치된 모델 목록을 보여 주고, 나머지는 직접 입력받는다.
    # 반환 None = "제공자 기본 모델 사용".
    if provider != "ollama":
        return input("Model (Enter for provider default): ").strip() or None

    models = get_ollama_models()
    if not models:
        # 목록을 못 가져오면 수동 입력으로 폴백.
        print("No local Ollama models found. Enter a model manually.")
        return input("Model (Enter for provider default): ").strip() or None

    options = [(name, "local Ollama model") for name in models]
    selected = select_option("Select Ollama model:", options)
    if selected is not None:
        return selected
    print("Using provider default model.\n")
    return None


def choose_provider(
    current_provider: str,
    current_model: str,
) -> tuple[str, str | None, dict] | None:
    # 제공자 → 모델 → (필요시) API 키 순으로 물어보는 대화형 흐름.
    # 반환값: (제공자, 모델, 생성자에 넘길 추가 kwargs). 취소·실패 시 None.
    print(f"Current provider={current_provider}  model={current_model}")
    set_completion_mode("provider")  # 자동완성을 '제공자 이름' 모드로 전환.
    # 메뉴에 각 제공자와 필요한 API 키를 함께 표시한다.
    provider_options = []
    for name, label, env_key in PROVIDERS:
        key_label = "no API key" if env_key is None else env_key
        provider_options.append((name, f"{label}, {key_label}"))

    choice = select_option("Select provider:", provider_options)
    if choice is None:                          # 사용자가 취소한 경우.
        set_completion_mode("command")
        print("Provider unchanged.\n")
        return None

    # 고른 이름에 해당하는 PROVIDERS 항목을 찾는다.
    selected = next((item for item in PROVIDERS if item[0] == choice), None)
    if selected is None:
        set_completion_mode("command")
        print("Invalid provider selection.\n")
        return None

    provider, _, env_key = selected
    set_completion_mode("none")                 # 모델/키 입력 동안엔 자동완성 끔.
    model = choose_model(provider)
    kwargs = {}

    # 이 제공자가 API 키를 요구하는데 환경변수에 없으면, 그 자리에서 안전하게 입력받는다.
    if env_key is not None and not os.environ.get(env_key):
        api_key = getpass.getpass(f"{env_key}: ").strip()  # getpass: 입력이 화면에 안 보임.
        if not api_key:
            set_completion_mode("command")
            print("API key is required for this provider.\n")
            return None
        kwargs["api_key"] = api_key

    set_completion_mode("command")              # 자동완성을 다시 명령어 모드로 복귀.
    return provider, model, kwargs


def main() -> None:
    # 프로그램 진입점: 설정을 불러와 에이전트를 만들고, 사용자와 주고받는 REPL 루프를 돈다.
    restore_autocomplete = setup_autocomplete()
    saved_settings = load_saved_settings()
    # 제공자/모델 우선순위: 저장된 설정 > 환경변수 > 기본값(claude).
    provider = saved_settings.get("provider") or os.environ.get("LLM_PROVIDER") or "claude"
    model = saved_settings.get("model") or os.environ.get("LLM_MODEL")

    try:
        llm = create_llm(provider, model=model)
    except ValueError as exc:
        # 알 수 없는 제공자 등 설정 오류면 메시지를 남기고 종료.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    tools = build_default_tools()
    agent = Agent(
        llm=llm,
        system_prompt="You are a helpful AI assistant.",
        tools=tools,
    )

    model_label = model or llm.model
    print(f"Base Agent  |  provider={provider}  model={model_label}")
    print("Commands: /reset  /provider  /quit\n")

    # 메인 REPL 루프: 입력을 받아 명령어면 처리하고, 아니면 에이전트에게 전달한다.
    while True:
        try:
            try:
                user_input = read_repl_input("> ")
            except (EOFError, KeyboardInterrupt):
                # Ctrl+D/Ctrl+C는 정상 종료로 처리.
                print("\nGoodbye.")
                break

            if not user_input:
                continue  # 빈 입력은 무시.

            # '/'로 시작하면 일반 대화가 아니라 명령어로 해석한다.
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lstrip("/").lower()
                if cmd == "quit":
                    print("Goodbye.")
                    break
                elif cmd == "reset":
                    agent.reset()  # 대화 기록 초기화.
                    print("Conversation cleared.\n")
                elif cmd == "provider":
                    # 제공자/모델을 바꾸는 명령. 새 설정을 받아 LLM과 에이전트를 다시 만든다.
                    selection = choose_provider(provider, llm.model)
                    if selection is None:
                        continue  # 취소된 경우 기존 설정 유지.

                    new_provider, new_model, llm_kwargs = selection
                    try:
                        new_llm = create_llm(
                            new_provider,
                            model=new_model,
                            **llm_kwargs,
                        )
                    except ValueError as exc:
                        print(f"Error: {exc}\n")
                        continue

                    # 성공했을 때만 실제 상태를 교체하고 설정을 저장한다.
                    provider = new_provider
                    model = new_model
                    llm = new_llm
                    save_settings(provider, llm.model)
                    # 새 LLM으로 에이전트를 재생성(대화는 새로 시작됨).
                    agent = Agent(
                        llm=llm,
                        system_prompt="You are a helpful AI assistant.",
                        tools=tools,
                    )
                    print(f"provider={provider}  model={llm.model}\n")
                else:
                    print(f"Unknown command: {user_input}\n")  # 모르는 명령.
                continue

            # 일반 입력이면 에이전트를 실행해 응답을 출력한다.
            try:
                response = agent.run(user_input)
                print(f"Agent: {response}\n")
            except Exception as exc:
                # 한 번의 호출 실패로 REPL 전체가 죽지 않도록 에러를 잡아 표시만 한다.
                print(f"Error: {exc}\n")
        finally:
            # 비대화형(파이프) 입력일 때는 자동완성 복원을 한 번만 하고 이후엔 생략.
            if not sys.stdin.isatty():
                restore_autocomplete()
                restore_autocomplete = lambda: None

    restore_autocomplete()  # 루프 종료 시 터미널 자동완성 설정을 원래대로 되돌린다.


if __name__ == "__main__":
    main()
