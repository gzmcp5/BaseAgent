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

load_dotenv()


PROVIDERS = [
    ("claude", "Anthropic Claude", "ANTHROPIC_API_KEY"),
    ("openai", "OpenAI", "OPENAI_API_KEY"),
    ("google", "Google Gemini", "GOOGLE_API_KEY"),
    ("ollama", "Ollama local", None),
    ("openrouter", "OpenRouter", "OPENROUTER_API_KEY"),
]
PROVIDER_KEYS = {name for name, _, _ in PROVIDERS}
COMMAND_OPTIONS = [
    ("/reset", "Clear conversation history"),
    ("/provider", "Choose LLM provider and model"),
    ("/quit", "Exit"),
]
COMMANDS = [command for command, _ in COMMAND_OPTIONS]
PROVIDER_NAMES = [name for name, _, _ in PROVIDERS]
STATE_PATH = Path(".baseagent.json")
COMPLETION_MODE = "command"
RESET = "\x1b[0m"
PROMPT_COLOR = "\x1b[1;36m"
INPUT_COLOR = "\x1b[38;5;250m"
MENU_COLOR = "\x1b[38;5;244m"
SELECTED_COLOR = "\x1b[1;32m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _color(text: str, color: str) -> str:
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
    try:
        import readline
    except ImportError:
        return lambda: None

    previous_completer = readline.get_completer()
    previous_delims = readline.get_completer_delims()
    readline.set_completer_delims(" \t\n")

    def complete(text: str, state: int) -> str | None:
        line = readline.get_line_buffer()
        stripped = line.lstrip()

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
    readline.parse_and_bind("tab: complete")

    def restore() -> None:
        readline.set_completer(previous_completer)
        readline.set_completer_delims(previous_delims)

    return restore


def _clear_lines(count: int) -> None:
    for _ in range(count):
        sys.stdout.write("\x1b[F\x1b[2K")


def _read_key() -> str:
    first = os.read(sys.stdin.fileno(), 1)
    if not first:
        return ""

    if first == b"\x1b":
        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not ready:
            return "\x1b"

        char = "\x1b" + os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if len(char) == 2 and char[1] not in {"[", "O"}:
            return char

        while len(char) < 8:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                break
            char += os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
            if "@" <= char[-1] <= "~":
                break
        return char

    lead = first[0]
    if lead < 0x80:
        expected = 1
    elif 0xC0 <= lead < 0xE0:
        expected = 2
    elif 0xE0 <= lead < 0xF0:
        expected = 3
    elif 0xF0 <= lead < 0xF8:
        expected = 4
    else:
        expected = 1

    data = first
    while len(data) < expected:
        data += os.read(sys.stdin.fileno(), expected - len(data))
    return data.decode(errors="ignore")


def _select_command(prompt: str) -> str:
    query = "/"
    selected = 0
    rendered_lines = 0

    def matches() -> list[tuple[str, str]]:
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

    while True:
        key = _read_key()
        if key in {"\r", "\n"}:
            current_matches = matches()
            command = current_matches[selected][0] if current_matches else query
            _clear_lines(rendered_lines)
            print(f"{_prompt_text(prompt)}{_input_text(command)}")
            return command
        if key in {"\x1b", "\x1b["}:
            _clear_lines(rendered_lines)
            return ""
        if key in {"\x1b[A", "\x1bOA"}:
            current_matches = matches()
            if current_matches:
                selected = (selected - 1) % len(current_matches)
        elif key in {"\x1b[B", "\x1bOB"}:
            current_matches = matches()
            if current_matches:
                selected = (selected + 1) % len(current_matches)
        elif key == "\x03":
            raise KeyboardInterrupt
        elif key == "\x04":
            raise EOFError
        elif key in {"\x7f", "\b"}:
            if len(query) == 1:
                _clear_lines(rendered_lines)
                return ""
            query = query[:-1]
            selected = 0
        elif key == "\t":
            current_matches = matches()
            if current_matches:
                query = current_matches[selected][0]
                selected = 0
        elif len(key) == 1 and key.isprintable() and not key.isspace():
            query += key
            selected = 0
        else:
            continue

        _clear_lines(rendered_lines)
        rendered_lines = render()


def select_option(
    title: str,
    options: list[tuple[str, str]],
    *,
    cancel_label: str = "Cancel",
) -> str | None:
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

    try:
        import termios
        import tty
    except ImportError:
        return None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    selected = 0
    rendered_lines = 0
    menu_options = options + [("", cancel_label)]

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
        tty.setcbreak(fd)
        rendered_lines = render()
        while True:
            key = _read_key()
            if key in {"\r", "\n"}:
                value = menu_options[selected][0]
                _clear_lines(rendered_lines)
                if value:
                    print(f"{_prompt_text('> ')}{_input_text(value)}")
                return value or None
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
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_repl_input(prompt: str) -> str:
    if not sys.stdin.isatty():
        return input(prompt).strip()

    try:
        import termios
        import tty
    except ImportError:
        return input(prompt).strip()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buffer = ""
    print(f"{_prompt_text(prompt)}{INPUT_COLOR}", end="", flush=True)

    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key()
            if key in {"\r", "\n"}:
                print(RESET if _supports_color() else "")
                return buffer.strip()
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x04":
                raise EOFError
            if key in {"\x7f", "\b"}:
                if buffer:
                    buffer = buffer[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if not buffer and key == "/":
                if _supports_color():
                    sys.stdout.write(RESET)
                return _select_command(prompt).strip()
            if key.startswith("\x1b"):
                continue

            buffer += key
            print(key, end="", flush=True)
    finally:
        if _supports_color():
            sys.stdout.write(RESET)
            sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def build_default_tools() -> ToolRegistry:
    tools = ToolRegistry()

    @tools.register(description="Return the current date and time.")
    def get_current_time() -> str:
        return datetime.now().isoformat()

    return tools


def load_saved_settings() -> dict[str, str]:
    if not STATE_PATH.exists():
        return {}
    try:
        with STATE_PATH.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    settings = {}
    provider = data.get("provider")
    model = data.get("model")
    if isinstance(provider, str) and provider in PROVIDER_KEYS:
        settings["provider"] = provider
    if isinstance(model, str) and model:
        settings["model"] = model
    return settings


def save_settings(provider: str, model: str) -> None:
    payload = {"provider": provider, "model": model}
    try:
        with STATE_PATH.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except OSError as exc:
        print(f"Warning: failed to save settings: {exc}\n")


def get_ollama_models() -> list[str]:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []

    models = data.get("models", [])
    names = [item.get("name", "") for item in models if item.get("name")]
    return sorted(names)


def choose_model(provider: str) -> str | None:
    if provider != "ollama":
        return input("Model (Enter for provider default): ").strip() or None

    models = get_ollama_models()
    if not models:
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
    print(f"Current provider={current_provider}  model={current_model}")
    set_completion_mode("provider")
    provider_options = []
    for name, label, env_key in PROVIDERS:
        key_label = "no API key" if env_key is None else env_key
        provider_options.append((name, f"{label}, {key_label}"))

    choice = select_option("Select provider:", provider_options)
    if choice is None:
        set_completion_mode("command")
        print("Provider unchanged.\n")
        return None

    selected = next((item for item in PROVIDERS if item[0] == choice), None)
    if selected is None:
        set_completion_mode("command")
        print("Invalid provider selection.\n")
        return None

    provider, _, env_key = selected
    set_completion_mode("none")
    model = choose_model(provider)
    kwargs = {}

    if env_key is not None and not os.environ.get(env_key):
        api_key = getpass.getpass(f"{env_key}: ").strip()
        if not api_key:
            set_completion_mode("command")
            print("API key is required for this provider.\n")
            return None
        kwargs["api_key"] = api_key

    set_completion_mode("command")
    return provider, model, kwargs


def main() -> None:
    restore_autocomplete = setup_autocomplete()
    saved_settings = load_saved_settings()
    provider = os.environ.get("LLM_PROVIDER") or saved_settings.get("provider", "claude")
    model = os.environ.get("LLM_MODEL") or saved_settings.get("model")

    try:
        llm = create_llm(provider, model=model)
    except ValueError as exc:
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

    while True:
        try:
            try:
                user_input = read_repl_input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lstrip("/").lower()
                if cmd == "quit":
                    print("Goodbye.")
                    break
                elif cmd == "reset":
                    agent.reset()
                    print("Conversation cleared.\n")
                elif cmd == "provider":
                    selection = choose_provider(provider, llm.model)
                    if selection is None:
                        continue

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

                    provider = new_provider
                    model = new_model
                    llm = new_llm
                    save_settings(provider, llm.model)
                    agent = Agent(
                        llm=llm,
                        system_prompt="You are a helpful AI assistant.",
                        tools=tools,
                    )
                    print(f"provider={provider}  model={llm.model}\n")
                else:
                    print(f"Unknown command: {user_input}\n")
                continue

            try:
                response = agent.run(user_input)
                print(f"Agent: {response}\n")
            except Exception as exc:
                print(f"Error: {exc}\n")
        finally:
            if not sys.stdin.isatty():
                restore_autocomplete()
                restore_autocomplete = lambda: None

    restore_autocomplete()


if __name__ == "__main__":
    main()
