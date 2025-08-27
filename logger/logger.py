"""Unified, colorized logger utilities with streaming support (ANSI codes)."""

from typing import Iterable, Union

MODEL = "qwen2.5:0.5b" # Default

RESET = "\033[0m"
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
WHITE = "\033[37m"
GRAY = "\033[90m"

ITALICS = "\033[3m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

def set_model(model: str):
    global MODEL
    MODEL = model


def user_log(message: str):
    print(f"{BLUE}[{WHITE}user{BLUE}] > {ITALICS}{message}{RESET}")


def system_log(message: str):
    print(f"{GRAY}[>] {message}{RESET}")


def error_log(message: str):
    print(f"{BOLD}{UNDERLINE}{RED}[!] {message}{RESET}")


def llm_log(
    data: Union[str, Iterable[str]],
    *,
    stream: bool = False,
    prefix: str = f"[{WHITE}{MODEL}{GREEN}] > ",
    newline: bool = True,
):
    if not stream:
        if isinstance(data, str):
            print(f"{GREEN}{prefix}{data}{RESET}" if prefix else f"{GREEN}{data}{RESET}")
        else:
            text = "".join(data)
            print(f"{GREEN}{prefix}{text}{RESET}" if prefix else f"{GREEN}{text}{RESET}")
        return

    print(f"{GREEN}{prefix}{RESET}", end="", flush=True)
    for chunk in data:  # type: ignore[assignment]
        print(f"{GREEN}{chunk}{RESET}", end="", flush=True)
    if newline:
        print()