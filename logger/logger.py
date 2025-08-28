"""Unified, colorized logger utilities with streaming support (ANSI codes)."""

from typing import Iterable, Union, Optional
from . import styles

MODEL = "qwen2.5:0.5b" # Default

def set_model(model: str):
    global MODEL
    MODEL = model


def user_log(message: str):
    print(f"{styles.RESET}{styles.BLUE}[{styles.WHITE}user{styles.BLUE}] > {styles.ITALICS}{message}{styles.RESET}")


def system_log(message: str):
    print(f"{styles.RESET}{styles.GRAY}[>] {message}{styles.RESET}")


def error_log(message: str):
    print(f"{styles.RESET}{styles.BOLD}{styles.UNDERLINE}{styles.RED}[!] {message}{styles.RESET}")


def llm_log(
    data: Union[str, Iterable[str]],
    *,
    stream: bool = False,
    prefix: Optional[str] = None,
    newline: bool = True,
):
    if prefix is None:
        prefix = f"{styles.RESET}{styles.GREEN}[{styles.WHITE}{MODEL}{styles.GREEN}] "
    if not stream:
        if isinstance(data, str):
            print(f"{styles.RESET}{styles.GREEN}{prefix}{data}{styles.RESET}" if prefix else f"{styles.RESET}{styles.GREEN}{data}{styles.RESET}")
        else:
            text = "".join(data)
            print(f"{styles.RESET}{styles.GREEN}{prefix}{text}{styles.RESET}" if prefix else f"{styles.RESET}{styles.GREEN}{text}{styles.RESET}")
        return

    print(f"{styles.RESET}{styles.GREEN}{prefix}{styles.RESET}", end="", flush=True)
    for chunk in data:
        print(f"{styles.RESET}{styles.GREEN}{chunk}{styles.RESET}", end="", flush=True)
    if newline:
        print()