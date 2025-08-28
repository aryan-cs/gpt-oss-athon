import sys
from typing import Optional

from engine import ChatSession, print_stream, SYSTEM_PROMPT, MODEL
from logger import logger, styles


def main() -> int:
    logger.system_log("Starting continuous chat. Type '/bye' to stop.")
    logger.set_model(MODEL)

    session = ChatSession(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        terminal_logging=True,
    )

    while True:
        try:
            user = input(f"{styles.RESET}{styles.BLUE}[{styles.WHITE}user{styles.BLUE}] > {styles.ITALICS}")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user.strip().lower() in {"/bye"}:
            break
        
        gen = session.ask_stream(user)
        print_stream(gen)

    logger.system_log("Chat ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
