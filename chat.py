import sys
from typing import Iterable, Generator

from engine import ChatSession, SYSTEM_PROMPT, MODEL
from logger import logger, styles

import importlib.util
from pathlib import Path

_output_mod = None
try:
    _output_path = Path(__file__).resolve().parent / "io" / "output.py"
    _spec = importlib.util.spec_from_file_location("app_output", str(_output_path))
    if _spec and _spec.loader:
        _output_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_output_mod)
except Exception:
    _output_mod = None


def main() -> int:
    logger.system_log("Starting continuous chat. Type '/bye' to stop.")
    logger.set_model(MODEL)

    tts_available = bool(_output_mod and getattr(_output_mod, "is_tts_available", lambda: False)())
    if tts_available:
        logger.system_log("TTS enabled - responses will be spoken after generation.")
    else:
        logger.system_log("TTS not available - pyttsx3 not installed.")

    session = ChatSession(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        terminal_logging=True,
    )

    print()
    while True:
        try:
            user = input(f"{styles.RESET}{styles.BLUE}[{styles.WHITE}user{styles.BLUE}] > {styles.ITALICS}")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        cmd = user.strip().lower()
        if cmd in {"/bye"}:
            break

        if cmd == "/voice":
            if not _output_mod or not getattr(_output_mod, "is_tts_available", lambda: False)():
                logger.system_log("TTS not available - cannot open voice menu.")
                continue
            try:
                voices = _output_mod.list_voices()
            except Exception:
                voices = []
            if not voices:
                logger.system_log("No voices found.")
                continue

            idx = 0
            total = len(voices)
            logger.system_log("Voice menu: (n)ext, (p)rev, (t)est, (s)elect, (q)uit")
            while True:
                v = voices[idx]
                vid = str(v.get('id'))
                vname = str(v.get('name'))
                logger.system_log(f"[{idx+1}/{total}] {vname} | {vid}")
                try:
                    choice = input(f"{styles.RESET}{styles.GRAY}[voice] (n/p/t/s/q) > {styles.ITALICS}").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if choice in {"q", "quit"}:
                    break
                if choice in {"n", "next"}:
                    idx = (idx + 1) % total
                    continue
                if choice in {"p", "prev", "previous"}:
                    idx = (idx - 1) % total
                    continue
                if choice in {"t", "test"}:
                    try:
                        _output_mod.speak(f"Testing {vname}")
                    except Exception:
                        logger.system_log("Failed to speak test sample.")
                    continue
                if choice in {"s", "select"}:
                    try:
                        _output_mod.set_tts_config(voice_id=vid, voice_name=None)
                        logger.system_log(f"Selected voice: {vname}")
                    except Exception:
                        logger.system_log("Failed to set voice.")
                    break
            continue

        gen = session.ask_stream(user)

        collected: list[str] = []
        def capture_and_yield(stream: Iterable[str]) -> Generator[str, None, None]:
            for chunk in stream:
                collected.append(chunk)
                yield chunk

        logger.llm_log(capture_and_yield(gen), stream=True)

        if tts_available and collected and _output_mod:
            full_text = "".join(collected)
            if full_text.strip():
                try:
                    _output_mod.speak(full_text)
                except Exception:
                    pass

    logger.system_log("Chat ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
