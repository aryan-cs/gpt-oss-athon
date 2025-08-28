import os
import sys
import threading
import time
from queue import Queue, Empty
from typing import Optional, Generator, Iterable

from engine import ChatSession, SYSTEM_PROMPT, MODEL
from logger import logger, styles

# TTS setup
_tts_engine = None
_tts_init_error = None

def _get_tts_engine():
    global _tts_engine, _tts_init_error
    if _tts_engine is not None:
        return _tts_engine
    
    try:
        import pyttsx3
        _tts_engine = pyttsx3.init()
        # Set volume to maximum for better audibility
        _tts_engine.setProperty('volume', 1.0)
        return _tts_engine
    except Exception as e:
        _tts_init_error = str(e)
        return None

def speak_text(text: str) -> bool:
    """Speak the given text using TTS. Returns True if successful.
    Use a fresh engine per call to avoid state issues across turns.
    """
    if not text.strip():
        return False
    try:
        import pyttsx3
        engine = pyttsx3.init(driverName='sapi5') if sys.platform.startswith('win') else pyttsx3.init()
        engine.setProperty('volume', 1.0)
        try:
            engine.setProperty('rate', 190)
        except Exception:
            pass
        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        return True
    except Exception:
        return False

def is_tts_available() -> bool:
    """Check if TTS is actually working by trying to initialize it."""
    return _get_tts_engine() is not None


def main() -> int:
    logger.system_log("Starting continuous chat. Type '/bye' to stop.")
    logger.set_model(MODEL)
    
    tts_available = is_tts_available()
    if tts_available:
        logger.system_log("TTS enabled - responses will be spoken after generation.")
    else:
        if _tts_init_error:
            logger.system_log(f"TTS not available: {_tts_init_error}")
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

        gen = session.ask_stream(user)
        
        # Capture the response while streaming for TTS
        response_chunks = []
        def capture_and_yield(stream: Iterable[str]) -> Generator[str, None, None]:
            for chunk in stream:
                response_chunks.append(chunk)
                yield chunk
        
        # Stream the response to console
        logger.llm_log(capture_and_yield(gen), stream=True)
        
        # Speak the full response after streaming completes
        if tts_available and response_chunks:
            full_response = "".join(response_chunks)
            if full_response.strip():
                speak_text(full_response)

    logger.system_log("Chat ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
