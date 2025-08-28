import sys
import time
from typing import Iterable, Generator
from queue import Empty

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

_input_mod = None
try:
    _input_path = Path(__file__).resolve().parent / "io" / "input.py"
    _ispec = importlib.util.spec_from_file_location("app_input", str(_input_path))
    if _ispec and _ispec.loader:
        _input_mod = importlib.util.module_from_spec(_ispec)
        _ispec.loader.exec_module(_input_mod)
except Exception:
    _input_mod = None

def _voice_menu() -> None:
    if not _output_mod or not _input_mod:
        return
    try:
        voices = _output_mod.list_voices()
    except Exception:
        voices = []
    if not voices:
        try:
            _output_mod.speak("No voices found.")
        except Exception:
            pass
        return
    idx = 0
    total = len(voices)
    while True:
        v = voices[idx]
        vid = str(v.get('id'))
        vname = str(v.get('name'))
        try:
            _output_mod.speak(f"Voice {idx+1} of {total}: {vname}. Say next, previous, test, select, or quit.")
        except Exception:
            pass
        try:
            choice = _input_mod.recognize_once(timeout=5.0, phrase_time_limit=4.0)
        except Exception:
            choice = None
        cmd = (choice or "").strip().lower()
        if not cmd:
            continue
        if any(k in cmd for k in ("quit", "exit", "stop")):
            return
        if any(k in cmd for k in ("next", "forward")):
            idx = (idx + 1) % total
            continue
        if any(k in cmd for k in ("previous", "back", "prior")):
            idx = (idx - 1) % total
            continue
        if "test" in cmd:
            try:
                _output_mod.speak(f"Testing {vname}")
            except Exception:
                pass
            continue
        if "select" in cmd or "choose" in cmd:
            try:
                _output_mod.set_tts_config(voice_id=vid, voice_name=None)
                _output_mod.speak(f"Selected {vname}")
            except Exception:
                pass
            return


def main() -> int:
    logger.system_log("Starting continuous chat. Say 'Goodbye' to stop.")
    logger.set_model(MODEL)

    tts_available = bool(_output_mod and getattr(_output_mod, "is_tts_available", lambda: False)())
    if tts_available:
        logger.system_log("TTS enabled - responses will be spoken after generation.")
    else:
        logger.system_log("TTS not available - pyttsx3 not installed.")

    stt_available = bool(_input_mod and getattr(_input_mod, "is_stt_available", lambda: False)())
    if stt_available:
        try:
            if _input_mod and hasattr(_input_mod, "set_stt_config"):
                _input_mod.set_stt_config(engine="whisper", whisper_model="base.en", whisper_device="cpu")
                if hasattr(_input_mod, "preload_stt_models"):
                    _input_mod.preload_stt_models()
        except Exception:
            pass

        try:
            engine_name = getattr(_input_mod, "STT_ENGINE", "?")
            whisper_model = getattr(_input_mod, "STT_WHISPER_MODEL", None)
            if engine_name in {"whisper", "faster-whisper"} and whisper_model:
                logger.system_log(f"STT enabled ({engine_name}, model={whisper_model}) - always listening. Speak your requests and commands.")
            else:
                logger.system_log(f"STT enabled ({engine_name}) - always listening. Speak your requests and commands.")
        except Exception:
            logger.system_log("STT enabled - always listening. Speak your requests and commands.")
    else:
        logger.system_log("STT not available - 'speech_recognition' not installed or no mic.")
        logger.system_log("Voice-only mode requires STT. Exiting.")
        return 1

    session = ChatSession(
        system_prompt=SYSTEM_PROMPT,
        model=MODEL,
        terminal_logging=True,
    )

    stt_mode = bool(stt_available)
    stt_queue = None
    stt_stop = None
    if stt_available:
        try:
            start_q = getattr(_input_mod, 'start_background_queue', None)
            if start_q:
                stt_queue, stt_stop = start_q()
        except Exception:
            stt_queue, stt_stop = None, None

    def get_next_user_text() -> str:
        nonlocal stt_queue, stt_stop
        last_tick = 0.0
        try:
            if _input_mod:
                getattr(_input_mod, 'set_capture_enabled', lambda x: None)(True)
        except Exception:
            pass
        try:
            if stt_queue is not None:
                while True:
                    stt_queue.get_nowait()
        except Empty:
            pass
        except Exception:
            pass
        SILENCE_GAP = 1.5
        while True:
            if stt_mode and stt_queue is not None:
                try:
                    text = stt_queue.get(timeout=0.25)
                    if isinstance(text, str) and text.strip():
                        parts: list[str] = [text.strip()]
                        last_add = time.time()
                        while True:
                            try:
                                more = stt_queue.get(timeout=0.35)
                                if isinstance(more, str) and more.strip():
                                    parts.append(more.strip())
                                    last_add = time.time()
                                    try:
                                        sys.stdout.write("\r" + f"{styles.RESET}{styles.BLUE}[{styles.WHITE}user{styles.BLUE}] > {styles.ITALICS}" + " ".join(parts) + styles.RESET + "\x1b[K")
                                        sys.stdout.flush()
                                    except Exception:
                                        pass
                                    continue
                            except Empty:
                                pass
                            if time.time() - last_add >= SILENCE_GAP:
                                break
                        return " ".join(parts)
                except Empty:
                    now = time.time()
                    if now - last_tick >= 0.8:
                        try:
                            sys.stdout.write(".")
                            sys.stdout.flush()
                        except Exception:
                            pass
                        last_tick = now
                    continue
            else:
                try:
                    start_q = getattr(_input_mod, 'start_background_queue', None)
                    if start_q:
                        stt_queue, stt_stop = start_q()
                        continue
                except Exception:
                    pass
                time.sleep(0.25)

    print()
    while True:
        try:
            prompt_prefix = f"{styles.RESET}{styles.BLUE}[{styles.WHITE}user{styles.BLUE}] > {styles.ITALICS}"
            try:
                sys.stdout.write(prompt_prefix)
                sys.stdout.flush()
            except Exception:
                pass

            user = get_next_user_text()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        try:
            typed = user.strip()
            sys.stdout.write("\r" + prompt_prefix + typed + styles.RESET + "\x1b[K\n")
            sys.stdout.flush()
        except Exception:
            pass

        cmd = user.strip().lower()
        if cmd in {"/bye"} or "goodbye" in cmd or "exit" in cmd or "quit" in cmd:
            break

        if cmd.startswith("/voice") or "voice menu" in cmd or cmd == "voice":
            _voice_menu()
            continue

        if cmd.startswith("/mics") or "microphones" in cmd or "list microphones" in cmd:
            if not stt_available:
                logger.system_log("STT not available.")
                continue
            try:
                mics = _input_mod.list_microphones()
            except Exception:
                mics = []
            if not mics:
                logger.system_log("No microphones found.")
                continue
            for m in mics:
                logger.system_log(f"mic[{m.get('index')}] {m.get('name')}")
            continue

        if cmd.startswith("/micset") or "set microphone" in cmd or cmd.startswith("microphone "):
            if not stt_available:
                logger.system_log("STT not available.")
                continue
            idx = None
            import re
            m = re.search(r"(microphone|mic)\s*(\d+)", cmd)
            if m:
                try:
                    idx = int(m.group(2))
                except Exception:
                    idx = None
            if idx is None and cmd.startswith("/micset"):
                parts = user.strip().split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[1])
                    except Exception:
                        idx = None
            if idx is None:
                logger.system_log("Say: set microphone <number>.")
                continue
            try:
                _input_mod.set_stt_config(mic_index=idx)
                logger.system_log(f"Microphone set to index {idx}.")
            except Exception:
                logger.system_log("Failed to set microphone index.")
            continue

        try:
            if _input_mod:
                getattr(_input_mod, 'set_capture_enabled', lambda x: None)(False)
        except Exception:
            pass

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
            time.sleep(1.2)

        try:
            if _input_mod:
                getattr(_input_mod, 'set_capture_enabled', lambda x: None)(True)
        except Exception:
            pass

    logger.system_log("Chat ended.")
    try:
        if stt_stop:
            try:
                stt_stop(wait_for_stop=False)
            except TypeError:
                stt_stop()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
