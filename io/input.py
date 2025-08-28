"""Speech-to-Text (STT) input utilities.

This module provides optional STT using the `speech_recognition` package.
If not installed, functions fail gracefully.

Exports:
- is_stt_available() -> bool
- list_microphones() -> list[dict]
- set_stt_config(...)
- recognize_once(...) -> Optional[str]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing import Tuple
import os
import warnings

# Global STT settings
STT_ENABLED: bool = True
STT_ENGINE: str = "whisper"  # use OpenAI Whisper by default; fallback if unavailable
STT_MIC_INDEX: Optional[int] = None  # default system mic when None
STT_LANGUAGE: str = "en-US"  # used by online recognizers (e.g., Google)
STT_ENERGY_THRESHOLD: Optional[int] = None  # static energy threshold when set
STT_DYNAMIC_ENERGY: bool = True  # let recognizer auto-adjust if True
STT_ADJUST_DURATION: float = 0.25  # seconds to calibrate ambient noise
CAPTURE_ENABLED: bool = True  # gate to drop recognitions when disabled

# Whisper settings
STT_WHISPER_MODEL: str = "base.en"  # tiny.en for English per request
# Other models:
# - tiny.en
# - base.en
# - small.en
# - medium.en

STT_WHISPER_DEVICE: str = "cpu"  # "cpu" or "cuda"

# Internal model caches
_WHISPER_MODEL = None  # type: ignore
_FASTER_WHISPER_MODEL = None  # type: ignore

# Reduce noisy library warnings/logging and progress bars
os.environ.setdefault("CT2_VERBOSE", "0")  # ctranslate2 verbosity
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"ctranslate2(\..*)?$")
warnings.filterwarnings("ignore", category=UserWarning, module=r"faster_whisper(\..*)?$")
warnings.filterwarnings("ignore", message=r"The current model is English-only.*")


def is_stt_available() -> bool:
	global STT_ENGINE
	if not STT_ENABLED:
		return False
	try:
		if STT_ENGINE == "whisper":
			import numpy  # noqa: F401
			import whisper  # type: ignore
			return True
		elif STT_ENGINE == "faster-whisper":
			import numpy  # noqa: F401
			from faster_whisper import WhisperModel  # noqa: F401
			return True
		else:
			import importlib
			importlib.import_module('speech_recognition')
			return True
	except Exception:
		# Try faster-whisper before falling back to SR
		try:
			import numpy  # noqa: F401
			from faster_whisper import WhisperModel  # noqa: F401
			STT_ENGINE = "faster-whisper"
			return True
		except Exception:
			try:
				import importlib
				importlib.import_module('speech_recognition')
				STT_ENGINE = "sr"
				return True
			except Exception:
				return False


def list_microphones() -> List[Dict[str, Any]]:
	"""List available microphone devices.

	Returns a list of dicts: { index, name }.
	"""
	try:
		import importlib
		sr = importlib.import_module('speech_recognition')
		names = sr.Microphone.list_microphone_names() or []
		return [{"index": i, "name": n} for i, n in enumerate(names)]
	except Exception:
		return []


def set_stt_config(
	*,
	enabled: Optional[bool] = None,
	engine: Optional[str] = None,
	mic_index: Optional[int] = None,
	language: Optional[str] = None,
	energy_threshold: Optional[int] = None,
	dynamic_energy: Optional[bool] = None,
	adjust_duration: Optional[float] = None,
	whisper_model: Optional[str] = None,
	whisper_device: Optional[str] = None,
) -> None:
	"""Update global STT configuration variables."""
	global STT_ENABLED, STT_ENGINE, STT_MIC_INDEX, STT_LANGUAGE, STT_ENERGY_THRESHOLD, STT_DYNAMIC_ENERGY, STT_ADJUST_DURATION, STT_WHISPER_MODEL, STT_WHISPER_DEVICE
	if enabled is not None:
		STT_ENABLED = bool(enabled)
	if engine is not None:
		STT_ENGINE = engine
	if mic_index is not None:
		STT_MIC_INDEX = int(mic_index)
	if language is not None:
		STT_LANGUAGE = language
	if energy_threshold is not None:
		STT_ENERGY_THRESHOLD = int(energy_threshold)
	if dynamic_energy is not None:
		STT_DYNAMIC_ENERGY = bool(dynamic_energy)
	if adjust_duration is not None:
		STT_ADJUST_DURATION = float(adjust_duration)
	if whisper_model is not None:
		STT_WHISPER_MODEL = str(whisper_model)
	if whisper_device is not None:
		STT_WHISPER_DEVICE = str(whisper_device)


def set_capture_enabled(enabled: bool) -> None:
	"""Enable/disable pushing recognized text into the queue.

	When disabled, background recognition still runs but results are ignored.
	"""
	global CAPTURE_ENABLED
	CAPTURE_ENABLED = bool(enabled)


def _recognize_with_available_engines(recognizer, audio, *, language: str) -> Optional[str]:
	"""Try offline (Sphinx) first, then online (Google) if available."""
	# Offline: PocketSphinx
	try:
		import importlib
		importlib.import_module('pocketsphinx')
		try:
			return recognizer.recognize_sphinx(audio, language=language)
		except Exception:
			pass
	except Exception:
		pass

	# Online: Google Web Speech API (no key needed, but requires internet)
	try:
		return recognizer.recognize_google(audio, language=language)
	except Exception:
		return None


def _sr_audio_to_float32(audio_obj) -> Optional[Tuple["np.ndarray", int]]:  # type: ignore
	"""Convert SpeechRecognition.AudioData to float32 numpy array and sample rate.

	Returns tuple (audio_float32, sample_rate) or None on failure.
	"""
	try:
		import numpy as np
		raw = audio_obj.get_raw_data(convert_rate=16000, convert_width=2)
		# int16 PCM -> float32 [-1, 1]
		arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
		return arr, 16000
	except Exception:
		return None


def _whisper_transcribe_array(audio_arr, sample_rate: int, *, language: Optional[str]) -> Optional[str]:
	"""Transcribe using OpenAI Whisper if available, else try faster-whisper."""
	# Try openai-whisper first
	try:
		import numpy as np  # noqa: F401
		import whisper
		global _WHISPER_MODEL
		if _WHISPER_MODEL is None:
			_WHISPER_MODEL = whisper.load_model(STT_WHISPER_MODEL, device=STT_WHISPER_DEVICE)
		lang = (language or STT_LANGUAGE)
		if isinstance(lang, str) and '-' in lang:
			lang = lang.split('-')[0]
		result = _WHISPER_MODEL.transcribe(audio_arr, language=lang, fp16=False if STT_WHISPER_DEVICE == 'cpu' else None)
		text = (result.get('text') or '').strip()
		return text or None
	except Exception:
		pass
	# Fallback to faster-whisper
	try:
		from faster_whisper import WhisperModel
		import numpy as np  # noqa: F401
		global _FASTER_WHISPER_MODEL
		if _FASTER_WHISPER_MODEL is None:
			_FASTER_WHISPER_MODEL = WhisperModel(STT_WHISPER_MODEL, device=STT_WHISPER_DEVICE, compute_type="int8" if STT_WHISPER_DEVICE == 'cpu' else "float16")
		lang = (language or STT_LANGUAGE)
		if isinstance(lang, str) and '-' in lang:
			lang = lang.split('-')[0]
		segments, info = _FASTER_WHISPER_MODEL.transcribe(audio_arr, language=lang, beam_size=1)
		parts = []
		for seg in segments:
			try:
				parts.append(seg.text)
			except Exception:
				pass
		text = " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
		return text or None
	except Exception:
		return None


def preload_stt_models() -> bool:
	"""Load the configured STT model(s) at startup to avoid runtime noise and latency."""
	try:
		# Create a tiny silent buffer for a no-op warm-up if needed
		import numpy as np
		silent = (np.zeros(16000, dtype=np.float32), 16000)
		# Warm the chosen engine
		if STT_ENGINE in {"whisper", "faster-whisper"}:
			_ = _whisper_transcribe_array(silent[0], silent[1], language=STT_LANGUAGE)
		return True
	except Exception:
		return False


def recognize_once(
	*,
	timeout: Optional[float] = None,
	phrase_time_limit: Optional[float] = None,
	language: Optional[str] = None,
) -> Optional[str]:
	"""Listen from the microphone once and return recognized text or None.

	- timeout: maximum seconds to wait for phrase start (None = wait indefinitely)
	- phrase_time_limit: limit max seconds of recorded phrase
	- language: BCP-47 code like 'en-US'; defaults to STT_LANGUAGE
	"""
	if not is_stt_available():
		return None
	try:
		import importlib
		sr = importlib.import_module('speech_recognition')
		r = sr.Recognizer()
		# Configure energy thresholds
		if STT_ENERGY_THRESHOLD is not None:
			r.energy_threshold = int(STT_ENERGY_THRESHOLD)
		r.dynamic_energy_threshold = bool(STT_DYNAMIC_ENERGY)

		# Open microphone
		with sr.Microphone(device_index=STT_MIC_INDEX, sample_rate=16000) as source:
			# Optional ambient noise calibration
			try:
				if STT_DYNAMIC_ENERGY and STT_ADJUST_DURATION > 0:
					r.adjust_for_ambient_noise(source, duration=STT_ADJUST_DURATION)
			except Exception:
				pass

			audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

		# Recognize
		if STT_ENGINE in {"whisper", "faster-whisper"}:
			arr_sr = _sr_audio_to_float32(audio)
			if arr_sr is None:
				return None
			arr, sr_val = arr_sr
			text = _whisper_transcribe_array(arr, sr_val, language=(language or STT_LANGUAGE))
		else:
			text = _recognize_with_available_engines(r, audio, language=(language or STT_LANGUAGE))
		return text
	except sr.WaitTimeoutError:
		return None
	except Exception:
		return None


__all__ = [
	"is_stt_available",
	"list_microphones",
	"set_stt_config",
	"set_capture_enabled",
	"preload_stt_models",
	"recognize_once",
	"start_background_queue",
	# Globals
	"STT_ENABLED",
	"STT_ENGINE",
	"STT_MIC_INDEX",
	"STT_LANGUAGE",
	"STT_ENERGY_THRESHOLD",
	"STT_DYNAMIC_ENERGY",
	"STT_ADJUST_DURATION",
	"STT_WHISPER_MODEL",
	"STT_WHISPER_DEVICE",
]


def start_background_queue(
	*,
	mic_index: Optional[int] = None,
	language: Optional[str] = None,
	phrase_time_limit: Optional[float] = 10.0,
):
	"""Start an always-listening background recognizer.

	Returns a tuple of (queue, stop_function).
	The queue will receive recognized strings.
	The stop_function can be called with wait_for_stop: bool = False.
	"""
	if not is_stt_available():
		raise RuntimeError("STT not available")
	import importlib
	from queue import Queue
	sr = importlib.import_module('speech_recognition')

	r = sr.Recognizer()
	if STT_ENERGY_THRESHOLD is not None:
		r.energy_threshold = int(STT_ENERGY_THRESHOLD)
	r.dynamic_energy_threshold = bool(STT_DYNAMIC_ENERGY)

	q: Queue[str] = Queue()

	def callback(recognizer, audio):
		try:
			if not CAPTURE_ENABLED:
				return
			if STT_ENGINE in {"whisper", "faster-whisper"}:
				arr_sr = _sr_audio_to_float32(audio)
				if arr_sr is None:
					return
				arr, sr_val = arr_sr
				text = _whisper_transcribe_array(arr, sr_val, language=(language or STT_LANGUAGE))
			else:
				text = _recognize_with_available_engines(recognizer, audio, language=(language or STT_LANGUAGE))
			if isinstance(text, str) and text.strip():
				q.put(text)
		except Exception:
			pass

	# Prepare mic and start
	mic_idx = STT_MIC_INDEX if mic_index is None else mic_index
	mic = sr.Microphone(device_index=mic_idx, sample_rate=16000)
	# Calibrate on an opened source, but start background listener on the mic object
	try:
		with mic as source:
			try:
				if STT_DYNAMIC_ENERGY and STT_ADJUST_DURATION > 0:
					r.adjust_for_ambient_noise(source, duration=STT_ADJUST_DURATION)
			except Exception:
				pass
	except Exception:
		pass
	# Start background listener (it will manage opening/closing the mic stream)
	stop_listening = r.listen_in_background(mic, callback, phrase_time_limit=phrase_time_limit)

	def stop_fn(wait_for_stop: bool = False):
		try:
			stop_listening(wait_for_stop)
		except TypeError:
			# Older SR versions without parameter
			stop_listening()

	return q, stop_fn