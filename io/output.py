from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

TTS_ENABLED: bool = True
TTS_DRIVER: Optional[str] = 'sapi5' if sys.platform.startswith('win') else None
TTS_RATE: int = 190
TTS_VOLUME: float = 1.0
TTS_VOICE_ID: Optional[str] = None
TTS_VOICE_NAME: Optional[str] = None


def is_tts_available() -> bool:
	if not TTS_ENABLED:
		return False
	try:
		import pyttsx3
		return True
	except Exception:
		return False


def list_voices() -> List[Dict[str, Any]]:
	try:
		import pyttsx3
		engine = pyttsx3.init(driverName=TTS_DRIVER) if TTS_DRIVER else pyttsx3.init()
		voices = []
		for v in engine.getProperty('voices') or []:
			voices.append({
				'id': getattr(v, 'id', None),
				'name': getattr(v, 'name', None),
				'languages': getattr(v, 'languages', None),
				'gender': getattr(v, 'gender', None),
				'age': getattr(v, 'age', None),
			})
		try:
			engine.stop()
		except Exception:
			pass
		return voices
	except Exception:
		return []


def set_tts_config(
	*,
	enabled: Optional[bool] = None,
	driver: Optional[str] = None,
	rate: Optional[int] = None,
	volume: Optional[float] = None,
	voice_id: Optional[str] = None,
	voice_name: Optional[str] = None,
) -> None:
	global TTS_ENABLED, TTS_DRIVER, TTS_RATE, TTS_VOLUME, TTS_VOICE_ID, TTS_VOICE_NAME
	if enabled is not None:
		TTS_ENABLED = bool(enabled)
	if driver is not None:
		TTS_DRIVER = driver
	if rate is not None:
		TTS_RATE = int(rate)
	if volume is not None:
		TTS_VOLUME = float(volume)
	if voice_id is not None:
		TTS_VOICE_ID = voice_id
	if voice_name is not None:
		TTS_VOICE_NAME = voice_name


def _select_voice(engine, *, voice_id: Optional[str], voice_name: Optional[str]) -> None:
	try:
		voices = engine.getProperty('voices') or []
		if voice_id:
			for v in voices:
				if getattr(v, 'id', None) == voice_id:
					engine.setProperty('voice', v.id)
					return
		if voice_name:
			target = voice_name.lower()
			for v in voices:
				name = (getattr(v, 'name', '') or '').lower()
				vid = (getattr(v, 'id', '') or '').lower()
				if target in name or target in vid:
					engine.setProperty('voice', v.id)
					return
	except Exception:
		pass


def speak(
	text: str,
	*,
	rate: Optional[int] = None,
	volume: Optional[float] = None,
	voice_id: Optional[str] = None,
	voice_name: Optional[str] = None,
) -> bool:
	if not TTS_ENABLED:
		return False
	if not isinstance(text, str) or not text.strip():
		return False
	try:
		import pyttsx3
		engine = pyttsx3.init(driverName=TTS_DRIVER) if TTS_DRIVER else pyttsx3.init()
		try:
			engine.setProperty('volume', float(TTS_VOLUME if volume is None else volume))
		except Exception:
			pass
		try:
			engine.setProperty('rate', int(TTS_RATE if rate is None else rate))
		except Exception:
			pass
		_select_voice(
			engine,
			voice_id=voice_id if voice_id is not None else TTS_VOICE_ID,
			voice_name=voice_name if voice_name is not None else TTS_VOICE_NAME,
		)
		engine.say(text)
		engine.runAndWait()
		try:
			engine.stop()
		except Exception:
			pass
		return True
	except Exception:
		return False


__all__ = [
	"speak",
	"is_tts_available",
	"list_voices",
	"set_tts_config",
	"TTS_ENABLED",
	"TTS_DRIVER",
	"TTS_RATE",
	"TTS_VOLUME",
	"TTS_VOICE_ID",
	"TTS_VOICE_NAME",
]