"""Output helpers (currently text-only)."""

def speak(text: str, **_: object) -> bool:
	"""Stub TTS function (disabled). Returns False to indicate no-op."""
	return False


__all__ = ["speak"]