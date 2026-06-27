"""Voice enrollment registry with the ``consent_verified`` gate (FR-011)."""

from __future__ import annotations

from collections.abc import Iterable

from voders.voices.models import Voice, VoiceKind

__all__ = ["Voice", "VoiceKind", "VoiceRegistry", "ConsentRefusedError"]


class ConsentRefusedError(PermissionError):
    """Raised when a lane tries to use a voice whose consent is not verified (FR-011)."""


class VoiceRegistry:
    """Holds the enrolled voice pool and enforces the consent gate (FR-011, SC-008)."""

    def __init__(self, voices: Iterable[Voice] = ()) -> None:
        self._voices: dict[str, Voice] = {}
        for v in voices:
            self.enroll(v)

    def enroll(self, voice: Voice) -> None:
        self._voices[voice.voice_id] = voice

    def get(self, voice_id: str) -> Voice | None:
        return self._voices.get(voice_id)

    def __contains__(self, voice_id: object) -> bool:
        return voice_id in self._voices

    def __len__(self) -> int:
        return len(self._voices)

    def all(self) -> list[Voice]:
        return list(self._voices.values())

    def usable(self) -> list[Voice]:
        """Voices that pass the consent gate (FR-011)."""
        return [v for v in self._voices.values() if v.is_usable()]

    def require_usable(self, voice_id: str) -> Voice:
        """Return the voice, or raise if it is missing or fails the consent gate (FR-011).

        Missing-at-runtime and consent-refused are distinct: a missing voice is skipped+logged
        (Edge Cases), a present-but-unconsented voice is ``license_refused`` (FR-011, SC-008).
        """
        voice = self._voices.get(voice_id)
        if voice is None:
            raise KeyError(f"voice {voice_id!r} is not enrolled")
        if not voice.is_usable():
            raise ConsentRefusedError(
                f"voice {voice_id!r} has consent_verified=false/missing; refused (FR-011)"
            )
        return voice
