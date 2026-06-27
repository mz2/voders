"""Voice (timbre) model with the consent gate (FR-011)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class VoiceKind(StrEnum):
    SVS_VOICEBANK = "svs_voicebank"
    VOICE_CONVERSION = "voice_conversion"
    DETERMINISTIC_DONOR = "deterministic_donor"


class Voice(BaseModel):
    """A named singer identity used by a renderer lane (data-model.md).

    A voice whose ``consent_verified`` is false or missing is refused by any lane (FR-011, SC-008).
    """

    voice_id: str
    kind: VoiceKind
    license: str = ""
    consent_verified: bool = False
    model_ref: str = ""

    def is_usable(self) -> bool:
        """True only when consent is verified (FR-011)."""
        return bool(self.consent_verified)
