"""ProvenanceRecord + ValidationVerdict models (FR-008, FR-006).

Contract: contracts/manifest-record.schema.json.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class VerdictStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    FLAGGED = "flagged"
    LICENSE_REFUSED = "license_refused"


class ValidationVerdict(BaseModel):
    """Per-sample gate result; lives inside the provenance record (FR-006)."""

    status: VerdictStatus
    onset_ok: bool = False
    offset_ok: bool = False
    f0_ok: bool = False
    snr_db: float | None = None
    reason: str | None = None
    max_onset_dev_ms: float = 0.0

    @property
    def is_accepted(self) -> bool:
        return self.status == VerdictStatus.ACCEPTED


class AccompanimentProvenance(BaseModel):
    """Accompaniment-stage provenance; present iff ``lane == "accompaniment"`` (FR-007, FR-006a).

    Contract: contracts/manifest-accompaniment.md. ``vocal_bit_exact`` and ``stem_available`` are
    both true only for Lego mode (Complete re-encodes the vocal and emits no separable stem).
    """

    model_config = ConfigDict(extra="forbid")

    mode: str  # "lego" | "complete"
    model_id: str
    model_version: str = ""
    model_license: str = ""
    attribution_text: str | None = None  # required for CC-BY-class models (FR-006a)
    vocal_bit_exact: bool = False
    source_vocal_sample_id: str = ""
    takes_tried: int = 1
    max_note_shift_ms: float = 0.0  # measured note displacement on the mix (FR-012, SC-005)
    free_time: bool = False
    target_instrument: str = ""
    stem_available: bool = False


class ProvenanceRecord(BaseModel):
    """One line of ``manifest.jsonl`` — provenance for one attempted sample (FR-008, FR-006a)."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    score_id: str
    score_path: str
    audio_path: str
    lane: str
    voice_id: str
    augmentation_profile: str | None = None
    seed: int
    voice_license: str = ""
    consent_verified: bool = False
    config_hash: str
    verdict: ValidationVerdict
    lyric_source: str = "vowel"
    lyric_hash: str | None = None
    lyric_model: str | None = None
    lyric_model_license: str | None = None
    lyric_articulated: bool = False
    lyric_multisyllable_supplied: int = 0
    lyric_language: str = "en-us"
    accompaniment: AccompanimentProvenance | None = None
    notes: dict[str, object] = Field(default_factory=dict)
