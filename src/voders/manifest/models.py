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
    # Score-domain augmentation lineage (feature 003). All default to "original/none", so a base
    # record (or any feature-off run) serialises with these at their inert defaults (SC-001).
    base_score_id: str | None = None  # the originating base score (None => this IS an original)
    score_aug_profile: str | None = None  # ScoreAugmentationProfile id
    score_aug_axis: str | None = None  # transpose | humanize | volume
    score_aug_transform: str | None = None  # t+12 | hum0 | vol — the applied transform
    score_aug_seed: int | None = None  # the variant seed (reproducibility audit)
    dynamics_applied: bool = False  # did the lane honour a per-note gain (volume axis)
    notes: dict[str, object] = Field(default_factory=dict)
