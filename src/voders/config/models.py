"""Run Config models (FR-016, contract: contracts/run-config.schema.yaml).

A run is specified by a single declarative YAML file: score set, voice pool, augmentation profiles,
master seed, and lane toggles. The resolved config is hashed into the manifest (``config_hash``) and
written to ``config.resolved.yaml`` so a run replays from the manifest + source scores.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from voders.lyrics.models import LyricModel, LyricSource
from voders.voices.models import Voice


class LaneToggle(BaseModel):
    """Per-lane enable flag plus free-form options passed to the lane (FR-015)."""

    model_config = ConfigDict(extra="allow")
    enabled: bool = False


class ValidatorConfig(BaseModel):
    """Validator tolerances (FR-006)."""

    onset_ms: float = 50.0
    offset_min_ms: float = 50.0
    offset_fraction: float = 0.20
    min_note_ms: float | None = None  # None -> learn from the annotation distribution (FR-018)
    min_note_percentile: float = 1.0
    f0_cents: float = 25.0
    f0_coverage: float = 0.80
    snr_floor_db: float = 0.0
    f0_method: str = "pyin_f0"  # pyin_f0 (CPU) | crepe_f0 (GPU) | auto
    f0_device: str = "auto"  # auto (cuda->xpu->dml->cpu) | cuda | xpu | dml | cpu (for crepe_f0)


class NeuralReproduction(BaseModel):
    """SC-009 tolerance for non-bit-deterministic neural/GPU lanes."""

    same_verdict: bool = True
    f0_cents: float = 25.0
    onset_ms: float = 10.0


class ReproductionConfig(BaseModel):
    """SC-009 reproduction tolerance per lane class."""

    deterministic: str = "bit_exact"  # deterministic + voice-conversion lanes
    neural: NeuralReproduction = Field(default_factory=NeuralReproduction)


class AugmentationProfileConfig(BaseModel):
    """An ordered, label-preserving augmentation profile (FR-005, data-model.md)."""

    profile_id: str
    steps: list[str] = Field(default_factory=list)
    params: dict[str, object] = Field(default_factory=dict)


class LyricsConfig(BaseModel):
    """Optional per-run lyric source selector and articulation settings (FR-013, data-model.md).

    Defaults keep a run lyric-free and byte-identical to the pre-feature behaviour (SC-001): the
    ``vowel`` source imports no lyric module. ``theme`` and ``model`` are valid only with the
    ``generated`` source; ``melisma`` accepts only ``per_note`` in v1.
    """

    source: LyricSource = LyricSource.VOWEL
    inventory: str = "en_cv"
    g2p_backend: str = "espeak"
    melisma: str = "per_note"
    theme: str | None = None
    model: LyricModel | None = None
    cache_dir: str = "lyrics"

    @model_validator(mode="after")
    def _check_source_constraints(self) -> LyricsConfig:
        has_generated_field = self.theme is not None or self.model is not None
        if has_generated_field and self.source != LyricSource.GENERATED:
            raise ValueError(
                "lyrics.theme/model are only valid with source='generated' "
                f"(got source={self.source.value!r})"
            )
        if self.melisma not in {"per_note"}:
            raise ValueError(
                f"lyrics.melisma={self.melisma!r} is not yet supported; "
                "'sustain_ties' is reserved and v1 accepts only 'per_note'"
            )
        return self


class RunConfig(BaseModel):
    """A single declarative run specification (FR-016)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    master_seed: int
    scores: str  # path or glob to .tsv scores (FR-001)
    output_root: str
    voices: list[Voice] = Field(default_factory=list)
    lanes: dict[str, LaneToggle] = Field(default_factory=dict)
    augmentation_profiles: list[AugmentationProfileConfig] = Field(default_factory=list)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    reproduction: ReproductionConfig = Field(default_factory=ReproductionConfig)
    lyrics: LyricsConfig = Field(default_factory=LyricsConfig)

    def enabled_lanes(self) -> list[str]:
        return [name for name, t in self.lanes.items() if t.enabled]

    def lane_options(self, name: str) -> dict[str, object]:
        toggle = self.lanes.get(name)
        if toggle is None:
            return {}
        return toggle.model_dump(exclude={"enabled"})
