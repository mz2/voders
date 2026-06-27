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
    f0_model: str = "full"  # CREPE capacity: full (accurate) | tiny (~5-10x faster, for gating)
    f0_decoder: str = "viterbi"  # CREPE decode: viterbi (smooth) | argmax (~15x faster, for gating)


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
    syllabifier: str = "en_rule"
    # espeak language codes to spread phonetic coverage across (multilingual eval match). One
    # language is chosen per sample, seeded. Default keeps runs English/single-language.
    languages: list[str] = Field(default_factory=lambda: ["en-us"])
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


class TransposeKnob(BaseModel):
    """Semitone transposition axis: one variant per offset (FR-004/005, data-model.md)."""

    model_config = ConfigDict(extra="forbid")

    offsets: list[int] = Field(default_factory=list)
    policy: str = "drop"  # drop | clamp
    window: tuple[int, int] = (0, 127)  # inclusive singable MIDI range

    @model_validator(mode="after")
    def _check(self) -> TransposeKnob:
        if self.policy not in {"drop", "clamp"}:
            raise ValueError(
                f"transpose.policy={self.policy!r} invalid; expected 'drop' or 'clamp'"
            )
        lo, hi = self.window
        if not (0 <= lo <= hi <= 127):
            raise ValueError(f"transpose.window={self.window!r} must satisfy 0 <= lo <= hi <= 127")
        return self


class HumanizeKnob(BaseModel):
    """Time-humanisation axis: seeded onset/duration jitter, N draws (FR-006/007, data-model.md)."""

    model_config = ConfigDict(extra="forbid")

    onset_sigma_s: float = 0.02
    duration_sigma_s: float = 0.02
    max_dev_s: float = 0.05
    draws: int = 1
    overlap: str = "forbid"  # v1 accepts only "forbid"
    min_dur_s: float = 0.01

    @model_validator(mode="after")
    def _check(self) -> HumanizeKnob:
        if self.draws < 1:
            raise ValueError(f"humanize_time.draws={self.draws} must be >= 1")
        if self.max_dev_s <= 0:
            raise ValueError(f"humanize_time.max_dev_s={self.max_dev_s} must be > 0")
        if self.min_dur_s <= 0:
            raise ValueError(f"humanize_time.min_dur_s={self.min_dur_s} must be > 0")
        if self.overlap != "forbid":
            raise ValueError(
                f"humanize_time.overlap={self.overlap!r} is not yet supported; "
                "v1 accepts only 'forbid'"
            )
        return self


class VolumeKnob(BaseModel):
    """Per-note dynamics axis: seeded gain within a dB range (FR-008/009, data-model.md)."""

    model_config = ConfigDict(extra="forbid")

    gain_db_range: tuple[float, float] = (-6.0, 6.0)
    distribution: str = "uniform"

    @model_validator(mode="after")
    def _check(self) -> VolumeKnob:
        lo, hi = self.gain_db_range
        if lo > hi:
            raise ValueError(
                f"volume.gain_db_range={self.gain_db_range!r} must satisfy low <= high"
            )
        if self.distribution not in {"uniform"}:
            raise ValueError(
                f"volume.distribution={self.distribution!r} is not yet supported; "
                "v1 accepts only 'uniform'"
            )
        return self


class ScoreAugmentationProfile(BaseModel):
    """One seeded score-augmentation profile; each enabled knob adds an axis (data-model.md).

    A knob left ``None`` disables that axis. A profile with all three ``None`` emits only the base
    score. Profiles are independently seeded from ``profile_id`` so order does not affect output.
    """

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    transpose: TransposeKnob | None = None
    humanize_time: HumanizeKnob | None = None
    volume: VolumeKnob | None = None


class AccompanimentOptions(BaseModel):
    """Accompaniment-stage options (contract: contracts/run-config-accompaniment.md).

    Parsed from the ``accompaniment`` lane toggle's free-form options (``LaneToggle`` is
    ``extra="allow"``), so enabling the stage is a non-breaking config addition (FR-008).
    """

    model_config = ConfigDict(extra="forbid")

    mode: str = "lego"  # lego (vocal-preserving) | complete (one-pass)
    backend: str = "fake"  # fake (CPU/CI) | acestep (GPU, `accomp` extra)
    model_id: str = ""
    license_policy: list[str] = Field(default_factory=lambda: ["MIT", "Apache-2.0", "CC-BY-4.0"])
    target_instrument: str = "sustained pad"
    free_time: bool = True
    bpm: float | None = None
    takes: int = 1
    target_snr_db: float | list[float] = 12.0

    @model_validator(mode="after")
    def _check(self) -> AccompanimentOptions:
        if self.mode not in ("lego", "complete"):
            raise ValueError(f"accompaniment.mode must be lego|complete, got {self.mode!r}")
        if self.free_time and self.bpm is not None:
            raise ValueError("accompaniment.bpm must be null when free_time is true (FR-005)")
        if self.takes < 1:
            raise ValueError(f"accompaniment.takes must be >= 1, got {self.takes}")
        return self


def parse_accompaniment_options(options: dict[str, object]) -> AccompanimentOptions:
    """Validate the ``accompaniment`` lane options into a typed model (FR-005/008)."""
    return AccompanimentOptions.model_validate(options)


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
    score_augmentation: list[ScoreAugmentationProfile] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_profile_ids(self) -> RunConfig:
        ids = [p.profile_id for p in self.score_augmentation]
        if len(ids) != len(set(ids)):
            raise ValueError(f"score_augmentation profile_id values must be unique (got {ids})")
        return self

    def enabled_lanes(self) -> list[str]:
        return [name for name, t in self.lanes.items() if t.enabled]

    def lane_options(self, name: str) -> dict[str, object]:
        toggle = self.lanes.get(name)
        if toggle is None:
            return {}
        return toggle.model_dump(exclude={"enabled"})
