"""Contract: RunConfig.score_augmentation + ProvenanceRecord score-aug fields (T003).

Covers contracts/config-manifest-store.md: the config block defaults to ``[]`` (absent ⇒
unchanged), knob/profile validation, and the additive provenance fields with backward-compatible
defaults under ``extra="forbid"``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voders.config.models import (
    HumanizeKnob,
    RunConfig,
    ScoreAugmentationProfile,
    TransposeKnob,
    VolumeKnob,
)
from voders.manifest.models import ProvenanceRecord, ValidationVerdict, VerdictStatus


def _run_config(**overrides: object) -> RunConfig:
    base: dict[str, object] = {
        "run_id": "r",
        "master_seed": 1,
        "scores": "evals/fixtures/scores",
        "output_root": "/tmp/x",
    }
    base.update(overrides)
    return RunConfig.model_validate(base)


def test_score_augmentation_defaults_to_empty():
    cfg = _run_config()
    assert cfg.score_augmentation == []


def test_score_augmentation_profile_round_trips():
    cfg = _run_config(
        score_augmentation=[
            {
                "profile_id": "all-axes",
                "transpose": {"offsets": [-12, 12], "policy": "drop", "window": [0, 127]},
                "humanize_time": {"max_dev_s": 0.05, "draws": 2},
                "volume": {"gain_db_range": [-6.0, 6.0]},
            }
        ]
    )
    prof = cfg.score_augmentation[0]
    assert prof.transpose is not None and prof.transpose.offsets == [-12, 12]
    assert prof.humanize_time is not None and prof.humanize_time.draws == 2
    assert prof.volume is not None and prof.volume.gain_db_range == (-6.0, 6.0)


def test_transpose_policy_must_be_drop_or_clamp():
    with pytest.raises(ValidationError):
        TransposeKnob(offsets=[1], policy="wrap")


def test_transpose_window_must_be_ordered_within_range():
    with pytest.raises(ValidationError):
        TransposeKnob(window=(50, 10))
    with pytest.raises(ValidationError):
        TransposeKnob(window=(0, 200))


def test_humanize_requires_positive_budget_and_draws():
    with pytest.raises(ValidationError):
        HumanizeKnob(draws=0)
    with pytest.raises(ValidationError):
        HumanizeKnob(max_dev_s=0.0)
    with pytest.raises(ValidationError):
        HumanizeKnob(min_dur_s=0.0)


def test_humanize_overlap_only_forbid_in_v1():
    with pytest.raises(ValidationError, match="not yet supported"):
        HumanizeKnob(overlap="allow")


def test_volume_range_must_be_ordered():
    with pytest.raises(ValidationError):
        VolumeKnob(gain_db_range=(6.0, -6.0))


def test_profile_ids_must_be_unique():
    with pytest.raises(ValidationError, match="unique"):
        _run_config(
            score_augmentation=[
                {"profile_id": "dup"},
                {"profile_id": "dup"},
            ]
        )


def test_empty_profile_is_valid():
    prof = ScoreAugmentationProfile(profile_id="noop")
    assert prof.transpose is None and prof.humanize_time is None and prof.volume is None


def test_provenance_record_score_aug_fields_default_to_original():
    rec = ProvenanceRecord(
        sample_id="s",
        score_id="s",
        score_path="",
        audio_path="",
        lane="deterministic",
        voice_id="v",
        seed=1,
        config_hash="sha256:x",
        verdict=ValidationVerdict(status=VerdictStatus.ACCEPTED),
    )
    assert rec.base_score_id is None
    assert rec.score_aug_profile is None
    assert rec.score_aug_axis is None
    assert rec.score_aug_transform is None
    assert rec.score_aug_seed is None
    assert rec.dynamics_applied is False


def test_provenance_record_still_forbids_extra():
    with pytest.raises(ValidationError):
        ProvenanceRecord(
            sample_id="s",
            score_id="s",
            score_path="",
            audio_path="",
            lane="deterministic",
            voice_id="v",
            seed=1,
            config_hash="sha256:x",
            verdict=ValidationVerdict(status=VerdictStatus.ACCEPTED),
            not_a_real_field=True,
        )
