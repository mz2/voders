"""Contract tests for the label-safe augmentation chain (T029, FR-005, FR-014, US3).

The augmentation chain post-processes an accepted base render: it adds reverb, codec band-limiting,
and accompaniment, but it MUST NOT move the vocal's onsets/offsets or shift its pitch (FR-005), and
it MUST be reproducible from the seed (FR-013). The validator gates the resulting SNR (FR-014).
"""

from __future__ import annotations

import math

import numpy as np

from voders.config.models import AugmentationProfileConfig, ValidatorConfig
from voders.manifest.models import VerdictStatus
from voders.render.augment import AugmentationChain
from voders.render.augmentor import Augmentor
from voders.render.base import RenderRequest
from voders.render.deterministic import DeterministicLane
from voders.validate.validator import Validator


def _base_render(small_score, donor_ah) -> tuple[np.ndarray, object]:
    result = DeterministicLane().render(RenderRequest(score=small_score, voice=donor_ah, seed=0))
    return result.audio, result.label_score


def _chain() -> AugmentationChain:
    profile = AugmentationProfileConfig(
        profile_id="pop_mix",
        steps=["reverb_ir", "codec", "accompaniment_mix"],
        params={"snr_db": [12.0]},
    )
    return AugmentationChain([profile], None)


def test_chain_satisfies_augmentor_protocol() -> None:
    chain = _chain()
    assert isinstance(chain, Augmentor)
    assert chain.profile_ids() == ["pop_mix"]


def test_apply_returns_float32_audio_and_finite_snr(small_score, donor_ah) -> None:
    """apply() returns float32 audio and a finite vocal-to-accompaniment SNR (FR-005/FR-014)."""
    audio, _ = _base_render(small_score, donor_ah)
    out, snr_db = _chain().apply(audio, "pop_mix", seed=7)

    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == audio.size  # label-safe: no length change -> onsets/offsets preserved
    assert snr_db is not None
    assert math.isfinite(snr_db)


def test_labels_preserved_after_augmentation(small_score, donor_ah) -> None:
    """The augmentation only returns audio; re-validating against the SAME score keeps labels.

    Onset/offset alignment must still hold for the augmented audio (FR-005).
    """
    audio, label_score = _base_render(small_score, donor_ah)
    out, snr_db = _chain().apply(audio, "pop_mix", seed=7)

    verdict = Validator(ValidatorConfig(snr_floor_db=0.0), min_note_ms=50.0).validate(
        out, label_score, snr_db=snr_db
    )
    assert verdict.onset_ok is True
    assert verdict.offset_ok is True


def test_apply_is_deterministic_from_seed(small_score, donor_ah) -> None:
    """Same seed -> byte-identical arrays (FR-013)."""
    audio, _ = _base_render(small_score, donor_ah)
    chain = _chain()
    out_a, snr_a = chain.apply(audio, "pop_mix", seed=99)
    out_b, snr_b = chain.apply(audio, "pop_mix", seed=99)

    assert np.array_equal(out_a, out_b)
    assert snr_a == snr_b


def test_low_snr_accompaniment_quarantines(small_score, donor_ah) -> None:
    """Accompaniment far above the vocal pushes SNR below the floor -> QUARANTINED (FR-014)."""
    audio, label_score = _base_render(small_score, donor_ah)
    profile = AugmentationProfileConfig(
        profile_id="loud_band",
        steps=["accompaniment_mix"],
        params={"snr_db": [-30.0]},
    )
    out, snr_db = AugmentationChain([profile]).apply(audio, "loud_band", seed=1)

    assert snr_db is not None and snr_db < 0.0
    verdict = Validator(ValidatorConfig(snr_floor_db=0.0), min_note_ms=50.0).validate(
        out, label_score, snr_db=snr_db
    )
    assert verdict.status == VerdictStatus.QUARANTINED
