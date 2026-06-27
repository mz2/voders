"""Contract tests for the deterministic WORLD lane (T014).

The deterministic lane is the load-bearing CPU baseline: its ``label_score`` equals the input
score by construction (FR-003), it never needs a GPU (FR-009), and importing it must not pull in
torch (FR-009).
"""

from __future__ import annotations

import sys

import numpy as np

from voders.constants import SAMPLE_RATE
from voders.render.base import RenderRequest, RenderResult
from voders.render.deterministic import DeterministicLane


def test_render_returns_result_with_label_equal_to_input(small_score, donor_ah) -> None:
    """render() returns a RenderResult whose label_score is the input score (FR-003)."""
    lane = DeterministicLane()
    result = lane.render(RenderRequest(score=small_score, voice=donor_ah, seed=0))

    assert isinstance(result, RenderResult)
    # label_score == input by construction: same notes, onsets, offsets, pitches.
    assert result.label_score == small_score
    assert result.label_score.notes == small_score.notes


def test_requires_gpu_is_false() -> None:
    """The deterministic lane is CPU-only (FR-009)."""
    assert DeterministicLane().requires_gpu() is False


def test_audio_is_mono_float32_and_length_matches_score(small_score, donor_ah) -> None:
    """Audio is non-empty 1-D float32 with length ≈ score duration * sample rate (FR-002)."""
    lane = DeterministicLane()
    audio = lane.render(RenderRequest(score=small_score, voice=donor_ah, seed=0)).audio

    assert audio.dtype == np.float32
    assert audio.ndim == 1  # mono
    assert audio.size > 0

    expected = small_score.duration_s * SAMPLE_RATE
    # The lane allocates round(duration * SR) + 1 samples; allow a small tolerance.
    assert abs(audio.size - expected) <= 0.02 * SAMPLE_RATE


def test_importing_lane_and_validator_does_not_import_torch() -> None:
    """Neither the deterministic lane nor the validator may import torch (FR-009)."""
    import voders.render.deterministic  # noqa: F401
    import voders.validate.validator  # noqa: F401

    assert "torch" not in sys.modules
