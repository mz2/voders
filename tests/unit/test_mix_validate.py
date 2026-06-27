"""Unit tests for validate-on-mix (FR-004, FR-012; research Decisions 2, 5).

``validate_on_mix`` runs the existing ``Validator`` on the final mix and measures the largest
per-note timing shift via ``derive_labels``.
"""

from __future__ import annotations

import math

import numpy as np

from voders.config.models import ValidatorConfig
from voders.manifest.models import ValidationVerdict, VerdictStatus
from voders.render.deterministic import midi_to_hz
from voders.scores.models import Note, Score
from voders.validate.mix import MixValidation, validate_on_mix
from voders.validate.validator import Validator

SR = 22_050


def _vocal_and_score() -> tuple[np.ndarray, Score]:
    """A 1-note score and a sine 'vocal' at that pitch, voiced only over the note."""
    onset_s, offset_s = 0.2, 0.8
    pitch = 60
    total = np.zeros(SR, dtype=np.float32)  # ~1.0 s buffer
    start, end = int(onset_s * SR), int(offset_s * SR)
    t = np.arange(end - start) / SR
    total[start:end] = 0.5 * np.sin(2.0 * np.pi * midi_to_hz(pitch) * t).astype(np.float32)
    score = Score(
        score_id="score_mix",
        notes=[Note(onset_s=onset_s, offset_s=offset_s, pitch_midi=pitch)],
    )
    return total, score


def test_validate_on_mix_returns_structured_result_with_finite_shift() -> None:
    """The result is a MixValidation with a ValidationVerdict and a finite, non-negative shift."""
    vocal, score = _vocal_and_score()
    validator = Validator(ValidatorConfig())

    result = validate_on_mix(mix=vocal, label_score=score, validator=validator)

    assert isinstance(result, MixValidation)
    assert isinstance(result.verdict, ValidationVerdict)
    assert isinstance(result.max_note_shift_ms, float)
    assert result.max_note_shift_ms >= 0.0
    assert math.isfinite(result.max_note_shift_ms)


def test_snr_below_floor_quarantines_the_verdict() -> None:
    """An SNR below the validator floor forces a QUARANTINED verdict (FR-014)."""
    vocal, score = _vocal_and_score()
    validator = Validator(ValidatorConfig(snr_floor_db=0.0))

    result = validate_on_mix(mix=vocal, label_score=score, validator=validator, snr_db=-30.0)

    assert result.verdict.status == VerdictStatus.QUARANTINED
    assert math.isfinite(result.max_note_shift_ms)
