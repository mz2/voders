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


# --- vocal-preserving (Lego) timing re-judgement (research Decision 2) ---


def _onset_reject() -> ValidationVerdict:
    # Rejected purely on an on-mix onset-detection artifact; pitch coverage is fine.
    return ValidationVerdict(
        status=VerdictStatus.REJECTED,
        onset_ok=False,
        offset_ok=True,
        f0_ok=True,
        reason="note 0: onset off by 107 ms (> 50 ms)",
        max_onset_dev_ms=107.0,
    )


def test_preserved_timing_admits_unmoved_vocal() -> None:
    """Lego: a small measured shift overrides an on-mix onset-detection false reject."""
    from voders.validate.mix import _preserved_timing_verdict

    out = _preserved_timing_verdict(_onset_reject(), shift_ms=15.0, onset_tol_ms=50.0)
    assert out.status == VerdictStatus.ACCEPTED
    assert out.onset_ok and out.offset_ok


def test_preserved_timing_keeps_reject_when_note_moved() -> None:
    """Lego: a measured shift beyond tolerance keeps the rejection (the note really moved)."""
    from voders.validate.mix import _preserved_timing_verdict

    out = _preserved_timing_verdict(_onset_reject(), shift_ms=80.0, onset_tol_ms=50.0)
    assert out.status == VerdictStatus.REJECTED


def test_preserved_timing_keeps_reject_when_masked() -> None:
    """Lego: a masked vocal (f0 coverage fails) stays rejected even with a small shift."""
    from voders.validate.mix import _preserved_timing_verdict

    masked = _onset_reject().model_copy(update={"f0_ok": False, "reason": "note 0: f0 in tune 10%"})
    out = _preserved_timing_verdict(masked, shift_ms=10.0, onset_tol_ms=50.0)
    assert out.status == VerdictStatus.REJECTED
