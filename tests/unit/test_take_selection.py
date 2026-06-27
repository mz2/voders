"""Unit tests for accompaniment take selection (FR-010, SC-001, SC-005).

The Accompanist generates up to ``takes`` takes, validates each on the final mix, and admits the
best-aligned validator-passing take. These tests exercise: takes are all tried; a passing Lego take
is admitted bit-exact with a retained stem; an all-fail run admits nothing yet still reports every
take tried; and an admitted take carries a finite, non-negative note-shift measurement.
"""

from __future__ import annotations

import math

import numpy as np

from voders.config.models import AccompanimentOptions, ValidatorConfig
from voders.manifest.models import VerdictStatus
from voders.render.accompaniment import AccompanimentResult, Accompanist
from voders.render.accompaniment_backend import FakeAccompanimentBackend
from voders.render.deterministic import midi_to_hz
from voders.scores.models import Note, Score
from voders.validate.validator import Validator

SR = 22_050


def _vocal_and_score() -> tuple[np.ndarray, Score]:
    """A 1-note (~0.6 s) sine 'vocal' at MIDI 60 in a ~1.0 s buffer at 22,050 Hz."""
    onset_s, offset_s, pitch = 0.2, 0.8, 60
    total = np.zeros(SR, dtype=np.float32)  # ~1.0 s buffer
    start, end = int(onset_s * SR), int(offset_s * SR)
    t = np.arange(end - start) / SR
    total[start:end] = 0.3 * np.sin(2.0 * np.pi * midi_to_hz(pitch) * t).astype(np.float32)
    score = Score(
        score_id="score_take",
        notes=[Note(onset_s=onset_s, offset_s=offset_s, pitch_midi=pitch)],
    )
    return total, score


def test_all_takes_tried_and_passing_lego_take_admitted_bit_exact() -> None:
    """takes=3 Lego: every take is tried; a passing take is admitted bit-exact with a stem."""
    vocal, score = _vocal_and_score()
    options = AccompanimentOptions(mode="lego", free_time=True, takes=3, target_snr_db=12.0)
    accompanist = Accompanist(FakeAccompanimentBackend(), options, Validator(ValidatorConfig()))

    result = accompanist.generate(vocal, score, source_vocal_sample_id="vocal-1", base_seed=7)

    assert isinstance(result, AccompanimentResult)
    assert result.provenance.takes_tried == 3
    assert result.admitted is True
    assert result.verdict.status == VerdictStatus.ACCEPTED
    assert result.stem is not None  # Lego retains a separable stem (SC-008)
    assert result.provenance.vocal_bit_exact is True  # vocal never re-encoded (SC-001)


def test_all_takes_fail_when_accompaniment_dominates() -> None:
    """A very negative target SNR lets the accompaniment swamp the vocal: nothing is admitted."""
    vocal, score = _vocal_and_score()
    options = AccompanimentOptions(mode="lego", free_time=True, takes=3, target_snr_db=-40.0)
    accompanist = Accompanist(FakeAccompanimentBackend(), options, Validator(ValidatorConfig()))

    result = accompanist.generate(vocal, score, source_vocal_sample_id="vocal-1", base_seed=7)

    assert result.admitted is False
    assert result.verdict.status != VerdictStatus.ACCEPTED
    assert result.provenance.takes_tried == 3  # all takes still tried (FR-010)


def test_admitted_result_reports_finite_non_negative_note_shift() -> None:
    """The best-aligned admitted take carries a finite, non-negative note-shift (SC-005)."""
    vocal, score = _vocal_and_score()
    options = AccompanimentOptions(mode="lego", free_time=True, takes=3, target_snr_db=12.0)
    accompanist = Accompanist(FakeAccompanimentBackend(), options, Validator(ValidatorConfig()))

    result = accompanist.generate(vocal, score, source_vocal_sample_id="vocal-1", base_seed=7)

    assert result.admitted is True
    shift = result.provenance.max_note_shift_ms
    assert isinstance(shift, float)
    assert shift >= 0.0
    assert math.isfinite(shift)
