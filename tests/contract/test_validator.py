"""Contract tests for the alignment validator (T015).

The validator is the single gate every sample passes (FR-006): it emits a ValidationVerdict whose
onset/offset/f0 checks reflect whether the rendered audio matches its paired score within tolerance.
"""

from __future__ import annotations

import numpy as np

from voders.config.models import ValidatorConfig
from voders.manifest.models import VerdictStatus
from voders.render.base import RenderRequest
from voders.render.deterministic import DeterministicLane
from voders.scores.models import Note, Score
from voders.validate.validator import Validator


def test_clean_render_is_accepted(small_score, donor_ah) -> None:
    """A deterministic render of a clean score validates as ACCEPTED with all checks True."""
    audio = (
        DeterministicLane().render(RenderRequest(score=small_score, voice=donor_ah, seed=0)).audio
    )
    verdict = Validator(ValidatorConfig(), min_note_ms=50.0).validate(audio, small_score)

    assert verdict.status == VerdictStatus.ACCEPTED
    assert verdict.onset_ok is True
    assert verdict.offset_ok is True
    assert verdict.f0_ok is True
    assert isinstance(verdict.max_onset_dev_ms, float)
    assert verdict.max_onset_dev_ms <= 50.0


def test_empty_score_is_rejected() -> None:
    """A score with no notes cannot be validated and is REJECTED (FR-006)."""
    verdict = Validator(ValidatorConfig(), min_note_ms=50.0).validate(
        np.zeros(0, dtype=np.float32), Score(score_id="empty", source="test", notes=[])
    )
    assert verdict.status == VerdictStatus.REJECTED


def test_wrong_pitch_fails_f0_check(small_score, donor_ah) -> None:
    """Audio at the score pitch but labelled an octave higher fails the f0 check (not ACCEPTED)."""
    audio = (
        DeterministicLane().render(RenderRequest(score=small_score, voice=donor_ah, seed=0)).audio
    )
    transposed = Score(
        score_id="transposed",
        source="test",
        notes=[
            Note(onset_s=n.onset_s, offset_s=n.offset_s, pitch_midi=n.pitch_midi + 12)
            for n in small_score.notes
        ],
    )
    verdict = Validator(ValidatorConfig(), min_note_ms=50.0).validate(audio, transposed)

    assert verdict.f0_ok is False
    assert verdict.status != VerdictStatus.ACCEPTED
