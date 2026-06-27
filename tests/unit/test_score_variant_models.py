"""Unit tests for ``ScoreAxis`` + ``ScoreVariant`` invariants (T005, data-model.md)."""

from __future__ import annotations

import pytest

from voders.scoreaug.models import ScoreAxis, ScoreVariant
from voders.scores.models import Note, Score


def _score(score_id: str) -> Score:
    return Score(score_id=score_id, notes=[Note(onset_s=0.0, offset_s=0.5, pitch_midi=60)])


def test_axis_enum_values():
    assert ScoreAxis.TRANSPOSE.value == "transpose"
    assert ScoreAxis.HUMANIZE.value == "humanize"
    assert ScoreAxis.VOLUME.value == "volume"


def test_variant_id_is_base_plus_transform():
    v = ScoreVariant(
        score=_score("song__t+12"),
        base_score_id="song",
        profile_id="p",
        axis=ScoreAxis.TRANSPOSE,
        transform="t+12",
        seed=7,
    )
    assert v.score.score_id == "song__t+12"
    assert v.axis is ScoreAxis.TRANSPOSE


def test_variant_rejects_mismatched_id():
    with pytest.raises(ValueError, match="must equal"):
        ScoreVariant(
            score=_score("song__hum0"),
            base_score_id="song",
            profile_id="p",
            axis=ScoreAxis.TRANSPOSE,
            transform="t+12",
            seed=7,
        )


def test_variant_rejects_reserved_separator_in_base_id():
    with pytest.raises(ValueError, match="reserved"):
        ScoreVariant(
            score=_score("a__b__vol"),
            base_score_id="a__b",
            profile_id="p",
            axis=ScoreAxis.VOLUME,
            transform="vol",
            seed=7,
        )
