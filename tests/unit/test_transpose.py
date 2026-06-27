"""Unit tests for the transposition axis (T012, FR-004/005)."""

from __future__ import annotations

from voders.scoreaug.transpose import transpose
from voders.scores.models import Note, Score


def _score(pitches: list[int]) -> Score:
    return Score(
        score_id="s",
        notes=[
            Note(onset_s=i * 0.5, offset_s=i * 0.5 + 0.4, pitch_midi=p, lyric="x")
            for i, p in enumerate(pitches)
        ],
    )


def test_shifts_every_pitch_and_preserves_timing_and_lyric():
    out = transpose(_score([60, 62, 64]), 12, policy="drop", window=(0, 127))
    assert out is not None
    assert [n.pitch_midi for n in out.notes] == [72, 74, 76]
    base = _score([60, 62, 64])
    assert [(n.onset_s, n.offset_s, n.lyric) for n in out.notes] == [
        (n.onset_s, n.offset_s, n.lyric) for n in base.notes
    ]


def test_drop_policy_returns_none_when_out_of_window():
    assert transpose(_score([120]), 12, policy="drop", window=(0, 127)) is None
    assert transpose(_score([5]), -12, policy="drop", window=(0, 127)) is None


def test_drop_policy_respects_narrow_window():
    assert transpose(_score([60]), 0, policy="drop", window=(48, 72)) is not None
    assert transpose(_score([60]), 24, policy="drop", window=(48, 72)) is None


def test_clamp_policy_pins_into_window_and_never_exceeds_hard_bound():
    out = transpose(_score([120, 60]), 12, policy="clamp", window=(0, 127))
    assert out is not None
    assert [n.pitch_midi for n in out.notes] == [127, 72]
    assert all(0 <= n.pitch_midi <= 127 for n in out.notes)
