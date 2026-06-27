"""Unit tests for the per-note volume axis (T020, FR-008/009)."""

from __future__ import annotations

from voders.config.models import VolumeKnob
from voders.scoreaug.volume import volume
from voders.scores.models import Note, Score


def _score() -> Score:
    return Score(
        score_id="s",
        notes=[
            Note(onset_s=i * 0.5, offset_s=i * 0.5 + 0.4, pitch_midi=60 + i, lyric=f"s{i}")
            for i in range(5)
        ],
    )


def test_assigns_gain_within_linear_range():
    out = volume(_score(), VolumeKnob(gain_db_range=(-6.0, 6.0)), sub_seed=7)
    lo_lin = 10.0 ** (-6.0 / 20.0)
    hi_lin = 10.0 ** (6.0 / 20.0)
    assert all(n.gain is not None for n in out.notes)
    assert all(lo_lin - 1e-9 <= n.gain <= hi_lin + 1e-9 for n in out.notes)  # type: ignore[operator]
    # Distinct per-note draws (not a single constant gain).
    assert len({round(n.gain, 6) for n in out.notes}) > 1  # type: ignore[arg-type]


def test_labels_unchanged():
    base = _score()
    out = volume(base, VolumeKnob(gain_db_range=(-6.0, 6.0)), sub_seed=7)
    assert [(n.onset_s, n.offset_s, n.pitch_midi, n.lyric) for n in out.notes] == [
        (n.onset_s, n.offset_s, n.pitch_midi, n.lyric) for n in base.notes
    ]


def test_deterministic_for_fixed_seed():
    a = volume(_score(), VolumeKnob(gain_db_range=(-6.0, 6.0)), sub_seed=42)
    b = volume(_score(), VolumeKnob(gain_db_range=(-6.0, 6.0)), sub_seed=42)
    assert [n.gain for n in a.notes] == [n.gain for n in b.notes]
