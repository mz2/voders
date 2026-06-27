"""Unit tests for the time-humanisation constraint solver (T016, FR-006/007)."""

from __future__ import annotations

from voders.config.models import HumanizeKnob
from voders.scoreaug.humanize import _humanize, humanize
from voders.scores.models import Note, Score


def _score(n: int = 5, *, gap: float = 0.08, dur: float = 0.45) -> Score:
    notes = []
    t = 0.2
    for i in range(n):
        notes.append(Note(onset_s=t, offset_s=t + dur, pitch_midi=60 + i, lyric=f"s{i}"))
        t += dur + gap
    return Score(score_id="s", notes=notes)


def _knob(**kw: object) -> HumanizeKnob:
    base: dict[str, object] = {
        "onset_sigma_s": 0.02,
        "duration_sigma_s": 0.02,
        "max_dev_s": 0.05,
        "draws": 1,
    }
    base.update(kw)
    return HumanizeKnob(**base)  # type: ignore[arg-type]


def test_jitter_present_and_within_budget():
    base = _score()
    out = humanize(base, _knob(onset_sigma_s=0.03, duration_sigma_s=0.03), sub_seed=7)
    devs = [abs(o.onset_s - b.onset_s) for o, b in zip(out.notes, base.notes, strict=True)] + [
        abs(o.offset_s - b.offset_s) for o, b in zip(out.notes, base.notes, strict=True)
    ]
    assert max(devs) <= 0.05 + 1e-9  # within max_dev_s budget
    assert any(d > 1e-6 for d in devs)  # jitter actually present


def test_output_is_valid_monophonic_with_positive_min_duration():
    out = humanize(_score(), _knob(onset_sigma_s=0.05, duration_sigma_s=0.05), sub_seed=3)
    assert out.is_monophonic()
    assert all(n.duration_s >= 0.01 - 1e-9 for n in out.notes)


def test_pitch_and_lyric_unchanged():
    base = _score()
    out = humanize(base, _knob(), sub_seed=11)
    assert [n.pitch_midi for n in out.notes] == [n.pitch_midi for n in base.notes]
    assert [n.lyric for n in out.notes] == [n.lyric for n in base.notes]


def test_deterministic_for_fixed_seed():
    a = humanize(_score(), _knob(), sub_seed=99)
    b = humanize(_score(), _knob(), sub_seed=99)
    assert [(n.onset_s, n.offset_s) for n in a.notes] == [(n.onset_s, n.offset_s) for n in b.notes]


def test_tight_score_forces_constraint_hits_but_stays_valid():
    # Zero-gap notes with large sigma force the projection to override draws (constraint hits),
    # yet the result must remain a valid monophonic score within budget.
    tight = _score(n=6, gap=0.0, dur=0.3)
    out, meta = _humanize(tight, _knob(onset_sigma_s=0.05, duration_sigma_s=0.05), sub_seed=5)
    assert out.is_monophonic()
    assert meta["constraint_hit"] >= 1
