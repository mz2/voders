"""Unit tests for long-score segmentation into singable phrases (Klangio ingest, issue #9)."""

from __future__ import annotations

from voders.scores.models import Note, Score


def _line(start: float, n: int, pitch: int, dur: float = 0.4, gap: float = 0.05) -> list[Note]:
    notes = []
    t = start
    for _ in range(n):
        notes.append(Note(onset_s=round(t, 3), offset_s=round(t + dur, 3), pitch_midi=pitch))
        t += dur + gap
    return notes


def test_splits_at_breath_rest() -> None:
    from voders.scores.segment import segment_score

    notes = _line(0.0, 5, 64) + _line(10.0, 5, 67)  # ~8s rest between the two runs
    score = Score(score_id="s", notes=notes)
    phrases = segment_score(score, gap_s=0.4)
    assert len(phrases) == 2
    assert all(p.is_monophonic() for p in phrases)
    assert all(p.notes[0].onset_s < 0.3 for p in phrases)  # each phrase re-zeroed


def test_octave_centres_into_range() -> None:
    from voders.scores.segment import segment_score

    score = Score(score_id="low", notes=_line(0.0, 6, 33))  # an octave+ below range
    phrases = segment_score(score, lo=55, hi=79)
    assert phrases
    assert all(55 <= n.pitch_midi <= 79 for p in phrases for n in p.notes)


def test_drops_short_or_sparse_phrases() -> None:
    from voders.scores.segment import segment_score

    score = Score(score_id="tiny", notes=_line(0.0, 2, 64))  # only 2 notes < min_notes
    assert segment_score(score, min_notes=4) == []


def test_hard_length_cap_splits_long_runs() -> None:
    from voders.scores.segment import segment_score

    score = Score(score_id="long", notes=_line(0.0, 60, 64))  # one long gapless run
    phrases = segment_score(score, gap_s=0.4, max_phrase_s=10.0)
    assert len(phrases) >= 2
    assert all((p.notes[-1].offset_s - p.notes[0].onset_s) <= 12.0 for p in phrases)
