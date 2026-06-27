"""Unit tests for external vocal-melody MIDI extraction (issue #9).

Built on synthesised MIDI so no external corpus is needed: a named melody track is preferred, a
polyphonic-only file is reduced to a monophonic skyline, and out-of-range lines are octave-centred
into the singable tessitura — with the yield/bias accounting populated.
"""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from voders.scores.midi_import import extract_melody


def _write_midi(path: Path, tracks: list[tuple[str, list[tuple[float, float, int]]]]) -> None:
    pm = pretty_midi.PrettyMIDI()
    for name, notes in tracks:
        inst = pretty_midi.Instrument(program=0, name=name)
        for start, end, pitch in notes:
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
        pm.instruments.append(inst)
    pm.write(str(path))


def test_prefers_named_melody_track(tmp_path: Path) -> None:
    mid = tmp_path / "song.mid"
    _write_midi(
        mid,
        [
            ("Melody", [(0.0, 0.5, 64), (0.5, 1.0, 67), (1.0, 1.5, 65)]),
            ("Piano", [(0.0, 1.5, 48), (0.0, 1.5, 52)]),  # held chord — must be ignored
        ],
    )
    score, stats = extract_melody(mid, lo=55, hi=79)
    assert score is not None
    assert stats.method == "melody_track"
    assert score.is_monophonic()
    assert [n.pitch_midi for n in score.notes] == [64, 67, 65]


def test_skyline_when_no_melody_track(tmp_path: Path) -> None:
    mid = tmp_path / "poly.mid"
    # Two overlapping voices; the skyline is the higher line.
    _write_midi(
        mid,
        [
            ("Piano", [(0.0, 1.0, 60), (0.0, 1.0, 67), (1.0, 2.0, 62), (1.0, 2.0, 69)]),
        ],
    )
    score, stats = extract_melody(mid, lo=55, hi=79)
    assert score is not None
    assert stats.method == "skyline"
    assert score.is_monophonic()
    assert max(n.pitch_midi for n in score.notes) >= 67  # took the top line


def test_out_of_range_is_octave_centred(tmp_path: Path) -> None:
    mid = tmp_path / "bass.mid"
    _write_midi(mid, [("Melody", [(0.0, 0.5, 31), (0.5, 1.0, 35), (1.0, 1.5, 33)])])
    score, stats = extract_melody(mid, lo=55, hi=79)
    assert score is not None
    assert all(55 <= n.pitch_midi <= 79 for n in score.notes)
    assert stats.transposed_semitones != 0
    assert stats.pitch_in[1] < 55 < stats.pitch_out[0] + 24  # was below range, now inside


def test_empty_midi_rejected(tmp_path: Path) -> None:
    mid = tmp_path / "empty.mid"
    _write_midi(mid, [("Drums", [])])
    score, stats = extract_melody(mid, lo=55, hi=79)
    assert score is None
    assert not stats.accepted
    assert stats.reject_reason


def test_unreadable_file_rejected_not_raised(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"not a midi file")
    score, stats = extract_melody(bad)
    assert score is None
    assert "unreadable" in stats.reject_reason
