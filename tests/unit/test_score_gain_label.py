"""The per-note ``gain`` is excluded from the label ``.tsv`` (T007, FR-009).

``serialize_score`` must emit byte-identical output whether or not a note carries a gain, and
``parse_tsv`` never reads a gain (a parsed score always has ``gain=None``). The gain rides in-memory
to the renderer only; the transcription label schema stays onset/offset/pitch[/lyric].
"""

from __future__ import annotations

from pathlib import Path

from voders.scores.models import Note, Score
from voders.scores.parse import parse_tsv, serialize_score


def _score(*, with_gain: bool) -> Score:
    return Score(
        score_id="s",
        notes=[
            Note(onset_s=0.0, offset_s=0.5, pitch_midi=60, gain=0.5 if with_gain else None),
            Note(onset_s=0.5, offset_s=1.0, pitch_midi=62, gain=1.7 if with_gain else None),
        ],
    )


def test_serialize_score_is_byte_identical_with_and_without_gain():
    assert serialize_score(_score(with_gain=False)) == serialize_score(_score(with_gain=True))


def test_parse_tsv_never_reads_a_gain(tmp_path: Path):
    p = tmp_path / "s.tsv"
    p.write_bytes(serialize_score(_score(with_gain=True)))
    parsed = parse_tsv(p)
    assert all(n.gain is None for n in parsed.score.notes)
