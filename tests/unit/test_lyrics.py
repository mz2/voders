"""Lyric field on notes + optional 4th TSV column (foundation for the lyric-driven SVS path)."""

from __future__ import annotations

from pathlib import Path

from voders.scores.models import Note, Score
from voders.scores.parse import parse_tsv, serialize_score


def test_three_column_score_parses_with_no_lyric(tmp_path: Path) -> None:
    p = tmp_path / "s.tsv"
    p.write_text("0.2\t0.7\t60\n0.8\t1.3\t64\n", encoding="utf-8")
    score = parse_tsv(p).score
    assert [n.lyric for n in score.notes] == [None, None]


def test_fourth_column_is_parsed_as_lyric(tmp_path: Path) -> None:
    p = tmp_path / "s.tsv"
    p.write_text("0.2\t0.7\t60\tla\n0.8\t1.3\t64\tdi\n", encoding="utf-8")
    score = parse_tsv(p).score
    assert [n.lyric for n in score.notes] == ["la", "di"]


def test_serialize_roundtrip_preserves_lyrics() -> None:
    score = Score(
        score_id="t",
        notes=[
            Note(onset_s=0.2, offset_s=0.7, pitch_midi=60, lyric="la"),
            Note(onset_s=0.8, offset_s=1.3, pitch_midi=64, lyric="di"),
        ],
    )
    data = serialize_score(score)
    assert data.decode().count("\t") == 6  # 4 columns x 2 rows minus newlines -> 3 tabs/row
    # lyric-free scores stay 3-column (backward compatible)
    plain = Score(score_id="t", notes=[Note(onset_s=0.0, offset_s=0.5, pitch_midi=60)])
    assert serialize_score(plain).decode().strip().count("\t") == 2
