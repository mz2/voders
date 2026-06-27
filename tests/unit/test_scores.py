"""Unit tests for score parsing and models (T048, FR-001, FR-002)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voders.scores.models import Note, Score
from voders.scores.parse import PolyphonyError, parse_tsv, serialize_score

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"
EDGE_DIR = REPO_ROOT / "evals" / "fixtures" / "edge"


def test_parse_tsv_clean_fixture() -> None:
    """A clean monophonic fixture parses into an ordered, non-empty Score."""
    parsed = parse_tsv(SCORES_DIR / "score_000.tsv")
    assert parsed.score.score_id == "score_000"
    assert not parsed.score.is_empty
    assert parsed.score.is_monophonic()
    onsets = [n.onset_s for n in parsed.score.notes]
    assert onsets == sorted(onsets)
    # raw_bytes are retained verbatim for byte-identical re-emission (FR-002).
    assert parsed.raw_bytes == (SCORES_DIR / "score_000.tsv").read_bytes()


def test_parse_tsv_polyphony_raises() -> None:
    """Overlapping notes are rejected: singing is monophonic (FR-001)."""
    with pytest.raises(PolyphonyError):
        parse_tsv(EDGE_DIR / "polyphony.tsv")


def test_parse_tsv_legato_parses() -> None:
    """A legato score (note offset touching the next onset) is valid monophony."""
    parsed = parse_tsv(EDGE_DIR / "legato.tsv")
    assert parsed.score.is_monophonic()
    assert len(parsed.score.notes) == 3


def test_empty_score_is_empty() -> None:
    """A Score with no notes reports is_empty True."""
    score = Score(score_id="empty", source="test", notes=[])
    assert score.is_empty
    assert score.duration_s == 0.0


def test_note_rejects_non_positive_duration() -> None:
    """Note offset must be strictly greater than onset (pydantic ValidationError)."""
    with pytest.raises(ValidationError):
        Note(onset_s=0.5, offset_s=0.5, pitch_midi=60)
    with pytest.raises(ValidationError):
        Note(onset_s=0.5, offset_s=0.4, pitch_midi=60)


def test_serialize_score_roundtrips_note_count(tmp_path: Path) -> None:
    """serialize_score emits one row per note and re-parses to the same note count."""
    parsed = parse_tsv(SCORES_DIR / "score_000.tsv")
    data = serialize_score(parsed.score)
    out = tmp_path / "roundtrip.tsv"
    out.write_bytes(data)
    reparsed = parse_tsv(out)
    assert len(reparsed.score.notes) == len(parsed.score.notes)
    for a, b in zip(reparsed.score.notes, parsed.score.notes, strict=True):
        assert a.pitch_midi == b.pitch_midi
        assert a.onset_s == pytest.approx(b.onset_s)
        assert a.offset_s == pytest.approx(b.offset_s)
