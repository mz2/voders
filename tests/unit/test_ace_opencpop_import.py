from __future__ import annotations

import pytest
from evals.prepare_ace_opencpop import score_filename, score_from_ace_row, write_scores

from voders.scores.parse import parse_tsv


def _row(**updates):
    row = {
        "segment_id": "acesinger_1#example",
        "note_midi": [60.0, 60.0, 62.0, 0.0, 64.0],
        "note_lyrics": ["n_i", "—", "h_ao", "SP", "m_a"],
        "note_start_times": [0.0, 0.2, 0.5, 0.8, 0.9],
        "note_end_times": [0.2, 0.5, 0.8, 0.9, 1.2],
    }
    row.update(updates)
    return row


def test_score_conversion_drops_rests_and_merges_same_pitch_slur() -> None:
    score = score_from_ace_row(_row())
    assert [(n.onset_s, n.offset_s, n.pitch_midi) for n in score.notes] == [
        (0.0, 0.5, 60),
        (0.5, 0.8, 62),
        (0.9, 1.2, 64),
    ]


def test_pitch_changing_slur_remains_a_new_transcription_note() -> None:
    score = score_from_ace_row(
        _row(
            note_midi=[60, 62],
            note_lyrics=["n_i", "—"],
            note_start_times=[0, 0.2],
            note_end_times=[0.2, 0.5],
        )
    )
    assert [n.pitch_midi for n in score.notes] == [60, 62]


def test_mismatched_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        score_from_ace_row(_row(note_end_times=[0.2]))


def test_filename_is_safe_stable_and_collision_resistant() -> None:
    name = score_filename("acesinger_1#2001000004")
    assert name == score_filename("acesinger_1#2001000004")
    assert "#" not in name and name.endswith(".tsv")
    assert name != score_filename("acesinger_1/2001000004")


def test_write_scores_resumes_and_emits_trainer_compatible_tsv(tmp_path) -> None:
    out = tmp_path / "scores"
    out.mkdir()
    first = write_scores([_row()], out)
    second = write_scores([_row()], out)
    assert first == {
        "selected": 1,
        "written": 1,
        "existing": 0,
        "invalid": 0,
        "empty": 0,
        "usable": 1,
    }
    assert second["existing"] == 1 and second["written"] == 0
    parsed = parse_tsv(next(out.glob("*.tsv"))).score
    assert [n.pitch_midi for n in parsed.notes] == [60, 62, 64]
