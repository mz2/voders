"""Contract: ``expand()`` guarantees G1–G5 (T008, contracts/score-augmentation.md)."""

from __future__ import annotations

from pathlib import Path

from voders.config.models import HumanizeKnob, ScoreAugmentationProfile, TransposeKnob, VolumeKnob
from voders.scoreaug.expand import expand
from voders.scores.models import Note, Score
from voders.scores.parse import ParsedScore, parse_tsv, serialize_score
from voders.seeds import derive_seed


def _base() -> ParsedScore:
    score = Score(
        score_id="t1",
        notes=[
            Note(onset_s=0.20, offset_s=0.65, pitch_midi=60, lyric="la"),
            Note(onset_s=0.73, offset_s=1.18, pitch_midi=62, lyric="le"),
            Note(onset_s=1.26, offset_s=1.71, pitch_midi=64, lyric="li"),
        ],
    )
    return ParsedScore(score=score, source_path=Path("t1.tsv"), raw_bytes=b"")


def _profile() -> ScoreAugmentationProfile:
    return ScoreAugmentationProfile(
        profile_id="p",
        transpose=TransposeKnob(offsets=[12], policy="drop"),
        humanize_time=HumanizeKnob(
            onset_sigma_s=0.02, duration_sigma_s=0.02, max_dev_s=0.05, draws=2
        ),
        volume=VolumeKnob(gain_db_range=(-6.0, 6.0)),
    )


def _seed() -> int:
    return derive_seed(42, "t1", "score_aug", "p")


def test_g1_reexpansion_is_byte_identical():
    base, profile, seed = _base(), _profile(), _seed()
    a = expand(base, profile, seed)
    b = expand(base, profile, seed)
    assert [v.score.model_dump() for v in a] == [v.score.model_dump() for v in b]
    assert [v.transform for v in a] == [v.transform for v in b]


def test_g2_every_pitch_in_range():
    out = expand(_base(), _profile(), _seed())
    assert all(0 <= n.pitch_midi <= 127 for v in out for n in v.score.notes)


def test_g3_every_variant_is_valid_monophonic(tmp_path: Path):
    out = expand(_base(), _profile(), _seed())
    assert out, "profile with all axes must emit variants"
    for v in out:
        assert v.score.is_monophonic()
        # The parser accepts the serialized variant without modification.
        p = tmp_path / f"{v.score.score_id}.tsv"
        p.write_bytes(serialize_score(v.score))
        parse_tsv(p)


def test_g4_note_count_and_lyric_preserved():
    out = expand(_base(), _profile(), _seed())
    for v in out:
        assert len(v.score.notes) == 3
        assert [n.lyric for n in v.score.notes] == ["la", "le", "li"]


def test_g5_variant_ids_are_self_describing():
    out = expand(_base(), _profile(), _seed())
    ids = {v.score.score_id for v in out}
    assert "t1__t+12" in ids
    assert "t1__hum0" in ids and "t1__hum1" in ids
    assert "t1__vol" in ids


def test_empty_profile_emits_no_variants():
    assert expand(_base(), ScoreAugmentationProfile(profile_id="noop"), _seed()) == []
