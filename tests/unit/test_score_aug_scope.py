"""Scope guard: v1 composition is per-axis only — no cross-axis products (T030).

Locks the documented v1 scope: ``expand()`` emits one variant per transpose offset, per humanise
draw, and one volume variant — never a transpose×humanize (or any cross-axis) combination.
"""

from __future__ import annotations

from pathlib import Path

from voders.config.models import (
    HumanizeKnob,
    ScoreAugmentationProfile,
    TransposeKnob,
    VolumeKnob,
)
from voders.scoreaug.expand import expand
from voders.scores.models import Note, Score
from voders.scores.parse import ParsedScore
from voders.seeds import derive_seed


def _base() -> ParsedScore:
    score = Score(
        score_id="b",
        notes=[
            Note(onset_s=0.2, offset_s=0.7, pitch_midi=60),
            Note(onset_s=0.8, offset_s=1.3, pitch_midi=62),
        ],
    )
    return ParsedScore(score=score, source_path=Path("b.tsv"), raw_bytes=b"")


def test_expand_emits_per_axis_variants_only():
    profile = ScoreAugmentationProfile(
        profile_id="p",
        transpose=TransposeKnob(offsets=[-12, 12]),
        humanize_time=HumanizeKnob(max_dev_s=0.05, draws=3),
        volume=VolumeKnob(gain_db_range=(-6.0, 6.0)),
    )
    out = expand(_base(), profile, derive_seed(1, "b", "score_aug", "p"))
    transforms = sorted(v.transform for v in out)

    # Exactly: 2 transpose + 3 humanise + 1 volume = 6 variants, no cross-products.
    assert transforms == ["hum0", "hum1", "hum2", "t+12", "t-12", "vol"]
    assert len(out) == 2 + 3 + 1
    # No transform descriptor combines two axes (e.g. "t+12_hum0").
    assert all(v.transform.count("_") == 0 for v in out)
