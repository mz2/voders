"""US1 integration: octave/semitone transposition variants (T013, SC-002, FR-004/005)."""

from __future__ import annotations

from pathlib import Path

from voders.config.models import ScoreAugmentationProfile, TransposeKnob
from voders.manifest.models import VerdictStatus
from voders.scores.parse import parse_tsv

from .scoreaug_util import run_score_aug, write_scores


def test_transpose_emits_shifted_variants_and_drops_out_of_range(tmp_path: Path, donor_ah) -> None:
    scores = tmp_path / "scores"
    # A normal score plus an extreme one whose +12 shift (pitch 120 -> 132) must be dropped.
    write_scores(
        scores,
        ["score_000"],
        extra={"high": "0.200\t0.650\t120\n0.730\t1.180\t118\n1.260\t1.710\t119\n"},
    )
    out = tmp_path / "out"
    records = run_score_aug(
        out,
        scores,
        donor_ah,
        ScoreAugmentationProfile(
            profile_id="tp", transpose=TransposeKnob(offsets=[-12, 12], policy="drop")
        ),
    )

    variants = [r for r in records if r.score_aug_axis == "transpose" and r.score_path]
    # score_000 yields both offsets; high yields only -12 (the +12 is dropped).
    transforms = sorted(r.score_aug_transform for r in variants if r.base_score_id == "score_000")
    assert transforms == ["t+12", "t-12"]

    # Each emitted variant: pitch shifted exactly, onsets/offsets byte-identical to the base.
    base_notes = parse_tsv(scores / "score_000.tsv").score.notes
    for r in variants:
        if r.base_score_id != "score_000":
            continue
        offset = int(r.score_aug_transform[1:])
        vn = parse_tsv(out / r.score_path).score.notes
        assert [n.pitch_midi for n in vn] == [n.pitch_midi + offset for n in base_notes]
        assert [(n.onset_s, n.offset_s) for n in vn] == [
            (n.onset_s, n.offset_s) for n in base_notes
        ]
        assert all(0 <= n.pitch_midi <= 127 for n in vn)
        assert "augmented/transpose/" in r.score_path

    # The out-of-range +12 on the extreme score is dropped, recorded in provenance with a reason.
    drops = [
        r
        for r in records
        if r.lane == "score_augmentation"
        and r.base_score_id == "high"
        and r.score_aug_transform == "t+12"
    ]
    assert len(drops) == 1
    assert drops[0].verdict.status == VerdictStatus.REJECTED
    assert "window" in (drops[0].verdict.reason or "")
    # No high__t+12 variant was rendered.
    assert not any(
        r.base_score_id == "high" and r.score_aug_transform == "t+12" and r.score_path
        for r in records
    )
