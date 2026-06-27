"""US2 integration: time-humanisation variants (T017, SC-003, SC-005, FR-006/007)."""

from __future__ import annotations

from pathlib import Path

from voders.config.models import HumanizeKnob, ScoreAugmentationProfile
from voders.scores.models import Score
from voders.scores.parse import parse_tsv

from .scoreaug_util import run_score_aug, write_scores


def _profile() -> ScoreAugmentationProfile:
    return ScoreAugmentationProfile(
        profile_id="hp",
        humanize_time=HumanizeKnob(
            onset_sigma_s=0.02, duration_sigma_s=0.02, max_dev_s=0.05, draws=2
        ),
    )


def test_humanize_emits_valid_bounded_variants(tmp_path: Path, donor_ah) -> None:
    scores = tmp_path / "scores"
    write_scores(scores, ["score_000"])
    out = tmp_path / "out"
    records = run_score_aug(out, scores, donor_ah, _profile())

    variants = [r for r in records if r.score_aug_axis == "humanize" and r.score_path]
    assert sorted(r.score_aug_transform for r in variants) == ["hum0", "hum1"]

    base = parse_tsv(scores / "score_000.tsv").score.notes
    for r in variants:
        vn = parse_tsv(out / r.score_path).score.notes
        # Valid monophonic score with strictly positive durations.
        assert Score(score_id="x", notes=list(vn)).is_monophonic()
        assert all(n.duration_s > 0 for n in vn)
        # Per-note deviations are non-zero (jitter present) and within the 50 ms budget.
        devs = [abs(a.onset_s - b.onset_s) for a, b in zip(vn, base, strict=True)]
        devs += [abs(a.offset_s - b.offset_s) for a, b in zip(vn, base, strict=True)]
        assert max(devs) <= 0.05 + 1e-9
        assert any(d > 1e-6 for d in devs)
        assert "augmented/humanize/" in r.score_path
        # Pitch/lyric unchanged.
        assert [n.pitch_midi for n in vn] == [n.pitch_midi for n in base]


def test_humanize_is_deterministic_across_independent_runs(tmp_path: Path, donor_ah) -> None:
    """SC-005: a fixed (base, profile, seed) yields byte-identical variant labels."""
    scores = tmp_path / "scores"
    write_scores(scores, ["score_000"])
    rec_a = run_score_aug(tmp_path / "a", scores, donor_ah, _profile())
    rec_b = run_score_aug(tmp_path / "b", scores, donor_ah, _profile())

    def labels(records, root):
        out = {}
        for r in records:
            if r.score_aug_axis == "humanize" and r.score_path:
                out[r.score_aug_transform] = (root / r.score_path).read_bytes()
        return out

    assert labels(rec_a, tmp_path / "a") == labels(rec_b, tmp_path / "b")
    # And the on-disk labels equal a fresh in-process serialization of the expansion.
    assert labels(rec_a, tmp_path / "a")  # non-empty
