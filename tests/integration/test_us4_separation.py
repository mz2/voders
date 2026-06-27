"""US4 integration: physical separation, lineage, and off-control safety (T024, SC-001/006/009)."""

from __future__ import annotations

from pathlib import Path

from voders.config.models import (
    HumanizeKnob,
    LaneToggle,
    RunConfig,
    ScoreAugmentationProfile,
    TransposeKnob,
    ValidatorConfig,
    VolumeKnob,
)
from voders.corpus.orchestrator import Orchestrator
from voders.manifest.io import load_manifest
from voders.render.registry import build_lanes
from voders.scoreaug.expand import expand_full
from voders.scores.parse import parse_tsv, serialize_score
from voders.seeds import derive_seed

from .scoreaug_util import run_score_aug, write_scores


def _all_axes() -> ScoreAugmentationProfile:
    return ScoreAugmentationProfile(
        profile_id="all",
        transpose=TransposeKnob(offsets=[-12, 12]),
        humanize_time=HumanizeKnob(max_dev_s=0.05, draws=2),
        volume=VolumeKnob(gain_db_range=(-6.0, 6.0)),
    )


def test_originals_and_variants_are_separated_with_full_lineage(tmp_path: Path, donor_ah) -> None:
    scores = tmp_path / "scores"
    write_scores(scores, ["score_000"])
    out = tmp_path / "out"
    records = run_score_aug(out, scores, donor_ah, _all_axes())

    originals = [r for r in records if r.base_score_id is None and r.lane != "score_augmentation"]
    variants = [r for r in records if r.base_score_id is not None and r.score_path]

    # Originals only under corpus/shard (never augmented/); variants only under augmented/<axis>/.
    for r in originals:
        assert "augmented/" not in r.score_path
    for r in variants:
        assert "/augmented/" in r.score_path
        # Self-describing, fully traceable lineage.
        assert r.base_score_id == "score_000"
        assert r.score_aug_profile == "all"
        assert r.score_aug_seed is not None
        assert r.score_aug_transform and r.score_aug_axis

    # Zero path collisions across the whole run.
    paths = [r.score_path for r in records if r.score_path] + [
        r.audio_path for r in records if r.audio_path
    ]
    assert len(paths) == len(set(paths))

    # Regenerating from the manifest reproduces every variant label byte-identically (in-process).
    base = parse_tsv(scores / "score_000.tsv")
    seed = derive_seed(42, "score_000", "score_aug", "all")
    fresh = {
        v.transform: serialize_score(v.score) for v in expand_full(base, _all_axes(), seed).variants
    }
    for r in variants:
        assert (out / r.score_path).read_bytes() == fresh[r.score_aug_transform]


def test_feature_off_creates_no_augmented_subtree_and_identical_labels(
    tmp_path: Path, donor_ah
) -> None:
    """SC-001: with no score_augmentation, originals are untouched and no augmented/ appears."""
    scores = tmp_path / "scores"
    write_scores(scores, ["score_000"])

    # Feature on (originals subset) vs feature off — original labels must be byte-identical.
    on = run_score_aug(tmp_path / "on", scores, donor_ah, _all_axes())

    off_root = tmp_path / "off"
    config = RunConfig(
        run_id="off",
        master_seed=42,
        scores=str(scores / "*.tsv"),
        output_root=str(off_root),
        voices=[donor_ah],
        lanes={"deterministic": LaneToggle(enabled=True)},
        validator=ValidatorConfig(min_note_ms=300.0),
    )
    Orchestrator(config, build_lanes(config)).run()
    off = load_manifest(off_root / "manifest.jsonl")

    assert not (off_root / "corpus" / "augmented").exists()
    assert all(r.base_score_id is None and r.score_aug_axis is None for r in off)

    on_orig = next(r for r in on if r.base_score_id is None and r.score_id == "score_000")
    off_orig = next(r for r in off if r.score_id == "score_000")
    assert (tmp_path / "on" / on_orig.score_path).read_bytes() == (
        off_root / off_orig.score_path
    ).read_bytes()
