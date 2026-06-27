"""Source-stratified train/val splits (issue #7)."""

from __future__ import annotations

from pathlib import Path

from voders.corpus.splits import build_splits
from voders.manifest.io import ManifestWriter
from voders.manifest.models import ProvenanceRecord, ValidationVerdict, VerdictStatus


def _manifest(tmp_path: Path) -> Path:
    """A manifest with two sources (5 deterministic + 5 voiceconv accepted, 1 rejected)."""
    w = ManifestWriter(tmp_path / "manifest.jsonl")
    for lane, voice, n in [("deterministic", "ah", 5), ("voice_conversion", "oo", 5)]:
        for i in range(n):
            sid = f"s{i}_{voice}"
            w.append(
                ProvenanceRecord(
                    sample_id=sid,
                    score_id=f"s{i}",
                    score_path=f"corpus/shard=000/{sid}.tsv",
                    audio_path=f"corpus/shard=000/{sid}.wav",
                    lane=lane,
                    voice_id=voice,
                    seed=1,
                    config_hash="x",
                    verdict=ValidationVerdict(status=VerdictStatus.ACCEPTED),
                )
            )
    # one rejected sample must be excluded from splits
    w.append(
        ProvenanceRecord(
            sample_id="bad",
            score_id="s9",
            score_path="rejected/shard=000/bad.tsv",
            audio_path="rejected/shard=000/bad.wav",
            lane="svs",
            voice_id="x",
            seed=1,
            config_hash="x",
            verdict=ValidationVerdict(status=VerdictStatus.REJECTED),
        )
    )
    return tmp_path / "manifest.jsonl"


def test_split_is_stratified_complete_and_disjoint(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    res = build_splits(manifest, val_fraction=0.4, seed=0)
    assert res["n_accepted"] == 10  # the rejected sample is excluded
    train_ids = {e["sample_id"] for e in res["train"]}
    val_ids = {e["sample_id"] for e in res["val"]}
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) + len(val_ids) == 10
    # Each source contributes to both sides (stratified): 40% of 5 -> 2 val, 3 train per source.
    for src, c in res["per_source"].items():
        assert c["val"] == 2, src
        assert c["train"] == 3, src


def test_split_is_deterministic_in_seed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)  # build once; only the seed varies below
    a = build_splits(manifest, seed=7)
    b = build_splits(manifest, seed=7)
    assert [e["sample_id"] for e in a["val"]] == [e["sample_id"] for e in b["val"]]
    # Some seed yields a different partition (the shuffle actually depends on the seed).
    base = [e["sample_id"] for e in a["val"]]
    assert any(
        [e["sample_id"] for e in build_splits(manifest, seed=s)["val"]] != base for s in range(20)
    )
