"""Unit test for the stats ``score_augmentation`` coverage block (T025, FR-012)."""

from __future__ import annotations

import json
from pathlib import Path

from voders.corpus.stats import build_stats


def _rec(sample_id: str, *, axis: str | None = None, transform: str | None = None) -> dict:
    base = {
        "sample_id": sample_id,
        "score_id": sample_id,
        "score_path": "",
        "audio_path": "",
        "lane": "deterministic",
        "voice_id": "v",
        "seed": 1,
        "config_hash": "sha256:x",
        "verdict": {"status": "accepted"},
    }
    if axis is not None:
        base.update(
            base_score_id="s",
            score_aug_profile="p",
            score_aug_axis=axis,
            score_aug_transform=transform,
            score_aug_seed=99,
        )
    return base


def test_score_augmentation_block(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        _rec("s"),  # original
        _rec("s__t+12", axis="transpose", transform="t+12"),
        _rec("s__t-12", axis="transpose", transform="t-12"),
        _rec("s__hum0", axis="humanize", transform="hum0"),
        _rec("s__vol", axis="volume", transform="vol"),
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    block = build_stats(manifest)["score_augmentation"]
    assert block["enabled"] is True
    assert block["variant_share"] == 4 / 5
    assert block["by_transform"] == {"t+12": 1, "t-12": 1, "hum0": 1, "vol": 1}
    assert block["effective_multiplier"] == 5 / 1  # 5 accepted / 1 original


def test_score_augmentation_block_disabled_when_no_variants(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_rec("s")) + "\n")
    block = build_stats(manifest)["score_augmentation"]
    assert block["enabled"] is False
    assert block["by_transform"] == {}
