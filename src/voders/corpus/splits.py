"""Stratified train/validation splits by source (issue #7).

A downstream training run needs to attribute errors to specific generators (lane / voice /
augmentation profile). Splitting *within* each source group — so train and val both hold a
proportional slice of every source — lets per-source metrics be computed on held-out data without
one source dominating either side. The split is deterministic in ``seed`` (FR-013 spirit), so a
manifest reproduces the same split.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from voders.manifest.io import load_manifest
from voders.manifest.models import ProvenanceRecord, VerdictStatus
from voders.seeds import rng

_DEFAULT_STRATIFY = ("lane", "voice_id", "augmentation_profile")


def _source_key(record: ProvenanceRecord, stratify: tuple[str, ...]) -> str:
    parts = []
    for field in stratify:
        value = getattr(record, field, None)
        parts.append(str(value) if value not in (None, "") else "-")
    return "/".join(parts)


def build_splits(
    manifest_path: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    stratify: tuple[str, ...] = _DEFAULT_STRATIFY,
) -> dict[str, Any]:
    """Build a deterministic, source-stratified train/val split over accepted samples.

    Each source group is shuffled by the seeded RNG and split so val gets ``val_fraction`` of it
    (at least one sample when the group has ≥2). Returns train/val entry lists plus per-source
    counts; the entries carry ``sample_id``/``audio_path``/``score_path``/``source`` for a trainer.
    """
    root = Path(manifest_path).parent
    accepted = [
        r for r in load_manifest(manifest_path) if r.verdict.status == VerdictStatus.ACCEPTED
    ]

    groups: dict[str, list[ProvenanceRecord]] = defaultdict(list)
    for r in accepted:
        groups[_source_key(r, stratify)].append(r)

    gen = rng(seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    per_source: dict[str, dict[str, int]] = {}

    def entry(r: ProvenanceRecord, source: str) -> dict[str, str]:
        return {
            "sample_id": r.sample_id,
            "audio_path": str(root / r.audio_path),
            "score_path": str(root / r.score_path),
            "source": source,
        }

    for source in sorted(groups):
        recs = sorted(groups[source], key=lambda r: r.sample_id)
        order = gen.permutation(len(recs))
        n_val = min(len(recs) - 1, round(len(recs) * val_fraction)) if len(recs) >= 2 else 0
        val_idx = set(order[:n_val].tolist())
        for i, r in enumerate(recs):
            (val if i in val_idx else train).append(entry(r, source))
        per_source[source] = {"total": len(recs), "train": len(recs) - n_val, "val": n_val}

    return {
        "manifest": str(manifest_path),
        "val_fraction": val_fraction,
        "seed": seed,
        "stratify": list(stratify),
        "n_accepted": len(accepted),
        "n_train": len(train),
        "n_val": len(val),
        "per_source": per_source,
        "train": train,
        "val": val,
    }


def write_splits(
    manifest_path: str | Path,
    out: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Write the split to ``splits.json`` next to the manifest (or ``out``)."""
    result = build_splits(manifest_path, **kwargs)
    out_path = Path(out) if out else Path(manifest_path).parent / "splits.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
