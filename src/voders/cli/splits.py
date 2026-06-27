"""``voders splits`` — write source-stratified train/val splits (issue #7)."""

from __future__ import annotations

from voders.corpus.splits import build_splits, write_splits


def splits_command(
    manifest_path: str,
    *,
    out: str | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
    stratify: tuple[str, ...] = ("lane", "voice_id", "augmentation_profile"),
) -> int:
    result = build_splits(manifest_path, val_fraction=val_fraction, seed=seed, stratify=stratify)
    path = write_splits(manifest_path, out, val_fraction=val_fraction, seed=seed, stratify=stratify)
    print(
        f"splits: {result['n_train']} train / {result['n_val']} val "
        f"from {result['n_accepted']} accepted, stratified by {','.join(result['stratify'])}"
    )
    print(f"{'source':<48}{'total':>6}{'train':>6}{'val':>5}")
    print("-" * 65)
    for source, c in sorted(result["per_source"].items()):
        print(f"{source:<48}{c['total']:>6}{c['train']:>6}{c['val']:>5}")
    print(f"wrote {path}")
    return 0
