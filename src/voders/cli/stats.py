"""``voders stats`` — emit aggregate corpus statistics (FR-012)."""

from __future__ import annotations

import json

from voders.corpus.stats import build_stats, write_stats


def stats_command(manifest_path: str, *, out: str | None = None) -> int:
    stats = build_stats(manifest_path)
    print(json.dumps(stats, indent=2, sort_keys=True))
    path = write_stats(manifest_path, out)
    print(f"wrote {path}")
    return 0
