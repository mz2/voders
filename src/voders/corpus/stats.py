"""Aggregate corpus statistics (FR-012, SC-004, SC-006).

Computed by scanning the manifest (FR-008). Reports totals, unique scores/voices, timbre
identities (voice × augmentation profile, SC-004), pitch/duration distributions, augmentation
coverage (SC-006), and accept/reject counts (SC-007).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voders.manifest.io import load_manifest
from voders.manifest.models import ProvenanceRecord, VerdictStatus
from voders.scores.parse import parse_tsv


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def build_stats(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    records = load_manifest(manifest_path)
    accepted = [r for r in records if r.verdict.status == VerdictStatus.ACCEPTED]

    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.verdict.status.value] = by_status.get(r.verdict.status.value, 0) + 1

    timbre_identities = {(r.voice_id, r.augmentation_profile or "") for r in accepted}
    augmented = [r for r in accepted if r.augmentation_profile]

    pitches: list[float] = []
    durations: list[float] = []
    for r in accepted:
        for p in _read_pitches_durations(root, r):
            pitches.append(p[0])
            durations.append(p[1])

    return {
        "total_samples": len(records),
        "accepted": len(accepted),
        "by_status": by_status,
        "unique_scores": len({r.score_id for r in accepted}),
        "unique_voices": len({r.voice_id for r in accepted}),
        "timbre_identities": len(timbre_identities),
        "augmentation_coverage": (len(augmented) / len(accepted)) if accepted else 0.0,
        "pitch_distribution": _distribution(pitches),
        "duration_ms_distribution": _distribution([d * 1000.0 for d in durations]),
        "lyrics": _lyric_stats(records, _gather_syllables(root, accepted)),
    }


def _gather_syllables(root: Path, records: list[ProvenanceRecord]) -> list[str | None]:
    """Collect per-note syllables carried in each accepted record's label ``.tsv``."""
    syllables: list[str | None] = []
    for r in records:
        if not r.score_path:
            continue
        p = root / r.score_path
        if not p.exists():
            continue
        try:
            parsed = parse_tsv(p)
        except Exception:
            continue
        syllables.extend(n.lyric for n in parsed.score.notes)
    return syllables


def _lyric_stats(records: list[ProvenanceRecord], syllables: list[str | None]) -> dict[str, Any]:
    """Lyric-source breakdown, supplied multi-syllable count, and phonetic coverage (FR-014).

    Phonetic coverage is computed over syllables carried in the label ``.tsv`` (the ``supplied``
    source); ``automatic``/``generated`` syllables ride in provenance/cache, so coverage there is
    surfaced by the source-level SC-003 check rather than this manifest scan.
    """
    from voders.lyrics.coverage import phonetic_coverage

    by_source: dict[str, int] = {}
    by_language: dict[str, int] = {}
    for r in records:
        by_source[r.lyric_source] = by_source.get(r.lyric_source, 0) + 1
        by_language[r.lyric_language] = by_language.get(r.lyric_language, 0) + 1
    return {
        "by_source": by_source,
        "by_language": by_language,
        "multisyllable_supplied_total": sum(r.lyric_multisyllable_supplied for r in records),
        "coverage": phonetic_coverage(syllables),
    }


def _read_pitches_durations(root: Path, record: ProvenanceRecord) -> list[tuple[float, float]]:
    if not record.score_path:
        return []
    p = root / record.score_path
    if not p.exists():
        return []
    try:
        parsed = parse_tsv(p)
    except Exception:
        return []
    return [(float(n.pitch_midi), n.duration_s) for n in parsed.score.notes]


def write_stats(manifest_path: str | Path, out: str | Path | None = None) -> Path:
    stats = build_stats(manifest_path)
    out_path = Path(out) if out else Path(manifest_path).parent / "stats.json"
    out_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
