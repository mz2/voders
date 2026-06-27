"""Shared helpers for the feature-003 score-augmentation integration tests."""

from __future__ import annotations

from pathlib import Path

from voders.config.models import (
    LaneToggle,
    RunConfig,
    ScoreAugmentationProfile,
    ValidatorConfig,
)
from voders.corpus.orchestrator import Orchestrator
from voders.manifest.io import load_manifest
from voders.manifest.models import ProvenanceRecord
from voders.render.registry import build_lanes
from voders.voices.models import Voice

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"


def write_scores(dest: Path, score_ids: list[str], extra: dict[str, str] | None = None) -> None:
    """Copy named fixture scores into ``dest`` plus any ``extra`` {stem: tsv-text} scores."""
    dest.mkdir(parents=True, exist_ok=True)
    for sid in score_ids:
        (dest / f"{sid}.tsv").write_bytes((SCORES_DIR / f"{sid}.tsv").read_bytes())
    for stem, text in (extra or {}).items():
        (dest / f"{stem}.tsv").write_text(text)


def run_score_aug(
    out_root: Path,
    scores_dir: Path,
    voice: Voice,
    profile: ScoreAugmentationProfile,
    *,
    min_note_ms: float = 300.0,
) -> list[ProvenanceRecord]:
    """Run the deterministic lane with one score-augmentation profile; return manifest records."""
    config = RunConfig(
        run_id="itest",
        master_seed=42,
        scores=str(scores_dir / "*.tsv"),
        output_root=str(out_root),
        voices=[voice],
        lanes={"deterministic": LaneToggle(enabled=True)},
        validator=ValidatorConfig(min_note_ms=min_note_ms),
        score_augmentation=[profile],
    )
    Orchestrator(config, build_lanes(config)).run()
    return load_manifest(out_root / "manifest.jsonl")
