"""End-to-end SVS articulation through the out-of-process backend (US1, FR-006/007).

Runs a real orchestrator pass on the SVS lane with automatic lyrics: core-side espeak G2P feeds the
backend, which articulates the phonemes at score-derived pitch. Skipped unless the SVS backend
project is synced (it runs in its own uv project) and espeak-ng is available.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from voders.render.backend_bridge import backend_available

try:
    from voders.lyrics.g2p import text_to_phonemes

    _ESPEAK = text_to_phonemes("la")[:1] == ["l"]
except Exception:
    _ESPEAK = False

pytestmark = pytest.mark.skipif(
    not (backend_available("svs") and _ESPEAK),
    reason="needs the synced svs backend + espeak-ng",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"


def test_svs_lane_articulates_lyrics_end_to_end(tmp_path: Path, donor_ah) -> None:
    from voders.config.models import LaneToggle, LyricsConfig, RunConfig
    from voders.corpus.orchestrator import Orchestrator
    from voders.lyrics.models import LyricSource
    from voders.manifest.io import load_manifest
    from voders.render.registry import build_lanes
    from voders.voices.models import Voice, VoiceKind

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SCORES_DIR / "score_000.tsv", scores_dir / "score_000.tsv")
    voice = Voice(
        voice_id="svs_ah",
        kind=VoiceKind.SVS_VOICEBANK,
        license="CC0",
        consent_verified=True,
        model_ref=donor_ah.model_ref,
    )
    config = RunConfig(
        run_id="svs-e2e",
        master_seed=20260627,
        scores=str(scores_dir / "*.tsv"),
        output_root=str(tmp_path / "out"),
        voices=[voice],
        lanes={"svs": LaneToggle(enabled=True, backend="nnsvs", mode="force_score_f0")},
        lyrics=LyricsConfig(source=LyricSource.AUTOMATIC, inventory="scat"),
    )
    Orchestrator(config, build_lanes(config)).run()
    rec = load_manifest(Path(config.output_root) / "manifest.jsonl")[0]
    assert rec.lyric_source == "automatic"
    assert rec.lyric_articulated is True  # the SVS backend sang the phonemes
    assert rec.lyric_hash is not None
