"""US4: themed generated lyrics — segmentation, cache replay, and the license gate.

The generated source segments model text into one syllable per note (FR-019/SC-010), pins it so a
replay reuses the pinned text with zero model re-invocations (FR-012/SC-007), and refuses a model
whose license is unverified (FR-010/SC-006).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from voders.config.models import LaneToggle, LyricsConfig, RunConfig
from voders.corpus.orchestrator import Orchestrator
from voders.lyrics.models import LyricModel, LyricSource
from voders.lyrics.sources import GeneratedSource, LyricLicenseRefused
from voders.manifest.io import load_manifest
from voders.manifest.models import VerdictStatus
from voders.render.registry import build_lanes
from voders.scores.models import Note, Score

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"
MASTER_SEED = 20260627


def _score(n: int) -> Score:
    notes = [Note(onset_s=float(i), offset_s=float(i) + 0.5, pitch_midi=60) for i in range(n)]
    return Score(score_id="s1", notes=notes)


def _usable_model() -> LyricModel:
    return LyricModel(model_id="stub-lm", license="apache-2.0", license_ok=True, model_ref="ref")


def test_generated_segments_one_syllable_per_note_and_caches(tmp_path: Path):
    calls = {"n": 0}

    def stub_generator(theme: str, score: Score, seed: int) -> str:
        calls["n"] += 1
        return "winter longing snow falls"

    src = GeneratedSource(
        theme="winter",
        model=_usable_model(),
        output_root=str(tmp_path),
        generator=stub_generator,
    )
    score = _score(3)

    plan1 = src.resolve(score, master_seed=MASTER_SEED, voice_id="v1")
    # One syllable per note (structural SC-010): length == notes, no multisyllable flags.
    assert len(plan1.syllables) == len(score.notes)
    assert plan1.multisyllable_notes == []
    assert all(s is None or " " not in s for s in plan1.syllables)
    assert plan1.source == LyricSource.GENERATED
    assert calls["n"] == 1

    # SC-007: a second resolve hits the pinned artifact — the model is not re-invoked.
    plan2 = src.resolve(score, master_seed=MASTER_SEED, voice_id="v1")
    assert plan2.syllables == plan1.syllables
    assert calls["n"] == 1


def test_generated_refuses_unverified_license():
    bad = LyricModel(model_id="x", license="proprietary", license_ok=False, model_ref="r")
    src = GeneratedSource(theme="t", model=bad, generator=lambda *_: "la la")
    with pytest.raises(LyricLicenseRefused):
        src.resolve(_score(2), master_seed=1, voice_id="v1")


def test_orchestrator_records_license_refused_for_bad_model(tmp_path: Path, donor_ah):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SCORES_DIR / "score_000.tsv", scores_dir / "score_000.tsv")
    config = RunConfig(
        run_id="us4-refuse",
        master_seed=MASTER_SEED,
        scores=str(scores_dir / "*.tsv"),
        output_root=str(tmp_path / "out"),
        voices=[donor_ah],
        lanes={"deterministic": LaneToggle(enabled=True)},
        lyrics=LyricsConfig(
            source=LyricSource.GENERATED,
            theme="winter",
            model=LyricModel(model_id="x", license="proprietary", license_ok=False, model_ref="r"),
        ),
    )
    Orchestrator(config, build_lanes(config)).run()
    rec = load_manifest(Path(config.output_root) / "manifest.jsonl")[0]
    # SC-006: refusal is surfaced in the manifest, no sample produced.
    assert rec.verdict.status == VerdictStatus.LICENSE_REFUSED
    assert rec.lyric_source == "generated"
    assert not rec.audio_path
