"""US1 integration: the lyric layer is wired through the orchestrator (FR-006/009, SC-001/004).

These tests prove the lyric source is actually invoked by a run and its provenance recorded: the
``vowel`` default stays byte-identical (SC-001), ``automatic`` records a source + stable hash that
reproduces across runs (SC-004), and ``supplied`` cells are taken as authored with multi-syllable
cells flagged (FR-019). The deterministic lane does not articulate (``lyric_articulated`` is False).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from voders.config.models import LaneToggle, LyricsConfig, RunConfig
from voders.corpus.orchestrator import Orchestrator
from voders.lyrics.models import LyricSource
from voders.manifest.io import load_manifest
from voders.render.registry import build_lanes

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"
MASTER_SEED = 20260627


def _config(tmp_path: Path, voice, lyrics: LyricsConfig | None = None) -> RunConfig:
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SCORES_DIR / "score_000.tsv", scores_dir / "score_000.tsv")
    kwargs: dict = {}
    if lyrics is not None:
        kwargs["lyrics"] = lyrics
    return RunConfig(
        run_id="us1-lyrics",
        master_seed=MASTER_SEED,
        scores=str(scores_dir / "*.tsv"),
        output_root=str(tmp_path / "out"),
        voices=[voice],
        lanes={"deterministic": LaneToggle(enabled=True)},
        **kwargs,
    )


def _run(config: RunConfig):
    Orchestrator(config, build_lanes(config)).run()
    return load_manifest(Path(config.output_root) / "manifest.jsonl")


def test_vowel_run_keeps_inert_lyric_axis_and_byte_identity(tmp_path: Path, donor_ah) -> None:
    records = _run(_config(tmp_path, donor_ah))
    rec = records[0]
    assert rec.lyric_source == "vowel"
    assert rec.lyric_hash is None
    assert rec.lyric_articulated is False
    assert rec.lyric_multisyllable_supplied == 0
    # SC-001: the produced label .tsv is byte-identical to the original 3-column source.
    produced = list((tmp_path / "out").rglob("*.tsv"))
    original = (SCORES_DIR / "score_000.tsv").read_bytes()
    assert produced and any(p.read_bytes() == original for p in produced)


def test_automatic_run_records_source_and_stable_hash(tmp_path: Path, donor_ah) -> None:
    cfg = _config(tmp_path, donor_ah, LyricsConfig(source=LyricSource.AUTOMATIC))
    rec = _run(cfg)[0]
    assert rec.lyric_source == "automatic"
    assert rec.lyric_hash is not None
    # Deterministic lane does not articulate phonemes (FR-006).
    assert rec.lyric_articulated is False


def test_automatic_hash_is_reproducible_across_runs(tmp_path: Path, donor_ah) -> None:
    # SC-004: the same (master_seed, score, voice) yields the same assigned syllables (hash).
    rec_a = _run(_config(tmp_path / "a", donor_ah, LyricsConfig(source=LyricSource.AUTOMATIC)))[0]
    rec_b = _run(_config(tmp_path / "b", donor_ah, LyricsConfig(source=LyricSource.AUTOMATIC)))[0]
    assert rec_a.lyric_hash == rec_b.lyric_hash


def test_supplied_run_flags_multisyllable_cells(tmp_path: Path, donor_ah) -> None:
    # A 4-column supplied score with one multi-syllable cell ("winter").
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "sup.tsv").write_text(
        "0.000000\t0.500000\t60\tla\n0.500000\t1.000000\t62\twinter\n", encoding="utf-8"
    )
    cfg = RunConfig(
        run_id="us1-sup",
        master_seed=MASTER_SEED,
        scores=str(scores_dir / "*.tsv"),
        output_root=str(tmp_path / "out"),
        voices=[donor_ah],
        lanes={"deterministic": LaneToggle(enabled=True)},
        lyrics=LyricsConfig(source=LyricSource.SUPPLIED),
    )
    rec = _run(cfg)[0]
    assert rec.lyric_source == "supplied"
    assert rec.lyric_hash is not None
    assert rec.lyric_multisyllable_supplied == 1
