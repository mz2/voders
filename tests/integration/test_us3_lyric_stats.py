"""US3 integration: the run stats report surfaces the lyric axis (FR-014).

A run's stats include a lyric-source breakdown, the supplied multi-syllable count, and a
phonetic-coverage summary over the label-borne syllables (the SC-003 signal vs the vowel baseline).
"""

from __future__ import annotations

from pathlib import Path

from voders.config.models import LaneToggle, LyricsConfig, RunConfig
from voders.corpus.orchestrator import Orchestrator
from voders.corpus.stats import build_stats
from voders.lyrics.models import LyricSource
from voders.render.registry import build_lanes

MASTER_SEED = 20260627


def _supplied_corpus(tmp_path: Path, donor_ah) -> RunConfig:
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    # Two single-syllable cells + one multi-syllable cell ("winter") taken as authored.
    (scores_dir / "sup.tsv").write_text(
        "0.000000\t0.500000\t60\tla\n"
        "0.500000\t1.000000\t62\tdee\n"
        "1.000000\t1.500000\t64\twinter\n",
        encoding="utf-8",
    )
    return RunConfig(
        run_id="us3-stats",
        master_seed=MASTER_SEED,
        scores=str(scores_dir / "*.tsv"),
        output_root=str(tmp_path / "out"),
        voices=[donor_ah],
        lanes={"deterministic": LaneToggle(enabled=True)},
        lyrics=LyricsConfig(source=LyricSource.SUPPLIED),
    )


def test_stats_report_includes_lyric_axis(tmp_path: Path, donor_ah) -> None:
    config = _supplied_corpus(tmp_path, donor_ah)
    Orchestrator(config, build_lanes(config)).run()

    stats = build_stats(Path(config.output_root) / "manifest.jsonl")
    lyrics = stats["lyrics"]

    assert lyrics["by_source"].get("supplied", 0) >= 1
    # One multi-syllable supplied cell ("winter") across the corpus.
    assert lyrics["multisyllable_supplied_total"] == 1
    # Coverage over label-borne syllables: distinct syllables beyond a single vowel (SC-003 signal).
    assert lyrics["coverage"]["distinct_syllables"] >= 3
    assert lyrics["coverage"]["distinct_onsets"] >= 2
