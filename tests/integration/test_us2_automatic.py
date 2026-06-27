"""Integration tests for the automatic lyric source (US2, T025, FR-003/FR-004, SC-003/SC-004).

US2 is the seed-deterministic, CPU-only automatic syllable source that needs no input data and
broadens phonetic coverage. Articulation (G2P/SVS) is out of scope here: these tests cover only
assigning syllables to notes, the resulting coverage signal, and order-independent reproducibility.
"""

from __future__ import annotations

from pathlib import Path

from voders.config.loader import load_config
from voders.lyrics.coverage import phonetic_coverage
from voders.lyrics.models import LyricSource
from voders.lyrics.sampler import load_inventory
from voders.lyrics.sources import AutomaticSource, resolve_source
from voders.scores.models import Note, Score

_MASTER_SEED = 20260627
_FIXTURE_DIR = Path(__file__).resolve().parents[1].parent / "evals" / "fixtures"


def _score(score_id: str, n_notes: int) -> Score:
    notes = [
        Note(onset_s=float(i), offset_s=float(i) + 0.5, pitch_midi=60 + (i % 12))
        for i in range(n_notes)
    ]
    return Score(score_id=score_id, notes=notes)


def test_resolve_source_returns_automatic() -> None:
    source = resolve_source("automatic")
    assert isinstance(source, AutomaticSource)
    assert source.name == "automatic"
    assert source.requires_gpu() is False
    assert resolve_source(LyricSource.AUTOMATIC).name == "automatic"


def test_automatic_resolve_assigns_one_inventory_syllable_per_note() -> None:
    """Every note gets a non-None syllable drawn from the inventory; lengths match (FR-008)."""
    inventory = set(load_inventory("en_cv"))
    score = _score("score_abc", 25)
    plan = AutomaticSource().resolve(score, master_seed=_MASTER_SEED, voice_id="svs_ah")
    assert plan.source is LyricSource.AUTOMATIC
    assert plan.score_id == "score_abc"
    assert len(plan.syllables) == len(score.notes)
    assert all(s is not None for s in plan.syllables)
    assert all(s in inventory for s in plan.syllables)


def test_automatic_resolve_is_deterministic() -> None:
    score = _score("score_abc", 30)
    a = AutomaticSource().resolve(score, master_seed=_MASTER_SEED, voice_id="svs_ah")
    b = AutomaticSource().resolve(score, master_seed=_MASTER_SEED, voice_id="svs_ah")
    assert a.syllables == b.syllables
    assert a.text_hash == b.text_hash


def test_sc003_corpus_coverage_is_order_of_magnitude_over_vowel_baseline() -> None:
    """SC-003 proxy: distinct syllables across a corpus far exceed the vowel baseline (= 1)."""
    source = AutomaticSource()
    scores = [_score(f"score_{i:03d}", 24) for i in range(8)]
    all_syllables: list[str | None] = []
    for score in scores:
        plan = source.resolve(score, master_seed=_MASTER_SEED, voice_id="svs_ah")
        all_syllables.extend(plan.syllables)

    distinct = len({s for s in all_syllables if s is not None})
    vowel_baseline = 1
    assert distinct >= 10 * vowel_baseline  # >= 10x the vowel-only baseline

    coverage = phonetic_coverage(all_syllables)
    assert coverage["distinct_syllables"] == distinct
    assert coverage["distinct_onsets"] >= 5
    assert coverage["distinct_vowels"] >= 4


def test_sc004_reordering_scores_yields_identical_per_score_syllables() -> None:
    """SC-004: assignment depends only on (seed, score_id, voice_id), not processing order."""
    source = AutomaticSource()
    scores = [_score(f"score_{i:03d}", 18) for i in range(6)]

    forward = {
        s.score_id: source.resolve(s, master_seed=_MASTER_SEED, voice_id="svs_ah").syllables
        for s in scores
    }
    reverse = {
        s.score_id: source.resolve(s, master_seed=_MASTER_SEED, voice_id="svs_ah").syllables
        for s in reversed(scores)
    }
    assert forward == reverse


def test_lyrics_smoke_fixture_loads_with_automatic_source() -> None:
    """The lyrics-smoke fixture is a valid RunConfig selecting the automatic source (T026)."""
    config = load_config(_FIXTURE_DIR / "lyrics-smoke.yaml")
    assert config.lyrics.source == LyricSource.AUTOMATIC
    assert config.lyrics.inventory == "en_cv"
    assert config.lanes["svs"].enabled is True
