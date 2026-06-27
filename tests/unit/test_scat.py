"""The scat-singing syllable style for the automatic source (Scatman-flavoured).

Selected via ``lyrics.inventory: scat``; a jazz-vocalese nonsense-syllable inventory whose only job
is phonetic diversity. Each entry is a single singable syllable (one-syllable-per-note, FR-019).
"""

from __future__ import annotations

from voders.lyrics.sampler import load_inventory, sample_syllables
from voders.lyrics.sources import AutomaticSource
from voders.lyrics.syllabify import syllable_count
from voders.scores.models import Note, Score


def test_scat_inventory_loads_and_is_scatty():
    inv = load_inventory("scat")
    assert len(inv) >= 30
    # Signature Scatman syllables are present.
    for syl in ("ski", "ba", "bop", "dop", "doo"):
        assert syl in inv


def test_scat_entries_are_single_syllables():
    for syl in load_inventory("scat"):
        assert syllable_count(syl) == 1, syl


def test_scat_sampler_is_deterministic_and_in_inventory():
    inv = load_inventory("scat")
    a = sample_syllables(12, 42, inv)
    assert a == sample_syllables(12, 42, inv)  # deterministic (FR-004)
    assert all(s in inv for s in a)


def test_automatic_source_with_scat_style():
    score = Score(
        score_id="s", notes=[Note(onset_s=i, offset_s=i + 0.5, pitch_midi=60) for i in range(8)]
    )
    plan = AutomaticSource(inventory="scat").resolve(score, master_seed=1, voice_id="v")
    inv = set(load_inventory("scat"))
    assert len(plan.syllables) == 8
    assert all(s in inv for s in plan.syllables)
    assert plan.multisyllable_notes == []  # one syllable per note (SC-010)
