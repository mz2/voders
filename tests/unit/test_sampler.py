"""Unit tests for the automatic CV-syllable sampler (T024, FR-003/FR-004, SC-003/SC-004).

The sampler is the seed-deterministic, CPU-only, coverage-biased syllable source that needs no
input data. These tests pin its three contracts: determinism (independent of call order/global
state), seed-sensitivity, and phonetic-coverage spread across the inventory.
"""

from __future__ import annotations

from voders.lyrics.sampler import load_inventory, sample_syllables
from voders.seeds import sample_seed


def test_load_inventory_is_broad_and_clean() -> None:
    """The checked-in en_cv inventory loads to many distinct, comment-free ASCII syllables."""
    inventory = load_inventory("en_cv")
    assert len(inventory) >= 80
    assert len(set(inventory)) == len(inventory)  # all distinct
    for syllable in inventory:
        assert syllable  # non-empty
        assert not syllable.startswith("#")  # comments stripped
        assert syllable.isascii()
        assert syllable == syllable.strip()


def test_sampler_is_deterministic_for_a_seed() -> None:
    """Same (n, seed, inventory) => identical syllable list across repeated calls (FR-004)."""
    inventory = load_inventory("en_cv")
    seed = sample_seed(20260627, "score_000", "svs_ah", "lyrics")
    first = sample_syllables(64, seed, inventory)
    second = sample_syllables(64, seed, inventory)
    assert first == second
    assert len(first) == 64


def test_sampler_is_independent_of_call_order_and_global_state() -> None:
    """Interleaving other draws/seeds does not perturb a later draw (FR-004, SC-004)."""
    inventory = load_inventory("en_cv")
    seed = sample_seed(20260627, "score_000", "svs_ah", "lyrics")
    expected = sample_syllables(40, seed, inventory)

    # Draw unrelated seeds in between; the target draw must be unchanged.
    sample_syllables(123, sample_seed(20260627, "score_999", "other", "lyrics"), inventory)
    sample_syllables(7, sample_seed(1, "x", "y", "lyrics"), inventory)
    again = sample_syllables(40, seed, inventory)
    assert again == expected


def test_sampler_differs_by_score_and_voice() -> None:
    """Different score_id or voice_id (=> different seed) generally differs (FR-004)."""
    inventory = load_inventory("en_cv")
    base = sample_syllables(64, sample_seed(42, "score_000", "v1", "lyrics"), inventory)
    other_score = sample_syllables(64, sample_seed(42, "score_001", "v1", "lyrics"), inventory)
    other_voice = sample_syllables(64, sample_seed(42, "score_000", "v2", "lyrics"), inventory)
    assert base != other_score
    assert base != other_voice


def test_sampler_is_coverage_biased_over_a_long_score() -> None:
    """Across 200 notes the draw spreads across the inventory, not one repeat (SC-003)."""
    inventory = load_inventory("en_cv")
    seed = sample_seed(20260627, "long_score", "svs_ah", "lyrics")
    syllables = sample_syllables(200, seed, inventory)
    assert len(syllables) == 200
    distinct = len(set(syllables))
    assert distinct >= 20
    # Coverage bias: a long score should reach (nearly) the whole inventory.
    assert distinct >= min(len(inventory), 80)


def test_sampler_returns_only_inventory_members() -> None:
    inventory = load_inventory("en_cv")
    seed = sample_seed(1, "s", "v", "lyrics")
    drawn = sample_syllables(50, seed, inventory)
    assert all(s in set(inventory) for s in drawn)


def test_sampler_handles_empty_and_short_scores() -> None:
    inventory = load_inventory("en_cv")
    assert sample_syllables(0, 123, inventory) == []
    one = sample_syllables(1, 123, inventory)
    assert len(one) == 1 and one[0] in set(inventory)


def test_mandarin_inventory_is_single_character_and_distinct() -> None:
    inventory = load_inventory("zh_cv")
    assert len(inventory) >= 64
    assert len(set(inventory)) == len(inventory)
    assert all(len(syllable) == 1 and "\u4e00" <= syllable <= "\u9fff" for syllable in inventory)
