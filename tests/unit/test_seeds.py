"""Unit tests for deterministic seed derivation (T048, FR-013)."""

from __future__ import annotations

import numpy as np

from voders.seeds import derive_seed, rng, sample_seed


def test_derive_seed_is_deterministic() -> None:
    """The same (master_seed, parts) always yields the same seed."""
    assert derive_seed(42, "a", "b") == derive_seed(42, "a", "b")
    assert derive_seed(42, "score", "voice", "deterministic") == derive_seed(
        42, "score", "voice", "deterministic"
    )


def test_derive_seed_is_non_negative_63_bit() -> None:
    seed = derive_seed(42, "a", "b")
    assert 0 <= seed < (1 << 63)


def test_derive_seed_is_order_sensitive() -> None:
    """A unit separator means ("a","b") differs from ("ab",) and from ("b","a")."""
    assert derive_seed(42, "a", "b") != derive_seed(42, "ab")
    assert derive_seed(42, "a", "b") != derive_seed(42, "b", "a")


def test_derive_seed_depends_on_master_seed() -> None:
    assert derive_seed(1, "a") != derive_seed(2, "a")


def test_sample_seed_is_stable_and_order_independent() -> None:
    """sample_seed is stable across calls and independent of call order (FR-013)."""
    a = sample_seed(7, "score_000", "donor_ah", "deterministic")
    b = sample_seed(7, "score_000", "donor_ah", "deterministic")
    assert a == b

    # Deriving other samples in between does not change a later derivation.
    sample_seed(7, "score_999", "donor_oo", "svs")
    c = sample_seed(7, "score_000", "donor_ah", "deterministic")
    assert c == a


def test_sample_seed_differs_by_stage_and_identity() -> None:
    base = sample_seed(7, "s", "v", "deterministic")
    assert base != sample_seed(7, "s", "v", "svs")
    assert base != sample_seed(7, "s", "v2", "deterministic")
    assert base != sample_seed(7, "s2", "v", "deterministic")


def test_rng_is_reproducible() -> None:
    """rng(seed) produces the same draws given the same seed."""
    seed = sample_seed(7, "s", "v", "deterministic")
    first = rng(seed).standard_normal(16)
    second = rng(seed).standard_normal(16)
    assert np.array_equal(first, second)
    # A different seed gives different draws.
    other = rng(seed + 1).standard_normal(16)
    assert not np.array_equal(first, other)
