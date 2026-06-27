"""Automatic CV-syllable sampler (T027, FR-003/FR-004, SC-003/SC-004).

The ``automatic`` lyric source needs no input data: it draws singable consonant–vowel syllables from
a checked-in inventory, seed-deterministically and biased toward broad phonetic coverage. CPU-only —
no ``torch``/``phonemizer``; only the standard library, ``numpy`` (via :mod:`voders.seeds`), and the
in-package text inventory are used here, so importing this module stays light (FR-005).
"""

from __future__ import annotations

from functools import cache
from importlib import resources

from voders.seeds import rng


@cache
def _load_inventory_cached(name: str) -> tuple[str, ...]:
    """Parse and cache the named inventory's syllables (comments/blank lines stripped)."""
    data = resources.files("voders.lyrics") / "data" / f"{name}.txt"
    text = data.read_text(encoding="utf-8")
    syllables = [
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]
    if not syllables:
        raise ValueError(f"lyric inventory {name!r} is empty")
    return tuple(syllables)


def load_inventory(name: str = "en_cv") -> list[str]:
    """Return the syllables of the named checked-in inventory (cached; FR-003).

    The result is a fresh list each call (callers may keep/mutate it without disturbing the cache).
    """
    return list(_load_inventory_cached(name))


def sample_syllables(n: int, seed: int, inventory: list[str]) -> list[str]:
    """Draw ``n`` coverage-biased syllables deterministically from ``inventory`` (FR-004).

    Pure function of ``(n, seed, inventory)``: independent of call order or any global state. The
    draw shuffles the whole inventory per chunk and concatenates chunks until ``n`` syllables are
    produced, so a long score spans every distinct syllable before any one repeats — maximising
    phonetic coverage (SC-003) — while a short score still draws distinct syllables.
    """
    if n <= 0:
        return []
    if not inventory:
        raise ValueError("cannot sample from an empty inventory")
    generator = rng(seed)
    out: list[str] = []
    while len(out) < n:
        chunk = list(inventory)
        generator.shuffle(chunk)
        out.extend(chunk)
    return out[:n]
