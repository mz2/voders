"""Phonetic-coverage signal for the lyric corpus (T028, FR-014, SC-003).

Summarises how much phonetic variety a corpus's syllables carry, versus the vowel-only baseline
(a corpus of open vowels yields a single distinct vowel — coverage 1). This is the SC-003 signal
the run stats report surfaces. CPU-only and dependency-free: no ``phonemizer``/``espeak``. The
onset/vowel split is a coarse, letter-based approximation (not true G2P) — adequate as a coverage
proxy but NOT a phonetic transcription.
"""

from __future__ import annotations

from collections.abc import Iterable

# Vowel letters used for the coarse onset/nucleus split. Approximate: ignores 'y' as a vowel and
# does not model digraphs (e.g. "ee", "oo") as single phonemes — both nuclei map to their first
# vowel letter, which is fine for a coverage signal.
_VOWEL_LETTERS = frozenset("aeiou")


def _onset(syllable: str) -> str:
    """The leading run of consonant letters (the approximate onset); "" if vowel-initial."""
    onset: list[str] = []
    for ch in syllable.lower():
        if ch in _VOWEL_LETTERS:
            break
        if ch.isalpha():
            onset.append(ch)
    return "".join(onset)


def _vowel(syllable: str) -> str:
    """The first vowel letter (the approximate nucleus); "" if none present."""
    for ch in syllable.lower():
        if ch in _VOWEL_LETTERS:
            return ch
    return ""


def phonetic_coverage(syllables: Iterable[str | None]) -> dict[str, int]:
    """Count distinct syllables, onset consonants, and vowels observed (``None`` ignored).

    Returns a dict with ``distinct_syllables``, ``distinct_onsets``, ``distinct_vowels``. The
    onset/vowel extraction is the coarse letter-based approximation documented above, intended as
    the SC-003 coverage signal against the vowel-only baseline (which yields 1).
    """
    seen_syllables: set[str] = set()
    seen_onsets: set[str] = set()
    seen_vowels: set[str] = set()
    for syllable in syllables:
        if syllable is None:
            continue
        seen_syllables.add(syllable)
        onset = _onset(syllable)
        if onset:
            seen_onsets.add(onset)
        vowel = _vowel(syllable)
        if vowel:
            seen_vowels.add(vowel)
    return {
        "distinct_syllables": len(seen_syllables),
        "distinct_onsets": len(seen_onsets),
        "distinct_vowels": len(seen_vowels),
    }
