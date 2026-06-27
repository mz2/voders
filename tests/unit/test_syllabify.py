"""Unit tests for the deterministic syllable segmenter (FR-019, SC-010).

``segment`` splits words/free text into an ordered list of singable syllables; ``syllable_count``
reports how many syllables a token holds (used to flag non-single-syllable supplied cells). Both are
pure, deterministic CPU functions with no heavy imports (FR-005).
"""

from __future__ import annotations

from voders.lyrics.syllabify import segment, syllable_count


def test_segment_is_deterministic():
    assert segment("winter longing") == segment("winter longing")


def test_segment_splits_into_one_vowel_group_per_syllable():
    syllables = segment("winter longing")
    # Two 2-syllable words → 4 syllables; the exact consonant split is not constrained (singable,
    # not linguistically authoritative — spec Assumptions), but each token is one vowel group.
    assert len(syllables) == 4
    assert "".join(syllables) == "winterlonging"
    vowels = set("aeiouy")
    for syl in syllables:
        groups = 0
        prev = False
        for ch in syl:
            v = ch in vowels
            if v and not prev:
                groups += 1
            prev = v
        assert groups == 1, f"{syl!r} is not a single vowel group"


def test_segment_canonical_simple_split():
    # Where a single inter-vowel consonant exists, the split is canonical V-CV / VC-CV.
    assert segment("winter") == ["win", "ter"]
    assert segment("water") == ["wa", "ter"]


def test_segment_single_syllable_word_is_one_token():
    assert segment("la") == ["la"]
    assert segment("cat") == ["cat"]


def test_segment_preserves_letters():
    assert "".join(segment("winter")) == "winter"


def test_segment_drops_non_alpha_tokens():
    assert segment("123") == []
    assert segment("!!!") == []
    assert segment("") == []


def test_syllable_count_single_syllable_is_one():
    for word in ("la", "cat", "the", "size", "make"):
        assert syllable_count(word) == 1, word


def test_syllable_count_multi_syllable():
    assert syllable_count("winter") == 2
    assert syllable_count("water") == 2
    assert syllable_count("table") == 2  # syllabic -le
    assert syllable_count("longing") == 2


def test_syllable_count_non_alpha_is_zero():
    assert syllable_count("123") == 0
    assert syllable_count("") == 0
    assert syllable_count("   ") == 0


# FR-005 import-safety for syllabify is covered robustly (subprocess-style pop) in
# tests/unit/test_lyrics_imports.py; not duplicated here to avoid cross-test sys.modules pollution.
