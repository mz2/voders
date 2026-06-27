"""Unit tests for the phoneme->note mapping (FR-006, Decision L3).

The vowel-on-the-beat mapping is a pure function and needs no G2P backend, so these tests inject
phonemes directly (no espeak-ng dependency, FR-005).
"""

from __future__ import annotations

from voders.lyrics.g2p import is_vowel_phoneme, map_syllable


def test_vowel_nucleus_is_on_the_beat_with_leading_consonant():
    run = map_syllable(["l", "a"], onset_s=0.5, offset_s=1.0, note_index=2)
    assert run.note_index == 2
    assert run.nucleus_onset_s == 0.5  # the labelled onset is the vowel, on the score beat
    assert run.lead_consonants == ["l"]
    assert run.tail_consonants == []


def test_leading_and_trailing_consonants_split_around_nucleus():
    run = map_syllable(["s", "t", "a", "p"], onset_s=1.0, offset_s=1.5, note_index=0)
    assert run.lead_consonants == ["s", "t"]
    assert run.tail_consonants == ["p"]
    assert run.nucleus_onset_s == 1.0


def test_all_consonant_token_keeps_nucleus_on_beat():
    run = map_syllable(["m", "m"], onset_s=2.0, offset_s=2.4, note_index=1)
    assert run.nucleus_onset_s == 2.0  # onset unaffected by absence of a vowel
    assert run.lead_consonants == ["m", "m"]


def test_is_vowel_phoneme():
    assert is_vowel_phoneme("a")
    assert is_vowel_phoneme("ɑ")
    assert not is_vowel_phoneme("t")
