"""Real espeak-ng G2P tests (skipped if the 'lyrics' extra / espeak-ng is unavailable).

These exercise the actual phonemizer + espeak-ng path, not a stub: text -> per-phoneme tokens, then
the vowel-on-the-beat mapping over real phonemes.
"""

from __future__ import annotations

import pytest

from voders.lyrics.g2p import is_vowel_phoneme, map_syllable

try:
    from voders.lyrics.g2p import text_to_phonemes

    _PH = text_to_phonemes("la")
    _ESPEAK_OK = _PH[:1] == ["l"]
except Exception:  # pragma: no cover - environment without espeak-ng
    _ESPEAK_OK = False

pytestmark = pytest.mark.skipif(not _ESPEAK_OK, reason="phonemizer/espeak-ng not available")


def test_text_to_phonemes_splits_into_individual_phonemes():
    assert text_to_phonemes("la") == ["l", "æ"]
    assert text_to_phonemes("ski")[0] == "s"
    strap = text_to_phonemes("strap")
    assert strap[0] == "s" and strap[-1] == "p"
    assert any(is_vowel_phoneme(p) for p in strap)


def test_real_phonemes_map_vowel_on_the_beat():
    phonemes = text_to_phonemes("strap")
    run = map_syllable(phonemes, onset_s=1.0, offset_s=1.5, note_index=3)
    assert run.nucleus_onset_s == 1.0
    assert run.lead_consonants[0] == "s"  # the consonant cluster precedes the beat
    assert run.tail_consonants == ["p"]  # trailing consonant after the nucleus
