"""Multilingual lyric coverage: language selection, segmentation, and per-language G2P.

CPU tests for segmentation + seeded language choice; the real espeak per-language G2P test is gated.
"""

from __future__ import annotations

import pytest

from voders.lyrics.models import LyricSource
from voders.lyrics.sources import AutomaticSource, _pick_language
from voders.lyrics.syllabify import segment_multilingual
from voders.scores.models import Note, Score


def _score(n: int, sid: str = "s1") -> Score:
    return Score(
        score_id=sid, notes=[Note(onset_s=i, offset_s=i + 0.5, pitch_midi=60) for i in range(n)]
    )


def test_segment_cjk_is_per_character():
    assert segment_multilingual("さくら", "ja") == ["さ", "く", "ら"]
    assert segment_multilingual("春天", "cmn") == ["春", "天"]
    assert segment_multilingual("사랑", "ko") == ["사", "랑"]


def test_segment_european_is_accent_aware():
    # The English splitter would drop the umlaut; the multilingual path keeps it.
    syl = segment_multilingual("schön", "de")
    assert "".join(syl) == "schön"
    assert any("ö" in s for s in syl)


def test_pick_language_is_deterministic_and_spreads():
    langs = ["en-us", "de", "ja", "es", "ru"]
    a = _pick_language(langs, 7, "s1", "v1")
    assert a == _pick_language(langs, 7, "s1", "v1")  # deterministic
    assert a in langs
    chosen = {_pick_language(langs, 7, f"s{i}", "v1") for i in range(40)}
    assert len(chosen) >= 3  # spreads across several languages
    assert _pick_language(["en-us"], 7, "s1", "v1") == "en-us"  # single language is stable


def test_automatic_source_records_language():
    src = AutomaticSource(inventory="en_cv", languages=("de", "ja", "es"))
    plan = src.resolve(_score(4), master_seed=7, voice_id="v1")
    assert plan.source == LyricSource.AUTOMATIC
    assert plan.language in ("de", "ja", "es")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("phonemizer") is None,
    reason="needs the lyrics extra (phonemizer) + espeak-ng",
)
def test_espeak_phonemes_differ_by_language():
    from voders.lyrics.g2p import text_to_phonemes

    try:
        de = text_to_phonemes("schön", language="de")
        en = text_to_phonemes("schon", language="en-us")
    except RuntimeError:
        pytest.skip("espeak-ng not available")
    assert de and en and de != en
