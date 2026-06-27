"""Unit tests for the lyric data carriers (T006, FR-008/FR-010, data-model.md).

Covers the ``LyricPlan`` length invariant and deterministic ``text_hash``, the ``LyricModel``
license gate (mirroring ``Voice.is_usable``), and the ``PhonemeRun`` dataclass shape.
"""

from __future__ import annotations

from voders.lyrics.models import LyricModel, LyricPlan, LyricSource, PhonemeRun


def test_lyric_source_values():
    assert LyricSource.VOWEL == "vowel"
    assert LyricSource.SUPPLIED == "supplied"
    assert LyricSource.AUTOMATIC == "automatic"
    assert LyricSource.GENERATED == "generated"


def test_text_hash_is_deterministic():
    a = LyricPlan.build("s1", LyricSource.SUPPLIED, ["la", "le", None])
    b = LyricPlan.build("s1", LyricSource.SUPPLIED, ["la", "le", None])
    assert a.text_hash == b.text_hash
    assert isinstance(a.text_hash, str) and a.text_hash


def test_text_hash_differs_when_syllables_differ():
    a = LyricPlan.build("s1", LyricSource.SUPPLIED, ["la", "le", None])
    b = LyricPlan.build("s1", LyricSource.SUPPLIED, ["la", "lo", None])
    assert a.text_hash != b.text_hash


def test_text_hash_distinguishes_none_from_empty_and_order():
    # None must hash distinctly from the empty string ...
    assert (
        LyricPlan.build("s1", LyricSource.SUPPLIED, [None]).text_hash
        != LyricPlan.build("s1", LyricSource.SUPPLIED, [""]).text_hash
    )
    # ... and order matters (note-ordered canonical list).
    assert (
        LyricPlan.build("s1", LyricSource.SUPPLIED, ["la", "le"]).text_hash
        != LyricPlan.build("s1", LyricSource.SUPPLIED, ["le", "la"]).text_hash
    )


def test_build_keeps_syllables_length():
    syllables = ["la", None, "le", None, "lo"]
    plan = LyricPlan.build("s1", LyricSource.AUTOMATIC, syllables)
    assert len(plan.syllables) == len(syllables)
    assert plan.syllables == syllables
    assert plan.score_id == "s1"
    assert plan.source == LyricSource.AUTOMATIC
    assert plan.mismatch is False
    assert plan.model is None


def test_build_records_mismatch_and_model():
    model = LyricModel(model_id="m", license_ok=True)
    plan = LyricPlan.build("s1", LyricSource.GENERATED, ["la"], model=model, mismatch=True)
    assert plan.mismatch is True
    assert plan.model is model


def test_lyric_model_is_usable_requires_license_ok():
    assert LyricModel(model_id="m", license="cc", license_ok=True).is_usable() is True
    assert LyricModel(model_id="m", license="cc", license_ok=False).is_usable() is False
    assert LyricModel().is_usable() is False


def test_phoneme_run_shape():
    run = PhonemeRun(
        note_index=2,
        phonemes=["l", "a"],
        nucleus_onset_s=1.5,
        lead_consonants=["l"],
        tail_consonants=[],
    )
    assert run.note_index == 2
    assert run.phonemes == ["l", "a"]
    assert run.nucleus_onset_s == 1.5
    assert run.lead_consonants == ["l"]
    assert run.tail_consonants == []
