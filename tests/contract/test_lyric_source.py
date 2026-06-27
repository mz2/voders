"""Contract tests for the lyric source interface (T008, contracts/lyric-source.md).

Every source returns a ``LyricPlan`` whose ``syllables`` length equals the note count; ``vowel``
returns an all-``None`` plan (byte-identity path, SC-001); count reconciliation sets
``mismatch=True`` and never drops, adds, or reorders notes (FR-008, SC-005).
"""

from __future__ import annotations

import pytest

from voders.lyrics.models import LyricPlan, LyricSource
from voders.lyrics.sources import (
    LyricSourceProtocol,
    VowelSource,
    reconcile,
    resolve_source,
)
from voders.scores.models import Note, Score


def _score(n_notes: int, score_id: str = "s1") -> Score:
    notes = [Note(onset_s=float(i), offset_s=float(i) + 0.5, pitch_midi=60) for i in range(n_notes)]
    return Score(score_id=score_id, notes=notes)


def test_vowel_source_protocol_and_attributes():
    src = VowelSource()
    assert isinstance(src, LyricSourceProtocol)
    assert src.name == "vowel"
    assert src.requires_gpu() is False


def test_vowel_source_returns_all_none_plan_of_note_length():
    score = _score(4)
    plan = src_plan = VowelSource().resolve(score, master_seed=7, voice_id="v1")
    assert isinstance(plan, LyricPlan)
    assert plan.source == LyricSource.VOWEL
    assert plan.score_id == "s1"
    assert len(plan.syllables) == len(score.notes)
    assert all(s is None for s in src_plan.syllables)
    assert plan.mismatch is False


def test_reconcile_pads_when_too_few():
    out, mismatch = reconcile(["la", "le"], 4)
    assert out == ["la", "le", None, None]
    assert mismatch is True


def test_reconcile_truncates_when_too_many():
    out, mismatch = reconcile(["la", "le", "lo", "lu"], 2)
    assert out == ["la", "le"]
    assert mismatch is True


def test_reconcile_equal_has_no_mismatch_and_preserves_order():
    syllables = ["la", None, "lo"]
    out, mismatch = reconcile(syllables, 3)
    assert out == syllables
    assert mismatch is False


def test_reconcile_never_reorders():
    out, _ = reconcile(["z", "a", "m"], 3)
    assert out == ["z", "a", "m"]


def test_resolve_source_vowel():
    src = resolve_source("vowel")
    assert isinstance(src, VowelSource)
    assert resolve_source(LyricSource.VOWEL).name == "vowel"


def test_resolve_source_automatic_is_implemented():
    """``automatic`` lands in US2 (FR-003); it resolves to a CPU source, not NotImplementedError."""
    src = resolve_source("automatic")
    assert src.name == "automatic"
    assert src.requires_gpu() is False
    assert resolve_source(LyricSource.AUTOMATIC).name == "automatic"


@pytest.mark.parametrize("name", ["supplied", "generated"])
def test_resolve_source_unimplemented_raises(name: str):
    with pytest.raises(NotImplementedError):
        resolve_source(name)
