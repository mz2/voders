"""Transposition axis: shift every pitch by a semitone offset, range-guarded (FR-004/005).

One variant per offset. Onsets/offsets/lyrics are untouched — only ``pitch_midi`` moves — so a
transposition variant's timing labels are byte-identical to its base (SC-002). The system MUST NEVER
emit a note outside MIDI ``0..127``: the ``drop`` policy discards the whole variant, ``clamp`` pins
offending pitches into the window and records how many were clamped.
"""

from __future__ import annotations

from voders.scores.models import Score

_HARD_LO, _HARD_HI = 0, 127


def transpose(
    score: Score,
    offset: int,
    *,
    policy: str = "drop",
    window: tuple[int, int] = (_HARD_LO, _HARD_HI),
) -> Score | None:
    """Return ``score`` with every pitch shifted by ``offset``, or ``None`` if dropped.

    ``policy="drop"`` (default): returns ``None`` when any shifted pitch leaves ``window``.
    ``policy="clamp"``: clamps offending pitches into ``window``. Either way no emitted pitch can
    fall outside the hard ``0..127`` bound.
    """
    result, _clamped = _transpose(score, offset, policy=policy, window=window)
    return result


def _transpose(
    score: Score, offset: int, *, policy: str, window: tuple[int, int]
) -> tuple[Score | None, int]:
    """Internal: returns ``(shifted score | None, clamped_count)`` for provenance (FR-005)."""
    lo, hi = window
    lo, hi = max(lo, _HARD_LO), min(hi, _HARD_HI)
    shifted = [n.pitch_midi + offset for n in score.notes]

    if policy == "drop":
        if any(p < lo or p > hi for p in shifted):
            return None, 0
        new_notes = [
            n.model_copy(update={"pitch_midi": p})
            for n, p in zip(score.notes, shifted, strict=True)
        ]
        return score.model_copy(update={"notes": new_notes}), 0

    # clamp policy: pin offending pitches into the window and count them.
    clamped = 0
    new_notes = []
    for n, p in zip(score.notes, shifted, strict=True):
        capped = min(max(p, lo), hi)
        if capped != p:
            clamped += 1
        new_notes.append(n.model_copy(update={"pitch_midi": capped}))
    return score.model_copy(update={"notes": new_notes}), clamped
