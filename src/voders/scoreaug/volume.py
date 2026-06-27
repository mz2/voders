"""Per-note volume / dynamics axis: a seeded gain the renderer honours (FR-008/009).

Assigns each note a seeded linear gain drawn from ``gain_db_range`` (in dB, converted to a linear
amplitude multiplier). The timing/pitch/lyric labels are untouched — the gain rides in-memory to the
renderer and is *never* serialised into the label ``.tsv`` — so ``(onset, offset, pitch)`` stay
byte-identical to the un-varied render (SC-004). Identical output for a fixed ``sub_seed``.
"""

from __future__ import annotations

import numpy as np

from voders.config.models import VolumeKnob
from voders.scores.models import Note, Score


def _db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (gain_db / 20.0))


def volume(score: Score, knob: VolumeKnob, sub_seed: int) -> Score:
    """Return ``score`` with each note carrying a seeded per-note ``gain`` (label-preserving)."""
    rng = np.random.default_rng(sub_seed)
    lo, hi = knob.gain_db_range
    new_notes: list[Note] = []
    for n in score.notes:
        gain_db = float(rng.uniform(lo, hi))
        new_notes.append(n.model_copy(update={"gain": _db_to_linear(gain_db)}))
    return score.model_copy(update={"notes": new_notes})
