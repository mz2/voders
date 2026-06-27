"""Time-humanisation axis: seeded onset/duration jitter, always a valid monophonic score (FR-006/7).

A *clamp-then-project* constraint solver: each note's onset and duration are jittered by a seeded
Gaussian draw clamped to ``±max_dev_s``, then projected left-to-right so the emitted score stays
ordered, non-overlapping (``overlap="forbid"``), and positive in duration (``>= min_dur_s``).
When projection has to override a draw to keep the score valid, the note is pinned to base timing
(clamped for non-overlap) and a ``constraint_hit`` is recorded — an invalid score is never emitted.
Pitch and lyric are untouched (FR-006). Identical output for a fixed ``sub_seed`` (G1/SC-005).
"""

from __future__ import annotations

import numpy as np

from voders.config.models import HumanizeKnob
from voders.scores.models import Note, Score

_EPS = 1e-9


def humanize(score: Score, knob: HumanizeKnob, sub_seed: int) -> Score:
    """Return one humanised variant of ``score`` for the given draw seed."""
    result, _meta = _humanize(score, knob, sub_seed)
    return result


def _humanize(score: Score, knob: HumanizeKnob, sub_seed: int) -> tuple[Score, dict[str, int]]:
    """Internal: returns ``(variant score, {"constraint_hit": n})`` for provenance."""
    rng = np.random.default_rng(sub_seed)
    max_dev = knob.max_dev_s
    min_dur = knob.min_dur_s

    new_notes: list[Note] = []
    prev_offset = 0.0
    constraint_hit = 0

    for n in score.notes:
        d_on = float(np.clip(rng.normal(0.0, knob.onset_sigma_s), -max_dev, max_dev))
        d_dur = float(np.clip(rng.normal(0.0, knob.duration_sigma_s), -max_dev, max_dev))

        # Jittered targets, each clamped to within the per-note deviation budget of the base.
        onset = n.onset_s + d_on
        offset = n.onset_s + (n.duration_s + d_dur)

        hit = False
        # Non-overlap (forbid): onset cannot precede the previous note's emitted offset.
        if onset < prev_offset - _EPS:
            onset = prev_offset
            hit = True
        # Stay within the onset budget after projection (pin to base timing if pushed too far).
        if abs(onset - n.onset_s) > max_dev + _EPS:
            onset = max(n.onset_s, prev_offset)
            hit = True
        # Strictly positive, minimum duration.
        if offset < onset + min_dur:
            offset = onset + min_dur
            hit = True
        # Keep the offset within the duration budget of the base.
        if abs(offset - n.offset_s) > max_dev + _EPS:
            offset = min(max(n.offset_s, onset + min_dur), onset + n.duration_s + max_dev)
            hit = True

        constraint_hit += int(hit)
        new_notes.append(n.model_copy(update={"onset_s": onset, "offset_s": offset}))
        prev_offset = offset

    variant = score.model_copy(update={"notes": new_notes})
    return variant, {"constraint_hit": constraint_hit}
