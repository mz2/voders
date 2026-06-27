"""Procedural singable monophonic melody generator — coverage for the score/label axis.

Generates deterministic, in-domain vocal melodies as :class:`Score` objects: diatonic, within a
singable tessitura, monophonic by construction, with varied contour (mostly stepwise with occasional
leaps), varied rhythm, and the odd breath rest. This complements the score-domain augmentations
(#8, a *density* lever that perturbs existing scores) and external-MIDI ingest (#9, real-world
coverage): it *invents* new melodic structure — keys, intervals, phrase shapes — that the handful of
seed fixtures don't contain, while staying singable so the validator gate keeps near-100% yield.
"""

from __future__ import annotations

import random

from voders.scores.models import Note, Score

# Diatonic modes as the semitone offsets of their scale degrees from the tonic.
_MODES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
}
# Note durations in seconds, weighted toward eighth/quarter values (a free-time singing feel).
_DURATIONS: tuple[float, ...] = (0.25, 0.375, 0.5, 0.5, 0.75, 1.0)
# Scale-degree moves: mostly steps, occasionally a small leap.
_STEPS: tuple[int, ...] = (-2, -1, -1, -1, 1, 1, 1, 2)
_LEAPS: tuple[int, ...] = (-4, -3, 3, 4)
_REST_S: tuple[float, ...] = (0.15, 0.25, 0.3)


def _scale_pitches(tonic_pc: int, intervals: tuple[int, ...], lo: int, hi: int) -> list[int]:
    """Every MIDI pitch of the given mode/tonic within the inclusive singable range [lo, hi]."""
    pitches = {
        octave * 12 + tonic_pc + iv
        for octave in range(11)
        for iv in intervals
        if lo <= octave * 12 + tonic_pc + iv <= hi
    }
    return sorted(pitches)


def generate_melody(
    seed: int,
    *,
    score_id: str,
    lo: int = 55,
    hi: int = 79,
    min_notes: int = 6,
    max_notes: int = 16,
) -> Score:
    """A deterministic singable melody for ``seed``: diatonic, monophonic, in [lo, hi] (FR-001).

    The contour is a bounded random walk over the chosen mode's scale degrees (mostly stepwise,
    reflected at the tessitura edges), the final note resolves to the nearest tonic, and onsets are
    laid end-to-end (with occasional breath rests) so the result never overlaps.
    """
    rng = random.Random(seed)
    mode = rng.choice(list(_MODES))
    tonic_pc = rng.randrange(12)
    scale = _scale_pitches(tonic_pc, _MODES[mode], lo, hi)
    if len(scale) < 5:  # tessitura too narrow for this tonic — fall back to a chromatic walk
        scale = list(range(lo, hi + 1))

    n_notes = rng.randint(min_notes, max_notes)
    idx = rng.randrange(len(scale) // 4, max(len(scale) // 4 + 1, 3 * len(scale) // 4))
    onset = round(rng.uniform(0.1, 0.3), 3)
    notes: list[Note] = []
    for i in range(n_notes):
        if i == n_notes - 1:  # cadence: resolve to the nearest tonic-degree pitch
            tonics = [j for j, p in enumerate(scale) if (p - tonic_pc) % 12 == 0]
            if tonics:
                idx = min(tonics, key=lambda j: abs(j - idx))
        dur = rng.choice(_DURATIONS)
        notes.append(
            Note(onset_s=round(onset, 3), offset_s=round(onset + dur, 3), pitch_midi=scale[idx])
        )
        onset = round(onset + dur, 3)
        if rng.random() < 0.15:  # occasional breath rest
            onset = round(onset + rng.choice(_REST_S), 3)
        move = rng.choice(_STEPS) if rng.random() < 0.85 else rng.choice(_LEAPS)
        idx = max(0, min(len(scale) - 1, idx + move))

    return Score(score_id=score_id, source=f"generated:melody:{mode}", notes=notes)
