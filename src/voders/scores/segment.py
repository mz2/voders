"""Segment long real-singing scores into singable, renderable phrases (Klangio ingest, issue #9).

The Klangio challenge ships ~4-minute monophonic vocal annotations spanning several octaves — too
long for the SVS pipeline and too wide for any one voicebank. This splits a Score into phrases at
breath rests (and a hard length cap), octave-centres each phrase into the singable tessitura, drops
whatever is still out of range, and re-zeros onsets. Labels stay correct by construction: the phrase
Score *is* the label, and the renderer pitch-locks to it.
"""

from __future__ import annotations

from voders.scores.models import Note, Score


def segment_score(
    score: Score,
    *,
    gap_s: float = 0.4,
    max_phrase_s: float = 12.0,
    min_notes: int = 4,
    min_phrase_s: float = 1.0,
    lo: int = 55,
    hi: int = 79,
    max_out_of_range_frac: float = 0.2,
) -> list[Score]:
    """Split ``score`` into octave-centred singable phrase Scores (empty list if none qualify)."""
    notes = sorted(score.notes, key=lambda n: n.onset_s)
    if not notes:
        return []

    # 1. Group into phrases: break at a rest >= gap_s, or when the phrase would exceed max_phrase_s.
    groups: list[list[Note]] = []
    cur: list[Note] = [notes[0]]
    for prev, n in zip(notes, notes[1:], strict=False):
        rest = n.onset_s - prev.offset_s
        span = n.offset_s - cur[0].onset_s
        if rest >= gap_s or span > max_phrase_s:
            groups.append(cur)
            cur = [n]
        else:
            cur.append(n)
    groups.append(cur)

    phrases: list[Score] = []
    pi = 0
    for g in groups:
        if len(g) < min_notes or (g[-1].offset_s - g[0].onset_s) < min_phrase_s:
            continue
        centred = _octave_centre(g, lo, hi, max_out_of_range_frac)
        if centred is None:
            continue
        t0 = centred[0].onset_s - 0.2
        notes_out = [
            Note(onset_s=round(n.onset_s - t0, 3), offset_s=round(n.offset_s - t0, 3),
                 pitch_midi=n.pitch_midi)
            for n in centred
        ]
        phrases.append(
            Score(score_id=f"{score.score_id}_p{pi:03d}", source=score.source, notes=notes_out)
        )
        pi += 1
    return phrases


def _octave_centre(group: list[Note], lo: int, hi: int, max_out_frac: float) -> list[Note] | None:
    """Shift a phrase by whole octaves to centre it in [lo, hi]; drop out-of-range notes.

    Returns ``None`` if more than ``max_out_frac`` of notes still fall out of range after centring
    (the phrase is too wide to be singable in one tessitura).
    """
    pitches = sorted(n.pitch_midi for n in group)
    median = pitches[len(pitches) // 2]
    shift = 12 * round(((lo + hi) / 2 - median) / 12)
    kept = [
        Note(onset_s=n.onset_s, offset_s=n.offset_s, pitch_midi=n.pitch_midi + shift)
        for n in group
        if lo <= n.pitch_midi + shift <= hi
    ]
    if not kept or (len(group) - len(kept)) / len(group) > max_out_frac:
        return None
    return kept
