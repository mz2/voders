"""Re-derive note onsets for legato/generative singing by aligning the score to the measured f0.

Energy-based onset detection fails on legato singing (SoulX): there are no gaps between notes, so
the RMS envelope is one continuous run. But a singer following a score moves through the score's
*pitches* in order, so the f0 carries the note boundaries. This DTW-aligns the score's pitch
sequence to the per-frame f0 (what the validator measures), reads off each note's onset where its
pitch begins, and returns a re-derived Score (score pitches kept, timing from the audio). The result
is self-consistent with the audio — exactly what option 1 needs for generative SVS.
"""

from __future__ import annotations

import numpy as np

from voders.constants import SAMPLE_RATE
from voders.scores.models import Note, Score

_UNVOICED_COST = 1e3  # cost of assigning a note onset to an unvoiced / wrong-pitch frame


def align_score_to_f0(
    audio: np.ndarray, score: Score, *, sr: int = SAMPLE_RATE, device: str = "auto"
) -> tuple[Score, float]:
    """Return ``(rederived_score, max_onset_dev_ms)`` aligning ``score`` to the audio's f0.

    Pitches are the score's (the backend sang them); onsets/offsets are detected from the f0 by a
    monotonic DTW so consecutive same-or-different-pitch notes are separated by their pitch contour.
    """
    from voders.validate.validator import _measure_f0

    notes = score.notes
    if not notes or audio.size == 0:
        return score, 0.0

    meas = _measure_f0(audio, sr, 0.0, device=device)
    times = np.asarray(meas.times, dtype=np.float64)
    f0 = np.asarray(meas.f0, dtype=np.float64)
    m = times.size
    n = len(notes)
    if m < n:  # too few frames to resolve every note — keep the score as-is
        return score, 0.0

    frame_midi = np.full(m, np.nan)
    voiced = np.isfinite(f0) & (f0 > 0)
    frame_midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)

    # Per-(note, frame) cost: absolute pitch distance, large where unvoiced.
    pitches = np.array([note.pitch_midi for note in notes], dtype=np.float64)
    diff = np.abs(frame_midi[None, :] - pitches[:, None])
    cost = np.where(np.isfinite(diff), np.minimum(diff, _UNVOICED_COST), _UNVOICED_COST)

    # Monotonic DTW: each frame belongs to a note; advance to the next note or stay (legato).
    big = np.inf
    acc = np.full((n, m), big)
    back = np.zeros((n, m), dtype=np.int8)  # 1 = advanced from note i-1, 0 = stayed on note i
    acc[0] = np.cumsum(cost[0])
    for i in range(1, n):
        for j in range(i, m):
            stay = acc[i, j - 1]
            adv = acc[i - 1, j - 1]
            if adv <= stay:
                acc[i, j] = cost[i, j] + adv
                back[i, j] = 1
            else:
                acc[i, j] = cost[i, j] + stay
    # Backtrack the frame→note assignment.
    assign = np.zeros(m, dtype=int)
    i, j = n - 1, m - 1
    while j >= 0:
        assign[j] = i
        if i > 0 and back[i, j] == 1:
            i -= 1
        j -= 1

    onset_frame: list[int | None] = [None] * n
    for jj, a in enumerate(assign):
        if onset_frame[a] is None:
            onset_frame[a] = jj
    # Fill any note that got no frame by interpolating between its neighbours.
    for i in range(n):
        if onset_frame[i] is None:
            onset_frame[i] = onset_frame[i - 1] if i > 0 else 0

    rederived: list[Note] = []
    max_dev_ms = 0.0
    for i, note in enumerate(notes):
        onset = float(times[onset_frame[i]])  # type: ignore[index]
        if i + 1 < n:
            offset = float(times[onset_frame[i + 1]])  # type: ignore[index]
        else:
            offset = float(times[-1]) + (times[1] - times[0] if m > 1 else 0.01)
        if offset <= onset:  # guard degenerate spans
            offset = onset + max(0.02, note.duration_s * 0.5)
        rederived.append(Note(onset_s=round(onset, 3), offset_s=round(offset, 3),
                              pitch_midi=note.pitch_midi))
        max_dev_ms = max(max_dev_ms, abs(onset - note.onset_s) * 1000.0)

    return Score(score_id=score.score_id, source=score.source, notes=rederived), max_dev_ms
