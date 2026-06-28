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
_TIMING_WEIGHT = 6.0  # pitch-cost units (≈ semitones) per second a frame sits outside a note's span


def align_score_to_f0(
    audio: np.ndarray, score: Score, *, sr: int = SAMPLE_RATE, device: str = "auto"
) -> tuple[Score, float, float]:
    """Return ``(rederived_score, max_onset_dev_ms, max_offset_dev_ms)`` aligning ``score`` to the
    audio's f0.

    Pitches are the score's (the backend sang them); onsets/offsets are detected from the f0 by a
    monotonic DTW so consecutive same-or-different-pitch notes are separated by their pitch contour.
    The two deviations report how far the *sung* onset/offset drifted from the source score (the
    re-derived labels match the audio by construction, so these are provenance, not a gate).
    """
    from voders.validate.validator import _measure_f0

    notes = score.notes
    if not notes or audio.size == 0:
        return score, 0.0, 0.0

    meas = _measure_f0(audio, sr, 0.0, device=device)
    times = np.asarray(meas.times, dtype=np.float64)
    f0 = np.asarray(meas.f0, dtype=np.float64)
    m = times.size
    n = len(notes)
    if m < n:  # too few frames to resolve every note — keep the score as-is
        return score, 0.0, 0.0

    frame_midi = np.full(m, np.nan)
    voiced = np.isfinite(f0) & (f0 > 0)
    frame_midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)

    # Per-(note, frame) pitch cost: absolute pitch distance, large where unvoiced.
    pitches = np.array([note.pitch_midi for note in notes], dtype=np.float64)
    diff = np.abs(frame_midi[None, :] - pitches[:, None])
    pitch_cost = np.where(np.isfinite(diff), np.minimum(diff, _UNVOICED_COST), _UNVOICED_COST)

    # Soft timing anchor: penalise assigning a frame to a note far OUTSIDE that note's nominal score
    # span (zero inside it). Pure pitch cost cannot separate consecutive SAME-pitch notes — the
    # monotonic DTW collapses the whole same-pitch run into one note and starves the rest (two p72
    # notes in a row: the first swallows both notes' frames, the second is left with ~0, so its
    # re-derived onset/offset land on the wrong frames and validation fails "in tune 0%"). The score
    # shares the audio's timeline, so anchoring each note to its nominal span splits same-pitch
    # neighbours at the score boundary and bounds drift. The weight is in pitch-cost units (≈
    # semitones) per second of out-of-span distance, gentle enough that genuine micro-timing in
    # voiced, on-pitch frames can still move a boundary by a frame or two.
    onsets = np.array([note.onset_s for note in notes], dtype=np.float64)
    offsets = np.array([note.offset_s for note in notes], dtype=np.float64)
    span_dist = np.maximum.reduce(
        [
            np.zeros((n, m)),
            onsets[:, None] - times[None, :],
            times[None, :] - offsets[:, None],
        ]
    )
    cost = pitch_cost + _TIMING_WEIGHT * span_dist

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

    # Each note's onset is the first frame the DTW assigns to it that is actually VOICED and on its
    # pitch (within a semitone) — i.e. where the note's pitch begins, matching the validator's
    # rising-edge onset. Falling back to the first assigned frame (or a neighbour) only if none.
    note_frames: list[list[int]] = [[] for _ in range(n)]
    for jj, a in enumerate(assign):
        note_frames[a].append(jj)
    onset_frame: list[int | None] = [None] * n
    offset_frame: list[int | None] = [None] * n
    for i in range(n):
        on_pitch = [
            j for j in note_frames[i] if voiced[j] and abs(frame_midi[j] - pitches[i]) <= 1.0
        ]
        if on_pitch:
            onset_frame[i] = on_pitch[0]
            # Last voiced on-pitch frame = where THIS note's voicing actually ends.
            offset_frame[i] = on_pitch[-1]
        elif note_frames[i]:
            onset_frame[i] = note_frames[i][0]
            offset_frame[i] = note_frames[i][-1]
    for i in range(n):
        if onset_frame[i] is None:
            onset_frame[i] = onset_frame[i - 1] if i > 0 else 0

    frame_dur = float(times[1] - times[0]) if m > 1 else 0.01

    rederived: list[Note] = []
    max_dev_ms = 0.0
    max_offset_dev_ms = 0.0
    for i, note in enumerate(notes):
        onset = float(times[onset_frame[i]])  # type: ignore[index]
        # Offset = the end of THIS note's own voicing (one frame past its last voiced on-pitch
        # frame), NOT the next note's onset. Equating offset with the next onset (the previous
        # behaviour) forced every note fully contiguous: rests, breaths and staccato gaps were
        # absorbed into the preceding note, so the audio went silent while the label stayed
        # "active" — training the model to hold notes on past their true end (systematically late
        # offsets, hurting COnOff/COnPOff).
        if offset_frame[i] is not None:
            offset = float(times[offset_frame[i]]) + frame_dur  # type: ignore[index]
        else:
            offset = onset + max(0.02, note.duration_s * 0.5)
        # Never run into the next note's onset (keep a valid monophonic, non-overlapping label).
        if i + 1 < n and onset_frame[i + 1] is not None:
            next_onset = float(times[onset_frame[i + 1]])  # type: ignore[index]
            if next_onset > onset:
                offset = min(offset, next_onset)
        if offset <= onset:  # guard degenerate spans
            offset = onset + max(0.02, note.duration_s * 0.5)
        rederived.append(
            Note(onset_s=round(onset, 3), offset_s=round(offset, 3), pitch_midi=note.pitch_midi)
        )
        max_dev_ms = max(max_dev_ms, abs(onset - note.onset_s) * 1000.0)
        max_offset_dev_ms = max(max_offset_dev_ms, abs(offset - note.offset_s) * 1000.0)

    return (
        Score(score_id=score.score_id, source=score.source, notes=rederived),
        max_dev_ms,
        max_offset_dev_ms,
    )


def relabel_score(
    audio: np.ndarray,
    sr: int,
    score: Score,
    *,
    device: str = "auto",
    validator=None,
):
    """Re-derive a score's onset/offset labels from its OWN audio and validate the result.

    Returns ``(rederived_score, verdict, onset_drift_ms, offset_drift_ms)``. ``verdict`` is the
    validator's check of the re-derived label against the audio; callers keep the re-derived label
    only when ``verdict.status`` is ACCEPTED and otherwise fall back to the original (so a sample is
    never lost or corrupted by a poor alignment). This is the shared core of the offline relabel CLI
    and the staging-time relabel in ``just train``.
    """
    from voders.config.models import ValidatorConfig
    from voders.validate.validator import Validator

    rederived, onset_drift_ms, offset_drift_ms = align_score_to_f0(
        audio, score, sr=sr, device=device
    )
    validator = validator or Validator(ValidatorConfig(), sr=sr)
    verdict = validator.validate(audio, rederived, f0_device=device)
    return rederived, verdict, onset_drift_ms, offset_drift_ms
