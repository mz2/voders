"""Label re-derivation for the expressive SVS lane (FR-007, FR-019).

The expressive singing-voice-synthesis ("SVS") lane humanizes note timing, so the rendered note
boundaries no longer match the input score by construction. The ``rederive_labels`` safety mode
therefore measures where each note *actually* lands in the rendered audio — a CPU stand-in for the
forced-alignment / onset-detection ("MFA") step a production pipeline would run — and carries that
re-derived score as the label.

The expressive renderer produces clean voiced segments separated by near-silence, so note
boundaries are recovered from a short-time root-mean-square (RMS) energy envelope: each voiced run
above an energy floor is one note. A method's constant group delay is subtracted before recording,
per FR-019; the analysis frame hop and group delay come from the method's documented timing budget
(``mfa_align``: 10 ms MFCC frames, no constant delay), not invented here.
"""

from __future__ import annotations

import numpy as np

from voders.scores.models import Note, Score
from voders.validate.timing import MethodTimingBudget, TimingRegistry

_METHOD = "mfa_align"  # documented timing budget: 10 ms frames, 0 ms group delay
_REL_FLOOR = 0.12  # voiced when frame RMS exceeds this fraction of the peak RMS
_MIN_SEG_FRAMES = 2  # ignore single-frame energy blips


def _rms_envelope(audio: np.ndarray, sr: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame RMS energy and (frame-start) times for ``audio`` at the given hop in samples."""
    frame = max(2 * hop, 1)
    n = audio.size
    if n == 0:
        return np.zeros(0), np.zeros(0)
    n_frames = 1 + max(0, -(-(n - frame) // hop))  # ceil of remaining frames
    x = audio.astype(np.float64)
    rms = np.empty(n_frames, dtype=np.float64)
    starts = np.empty(n_frames, dtype=np.float64)
    for k in range(n_frames):
        seg = x[k * hop : k * hop + frame]
        rms[k] = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
        starts[k] = (k * hop) / sr
    return rms, starts


def _voiced_runs(rms: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges (first, last inclusive) of consecutive frames above the energy floor."""
    if rms.size == 0:
        return []
    peak = float(rms.max())
    if peak <= 0.0:
        return []
    voiced = rms > _REL_FLOOR * peak
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for k, v in enumerate(voiced):
        if v and start is None:
            start = k
        elif not v and start is not None:
            runs.append((start, k - 1))
            start = None
    if start is not None:
        runs.append((start, voiced.size - 1))
    return [(a, b) for a, b in runs if b - a + 1 >= _MIN_SEG_FRAMES]


def _activate_method(timing: TimingRegistry) -> None:
    try:
        timing.activate(_METHOD)
    except KeyError:  # pragma: no cover - default budgets always register mfa_align
        timing.register(MethodTimingBudget(_METHOD, frame_hop_ms=10.0, group_delay_ms=0.0))
        timing.activate(_METHOD)


def derive_labels(
    audio: np.ndarray,
    score: Score,
    sr: int = 22_050,
    timing: TimingRegistry | None = None,
) -> tuple[Score, float]:
    """Re-derive a label score from rendered audio (FR-007, FR-019).

    For each original note, the rendered voiced segment is located in the energy envelope and its
    onset/offset measured. The active methods' constant group delay is subtracted before recording
    (FR-019). Returns ``(rederived_score, max_onset_dev_ms)`` where ``max_onset_dev_ms`` is the
    largest ``|rederived_onset - original_onset|`` over the notes, in milliseconds.
    """
    timing = timing or TimingRegistry()
    _activate_method(timing)
    hop_ms = timing.get(_METHOD).frame_hop_ms
    group_delay_s = timing.total_active_group_delay_ms() / 1000.0
    hop = max(1, int(round(sr * hop_ms / 1000.0)))

    if score.is_empty or audio.size == 0:
        return score, 0.0

    rms, starts = _rms_envelope(audio, sr, hop)
    runs = _voiced_runs(rms)
    frame_s = (2 * hop) / sr

    def run_bounds(run: tuple[int, int]) -> tuple[float, float]:
        first, last = run
        onset = float(starts[first]) - group_delay_s
        offset = float(starts[last]) + frame_s - group_delay_s
        return onset, offset

    notes = score.notes
    rederived: list[Note] = []
    max_dev_ms = 0.0

    # When the segment count matches the note count, pair in onset order (the common, clean case);
    # otherwise fall back to the nearest segment by centre so a merged/missing segment degrades
    # gracefully rather than crashing.
    centres = [(float(starts[a]) + float(starts[b]) + frame_s) / 2.0 for a, b in runs]

    for idx, note in enumerate(notes):
        if not runs:
            rederived.append(note)
            continue
        if len(runs) == len(notes):
            run = runs[idx]
        else:
            note_centre = (note.onset_s + note.offset_s) / 2.0
            j = min(range(len(runs)), key=lambda r: abs(centres[r] - note_centre))
            run = runs[j]

        onset, offset = run_bounds(run)
        onset = max(0.0, onset)
        if offset <= onset:
            offset = onset + max(frame_s, note.duration_s)
        rederived.append(Note(onset_s=onset, offset_s=offset, pitch_midi=note.pitch_midi))
        max_dev_ms = max(max_dev_ms, abs(onset - note.onset_s) * 1000.0)

    rederived_score = Score(score_id=score.score_id, source=score.source, notes=rederived)
    return rederived_score, max_dev_ms
