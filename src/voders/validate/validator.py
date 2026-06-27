"""Alignment validator — the single gate every sample passes (FR-006, FR-014, FR-019).

Measures the rendered audio's pitch (``pyin`` on CPU; ``torchcrepe`` is the optional GPU upgrade)
and verifies, per note, that f0 and the note boundaries match the paired score within tolerance.
Method group delay is compensated before comparison (FR-019). Also gates SNR (FR-014) and clipping.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from voders.config.models import ValidatorConfig
from voders.manifest.models import ValidationVerdict, VerdictStatus
from voders.render.deterministic import midi_to_hz
from voders.scores.models import Score
from voders.validate.timing import TimingRegistry

_HOP = 512
_FRAME = 2048
_FMIN_MIDI = 36
_FMAX_MIDI = 96


@dataclass
class _Measurement:
    times: np.ndarray  # frame centre times (s), group-delay compensated
    f0: np.ndarray  # Hz, nan where unvoiced
    voiced: np.ndarray  # bool


def _measure_f0(audio: np.ndarray, sr: int, group_delay_ms: float) -> _Measurement:
    f0, voiced, _ = librosa.pyin(
        audio,
        sr=sr,
        fmin=float(midi_to_hz(_FMIN_MIDI)),
        fmax=float(midi_to_hz(_FMAX_MIDI)),
        frame_length=_FRAME,
        hop_length=_HOP,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=_HOP) - group_delay_ms / 1000.0
    return _Measurement(times=times, f0=np.asarray(f0), voiced=np.asarray(voiced, dtype=bool))


def _cents(f_meas: np.ndarray, f_ref: float) -> np.ndarray:
    return 1200.0 * np.log2(np.where(f_meas > 0, f_meas, np.nan) / f_ref)


class Validator:
    """Validates a rendered sample against its paired score (FR-006)."""

    def __init__(
        self,
        config: ValidatorConfig,
        timing: TimingRegistry | None = None,
        min_note_ms: float = 50.0,
        sr: int = 22_050,
    ) -> None:
        self.cfg = config
        self.timing = timing or TimingRegistry()
        self.min_note_ms = min_note_ms
        self.sr = sr

    def _offset_tol_ms(self, note_dur_ms: float) -> float:
        return max(self.cfg.offset_min_ms, self.cfg.offset_fraction * note_dur_ms)

    def validate(
        self,
        audio: np.ndarray,
        label_score: Score,
        *,
        snr_db: float | None = None,
        f0_method: str = "pyin_f0",
    ) -> ValidationVerdict:
        reasons: list[str] = []
        if label_score.is_empty:
            return ValidationVerdict(status=VerdictStatus.REJECTED, reason="empty score (no notes)")

        # Clipping gate (US3 scenario 3): never silently distort into the corpus.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        clipping = peak > 1.0 + 1e-6

        group_delay = self.timing.total_active_group_delay_ms()
        meas = _measure_f0(audio.astype(float), self.sr, group_delay)

        onset_ok = True
        offset_ok = True
        f0_ok = True
        max_onset_dev = 0.0
        short_notes: list[int] = []

        win = max(0.15, 3.0 * self.cfg.onset_ms / 1000.0)
        for idx, note in enumerate(label_score.notes):
            if note.duration_ms + 1e-6 < self.min_note_ms:
                short_notes.append(idx)

            f_ref = midi_to_hz(note.pitch_midi)
            cents = _cents(meas.f0, f_ref)
            in_tune = meas.voiced & np.isfinite(cents) & (np.abs(cents) <= self.cfg.f0_cents)

            # Onset/offset: first/last in-tune frame within a search window around the score time.
            on_lo, on_hi = note.onset_s - win, note.onset_s + win
            off_lo, off_hi = note.offset_s - win, note.offset_s + win
            on_idx = np.where(in_tune & (meas.times >= on_lo) & (meas.times <= on_hi))[0]
            off_idx = np.where(in_tune & (meas.times >= off_lo) & (meas.times <= off_hi))[0]

            if on_idx.size == 0:
                onset_ok = False
                reasons.append(f"note {idx}: no in-tune onset detected")
            else:
                dev = abs(float(meas.times[on_idx[0]]) - note.onset_s) * 1000.0
                max_onset_dev = max(max_onset_dev, dev)
                if dev > self.cfg.onset_ms:
                    onset_ok = False
                    reasons.append(
                        f"note {idx}: onset off by {dev:.0f} ms (> {self.cfg.onset_ms:.0f} ms)"
                    )

            if off_idx.size == 0:
                offset_ok = False
                reasons.append(f"note {idx}: no in-tune offset detected")
            else:
                dev = abs(float(meas.times[off_idx[-1]]) - note.offset_s) * 1000.0
                tol = self._offset_tol_ms(note.duration_ms)
                if dev > tol:
                    offset_ok = False
                    reasons.append(f"note {idx}: offset off by {dev:.0f} ms (> {tol:.0f} ms)")

            # f0 coverage over the central (sustained) portion of the note.
            pad = 0.1 * note.duration_s
            sus = (meas.times >= note.onset_s + pad) & (meas.times <= note.offset_s - pad)
            n_sus = int(np.count_nonzero(sus))
            if n_sus == 0:
                sus = (meas.times >= note.onset_s) & (meas.times <= note.offset_s)
                n_sus = int(np.count_nonzero(sus))
            covered = int(np.count_nonzero(in_tune & sus))
            coverage = (covered / n_sus) if n_sus else 0.0
            if n_sus == 0 or coverage < self.cfg.f0_coverage:
                f0_ok = False
                reasons.append(
                    f"note {idx}: f0 in tune for {coverage:.0%} of the note "
                    f"(< {self.cfg.f0_coverage:.0%})"
                )

        snr_fail = snr_db is not None and snr_db < self.cfg.snr_floor_db

        status = self._status(
            onset_ok=onset_ok,
            offset_ok=offset_ok,
            f0_ok=f0_ok,
            clipping=clipping,
            snr_fail=snr_fail,
            short_notes=short_notes,
        )
        if clipping:
            reasons.append(f"clipping: peak={peak:.3f} > 1.0")
        if snr_fail:
            reasons.append(f"snr {snr_db:.1f} dB below floor {self.cfg.snr_floor_db} dB")
        if short_notes and status != VerdictStatus.REJECTED:
            reasons.append(f"notes shorter than min_note_ms={self.min_note_ms:.1f}: {short_notes}")

        return ValidationVerdict(
            status=status,
            onset_ok=onset_ok,
            offset_ok=offset_ok,
            f0_ok=f0_ok,
            snr_db=snr_db,
            reason="; ".join(reasons) or None,
            max_onset_dev_ms=max_onset_dev,
        )

    def _status(
        self,
        *,
        onset_ok: bool,
        offset_ok: bool,
        f0_ok: bool,
        clipping: bool,
        snr_fail: bool,
        short_notes: list[int],
    ) -> VerdictStatus:
        if snr_fail:
            return VerdictStatus.QUARANTINED
        if clipping or not (onset_ok and offset_ok and f0_ok):
            return VerdictStatus.REJECTED
        if short_notes:
            return VerdictStatus.FLAGGED
        return VerdictStatus.ACCEPTED
