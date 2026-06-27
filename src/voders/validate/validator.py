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

_CREPE_SR = 16_000  # CREPE operates at 16 kHz internally
_CREPE_HOP = 160  # 10 ms at 16 kHz — matches the crepe_f0 timing budget (FR-019)
_CREPE_PERIODICITY_THRESH = 0.30  # voiced when CREPE periodicity exceeds this


@dataclass
class _Measurement:
    times: np.ndarray  # frame centre times (s), group-delay compensated
    f0: np.ndarray  # Hz, nan where unvoiced
    voiced: np.ndarray  # bool


def _measure_pyin(audio: np.ndarray, sr: int, group_delay_ms: float) -> _Measurement:
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


def _select_torch_device(pref: str = "auto"):  # noqa: ANN202 - returns str | torch.device
    """Resolve a torch device for the f0 estimator.

    An explicit preference (``cuda`` / ``xpu`` / ``dml`` / ``cpu``) is returned as-is. ``auto``
    probes, in order: NVIDIA CUDA, Intel XPU (Arc / recent iGPUs via intel-extension-for-pytorch),
    DirectML (Intel/AMD iGPUs on Windows via torch-directml), then CPU. Only ``cuda`` and ``cpu``
    are verified here; ``xpu`` / ``dml`` are wired to their documented APIs.
    """
    if pref and pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    try:
        import torch_directml

        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return "cpu"


def _measure_crepe(
    audio: np.ndarray, sr: int, group_delay_ms: float, device: str = "auto", model: str = "full"
) -> _Measurement:
    """Measure f0 with CREPE (a neural pitch estimator) on an accelerator when available.

    torchcrepe is imported lazily so the CPU baseline never pulls in torch at module load (FR-009).
    Audio is resampled to 16 kHz (CREPE's native rate); frames are 10 ms apart, matching the
    ``crepe_f0`` timing budget. ``model`` selects the CREPE capacity: ``full`` (most accurate) or
    ``tiny`` (~5-10x faster, slightly less accurate — adequate for a gating check).
    """
    import torch
    import torchcrepe

    device = _select_torch_device(device)

    x = audio.astype(np.float32)
    if sr != _CREPE_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=_CREPE_SR)
    tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    f0_t, periodicity = torchcrepe.predict(
        tensor,
        _CREPE_SR,
        hop_length=_CREPE_HOP,
        fmin=float(midi_to_hz(_FMIN_MIDI)),
        fmax=float(midi_to_hz(_FMAX_MIDI)),
        model=model,
        return_periodicity=True,
        device=device,
        pad=True,
    )
    f0 = f0_t.squeeze(0).cpu().numpy().astype(float)
    per = periodicity.squeeze(0).cpu().numpy().astype(float)
    voiced = per > _CREPE_PERIODICITY_THRESH
    f0 = np.where(voiced, f0, np.nan)
    times = np.arange(f0.size) * (_CREPE_HOP / _CREPE_SR) - group_delay_ms / 1000.0
    return _Measurement(times=times, f0=f0, voiced=voiced)


def _measure_f0(
    audio: np.ndarray,
    sr: int,
    group_delay_ms: float,
    method: str = "pyin_f0",
    device: str = "auto",
    model: str = "full",
) -> _Measurement:
    """Dispatch f0 measurement. ``pyin_f0`` (CPU) is the default; ``crepe_f0`` is the GPU upgrade.

    ``auto`` uses CREPE when torch + torchcrepe import, else falls back to pyin. ``crepe_f0`` is
    explicit and raises if torchcrepe is unavailable (no silent downgrade). ``model`` selects the
    CREPE capacity (``full`` | ``tiny``) and is ignored by pyin.
    """
    if method == "pyin_f0":
        return _measure_pyin(audio, sr, group_delay_ms)
    if method == "crepe_f0":
        return _measure_crepe(audio, sr, group_delay_ms, device, model)
    if method == "auto":
        try:
            return _measure_crepe(audio, sr, group_delay_ms, device, model)
        except ImportError:
            return _measure_pyin(audio, sr, group_delay_ms)
    raise ValueError(f"unknown f0 method {method!r}; expected pyin_f0 | crepe_f0 | auto")


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
        f0_method: str | None = None,
        f0_device: str | None = None,
    ) -> ValidationVerdict:
        reasons: list[str] = []
        if label_score.is_empty:
            return ValidationVerdict(status=VerdictStatus.REJECTED, reason="empty score (no notes)")

        method = f0_method or self.cfg.f0_method
        device = f0_device or self.cfg.f0_device
        model = self.cfg.f0_model

        # Clipping gate (US3 scenario 3): never silently distort into the corpus.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        clipping = peak > 1.0 + 1e-6

        group_delay = self.timing.total_active_group_delay_ms()
        meas = _measure_f0(
            audio.astype(float), self.sr, group_delay, method=method, device=device, model=model
        )

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
