"""Deterministic WORLD f0-driven renderer (FR-003, FR-009; research Decision 1).

The load-bearing baseline. A donor vowel is decomposed by the WORLD vocoder into pitch (f0),
spectral envelope, and aperiodicity; the f0 is replaced by a contour constructed directly from the
score and the signal is resynthesised. Because pitch, onset, and offset come from the score, they
are correct *by construction* — this lane cannot drift. CPU-only: no GPU/torch import at module
load (FR-009).
"""

from __future__ import annotations

import functools

import numpy as np
import pyworld as pw

from voders.audio import read_wav, to_mono_float32
from voders.constants import SAMPLE_RATE
from voders.render.base import RenderRequest, RenderResult
from voders.scores.models import Note

_FRAME_PERIOD_MS = 5.0
_ARTICULATION_DIP_MS = 18.0  # amplitude dip between same-pitch legato notes (edge case)

# Realism, kept inside the alignment budget. Vibrato depth stays under the validator's ±25-cent
# tolerance, and the attack/release are short relative to the onset/offset tolerances, so the audio
# is livelier without the labels ever drifting (FR-003). All deterministic → SC-009 stays bit-exact.
_VIBRATO_RATE_HZ = 5.5
_VIBRATO_CENTS = 18.0  # peak deviation; < the 25-cent f0 tolerance
_VIBRATO_ONSET_S = 0.12  # vibrato fades in after the note's attack (natural)
# Kept short so the re-derive lane's energy-based onset detector and the validator's f0-based one
# agree to well within the 50 ms onset tolerance (a longer attack drifts the detected onset).
_ATTACK_S = 0.012
_RELEASE_S = 0.030


def midi_to_hz(pitch_midi: int) -> float:
    """Equal-tempered MIDI pitch to frequency in Hz (A4 = MIDI 69 = 440 Hz)."""
    return 440.0 * (2.0 ** ((pitch_midi - 69) / 12.0))


@functools.lru_cache(maxsize=16)
def _donor_timbre(model_ref: str) -> tuple[np.ndarray, np.ndarray]:
    """The donor's full voiced (spectral envelope, aperiodicity) *trajectories*.

    Cached per donor path. Keeping the whole voiced sequence (not a single median frame) preserves
    the donor's natural frame-to-frame spectral motion, which is what makes the render sound alive
    rather than a flat, buzzy held vowel.
    """
    audio, sr = read_wav(model_ref)
    x = np.ascontiguousarray(audio.astype(np.float64))
    if sr != SAMPLE_RATE:  # pragma: no cover - fixtures are already 22,050 Hz
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float64)
    _f0, t = pw.harvest(x, SAMPLE_RATE, frame_period=_FRAME_PERIOD_MS)
    f0 = pw.stonemask(x, _f0, t, SAMPLE_RATE)
    sp = pw.cheaptrick(x, f0, t, SAMPLE_RATE)
    ap = pw.d4c(x, f0, t, SAMPLE_RATE)
    voiced = f0 > 0
    if not np.any(voiced):  # pragma: no cover - donor fixtures are voiced
        raise ValueError(f"donor {model_ref!r} has no voiced frames; cannot extract timbre")
    sp_v = sp[voiced]
    ap_v = ap[voiced]
    # Restrict to the donor's loud, stable core: drop low-energy edge frames (a donor's quiet
    # attack/decay) so every rendered note starts at full energy and its onset is detected crisply
    # even after augmentation masks it. The remaining frames still carry real spectral motion.
    energy = sp_v.sum(axis=1)
    keep = energy >= 0.5 * float(energy.max())
    if np.count_nonzero(keep) >= 2:
        sp_v, ap_v = sp_v[keep], ap_v[keep]
    return np.ascontiguousarray(sp_v), np.ascontiguousarray(ap_v)


def _pingpong(n: int, period: int) -> np.ndarray:
    """Indices sweeping 0→period-1→0… so the donor trajectory loops without a seam discontinuity."""
    if period <= 1:
        return np.zeros(n, dtype=int)
    cycle = np.concatenate([np.arange(period), np.arange(period - 2, 0, -1)])
    return cycle[np.arange(n) % cycle.size]


def _amp_envelope(n: int) -> np.ndarray:
    """A singer-like amplitude envelope: soft attack, gentle mid-note swell, soft release."""
    env = np.ones(n, dtype=np.float32)
    att = min(int(_ATTACK_S * SAMPLE_RATE), n // 2)
    rel = min(int(_RELEASE_S * SAMPLE_RATE), n // 2)
    if att > 0:
        env[:att] = np.linspace(0.0, 1.0, att, dtype=np.float32) ** 1.5
    if rel > 0:
        env[-rel:] = np.linspace(1.0, 0.0, rel, dtype=np.float32) ** 1.5
    swell = 0.92 + 0.08 * np.sin(np.pi * np.linspace(0.0, 1.0, n, dtype=np.float32))
    return env * swell


def _render_note(note: Note, sp_seq: np.ndarray, ap_seq: np.ndarray) -> np.ndarray:
    """Synthesise one note: score pitch (+ light vibrato), donor timbre trajectory, envelope."""
    n_samples = max(1, int(round(note.duration_s * SAMPLE_RATE)))
    n_frames = max(1, int(round(note.duration_ms / _FRAME_PERIOD_MS)))
    base = midi_to_hz(note.pitch_midi)

    tf = np.arange(n_frames) * (_FRAME_PERIOD_MS / 1000.0)
    depth = 2.0 ** (_VIBRATO_CENTS / 1200.0) - 1.0
    onset_gain = np.clip(tf / _VIBRATO_ONSET_S, 0.0, 1.0)
    f0 = base * (1.0 + depth * onset_gain * np.sin(2.0 * np.pi * _VIBRATO_RATE_HZ * tf))

    idx = _pingpong(n_frames, sp_seq.shape[0])
    sp = np.ascontiguousarray(sp_seq[idx], dtype=np.float64)
    ap = np.ascontiguousarray(ap_seq[idx], dtype=np.float64)
    y = pw.synthesize(np.ascontiguousarray(f0), sp, ap, SAMPLE_RATE, frame_period=_FRAME_PERIOD_MS)
    y = np.asarray(y, dtype=np.float32)
    y = y[:n_samples] if y.size >= n_samples else np.pad(y, (0, n_samples - y.size))
    return y * _amp_envelope(n_samples)


class DeterministicLane:
    """Score-f0 → WORLD resynthesis. ``label_score == input`` by construction (FR-003)."""

    name = "deterministic"

    def requires_gpu(self) -> bool:
        return False

    def render(self, req: RenderRequest) -> RenderResult:
        score = req.score
        if score.is_empty:
            return RenderResult(
                audio=np.zeros(0, dtype=np.float32), label_score=score, notes={"empty": True}
            )

        sp_seq, ap_seq = _donor_timbre(req.voice.model_ref)
        total_samples = int(round(score.duration_s * SAMPLE_RATE)) + 1
        out = np.zeros(total_samples, dtype=np.float32)

        dip_n = int(_ARTICULATION_DIP_MS * SAMPLE_RATE / 1000.0)
        dynamics_applied = False
        for i, note in enumerate(score.notes):
            y = _render_note(note, sp_seq, ap_seq)
            # Honour the optional per-note gain from the score-augmentation volume axis (FR-008).
            # The trailing peak-normalise rescales the whole signal, so the relative per-note level
            # differences survive — widening the dynamics without touching the labels (SC-004).
            if note.gain is not None:
                y = (y * np.float32(note.gain)).astype(np.float32)
                dynamics_applied = True
            start = int(round(note.onset_s * SAMPLE_RATE))
            end = min(start + y.size, out.size)
            out[start:end] += y[: end - start]
            # Articulate same-pitch legato pairs with a brief amplitude dip (edge case).
            prev = score.notes[i - 1] if i > 0 else None
            if (
                prev is not None
                and prev.pitch_midi == note.pitch_midi
                and abs(prev.offset_s - note.onset_s) < 1e-3
            ):
                d0 = max(0, start - dip_n // 2)
                d1 = min(out.size, start + dip_n // 2)
                out[d0:d1] *= np.linspace(0.3, 1.0, d1 - d0, dtype=np.float32)

        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0:
            out = (out * (0.9 / peak)).astype(np.float32)
        notes: dict[str, object] = {"dynamics_applied": True} if dynamics_applied else {}
        # Optional neural-vocoder enhancement (alignment-safe: mel preserves timing/pitch). Lazily
        # imported so the CPU baseline never loads torch (FR-009).
        if str(req.options.get("vocoder", "world")) == "vocos":
            from voders.render.vocos_enhance import enhance

            out = enhance(out, SAMPLE_RATE, device=str(req.options.get("device", "auto")))
            notes["vocoder"] = "vocos"
        return RenderResult(audio=to_mono_float32(out), label_score=score, notes=notes)
