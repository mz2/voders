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
_FADE_MS = 5.0  # click-avoidance fade; well within the offset tolerance
_ARTICULATION_DIP_MS = 18.0  # amplitude dip between same-pitch legato notes (edge case)


def midi_to_hz(pitch_midi: int) -> float:
    """Equal-tempered MIDI pitch to frequency in Hz (A4 = MIDI 69 = 440 Hz)."""
    return 440.0 * (2.0 ** ((pitch_midi - 69) / 12.0))


@functools.lru_cache(maxsize=16)
def _donor_timbre(model_ref: str) -> tuple[np.ndarray, np.ndarray]:
    """Representative (spectral envelope, aperiodicity) frame for a donor vowel.

    Cached per donor path. The median over voiced frames gives a pitch-independent vowel timbre.
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
    sp_rep = np.median(sp[voiced], axis=0)
    ap_rep = np.median(ap[voiced], axis=0)
    return sp_rep, ap_rep


def _fade(audio: np.ndarray, n_fade: int) -> np.ndarray:
    if n_fade <= 0 or audio.size < 2 * n_fade:
        return audio
    ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    audio[:n_fade] *= ramp
    audio[-n_fade:] *= ramp[::-1]
    return audio


def _render_note(note: Note, sp_rep: np.ndarray, ap_rep: np.ndarray) -> np.ndarray:
    """Synthesise one note as a sustained tone at its exact pitch with the donor timbre."""
    n_samples = max(1, int(round(note.duration_s * SAMPLE_RATE)))
    n_frames = max(1, int(round(note.duration_ms / _FRAME_PERIOD_MS)))
    f0 = np.full(n_frames, midi_to_hz(note.pitch_midi), dtype=np.float64)
    sp = np.tile(sp_rep, (n_frames, 1)).astype(np.float64)
    ap = np.tile(ap_rep, (n_frames, 1)).astype(np.float64)
    y = pw.synthesize(f0, sp, ap, SAMPLE_RATE, frame_period=_FRAME_PERIOD_MS)
    y = np.asarray(y, dtype=np.float32)
    y = y[:n_samples] if y.size >= n_samples else np.pad(y, (0, n_samples - y.size))
    return _fade(y, int(_FADE_MS * SAMPLE_RATE / 1000.0))


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

        sp_rep, ap_rep = _donor_timbre(req.voice.model_ref)
        total_samples = int(round(score.duration_s * SAMPLE_RATE)) + 1
        out = np.zeros(total_samples, dtype=np.float32)

        dip_n = int(_ARTICULATION_DIP_MS * SAMPLE_RATE / 1000.0)
        for i, note in enumerate(score.notes):
            y = _render_note(note, sp_rep, ap_rep)
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
        return RenderResult(audio=to_mono_float32(out), label_score=score, notes={})
