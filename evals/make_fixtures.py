"""Generate the checked-in test fixtures (T004).

Run via uv::

    uv run python evals/make_fixtures.py

Produces:
  - evals/fixtures/voices/donor_ah_synth.wav   one synthetic, consented "ah" vowel (22,050 Hz)
  - evals/fixtures/scores/score_*.tsv          ~10 clean monophonic melodies (smoke score set)
  - evals/fixtures/edge/*.tsv                   edge-case scores (short note, polyphony, legato)

These are small, intentionally-versioned assets (Constitution: Binary Assets via LFS).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22_050
FIX = Path(__file__).parent / "fixtures"


def synth_ah_vowel(f0: float = 146.83, dur_s: float = 1.6) -> np.ndarray:
    """A sustained 'ah' vowel: harmonic glottal source shaped by three formant resonators."""
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    # Slight vibrato so WORLD sees a natural, clearly-voiced signal.
    vib = 1.0 + 0.005 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * f0 * np.cumsum(vib) / SR
    source = np.zeros(n)
    for k in range(1, 40):
        if k * f0 > SR / 2:
            break
        source += (1.0 / k) * np.sin(k * phase)
    # "ah" formants (Hz): F1 730, F2 1090, F3 2440.
    out = np.zeros(n)
    for fc, bw, gain in [(730, 90, 1.0), (1090, 110, 0.6), (2440, 160, 0.3)]:
        out += gain * _resonator(source, fc, bw)
    out *= np.hanning(n) ** 0.25  # gentle taper, keep the body sustained
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.9).astype(np.float32)


def _resonator(x: np.ndarray, fc: float, bw: float) -> np.ndarray:
    from scipy.signal import lfilter

    r = np.exp(-np.pi * bw / SR)
    theta = 2 * np.pi * fc / SR
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1.0 - r]
    return lfilter(b, a, x)


def _write_tsv(path: Path, rows: list[tuple[float, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{on:.3f}\t{off:.3f}\t{p}" for on, off, p in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_scores() -> None:
    scales = [
        [60, 62, 64, 65, 67],
        [67, 65, 64, 62, 60],
        [60, 64, 67, 64, 60],
        [69, 71, 72, 71, 69],
        [55, 57, 59, 60, 62],
        [72, 71, 69, 67, 65],
        [62, 64, 65, 67, 69],
        [64, 62, 60, 62, 64],
        [57, 60, 64, 60, 57],
        [65, 67, 69, 71, 72],
    ]
    for i, pitches in enumerate(scales):
        rows: list[tuple[float, float, int]] = []
        t = 0.2
        for p in pitches:
            dur = 0.45
            rows.append((round(t, 3), round(t + dur, 3), p))
            t += dur + 0.08
        _write_tsv(FIX / "scores" / f"score_{i:03d}.tsv", rows)


def make_edge_cases() -> None:
    # A note shorter than the learned/floor min_note_ms (US1 scenario 3).
    _write_tsv(
        FIX / "edge" / "short_note.tsv",
        [(0.2, 0.62, 60), (0.7, 0.72, 62), (0.8, 1.2, 64)],  # middle note ~20 ms
    )
    # Overlapping notes (polyphony) — must be refused (FR-001 edge case).
    _write_tsv(
        FIX / "edge" / "polyphony.tsv",
        [(0.2, 0.8, 60), (0.5, 1.1, 64)],
    )
    # Legato pair at the same pitch (edge case).
    _write_tsv(
        FIX / "edge" / "legato.tsv",
        [(0.2, 0.7, 62), (0.7, 1.2, 62), (1.25, 1.7, 64)],
    )


def synth_oo_vowel(f0: float = 110.0, dur_s: float = 1.6) -> np.ndarray:
    """A second, distinct timbre ('oo' vowel, lower formants) for voice-conversion fan-out (US2)."""
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * 6.0 * t)
    phase = 2 * np.pi * f0 * np.cumsum(vib) / SR
    source = np.zeros(n)
    for k in range(1, 50):
        if k * f0 > SR / 2:
            break
        source += (1.0 / k) * np.sin(k * phase)
    out = np.zeros(n)
    # "oo" formants (Hz): F1 300, F2 870, F3 2240.
    for fc, bw, gain in [(300, 70, 1.0), (870, 90, 0.5), (2240, 140, 0.2)]:
        out += gain * _resonator(source, fc, bw)
    out *= np.hanning(n) ** 0.25
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.9).astype(np.float32)


def main() -> None:
    voices = FIX / "voices"
    voices.mkdir(parents=True, exist_ok=True)
    sf.write(str(voices / "donor_ah_synth.wav"), synth_ah_vowel(), SR, subtype="FLOAT")
    sf.write(str(voices / "donor_oo_synth.wav"), synth_oo_vowel(), SR, subtype="FLOAT")
    make_scores()
    make_edge_cases()
    print(f"wrote donors + scores + edge cases under {FIX}")


if __name__ == "__main__":
    main()
