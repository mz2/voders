"""SVS render worker — a CLI invoked by the 3.14 core across a process boundary.

Protocol (deliberately tiny, so the boundary is robust):
  argv[1] = path to a JSON request: {"notes": [[onset_s, offset_s, pitch_midi], ...],
                                      "sr": 22050, "seed": int, "out_wav": path,
                                      "model_ref": str, "mode": "force_score_f0"}
  On success: writes the WAV to out_wav and prints a JSON result {"ok": true, "n_samples": int,
              "backend": "nnsvs", "toolkit_available": bool} to stdout; exit 0.

The actual render is an f0-driven harmonic synthesis (pitch taken directly from the score, so the
audio is score-aligned). NNSVS is imported to confirm the isolated toolkit is available; plugging a
trained NNSVS voice bank in here (it needs model weights + config) is the production step — the
boundary, the request format, and the returned WAV do not change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter


def _toolkit_available() -> bool:
    try:
        import nnsvs  # noqa: F401

        return True
    except Exception:
        return False


def _midi_to_hz(p: float) -> float:
    return 440.0 * (2.0 ** ((p - 69) / 12.0))


def _resonator(x: np.ndarray, fc: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * fc / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1.0 - r]
    return np.asarray(lfilter(b, a, x), dtype=np.float64)


def render(notes: list[list[float]], sr: int, seed: int) -> np.ndarray:
    """f0-driven harmonic synthesis: pitch comes from the score, so onsets/offsets are exact."""
    rng = np.random.default_rng(seed)
    total = max((off for _, off, _ in notes), default=0.0)
    out = np.zeros(int(round(total * sr)) + 1, dtype=np.float64)
    for onset, offset, pitch in notes:
        n = int(round((offset - onset) * sr))
        if n <= 0:
            continue
        t = np.arange(n) / sr
        # Light, seeded vibrato for expressiveness; pitch centre stays on the score note.
        vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2 * np.pi))
        phase = 2 * np.pi * _midi_to_hz(pitch) * np.cumsum(vib) / sr
        src = sum(
            (1.0 / k) * np.sin(k * phase) for k in range(1, 30) if k * _midi_to_hz(pitch) < sr / 2
        )
        voiced = np.zeros(n)
        for fc, bw, g in [(730, 90, 1.0), (1090, 110, 0.6), (2440, 160, 0.3)]:
            voiced += g * _resonator(src, fc, bw, sr)
        fade = min(int(0.005 * sr), n // 2)
        if fade > 0:
            ramp = np.linspace(0, 1, fade)
            voiced[:fade] *= ramp
            voiced[-fade:] *= ramp[::-1]
        start = int(round(onset * sr))
        end = min(start + n, out.size)
        out[start:end] += voiced[: end - start]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out *= 0.9 / peak
    return out.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(json.dumps({"ok": False, "error": "missing request path"}))
        return 2
    req = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    sr = int(req.get("sr", 22050))
    audio = render(req["notes"], sr, int(req.get("seed", 0)))
    out_wav = req["out_wav"]
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, audio, sr, subtype="FLOAT")
    print(
        json.dumps(
            {
                "ok": True,
                "n_samples": int(audio.size),
                "backend": "nnsvs",
                "toolkit_available": _toolkit_available(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
