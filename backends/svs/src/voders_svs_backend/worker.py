"""SVS render worker — a CLI invoked by the 3.14 core across a process boundary.

Protocol (deliberately tiny, so the boundary is robust):
  argv[1] = path to a JSON request: {"notes": [[onset_s, offset_s, pitch_midi], ...],
                                      "sr": 22050, "seed": int, "out_wav": path,
                                      "model_ref": str, "mode": "force_score_f0",
                                      "phonemes": [{"note_index", "phonemes", "lead", "tail"}, ...]}
  On success: writes the WAV to out_wav and prints a JSON result {"ok": true, "n_samples": int,
              "backend": "nnsvs", "toolkit_available": bool, "articulated": bool} to stdout; exit 0.

The render is **f0-driven source-filter articulatory synthesis**: pitch comes straight from the
score (so onsets/offsets/pitch are exact), and when the core supplies per-note ``phonemes`` (espeak
G2P, vowel-on-the-beat), the synth articulates them — vowel-specific formants for the nucleus plus
consonant bursts/murmurs for leading/trailing consonants — so the audio carries real phonetic
content, not a single open vowel. A trained NNSVS/DiffSinger voicebank is a drop-in behind this
exact request format (notes + phonemes in, WAV out); the boundary does not change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

# Vowel formants (F1, F2, F3) in Hz, keyed by espeak-ng IPA nucleus symbols. Distinct formant
# targets are what make different vowels audibly different — the core of phonetic diversity.
_VOWEL_FORMANTS: dict[str, tuple[float, float, float]] = {
    "i": (270, 2290, 3010),
    "iː": (270, 2290, 3010),
    "ɪ": (390, 1990, 2550),
    "e": (530, 1840, 2480),
    "ɛ": (530, 1840, 2480),
    "æ": (660, 1720, 2410),
    "a": (730, 1090, 2440),
    "ɑ": (730, 1090, 2440),
    "ɑː": (730, 1090, 2440),
    "ʌ": (640, 1190, 2390),
    "ɔ": (570, 840, 2410),
    "ɒ": (570, 840, 2410),
    "o": (450, 800, 2400),
    "oʊ": (450, 800, 2400),
    "ʊ": (440, 1020, 2240),
    "u": (300, 870, 2240),
    "uː": (300, 870, 2240),
    "ə": (500, 1500, 2500),
    "ɚ": (490, 1350, 1690),
    "ɝ": (490, 1350, 1690),
    "ɜ": (490, 1350, 1690),
}
_DEFAULT_FORMANTS = (730, 1090, 2440)  # open "ah"

_VOWELS = set("aeiouɑɐɒæɛɜɝɪɔʊʌəɚɘɞyøœɶ")
_NASALS = {"m", "n", "ŋ"}
_APPROX = {"l", "r", "ɹ", "w", "j", "ɫ"}
_FRIC_UNVOICED = {"s", "ʃ", "f", "θ", "h", "tʃ"}
_FRIC_VOICED = {"z", "ʒ", "v", "ð", "dʒ"}
_PLOSIVE = {"p", "t", "k", "b", "d", "ɡ", "g"}


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


def _is_vowel(ph: str) -> bool:
    return any(c in _VOWELS for c in ph)


def _glottal_source(n: int, f0: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A harmonic glottal source at ``f0`` with light seeded vibrato (the voiced excitation)."""
    t = np.arange(n) / sr
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * f0 * np.cumsum(vib) / sr
    nyq = sr / 2
    return sum((1.0 / k) * np.sin(k * phase) for k in range(1, 30) if k * f0 < nyq)


def _formants(src: np.ndarray, formants: tuple[float, float, float], sr: int) -> np.ndarray:
    out = np.zeros_like(src)
    for fc, g in zip(formants, (1.0, 0.6, 0.3), strict=True):
        out += g * _resonator(src, fc, 90.0 + 0.1 * fc, sr)
    return out


def _vowel_segment(
    n: int, f0: float, formants: tuple[float, float, float], sr: int, rng: np.random.Generator
) -> np.ndarray:
    return _formants(_glottal_source(n, f0, sr, rng), formants, sr)


def _consonant_segment(
    ph: str, n: int, f0: float, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Synthesize a short consonant: noise for fricatives/plosives, murmur for nasals/liquids."""
    if n <= 0:
        return np.zeros(0)
    noise = rng.standard_normal(n)
    if ph in _PLOSIVE:
        seg = np.zeros(n)
        burst = max(1, n // 3)
        seg[-burst:] = rng.standard_normal(burst)
        fc = 2500.0 if ph in {"t", "d"} else (1500.0 if ph in {"k", "ɡ", "g"} else 800.0)
        return _resonator(seg, fc, 800.0, sr)
    if ph in _FRIC_UNVOICED or ph in _FRIC_VOICED:
        if ph in {"s", "z"}:
            band = _resonator(noise, 6500.0, 1500.0, sr)
        elif ph in {"ʃ", "ʒ", "tʃ", "dʒ"}:
            band = _resonator(noise, 3000.0, 1200.0, sr)
        else:
            band = _resonator(noise, 1800.0, 1500.0, sr)
        if ph in _FRIC_VOICED:  # add voicing for z/ʒ/v/ð
            band = band + 0.5 * _glottal_source(n, f0, sr, rng)
        return band
    if ph in _NASALS:  # voiced low-formant murmur
        return _formants(_glottal_source(n, f0, sr, rng), (250.0, 1000.0, 2200.0), sr)
    # approximants / liquids / unknown: voiced through a neutral-ish formant
    return _formants(_glottal_source(n, f0, sr, rng), (400.0, 1200.0, 2500.0), sr)


def _fade(seg: np.ndarray, sr: int) -> np.ndarray:
    fade = min(int(0.005 * sr), seg.size // 2)
    if fade > 0:
        ramp = np.linspace(0, 1, fade)
        seg[:fade] *= ramp
        seg[-fade:] *= ramp[::-1]
    return seg


def render(
    notes: list[list[float]],
    sr: int,
    seed: int,
    phonemes: list[dict] | None = None,
) -> np.ndarray:
    """f0-driven articulatory synthesis: score pitch (exact onsets/offsets), phonemes sung."""
    rng = np.random.default_rng(seed)
    by_note = {int(p["note_index"]): p for p in (phonemes or [])}
    total = max((off for _, off, _ in notes), default=0.0)
    out = np.zeros(int(round(total * sr)) + 1, dtype=np.float64)

    c_len = int(0.035 * sr)  # ~35 ms per consonant

    def _add(pos: int, seg: np.ndarray) -> None:
        a = max(0, pos)
        b = min(pos + seg.size, out.size)
        if b > a:
            out[a:b] += seg[a - pos : b - pos]

    for i, (onset, offset, pitch) in enumerate(notes):
        n = int(round((offset - onset) * sr))
        if n <= 0:
            continue
        f0 = _midi_to_hz(pitch)
        start = int(round(onset * sr))
        entry = by_note.get(i)

        if entry is None:
            _add(start, _fade(_vowel_segment(n, f0, _DEFAULT_FORMANTS, sr, rng), sr))
            continue

        lead = [p for p in entry.get("lead", []) if p]
        tail = [p for p in entry.get("tail", []) if p]
        nucleus = next((p for p in entry.get("phonemes", []) if _is_vowel(p)), None)
        formants = _VOWEL_FORMANTS.get(nucleus or "", _DEFAULT_FORMANTS)

        # Vowel-on-the-beat (Decision L3): the vowel nucleus fills the whole labelled note, so the
        # measured (voiced) onset stays on the score beat and f0 coverage stays high.
        _add(start, _fade(_vowel_segment(n, f0, formants, sr, rng), sr))
        # Leading consonants ride in a PRE-onset window (before the score onset); trailing ones in a
        # post-offset window — outside the labelled interval, so they don't move the label. Clamp to
        # 0 so a note at t=0 still gets its consonant (overlapping the onset when there's no room).
        cpos = max(0, start - c_len * len(lead))
        for ph in lead:
            _add(cpos, _fade(_consonant_segment(ph, c_len, f0, sr, rng), sr))
            cpos += c_len
        cpos = start + n
        for ph in tail:
            _add(cpos, _fade(_consonant_segment(ph, c_len, f0, sr, rng), sr))
            cpos += c_len

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out *= 0.9 / peak
    return out.astype(np.float32)


def _handle(req: dict) -> dict:
    """Render one request to its ``out_wav`` and return the JSON response dict.

    Shared by the one-shot CLI and the persistent ``--serve`` loop. Because the NNSVS engine is
    cached per process, calling this repeatedly in one ``--serve`` process loads the model once and
    reuses it — eliminating the per-sample reload that dominates batch renders.
    """
    sr = int(req.get("sr", 22050))
    phonemes = req.get("phonemes")
    lyrics = req.get("lyrics")
    notes = req["notes"]
    seed = int(req.get("seed", 0))

    # The voice's model_ref selects real NNSVS singing (and a voice character), else the in-process
    # formant articulator. Forms: "nnsvs:<char>" (male|female|tenor|... -> formant warp),
    # "nnsvs:donor:<wav>" (character from a freesound/VocalSet donor), a real NNSVS model id
    # ("<a>/<b>"), "yoko"/""; a plain donor ".wav" stays on the formant path (back-compat).
    model_ref = str(req.get("model_ref", ""))
    if _wants_nnsvs(model_ref):
        # nnsvs was explicitly requested: render with the real singing model or FAIL — never degrade
        # to the formant synth (its robotic timbre is exactly what we don't want in the corpus). A
        # failed sample is reported as an error so the orchestrator rejects it (no audio written).
        if not lyrics:
            return {"ok": False, "error": "nnsvs requires per-note lyrics/syllables"}
        from voders_svs_backend.nnsvs_engine import (
            DEFAULT_MODEL,
            character_factor,
            donor_formant_factor,
            render_nnsvs,
        )

        base, factor = DEFAULT_MODEL, 1.0
        if model_ref.startswith("nnsvs:donor:"):
            factor = donor_formant_factor(model_ref[len("nnsvs:donor:") :])
        elif model_ref.startswith("nnsvs:"):
            factor = character_factor(model_ref[len("nnsvs:") :])
        elif "/" in model_ref and not model_ref.endswith(".wav"):
            base = model_ref  # a real NNSVS voicebank id
        audio = render_nnsvs(notes, lyrics, sr, model_ref=base, formant_factor=factor)
        engine = "nnsvs"
    else:
        # Formant synth only when nnsvs was NOT requested (a config that explicitly chose it).
        audio = render(notes, sr, seed, phonemes=phonemes)
        engine = "formant"

    out_wav = req["out_wav"]
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, audio, sr, subtype="FLOAT")
    return {
        "ok": True,
        "n_samples": int(audio.size),
        "backend": engine,
        "toolkit_available": _toolkit_available(),
        "articulated": bool(phonemes) or engine == "nnsvs",
    }


def _serve() -> int:
    """Persistent mode: one JSON request per stdin line, one JSON response per stdout line.

    The model loads on the first request and is reused for the rest (the engine is process-cached),
    so a long batch render pays the model-load cost once instead of per sample. The core's
    backend_bridge spawns one of these and streams requests to it.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = _handle(json.loads(line))
        except Exception as exc:  # noqa: BLE001 — report per-request, keep the worker alive
            resp = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "--serve":
        return _serve()
    if not args:
        print(json.dumps({"ok": False, "error": "missing request path"}))
        return 2
    print(json.dumps(_handle(json.loads(Path(args[0]).read_text(encoding="utf-8")))))
    return 0


def _wants_nnsvs(model_ref: str) -> bool:
    """True when the voice's model_ref selects an NNSVS voicebank (id), not a donor .wav path."""
    if not model_ref:
        return False
    return model_ref in ("yoko", "nnsvs") or model_ref.startswith("nnsvs:") or (
        "/" in model_ref and not model_ref.endswith(".wav")
    )


if __name__ == "__main__":
    sys.exit(main())
