"""Enroll a consented donor vowel as a voders ``Voice``.

Run via uv (record from the mic needs the ``record`` extra for ``sounddevice``)::

    # Import an existing WAV:
    uv run --extra cpu python evals/record_donor.py --input my_voice.wav --voice-id alice

    # Record a sustained vowel from the microphone:
    uv run --extra cpu --extra record python evals/record_donor.py --record --voice-id alice

The script loads or records a sustained vowel (~3 s), isolates the steady portion, normalizes,
resamples to 22,050 Hz mono float32 (FR-002), validates the take (voiced, low noise), captures the
operator's consent + a license string, writes the WAV under ``models/donors/`` (git-ignored), and
prints a ready-to-paste ``Voice`` entry. Because the operator records and consents themselves, the
consent gate (FR-011, SC-008) is satisfied by construction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from voders.audio import read_wav, to_mono_float32, write_wav
from voders.constants import SAMPLE_RATE
from voders.voices.models import Voice, VoiceKind

DONORS_ROOT = Path(__file__).resolve().parents[1] / "models" / "donors"


# ── Capture ───────────────────────────────────────────────────────────────────


def record_mic(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Record a mono take from the default microphone (needs the ``record`` extra)."""
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "ERROR: sounddevice not installed. Re-run with the record extra:\n"
            "  uv run --extra cpu --extra record python evals/record_donor.py ...",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    print(f"Recording {duration:.1f}s — hold a steady 'Aaaah' now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("Recording done.")
    return to_mono_float32(audio)


# ── Processing ────────────────────────────────────────────────────────────────


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample to ``target_sr`` (librosa, matching ``evals/download_donors.py``)."""
    if orig_sr == target_sr:
        return to_mono_float32(audio)
    import librosa

    out = librosa.resample(to_mono_float32(audio), orig_sr=orig_sr, target_sr=target_sr)
    return to_mono_float32(out)


def _frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Short-time RMS envelope."""
    if len(audio) < frame:
        return np.array([float(np.sqrt(np.mean(audio**2)))]) if len(audio) else np.array([0.0])
    starts = range(0, len(audio) - frame + 1, hop)
    return np.array([float(np.sqrt(np.mean(audio[s : s + frame] ** 2))) for s in starts])


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading/trailing silence below ``threshold_db`` of full scale."""
    threshold = 10 ** (threshold_db / 20.0)
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0:
        return audio
    margin = int(0.05 * sr)
    start = max(0, above[0] - margin)
    end = min(len(audio), above[-1] + margin)
    return audio[start:end]


def extract_steady(audio: np.ndarray, sr: int, target_s: float = 3.0) -> np.ndarray:
    """Isolate the steadiest ``target_s`` window — the sustained portion of the vowel.

    Slides a window over the short-time RMS envelope and keeps the one with the lowest coefficient
    of variation (flattest loudness), so onset swells and trailing decay are dropped.
    """
    want = int(target_s * sr)
    if len(audio) <= want:
        return audio
    frame, hop = int(0.05 * sr), int(0.025 * sr)
    env = _frame_rms(audio, frame, hop)
    win_frames = max(1, want // hop)
    if len(env) <= win_frames:
        return audio
    best_start, best_cv = 0, np.inf
    for i in range(len(env) - win_frames + 1):
        seg = env[i : i + win_frames]
        mean = float(seg.mean())
        cv = float(seg.std() / mean) if mean > 1e-9 else np.inf
        if cv < best_cv:
            best_cv, best_start = cv, i
    start = best_start * hop
    return audio[start : start + want]


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Peak-normalize to ``target_peak`` (matches ``download_donors`` 0.9)."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    return to_mono_float32(audio * (target_peak / peak))


# ── Validation ────────────────────────────────────────────────────────────────


def periodicity(audio: np.ndarray, sr: int) -> float:
    """Autocorrelation peak in the 80–1000 Hz f0 range (1.0 = perfectly periodic)."""
    frame = audio[: min(len(audio), int(0.1 * sr))]
    if frame.size == 0:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2 :]
    corr = corr / (corr[0] + 1e-9)
    min_lag, max_lag = sr // 1000, sr // 80
    if max_lag >= len(corr):
        return 0.0
    return float(corr[min_lag:max_lag].max())


def hnr_db(periodicity_peak: float) -> float:
    """Harmonics-to-noise ratio (dB) from the autocorrelation peak (Boersma estimate).

    A clean periodic vowel concentrates energy in the periodic component (peak → 1, HNR → high);
    additive noise and reverb smear the autocorrelation peak down, so this works on a single steady
    clip where an envelope-based SNR cannot (a flat sustained tone has no quiet frames to compare).
    """
    r = min(max(periodicity_peak, 0.0), 1.0 - 1e-6)
    return 10.0 * float(np.log10(r / (1.0 - r))) if r > 0 else -60.0


def validate(audio: np.ndarray, sr: int) -> list[str]:
    """Return a list of problems (empty = usable as a donor voice)."""
    errors: list[str] = []
    duration = len(audio) / sr
    if duration < 0.5:
        errors.append(f"Too short: {duration:.2f}s (need at least 0.5s)")
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    if rms < 0.01:
        errors.append(f"Too quiet: RMS={rms:.4f} (is the microphone working?)")
    voiced = periodicity(audio, sr)
    if voiced < 0.4:
        errors.append(f"Low periodicity ({voiced:.2f}) — hold a clear, steady vowel")
    elif hnr_db(voiced) < 7.0:
        errors.append(
            f"Noisy/reverberant: HNR≈{hnr_db(voiced):.0f} dB (record in a quiet, dry room)"
        )
    return errors


# ── Voice entry ───────────────────────────────────────────────────────────────


def voice_entry_yaml(voice: Voice) -> str:
    """Render a single ``Voice`` as a ready-to-paste YAML list item."""
    entry = voice.model_dump(mode="json")
    return yaml.safe_dump([entry], default_flow_style=False, sort_keys=False, allow_unicode=True)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enroll a donor voice for voders")
    parser.add_argument("--input", "-i", help="Path to an existing WAV file")
    parser.add_argument("--record", "-r", action="store_true", help="Record from the microphone")
    parser.add_argument(
        "--voice-id", required=True, help="Unique name for this voice (e.g. 'alice')"
    )
    parser.add_argument(
        "--license",
        default="personal use, research only",
        help="License string (default: 'personal use, research only')",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DONORS_ROOT),
        help="Where to save the enrolled WAV (default: models/donors)",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0, help="Recording duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive consent prompt (you are consenting)",
    )
    args = parser.parse_args(argv)

    if not args.input and not args.record:
        parser.error("provide either --input <file> or --record")

    # 1. Load or record.
    if args.record:
        audio = record_mic(args.duration)
        sr = SAMPLE_RATE
    else:
        print(f"Loading {args.input} ...")
        audio, sr = read_wav(args.input)
    print(f"  loaded {len(audio) / sr:.2f}s @ {sr} Hz")

    # 2. Resample to the consumer format.
    if sr != SAMPLE_RATE:
        print(f"  resampling {sr} Hz -> {SAMPLE_RATE} Hz ...")
        audio = resample(audio, sr)

    # 3. Trim silence, then isolate the steady sustained portion.
    audio = trim_silence(audio, SAMPLE_RATE)
    audio = extract_steady(audio, SAMPLE_RATE)
    print(f"  steady take: {len(audio) / SAMPLE_RATE:.2f}s")

    # 4. Normalize.
    audio = normalize(audio)

    # 5. Validate.
    print("Validating take ...")
    errors = validate(audio, SAMPLE_RATE)
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        print("\nTip: record in a quiet room, hold a steady 'Aaaah' for ~3 seconds.")
        return 1
    print("  ✓ validation passed")

    # 6. Consent.
    print(f"\nConsent check:\n  voice_id : {args.voice_id}\n  license  : {args.license}")
    if not args.yes:
        answer = input(
            "  Consent to this voice being used for synthetic data generation? [yes/no]: "
        )
        if answer.strip().lower() not in ("yes", "y"):
            print("Consent not given — aborting.")
            return 0

    # 7. Save the WAV (22,050 Hz mono float32).
    out_path = Path(args.output_dir) / f"{args.voice_id}.wav"
    write_wav(out_path, audio)
    print(f"\n  saved {out_path}")

    # 8. Print a ready-to-paste Voice entry — do not mutate any shared config.
    voice = Voice(
        voice_id=args.voice_id,
        kind=VoiceKind.DETERMINISTIC_DONOR,
        license=args.license,
        consent_verified=True,
        model_ref=str(out_path),
    )
    print("\n" + "=" * 60)
    print("Add this under `voices:` in your run config:")
    print("=" * 60)
    print(voice_entry_yaml(voice).rstrip())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
