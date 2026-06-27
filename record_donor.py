#!/usr/bin/env python3
"""
record_donor.py — Enroll a donor voice for voders.

Usage:
    # Import an existing WAV file:
    python record_donor.py --input my_voice.wav --voice-id my_name --license "personal use"

    # Record from microphone (requires sounddevice):
    python record_donor.py --record --voice-id my_name --license "personal use"

The script:
1. Loads or records a sustained vowel (~3 seconds)
2. Trims silence from start/end
3. Normalizes loudness
4. Resamples to 22,050 Hz mono
5. Validates the take (voiced, low noise)
6. Saves the WAV + prints a ready-to-use Voice YAML entry
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import scipy.signal
import yaml


TARGET_SR = 22_050  # voders expects 22,050 Hz mono float64


# ─── Audio loading ────────────────────────────────────────────────────────────

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as float32 mono."""
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    # Mix down to mono
    audio = audio.mean(axis=1)
    return audio, sr


def record_mic(duration: float = 4.0, sr: int = TARGET_SR) -> np.ndarray:
    """Record from the default microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print(f"Recording {duration}s — say a sustained 'Aaaah' now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float64")
    sd.wait()
    print("Recording done.")
    return audio.flatten()


# ─── Processing ───────────────────────────────────────────────────────────────

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return scipy.signal.resample(audio, num_samples).astype(np.float64)


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading/trailing silence."""
    threshold = 10 ** (threshold_db / 20.0)
    above = np.where(np.abs(audio) > threshold)[0]
    if len(above) == 0:
        return audio
    # Add small margin
    margin = int(0.05 * sr)
    start = max(0, above[0] - margin)
    end = min(len(audio), above[-1] + margin)
    trimmed = audio[start:end]
    print(f"  Trimmed: {len(audio)/sr:.2f}s → {len(trimmed)/sr:.2f}s")
    return trimmed


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize peak loudness."""
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio
    return (audio * (target_peak / peak)).astype(np.float64)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate(audio: np.ndarray, sr: int) -> list[str]:
    """
    Check if the recording is usable as a donor voice.
    Returns a list of error strings (empty = all good).
    """
    errors = []

    duration = len(audio) / sr
    if duration < 0.5:
        errors.append(f"Too short: {duration:.2f}s (need at least 0.5s)")

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        errors.append(f"Too quiet: RMS={rms:.4f} (is microphone working?)")

    # Simple voiced check: look for periodicity via autocorrelation
    frame = audio[:min(len(audio), int(0.1 * sr))]
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2:]
    corr /= corr[0] + 1e-9
    # Check if there's a clear peak in the f0 range (80–1000 Hz)
    min_lag = sr // 1000
    max_lag = sr // 80
    peak = corr[min_lag:max_lag].max() if max_lag < len(corr) else 0.0
    if peak < 0.3:
        errors.append(
            f"Low periodicity ({peak:.2f}) — make sure you're singing/humming a clear vowel"
        )

    return errors


# ─── YAML output ──────────────────────────────────────────────────────────────

def print_voice_yaml(voice_id: str, wav_path: str, license_str: str) -> None:
    """Print a ready-to-paste Voice entry for smoke.yaml or any run config."""
    print("\n" + "=" * 60)
    print("Add this to your run config YAML under `voices:`:")
    print("=" * 60)
    print(f"""  - voice_id: {voice_id}
    kind: deterministic_donor
    license: "{license_str}"
    consent_verified: true
    model_ref: {wav_path}""")
    print("=" * 60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a donor voice for voders")
    parser.add_argument("--input", "-i", help="Path to an existing WAV file")
    parser.add_argument("--record", "-r", action="store_true", help="Record from microphone")
    parser.add_argument("--voice-id", required=True, help="Unique name for this voice (e.g. 'alice')")
    parser.add_argument("--license", default="personal use, research only",
                        help="License string (default: 'personal use, research only')")
    parser.add_argument("--output-dir", default="evals/fixtures/voices",
                        help="Where to save the enrolled WAV (default: evals/fixtures/voices)")
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Recording duration in seconds (default: 4.0)")
    args = parser.parse_args()

    if not args.input and not args.record:
        parser.error("Provide either --input <file> or --record")

    # 1. Load or record
    if args.record:
        audio = record_mic(duration=args.duration)
        sr = TARGET_SR
    else:
        print(f"Loading {args.input} ...")
        audio, sr = load_wav(args.input)

    print(f"  Loaded: {len(audio)/sr:.2f}s at {sr} Hz")

    # 2. Resample
    if sr != TARGET_SR:
        print(f"  Resampling {sr} Hz → {TARGET_SR} Hz ...")
        audio = resample(audio, sr, TARGET_SR)

    # 3. Trim silence
    audio = trim_silence(audio, TARGET_SR)

    # 4. Normalize
    audio = normalize(audio)

    # 5. Validate
    print("Validating take ...")
    errors = validate(audio, TARGET_SR)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nTip: record in a quiet room, hold a steady 'Aaaah' for 3 seconds.")
        sys.exit(1)
    print("  ✓ Validation passed!")

    # 6. Consent confirmation
    print(f"\nConsent check:")
    print(f"  Voice ID : {args.voice_id}")
    print(f"  License  : {args.license}")
    answer = input("  Do you consent to this voice being used for synthetic data generation? [yes/no]: ")
    if answer.strip().lower() not in ("yes", "y"):
        print("Consent not given — aborting.")
        sys.exit(0)

    # 7. Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.voice_id}.wav"
    sf.write(str(out_path), audio, TARGET_SR, subtype="FLOAT")
    print(f"\n  Saved: {out_path}")

    # 9. Auto-append to real_donor.yaml
    yaml_path = Path("evals/fixtures/real_donor.yaml")
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
             config = yaml.safe_load(f)
        
        new_voice = new_voice = {
        "voice_id": args.voice_id,
        "kind": "deterministic_donor",
        "license": args.license,
        "consent_verified": True,
        "model_ref": str(out_path)
        }

        config["voices"].append(new_voice)

        with open(yaml_path, "w") as f:
            yaml.dump(config, f,
            default_flow_style=False,
            allow_unicode=True)
        print(f"  Added to {yaml_path}")


if __name__ == "__main__":
    main()