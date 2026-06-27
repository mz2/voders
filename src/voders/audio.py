"""Audio I/O helpers in the fixed consumer format: 22,050 Hz mono float32 (FR-002)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voders.constants import SAMPLE_RATE


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Coerce to 1-D float32 (mono). Multi-channel input is averaged to mono."""
    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def bridge_to_corpus_format(
    audio: np.ndarray, native_sr: int, target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Down-mix to mono and resample to the corpus rate (research.md Decision 3).

    The accompaniment models run at 48 kHz stereo; the corpus is 22,050 Hz mono float32. This
    bridges a backend's native-rate output into the corpus format before mixing/validation/storage.
    Mono coercion happens first (channels averaged), the DC offset is removed (model/separator
    output can carry a large inaudible DC component that wastes headroom), then a band-limited
    resample.
    """
    mono = to_mono_float32(audio)
    if mono.size == 0:
        return mono
    mono = mono - float(mono.mean())  # DC block — keep the audible content, drop the 0 Hz offset
    if native_sr == target_sr:
        return to_mono_float32(mono)
    import librosa

    resampled = librosa.resample(mono.astype(np.float32), orig_sr=native_sr, target_sr=target_sr)
    return to_mono_float32(resampled)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write 22,050 Hz mono float32 WAV (FR-002)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), to_mono_float32(audio), sr, subtype="FLOAT")


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float32 plus its sample rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_float32(data), int(sr)
