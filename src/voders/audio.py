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


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write 22,050 Hz mono float32 WAV (FR-002)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), to_mono_float32(audio), sr, subtype="FLOAT")


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float32 plus its sample rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_float32(data), int(sr)
