"""Contract tests for the AccompanimentBackend Protocol and FakeAccompanimentBackend.

Contract: contracts/accompaniment-backend.md. The fake backend is the torch-free, deterministic
CPU baseline the accompaniment stage depends on, so its mode/return invariants and bit-exact
determinism are load-bearing for CI.
"""

from __future__ import annotations

import numpy as np

from voders.render.accompaniment_backend import (
    MODE_COMPLETE,
    MODE_LEGO,
    AccompanimentBackend,
    BackendOutput,
    FakeAccompanimentBackend,
)
from voders.scores.models import Note, Score


def _vocal(seconds: float = 1.0, sr: int = 22050) -> np.ndarray:
    """A quiet 220 Hz sine with real RMS energy (zeros would have none)."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)


def _score() -> Score:
    return Score(
        score_id="s_accomp",
        notes=[
            Note(onset_s=0.1, offset_s=0.4, pitch_midi=57),
            Note(onset_s=0.5, offset_s=0.9, pitch_midi=60),
        ],
    )


def test_fake_backend_is_runtime_checkable_protocol_instance() -> None:
    """FakeAccompanimentBackend satisfies the runtime_checkable Protocol."""
    assert isinstance(FakeAccompanimentBackend(), AccompanimentBackend)


def test_fake_backend_declares_capabilities() -> None:
    """CPU-only, supports both modes, MIT licensed, no attribution required."""
    backend = FakeAccompanimentBackend()
    assert backend.requires_gpu() is False
    assert backend.supports("lego") is True
    assert backend.supports("complete") is True
    assert backend.supports("bogus") is False
    assert backend.model_license == "MIT"
    assert backend.attribution_text is None


def test_lego_returns_stem_only() -> None:
    """Lego sets accompaniment_stem (float32 1-D, vocal-length) and leaves mix None."""
    backend = FakeAccompanimentBackend()
    vocal = _vocal()
    out = backend.generate(vocal, _score(), MODE_LEGO, seed=7, options={})

    assert isinstance(out, BackendOutput)
    assert out.mix is None
    stem = out.accompaniment_stem
    assert isinstance(stem, np.ndarray)
    assert stem.dtype == np.float32
    assert stem.ndim == 1
    assert stem.size == vocal.size


def test_complete_returns_mix_only_without_clipping() -> None:
    """Complete sets mix (stem None) and keeps the peak within [-1, 1]."""
    backend = FakeAccompanimentBackend()
    vocal = _vocal()
    out = backend.generate(vocal, _score(), MODE_COMPLETE, seed=7, options={})

    assert out.accompaniment_stem is None
    mix = out.mix
    assert isinstance(mix, np.ndarray)
    assert mix.dtype == np.float32
    assert mix.ndim == 1
    assert mix.size == vocal.size
    assert float(np.max(np.abs(mix))) <= 1.0


def test_same_seed_is_byte_identical() -> None:
    """Two generate() calls with the same seed return identical arrays (FR-013)."""
    backend = FakeAccompanimentBackend()
    vocal = _vocal()
    score = _score()
    a = backend.generate(vocal, score, MODE_LEGO, seed=42, options={})
    b = backend.generate(vocal, score, MODE_LEGO, seed=42, options={})
    assert np.array_equal(a.accompaniment_stem, b.accompaniment_stem)


def test_different_seeds_differ() -> None:
    """Different seeds yield different stems."""
    backend = FakeAccompanimentBackend()
    vocal = _vocal()
    score = _score()
    a = backend.generate(vocal, score, MODE_LEGO, seed=1, options={})
    b = backend.generate(vocal, score, MODE_LEGO, seed=2, options={})
    assert not np.array_equal(a.accompaniment_stem, b.accompaniment_stem)
