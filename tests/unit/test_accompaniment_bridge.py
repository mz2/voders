"""Unit tests for the corpus-format bridge (FR-002; research Decision 3).

``bridge_to_corpus_format`` down-mixes to mono and resamples a backend's native-rate output into the
fixed corpus format (22,050 Hz mono float32).
"""

from __future__ import annotations

import numpy as np

from voders.audio import bridge_to_corpus_format
from voders.constants import SAMPLE_RATE


def test_stereo_48k_downmixes_and_resamples_to_corpus_rate() -> None:
    """Stereo 48 kHz input becomes mono float32 at 22,050 Hz with the expected length."""
    native_sr = 48_000
    n = native_sr  # 1 second
    rng = np.random.default_rng(0)
    stereo = rng.standard_normal((n, 2)).astype(np.float32)

    out = bridge_to_corpus_format(stereo, native_sr)

    assert out.dtype == np.float32
    assert out.ndim == 1
    expected = int(round(n * SAMPLE_RATE / native_sr))
    assert abs(out.size - expected) <= 2


def test_equal_rate_averages_stereo_to_mono_unchanged_length() -> None:
    """When native == target, stereo is averaged to mono and the length is unchanged."""
    stereo = np.array([[0.0, 2.0], [1.0, 3.0], [4.0, 4.0]], dtype=np.float32)

    out = bridge_to_corpus_format(stereo, SAMPLE_RATE, SAMPLE_RATE)

    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == 3
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0])


def test_empty_input_returns_empty_float32() -> None:
    """An empty buffer bridges to an empty float32 array."""
    out = bridge_to_corpus_format(np.zeros(0, dtype=np.float32), 48_000)

    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == 0
