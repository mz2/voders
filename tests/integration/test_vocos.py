"""Vocos neural-vocoder enhancement (skipped unless the gpu extra with vocos is installed)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vocos") is None,
    reason="vocos not installed (needs the gpu extra)",
)


def test_enhance_runs_and_returns_audio() -> None:
    """enhance() round-trips audio through Vocos and returns a finite 1-D float32 signal.

    Acceptance is intentionally not asserted: mel Vocos is not f0-conditioned and the alignment
    gate rejects its (slightly detuned) output — this experiment exists to record that finding.
    """
    from voders.render.vocos_enhance import enhance

    sr = 22_050
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    out = enhance(tone, sr=sr, device="cpu")

    assert out.ndim == 1
    assert out.dtype == np.float32
    assert out.size > sr // 2
    assert np.all(np.isfinite(out))
