"""Donor enrollment helpers (evals/record_donor.py): processing + validation + Voice entry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from voders.constants import SAMPLE_RATE

# evals/ is a script dir, not an installed package — load the module by path.
_SPEC = importlib.util.spec_from_file_location(
    "record_donor", Path(__file__).resolve().parents[2] / "evals" / "record_donor.py"
)
rd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rd)


def _vowel(seconds: float, f0: float = 150.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A synthetic voiced vowel: f0 plus a couple of harmonics."""
    t = np.arange(int(seconds * sr)) / sr
    sig = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
    sig += 0.3 * np.sin(2 * np.pi * 3 * f0 * t)
    return (sig / np.abs(sig).max()).astype(np.float32)


def test_resample_changes_length_and_stays_float32():
    src = _vowel(1.0, sr=44_100)
    out = rd.resample(src, 44_100, SAMPLE_RATE)
    assert out.dtype == np.float32
    assert abs(len(out) / SAMPLE_RATE - 1.0) < 0.01  # ~1 second preserved
    # A no-op resample returns the same rate untouched (still float32 mono).
    same = rd.resample(src, SAMPLE_RATE, SAMPLE_RATE)
    assert same.dtype == np.float32


def test_trim_silence_removes_leading_and_trailing_quiet():
    sr = SAMPLE_RATE
    quiet = np.zeros(sr, dtype=np.float32)
    body = _vowel(1.0)
    padded = np.concatenate([quiet, body, quiet])
    trimmed = rd.trim_silence(padded, sr)
    # Most of the 2 s of silence is gone; the ~1 s vowel survives.
    assert len(trimmed) < len(padded)
    assert 0.9 * sr < len(trimmed) < 1.5 * sr


def test_extract_steady_picks_a_target_window_from_a_longer_take():
    sr = SAMPLE_RATE
    # A swelling onset (rising envelope) then a steady tail.
    onset = _vowel(1.0) * np.linspace(0.0, 1.0, sr, dtype=np.float32)
    steady = _vowel(3.0)
    take = np.concatenate([onset, steady])
    out = rd.extract_steady(take, sr, target_s=3.0)
    assert abs(len(out) / sr - 3.0) < 0.1

    # The chosen window's loudness envelope is flatter than the full take's (onset swell dropped).
    def env_cv(a: np.ndarray) -> float:
        env = rd._frame_rms(a, int(0.05 * sr), int(0.025 * sr))
        return float(env.std() / env.mean())

    assert env_cv(out) < env_cv(take)


def test_normalize_scales_peak_to_target():
    out = rd.normalize(_vowel(0.5) * 0.1, target_peak=0.9)
    assert np.isclose(np.abs(out).max(), 0.9, atol=1e-3)
    assert out.dtype == np.float32
    # Silence is left untouched (no divide-by-zero blowup).
    silent = np.zeros(100, dtype=np.float32)
    assert np.array_equal(rd.normalize(silent), silent)


def test_validate_accepts_a_clean_vowel():
    assert rd.validate(_vowel(3.0), SAMPLE_RATE) == []


def test_validate_rejects_short_quiet_and_noisy_takes():
    sr = SAMPLE_RATE
    # Too short.
    assert any("short" in e.lower() for e in rd.validate(_vowel(0.2), sr))
    # Too quiet.
    assert any("quiet" in e.lower() for e in rd.validate(_vowel(2.0) * 1e-4, sr))
    # White noise: not voiced.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(2 * sr).astype(np.float32) * 0.3
    assert any("periodicity" in e.lower() for e in rd.validate(noise, sr))


def test_validate_flags_a_noisy_vowel_as_reverberant():
    sr = SAMPLE_RATE
    rng = np.random.default_rng(1)
    # A voiced vowel with moderate noise: periodicity stays above the voiced floor but HNR drops,
    # so it is flagged as noisy rather than unvoiced.
    dirty = _vowel(2.0) + rng.standard_normal(2 * sr).astype(np.float32) * 0.4
    dirty = (dirty / np.abs(dirty).max()).astype(np.float32)
    assert rd.periodicity(dirty, sr) >= 0.4  # still "voiced", so the noise gate is what fires
    errors = rd.validate(dirty, sr)
    assert any("noisy" in e.lower() or "reverber" in e.lower() for e in errors)


def test_voice_entry_yaml_is_paste_ready_and_consented():
    from voders.voices.models import Voice, VoiceKind

    voice = Voice(
        voice_id="alice",
        kind=VoiceKind.DETERMINISTIC_DONOR,
        license="personal use",
        consent_verified=True,
        model_ref="models/donors/alice.wav",
    )
    parsed = yaml.safe_load(rd.voice_entry_yaml(voice))
    assert parsed == [
        {
            "voice_id": "alice",
            "kind": "deterministic_donor",
            "license": "personal use",
            "consent_verified": True,
            "model_ref": "models/donors/alice.wav",
        }
    ]


def test_hnr_is_monotonic_in_periodicity():
    assert rd.hnr_db(0.99) > rd.hnr_db(0.8) > rd.hnr_db(0.5)


@pytest.mark.parametrize("peak", [0.0, 1.0])
def test_hnr_handles_degenerate_peaks(peak):
    # Must not raise / produce inf at the boundaries.
    assert np.isfinite(rd.hnr_db(peak))
