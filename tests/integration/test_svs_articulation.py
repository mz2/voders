"""The SVS backend articulates phonemes into real phonetic content (US1, FR-006).

Drives the backend's source-filter synth directly (pure numpy/scipy, no GPU): different vowels yield
different spectra, and a leading fricative adds high-frequency energy the open vowel lacks. Pitch is
score-derived, so this is articulation *on top of* exact-pitch synthesis.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = REPO_ROOT / "backends" / "svs" / "src" / "voders_svs_backend" / "worker.py"

_spec = importlib.util.spec_from_file_location("voders_svs_backend_worker", _WORKER)
assert _spec and _spec.loader
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

SR = 22050
NOTE = [[0.0, 0.6, 60]]  # one ~0.6 s note at middle C


def _hf_energy(audio: np.ndarray, sr: int = SR, cutoff: float = 4000.0) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1 / sr)
    total = float(spec.sum()) or 1.0
    return float(spec[freqs >= cutoff].sum()) / total


def _phon(lead, nucleus, tail=None):
    tail = tail or []
    return [{"note_index": 0, "phonemes": lead + [nucleus] + tail, "lead": lead, "tail": tail}]


def test_different_vowels_produce_different_audio():
    ah = worker.render(NOTE, SR, 0, phonemes=_phon([], "ɑ"))
    ee = worker.render(NOTE, SR, 0, phonemes=_phon([], "iː"))
    assert ah.shape == ee.shape
    # Different formant targets => materially different waveforms.
    assert float(np.mean(np.abs(ah - ee))) > 1e-3


def test_fricative_adds_high_frequency_energy():
    vowel_only = worker.render(NOTE, SR, 0, phonemes=_phon([], "ɑ"))
    with_s = worker.render(NOTE, SR, 0, phonemes=_phon(["s"], "ɑ"))
    assert _hf_energy(with_s) > _hf_energy(vowel_only)


def test_output_is_finite_and_score_length():
    audio = worker.render(NOTE, SR, 0, phonemes=_phon(["s", "t"], "ɑ", ["p"]))
    assert np.all(np.isfinite(audio))
    assert abs(audio.size - 0.6 * SR) <= 0.02 * SR
    assert float(np.max(np.abs(audio))) <= 1.0


def test_no_phonemes_falls_back_to_open_vowel():
    a = worker.render(NOTE, SR, 0)
    b = worker.render(NOTE, SR, 0, phonemes=_phon([], "ɑ"))
    # Default (no phonemes) uses the open-"ah" formants, matching an explicit ɑ nucleus.
    assert float(np.mean(np.abs(a - b))) < 1e-6
