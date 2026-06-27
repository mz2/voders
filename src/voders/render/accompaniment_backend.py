"""Accompaniment backend boundary (contract: contracts/accompaniment-backend.md).

The accompaniment stage depends only on the ``AccompanimentBackend`` Protocol. Two implementations
exist: the torch-free deterministic ``FakeAccompanimentBackend`` (CI/tests, the load-bearing
baseline) and the GPU ACE-Step backend in ``voders.render.backends.acestep`` (lazy torch import,
``accomp`` extra). Backends return audio already bridged to the corpus format — 22,050 Hz mono
float32 (research.md Decision 3); the stage never sees 48 kHz stereo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.signal import butter, lfilter

from voders.audio import to_mono_float32
from voders.constants import SAMPLE_RATE
from voders.scores.models import Score
from voders.seeds import rng as rng_for_seed

# Conditioning modes (data-model.md §1).
MODE_LEGO = "lego"  # generate an isolated accompaniment stem; the stage sums it under the vocal
MODE_COMPLETE = "complete"  # generate the full mix in one pass (vocal re-encoded)


@dataclass
class BackendOutput:
    """A backend's result, already in 22,050 Hz mono float32 (contract §1).

    Exactly one of ``accompaniment_stem`` / ``mix`` is set per mode (contract §2):
    Lego sets ``accompaniment_stem``; Complete sets ``mix``.
    """

    accompaniment_stem: np.ndarray | None
    mix: np.ndarray | None
    native_sr: int = SAMPLE_RATE


@runtime_checkable
class AccompanimentBackend(Protocol):
    """The contract the accompaniment stage depends on (contract: accompaniment-backend.md)."""

    name: str
    model_id: str
    model_version: str
    model_license: str  # checked against license_policy BEFORE generate() (FR-006)
    attribution_text: str | None  # CC-BY-class credit, propagated to provenance (FR-006a)

    def requires_gpu(self) -> bool: ...

    def supports(self, mode: str) -> bool: ...

    def generate(
        self,
        vocal: np.ndarray,
        score: Score,
        mode: str,
        seed: int,
        options: dict[str, object],
    ) -> BackendOutput: ...


def _score_shaped_pad(vocal: np.ndarray, score: Score, seed: int) -> np.ndarray:
    """A deterministic, onset-aligned accompaniment pad: lowpass noise gated to the note spans.

    The pad has energy only while a note sounds, so it is plausibly "related" to the singing and
    does not introduce a competing pulse — keeping the fake backend honest about the note-timing
    invariant the real model must also respect. Reproducible from ``seed`` (FR-013).
    """
    n = int(vocal.size)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)
    rng = rng_for_seed(seed)
    noise = rng.standard_normal(n)
    b, a = butter(2, 1500.0 / (SAMPLE_RATE / 2.0), btype="low")
    pad = np.asarray(lfilter(b, a, noise), dtype=np.float64)
    gate = np.zeros(n, dtype=np.float64)
    for note in score.notes:
        lo = max(0, int(round(note.onset_s * SAMPLE_RATE)))
        hi = min(n, int(round(note.offset_s * SAMPLE_RATE)))
        if hi > lo:
            gate[lo:hi] = 1.0
    pad *= gate
    peak = float(np.max(np.abs(pad))) if pad.size else 0.0
    if peak > 0.0:
        pad = pad / peak * 0.5
    return pad.astype(np.float64)


class FakeAccompanimentBackend:
    """Deterministic, torch-free backend for tests/CI (contract: fake-backend section).

    Lego: a seeded, score-shaped band-limited pad as the stem. Complete: that pad summed under a
    copy of the input vocal as the mix. Declares an MIT license with no attribution requirement.
    """

    name = "fake"
    model_id = "fake-accompaniment"
    model_version = "0"
    model_license = "MIT"
    attribution_text: str | None = None

    def requires_gpu(self) -> bool:
        return False

    def supports(self, mode: str) -> bool:
        return mode in (MODE_LEGO, MODE_COMPLETE)

    def generate(
        self,
        vocal: np.ndarray,
        score: Score,
        mode: str,
        seed: int,
        options: dict[str, object],
    ) -> BackendOutput:
        vocal = to_mono_float32(vocal)
        pad = _score_shaped_pad(vocal, score, seed)
        if mode == MODE_LEGO:
            return BackendOutput(accompaniment_stem=pad.astype(np.float32), mix=None)
        if mode == MODE_COMPLETE:
            mixed = vocal.astype(np.float64) + pad
            # A real model emits well-formed audio; keep headroom so the mix never clips. The
            # uniform scale leaves pitch/timing unchanged (the validator's clipping gate is for
            # genuine distortion, not for the fake's summing).
            peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
            if peak > 0.99:
                mixed = mixed * (0.99 / peak)
            return BackendOutput(accompaniment_stem=None, mix=to_mono_float32(mixed))
        raise ValueError(f"unknown accompaniment mode {mode!r}; expected lego | complete")
