"""Augmentor protocol (FR-005, FR-014).

The augmentation post-processor contract the orchestrator depends on. The concrete chain lives in
``voders.render.augment``; keeping the protocol separate lets the orchestrator type against it
without importing the DSP implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Augmentor(Protocol):
    """Applies a named, label-preserving augmentation profile to a rendered sample (FR-005)."""

    def profile_ids(self) -> list[str]: ...

    def apply(
        self, audio: np.ndarray, profile_id: str, seed: int
    ) -> tuple[np.ndarray, float | None]:
        """Return (augmented audio, vocal-to-accompaniment SNR in dB or None).

        MUST NOT alter the paired score's note rows (FR-005). Returns an SNR for the quarantine
        gate (FR-014) when the profile mixes accompaniment, else None.
        """
        ...
