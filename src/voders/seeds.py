"""Deterministic seed derivation (FR-013).

A single run-level ``master_seed`` deterministically derives every per-sample and per-stage
seed from stable identifiers (score id + voice id + stage name), so any one sample reproduces
in isolation regardless of worker count or processing order.
"""

from __future__ import annotations

import hashlib

import numpy as np

_MASK_63 = (1 << 63) - 1


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable non-negative 63-bit seed from the master seed and identifier parts.

    The result depends only on the inputs (not on call order or worker count), so the same
    (master_seed, parts) always yields the same seed.
    """
    h = hashlib.sha256()
    h.update(str(int(master_seed)).encode("utf-8"))
    for part in parts:
        h.update(b"\x1f")  # unit separator so ("a","b") != ("ab",)
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big") & _MASK_63


def sample_seed(master_seed: int, score_id: str, voice_id: str, stage: str) -> int:
    """Per-sample/per-stage seed for one (score, voice, stage) triple (FR-013)."""
    return derive_seed(master_seed, score_id, voice_id, stage)


def rng(seed: int) -> np.random.Generator:
    """A numpy random generator bound to a derived seed."""
    return np.random.default_rng(seed)
