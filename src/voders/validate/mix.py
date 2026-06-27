"""Validate an accompaniment sample on the final mix (FR-004, FR-012; research Decisions 2, 5).

Admission for the accompaniment stage is decided on the **final vocal+accompaniment mix**: the sung
notes must stay within tolerance AND remain detectable through the accompaniment (a masked vocal is
rejected by the existing SNR/pitch gates). This module wraps the existing ``Validator`` and adds the
per-note timing-shift measurement the spec gates on (FR-012 / SC-005), reusing
``validate/rederive.py::derive_labels`` (Decision 5).

For Complete (one-pass) mode there is no separate vocal stem, so callers may pass a separated vocal
estimate to measure the shift against; absent one, the mix itself is used (the documented fallback,
Decision 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voders.constants import SAMPLE_RATE
from voders.manifest.models import ValidationVerdict
from voders.scores.models import Score
from voders.validate.rederive import derive_labels
from voders.validate.timing import TimingRegistry
from voders.validate.validator import Validator


@dataclass
class MixValidation:
    """Result of validating an accompaniment sample on the mix."""

    verdict: ValidationVerdict
    max_note_shift_ms: float


def validate_on_mix(
    mix: np.ndarray,
    label_score: Score,
    validator: Validator,
    *,
    snr_db: float | None = None,
    vocal_estimate: np.ndarray | None = None,
    timing: TimingRegistry | None = None,
    sr: int = SAMPLE_RATE,
) -> MixValidation:
    """Validate the mix and measure how far any sung note shifted in time (FR-004, FR-012).

    The verdict comes from the existing ``Validator`` run on the mix (so masking trips the SNR/pitch
    gates). ``max_note_shift_ms`` is the largest ``|rederived_onset - original_onset|`` measured by
    ``derive_labels`` on the vocal estimate (or the mix when none is given) — the operative
    "did a note move?" quantity (SC-005); the target is zero.
    """
    verdict = validator.validate(mix, label_score, snr_db=snr_db)
    shift_source = vocal_estimate if vocal_estimate is not None else mix
    _, max_note_shift_ms = derive_labels(shift_source, label_score, sr=sr, timing=timing)
    return MixValidation(verdict=verdict, max_note_shift_ms=max_note_shift_ms)
