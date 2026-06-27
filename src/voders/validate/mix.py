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
from voders.manifest.models import ValidationVerdict, VerdictStatus
from voders.scores.models import Score
from voders.validate.rederive import derive_labels
from voders.validate.timing import TimingRegistry
from voders.validate.validator import Validator


@dataclass
class MixValidation:
    """Result of validating an accompaniment sample on the mix."""

    verdict: ValidationVerdict
    max_note_shift_ms: float


def _preserved_timing_verdict(
    verdict: ValidationVerdict, shift_ms: float, onset_tol_ms: float
) -> ValidationVerdict:
    """Re-judge a Lego (vocal-preserving) verdict using the measured note shift for timing.

    In Lego mode the vocal is byte-identical, so its note timing is preserved by construction; the
    authoritative "did a note move?" signal is the measured ``max_note_shift_ms`` (Decision 5), not
    a re-detection of onsets on the *polyphonic* mix (where the accompaniment's tonal content can
    shift the first-in-tune frame even though the vocal never moved). So when the measured shift is
    within tolerance, onset/offset are satisfied by construction; masking still gates admission
    through f0 coverage (``f0_ok``), the SNR floor (QUARANTINED), and clipping.
    """
    if shift_ms > onset_tol_ms:
        return verdict  # a note genuinely moved beyond tolerance — keep the rejection
    if verdict.status == VerdictStatus.QUARANTINED:
        return verdict  # SNR floor (masked vocal) — keep
    reason = verdict.reason or ""
    if "clipping" in reason or not verdict.f0_ok:
        return verdict  # genuine distortion / masked pitch — keep the rejection
    flagged = "shorter than min_note_ms" in reason
    status = VerdictStatus.FLAGGED if flagged else VerdictStatus.ACCEPTED
    new_reason = verdict.reason if flagged else None
    return verdict.model_copy(
        update={"onset_ok": True, "offset_ok": True, "status": status, "reason": new_reason}
    )


def validate_on_mix(
    mix: np.ndarray,
    label_score: Score,
    validator: Validator,
    *,
    snr_db: float | None = None,
    vocal_estimate: np.ndarray | None = None,
    vocal_preserving: bool = False,
    timing: TimingRegistry | None = None,
    sr: int = SAMPLE_RATE,
) -> MixValidation:
    """Validate the mix and measure how far any sung note shifted in time (FR-004, FR-012).

    The verdict comes from the existing ``Validator`` run on the mix (so masking trips the SNR/pitch
    gates). ``max_note_shift_ms`` is the largest ``|rederived_onset - original_onset|`` measured by
    ``derive_labels`` on the vocal estimate (or the mix when none is given) — the operative
    "did a note move?" quantity (SC-005); the target is zero.

    ``vocal_preserving=True`` (Lego) judges timing by the measured shift rather than by re-detecting
    onsets on the polyphonic mix, so a byte-identical vocal isn't falsely rejected when the
    accompaniment confuses the onset detector; masking is still caught by f0 coverage and the SNR
    floor (see ``_preserved_timing_verdict``).
    """
    verdict = validator.validate(mix, label_score, snr_db=snr_db)
    shift_source = vocal_estimate if vocal_estimate is not None else mix
    _, max_note_shift_ms = derive_labels(shift_source, label_score, sr=sr, timing=timing)
    if vocal_preserving:
        verdict = _preserved_timing_verdict(verdict, max_note_shift_ms, validator.cfg.onset_ms)
    return MixValidation(verdict=verdict, max_note_shift_ms=max_note_shift_ms)
