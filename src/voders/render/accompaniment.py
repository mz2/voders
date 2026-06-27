"""Accompaniment stage — lay instrumental backing under an accepted corpus vocal (spec 002).

A post-acceptance stage (not a from-score ``RendererLane``): it consumes an accepted base render
(vocal audio + its correct-by-construction score) and emits a new sample with accompaniment, in one
of two modes:

* **Lego (vocal-preserving)** — the backend generates an isolated accompaniment stem; this stage
  sums it beneath the **untouched** vocal at a target vocal-to-accompaniment ratio. The vocal is
  never re-encoded, so its note timing is preserved by construction and the source vocal is retained
  bit-identically (SC-001). The stem is kept for label-safe re-mixing (FR-015, SC-008).
* **Complete (one-pass)** — the backend emits a single full mix (vocal re-encoded, mild coloration);
  admitted only if the sung notes still land within tolerance when validated on the mix (SC-002).

Admission is decided on the **final mix** (``validate/mix.py``); among up to ``takes`` validator-
passing takes the best-aligned (smallest note shift) is admitted (FR-010). The license policy is
checked before any generation (FR-006); a disallowed license or an unusable backend makes the stage
no-op with a logged skip (FR-013).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from voders.audio import to_mono_float32
from voders.config.models import AccompanimentOptions
from voders.manifest.models import AccompanimentProvenance, ValidationVerdict, VerdictStatus
from voders.render.accompaniment_backend import (
    MODE_LEGO,
    AccompanimentBackend,
    BackendOutput,
)
from voders.scores.models import Score
from voders.seeds import derive_seed
from voders.seeds import rng as rng_for_seed
from voders.validate.mix import validate_on_mix
from voders.validate.timing import TimingRegistry
from voders.validate.validator import Validator

_LOG = logging.getLogger(__name__)
_EPS = 1e-12


@dataclass
class AccompanimentResult:
    """One produced accompaniment sample, ready for the corpus store."""

    mix: np.ndarray
    stem: np.ndarray | None
    verdict: ValidationVerdict
    provenance: AccompanimentProvenance
    admitted: bool


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _pick_snr(target: float | list[float], seed: int) -> float:
    if isinstance(target, list | tuple) and len(target) > 0:
        choices = np.asarray([float(v) for v in target], dtype=np.float64)
        return float(rng_for_seed(seed).choice(choices))
    return float(target) if isinstance(target, int | float) else 12.0


def _mix_under_vocal(
    vocal: np.ndarray, stem: np.ndarray, target_snr_db: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sum a stem under the untouched vocal at a target vocal-to-accompaniment ratio.

    Only the stem is scaled to hit the ratio; the vocal is never re-encoded. The summed mix is
    uniformly scaled down if it would clip — a uniform scale leaves pitch, timing, and the reported
    SNR unchanged, and the bit-identical vocal is retained separately (SC-001). Returns
    ``(mix, scaled_stem, snr_db)``.
    """
    vocal = to_mono_float32(vocal)
    stem = to_mono_float32(stem)
    rms_v, rms_s = _rms(vocal), _rms(stem)
    if rms_v <= _EPS or rms_s <= _EPS:
        return vocal, stem, float("inf")
    desired_rms_s = rms_v / (10.0 ** (target_snr_db / 20.0))
    scaled = stem * (desired_rms_s / rms_s)
    mix = vocal.astype(np.float64) + scaled.astype(np.float64)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    snr_db = 20.0 * np.log10(rms_v / _rms(scaled)) if _rms(scaled) > _EPS else float("inf")
    return to_mono_float32(mix), to_mono_float32(scaled), snr_db


class Accompanist:
    """Generates accompaniment for an accepted vocal, validated on the mix (spec 002)."""

    name = "accompaniment"

    def __init__(
        self,
        backend: AccompanimentBackend,
        options: AccompanimentOptions,
        validator: Validator,
        *,
        timing: TimingRegistry | None = None,
    ) -> None:
        self.backend = backend
        self.options = options
        self.validator = validator
        self.timing = timing

    def skip_reason(self) -> str | None:
        """Why the stage cannot run this config, or None if it can (FR-006, FR-013)."""
        if self.backend.model_license not in self.options.license_policy:
            return (
                f"model license {self.backend.model_license!r} not in policy "
                f"{self.options.license_policy} — skipping (FR-006)"
            )
        if not self.backend.supports(self.options.mode):
            return f"backend {self.backend.name!r} does not support mode {self.options.mode!r}"
        return None

    def _provenance(
        self, *, source_vocal_sample_id: str, takes_tried: int, max_note_shift_ms: float
    ) -> AccompanimentProvenance:
        mode = self.options.mode
        return AccompanimentProvenance(
            mode=mode,
            model_id=self.backend.model_id or self.options.model_id,
            model_version=self.backend.model_version,
            model_license=self.backend.model_license,
            attribution_text=self.backend.attribution_text,
            vocal_bit_exact=(mode == MODE_LEGO),
            source_vocal_sample_id=source_vocal_sample_id,
            takes_tried=takes_tried,
            max_note_shift_ms=max_note_shift_ms,
            free_time=self.options.free_time,
            target_instrument=self.options.target_instrument,
            stem_available=(mode == MODE_LEGO),
        )

    def _one_take(
        self, vocal: np.ndarray, score: Score, seed: int
    ) -> tuple[np.ndarray, np.ndarray | None, float | None]:
        """Generate one take; return (mix, stem_or_None, snr_or_None)."""
        opts: dict[str, object] = {
            "target_instrument": self.options.target_instrument,
            "free_time": self.options.free_time,
            "bpm": self.options.bpm,
            "target_snr_db": self.options.target_snr_db,
        }
        out: BackendOutput = self.backend.generate(vocal, score, self.options.mode, seed, opts)
        if self.options.mode == MODE_LEGO:
            assert out.accompaniment_stem is not None, "lego backend must return a stem"
            snr_target = _pick_snr(self.options.target_snr_db, seed)
            mix, stem, snr = _mix_under_vocal(vocal, out.accompaniment_stem, snr_target)
            return mix, stem, snr
        assert out.mix is not None, "complete backend must return a mix"
        return to_mono_float32(out.mix), None, None

    def generate(
        self,
        vocal: np.ndarray,
        score: Score,
        *,
        source_vocal_sample_id: str,
        base_seed: int,
    ) -> AccompanimentResult:
        """Generate up to ``takes`` takes and return the best-aligned admitted one (FR-010).

        If no take passes the validator, the best-aligned (smallest note shift) failing take is
        returned for the rejected tree, with its verdict reason preserved.
        """
        vocal = to_mono_float32(vocal)
        best_pass: tuple[float, float, np.ndarray, np.ndarray | None, ValidationVerdict] | None = (
            None
        )
        best_fail: tuple[float, np.ndarray, np.ndarray | None, ValidationVerdict] | None = None

        for take in range(self.options.takes):
            seed = derive_seed(base_seed, "take", take)
            mix, stem, snr = self._one_take(vocal, score, seed)
            mv = validate_on_mix(mix, score, self.validator, snr_db=snr, timing=self.timing)
            verdict, shift = mv.verdict, mv.max_note_shift_ms
            snr_key = -(snr if snr is not None else -1e9)  # higher SNR -> smaller key (tie-break)
            if verdict.status == VerdictStatus.ACCEPTED:
                cand = (shift, snr_key, mix, stem, verdict)
                if best_pass is None or (shift, snr_key) < (best_pass[0], best_pass[1]):
                    best_pass = cand
            elif best_fail is None or shift < best_fail[0]:
                best_fail = (shift, mix, stem, verdict)

        takes_tried = self.options.takes
        if best_pass is not None:
            shift, _snr_key, mix, stem, verdict = best_pass
            prov = self._provenance(
                source_vocal_sample_id=source_vocal_sample_id,
                takes_tried=takes_tried,
                max_note_shift_ms=shift,
            )
            return AccompanimentResult(
                mix=mix, stem=stem, verdict=verdict, provenance=prov, admitted=True
            )

        assert best_fail is not None  # at least one take always runs (takes >= 1)
        shift, mix, stem, verdict = best_fail
        prov = self._provenance(
            source_vocal_sample_id=source_vocal_sample_id,
            takes_tried=takes_tried,
            max_note_shift_ms=shift,
        )
        return AccompanimentResult(
            mix=mix, stem=stem, verdict=verdict, provenance=prov, admitted=False
        )
