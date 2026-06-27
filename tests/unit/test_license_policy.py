"""Unit tests for accompaniment license policy + attribution (FR-006, FR-006a, FR-013).

The license policy is checked before any generation: a model whose license is outside the policy
makes the stage a logged no-op (``skip_reason`` returns a string). A backend that cannot serve the
requested mode is likewise skipped. For admitted samples, CC-BY-class attribution and the model
license propagate into the provenance record.
"""

from __future__ import annotations

import numpy as np

from voders.config.models import AccompanimentOptions, ValidatorConfig
from voders.render.accompaniment import Accompanist
from voders.render.accompaniment_backend import (
    MODE_COMPLETE,
    MODE_LEGO,
    BackendOutput,
    FakeAccompanimentBackend,
)
from voders.render.deterministic import midi_to_hz
from voders.scores.models import Note, Score
from voders.validate.validator import Validator

SR = 22_050


def _vocal_and_score() -> tuple[np.ndarray, Score]:
    """A 1-note (~0.6 s) sine 'vocal' at MIDI 60 in a ~1.0 s buffer at 22,050 Hz."""
    onset_s, offset_s, pitch = 0.2, 0.8, 60
    total = np.zeros(SR, dtype=np.float32)  # ~1.0 s buffer
    start, end = int(onset_s * SR), int(offset_s * SR)
    t = np.arange(end - start) / SR
    total[start:end] = 0.3 * np.sin(2.0 * np.pi * midi_to_hz(pitch) * t).astype(np.float32)
    score = Score(
        score_id="score_license",
        notes=[Note(onset_s=onset_s, offset_s=offset_s, pitch_midi=pitch)],
    )
    return total, score


class AttributionBackend:
    """A CC-BY-licensed backend with a required attribution; audio delegated to the fake backend."""

    name = "attrib"
    model_id = "attrib-model"
    model_version = "1"
    model_license = "CC-BY-4.0"
    attribution_text: str | None = "Credit: X"

    def __init__(self) -> None:
        self._inner = FakeAccompanimentBackend()

    def requires_gpu(self) -> bool:
        return False

    def supports(self, mode: str) -> bool:
        return self._inner.supports(mode)

    def generate(
        self,
        vocal: np.ndarray,
        score: Score,
        mode: str,
        seed: int,
        options: dict[str, object],
    ) -> BackendOutput:
        return self._inner.generate(vocal, score, mode, seed, options)


class LegoOnlyBackend:
    """An MIT backend that supports only Lego mode (rejects 'complete')."""

    name = "lego-only"
    model_id = "lego-only-model"
    model_version = "0"
    model_license = "MIT"
    attribution_text: str | None = None

    def requires_gpu(self) -> bool:
        return False

    def supports(self, mode: str) -> bool:
        return mode == MODE_LEGO  # False for MODE_COMPLETE

    def generate(
        self,
        vocal: np.ndarray,
        score: Score,
        mode: str,
        seed: int,
        options: dict[str, object],
    ) -> BackendOutput:
        return FakeAccompanimentBackend().generate(vocal, score, mode, seed, options)


def test_skip_reason_none_when_license_in_policy() -> None:
    """An MIT backend with MIT in the policy can run: no skip reason."""
    options = AccompanimentOptions(license_policy=["MIT", "Apache-2.0", "CC-BY-4.0"])
    accompanist = Accompanist(FakeAccompanimentBackend(), options, Validator(ValidatorConfig()))

    assert accompanist.skip_reason() is None


def test_skip_reason_set_when_license_outside_policy() -> None:
    """An MIT backend excluded by an Apache-only policy is skipped with a reason (FR-006)."""
    options = AccompanimentOptions(license_policy=["Apache-2.0"])
    accompanist = Accompanist(FakeAccompanimentBackend(), options, Validator(ValidatorConfig()))

    reason = accompanist.skip_reason()

    assert reason is not None
    assert "license" in reason
    assert "policy" in reason


def test_skip_reason_set_when_mode_unsupported() -> None:
    """A backend that cannot serve the requested mode is skipped (FR-013)."""
    options = AccompanimentOptions(mode=MODE_COMPLETE, license_policy=["MIT"])
    accompanist = Accompanist(LegoOnlyBackend(), options, Validator(ValidatorConfig()))

    reason = accompanist.skip_reason()

    assert reason is not None
    assert MODE_COMPLETE in reason


def test_attribution_and_license_propagate_to_provenance() -> None:
    """A CC-BY model's attribution and license are carried into provenance (FR-006a)."""
    vocal, score = _vocal_and_score()
    options = AccompanimentOptions(
        mode="lego", takes=1, target_snr_db=12.0, license_policy=["MIT", "CC-BY-4.0"]
    )
    accompanist = Accompanist(AttributionBackend(), options, Validator(ValidatorConfig()))

    assert accompanist.skip_reason() is None
    result = accompanist.generate(vocal, score, source_vocal_sample_id="vocal-1", base_seed=7)

    assert result.provenance.attribution_text == "Credit: X"
    assert result.provenance.model_license == "CC-BY-4.0"
