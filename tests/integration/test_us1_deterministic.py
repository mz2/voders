"""US1 acceptance scenarios for the deterministic singing corpus lane (T016).

Parse a Klangio-format ``.tsv`` score, render it with the deterministic lane, and verify the
consumer audio format, the f0 alignment gate, the short-note safeguard, and polyphony rejection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voders.config.models import ValidatorConfig
from voders.constants import SAMPLE_RATE
from voders.manifest.models import VerdictStatus
from voders.render.base import RenderRequest
from voders.render.deterministic import DeterministicLane
from voders.scores.parse import PolyphonyError, parse_tsv
from voders.validate.validator import Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"
EDGE_DIR = REPO_ROOT / "evals" / "fixtures" / "edge"


def _render(score, voice):
    return DeterministicLane().render(RenderRequest(score=score, voice=voice, seed=0)).audio


def test_scenario1_render_produces_consumer_format_audio_and_roundtrips(donor_ah) -> None:
    """Scenario 1: parse a fixture score, render it, get 22050 Hz mono float32 audio.

    The orchestrator copies the source bytes verbatim, so parse_tsv must round-trip the raw bytes.
    """
    parsed = parse_tsv(SCORES_DIR / "score_000.tsv")
    assert parsed.raw_bytes == parsed.source_path.read_bytes()

    audio = _render(parsed.score, donor_ah)

    assert audio.dtype == np.float32  # consumer format: float32
    assert audio.ndim == 1  # mono
    assert audio.size > 0
    # length corresponds to score duration at the fixed 22,050 Hz sample rate.
    assert abs(audio.size - parsed.score.duration_s * SAMPLE_RATE) <= 0.02 * SAMPLE_RATE


def test_scenario2_f0_within_tolerance(donor_ah) -> None:
    """Scenario 2: rendered f0 stays within ±25 cents over ≥80% of each note -> f0_ok True."""
    parsed = parse_tsv(SCORES_DIR / "score_000.tsv")
    audio = _render(parsed.score, donor_ah)

    verdict = Validator(ValidatorConfig(), min_note_ms=50.0).validate(audio, parsed.score)
    assert verdict.f0_ok is True


def test_scenario3_short_note_is_flagged_or_rejected(donor_ah) -> None:
    """Scenario 3: a note shorter than min_note_ms must never be silently accepted."""
    parsed = parse_tsv(EDGE_DIR / "short_note.tsv")
    assert any(n.duration_ms < 50.0 for n in parsed.score.notes)  # the ~20 ms note

    audio = _render(parsed.score, donor_ah)
    verdict = Validator(ValidatorConfig(), min_note_ms=50.0).validate(audio, parsed.score)

    assert verdict.status in {VerdictStatus.FLAGGED, VerdictStatus.REJECTED}
    assert verdict.status != VerdictStatus.ACCEPTED
    assert verdict.reason is not None
    assert "short" in verdict.reason.lower() or "min_note_ms" in verdict.reason


def test_polyphony_score_is_rejected_by_parser() -> None:
    """Overlapping notes are not valid monophonic singing: parse_tsv raises PolyphonyError."""
    with pytest.raises(PolyphonyError):
        parse_tsv(EDGE_DIR / "polyphony.tsv")
