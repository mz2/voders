"""Shared pytest fixtures for the voders test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from voders.scores.models import Note, Score
from voders.voices.models import Voice, VoiceKind

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "evals" / "fixtures"
DONOR_AH = FIXTURES / "voices" / "donor_ah_synth.wav"
DONOR_OO = FIXTURES / "voices" / "donor_oo_synth.wav"
SCORES_DIR = FIXTURES / "scores"
EDGE_DIR = FIXTURES / "edge"


@pytest.fixture(scope="session")
def donor_ah() -> Voice:
    return Voice(
        voice_id="donor_ah_synth",
        kind=VoiceKind.DETERMINISTIC_DONOR,
        license="CC0 synthetic vowel",
        consent_verified=True,
        model_ref=str(DONOR_AH),
    )


@pytest.fixture(scope="session")
def donor_oo() -> Voice:
    return Voice(
        voice_id="donor_oo_synth",
        kind=VoiceKind.VOICE_CONVERSION,
        license="CC0 synthetic vowel",
        consent_verified=True,
        model_ref=str(DONOR_OO),
    )


@pytest.fixture
def small_score() -> Score:
    """A short monophonic melody with comfortably-long notes."""
    return Score(
        score_id="tiny",
        source="test",
        notes=[
            Note(onset_s=0.2, offset_s=0.7, pitch_midi=60),
            Note(onset_s=0.8, offset_s=1.3, pitch_midi=64),
            Note(onset_s=1.4, offset_s=1.9, pitch_midi=67),
        ],
    )
