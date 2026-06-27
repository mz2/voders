"""Contract tests for the voice-conversion lane (T023, FR-004, US2).

Voice conversion changes timbre while keeping the score's f0 (``auto_predict_f0=False``), so the
label is the input score unchanged. The CPU ``world`` backend must never import ``torch``.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from voders.render.base import RenderRequest
from voders.render.voiceconv import VoiceConversionLane
from voders.scores.models import Note, Score
from voders.voices.models import Voice, VoiceKind
from voders.voices.registry import ConsentRefusedError, VoiceRegistry


def _small_score() -> Score:
    return Score(
        score_id="tiny",
        source="test",
        notes=[
            Note(onset_s=0.2, offset_s=0.7, pitch_midi=60),
            Note(onset_s=0.8, offset_s=1.3, pitch_midi=64),
            Note(onset_s=1.4, offset_s=1.9, pitch_midi=67),
        ],
    )


def test_world_backend_preserves_score_and_audio_format(donor_oo: Voice) -> None:
    """world backend: label_score == input (f0 preserved); audio is 22050 Hz mono float32."""
    lane = VoiceConversionLane()  # default backend == "world"
    assert lane.requires_gpu() is False

    score = _small_score()
    result = lane.render(RenderRequest(score=score, voice=donor_oo, seed=7))

    # auto_predict_f0=False semantics: the score (hence f0/onset/offset) is unchanged, row-for-row.
    assert result.label_score == score
    assert result.label_score.notes == score.notes

    audio = result.audio
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1  # mono
    assert audio.size > 0
    assert result.notes.get("backend") == "world"


def test_rvc_backend_runs_out_of_process_and_requires_a_model() -> None:
    """rvc is an out-of-process backend (core needs no GPU) and requires a consented model_ref."""
    lane = VoiceConversionLane({"backend": "rvc"})
    assert lane.requires_gpu() is False  # conversion runs in the backends/rvc subprocess

    voice = Voice(
        voice_id="vc_rvc",
        kind=VoiceKind.VOICE_CONVERSION,
        license="CC0",
        consent_verified=True,
        model_ref="",  # no consented target model -> fail fast before any rendering/subprocess
    )
    with pytest.raises(RuntimeError, match="model"):
        lane.render(RenderRequest(score=_small_score(), voice=voice, seed=1))


def test_consent_gate_refuses_unconsented_voice() -> None:
    """The orchestrator calls require_usable before render; unconsented voices are refused."""
    voice = Voice(
        voice_id="no_consent",
        kind=VoiceKind.VOICE_CONVERSION,
        license="unknown",
        consent_verified=False,
    )
    registry = VoiceRegistry([voice])
    with pytest.raises(ConsentRefusedError):
        registry.require_usable("no_consent")


def test_importing_module_does_not_import_torch() -> None:
    """The CPU baseline must not pull in torch at module load (FR-009)."""
    sys.modules.pop("torch", None)
    importlib.reload(importlib.import_module("voders.render.voiceconv"))
    assert "torch" not in sys.modules
