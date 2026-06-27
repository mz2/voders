"""US2 integration: voice-conversion timbre fan-out + consent gate (T024, FR-004, FR-011).

One score is rendered across several consented VOICE_CONVERSION voices; every variant keeps the
source score row-for-row and validates to accepted quality. An unconsented voice is skipped and
logged as ``license_refused`` in the manifest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voders.config.models import LaneToggle, RunConfig, ValidatorConfig
from voders.corpus.orchestrator import Orchestrator
from voders.manifest.io import load_manifest
from voders.manifest.models import VerdictStatus
from voders.render.base import RenderRequest
from voders.render.registry import build_lanes
from voders.render.voiceconv import VoiceConversionLane
from voders.scores.models import Note, Score
from voders.validate.validator import Validator
from voders.voices.models import Voice, VoiceKind

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "evals" / "fixtures"
DONOR_OO = FIXTURES / "voices" / "donor_oo_synth.wav"
DONOR_AH = FIXTURES / "voices" / "donor_ah_synth.wav"
SCORE_000 = "evals/fixtures/scores/score_000.tsv"


def _score() -> Score:
    return Score(
        score_id="tiny",
        source="test",
        notes=[
            Note(onset_s=0.2, offset_s=0.7, pitch_midi=60),
            Note(onset_s=0.8, offset_s=1.3, pitch_midi=64),
            Note(onset_s=1.4, offset_s=1.9, pitch_midi=67),
        ],
    )


def _vc_voice(voice_id: str, model_ref: Path, *, consent: bool = True) -> Voice:
    return Voice(
        voice_id=voice_id,
        kind=VoiceKind.VOICE_CONVERSION,
        license="CC0 synthetic vowel",
        consent_verified=consent,
        model_ref=str(model_ref),
    )


def test_timbre_fanout_preserves_score_and_validates() -> None:
    """Fan one score across two consented VC voices; each keeps the score and validates accepted."""
    score = _score()
    voices = [
        _vc_voice("vc_oo", DONOR_OO),
        _vc_voice("vc_ah", DONOR_AH),
    ]
    lane = VoiceConversionLane({"backend": "world"})
    validator = Validator(ValidatorConfig())

    for voice in voices:
        result = lane.render(RenderRequest(score=score, voice=voice, seed=42))
        # label_score identical row-for-row to the source score.
        assert result.label_score == score
        assert result.label_score.notes == score.notes

        verdict = validator.validate(result.audio, result.label_score, f0_method="pyin_f0")
        assert verdict.onset_ok and verdict.offset_ok and verdict.f0_ok
        assert verdict.status == VerdictStatus.ACCEPTED


def test_unconsented_voice_skipped_and_logged(tmp_path: Path) -> None:
    """Unconsented voice -> license_refused in the manifest; consented voices accepted."""
    config = RunConfig(
        run_id="us2_consent",
        master_seed=1234,
        scores=SCORE_000,
        output_root=str(tmp_path),
        voices=[
            _vc_voice("vc_oo", DONOR_OO),
            _vc_voice("vc_ah", DONOR_AH),
            _vc_voice("vc_blocked", DONOR_OO, consent=False),
        ],
        lanes={"voice_conversion": LaneToggle(enabled=True)},
    )
    lanes = build_lanes(config)
    summary = Orchestrator(config, lanes).run()
    assert summary.attempted == 3

    records = load_manifest(tmp_path / "manifest.jsonl")
    by_voice = {r.voice_id: r for r in records}

    # Unconsented voice: license_refused, no audio written.
    blocked = by_voice["vc_blocked"]
    assert blocked.verdict.status == VerdictStatus.LICENSE_REFUSED

    # Consented voices produced accepted samples sharing the same score_id.
    consented = [by_voice["vc_oo"], by_voice["vc_ah"]]
    score_ids = {r.score_id for r in consented}
    assert len(score_ids) == 1
    for r in consented:
        assert r.verdict.status == VerdictStatus.ACCEPTED
        assert r.consent_verified is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
