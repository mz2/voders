"""US4 integration: expressive neural-SVS lane with a label safety net (T036, FR-007, SC-010).

SVS = singing-voice synthesis; MFA = forced alignment (re-derives note boundaries from audio).
Three scenarios:

1. ``rederive_labels`` with modest timing humanization carries a re-derived ``.tsv`` and records the
   onset deviation; validating the audio against the *re-derived* labels passes (onset_ok).
2. A large humanization that drives an onset past the 50 ms tolerance is forced to REJECTED, end to
   end through the orchestrator and its manifest (SC-010).
3. ``force_score_f0`` behaves like the deterministic P1 lane: label == input and ACCEPTED.
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
from voders.render.svs import SvsLane
from voders.scores.models import Note, Score
from voders.validate.validator import Validator
from voders.voices.models import Voice, VoiceKind

REPO_ROOT = Path(__file__).resolve().parents[2]
DONOR_AH = REPO_ROOT / "evals" / "fixtures" / "voices" / "donor_ah_synth.wav"


def _svs_voice() -> Voice:
    return Voice(
        voice_id="svs_ah",
        kind=VoiceKind.SVS_VOICEBANK,
        license="CC0 synthetic vowel",
        consent_verified=True,
        model_ref=str(DONOR_AH),
    )


def _score() -> Score:
    """Comfortably-long notes with wide gaps so humanized onsets never merge segments."""
    return Score(
        score_id="tiny",
        source="test",
        notes=[
            Note(onset_s=0.3, offset_s=0.9, pitch_midi=60),
            Note(onset_s=1.4, offset_s=2.0, pitch_midi=64),
            Note(onset_s=2.5, offset_s=3.1, pitch_midi=67),
        ],
    )


def test_rederived_labels_carried_and_validate_against_audio() -> None:
    """Modest humanize: re-derived labels differ from input, deviation recorded, audio validates."""
    score = _score()
    lane = SvsLane({"mode": "rederive_labels", "backend": "cpu", "humanize_ms": 20.0})
    result = lane.render(RenderRequest(score=score, voice=_svs_voice(), seed=3))

    # The carried label is re-derived from the audio, not the input score.
    assert result.label_score != score
    dev = result.notes["max_onset_dev_ms"]
    assert isinstance(dev, float) and dev >= 0.0
    assert dev <= 50.0  # modest humanize stays within tolerance

    # Validating the audio against the re-derived labels passes (the labels match the audio).
    verdict = Validator(ValidatorConfig()).validate(result.audio, result.label_score)
    assert verdict.onset_ok is True


def test_large_onset_deviation_is_rejected_end_to_end(tmp_path: Path) -> None:
    """humanize > 50 ms drift -> lane forces REJECTED and the manifest records it (SC-010)."""
    score_path = tmp_path / "score_big.tsv"
    score_path.write_text("0.300\t0.900\t60\n1.400\t2.000\t64\n2.500\t3.100\t67\n")

    config = RunConfig(
        run_id="us4_reject",
        master_seed=2026,
        scores=str(score_path),
        output_root=str(tmp_path / "out"),
        voices=[_svs_voice()],
        lanes={
            "svs": LaneToggle(
                enabled=True, mode="rederive_labels", backend="cpu", humanize_ms=120.0
            )
        },
    )
    lanes = build_lanes(config)
    summary = Orchestrator(config, lanes).run()
    assert summary.attempted == 1

    records = load_manifest(Path(config.output_root) / "manifest.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record.lane == "svs"
    assert record.verdict.status == VerdictStatus.REJECTED
    assert record.notes.get("max_onset_dev_ms", 0.0) > 50.0


def test_force_score_f0_behaves_like_p1() -> None:
    """force_score_f0: label == input score and the sample validates to ACCEPTED."""
    score = _score()
    lane = SvsLane({"mode": "force_score_f0", "backend": "cpu"})
    result = lane.render(RenderRequest(score=score, voice=_svs_voice(), seed=5))

    assert result.label_score == score
    verdict = Validator(ValidatorConfig()).validate(result.audio, result.label_score)
    assert verdict.status == VerdictStatus.ACCEPTED


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
