"""US3 acceptance scenarios for the label-safe augmentation chain (T030, FR-005, FR-014).

Reverb + codec + accompaniment must keep the vocal dominant and the labels intact, the validator
must quarantine low-SNR mixes and reject clipping, and an orchestrated run must emit augmented
provenance records whose ``.tsv`` is byte-identical to the base sample's (labels untouched).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voders.config.models import (
    AugmentationProfileConfig,
    LaneToggle,
    RunConfig,
    ValidatorConfig,
)
from voders.manifest.io import load_manifest
from voders.manifest.models import VerdictStatus
from voders.render.augment import AugmentationChain
from voders.render.base import RenderRequest
from voders.render.deterministic import DeterministicLane
from voders.render.registry import build_augmentor, build_lanes
from voders.validate.validator import Validator
from voders.voices.models import Voice, VoiceKind

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "evals" / "fixtures"
DONOR_AH = FIXTURES / "voices" / "donor_ah_synth.wav"
SCORES_DIR = FIXTURES / "scores"


def _base_render(small_score, donor_ah):
    result = DeterministicLane().render(RenderRequest(score=small_score, voice=donor_ah, seed=0))
    return result.audio, result.label_score


def _normal_profile() -> AugmentationProfileConfig:
    return AugmentationProfileConfig(
        profile_id="pop_mix",
        steps=["reverb_ir", "codec", "accompaniment_mix"],
        params={"snr_db": [12.0]},
    )


def test_reverb_codec_accompaniment_preserves_labels_and_passes(small_score, donor_ah) -> None:
    """A normal profile keeps onsets/offsets aligned and the sample stays ACCEPTED (FR-005)."""
    audio, label_score = _base_render(small_score, donor_ah)
    out, snr_db = AugmentationChain([_normal_profile()]).apply(audio, "pop_mix", seed=3)

    verdict = Validator(ValidatorConfig(snr_floor_db=0.0), min_note_ms=50.0).validate(
        out, label_score, snr_db=snr_db
    )
    assert verdict.onset_ok is True
    assert verdict.offset_ok is True
    assert verdict.status == VerdictStatus.ACCEPTED


def test_vocal_stays_dominant_for_positive_snr_profile(small_score, donor_ah) -> None:
    """A positive-SNR profile leaves the vocal louder than the accompaniment (snr_db > 0)."""
    audio, _ = _base_render(small_score, donor_ah)
    _out, snr_db = AugmentationChain([_normal_profile()]).apply(audio, "pop_mix", seed=3)
    assert snr_db is not None
    assert snr_db > 0.0


def test_clipping_is_rejected_never_silently_admitted(small_score, donor_ah) -> None:
    """Audio that exceeds full scale must be rejected by the validator's clipping gate (US3 #3)."""
    audio, label_score = _base_render(small_score, donor_ah)
    peak = float(np.max(np.abs(audio)))
    clipped = (audio * (2.0 / peak)).astype(np.float32)  # peak ~2.0 -> clips
    assert float(np.max(np.abs(clipped))) > 1.0

    verdict = Validator(ValidatorConfig(), min_note_ms=50.0).validate(clipped, label_score)
    assert verdict.status == VerdictStatus.REJECTED


def test_orchestrated_run_emits_augmented_records_with_identical_labels(tmp_path) -> None:
    """End-to-end: deterministic + augmentation lanes; augmented .tsv == base .tsv (labels kept)."""
    config = RunConfig(
        run_id="us3-aug",
        master_seed=1234,
        scores=str(SCORES_DIR / "score_000.tsv"),
        output_root=str(tmp_path),
        voices=[
            Voice(
                voice_id="donor_ah_synth",
                kind=VoiceKind.DETERMINISTIC_DONOR,
                license="CC0 synthetic vowel",
                consent_verified=True,
                model_ref=str(DONOR_AH),
            )
        ],
        lanes={
            "deterministic": LaneToggle(enabled=True),
            "augmentation": LaneToggle(enabled=True),
        },
        augmentation_profiles=[_normal_profile()],
        validator=ValidatorConfig(min_note_ms=1.0),
    )

    # Construct lazily, exactly as the CLI does.
    from voders.corpus.orchestrator import Orchestrator

    orch = Orchestrator(
        config,
        lanes=build_lanes(config),
        augmentor=build_augmentor(config),
    )
    summary = orch.run()
    assert summary.attempted >= 2  # base + at least one augmented sample

    records = load_manifest(tmp_path / "manifest.jsonl")
    aug = [r for r in records if r.lane == "augmentation"]
    base = [r for r in records if r.lane == "deterministic"]
    assert aug, "expected at least one augmented provenance record"
    assert base, "expected the base deterministic record"
    for r in aug:
        assert r.augmentation_profile == "pop_mix"

    # The augmented sample's labels are byte-identical to the base sample's (FR-005).
    base_tsv = (tmp_path / base[0].score_path).read_bytes()
    aug_tsv = (tmp_path / aug[0].score_path).read_bytes()
    assert aug_tsv == base_tsv
