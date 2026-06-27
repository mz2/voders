"""US5 acceptance: provenance/audit and isolated reproduction (T042, FR-013, SC-009).

Run a small deterministic corpus end-to-end, then (a) assert the manifest exposes the full
provenance chain and (b) re-render one accepted sample in isolation from its derived seed and
assert a bit-exact match.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from voders.config.models import LaneToggle, RunConfig
from voders.corpus.orchestrator import Orchestrator
from voders.corpus.reproduce import reproduce_sample
from voders.manifest.io import load_manifest
from voders.manifest.models import ProvenanceRecord, VerdictStatus
from voders.render.registry import build_lanes
from voders.seeds import sample_seed
from voders.validate.validator import Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"

MASTER_SEED = 20260627


def _config(tmp_path: Path, voice) -> RunConfig:
    """A one-score, one-voice deterministic run rooted under tmp_path."""
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    shutil.copy(SCORES_DIR / "score_000.tsv", scores_dir / "score_000.tsv")
    return RunConfig(
        run_id="us5",
        master_seed=MASTER_SEED,
        scores=str(scores_dir / "*.tsv"),
        output_root=str(tmp_path / "out"),
        voices=[voice],
        lanes={"deterministic": LaneToggle(enabled=True)},
    )


def test_manifest_exposes_full_provenance_chain(tmp_path: Path, donor_ah) -> None:
    config = _config(tmp_path, donor_ah)
    summary = Orchestrator(config, build_lanes(config)).run()
    assert summary.attempted == 1
    assert summary.accepted == 1

    records = load_manifest(Path(config.output_root) / "manifest.jsonl")
    assert len(records) == 1
    rec = records[0]

    # Full provenance chain: renderer/lane, voice, seed, license, consent, config_hash, verdict.
    assert rec.lane == "deterministic"
    assert rec.voice_id == donor_ah.voice_id
    assert rec.score_id == "score_000"
    assert rec.seed == sample_seed(MASTER_SEED, "score_000", donor_ah.voice_id, "deterministic")
    assert rec.voice_license == donor_ah.license
    assert rec.consent_verified is True
    assert rec.config_hash.startswith("sha256:")
    assert rec.verdict.status == VerdictStatus.ACCEPTED
    assert rec.audio_path and rec.score_path

    # The record validates against the manifest schema (mirrored by ProvenanceRecord).
    assert ProvenanceRecord.model_validate(rec.model_dump()) == rec


def test_seed_derivation_is_stable(tmp_path: Path, donor_ah) -> None:
    """FR-013: the recorded seed is exactly the derived per-sample seed."""
    config = _config(tmp_path, donor_ah)
    Orchestrator(config, build_lanes(config)).run()
    rec = load_manifest(Path(config.output_root) / "manifest.jsonl")[0]
    expected = sample_seed(MASTER_SEED, rec.score_id, rec.voice_id, "deterministic")
    assert rec.seed == expected


def test_reproduce_accepted_sample_is_bit_exact(tmp_path: Path, donor_ah) -> None:
    """SC-009: re-render one accepted deterministic sample in isolation, bit-exact match."""
    config = _config(tmp_path, donor_ah)
    Orchestrator(config, build_lanes(config)).run()
    rec = load_manifest(Path(config.output_root) / "manifest.jsonl")[0]
    assert rec.verdict.status == VerdictStatus.ACCEPTED

    lanes = build_lanes(config)
    validator = Validator(config.validator)
    result = reproduce_sample(config.output_root, rec, config, lanes, validator)

    assert result.mode == "bit_exact"
    assert result.matched is True, result.detail
