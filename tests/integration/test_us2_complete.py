"""US2 (spec 002): Complete one-pass mode re-encodes the vocal and emits no separable stem.

End-to-end through ``run_command`` with the fake CPU backend on a single score: the accompaniment
record is ACCEPTED (the fake Complete backend keeps headroom, so the sung notes still land within
tolerance on the mix, SC-002), but the vocal is not bit-exact and no stem is retained.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from voders.cli.run import run_command
from voders.manifest.io import load_manifest
from voders.manifest.models import VerdictStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = REPO_ROOT / "evals" / "fixtures" / "accompaniment-smoke.yaml"


def _write_config(tmp_path: Path, *, mode: str, free_time: bool, bpm: float | None) -> Path:
    config = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config["scores"] = "evals/fixtures/scores/score_000.tsv"
    config["output_root"] = str(tmp_path / "out")
    accomp = config["lanes"]["accompaniment"]
    accomp["mode"] = mode
    accomp["free_time"] = free_time
    accomp["bpm"] = bpm
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_complete_reencodes_vocal_and_emits_no_stem(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, mode="complete", free_time=True, bpm=None)
    out_root = tmp_path / "out"

    assert run_command(str(config_path)) == 0

    records = load_manifest(out_root / "manifest.jsonl")
    accomp = [r for r in records if r.lane == "accompaniment"]
    assert len(accomp) == 1
    rec = accomp[0]

    prov = rec.accompaniment
    assert prov is not None
    assert prov.mode == "complete"
    assert prov.vocal_bit_exact is False
    assert prov.stem_available is False

    # The fake Complete backend keeps headroom -> ACCEPTED with notes within tolerance on the mix.
    assert rec.verdict.status == VerdictStatus.ACCEPTED
    assert rec.verdict.onset_ok is True
    assert rec.verdict.offset_ok is True

    # No separable stem is written for a Complete-mode sample.
    shard = out_root / "corpus" / "shard=000"
    stem_path = shard / f"{rec.sample_id}.stem.wav"
    assert not stem_path.exists()
