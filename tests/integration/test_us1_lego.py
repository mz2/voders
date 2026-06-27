"""US1 (spec 002): Lego accompaniment preserves the vocal bit-exactly and keeps a stem.

End-to-end through ``run_command`` with the fake CPU backend on a single score: the accompaniment
record is ACCEPTED in Lego mode, the source vocal is retained bit-identically (SC-001), and a
separable stem is written alongside the mix (FR-015, SC-008).
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
    """Single-score variant of the smoke config under ``tmp_path`` (fast: one base + one accomp)."""
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


def test_lego_preserves_vocal_and_keeps_stem(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, mode="lego", free_time=True, bpm=None)
    out_root = tmp_path / "out"

    assert run_command(str(config_path)) == 0

    records = load_manifest(out_root / "manifest.jsonl")
    accomp = [r for r in records if r.lane == "accompaniment"]
    assert len(accomp) == 1
    rec = accomp[0]

    # Base deterministic render: the record whose sample_id has no "_accomp" suffix.
    base = next(r for r in records if r.lane != "accompaniment" and "_accomp" not in r.sample_id)

    prov = rec.accompaniment
    assert prov is not None
    assert rec.verdict.status == VerdictStatus.ACCEPTED
    assert prov.mode == "lego"
    assert prov.vocal_bit_exact is True
    assert prov.stem_available is True
    assert prov.source_vocal_sample_id == base.sample_id

    # The separable Lego stem is written next to the mix under corpus/shard=000/.
    shard = out_root / "corpus" / "shard=000"
    stem_path = shard / f"{rec.sample_id}.stem.wav"
    assert stem_path.exists()

    # SC-001 spirit: the retained source vocal is the base render's wav, which must exist.
    base_wav = shard / f"{base.sample_id}.wav"
    assert base_wav.exists()
