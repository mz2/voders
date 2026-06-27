"""US3 (spec 002): free-time accompaniment carries no fixed tempo and stays note-aligned.

End-to-end through ``run_command`` in Lego mode with ``free_time: true`` / ``bpm: null``: the
accompaniment provenance records ``free_time`` true, keeps the measured note displacement within
tolerance (SC-005), and carries no fixed-tempo (``bpm``) metadata.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from voders.cli.run import run_command
from voders.manifest.io import load_manifest

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


def test_free_time_is_recorded_with_no_fixed_tempo(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, mode="lego", free_time=True, bpm=None)
    out_root = tmp_path / "out"

    assert run_command(str(config_path)) == 0

    records = load_manifest(out_root / "manifest.jsonl")
    accomp = [r for r in records if r.lane == "accompaniment"]
    assert len(accomp) == 1
    prov = accomp[0].accompaniment
    assert prov is not None

    assert prov.free_time is True
    # SC-005: measured note displacement on the mix stays within tolerance.
    assert prov.max_note_shift_ms <= 50.0
    # Free-time samples carry no fixed-tempo metadata: there is no ``bpm`` field in provenance.
    assert "bpm" not in prov.model_dump()
