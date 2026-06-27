"""US4 (spec 002): the license/consent audit passes for the in-policy fake accompaniment.

End-to-end through ``run_command`` then ``audit_command`` over the manifest: the fake backend is
MIT (in policy, no attribution needed), so the audit returns 0 (FR-011, SC-008), and every accepted
accompaniment record carries a non-empty, commercially-usable license.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from voders.cli.audit import audit_command
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


def test_audit_passes_for_in_policy_accompaniment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path, mode="lego", free_time=True, bpm=None)
    out_root = tmp_path / "out"

    assert run_command(str(config_path)) == 0
    manifest_path = out_root / "manifest.jsonl"

    assert audit_command(str(manifest_path)) == 0
    out = capsys.readouterr().out
    assert "AUDIT PASSED" in out

    records = load_manifest(manifest_path)
    accepted_accomp = [
        r
        for r in records
        if r.lane == "accompaniment"
        and r.accompaniment is not None
        and r.verdict.status == VerdictStatus.ACCEPTED
    ]
    assert accepted_accomp  # at least one accepted accompaniment sample
    for r in accepted_accomp:
        prov = r.accompaniment
        assert prov.model_id  # non-empty
        assert prov.model_license  # non-empty
        assert "-NC" not in prov.model_license.upper()  # commercially usable
