"""Issue #9: an external score source's license tag rides into the manifest (FR-008).

A score directory carrying a ``_source.json`` sidecar (as written by ``evals/import_midi.py``) must
have its ``source``/``license`` stamped onto every provenance record derived from those scores, so
ingested corpora stay license-auditable. Scores without a sidecar serialise with empty tags.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from voders.cli.run import run_command
from voders.manifest.io import load_manifest


def _config(tmp_path: Path, scores_dir: Path) -> Path:
    config = {
        "run_id": "midi_lic",
        "master_seed": 20260628,
        "scores": f"{scores_dir}/*.tsv",
        "output_root": str(tmp_path / "out"),
        "voices": [
            {
                "voice_id": "donor_ah_synth",
                "kind": "deterministic_donor",
                "license": "CC0 synthetic vowel",
                "consent_verified": True,
                "model_ref": "evals/fixtures/voices/donor_ah_synth.wav",
            }
        ],
        "lanes": {"deterministic": {"enabled": True, "synth": "world"}},
        "validator": {
            "onset_ms": 50,
            "min_note_ms": None,
            "min_note_percentile": 1,
            "f0_cents": 25,
            "f0_coverage": 0.80,
            "snr_floor_db": 0,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_source_license_stamped_on_records(tmp_path: Path) -> None:
    scores = tmp_path / "scores_ext"
    scores.mkdir()
    # A short singable melody + the license sidecar the importer writes.
    (scores / "tune.tsv").write_text("0.200\t0.700\t60\n0.800\t1.300\t64\n", encoding="utf-8")
    (scores / "_source.json").write_text(
        json.dumps({"source": "POP909", "license": "research-only", "url": "http://x"}),
        encoding="utf-8",
    )

    assert run_command(str(_config(tmp_path, scores))) == 0

    records = load_manifest(tmp_path / "out" / "manifest.jsonl")
    assert records
    assert all(r.score_source == "POP909" for r in records)
    assert all(r.score_license == "research-only" for r in records)


def test_no_sidecar_leaves_tags_empty(tmp_path: Path) -> None:
    scores = tmp_path / "scores_plain"
    scores.mkdir()
    (scores / "tune.tsv").write_text("0.200\t0.700\t60\n0.800\t1.300\t64\n", encoding="utf-8")

    assert run_command(str(_config(tmp_path, scores))) == 0

    records = load_manifest(tmp_path / "out" / "manifest.jsonl")
    assert records
    assert all(r.score_source == "" and r.score_license == "" for r in records)
