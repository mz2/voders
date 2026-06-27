"""Unit tests for the JSON Lines manifest writer/reader (T048, FR-008)."""

from __future__ import annotations

from pathlib import Path

from voders.manifest.io import ManifestWriter, load_manifest
from voders.manifest.models import ProvenanceRecord, ValidationVerdict, VerdictStatus


def _record(sample_id: str, status: VerdictStatus = VerdictStatus.ACCEPTED) -> ProvenanceRecord:
    return ProvenanceRecord(
        sample_id=sample_id,
        score_id="score_000",
        score_path=f"corpus/shard=000/{sample_id}/score.tsv",
        audio_path=f"corpus/shard=000/{sample_id}/audio.wav",
        lane="deterministic",
        voice_id="donor_ah",
        seed=12345,
        voice_license="CC0 synthetic vowel",
        consent_verified=True,
        config_hash="sha256:deadbeef",
        verdict=ValidationVerdict(
            status=status, onset_ok=True, offset_ok=True, f0_ok=True, max_onset_dev_ms=3.5
        ),
        notes={"max_onset_dev_ms": 3.5},
    )


def test_writer_append_and_load_roundtrip_preserves_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    original = _record("score_000_singer_donor_ah")
    with ManifestWriter(path) as writer:
        writer.append(original)

    records = load_manifest(path)
    assert len(records) == 1
    loaded = records[0]
    assert loaded == original
    assert loaded.seed == 12345
    assert loaded.voice_license == "CC0 synthetic vowel"
    assert loaded.consent_verified is True
    assert loaded.verdict.status == VerdictStatus.ACCEPTED
    assert loaded.verdict.max_onset_dev_ms == 3.5
    assert loaded.notes == {"max_onset_dev_ms": 3.5}


def test_scanning_yields_records_in_order(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    ids = ["a", "b", "c", "d"]
    writer = ManifestWriter(path)
    for i in ids:
        writer.append(_record(i))

    records = load_manifest(path)
    assert [r.sample_id for r in records] == ids


def test_writer_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "manifest.jsonl"
    ManifestWriter(path).append(_record("x"))
    assert path.exists()
    assert len(load_manifest(path)) == 1
