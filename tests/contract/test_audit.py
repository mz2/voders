"""Contract tests for ``voders audit`` (T041, FR-011, SC-008).

The audit gate fails iff an unconsented voice reached the *accepted* corpus. A logged
``license_refused`` record (consent missing, no audio) is the expected, passing outcome.
Every manifest line must validate against the schema mirrored by ``ProvenanceRecord``.
"""

from __future__ import annotations

from pathlib import Path

from voders.cli.audit import audit_command
from voders.manifest.io import ManifestWriter
from voders.manifest.models import ProvenanceRecord, ValidationVerdict, VerdictStatus


def _record(
    sample_id: str,
    *,
    status: VerdictStatus,
    consent_verified: bool,
    voice_id: str = "donor_ah",
) -> ProvenanceRecord:
    accepted = status == VerdictStatus.ACCEPTED
    return ProvenanceRecord(
        sample_id=sample_id,
        score_id="score_000",
        score_path="corpus/shard=000/x.tsv" if accepted else "",
        audio_path="corpus/shard=000/x.wav" if accepted else "",
        lane="deterministic",
        voice_id=voice_id,
        seed=1,
        voice_license="CC0 synthetic vowel" if consent_verified else "unknown",
        consent_verified=consent_verified,
        config_hash="sha256:abc",
        verdict=ValidationVerdict(status=status, onset_ok=accepted, f0_ok=accepted),
    )


def _write(path: Path, records: list[ProvenanceRecord]) -> None:
    writer = ManifestWriter(path)
    for r in records:
        writer.append(r)


def test_audit_passes_when_no_unconsented_voice_is_accepted(tmp_path: Path, capsys) -> None:
    """ACCEPTED+consented alongside a logged LICENSE_REFUSED (unconsented) -> exit 0."""
    path = tmp_path / "manifest.jsonl"
    _write(
        path,
        [
            _record(
                "ok", status=VerdictStatus.ACCEPTED, consent_verified=True, voice_id="donor_ah"
            ),
            _record(
                "refused",
                status=VerdictStatus.LICENSE_REFUSED,
                consent_verified=False,
                voice_id="no_consent",
            ),
        ],
    )
    assert audit_command(str(path)) == 0
    assert "AUDIT PASSED" in capsys.readouterr().out


def test_audit_fails_when_unconsented_voice_reaches_accepted(tmp_path: Path, capsys) -> None:
    """An ACCEPTED record with consent_verified=False fails the audit (SC-008) -> exit 1."""
    path = tmp_path / "manifest.jsonl"
    _write(
        path,
        [
            _record(
                "leak",
                status=VerdictStatus.ACCEPTED,
                consent_verified=False,
                voice_id="no_consent",
            ),
        ],
    )
    assert audit_command(str(path)) == 1
    assert "AUDIT FAILED" in capsys.readouterr().out


def test_manifest_lines_validate_against_schema(tmp_path: Path) -> None:
    """Every line round-trips through ProvenanceRecord.model_validate_json without error."""
    path = tmp_path / "manifest.jsonl"
    _write(
        path,
        [
            _record("ok", status=VerdictStatus.ACCEPTED, consent_verified=True),
            _record("refused", status=VerdictStatus.LICENSE_REFUSED, consent_verified=False),
        ],
    )
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    for line in lines:
        # Mirrors the manifest schema exactly (extra="forbid"): raises on any drift.
        ProvenanceRecord.model_validate_json(line)
