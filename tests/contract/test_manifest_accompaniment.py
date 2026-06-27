"""Contract tests for the accompaniment manifest additions.

Contract: contracts/manifest-accompaniment.md. ``AccompanimentProvenance`` is the new optional
nested object on ``ProvenanceRecord``, present iff ``lane == "accompaniment"``. Both models are
``extra="forbid"``, so they mirror the manifest schema exactly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voders.manifest.models import (
    AccompanimentProvenance,
    ProvenanceRecord,
    ValidationVerdict,
    VerdictStatus,
)


def _verdict() -> ValidationVerdict:
    return ValidationVerdict(
        status=VerdictStatus.ACCEPTED,
        onset_ok=True,
        offset_ok=True,
        f0_ok=True,
        snr_db=12.3,
        max_onset_dev_ms=7.0,
    )


def _lego_provenance() -> AccompanimentProvenance:
    return AccompanimentProvenance(
        mode="lego",
        model_id="ace-step-1.5-xl-base",
        model_version="1.5-xl-2026-04-02",
        model_license="MIT",
        attribution_text=None,
        vocal_bit_exact=True,
        source_vocal_sample_id="score007_singer_donor3",
        takes_tried=3,
        max_note_shift_ms=0.0,
        free_time=True,
        target_instrument="sustained bowed metal pad",
        stem_available=True,
    )


def test_accompaniment_provenance_round_trips() -> None:
    """model_dump() / model_validate() preserve every field."""
    prov = _lego_provenance()
    restored = AccompanimentProvenance.model_validate(prov.model_dump())
    assert restored == prov


def test_record_serializes_nested_accompaniment() -> None:
    """A lane="accompaniment" record carries the nested object through model_dump()."""
    record = ProvenanceRecord(
        sample_id="score007_singer_donor3_accomp_lego",
        score_id="score007",
        score_path="corpus/shard=000/x.tsv",
        audio_path="corpus/shard=000/x.wav",
        lane="accompaniment",
        voice_id="donor3",
        seed=184467,
        config_hash="sha256:abc",
        verdict=_verdict(),
        accompaniment=_lego_provenance(),
    )
    dumped = record.model_dump()
    assert dumped["accompaniment"]["mode"] == "lego"
    assert dumped["accompaniment"]["vocal_bit_exact"] is True

    restored = ProvenanceRecord.model_validate(dumped)
    assert restored.accompaniment == _lego_provenance()


def test_accompaniment_defaults_to_none_for_non_accompaniment_records() -> None:
    """Records on other lanes omit the accompaniment object (defaults to None)."""
    record = ProvenanceRecord(
        sample_id="score007_singer_donor3",
        score_id="score007",
        score_path="corpus/shard=000/x.tsv",
        audio_path="corpus/shard=000/x.wav",
        lane="deterministic",
        voice_id="donor3",
        seed=1,
        config_hash="sha256:abc",
        verdict=_verdict(),
    )
    assert record.accompaniment is None


def test_lego_carries_bit_exact_and_stem_available() -> None:
    """Lego provenance: vocal_bit_exact and stem_available are both true (SC-001)."""
    prov = _lego_provenance()
    assert prov.vocal_bit_exact is True
    assert prov.stem_available is True


def test_complete_carries_both_false() -> None:
    """Complete re-encodes the vocal and emits no separable stem: both false."""
    prov = AccompanimentProvenance(
        mode="complete",
        model_id="ace-step-1.5-xl-base",
        vocal_bit_exact=False,
        stem_available=False,
    )
    assert prov.vocal_bit_exact is False
    assert prov.stem_available is False


def test_record_forbids_unknown_fields() -> None:
    """ProvenanceRecord is extra="forbid": an unknown field raises ValidationError."""
    with pytest.raises(ValidationError):
        ProvenanceRecord(
            sample_id="x",
            score_id="score007",
            score_path="",
            audio_path="",
            lane="accompaniment",
            voice_id="donor3",
            seed=1,
            config_hash="sha256:abc",
            verdict=_verdict(),
            not_a_real_field="boom",
        )
