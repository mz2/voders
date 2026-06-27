"""Contract tests for the 002 lyric deltas to RunConfig and ProvenanceRecord (T010).

Covers the additive ``lyrics`` block on ``RunConfig`` (data-model.md / config deltas) and the five
new lyric-axis fields on ``ProvenanceRecord``. All new fields default so a lyric-free
run validates and serializes exactly as before (SC-001).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voders.config.models import LyricsConfig, RunConfig
from voders.lyrics.models import LyricModel, LyricSource
from voders.manifest.models import ProvenanceRecord, ValidationVerdict, VerdictStatus


def _run_config(**lyrics_kwargs: object) -> RunConfig:
    """A minimal RunConfig; ``lyrics_kwargs`` (when given) build the optional lyrics block."""
    kwargs: dict[str, object] = {
        "run_id": "r",
        "master_seed": 1,
        "scores": "scores/*.tsv",
        "output_root": "out",
    }
    if lyrics_kwargs:
        kwargs["lyrics"] = LyricsConfig(**lyrics_kwargs)
    return RunConfig(**kwargs)


def _record(**overrides: object) -> ProvenanceRecord:
    """A minimal ProvenanceRecord; ``overrides`` set the new lyric-axis fields."""
    kwargs: dict[str, object] = {
        "sample_id": "s",
        "score_id": "score_000",
        "score_path": "corpus/s.tsv",
        "audio_path": "corpus/s.wav",
        "lane": "deterministic",
        "voice_id": "donor_ah",
        "seed": 7,
        "config_hash": "sha256:abc",
        "verdict": ValidationVerdict(status=VerdictStatus.ACCEPTED),
    }
    kwargs.update(overrides)
    return ProvenanceRecord(**kwargs)


def test_lyrics_block_defaults_to_vowel() -> None:
    cfg = _run_config(source=LyricSource.VOWEL)
    assert cfg.lyrics.source == LyricSource.VOWEL


def test_omitting_lyrics_block_defaults_to_vowel() -> None:
    cfg = _run_config()
    assert cfg.lyrics.source == LyricSource.VOWEL
    assert cfg.lyrics.inventory == "en_cv"
    assert cfg.lyrics.melisma == "per_note"
    assert cfg.lyrics.theme is None
    assert cfg.lyrics.model is None


def test_theme_without_generated_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _run_config(source=LyricSource.VOWEL, theme="winter, longing")


def test_model_without_generated_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _run_config(
            source=LyricSource.AUTOMATIC,
            model=LyricModel(model_id="m", license="x", license_ok=True),
        )


def test_theme_and_model_accepted_with_generated_source() -> None:
    cfg = _run_config(
        source=LyricSource.GENERATED,
        theme="winter, longing",
        model=LyricModel(model_id="m", license="x", license_ok=True),
    )
    assert cfg.lyrics.source == LyricSource.GENERATED
    assert cfg.lyrics.theme == "winter, longing"


def test_sustain_ties_melisma_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _run_config(melisma="sustain_ties")


def test_provenance_record_accepts_new_lyric_fields() -> None:
    record = _record(
        lyric_source="generated",
        lyric_hash="sha256:deadbeef",
        lyric_model="m",
        lyric_model_license="CC-BY",
        lyric_articulated=True,
    )
    assert record.lyric_source == "generated"
    assert record.lyric_hash == "sha256:deadbeef"
    assert record.lyric_model == "m"
    assert record.lyric_model_license == "CC-BY"
    assert record.lyric_articulated is True


def test_provenance_record_lyric_fields_are_backward_compatible() -> None:
    record = _record()
    assert record.lyric_source == "vowel"
    assert record.lyric_hash is None
    assert record.lyric_model is None
    assert record.lyric_model_license is None
    assert record.lyric_articulated is False
