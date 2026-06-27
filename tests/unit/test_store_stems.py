"""Unit tests for accompaniment stem storage (FR-015, SC-008).

``CorpusStore.write_accompaniment_sample`` writes the mix as the corpus audio and, when a Lego
accompaniment stem is supplied, an adjacent ``{sample_id}.stem.wav`` for re-mixing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voders.audio import read_wav
from voders.corpus.store import CorpusStore
from voders.manifest.models import (
    AccompanimentProvenance,
    ProvenanceRecord,
    ValidationVerdict,
    VerdictStatus,
)


def _record() -> ProvenanceRecord:
    return ProvenanceRecord(
        sample_id="score_000_acc_lego",
        score_id="score_000",
        score_path="",
        audio_path="",
        lane="accompaniment",
        voice_id="donor_ah",
        seed=1,
        config_hash="x",
        verdict=ValidationVerdict(
            status=VerdictStatus.ACCEPTED, onset_ok=True, offset_ok=True, f0_ok=True
        ),
        accompaniment=AccompanimentProvenance(
            mode="lego",
            model_id="fake",
            stem_available=True,
            vocal_bit_exact=True,
        ),
    )


def test_writes_mix_and_stem_alongside(tmp_path: Path) -> None:
    """An accompaniment sample writes the mix wav plus an adjacent .stem.wav."""
    store = CorpusStore(tmp_path)
    record = _record()
    mix = np.linspace(-0.5, 0.5, 2048, dtype=np.float32)
    stem = np.linspace(0.1, -0.1, 2048, dtype=np.float32)

    written = store.write_accompaniment_sample(record, mix, b"score-tsv", 0, stem=stem)

    assert written.audio_path
    assert written.score_path
    audio_path, _ = store.paths_for(written, 0)
    stem_path = audio_path.with_suffix(".stem.wav")
    assert audio_path.exists()
    assert stem_path.exists()
    assert stem_path.name == "score_000_acc_lego.stem.wav"

    stem_back, sr = read_wav(stem_path)
    assert sr == 22_050
    assert stem_back.size == stem.size


def test_no_stem_written_when_stem_is_none(tmp_path: Path) -> None:
    """With stem=None the mix is written but no .stem.wav is produced."""
    store = CorpusStore(tmp_path)
    record = _record()
    mix = np.zeros(1024, dtype=np.float32)

    written = store.write_accompaniment_sample(record, mix, b"score-tsv", 0, stem=None)

    audio_path, _ = store.paths_for(written, 0)
    stem_path = audio_path.with_suffix(".stem.wav")
    assert audio_path.exists()
    assert not stem_path.exists()
