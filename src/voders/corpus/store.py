"""Sharded on-disk corpus layout with checkpoint/resume (FR-006a, SC-011; research Decision 7).

Layout under the output root::

    config.resolved.yaml   manifest.jsonl   stats.json
    corpus/shard=NNN/<sample_id>/      audio.wav + byte-identical score.tsv  (accepted)
    rejected/shard=NNN/<sample_id>/    non-accepted samples (audio + reason) (never trained on)
    checkpoints/                       streaming resume state (git-ignored scratch)

Accepted samples (and only those) land in ``corpus/``; every non-accepted status retains both its
provenance and its audio under ``rejected/`` (FR-006a).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voders.audio import write_wav
from voders.manifest.models import ProvenanceRecord, VerdictStatus

_CHECKPOINT_FILE = "completed.txt"


class CorpusStore:
    """Writes samples into the sharded corpus / rejected trees and tracks resume state."""

    def __init__(self, output_root: str | Path, shard_size: int = 1000) -> None:
        self.root = Path(output_root)
        self.shard_size = shard_size
        self.corpus_dir = self.root / "corpus"
        self.rejected_dir = self.root / "rejected"
        self.checkpoint_dir = self.root / "checkpoints"
        self.manifest_path = self.root / "manifest.jsonl"
        self.config_resolved_path = self.root / "config.resolved.yaml"
        self.stats_path = self.root / "stats.json"

    def ensure_dirs(self) -> None:
        for d in (self.corpus_dir, self.rejected_dir, self.checkpoint_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _shard_dir(self, base: Path, index: int) -> Path:
        d = base / f"shard={index // self.shard_size:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def paths_for(self, record: ProvenanceRecord, index: int) -> tuple[Path, Path]:
        """Return (audio_path, score_path) for a sample given its accept/reject status.

        Each sample gets its own directory ``shard=NNN/<sample_id>/`` with ``audio.wav`` and
        ``score.tsv`` — a flat per-sample layout downstream loaders can iterate as one dir per item.
        """
        base = (
            self.corpus_dir
            if record.verdict.status == VerdictStatus.ACCEPTED
            else self.rejected_dir
        )
        shard = self._shard_dir(base, index)
        sample_dir = shard / record.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        return sample_dir / "audio.wav", sample_dir / "score.tsv"

    def write_sample(
        self,
        record: ProvenanceRecord,
        audio: np.ndarray,
        score_tsv: bytes,
        index: int,
    ) -> ProvenanceRecord:
        """Write the (wav, tsv) pair and return the record with resolved relative paths."""
        self.ensure_dirs()
        audio_path, score_path = self.paths_for(record, index)
        write_wav(audio_path, audio)
        score_path.write_bytes(score_tsv)
        return record.model_copy(
            update={
                "audio_path": str(audio_path.relative_to(self.root)),
                "score_path": str(score_path.relative_to(self.root)),
            }
        )

    def write_accompaniment_sample(
        self,
        record: ProvenanceRecord,
        mix: np.ndarray,
        score_tsv: bytes,
        index: int,
        *,
        stem: np.ndarray | None = None,
    ) -> ProvenanceRecord:
        """Write the accompaniment mix (+ optional stem) and byte-identical score (FR-015).

        The mix is the corpus audio (the sample dir's ``audio.wav``); the Lego accompaniment stem,
        when present, is written alongside as ``audio.stem.wav`` so the sample can be re-mixed at a
        different vocal/accompaniment balance without regenerating (SC-008).
        """
        record = self.write_sample(record, mix, score_tsv, index)
        if stem is not None:
            audio_path, _ = self.paths_for(record, index)
            stem_path = audio_path.with_suffix(".stem.wav")
            write_wav(stem_path, stem)
        return record

    # --- checkpoint / resume (SC-011) ---

    def _checkpoint_path(self) -> Path:
        return self.checkpoint_dir / _CHECKPOINT_FILE

    def completed_ids(self) -> set[str]:
        p = self._checkpoint_path()
        if not p.exists():
            return set()
        return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}

    def mark_completed(self, sample_id: str) -> None:
        self.ensure_dirs()
        with self._checkpoint_path().open("a", encoding="utf-8") as fh:
            fh.write(sample_id + "\n")
