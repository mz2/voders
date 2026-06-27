"""JSON Lines manifest writer/reader (FR-008).

The manifest is append-only: one JSON object per attempted sample (accepted and non-accepted,
FR-006a). Per-sample lookup, license audits (FR-011/SC-008), and aggregate stats (FR-012) are
computed by scanning the file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from voders.manifest.models import ProvenanceRecord


class ManifestWriter:
    """Append ProvenanceRecords to a ``.jsonl`` file, one object per line (FR-008)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ProvenanceRecord) -> None:
        line = record.model_dump_json()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")

    def __enter__(self) -> ManifestWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def read_manifest(path: str | Path) -> Iterator[ProvenanceRecord]:
    """Scan a manifest file, yielding one ProvenanceRecord per non-empty line (FR-008)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield ProvenanceRecord.model_validate_json(line)


def load_manifest(path: str | Path) -> list[ProvenanceRecord]:
    """Read all records into a list."""
    return list(read_manifest(path))
