"""Parse ``.tsv`` scores and check monophony (FR-001).

Score files are row-oriented ``onset_s<TAB>offset_s<TAB>pitch_midi`` (the Klangio challenge input
format). A leading non-numeric header row is tolerated. The raw bytes are retained so the paired
output ``.tsv`` can be byte-identical to the input (FR-002).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voders.scores.models import Note, Score


class PolyphonyError(ValueError):
    """Raised when a score contains overlapping (simultaneous) notes (FR-001 edge case)."""


@dataclass(frozen=True)
class ParsedScore:
    """A parsed score plus the exact source path/bytes for byte-identical re-emission."""

    score: Score
    source_path: Path
    raw_bytes: bytes


def _split_row(line: str) -> list[str]:
    # Accept tab-separated (canonical) or any whitespace.
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    return line.split()


def parse_tsv(path: str | Path, *, allow_split: bool = False) -> ParsedScore:
    """Parse one ``.tsv`` score file into a :class:`ParsedScore`.

    Args:
        path: score file path; its stem becomes ``score_id``.
        allow_split: reserved for splitting polyphony into monophonic tracks; when False
            (default) an overlapping score raises :class:`PolyphonyError`.
    """
    p = Path(path)
    raw = p.read_bytes()
    text = raw.decode("utf-8")

    notes: list[Note] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cols = _split_row(stripped)
        if len(cols) < 3:
            continue
        try:
            onset = float(cols[0])
            offset = float(cols[1])
            pitch = int(round(float(cols[2])))
        except ValueError:
            # Header or comment row — skip.
            continue
        # Optional 4th column: a per-note lyric/syllable (backward-compatible — 3-column scores
        # parse exactly as before, lyric=None).
        lyric = cols[3].strip() if len(cols) >= 4 and cols[3].strip() else None
        notes.append(Note(onset_s=onset, offset_s=offset, pitch_midi=pitch, lyric=lyric))

    score = Score(score_id=p.stem, source=str(p), notes=notes)

    if not score.is_monophonic() and not allow_split:
        pairs = score.overlapping_pairs()
        raise PolyphonyError(
            f"score {score.score_id!r} has {len(pairs)} overlapping note pair(s); "
            "singing is monophonic (FR-001)"
        )

    return ParsedScore(score=score, source_path=p, raw_bytes=raw)


def serialize_score(score: Score) -> bytes:
    """Canonical ``.tsv`` serialization (used for re-derived labels, FR-007).

    Onsets/offsets are written with millisecond precision; pitch as an integer. A 4th lyric column
    is emitted only when at least one note carries a lyric, so lyric-free scores stay 3-column.
    """
    has_lyrics = any(n.lyric for n in score.notes)
    lines = []
    for n in score.notes:
        row = f"{n.onset_s:.6f}\t{n.offset_s:.6f}\t{n.pitch_midi}"
        if has_lyrics:
            row += f"\t{n.lyric or ''}"
        lines.append(row)
    return ("\n".join(lines) + "\n").encode("utf-8")
