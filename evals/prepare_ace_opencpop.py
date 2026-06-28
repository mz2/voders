"""Prepare ACE-Opencpop score metadata for synthetic SoulX rendering.

The source dataset stores audio and annotations together in large Parquet shards.  This importer
projects only the symbolic columns, so preparing scores does not download the ~43 GB source-audio
column.  It writes one Klangio-compatible TSV per segment for the existing ``voders run`` pipeline.

ACE-Opencpop is CC BY-NC 4.0.  The explicit license acknowledgement is intentional: this workflow
conservatively treats the generated corpus as subject to the source-score restrictions.

    uv run --extra donors python evals/prepare_ace_opencpop.py \
        --split validation --limit 10 --accept-license
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from voders.scores.models import Note, Score
from voders.scores.parse import serialize_score

ACE_REPO = "espnet/ace-opencpop-segments"
ACE_LICENSE = "CC BY-NC 4.0"
ACE_URL = "https://huggingface.co/datasets/espnet/ace-opencpop-segments"
DEFAULT_OUT = Path("generated/ace_opencpop_scores")
_COLUMNS = (
    "segment_id",
    "note_midi",
    "note_lyrics",
    "note_start_times",
    "note_end_times",
)
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_SLURS = frozenset({"—", "-", "_"})


def _as_list(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    raise ValueError(f"{key} must be a sequence")


def score_from_ace_row(row: Mapping[str, Any]) -> Score:
    """Convert one ACE metadata row to a monophonic transcription score.

    MIDI 0 denotes a rest and is omitted from the TSV.  A same-pitch slur (``—``) extends the
    preceding note instead of inventing an inaudible second onset; pitch-changing slurs remain
    separate notes because a transcription model must detect the pitch transition.
    """
    segment_id = str(row.get("segment_id", "")).strip()
    if not segment_id:
        raise ValueError("segment_id is empty")
    pitches = _as_list(row, "note_midi")
    starts = _as_list(row, "note_start_times")
    ends = _as_list(row, "note_end_times")
    lyrics = _as_list(row, "note_lyrics")
    lengths = {len(pitches), len(starts), len(ends), len(lyrics)}
    if len(lengths) != 1:
        raise ValueError(
            "note_midi, note_lyrics, note_start_times, and note_end_times differ in length"
        )

    notes: list[Note] = []
    for raw_pitch, raw_start, raw_end, raw_lyric in zip(pitches, starts, ends, lyrics, strict=True):
        pitch = int(round(float(raw_pitch)))
        onset = float(raw_start)
        offset = float(raw_end)
        if not (math.isfinite(onset) and math.isfinite(offset)):
            raise ValueError("note timing is not finite")
        if onset < 0 or offset <= onset:
            raise ValueError(f"invalid note timing {onset:.6f}..{offset:.6f}")
        if pitch <= 0:  # ACE uses MIDI 0 for SP/AP rests.
            continue
        if pitch > 127:
            raise ValueError(f"MIDI pitch outside 1..127: {pitch}")

        lyric = str(raw_lyric).strip()
        previous = notes[-1] if notes else None
        if (
            lyric in _SLURS
            and previous is not None
            and previous.pitch_midi == pitch
            and abs(previous.offset_s - onset) <= 0.002
        ):
            notes[-1] = previous.model_copy(update={"offset_s": offset})
            continue
        notes.append(Note(onset_s=onset, offset_s=offset, pitch_midi=pitch))

    score = Score(score_id=segment_id, source=f"{ACE_REPO}:{segment_id}", notes=notes)
    if not score.is_monophonic():
        raise ValueError("ACE row contains overlapping notes")
    return score


def score_filename(segment_id: str) -> str:
    """Return a readable, collision-resistant filename for an ACE segment id."""
    slug = _SAFE.sub("_", segment_id).strip("._-") or "segment"
    digest = hashlib.sha256(segment_id.encode()).hexdigest()[:10]
    return f"{slug[:100]}-{digest}.tsv"


def iter_ace_rows(
    split: str, *, repo_id: str = ACE_REPO, token: str | None = None
) -> Iterator[dict[str, Any]]:
    """Yield metadata-only ACE rows by projected Parquet range reads."""
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files

    files = sorted(
        name
        for name in list_repo_files(repo_id, repo_type="dataset", token=token)
        if name.startswith(f"data/{split}-") and name.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"no {split!r} Parquet shards found in {repo_id}")

    storage_options = {"token": token} if token else {}
    for shard in files:
        url = f"hf://datasets/{repo_id}/{shard}"
        with fsspec.open(url, "rb", **storage_options).open() as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(columns=list(_COLUMNS), batch_size=256):
                yield from batch.to_pylist()


def write_scores(
    rows: Iterable[Mapping[str, Any]], out_dir: Path, *, limit: int = 0
) -> dict[str, int]:
    """Write prepared scores, returning deterministic ingest counters."""
    selected = written = existing = invalid = empty = 0
    for row in rows:
        if limit and selected >= limit:
            break
        selected += 1
        try:
            score = score_from_ace_row(row)
        except (TypeError, ValueError) as exc:
            invalid += 1
            if invalid <= 10:
                print(f"skip invalid ACE row {row.get('segment_id', '?')!r}: {exc}")
            continue
        if score.is_empty:
            empty += 1
            continue
        dest = out_dir / score_filename(score.score_id)
        if dest.exists():
            existing += 1
            continue
        dest.write_bytes(serialize_score(score))
        written += 1
        if (written + existing) % 1000 == 0:
            print(f"  prepared {written + existing}/{selected} usable scores")
    return {
        "selected": selected,
        "written": written,
        "existing": existing,
        "invalid": invalid,
        "empty": empty,
        "usable": written + existing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare ACE-Opencpop symbolic scores for SoulX synthetic rendering"
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--limit", type=int, default=0, help="maximum source rows (0 = full split)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clean", action="store_true", help="replace this split's prepared scores")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help=f"acknowledge that ACE-Opencpop and derived data are {ACE_LICENSE}",
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if not args.accept_license:
        parser.error(f"pass --accept-license after reviewing the {ACE_LICENSE} dataset terms")

    out_dir = args.out / args.split
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    report = write_scores(iter_ace_rows(args.split, token=token), out_dir, limit=args.limit)
    source = {
        "source": "ACE-Opencpop",
        "license": ACE_LICENSE,
        "url": ACE_URL,
        "repository": ACE_REPO,
        "split": args.split,
        "audio_downloaded": False,
    }
    (out_dir / "_source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    (out_dir / "_ingest_report.json").write_text(
        json.dumps({**source, **report}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"ACE-Opencpop {args.split}: {report['usable']} usable scores "
        f"({report['written']} new, {report['existing']} existing, "
        f"{report['invalid']} invalid, {report['empty']} empty) -> {out_dir}"
    )
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
