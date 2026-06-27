"""Ingest the Klangio MML26 challenge scores as a segmented, singable score source (issue #9).

The challenge (external/MML26-singing-synthesis) provides ~400 real-singing note annotations under
``scores/`` and explicitly invites synthesising singing from them to train on. They are full songs,
segments each into octave-centred singable phrases (``voders.scores.segment``), writing one ``.tsv``
per phrase plus a ``_source.json`` license tag (FR-008) and a yield report, ready to render.

    uv run --extra cpu python evals/ingest_klangio.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voders.scores.parse import parse_tsv, serialize_score
from voders.scores.segment import segment_score

_DEFAULT_SRC = "external/MML26-singing-synthesis/scores"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Segment Klangio MML26 scores into singable phrases")
    ap.add_argument("--src", default=_DEFAULT_SRC, help="Klangio scores/ directory")
    ap.add_argument("--out", default="evals/fixtures/scores_klangio", help="output phrase dir")
    ap.add_argument("--lo", type=int, default=55)
    ap.add_argument("--hi", type=int, default=79)
    ap.add_argument("--max-phrase-s", type=float, default=12.0)
    args = ap.parse_args(argv)

    src = Path(args.src)
    files = sorted(src.glob("*.tsv"))
    if not files:
        print(f"no .tsv scores under {src} — is the submodule initialised?")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    songs = phrases = notes_in = notes_out = 0
    for f in files:
        score = parse_tsv(f, allow_split=True).score
        notes_in += len(score.notes)
        songs += 1
        for phrase in segment_score(score, lo=args.lo, hi=args.hi, max_phrase_s=args.max_phrase_s):
            (out / f"{phrase.score_id}.tsv").write_bytes(serialize_score(phrase))
            phrases += 1
            notes_out += len(phrase.notes)

    (out / "_source.json").write_text(
        json.dumps(
            {
                "source": "Klangio-MML26",
                "license": "MML26 Singing Transcription Challenge (provided training annotations)",
                "url": "https://github.com/Klangio/MML26-singing-synthesis",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "songs": songs,
        "phrases": phrases,
        "phrases_per_song": round(phrases / songs, 1) if songs else 0,
        "notes_in": notes_in,
        "notes_out": notes_out,
        "note_retention": round(notes_out / notes_in, 3) if notes_in else 0,
    }
    (out / "_ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"segmented {songs} songs -> {phrases} phrases ({report['phrases_per_song']}/song), "
          f"note retention {report['note_retention']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
