"""Generate N singable melodies as ``.tsv`` scores — coverage for the score/label axis.

Each melody is deterministic in ``--seed`` + index, diatonic, and singable, so the rendered corpus
gains genuine melodic variety (keys, contours, rhythms) the 10 hand-built fixtures lack. Point a run
config's ``scores`` glob at the output dir to use them.

    uv run --extra cpu python evals/generate_melodies.py --n 200 --out evals/fixtures/scores_melody
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voders.scores.melody import generate_melody
from voders.scores.parse import serialize_score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate singable melodies as .tsv scores")
    ap.add_argument("--n", type=int, default=200, help="how many melodies to generate")
    ap.add_argument("--seed", type=int, default=20260629, help="master seed (per-melody = seed+i)")
    ap.add_argument("--out", default="evals/fixtures/scores_melody", help="output directory")
    ap.add_argument("--lo", type=int, default=55, help="lowest singable MIDI pitch")
    ap.add_argument("--hi", type=int, default=79, help="highest singable MIDI pitch")
    ap.add_argument("--min-notes", type=int, default=6)
    ap.add_argument("--max-notes", type=int, default=16)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        sid = f"melody_{i:04d}"
        score = generate_melody(
            args.seed + i,
            score_id=sid,
            lo=args.lo,
            hi=args.hi,
            min_notes=args.min_notes,
            max_notes=args.max_notes,
        )
        (out / f"{sid}.tsv").write_bytes(serialize_score(score))
    print(f"wrote {args.n} melodies -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
