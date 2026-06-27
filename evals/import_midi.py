"""Ingest a curated, license-tagged external vocal-melody MIDI corpus as .tsv scores (issue #9).

This is a *coverage* lever (new melodic structure), complementary to the score-domain augmentations
(#8, a density lever). Point it at a directory of curated, vocal-melody MIDI (e.g. POP909's MELODY
track, lead-sheet corpora) — NOT generic polyphonic MIDI — and tag the source's license so it rides
into the manifest (FR-008). It writes one ``.tsv`` per accepted file, a ``_source.json`` sidecar
(source/license/url the orchestrator stamps onto each record), and a yield+bias report so the
silent-filtering distribution shift (#9) is visible.

    uv run --extra cpu python evals/import_midi.py --midi-dir <dir> \
        --out evals/fixtures/scores_pop909 \
        --source POP909 --license "research-only (POP909)" \
        --url https://github.com/music-x-lab/POP909-Dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voders.scores.midi_import import ExtractionStats, extract_melody
from voders.scores.parse import serialize_score


def _report(stats: list[ExtractionStats]) -> dict:
    accepted = [s for s in stats if s.accepted]
    methods: dict[str, int] = {}
    for s in stats:
        methods[s.method] = methods.get(s.method, 0) + 1
    in_pitches = [p for s in accepted if s.pitch_in for p in s.pitch_in]
    out_pitches = [p for s in accepted if s.pitch_out for p in s.pitch_out]
    return {
        "files": len(stats),
        "accepted": len(accepted),
        "rejected": len(stats) - len(accepted),
        "yield": round(len(accepted) / len(stats), 3) if stats else 0.0,
        "methods": methods,
        "notes_in_total": sum(s.notes_in for s in accepted),
        "notes_out_total": sum(s.notes_out for s in accepted),
        "dropped_out_of_range": sum(s.dropped_out_of_range for s in accepted),
        "dropped_short": sum(s.dropped_short for s in accepted),
        "pitch_range_in": [min(in_pitches), max(in_pitches)] if in_pitches else None,
        "pitch_range_out": [min(out_pitches), max(out_pitches)] if out_pitches else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest external vocal-melody MIDI as scores (#9)")
    ap.add_argument("--midi-dir", required=True, help="directory of curated vocal-melody MIDI")
    ap.add_argument("--out", required=True, help="output score directory")
    ap.add_argument("--source", required=True, help="source name (e.g. POP909) — manifest tag")
    ap.add_argument("--license", required=True, help="license string — manifest tag (FR-008)")
    ap.add_argument("--url", default="", help="source URL for provenance")
    ap.add_argument("--lo", type=int, default=55, help="lowest singable MIDI pitch")
    ap.add_argument("--hi", type=int, default=79, help="highest singable MIDI pitch")
    ap.add_argument("--min-note-s", type=float, default=0.08)
    args = ap.parse_args(argv)

    midi_dir = Path(args.midi_dir)
    files = sorted(p for p in midi_dir.rglob("*") if p.suffix.lower() in (".mid", ".midi"))
    if not files:
        print(f"no .mid/.midi files under {midi_dir}")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_stats: list[ExtractionStats] = []
    for f in files:
        score, stats = extract_melody(f, lo=args.lo, hi=args.hi, min_note_s=args.min_note_s)
        all_stats.append(stats)
        if score is not None:
            (out / f"{score.score_id}.tsv").write_bytes(serialize_score(score))

    # License/provenance sidecar — read by the orchestrator and stamped onto every record (FR-008).
    (out / "_source.json").write_text(
        json.dumps({"source": args.source, "license": args.license, "url": args.url}, indent=2),
        encoding="utf-8",
    )
    report = _report(all_stats)
    (out / "_ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ingested {report['accepted']}/{report['files']} (yield {report['yield']}) -> {out}")
    print(f"  methods={report['methods']}  pitch in={report['pitch_range_in']} "
          f"out={report['pitch_range_out']}  dropped_range={report['dropped_out_of_range']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
