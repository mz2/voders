"""Re-derive note labels (onset + offset) from the rendered audio for a force_score_f0 corpus.

``force_score_f0`` pins every label to the rigid score grid, but the SVS backend (nnsvs / SoulX)
sings with its own micro-timing, so the labels drift from what the audio actually does. Measured on
``syntheticdataset_soulx`` (nnsvs / force_score_f0): median offset drift ~37 ms but ~39 % of notes
have a forced offset more than 50 ms from the audio's real voicing end. That is label noise on
exactly the signal the model already predicts worst (offsets).

This relabels each sample's onsets+offsets from its OWN audio using the production f0 alignment
(``voders.validate.f0_align.align_score_to_f0`` — now fixed so each offset tracks the note's actual
voicing end instead of snapping to the next onset), validates the re-derived label against the
audio, and writes a NEW corpus dir. Audio is never re-synthesised; only the ``.tsv`` is rewritten.

Layout (mirrors the staged training dirs)::

    <src>/<sample>/{audio.wav, score.tsv}  ->  <dst>/<sample>/{audio.wav (symlink), score.tsv}

plus ``<dst>/relabel_manifest.jsonl`` (one row per sample: drift vs the old label + validation
verdict) and a printed aggregate summary. Re-derivation is non-destructive: the source corpus is
left untouched so you can A/B train old-vs-relabelled.

Note: f0 is measured twice per sample (once to align, once to validate), so a full 6.7k-sample pass
is not cheap. Trial with ``--limit``/``--stride`` first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from voders.audio import read_wav
from voders.manifest.models import VerdictStatus
from voders.scores.parse import parse_tsv, serialize_score
from voders.validate.f0_align import relabel_score


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src", default="syntheticdataset_soulx", help="corpus dir of <sample>/ subdirs"
    )
    ap.add_argument("--dst", required=True, help="output corpus dir (created; must not be the src)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth sample (subset)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of samples (0 = all)")
    ap.add_argument("--device", default="cpu", help="f0 device for alignment + validation")
    ap.add_argument(
        "--only-accepted",
        action="store_true",
        help="write only samples whose re-derived label PASSES validation (all are still logged)",
    )
    args = ap.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.resolve() == src.resolve():
        ap.error("--dst must differ from --src (this never overwrites the source corpus)")
    dst.mkdir(parents=True, exist_ok=True)

    dirs = sorted(p for p in src.iterdir() if p.is_dir())[:: args.stride]
    if args.limit:
        dirs = dirs[: args.limit]

    manifest = (dst / "relabel_manifest.jsonl").open("w")

    n = written = accepted = 0
    onset_drifts: list[float] = []
    offset_drifts: list[float] = []
    for d in dirs:
        wav, tsv = d / "audio.wav", d / "score.tsv"
        if not (wav.exists() and tsv.exists()):
            continue
        audio, sr = read_wav(wav)
        score = parse_tsv(tsv).score
        rederived, verdict, onset_dev_ms, offset_dev_ms = relabel_score(
            audio, sr, score, device=args.device
        )
        is_accepted = verdict.status == VerdictStatus.ACCEPTED

        n += 1
        onset_drifts.append(onset_dev_ms)
        offset_drifts.append(offset_dev_ms)
        if is_accepted:
            accepted += 1

        manifest.write(
            json.dumps(
                {
                    "sample": d.name,
                    "n_notes": len(rederived.notes),
                    "onset_drift_ms": round(onset_dev_ms, 1),
                    "offset_drift_ms": round(offset_dev_ms, 1),
                    "status": verdict.status.value,
                    "onset_ok": verdict.onset_ok,
                    "offset_ok": verdict.offset_ok,
                    "f0_ok": verdict.f0_ok,
                    "reason": verdict.reason,
                }
            )
            + "\n"
        )

        if args.only_accepted and not is_accepted:
            continue
        out = dst / d.name
        out.mkdir(exist_ok=True)
        audio_link = out / "audio.wav"
        if not audio_link.exists():
            audio_link.symlink_to(wav.resolve())
        (out / "score.tsv").write_bytes(serialize_score(rederived))
        written += 1

    manifest.close()

    print(f"\nProcessed {n} samples; wrote {written} to {dst}")
    if n:
        print(
            f"Validation: {accepted}/{n} accepted ({100 * accepted / n:.0f}%)"
            + (" — only accepted were written" if args.only_accepted else "")
        )
        print(
            "Onset drift vs old label (ms):  "
            f"p50={_pct(onset_drifts, 50):.0f}  p90={_pct(onset_drifts, 90):.0f}  "
            f"p99={_pct(onset_drifts, 99):.0f}  max={max(onset_drifts):.0f}"
        )
        print(
            "Offset drift vs old label (ms): "
            f"p50={_pct(offset_drifts, 50):.0f}  p90={_pct(offset_drifts, 90):.0f}  "
            f"p99={_pct(offset_drifts, 99):.0f}  max={max(offset_drifts):.0f}"
        )
        print(f"Per-sample detail: {dst / 'relabel_manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
