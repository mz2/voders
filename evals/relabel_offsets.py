"""Build an offset-ablation pair from a staged training dir, changing ONLY the offset labels.

For a controlled A/B on whether offset-label noise hurts COnOff/COnPOff: copy a subset of staged
samples into two dirs with identical audio + onsets + pitches, differing only in each note's offset:
  --mode orig    : keep the score's offsets (the current, loose-gate labels)
  --mode offfix  : re-derive each offset from the audio (last in-tune frame of the note), so the
                   label matches where the note's voicing actually ends.
Train a model on each and compare COnOff on Klangio — any difference is purely the offset labels.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from voders.audio import read_wav
from voders.render.deterministic import midi_to_hz
from voders.scores.models import Note, Score
from voders.scores.parse import parse_tsv, serialize_score
from voders.validate.validator import _cents, _measure_f0


def _refine_offset(note, nxt_onset, t, f0, voiced) -> float:
    """Last in-tune frame of the note (within a small window), clamped to its span — the audio's
    actual voicing end for this note."""
    cents = _cents(f0, midi_to_hz(note.pitch_midi))
    intune = voiced & np.isfinite(cents) & (np.abs(cents) <= 100)
    hi = min(note.offset_s + 0.2, nxt_onset)
    idx = np.where(intune & (t >= note.onset_s) & (t <= hi))[0]
    off = float(t[idx[-1]]) if idx.size else note.offset_s
    off = max(note.onset_s + 0.03, off)
    if nxt_onset > note.onset_s:
        off = min(off, nxt_onset - 0.001)
    return round(off, 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build orig/offfix offset-ablation training dir")
    ap.add_argument("--src", default="syntheticdataset_soulx")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", choices=("orig", "offfix"), required=True)
    ap.add_argument("--stride", type=int, default=3, help="take every Nth sample (subset)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    src = Path(args.src)
    dirs = sorted(p for p in src.iterdir() if p.is_dir())[:: args.stride]
    if args.limit:
        dirs = dirs[: args.limit]
    dst = Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    n = 0
    for d in dirs:
        wav, tsv = d / "audio.wav", d / "score.tsv"
        if not (wav.exists() and tsv.exists()):
            continue
        out = dst / d.name
        out.mkdir()
        if args.mode == "orig":
            (out / "audio.wav").symlink_to(wav.resolve())
            shutil.copy2(tsv, out / "score.tsv")
        else:
            audio, sr = read_wav(wav)
            score = parse_tsv(tsv).score
            meas = _measure_f0(audio, sr, 0.0, device="cpu")
            t, f0, v = np.asarray(meas.times), np.asarray(meas.f0), np.asarray(meas.voiced)
            notes = score.notes
            new = [
                Note(
                    onset_s=note.onset_s,
                    offset_s=_refine_offset(
                        note,
                        notes[i + 1].onset_s if i + 1 < len(notes) else float("inf"),
                        t, f0, v,
                    ),
                    pitch_midi=note.pitch_midi,
                )
                for i, note in enumerate(notes)
            ]
            (out / "audio.wav").symlink_to(wav.resolve())
            (out / "score.tsv").write_bytes(
                serialize_score(Score(score_id=score.score_id, notes=new))
            )
        n += 1
    print(f"{args.mode}: wrote {n} samples -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
