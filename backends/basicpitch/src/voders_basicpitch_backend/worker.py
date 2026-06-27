"""Run Basic Pitch on a produced corpus and score note-F1 against the labels, per source.

This is the *consumer-side* check: Basic Pitch (the downstream transcription model) is run on each
accepted sample's audio, and its transcription is compared to the paired label with mir_eval's
note metric — onset within 50 ms, pitch within 50 cents (a quarter-tone; the challenge tolerance),
onset-only (COnP, no offset). Results are aggregated overall and broken down by source
(lane / voice / augmentation profile) so weak generators can be isolated.

It runs out-of-process (Basic Pitch pins old TensorFlow); the manifest is read as plain JSON so this
backend has no dependency on the ``voders`` package.

Usage: ``python -m voders_basicpitch_backend.worker <manifest.jsonl>``  -> prints a JSON report.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ONSET_TOL_S = 0.05
_PITCH_TOL_CENTS = 50.0


def _midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def _label_notes(tsv: Path) -> tuple[np.ndarray, np.ndarray]:
    intervals, pitches = [], []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        try:
            on, off, pitch = float(cols[0]), float(cols[1]), int(round(float(cols[2])))
        except ValueError:
            continue
        intervals.append([on, off])
        pitches.append(_midi_to_hz(pitch))
    return np.asarray(intervals, dtype=float).reshape(-1, 2), np.asarray(pitches, dtype=float)


def _basic_pitch_notes(wav: Path, model) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    from basic_pitch.inference import predict

    _, _, note_events = predict(str(wav), model)
    intervals, pitches = [], []
    for ev in note_events:  # (start_s, end_s, pitch_midi, amplitude, [bends])
        intervals.append([float(ev[0]), float(ev[1])])
        pitches.append(_midi_to_hz(float(ev[2])))
    return np.asarray(intervals, dtype=float).reshape(-1, 2), np.asarray(pitches, dtype=float)


def _note_f1(ref_i, ref_p, est_i, est_p) -> float | None:  # noqa: ANN001
    import mir_eval

    if ref_i.shape[0] == 0:
        return None
    _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i,
        ref_p,
        est_i,
        est_p,
        onset_tolerance=_ONSET_TOL_S,
        pitch_tolerance=_PITCH_TOL_CENTS,
        offset_ratio=None,  # COnP: onset + pitch, ignore offset
    )
    return float(f1)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(json.dumps({"ok": False, "error": "usage: worker <manifest.jsonl>"}))
        return 2
    manifest = Path(args[0])
    root = manifest.parent
    records = [
        json.loads(ln) for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    # Score every sample that has audio (accepted AND non-accepted), so we can see whether the
    # samples our validator rejected would actually transcribe fine for the downstream consumer.
    scored_records = [r for r in records if r.get("audio_path")]

    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    model = Model(ICASSP_2022_MODEL_PATH)

    overall: list[float] = []
    per_source: dict[str, list[float]] = defaultdict(list)
    per_status: dict[str, list[float]] = defaultdict(list)
    for r in scored_records:
        wav, tsv = root / r["audio_path"], root / r["score_path"]
        if not wav.exists() or not tsv.exists():
            continue
        ref_i, ref_p = _label_notes(tsv)
        est_i, est_p = _basic_pitch_notes(wav, model)
        f1 = _note_f1(ref_i, ref_p, est_i, est_p)
        if f1 is None:
            continue
        overall.append(f1)
        status = r.get("verdict", {}).get("status", "?")
        per_status[status].append(f1)
        key = f"{r['lane']}/{r['voice_id']}"
        if r.get("augmentation_profile"):
            key += f"/{r['augmentation_profile']}"
        per_source[key].append(f1)

    def _summ(vals: list[float]) -> dict[str, float]:
        return {"note_f1_mean": float(np.mean(vals)), "n": len(vals)}

    report = {
        "ok": True,
        "n_scored": len(overall),
        "note_f1_mean": float(np.mean(overall)) if overall else None,
        "by_status": {k: _summ(v) for k, v in sorted(per_status.items())},
        "per_source": {k: _summ(v) for k, v in sorted(per_source.items())},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
