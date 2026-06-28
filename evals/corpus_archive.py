"""Pack a rendered corpus into a committable OGG archive, or unpack it back to WAV.

Rendering the augmented corpus takes a while, so the accepted audio is archived under
``datasets/<run_id>/`` as **OGG/Vorbis** (~17× smaller than 22,050 Hz mono float32 WAV) and
versioned via Git LFS. Both archive and working tree use one directory per sample —
``corpus/shard=NNN/<sample_id>/`` containing ``audio.{wav,ogg}`` plus ``score.tsv``. The labels,
``manifest.jsonl``, ``stats.json``, and ``config.resolved.yaml`` are copied verbatim. ``unpack``
decompresses each ``audio.ogg`` back to ``audio.wav`` at the same per-sample path the renderer
would have written, so the training task consumes the manifest unchanged.

    uv run --extra cpu python evals/corpus_archive.py pack     # out/<id> -> datasets/<id>
    uv run --extra cpu python evals/corpus_archive.py unpack   # datasets/<id> -> out/<id>

OGG/Vorbis is lossy, so a round-trip is NOT bit-exact (bit-exact reproduction is what `just augment`
is for); the labels and provenance are exact. Only accepted ``corpus/`` audio is archived —
``rejected/`` is never trained on, so it is not stored.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
META_FILES = ("manifest.jsonl", "stats.json", "config.resolved.yaml")


def _size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def pack(run_dir: Path, archive: Path) -> int:
    corpus = run_dir / "corpus"
    if not corpus.is_dir():
        print(f"ERROR: no corpus at {corpus} — run `just augment` first.", file=sys.stderr)
        return 1
    arc_corpus = archive / "corpus"
    n_audio = n_label = 0
    for wav in sorted(corpus.rglob("audio.wav")):
        rel_dir = wav.parent.relative_to(corpus)
        dest_dir = arc_corpus / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        audio, sr = sf.read(wav, dtype="float32")
        sf.write(dest_dir / "audio.ogg", audio, sr, format="OGG", subtype="VORBIS")
        n_audio += 1
        tsv = wav.parent / "score.tsv"
        if tsv.exists():
            shutil.copy2(tsv, dest_dir / "score.tsv")
            n_label += 1
    for name in META_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, archive / name)
    print(f"packed {n_audio} audio + {n_label} labels -> {archive}")
    print(f"  WAV corpus {_size_mb(corpus):.1f} MB  ->  OGG archive {_size_mb(archive):.1f} MB")
    return 0


def unpack(archive: Path, run_dir: Path) -> int:
    arc_corpus = archive / "corpus"
    if not arc_corpus.is_dir():
        print(f"ERROR: no archive at {arc_corpus}.", file=sys.stderr)
        return 1
    corpus = run_dir / "corpus"
    n_audio = n_label = 0
    for ogg in sorted(arc_corpus.rglob("audio.ogg")):
        rel_dir = ogg.parent.relative_to(arc_corpus)
        dest_dir = corpus / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        audio, sr = sf.read(ogg, dtype="float32")
        sf.write(dest_dir / "audio.wav", audio, sr, subtype="FLOAT")
        n_audio += 1
        tsv = ogg.parent / "score.tsv"
        if tsv.exists():
            shutil.copy2(tsv, dest_dir / "score.tsv")
            n_label += 1
    for name in META_FILES:
        src = archive / name
        if src.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, run_dir / name)
    print(f"unpacked {n_audio} audio + {n_label} labels -> {corpus}")
    return 0


def _make_relabeler(device: str):
    """Return a ``(audio, sr, tsv_path) -> bool`` that REPAIRS only labels the audio contradicts.

    Labels that already validate against the audio are left untouched — the deterministic lane's
    exact labels and any on-grid SVS sample stay byte-for-byte, so relabeling never injects f0
    measurement noise into already-correct data. Only when the original label fails validation is
    the audio-derived label tried, and written solely if it too passes (else the original is kept).
    Heavy validation deps import here so plain (non-relabel) staging never pays for them.
    """
    from voders.config.models import ValidatorConfig
    from voders.manifest.models import VerdictStatus
    from voders.scores.parse import parse_tsv, serialize_score
    from voders.validate.f0_align import align_score_to_f0
    from voders.validate.validator import Validator

    def _relabel(audio, sr: int, tsv_path: Path) -> bool:
        mono = audio if audio.ndim == 1 else audio.mean(axis=1)
        score = parse_tsv(tsv_path).score
        if not score.notes:
            return False
        validator = Validator(ValidatorConfig(), sr=sr)
        if validator.validate(mono, score, f0_device=device).status == VerdictStatus.ACCEPTED:
            return False  # original label already matches the audio — leave it exact
        rederived, _onset_drift, _offset_drift = align_score_to_f0(
            mono, score, sr=sr, device=device
        )
        if validator.validate(mono, rederived, f0_device=device).status == VerdictStatus.ACCEPTED:
            tsv_path.write_bytes(serialize_score(rederived))
            return True
        return False

    return _relabel


def stage(
    run_ids: list[str],
    out_dir: Path,
    *,
    clean: bool = False,
    relabel_offsets: bool = False,
    relabel_device: str = "auto",
) -> int:
    """Assemble the trainer's flat input dir from committed datasets (OGG -> WAV).

    ``SyntheticDataset`` reads ``<out_dir>/<song>/{audio.wav,score.tsv}``, so every accepted sample
    from each ``datasets/<run_id>`` is decompressed into ``<out_dir>/<run_id>__<sample_id>/``. Pass
    several run ids to combine corpora (e.g. the deterministic ``donor_pool`` AND the lyric/SVS
    ``lyrics_pool``) into one training set. ``clean`` rebuilds ``out_dir`` from scratch so the
    staged set is exactly ``run_ids`` (no stale samples from a previous run).

    ``relabel_offsets`` REPAIRS labels the audio contradicts: the datasets are rendered
    ``force_score_f0`` (labels on the rigid score grid), but the SVS sings with its own timing.
    A label that already validates against its audio is kept byte-for-byte; only one that fails is
    replaced by an audio-derived label, and only if that re-derivation itself validates. No sample
    is lost, and already-correct labels (deterministic lane, on-grid SVS) are never disturbed.
    """
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    relabel = _make_relabeler(relabel_device) if relabel_offsets else None
    total = relabelled = 0
    for spec in run_ids:
        rid, _, cap = spec.partition(":")  # "donor_pool:100" caps that dataset to 100 songs
        limit = int(cap) if cap else None
        arc_corpus = REPO / "datasets" / rid / "corpus"
        if not arc_corpus.is_dir():
            print(f"skip {rid}: no archive at {arc_corpus}", file=sys.stderr)
            continue
        oggs = sorted(arc_corpus.rglob("audio.ogg"))
        if limit is not None:
            oggs = oggs[:limit]  # deterministic slice — a small label/timbre anchor, not the bulk
        n = 0
        for ogg in oggs:
            dest = out_dir / f"{rid}__{ogg.parent.name}"
            dest.mkdir(parents=True, exist_ok=True)
            audio, sr = sf.read(ogg, dtype="float32")
            sf.write(dest / "audio.wav", audio, sr, subtype="FLOAT")
            tsv = ogg.parent / "score.tsv"
            if tsv.exists():
                shutil.copy2(tsv, dest / "score.tsv")
                if relabel is not None and relabel(audio, sr, dest / "score.tsv"):
                    relabelled += 1
            n += 1
            total += 1
        print(f"staged {n} songs from {rid}" + (f" (capped at {limit})" if limit else ""))
    print(f"staged {total} songs -> {out_dir}")
    if relabel is not None:
        print(
            f"repaired labels from audio for {relabelled}/{total} songs "
            "(rest already matched the audio or kept their score labels)"
        )
    if total == 0:
        print(
            f"ERROR: staged 0 songs — none of {run_ids} resolved under datasets/. "
            "Did you pack the datasets (and pull LFS)?",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack/unpack/stage a rendered corpus")
    parser.add_argument("mode", choices=("pack", "unpack", "stage"))
    parser.add_argument("--run-id", default="donor_pool", help="run_id under out/ and datasets/")
    parser.add_argument("--run-dir", default=None, help="override out/<run_id>")
    parser.add_argument("--archive", default=None, help="override datasets/<run_id>")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=["donor_pool", "lyrics_pool"],
        help="datasets to combine into the training dir (stage mode)",
    )
    parser.add_argument(
        "--out", default="syntheticdataset_soulx", help="flat training dir (stage mode)"
    )
    parser.add_argument(
        "--clean", action="store_true", help="rebuild the staging dir from scratch (stage mode)"
    )
    parser.add_argument(
        "--relabel-offsets",
        action="store_true",
        help="re-derive onset/offset labels from each sample's audio during staging, keeping the "
        "re-derived label only when it passes validation (else the original score label). Fixes "
        "force_score_f0 label drift; adds an f0 pass per sample (stage mode)",
    )
    parser.add_argument(
        "--relabel-device",
        default="auto",
        help="f0 device for --relabel-offsets (auto uses CREPE/GPU when available, else pyin/CPU)",
    )
    args = parser.parse_args(argv)

    if args.mode == "stage":
        return stage(
            args.run_ids,
            Path(args.out),
            clean=args.clean,
            relabel_offsets=args.relabel_offsets,
            relabel_device=args.relabel_device,
        )

    run_dir = Path(args.run_dir) if args.run_dir else REPO / "out" / args.run_id
    archive = Path(args.archive) if args.archive else REPO / "datasets" / args.run_id
    if args.mode == "pack":
        return pack(run_dir, archive)
    return unpack(archive, run_dir)


if __name__ == "__main__":
    sys.exit(main())
