"""Pack a rendered corpus into a committable OGG archive, or unpack it back to WAV.

Rendering the augmented corpus takes a while, so the accepted audio is archived under
``datasets/<run_id>/`` as **OGG/Vorbis** (~17× smaller than 22,050 Hz mono float32 WAV) and
versioned via Git LFS. The labels (``*.tsv``), ``manifest.jsonl``, ``stats.json``, and
``config.resolved.yaml`` are copied verbatim. ``unpack`` decompresses the OGG back to WAV at the
exact paths the renderer would have written (``out/<run_id>/corpus/.../*.wav``), so the training
task consumes the manifest unchanged.

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
    for wav in sorted(corpus.rglob("*.wav")):
        rel = wav.relative_to(corpus)
        dest = arc_corpus / rel.with_suffix(".ogg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        audio, sr = sf.read(wav, dtype="float32")
        sf.write(dest, audio, sr, format="OGG", subtype="VORBIS")
        n_audio += 1
    for tsv in sorted(corpus.rglob("*.tsv")):
        rel = tsv.relative_to(corpus)
        dest = arc_corpus / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tsv, dest)
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
    for ogg in sorted(arc_corpus.rglob("*.ogg")):
        rel = ogg.relative_to(arc_corpus)
        dest = corpus / rel.with_suffix(".wav")
        dest.parent.mkdir(parents=True, exist_ok=True)
        audio, sr = sf.read(ogg, dtype="float32")
        sf.write(dest, audio, sr, subtype="FLOAT")
        n_audio += 1
    for tsv in sorted(arc_corpus.rglob("*.tsv")):
        rel = tsv.relative_to(arc_corpus)
        dest = corpus / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tsv, dest)
        n_label += 1
    for name in META_FILES:
        src = archive / name
        if src.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, run_dir / name)
    print(f"unpacked {n_audio} audio + {n_label} labels -> {corpus}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack/unpack a rendered corpus as an OGG archive")
    parser.add_argument("mode", choices=("pack", "unpack"))
    parser.add_argument("--run-id", default="donor_pool", help="run_id under out/ and datasets/")
    parser.add_argument("--run-dir", default=None, help="override out/<run_id>")
    parser.add_argument("--archive", default=None, help="override datasets/<run_id>")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else REPO / "out" / args.run_id
    archive = Path(args.archive) if args.archive else REPO / "datasets" / args.run_id

    if args.mode == "pack":
        return pack(run_dir, archive)
    return unpack(archive, run_dir)


if __name__ == "__main__":
    sys.exit(main())
