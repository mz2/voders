"""Fetch real, permissively-licensed donor voices (VocalSet / VCTK).

Run via uv with the donors extra::

    uv run --extra donors python evals/download_donors.py

Streams one example from each dataset (no full download), resamples to 22,050 Hz mono, and writes a
donor WAV under ``models/donors/`` (git-ignored). Both datasets are CC BY 4.0 — speakers consented
to open release — so the resulting voices pass the consent gate (FR-011) with attribution.

  - VocalSet (CC BY 4.0): professional singers sustaining vowels — the closest match to a "donor
    vowel". Mirror: Bill13579/vocalset-mirror.
  - VCTK (CC BY 4.0): read speech; a voiced utterance serves as a donor timbre. Mirror:
    CSTR-Edinburgh/vctk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from voders.constants import SAMPLE_RATE

DONORS_ROOT = Path(__file__).resolve().parents[1] / "models" / "donors"

# Default fetch: VocalSet (a reliable streaming mirror of sustained sung vowels).
# VCTK is primarily used as a voice-conversion *target* via the consented RVC p231 model
# (`just download-rvc-voice`); a VCTK raw-donor-wav is only reachable via the canonical CSTR
# mirror, which streams slowly, so it is opt-in (`... download_donors.py vctk`), not the default.
SOURCES = {
    "vocalset": ("Bill13579/vocalset-mirror", "train", "vocalset_singer.wav"),
    "vctk": ("CSTR-Edinburgh/vctk", "train", "vctk_speaker.wav"),
}
_DEFAULT = ["vocalset"]


def _first_audio(dataset_id: str, split: str):  # noqa: ANN202
    """Stream the first audio example and decode it with soundfile.

    The audio column is read with ``Audio(decode=False)`` so datasets hands back the raw file bytes
    (no torchcodec/torchaudio decoder dependency); soundfile decodes them here.
    """
    import io

    from datasets import Audio, load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    audio_cols = [k for k, v in (ds.features or {}).items() if isinstance(v, Audio)]
    for col in audio_cols:
        ds = ds.cast_column(col, Audio(decode=False))

    for example in ds:
        raw = [example[c] for c in audio_cols] if audio_cols else list(example.values())
        for value in raw:
            if not (isinstance(value, dict) and ("bytes" in value or "path" in value)):
                continue
            data = value.get("bytes")
            if data is None and value.get("path"):
                data = Path(value["path"]).read_bytes()
            if not data:
                continue
            arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            if arr.size > 0:
                return arr, int(sr)
    raise RuntimeError(f"no decodable audio example found in {dataset_id!r}")


def _save(arr: np.ndarray, sr: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if sr != SAMPLE_RATE:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = (arr / peak * 0.9).astype(np.float32)
    sf.write(str(dest), arr, SAMPLE_RATE, subtype="FLOAT")
    print(f"  wrote {dest} ({arr.size / SAMPLE_RATE:.1f} s @ {SAMPLE_RATE} Hz)")


_VOCALSET_REPO = "Bill13579/vocalset-mirror"


def _vocalset_singers(n: int):  # noqa: ANN202
    """Yield ``(label, arr, sr)`` for up to ``n`` distinct VocalSet singers.

    VocalSet is ~20 professional singers; the mirror's parquet ``label`` column is the singer id.
    Reads the parquet shards directly (fast — the audio bytes are embedded) and takes the first clip
    per distinct ``label`` until ``n`` singers are collected.
    """
    import io

    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files

    shards = sorted(
        f for f in list_repo_files(_VOCALSET_REPO, repo_type="dataset") if f.endswith(".parquet")
    )
    seen: set[int] = set()
    for shard in shards:
        if len(seen) >= n:
            return
        pf = pq.ParquetFile(fsspec.open(f"hf://datasets/{_VOCALSET_REPO}/{shard}").open())
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["audio", "label"])
            labels = tbl.column("label").to_pylist()
            if set(labels) <= seen:  # nothing new in this group — skip the audio decode
                continue
            audio = tbl.column("audio").to_pylist()
            for a, lab in zip(audio, labels, strict=False):
                lab = int(lab)
                if lab in seen or not isinstance(a, dict) or not a.get("bytes"):
                    continue
                arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
                if arr.size == 0:
                    continue
                seen.add(lab)
                yield lab, arr, sr
                if len(seen) >= n:
                    return


def _vocalset_dest(label: int) -> Path:
    """Singer 0 keeps the historical ``vocalset_singer.wav`` name; others are ``vocalset_NN``."""
    return DONORS_ROOT / ("vocalset_singer.wav" if label == 0 else f"vocalset_{label:02d}.wav")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch permissively-licensed donor voices")
    parser.add_argument("sources", nargs="*", help="donor sources to fetch (default: vocalset)")
    parser.add_argument(
        "--vocalset-singers",
        type=int,
        default=1,
        help="enroll this many distinct VocalSet singers (one clip each; ~20 available)",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    which = args.sources or _DEFAULT

    for name in which:
        if name not in SOURCES:
            print(f"unknown donor source {name!r}; known: {sorted(SOURCES)}")
            return 2
        if name == "vocalset" and args.vocalset_singers > 1:
            print(f"== vocalset: enrolling up to {args.vocalset_singers} distinct singers ==")
            enrolled = 0
            for label, arr, sr in _vocalset_singers(args.vocalset_singers):
                dest = _vocalset_dest(label)
                enrolled += 1
                if dest.exists():
                    print(f"  singer {label}: exists, skipping {dest.name}")
                    continue
                _save(arr, sr, dest)
            print(f"== vocalset: {enrolled} singer(s) enrolled under {DONORS_ROOT} ==")
            continue
        dataset_id, split, out = SOURCES[name]
        dest = DONORS_ROOT / out
        if dest.exists():
            print(f"== {name}: exists, skipping {dest.name} ==")
            continue
        print(f"== {name}: streaming one example from {dataset_id} ==")
        arr, sr = _first_audio(dataset_id, split)
        _save(arr, sr, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
