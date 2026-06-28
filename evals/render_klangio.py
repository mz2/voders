"""Render a slice of the Klangio phrases via the persistent nnsvs worker, in parallel shards (#9).

Pick a slice with ``--part P --of N`` (e.g. ``--of 5 --part 0`` = the first fifth, ``--part 1`` the
second, …). Phrases are interleaved (``files[P::N]``) so every slice is representative. Each shard
is a ``voders run`` process holding ONE resident nnsvs model (``VODERS_SVS_PERSISTENT=1`` — no
per-sample reload, no GPU-OOM from many transient models), and the formant fallback is off, so every
accepted sample is real nnsvs. After the shards finish, their corpora are merged into a single run
dir ``out/<run_id>/`` ready for ``corpus_archive pack``.

    uv run --extra cpu --extra lyrics --extra gpu python evals/render_klangio.py --of 5 --part 0
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_VOICES = [("female", "nnsvs:female"), ("male", "nnsvs:male"), ("tenor", "nnsvs:tenor")]

_CONFIG = """run_id: {run_id}_shard{k}
master_seed: {seed}
scores: {shard_dir}/*.tsv
output_root: {out}
voices:
  - voice_id: {vid}
    kind: svs_voicebank
    license: CC0
    consent_verified: true
    model_ref: "{vref}"
lanes: {{svs: {{enabled: true, backend: nnsvs, mode: force_score_f0}}}}
lyrics:
  source: automatic
  inventory: scat
  g2p_backend: espeak
  # Languages narrowed to English / German / Mandarin (one per phrase, seeded). SoulX covers en + zh
  # at top quality; German rides this nnsvs+espeak path. Keeping the set small concentrates coverage
  # where SoulX helps most.
  languages: [en-us, de, cmn]
validator: {{onset_ms: 50, offset_min_ms: 150, offset_fraction: 0.5, min_note_ms: null, \
min_note_percentile: 1, f0_cents: 25, f0_coverage: 0.75, snr_floor_db: 0}}
"""


def _merge(run_id: str, shard_outs: list[Path]) -> tuple[Path, int]:
    """Collect every accepted sample dir from the shard runs into one ``out/<run_id>/`` run dir."""
    run_dir = REPO / "out" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    corpus = run_dir / "corpus"
    corpus.mkdir(parents=True)
    manifest_lines: list[str] = []
    n = 0
    for out in shard_outs:
        sc = out / "corpus"
        if sc.is_dir():
            for sample in sorted(p for p in sc.rglob("*") if (p / "audio.wav").exists()):
                dest_shard = corpus / f"shard={n // 1000:03d}"
                dest_shard.mkdir(parents=True, exist_ok=True)
                shutil.move(str(sample), str(dest_shard / sample.name))
                n += 1
        mani = out / "manifest.jsonl"
        if mani.exists():
            manifest_lines.append(mani.read_text(encoding="utf-8"))
    (run_dir / "manifest.jsonl").write_text("".join(manifest_lines), encoding="utf-8")
    return run_dir, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render a slice of Klangio phrases (persistent nnsvs)")
    ap.add_argument("--src", default="evals/fixtures/scores_klangio")
    ap.add_argument("--of", type=int, default=1, help="split the phrases into this many slices")
    ap.add_argument("--part", type=int, default=0, help="which slice (0..of-1) to render")
    ap.add_argument("--shards", type=int, default=6, help="max concurrent render processes")
    ap.add_argument(
        "--chunk-size", type=int, default=150,
        help="phrases per render process; a fresh process per chunk caps the core's per-sample "
        "memory growth (it leaks ~50 MB/sample, so one long process OOMs)",
    )
    ap.add_argument("--limit", type=int, default=0, help="cap phrases (testing)")
    ap.add_argument("--run-id", default="", help="output run id (default klangio_fifth_<part>)")
    args = ap.parse_args(argv)

    src = REPO / args.src
    files = sorted(src.glob("*.tsv"))
    if not files:
        print(f"no phrases under {src} — run evals/ingest_klangio.py first")
        return 1
    files = files[args.part :: args.of] if args.of > 1 else files
    if args.limit:
        files = files[: args.limit]
    run_id = args.run_id or (
        f"klangio_fifth_{args.part}" if args.of == 5 else f"klangio_p{args.part}_of{args.of}"
    )
    sidecar = src / "_source.json"

    work = REPO / "out" / f"{run_id}_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    env = {**os.environ, "VODERS_SVS_PERSISTENT": "1"}  # model resident within each chunk process

    # One render PROCESS per chunk of phrases (fresh process => the core's per-sample leak is freed
    # when it exits). A pool keeps up to --shards running at once. Voice rotates per chunk.
    chunks = [files[i : i + args.chunk_size] for i in range(0, len(files), args.chunk_size)]
    specs, chunk_outs = [], []
    for ci, chunk in enumerate(chunks):
        cdir = work / f"chunk{ci}"
        cdir.mkdir()
        for f in chunk:
            (cdir / f.name).symlink_to(f)
        if sidecar.exists():
            (cdir / "_source.json").write_text(sidecar.read_text(encoding="utf-8"))
        vid, vref = _VOICES[ci % len(_VOICES)]
        out = REPO / "out" / f"{run_id}_chunk{ci}"
        chunk_outs.append(out)
        cfg = cdir / "config.yaml"
        cfg.write_text(
            _CONFIG.format(run_id=f"{run_id}_c{ci}", k=ci, seed=1000 + ci, shard_dir=cdir, out=out,
                           vid=vid, vref=vref),
            encoding="utf-8",
        )
        specs.append(cfg)

    print(f"{len(chunks)} chunks of <= {args.chunk_size} over {len(files)} phrases, "
          f"<= {args.shards} concurrent (slice {args.part}/{args.of})")
    start = time.monotonic()
    queue, running = list(specs), []
    while queue or running:
        while queue and len(running) < args.shards:
            cfg = queue.pop(0)
            running.append(subprocess.Popen(
                ["uv", "run", "--extra", "cpu", "--extra", "lyrics", "--extra", "gpu",
                 "voders", "run", "--config", str(cfg)],
                cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        time.sleep(2)
        running = [p for p in running if p.poll() is None]
    elapsed = time.monotonic() - start

    run_dir, accepted = _merge(run_id, chunk_outs)
    for out in chunk_outs:
        shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    rate = len(files) / elapsed if elapsed else 0
    print(f"{run_id}: {accepted}/{len(files)} accepted in {elapsed:.0f}s ({rate:.1f} phrases/s)")
    print(f"merged -> {run_dir}  (pack: corpus_archive pack --run-id {run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
