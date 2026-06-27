"""Accompaniment throughput benchmark (research.md Decision 7).

Times the accompaniment backend over a handful of fixture-sized clips and reports
samples/hour. GPU-gated: defaults to the CPU ``fake`` backend (a sanity floor); pass
``--backend acestep`` to benchmark the real ACE-Step model once the `accomp` extra and weights are
installed. Not part of CI — the planning target is >= 60 admitted samples/hour per A6000-class GPU.

    uv run python evals/bench_accompaniment.py --backend fake --n 10
    uv run --extra accomp python evals/bench_accompaniment.py --backend acestep --mode lego
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from voders.config.models import AccompanimentOptions, ValidatorConfig
from voders.constants import SAMPLE_RATE
from voders.render.accompaniment import Accompanist
from voders.render.accompaniment_backend import FakeAccompanimentBackend
from voders.scores.models import Note, Score
from voders.validate.validator import Validator


def _make_clip(seconds: float = 4.0, pitch_midi: int = 60) -> tuple[np.ndarray, Score]:
    from voders.render.deterministic import midi_to_hz

    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    vocal = (0.3 * np.sin(2 * np.pi * float(midi_to_hz(pitch_midi)) * t)).astype(np.float32)
    score = Score(
        score_id="bench", notes=[Note(onset_s=0.5, offset_s=seconds - 0.5, pitch_midi=pitch_midi)]
    )
    return vocal, score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="accompaniment throughput benchmark")
    parser.add_argument("--backend", default="fake", choices=["fake", "acestep"])
    parser.add_argument("--mode", default="lego", choices=["lego", "complete"])
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args(argv)

    if args.backend == "acestep":
        from voders.render.backends.acestep import AceStepBackend

        backend: object = AceStepBackend()
    else:
        backend = FakeAccompanimentBackend()

    options = AccompanimentOptions(mode=args.mode, backend=args.backend, takes=1)
    accompanist = Accompanist(backend, options, Validator(ValidatorConfig()))  # type: ignore[arg-type]
    vocal, score = _make_clip()

    start = time.perf_counter()
    for i in range(args.n):
        accompanist.generate(vocal, score, source_vocal_sample_id=f"bench_{i}", base_seed=i)
    elapsed = time.perf_counter() - start

    per_sample = elapsed / max(1, args.n)
    per_hour = 3600.0 / per_sample if per_sample > 0 else float("inf")
    print(f"backend={args.backend} mode={args.mode} n={args.n}")
    print(f"  {per_sample * 1000:.1f} ms/sample  ->  {per_hour:.0f} samples/hour")
    print("  target: >= 60 samples/hour per A6000-class GPU (research.md Decision 7)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
