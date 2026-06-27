"""Throughput benchmark for the deterministic lane (T049, SC-005).

Renders and validates a handful of deterministic ``(audio, score)`` pairs over the fixture
scores, measures wall-clock time with ``time.perf_counter``, and reports the extrapolated
throughput in pairs/hour. Exits non-zero if the rate falls below the SC-005 floor of
100 pairs/hour on this machine.

Run via uv::

    uv run python evals/bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from voders.config.models import ValidatorConfig
from voders.render.base import RenderRequest
from voders.render.deterministic import DeterministicLane
from voders.scores.parse import parse_tsv
from voders.validate.validator import Validator
from voders.voices.models import Voice, VoiceKind

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES_DIR = REPO_ROOT / "evals" / "fixtures" / "scores"
DONOR_WAV = REPO_ROOT / "evals" / "fixtures" / "voices" / "donor_ah_synth.wav"

MIN_PAIRS_PER_HOUR = 100.0  # SC-005 floor
N_SAMPLES = 4  # keep the benchmark short


def _donor() -> Voice:
    return Voice(
        voice_id="donor_ah_synth",
        kind=VoiceKind.DETERMINISTIC_DONOR,
        license="CC0 synthetic vowel",
        consent_verified=True,
        model_ref=str(DONOR_WAV),
    )


def main() -> int:
    voice = _donor()
    lane = DeterministicLane()
    validator = Validator(ValidatorConfig(), min_note_ms=50.0)

    score_paths = sorted(SCORES_DIR.glob("*.tsv"))[:N_SAMPLES]
    if not score_paths:
        print("bench: no fixture scores found; run `uv run python evals/make_fixtures.py` first")
        return 1
    scores = [parse_tsv(p).score for p in score_paths]

    start = time.perf_counter()
    produced = 0
    for score in scores:
        result = lane.render(RenderRequest(score=score, voice=voice, seed=0))
        validator.validate(result.audio, result.label_score, f0_method="pyin_f0")
        produced += 1
    elapsed_s = time.perf_counter() - start

    per_sample_s = elapsed_s / produced
    pairs_per_hour = 3600.0 / per_sample_s

    print(f"bench: rendered+validated {produced} deterministic pairs in {elapsed_s:.2f} s")
    print(f"bench: {per_sample_s:.3f} s/pair  ->  {pairs_per_hour:,.0f} pairs/hour")
    print(f"bench: SC-005 floor = {MIN_PAIRS_PER_HOUR:.0f} pairs/hour")

    if pairs_per_hour < MIN_PAIRS_PER_HOUR:
        print("bench: FAIL — throughput below SC-005 floor")
        return 1
    print("bench: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
