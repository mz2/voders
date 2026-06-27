"""Per-sample isolated reproduction (FR-013, SC-009).

Re-renders one manifest record from its derived seed and asserts the SC-009 tolerance per lane:
bit-exact for the deterministic and voice-conversion lanes; same ``ValidationVerdict.status`` plus
f0/onset within the validator's tolerances for non-bit-deterministic neural/GPU lanes.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voders.audio import read_wav
from voders.config.models import RunConfig
from voders.manifest.models import ProvenanceRecord
from voders.render.base import RendererLane, RenderRequest
from voders.scores.parse import parse_tsv
from voders.validate.validator import Validator
from voders.voices.registry import VoiceRegistry

_BIT_EXACT_LANES = {"deterministic", "voice_conversion"}


def _original_score_path(config: RunConfig, score_id: str) -> str | None:
    """Resolve the ORIGINAL input score for a score_id from the run's score set.

    Reproduction must drive from the original score, not the carried label .tsv — for the SVS
    re-derive mode the label is re-derived and differs from the input (FR-007).
    """
    pattern = config.scores
    paths = sorted(glob.glob(pattern)) or sorted(glob.glob(str(Path(pattern) / "*.tsv")))
    for p in paths:
        if Path(p).stem == score_id:
            return p
    return None


@dataclass
class ReproResult:
    sample_id: str
    lane: str
    mode: str  # "bit_exact" | "neural_tolerance"
    matched: bool
    detail: str = ""


def reproduce_sample(
    output_root: str | Path,
    record: ProvenanceRecord,
    config: RunConfig,
    lanes: dict[str, RendererLane],
    validator: Validator,
) -> ReproResult:
    root = Path(output_root)
    voices = VoiceRegistry(config.voices)
    voice = voices.get(record.voice_id)
    lane = lanes.get(record.lane)
    if voice is None or lane is None:
        return ReproResult(record.sample_id, record.lane, "n/a", False, "voice or lane unavailable")

    src_path = _original_score_path(config, record.score_id)
    if src_path is None:
        # Fall back to the carried label (correct for lanes whose label == input score).
        src_path = str(root / record.score_path)
    score = parse_tsv(src_path).score
    req = RenderRequest(
        score=score, voice=voice, seed=record.seed, options=config.lane_options(record.lane)
    )
    result = lane.render(req)

    original_audio, _ = read_wav(root / record.audio_path)

    if record.lane in _BIT_EXACT_LANES:
        matched = original_audio.shape == result.audio.shape and np.array_equal(
            original_audio, result.audio
        )
        return ReproResult(
            record.sample_id,
            record.lane,
            "bit_exact",
            matched,
            "" if matched else "re-rendered audio differs bit-for-bit",
        )

    # Neural/GPU lanes: same verdict status + within validator tolerances (SC-009).
    verdict = validator.validate(result.audio, result.label_score)
    same_status = verdict.status == record.verdict.status
    onset_ok = verdict.max_onset_dev_ms <= config.reproduction.neural.onset_ms or verdict.onset_ok
    matched = same_status and onset_ok and verdict.f0_ok
    return ReproResult(
        record.sample_id,
        record.lane,
        "neural_tolerance",
        matched,
        f"status {verdict.status.value} vs {record.verdict.status.value}",
    )
