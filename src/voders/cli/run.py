"""``voders run`` — render a corpus from a Run Config (contract: contracts/cli.md)."""

from __future__ import annotations

import glob
from pathlib import Path

from voders.config.loader import load_config
from voders.config.models import RunConfig
from voders.corpus.orchestrator import LANE_VOICE_KINDS, Orchestrator
from voders.render.registry import build_accompanist, build_augmentor, build_lanes
from voders.validate.validator import Validator


def _project_effective_count(config: RunConfig) -> tuple[int, int]:
    """Projected effective sample count before producing (FR-015): an informational upper bound.

    ``Σ_scores (1 + variants(score)) × voices(lane) × (1 + audio_profiles)`` using the *configured*
    knob counts (offsets length + humanise draws + a volume flag), so the operator can anticipate
    corpus size before any rendering. Transposition drops only ever reduce the realised count.
    """
    paths = sorted(glob.glob(config.scores)) or sorted(
        glob.glob(str(Path(config.scores) / "*.tsv"))
    )
    n_scores = len(paths)

    variants_per_score = 0
    for prof in config.score_augmentation:
        if prof.transpose is not None:
            variants_per_score += len(prof.transpose.offsets)
        if prof.humanize_time is not None:
            variants_per_score += prof.humanize_time.draws
        if prof.volume is not None:
            variants_per_score += 1
    per_score = 1 + variants_per_score

    voice_samples = 0
    for lane_name in config.enabled_lanes():
        kind = LANE_VOICE_KINDS.get(lane_name)
        if kind is None:
            continue
        voice_samples += sum(1 for v in config.voices if v.kind == kind)

    audio_multiplier = 1 + len(config.augmentation_profiles)
    projected = n_scores * per_score * voice_samples * audio_multiplier
    return projected, variants_per_score * n_scores


def run_command(
    config_path: str,
    *,
    output_root: str | None = None,
    resume: bool = False,
    lanes_override: list[str] | None = None,
) -> int:
    config = load_config(config_path)
    if output_root is not None:
        config = config.model_copy(update={"output_root": output_root})
    if lanes_override is not None:
        new_lanes = {
            name: toggle.model_copy(update={"enabled": name in lanes_override})
            for name, toggle in config.lanes.items()
        }
        config = config.model_copy(update={"lanes": new_lanes})

    if config.score_augmentation:
        projected, n_variants = _project_effective_count(config)
        print(
            f"score-augmentation: {len(config.score_augmentation)} profile(s), "
            f"~{n_variants} score variants → projected effective samples ~{projected}"
        )

    lanes = build_lanes(config)
    augmentor = build_augmentor(config)
    accompanist = build_accompanist(config, Validator(config.validator))
    orchestrator = Orchestrator(config, lanes, augmentor=augmentor, accompanist=accompanist)
    summary = orchestrator.run(resume=resume)

    print(f"run {summary.run_id}: attempted={summary.attempted} accepted={summary.accepted}")
    for status, count in sorted(summary.by_status.items()):
        print(f"  {status}: {count}")
    print(f"  learned min_note_ms={summary.learned_threshold.get('min_note_ms')}")
    print(f"  output: {summary.output_root}")
    return 0
