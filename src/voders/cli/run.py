"""``voders run`` — render a corpus from a Run Config (contract: contracts/cli.md)."""

from __future__ import annotations

from voders.config.loader import load_config
from voders.corpus.orchestrator import Orchestrator
from voders.render.registry import build_accompanist, build_augmentor, build_lanes
from voders.validate.validator import Validator


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
