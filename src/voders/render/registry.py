"""Lane factory with lazy imports (FR-009, FR-015).

Only the lanes named in the run config are constructed, and GPU lanes are imported lazily so the
CPU baseline (deterministic + validator) never pulls in ``torch`` at module load.
"""

from __future__ import annotations

from voders.config.models import RunConfig
from voders.render.augmentor import Augmentor
from voders.render.base import RendererLane


def build_augmentor(config: RunConfig) -> Augmentor | None:
    """Construct the augmentation post-processor when the augmentation lane is enabled (FR-005)."""
    toggle = config.lanes.get("augmentation")
    if toggle is None or not toggle.enabled or not config.augmentation_profiles:
        return None
    from voders.render.augment import AugmentationChain

    return AugmentationChain(config.augmentation_profiles, config.lane_options("augmentation"))


def build_lanes(config: RunConfig) -> dict[str, RendererLane]:
    """Construct the enabled renderer lanes (FR-015)."""
    lanes: dict[str, RendererLane] = {}
    for name in config.enabled_lanes():
        options = config.lane_options(name)
        if name == "deterministic":
            from voders.render.deterministic import DeterministicLane

            lanes[name] = DeterministicLane()
        elif name == "voice_conversion":
            from voders.render.voiceconv import VoiceConversionLane

            lanes[name] = VoiceConversionLane(options)
        elif name == "svs":
            from voders.render.svs import SvsLane

            lanes[name] = SvsLane(options)
        elif name == "augmentation":
            # The augmentation lane is a post-processor wired separately by the orchestrator.
            continue
    return lanes
