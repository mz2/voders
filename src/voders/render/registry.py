"""Lane factory with lazy imports (FR-009, FR-015).

Only the lanes named in the run config are constructed, and GPU lanes are imported lazily so the
CPU baseline (deterministic + validator) never pulls in ``torch`` at module load.
"""

from __future__ import annotations

import logging

from voders.config.models import RunConfig, parse_accompaniment_options
from voders.render.augmentor import Augmentor
from voders.render.base import RendererLane

_LOG = logging.getLogger(__name__)


def build_augmentor(config: RunConfig) -> Augmentor | None:
    """Construct the augmentation post-processor when the augmentation lane is enabled (FR-005)."""
    toggle = config.lanes.get("augmentation")
    if toggle is None or not toggle.enabled or not config.augmentation_profiles:
        return None
    from voders.render.augment import AugmentationChain

    return AugmentationChain(config.augmentation_profiles, config.lane_options("augmentation"))


def build_accompanist(config: RunConfig, validator: object, timing: object | None = None):  # noqa: ANN201
    """Construct the accompaniment stage when its lane is enabled (spec 002; FR-006/009/013).

    Returns ``None`` when the lane is disabled, the backend is unavailable (GPU/`accomp` extra not
    installed), or the license policy excludes the backend — so the run continues with other lanes
    and a logged skip. The GPU ACE-Step backend is imported lazily so the CPU baseline stays
    torch-free (FR-009).
    """
    toggle = config.lanes.get("accompaniment")
    if toggle is None or not toggle.enabled:
        return None
    from voders.render.accompaniment import Accompanist
    from voders.render.accompaniment_backend import FakeAccompanimentBackend

    options = parse_accompaniment_options(config.lane_options("accompaniment"))
    if options.backend == "fake":
        backend: object = FakeAccompanimentBackend()
    elif options.backend == "acestep":
        try:
            from voders.render.backends.acestep import AceStepBackend

            backend = AceStepBackend(options.model_id)
        except Exception as exc:  # noqa: BLE001 - any import/GPU failure → graceful skip (FR-013)
            _LOG.warning("accompaniment: acestep backend unavailable (%s) — skipping lane", exc)
            return None
        preflight = backend.preflight(options.mode)
        if preflight is not None:
            _LOG.warning("accompaniment: %s — skipping lane", preflight)
            return None
    else:
        _LOG.warning("accompaniment: unknown backend %r — skipping lane", options.backend)
        return None

    accompanist = Accompanist(backend, options, validator, timing=timing)  # type: ignore[arg-type]
    reason = accompanist.skip_reason()
    if reason is not None:
        _LOG.warning("accompaniment: %s", reason)
        return None
    return accompanist


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
