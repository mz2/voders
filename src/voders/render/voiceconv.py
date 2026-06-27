"""Voice-conversion (timbre) lane (FR-004, US2; research Decision 4).

Voice conversion ("VC") changes a singer's *timbre* while keeping the score-derived pitch contour
(fundamental frequency, "f0") unchanged. The load-bearing invariant is ``auto_predict_f0=False``
(the flag the production RVC / so-vits-svc toolkits expose): the converted audio keeps the score's
f0, so onset/offset/pitch alignment is preserved by construction and ``label_score == req.score``.

Backends (``options["backend"]``):

* ``"world"`` (default, CPU): delegates to the deterministic WORLD lane, which builds f0 directly
  from the score and uses the voice's donor recording as the timbre source. No GPU, no ``torch``.
* ``"rvc"`` / ``"sovits"`` (GPU): the production neural VC toolkits. ``torch`` is imported lazily
  *inside* ``render`` so the CPU baseline never pulls it in at module load; since the GPU toolkit is
  not installed here, ``render`` raises a clear :class:`RuntimeError`.
"""

from __future__ import annotations

from voders.render.base import RenderRequest, RenderResult

_CPU_BACKENDS = frozenset({"world"})
_GPU_BACKENDS = frozenset({"rvc", "sovits"})


class VoiceConversionLane:
    """Timbre fan-out lane keeping the score f0 (``auto_predict_f0=False``) (FR-004)."""

    name = "voice_conversion"

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}
        self.backend = str(self.options.get("backend", "world"))

    def requires_gpu(self) -> bool:
        """False for the CPU ``world`` backend; True for the GPU ``rvc`` / ``sovits`` backends."""
        return self.backend in _GPU_BACKENDS

    def render(self, req: RenderRequest) -> RenderResult:
        backend = str(req.options.get("backend", self.backend))

        if backend in _CPU_BACKENDS:
            # Delegate to the deterministic WORLD lane: it constructs f0 from the score (f0 is never
            # predicted/altered — ``auto_predict_f0=False``) using the voice's donor as the timbre.
            from voders.render.deterministic import DeterministicLane

            result = DeterministicLane().render(req)
            notes = dict(result.notes)
            notes["backend"] = backend
            # label_score == req.score by construction (exact, row-for-row).
            return RenderResult(audio=result.audio, label_score=req.score, notes=notes)

        if backend in _GPU_BACKENDS:
            # Lazily import torch so the CPU baseline never loads it at module import time.
            import importlib

            try:
                importlib.import_module("torch")
            except ImportError as exc:
                raise RuntimeError(
                    f"voice-conversion backend {backend!r} requires the 'gpu' extra "
                    f"(torch is not installed)"
                ) from exc
            raise RuntimeError(
                f"voice-conversion backend {backend!r} requires the GPU toolkit "
                f"(install the 'gpu' extra and the {backend} model weights); not available here"
            )

        raise ValueError(
            f"unknown voice-conversion backend {backend!r}; expected one of "
            f"{sorted(_CPU_BACKENDS | _GPU_BACKENDS)}"
        )
