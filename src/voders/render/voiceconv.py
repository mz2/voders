"""Voice-conversion (timbre) lane (FR-004, US2; research Decision 4).

Voice conversion ("VC") changes a singer's *timbre* while keeping the score-derived pitch contour
(fundamental frequency, "f0") unchanged, so onset/offset/pitch alignment is preserved and
``label_score == req.score``.

Backends (``options["backend"]``):

* ``"world"`` (default, CPU): delegates to the deterministic WORLD lane, which builds f0 directly
  from the score and uses the voice's donor recording as the timbre source. No GPU, no ``torch``.
* ``"rvc"``: the real RVC toolkit, run **out of process** in its own uv project (``backends/rvc``,
  Python 3.10) via :mod:`voders.render.backend_bridge`. The core renders score-aligned audio with
  WORLD, then RVC converts only the timbre using a *consented* target voice model
  (``voice.model_ref``, an RVC ``.pth``); RVC keeps the input audio's f0, so the labels are
  preserved (``auto_predict_f0=False`` equivalent).
* ``"sovits"``: so-vits-svc — a GitHub repo (not a PyPI package); wire it as another backend
  project when its weights are available.
"""

from __future__ import annotations

from voders.render.base import RenderRequest, RenderResult
from voders.voices.models import Voice, VoiceKind

_CPU_BACKENDS = frozenset({"world"})
# Out-of-process backends → (backend project dir under backends/, worker module). Each runs the
# real toolkit in its own uv project. "rvc" uses a trained .pth; "seedvc" is zero-shot (model_ref
# is a reference audio clip — e.g. a consented VocalSet/VCTK donor).
_SUBPROCESS_BACKENDS = {
    "rvc": ("rvc", "voders_rvc_backend.worker"),
    "seedvc": ("seedvc", "voders_seedvc_backend.worker"),
}
_REPO_BACKENDS = frozenset({"sovits"})  # GitHub repo + weights, not yet wired


class VoiceConversionLane:
    """Timbre fan-out lane keeping the score f0 (FR-004)."""

    name = "voice_conversion"

    def __init__(self, options: dict[str, object] | None = None) -> None:
        self.options = options or {}
        self.backend = str(self.options.get("backend", "world"))

    def requires_gpu(self) -> bool:
        """False: ``world`` is CPU and ``rvc`` runs out-of-process, so the core needs no GPU."""
        return False

    def render(self, req: RenderRequest) -> RenderResult:
        backend = str(req.options.get("backend", self.backend))
        from voders.render.deterministic import DeterministicLane

        if backend in _CPU_BACKENDS:
            # WORLD constructs f0 from the score (never predicted/altered) using the donor timbre.
            result = DeterministicLane().render(req)
            return RenderResult(
                audio=result.audio, label_score=req.score, notes={"backend": backend}
            )

        if backend in _SUBPROCESS_BACKENDS:
            if not req.voice.model_ref:
                raise RuntimeError(
                    f"voice-conversion backend {backend!r} requires a consented target voice "
                    "(voice.model_ref: an RVC .pth, or a reference clip for zero-shot seedvc) "
                    "(FR-011)"
                )
            base_donor = req.options.get("base_donor") or self.options.get("base_donor")
            if not base_donor:
                raise RuntimeError(
                    f"voice-conversion backend {backend!r} needs a 'base_donor' (a donor vowel "
                    "wav) for the WORLD source render that the converter then re-timbres"
                )
            from voders.render.backend_bridge import convert_via_backend

            # 1) WORLD renders score-aligned source audio from a donor vowel (timbre irrelevant —
            #    the converter replaces it); 2) the toolkit converts to the target voice, keeping
            #    its f0 so the labels are preserved.
            project, module = _SUBPROCESS_BACKENDS[backend]
            donor = Voice(
                voice_id=f"{req.voice.voice_id}__base",
                kind=VoiceKind.DETERMINISTIC_DONOR,
                consent_verified=True,
                model_ref=str(base_donor),
            )
            base = (
                DeterministicLane()
                .render(
                    RenderRequest(score=req.score, voice=donor, seed=req.seed, options=req.options)
                )
                .audio
            )
            converted = convert_via_backend(
                project,
                module,
                base,
                model_ref=req.voice.model_ref,
                device=str(req.options.get("device", "auto")),
            )
            return RenderResult(audio=converted, label_score=req.score, notes={"backend": backend})

        if backend in _REPO_BACKENDS:
            raise RuntimeError(
                f"voice-conversion backend {backend!r} is a GitHub repo + weights; "
                "wire it as a backend project under backends/ before use"
            )

        raise ValueError(
            f"unknown voice-conversion backend {backend!r}; expected one of "
            f"{sorted(_CPU_BACKENDS | set(_SUBPROCESS_BACKENDS) | _REPO_BACKENDS)}"
        )
