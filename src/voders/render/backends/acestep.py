"""ACE-Step 1.5 XL accompaniment backend (research.md Decision 1; GPU, `accomp` extra).

ACE-Step 1.5 XL-Base is a 4B-parameter Diffusion Transformer ("DiT" — a transformer that denoises
audio latents) over a 48 kHz stereo 1-D VAE. The **Lego** task generates an isolated accompaniment
stem conditioned on the vocal; the **Complete** task decodes a full mix with the vocal held in
place.

Torch and the ACE-Step package are imported lazily inside methods so the CPU baseline never pulls in
torch at module load (FR-009). Output is bridged to 22,050 Hz mono float32 before return
(``audio.bridge_to_corpus_format``, Decision 3).

⚠️ Implementation is a guarded stub. Before wiring the real model, VERIFY (research.md open items):
  1. the exact ``xl-base`` checkpoint LICENSE (must be MIT / Apache-2.0 / CC-BY-class per policy);
  2. that Lego and Complete are exposed by the pinned release's ``generate_music.py``;
  3. the ACE-Step Python package name / inference entry point and its Python 3.14 wheel.
"""

from __future__ import annotations

import numpy as np

from voders.audio import bridge_to_corpus_format
from voders.render.accompaniment_backend import MODE_COMPLETE, MODE_LEGO, BackendOutput
from voders.scores.models import Score

_NATIVE_SR = 48_000


class AceStepBackend:
    """Vocal-conditioned accompaniment via ACE-Step 1.5 XL-Base (GPU)."""

    name = "acestep"
    model_version = "1.5-xl"
    # MIT per the 1.5 repo (Apache-2.0 upstream) — VERIFY the checkpoint LICENSE (open item 1).
    model_license = "MIT"
    attribution_text: str | None = None

    def __init__(self, model_id: str = "ace-step-1.5-xl-base") -> None:
        self.model_id = model_id or "ace-step-1.5-xl-base"
        self._pipe = None  # lazily constructed on first generate()

    def requires_gpu(self) -> bool:
        return True

    def supports(self, mode: str) -> bool:
        return mode in (MODE_LEGO, MODE_COMPLETE)

    def _ensure_pipeline(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: F401  (presence check; real wiring uses it)
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "ACE-Step backend requires the `accomp` extra (uv sync --extra accomp) and a GPU"
            ) from exc
        # Real wiring (deferred — open items 2/3): load xl-base, set Lego/Complete task, captions.
        raise NotImplementedError(
            "AceStepBackend is a verified stub: wire ACE-Step 1.5 XL `generate_music.py` here "
            "after confirming the checkpoint LICENSE, the Lego/Complete tasks, and the package."
        )

    def generate(
        self,
        vocal: np.ndarray,
        score: Score,
        mode: str,
        seed: int,
        options: dict[str, object],
    ) -> BackendOutput:  # pragma: no cover - requires GPU + ACE-Step weights
        self._ensure_pipeline()
        # When wired: run the model at 48 kHz stereo, then bridge to the corpus format:
        #   stem_or_mix = bridge_to_corpus_format(raw_48k_stereo, _NATIVE_SR)
        _ = bridge_to_corpus_format  # referenced so the import is not flagged unused
        raise NotImplementedError
