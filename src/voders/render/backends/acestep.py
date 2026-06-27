"""ACE-Step 1.5 XL accompaniment backend (research.md Decision 1; GPU, `accomp` extra).

ACE-Step 1.5 XL-Base is a 4B-parameter Diffusion Transformer ("DiT" — a transformer that denoises
audio latents) over a 48 kHz stereo 1-D VAE. The **Complete** task conditions on the vocal and
decodes a full mix with the singing held in place; **Lego** generates an accompaniment stem to sum
under the untouched vocal — here realized by generating the conditioned audio and then extracting
the non-vocal stems with a source separator (a Demucs-class model that splits a mix into vocal /
instrument tracks), so the layer summed under the original vocal contains no duplicate voice.

Design for testability + honesty about hardware:
  * The model and the separator are injectable callables (``generator`` / ``separator``). The real
    ones are built lazily (``_AceStepGenerator`` / ``_DemucsSeparator``) and import torch / acestep
    / demucs only when first used, so the CPU baseline never pulls in torch at module load (FR-009).
  * Everything except the two library calls — caption construction, 22.05 kHz↔48 kHz bridging, mode
    dispatch, mono/stereo handling — is concrete and unit-tested on CPU with fakes injected.
  * The two library calls target the real ACE-Step / Demucs APIs. ⚠️ Their exact signatures and the
    1.5 XL Lego/Complete task names are unverified against a pinned release (research.md open items
    1–3); confirm on a GPU run with weights installed and adjust the adapters below if needed. The
    backend declares ``model_license`` honestly — VERIFY the checkpoint LICENSE before release.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from voders.audio import bridge_to_corpus_format, to_mono_float32
from voders.constants import SAMPLE_RATE
from voders.render.accompaniment_backend import MODE_COMPLETE, MODE_LEGO, BackendOutput
from voders.scores.models import Score

_NATIVE_SR = 48_000

# A generator maps (vocal_48k_stereo, caption, seed, duration_s) -> 48 kHz stereo audio (N, 2).
GeneratorFn = Callable[[np.ndarray, str, int, float], np.ndarray]
# A separator maps a 48 kHz stereo mix (N, 2) -> the 48 kHz stereo accompaniment (N, 2) (no vocal).
SeparatorFn = Callable[[np.ndarray], np.ndarray]


def build_caption(score: Score, options: dict[str, object]) -> str:
    """Build the ACE-Step text prompt from the run options (FR-005).

    Free-time material gets ambient / rubato / no-fixed-tempo tags and never a BPM, so the model is
    not nudged toward a metrical pulse that would fight the phrasing; otherwise a BPM tag is added
    when one was supplied. The target instrument is always named, biased toward sustained/textural
    accompaniment.
    """
    instrument = str(options.get("target_instrument") or "sustained pad")
    tags = [instrument, "instrumental accompaniment", "no vocals"]
    if options.get("free_time", True):
        tags = ["ambient", "rubato", "free time", "no fixed tempo", *tags]
    else:
        bpm = options.get("bpm")
        if isinstance(bpm, int | float):
            tags.append(f"{int(round(float(bpm)))} bpm")
    return ", ".join(tags)


def _to_native_stereo(vocal: np.ndarray, native_sr: int = _NATIVE_SR) -> np.ndarray:
    """Resample a 22,050 Hz mono vocal to ``native_sr`` and duplicate to stereo for conditioning."""
    import librosa

    mono = to_mono_float32(vocal)
    if mono.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    up = (
        mono
        if native_sr == SAMPLE_RATE
        else librosa.resample(mono, orig_sr=SAMPLE_RATE, target_sr=native_sr)
    )
    up = np.ascontiguousarray(up, dtype=np.float32)
    return np.stack([up, up], axis=-1)


class _AceStepGenerator:
    """Real ACE-Step 1.5 XL adapter (GPU). Built lazily; signature verified on a GPU run."""

    def __init__(
        self,
        model_id: str,
        *,
        checkpoint_dir: str | None,
        dtype: str,
        device: str,
        mode: str,
    ) -> None:
        self.model_id = model_id
        self.checkpoint_dir = checkpoint_dir
        self.dtype = dtype
        self.device = device
        self.mode = mode
        self._pipe = None

    def _ensure(self) -> None:
        if self._pipe is not None:
            return
        try:
            from acestep.pipeline_ace_step import ACEStepPipeline
        except ImportError as exc:  # pragma: no cover - exercised only without the extra/weights
            raise RuntimeError(
                "ACE-Step backend requires the `accomp` extra and the ACE-Step package "
                "(uv sync --extra accomp; install ACE-Step 1.5 XL weights). See research.md open "
                "item 3 for the package name / Python 3.14 wheel."
            ) from exc
        self._pipe = ACEStepPipeline(
            checkpoint_dir=self.checkpoint_dir, dtype=self.dtype, device=self.device
        )

    def __call__(
        self, vocal_48k_stereo: np.ndarray, caption: str, seed: int, duration_s: float
    ) -> np.ndarray:  # pragma: no cover - requires GPU + weights
        self._ensure()
        assert self._pipe is not None
        # ⚠️ VERIFY against the pinned release's generate_music.py (open item 2). Both Lego and
        # Complete condition on the vocal via ACE-Step's audio-to-audio editing path; "Complete"
        # holds the vocal in place (lower edit strength), "Lego" lets the generation drift further
        # since only its non-vocal stems are kept downstream.
        strength = 0.55 if self.mode == MODE_COMPLETE else 0.75
        result = self._pipe(
            prompt=caption,
            audio_duration=float(duration_s),
            audio2audio_enable=True,
            ref_audio_input=vocal_48k_stereo,
            ref_audio_strength=strength,
            manual_seeds=[int(seed)],
            infer_step=60,
            guidance_scale=7.5,
            format="wav",
        )
        return _normalize_pipeline_output(result)


def _normalize_pipeline_output(result: object) -> np.ndarray:  # pragma: no cover - GPU path
    """Coerce an ACE-Step return (ndarray, tensor, or saved path/list) to a 48 kHz stereo array."""
    if isinstance(result, list | tuple) and result:
        result = result[0]
    if isinstance(result, str):
        from voders.audio import read_wav

        data, _ = read_wav(result)
        return np.stack([data, data], axis=-1) if data.ndim == 1 else data
    arr = np.asarray(result, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=-1)
    return arr


class _DemucsSeparator:
    """Real Demucs adapter (GPU): returns the summed non-vocal stems of a 48 kHz stereo mix."""

    def __init__(self, device: str) -> None:
        self.device = device
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            from demucs.pretrained import get_model
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "Lego mode needs a Demucs-class separator; install it with the `accomp` extra."
            ) from exc
        self._model = get_model("htdemucs")
        self._model.to(self.device)

    def __call__(self, mix_48k_stereo: np.ndarray) -> np.ndarray:  # pragma: no cover - GPU path
        self._ensure()
        import torch
        from demucs.apply import apply_model

        assert self._model is not None
        wav = torch.from_numpy(np.ascontiguousarray(mix_48k_stereo.T, dtype=np.float32))
        sources = apply_model(self._model, wav[None], device=self.device)[0]
        names = list(self._model.sources)
        accomp = sum(
            sources[i] for i, n in enumerate(names) if n != "vocals"
        )  # drums + bass + other
        return np.ascontiguousarray(accomp.cpu().numpy().T, dtype=np.float32)


class AceStepBackend:
    """Vocal-conditioned accompaniment via ACE-Step 1.5 XL-Base + source separation (GPU)."""

    name = "acestep"
    model_version = "1.5-xl"
    # MIT per the 1.5 repo (Apache-2.0 upstream) — VERIFY the checkpoint LICENSE (open item 1).
    model_license = "MIT"
    attribution_text: str | None = None

    def __init__(
        self,
        model_id: str = "ace-step-1.5-xl-base",
        *,
        checkpoint_dir: str | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        generator: GeneratorFn | None = None,
        separator: SeparatorFn | None = None,
    ) -> None:
        self.model_id = model_id or "ace-step-1.5-xl-base"
        self.checkpoint_dir = checkpoint_dir
        self.dtype = dtype
        self.device = device
        self._generator = generator
        self._separator = separator

    def requires_gpu(self) -> bool:
        return True

    def supports(self, mode: str) -> bool:
        return mode in (MODE_LEGO, MODE_COMPLETE)

    def preflight(self, mode: str) -> str | None:
        """Return a skip reason if the backend can't run here, else None (FR-013).

        Probes for torch + an available GPU and the ACE-Step package (and Demucs for Lego) without
        loading any weights, so the run degrades gracefully instead of crashing mid-generation.
        Injected generators/separators (tests) skip the probe entirely.
        """
        if self._generator is not None and (mode == MODE_COMPLETE or self._separator is not None):
            return None
        try:
            import torch
        except ImportError:
            return "torch not installed — install the `accomp` extra (uv sync --extra accomp)"
        if not torch.cuda.is_available():
            return "no CUDA GPU available for the ACE-Step backend"
        import importlib.util

        if importlib.util.find_spec("acestep") is None:
            return "the ACE-Step package is not installed (research.md open item 3)"
        if mode == MODE_LEGO and importlib.util.find_spec("demucs") is None:
            return "Lego mode needs a Demucs-class separator (install the `accomp` extra)"
        return None

    def _get_generator(self, mode: str) -> GeneratorFn:
        if self._generator is not None:
            return self._generator
        return _AceStepGenerator(
            self.model_id,
            checkpoint_dir=self.checkpoint_dir,
            dtype=self.dtype,
            device=self.device,
            mode=mode,
        )

    def _get_separator(self) -> SeparatorFn:
        if self._separator is not None:
            return self._separator
        return _DemucsSeparator(self.device)

    def generate(
        self,
        vocal: np.ndarray,
        score: Score,
        mode: str,
        seed: int,
        options: dict[str, object],
    ) -> BackendOutput:
        """Generate accompaniment for the vocal and return it in the corpus format (Decision 3)."""
        if mode not in (MODE_LEGO, MODE_COMPLETE):
            raise ValueError(f"unknown accompaniment mode {mode!r}; expected lego | complete")
        vocal = to_mono_float32(vocal)
        duration_s = max(1.0, vocal.size / SAMPLE_RATE)
        caption = build_caption(score, options)
        ref = _to_native_stereo(vocal)

        raw_48k_stereo = self._get_generator(mode)(ref, caption, seed, duration_s)

        if mode == MODE_COMPLETE:
            mix = bridge_to_corpus_format(raw_48k_stereo, _NATIVE_SR)
            mix = _fit_length(mix, vocal.size)
            return BackendOutput(accompaniment_stem=None, mix=mix, native_sr=_NATIVE_SR)

        accomp_48k_stereo = self._get_separator()(raw_48k_stereo)
        stem = bridge_to_corpus_format(accomp_48k_stereo, _NATIVE_SR)
        stem = _fit_length(stem, vocal.size)
        return BackendOutput(accompaniment_stem=stem, mix=None, native_sr=_NATIVE_SR)


def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad mono audio to exactly ``n`` samples (generated length can drift ±frames)."""
    audio = to_mono_float32(audio)
    if audio.size == n:
        return audio
    if audio.size > n:
        return np.ascontiguousarray(audio[:n], dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: audio.size] = audio
    return out
