"""ACE-Step accompaniment backend (research.md Decision 1; GPU).

Uses the real **ACE-Step** music model (package ``ace_step`` v0.2.0, Apache-2.0) via its
audio-to-audio path, conditioned on the vocal. Two modes:
  * **Complete** — ACE-Step returns a full mix with the singing held in place (lower edit strength).
  * **Lego** — ACE-Step generates the conditioned audio, then a source separator (Demucs — a model
    that splits a mix into vocal/instrument stems) keeps only the non-vocal stems, which are summed
    under the *untouched* original vocal so the layer adds no duplicate voice.

Why a subprocess: ACE-Step pins a stack (``soundfile==0.13.1`` / ``transformers`` / ``spacy`` /
``pytorch_lightning``) that conflicts with this project's deps and lacks Python 3.14 / aarch64
wheels, so it cannot be a direct dependency. ACE-Step therefore runs in **its own environment**
(``acestep_python``) driven by ``acestep_runner.py`` (subprocess); Demucs runs in this project's
``accomp`` extra. The generator/separator are injectable callables so the orchestration — caption
construction, 22.05↔48 kHz bridging, mode dispatch, length-fit — is unit-tested on CPU with fakes.

Hardware-verified on a DGX Spark (NVIDIA GB10, CUDA 13): the Demucs separation path and a real
ACE-Step audio2audio generation both run on the GPU (research.md "Hardware verification").
"""

from __future__ import annotations

import os
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


def _runner_path() -> str:
    return os.path.join(os.path.dirname(__file__), "acestep_runner.py")


class _AceStepGenerator:
    """Real ACE-Step adapter via subprocess into the ACE-Step environment.

    ACE-Step (``ace_step`` v0.2.0, Apache-2.0) can't be a direct dependency of this project (pinned
    ``soundfile==0.13.1`` / ``transformers`` / ``spacy`` conflict and lack Python 3.14 / aarch64
    wheels). So generation runs in ACE-Step's own venv: this adapter writes the reference vocal + a
    JSON spec, runs ``acestep_runner.py`` with that venv's Python (``acestep_python``), and reads
    back the output wav. The runner call args were verified against the real
    ``ACEStepPipeline.__call__`` signature.
    """

    def __init__(
        self,
        model_id: str,
        *,
        acestep_python: str,
        checkpoint_dir: str | None,
        device_id: int,
        cpu_offload: bool,
        mode: str,
        timeout_s: float,
    ) -> None:
        self.model_id = model_id
        self.acestep_python = acestep_python
        self.checkpoint_dir = checkpoint_dir
        self.device_id = device_id
        self.cpu_offload = cpu_offload
        self.mode = mode
        self.timeout_s = timeout_s

    def __call__(
        self, vocal_48k_stereo: np.ndarray, caption: str, seed: int, duration_s: float
    ) -> np.ndarray:  # pragma: no cover - requires the ACE-Step env + weights
        import json
        import subprocess
        import tempfile

        import soundfile as sf

        # "Complete" holds the vocal in place (lower edit strength); "Lego" drifts further since
        # only its non-vocal stems are kept downstream after source separation.
        strength = 0.55 if self.mode == MODE_COMPLETE else 0.75
        with tempfile.TemporaryDirectory(prefix="acestep_") as tmp:
            ref_path = os.path.join(tmp, "ref.wav")
            out_path = os.path.join(tmp, "out.wav")
            spec_path = os.path.join(tmp, "spec.json")
            sf.write(ref_path, vocal_48k_stereo, _NATIVE_SR, subtype="FLOAT")
            spec = {
                "ref": ref_path,
                "out": out_path,
                "prompt": caption,
                "seed": int(seed),
                "duration": float(duration_s),
                "strength": strength,
                "checkpoint_dir": self.checkpoint_dir or "",
                "device_id": self.device_id,
                "cpu_offload": self.cpu_offload,
            }
            with open(spec_path, "w", encoding="utf-8") as fh:
                json.dump(spec, fh)
            subprocess.run(
                [self.acestep_python, _runner_path(), spec_path],
                check=True,
                timeout=self.timeout_s,
            )
            data, _ = sf.read(out_path, dtype="float32", always_2d=True)  # (N, channels)
        return data if data.shape[1] == 2 else np.repeat(data, 2, axis=1)


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
        from demucs.audio import convert_audio

        model = self._model
        assert model is not None
        # demucs htdemucs runs at 44.1 kHz / stereo (verified live): resample the 48 kHz input to
        # the model rate, separate, then resample the summed non-vocal stems back to 48 kHz.
        wav = torch.from_numpy(np.ascontiguousarray(mix_48k_stereo.T, dtype=np.float32))  # (2, N)
        ref = convert_audio(wav, _NATIVE_SR, model.samplerate, model.audio_channels)
        sources = apply_model(model, ref[None], device=self.device)[0]
        names = list(model.sources)
        accomp = sum(sources[i] for i, n in enumerate(names) if n != "vocals")  # drums+bass+other
        accomp = convert_audio(accomp, model.samplerate, _NATIVE_SR, 2)
        return np.ascontiguousarray(accomp.cpu().numpy().T, dtype=np.float32)  # (N, 2)


class AceStepBackend:
    """Vocal-conditioned accompaniment via real ACE-Step (subprocess) + Demucs source separation.

    ACE-Step generation runs in its own environment (``acestep_python``); Demucs separation for Lego
    runs in this project's ``accomp`` extra. ``acestep_python`` / ``checkpoint_dir`` default from
    the ``VODERS_ACESTEP_PYTHON`` / ``VODERS_ACESTEP_CHECKPOINT`` env vars.
    """

    name = "acestep"
    # Real package: ace_step v0.2.0, Apache-2.0 (verified from the repo's setup.py).
    model_version = "0.2.0"
    model_license = "Apache-2.0"
    attribution_text: str | None = None

    def __init__(
        self,
        model_id: str = "ace-step-v1-3.5b",
        *,
        acestep_python: str | None = None,
        checkpoint_dir: str | None = None,
        device_id: int = 0,
        cpu_offload: bool = False,
        timeout_s: float = 1800.0,
        generator: GeneratorFn | None = None,
        separator: SeparatorFn | None = None,
    ) -> None:
        self.model_id = model_id or "ace-step-v1-3.5b"
        self.acestep_python = acestep_python or os.environ.get("VODERS_ACESTEP_PYTHON")
        self.checkpoint_dir = checkpoint_dir or os.environ.get("VODERS_ACESTEP_CHECKPOINT") or None
        self.device_id = device_id
        self.cpu_offload = cpu_offload
        self.timeout_s = timeout_s
        self.device = f"cuda:{device_id}"
        self._generator = generator
        self._separator = separator

    def requires_gpu(self) -> bool:
        return True

    def supports(self, mode: str) -> bool:
        return mode in (MODE_LEGO, MODE_COMPLETE)

    def preflight(self, mode: str) -> str | None:
        """Return a skip reason if the backend can't run here, else None (FR-013).

        Checks that the ACE-Step environment is configured/reachable and (for Lego) that a Demucs
        separator is importable in this env, without loading any weights — so the run degrades
        gracefully. Injected generators/separators (tests) skip the probe entirely.
        """
        if self._generator is not None and (mode == MODE_COMPLETE or self._separator is not None):
            return None
        if not self.acestep_python:
            return "ACE-Step env not configured — set VODERS_ACESTEP_PYTHON to its venv python"
        if not os.path.exists(self.acestep_python):
            return f"ACE-Step python not found at {self.acestep_python!r}"
        if mode == MODE_LEGO:
            import importlib.util

            if importlib.util.find_spec("demucs") is None:
                return "Lego mode needs a Demucs-class separator (install the `accomp` extra)"
        return None

    def _get_generator(self, mode: str) -> GeneratorFn:
        if self._generator is not None:
            return self._generator
        if not self.acestep_python:
            raise RuntimeError("ACE-Step env not configured (set VODERS_ACESTEP_PYTHON)")
        return _AceStepGenerator(
            self.model_id,
            acestep_python=self.acestep_python,
            checkpoint_dir=self.checkpoint_dir,
            device_id=self.device_id,
            cpu_offload=self.cpu_offload,
            mode=mode,
            timeout_s=self.timeout_s,
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
