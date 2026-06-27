"""Neural mel-vocoder (Vocos) enhancement for the deterministic lane (alignment-safe).

Re-vocodes a rendered waveform through Vocos: audio → mel-spectrogram → ``Vocos.decode`` → audio.
The mel preserves timing and pitch, so onsets/offsets/f0 — the labels — are preserved; Vocos only
re-synthesises the waveform with a modern neural vocoder for a smoother, more natural timbre than
WORLD. ``torch``/``vocos`` are imported lazily so the CPU baseline never loads them (FR-009); the
GPU is used when available, else CPU.

Note: a neural vocoder is bounded by the mel it is given — it naturalises the waveform but cannot
add detail absent from the input. And Vocos output is not bit-exact across hardware, so a run that
enables ``vocoder: vocos`` reproduces under the neural tolerance (SC-009), not bit-exactly.

**Empirical finding (experimental):** the plain mel Vocos model is *not* f0-conditioned, and in
practice it detunes the pitch enough that the rendered notes fall outside the validator's ±25-cent
tolerance — so the alignment gate *rejects* these samples. The gate is working as designed; the
lesson is that an **f0-conditioned** vocoder (BigVGAN-f0 or a Neural Source-Filter model) is the
alignment-safe way to get neural-vocoder realism. This option is kept as a runnable experiment, not
a default.
"""

from __future__ import annotations

import functools

import numpy as np

from voders.constants import SAMPLE_RATE

_VOCOS_SR = 24_000
_VOCOS_MODEL = "charactr/vocos-mel-24khz"


@functools.lru_cache(maxsize=2)
def _load(device: str):  # noqa: ANN202 - returns a Vocos model
    from vocos import Vocos

    model = Vocos.from_pretrained(_VOCOS_MODEL).to(device)
    model.eval()
    return model


def enhance(audio: np.ndarray, sr: int = SAMPLE_RATE, device: str = "auto") -> np.ndarray:
    """Re-vocode ``audio`` through Vocos, returning audio at the original sample rate."""
    import torch
    import torchaudio

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load(device)

    x = torch.tensor(np.asarray(audio, dtype=np.float32), device=device)
    if sr != _VOCOS_SR:
        x = torchaudio.functional.resample(x, sr, _VOCOS_SR)
    with torch.no_grad():
        feats = model.feature_extractor(x.unsqueeze(0))
        out = model.decode(feats).squeeze(0)
    if sr != _VOCOS_SR:
        out = torchaudio.functional.resample(out, _VOCOS_SR, sr)
    result = out.detach().cpu().numpy().astype(np.float32)
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > 0:
        result = (result * (0.9 / peak)).astype(np.float32)
    return result
