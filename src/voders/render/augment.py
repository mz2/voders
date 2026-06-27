"""Label-safe augmentation chain (FR-005, FR-014, US3; research Decision 5).

Domain randomization makes synthetic vocals look like real "in-the-mix" pop singing: room
reverberation, lossy-codec band-limiting, and mixing against accompaniment at swept
signal-to-accompaniment ratios. Every step here is **label-safe** — it must NOT move the vocal's
onsets/offsets or shift its pitch, so the paired score's note rows stay byte-identical (FR-005).

Label safety is achieved with three deliberate DSP choices:

* ``reverb_ir`` convolves with a synthetic room impulse response whose **first sample is the direct
  impulse at t=0**, so the dry vocal arrives with zero added onset delay; only the decaying tail is
  added afterwards.
* ``codec`` band-limits with a **zero-phase** filter (``scipy.signal.filtfilt``), which has no group
  delay, so no onset/offset smear.
* ``accompaniment_mix`` only adds an independent accompaniment stem; the vocal stem is untouched.
  Any pitch/time perturbation (not enabled here) would be applied to the accompaniment alone.

This module is a CPU ``numpy``/``scipy.signal`` approximation. The production backends named in the
plan are ``pedalboard`` (reverb/EQ/compression/codec-like effects) and ``torchaudio`` (real
MP3/Opus codec round-trips); they are not imported here so the CPU baseline stays torch-free.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import butter, fftconvolve, filtfilt, lfilter

from voders.audio import to_mono_float32
from voders.config.models import AugmentationProfileConfig
from voders.constants import SAMPLE_RATE
from voders.seeds import rng as rng_for_seed

_EPS = 1e-12

# A step takes (vocal, accompaniment_or_None, rng, params) and returns
# (new_vocal, new_accompaniment_or_None). Accompaniment is carried as a separate stem so the
# vocal-to-accompaniment SNR can be reported and so perturbations never touch the vocal.
StepFn = Callable[
    [np.ndarray, "np.ndarray | None", np.random.Generator, dict[str, object], int],
    "tuple[np.ndarray, np.ndarray | None]",
]


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool):  # bool is an int subclass; treat as "not provided"
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _synth_room_ir(
    rng: np.random.Generator, *, decay_s: float, wet: float, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """A short decaying-noise room impulse response with the direct sound at sample 0.

    The first sample is the unit direct impulse, so convolving adds NO onset delay (label-safe);
    the exponentially-decaying noise tail is appended after it at level ``wet``.
    """
    n = max(2, int(round(decay_s * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-t / (decay_s / 4.0))
    tail = rng.standard_normal(n) * env
    tail[0] = 0.0  # the tail begins strictly after the direct sound
    peak = float(np.max(np.abs(tail)))
    if peak > _EPS:
        tail = tail / peak
    ir = np.zeros(n, dtype=np.float64)
    ir[0] = 1.0  # direct impulse at t=0 -> onsets stay byte-aligned
    ir += wet * tail
    return ir


def _step_reverb_ir(
    vocal: np.ndarray,
    accomp: np.ndarray | None,
    rng: np.random.Generator,
    params: dict[str, object],
    sr: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convolve the vocal with a synthetic room IR; onsets stay aligned (direct sound at t=0)."""
    decay_s = _as_float(params.get("reverb_decay_s"), 0.10)
    wet = _as_float(params.get("reverb_wet"), 0.18)
    ir = _synth_room_ir(rng, decay_s=decay_s, wet=wet, sr=sr)
    wet_vocal = fftconvolve(vocal.astype(np.float64), ir, mode="full")[: vocal.size]
    return wet_vocal.astype(np.float32), accomp


def _step_codec(
    vocal: np.ndarray,
    accomp: np.ndarray | None,
    rng: np.random.Generator,
    params: dict[str, object],
    sr: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Band-limit the vocal with a ZERO-PHASE lowpass to mimic lossy-codec roll-off.

    ``filtfilt`` is forward-backward, so there is no group delay and onsets/offsets do not move
    (label-safe). torchaudio provides the real MP3/Opus codec round-trip in production.
    """
    nyq = sr / 2.0
    bitrate_kbps = _as_float(params.get("bitrate_kbps"), 96.0)
    cutoff = _as_float(params.get("cutoff_hz"), min(nyq * 0.95, 80.0 * bitrate_kbps))
    cutoff = float(np.clip(cutoff, 1000.0, nyq * 0.99))
    order = 8
    b, a = butter(order, cutoff / nyq, btype="low")
    x = vocal.astype(np.float64)
    padlen = 3 * max(len(a), len(b))
    if x.size <= padlen:  # too short for filtfilt's edge padding; leave untouched
        return vocal, accomp
    filtered = filtfilt(b, a, x)
    return filtered.astype(np.float32), accomp


def _synth_accompaniment(n: int, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A seeded synthetic accompaniment stem: lowpass-filtered noise (a simple pad).

    The accompaniment may have its own group delay — it is not the vocal, so it is label-safe.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    noise = rng.standard_normal(n)
    b, a = butter(2, 2000.0 / (sr / 2.0), btype="low")
    pad = lfilter(b, a, noise)
    return np.asarray(pad, dtype=np.float64)


def _step_accompaniment_mix(
    vocal: np.ndarray,
    accomp: np.ndarray | None,
    rng: np.random.Generator,
    params: dict[str, object],
    sr: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Add a synthetic accompaniment at a target vocal-to-accompaniment SNR sampled from params.

    ``params["snr_db"]`` is a list (one is chosen via ``rng``) or a scalar target in dB. The
    accompaniment is scaled so 20*log10(rms_vocal/rms_accomp) equals the target.
    """
    snr_param = params.get("snr_db", 12.0)
    if isinstance(snr_param, list | tuple) and len(snr_param) > 0:
        choices = np.asarray([float(v) for v in snr_param], dtype=np.float64)
        target_snr = float(rng.choice(choices))
    else:
        target_snr = _as_float(snr_param, 12.0)

    pad = _synth_accompaniment(vocal.size, rng, sr)
    rms_vocal = _rms(vocal)
    rms_pad = _rms(pad)
    if rms_pad <= _EPS or rms_vocal <= _EPS:
        return vocal, accomp
    desired_rms_pad = rms_vocal / (10.0 ** (target_snr / 20.0))
    pad = pad * (desired_rms_pad / rms_pad)
    new_accomp = pad if accomp is None else accomp + pad
    return vocal, new_accomp


_STEPS: dict[str, StepFn] = {
    "reverb_ir": _step_reverb_ir,
    "codec": _step_codec,
    "accompaniment_mix": _step_accompaniment_mix,
}


class AugmentationChain:
    """An ordered registry of label-safe augmentation steps (satisfies the ``Augmentor`` protocol).

    Constructed by ``voders.render.registry.build_augmentor`` as
    ``AugmentationChain(config.augmentation_profiles, config.lane_options("augmentation"))``.
    """

    def __init__(
        self,
        profiles: list[AugmentationProfileConfig],
        options: dict[str, object] | None = None,
        sr: int = SAMPLE_RATE,
    ) -> None:
        self.profiles = {p.profile_id: p for p in profiles}
        self.options = options or {}
        # Filters (codec lowpass, reverb IR length, accompaniment lowpass) are designed at this
        # rate. Defaults to the corpus rate (22.05 kHz); training passes its own rate (16 kHz) so
        # cutoffs/decays are physically correct for the audio actually being augmented.
        self.sr = sr

    def profile_ids(self) -> list[str]:
        return list(self.profiles)

    def apply(
        self, audio: np.ndarray, profile_id: str, seed: int
    ) -> tuple[np.ndarray, float | None]:
        """Run a profile's steps in order; return (augmented float32 audio, accompaniment SNR dB).

        Randomness is threaded through ``voders.seeds.rng(seed)`` so the result is reproducible from
        the seed alone (FR-013): ``apply`` twice with the same seed yields byte-identical arrays.
        The returned SNR is the final vocal-to-accompaniment ratio when an ``accompaniment_mix``
        step ran, else ``None``. Unknown step names are skipped.
        """
        try:
            profile = self.profiles[profile_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown augmentation profile {profile_id!r}; configured: {sorted(self.profiles)}"
            ) from exc

        rng = rng_for_seed(seed)
        vocal = to_mono_float32(np.array(audio, copy=True))
        accomp: np.ndarray | None = None

        for step_name in profile.steps:
            step = _STEPS.get(step_name)
            if step is None:  # label-safe: silently skip unimplemented/unknown steps
                continue
            vocal, accomp = step(vocal, accomp, rng, profile.params, self.sr)

        snr_db: float | None = None
        if accomp is not None:
            rms_vocal = _rms(vocal)
            rms_accomp = _rms(accomp)
            if rms_accomp > _EPS and rms_vocal > _EPS:
                snr_db = 20.0 * np.log10(rms_vocal / rms_accomp)
            mixed = vocal + accomp
        else:
            mixed = vocal

        # Keep the normal output below full scale; scaling vocal and accompaniment together leaves
        # the reported SNR unchanged. Intentional clipping in a test is induced outside apply().
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 0.99:
            mixed = mixed * (0.99 / peak)

        return to_mono_float32(mixed), snr_db
