"""Dynamic, label-safe audio augmentation applied at training time.

Instead of consuming pre-baked augmented corpus variants from disk, this augments the
clean render on the fly inside the dataset. It reuses the offline corpus augmentation
chain (``voders.render.augment.AugmentationChain``) and, optionally, the alignment
validator (``voders.validate.Validator``) to gate every augmented sample by **rejection
sampling**: propose a profile + seed, augment, re-derive labels with pyin and compare to
the score, accept if within tolerance, otherwise resample up to a retry budget. When no
proposal passes, fall back to the clean audio (always a valid, label-correct sample).

The augmentation steps here are label-preserving by construction (FR-005), so validation
is optional: enable it (default) to also admit label-*risky* profiles safely, or disable
it to skip the pyin cost for the provably-safe reverb/codec/mix chain.

By default validation runs ``crepe_f0`` (torchcrepe) on the GPU; because that initializes
a CUDA context, the augment+validate loop must run in the main training process
(``num_workers=0`` for the train loader) rather than in forked DataLoader workers.
``pyin_f0`` is the CPU fallback.

Sample rate: both the augmentation chain and the validator are built at the training rate
(``training.constants.SAMPLE_RATE``, 16 kHz), not voders' 22.05 kHz corpus default, so codec
cutoffs / reverb decays and onset/offset timing are all physically correct for the audio
actually being augmented.
"""

from __future__ import annotations

import numpy as np

from ..constants import SAMPLE_RATE


def default_profiles():
    """A small built-in set of label-safe profiles, used when no config is supplied."""
    from voders.config.models import AugmentationProfileConfig

    return [
        AugmentationProfileConfig(
            profile_id="room_reverb",
            steps=["reverb_ir"],
            params={"reverb_decay_s": 0.18, "reverb_wet": 0.25},
        ),
        AugmentationProfileConfig(
            profile_id="phone_codec",
            steps=["codec"],
            params={"bitrate_kbps": 24.0},
        ),
        AugmentationProfileConfig(
            profile_id="noisy_codec",
            steps=["codec", "accompaniment_mix"],
            params={"bitrate_kbps": 64.0, "snr_db": [6.0, 12.0, 18.0]},
        ),
        AugmentationProfileConfig(
            profile_id="room_and_codec",
            steps=["reverb_ir", "codec"],
            params={"reverb_decay_s": 0.12, "reverb_wet": 0.20, "bitrate_kbps": 96.0},
        ),
        AugmentationProfileConfig(
            profile_id="gain_room",
            steps=["gain", "reverb_ir"],
            params={"gain_db": [-12.0, 3.0], "reverb_decay_s": 0.15, "reverb_wet": 0.2},
        ),
    ]


def profiles_from_config(path):
    """Load ``augmentation_profiles`` from a run config (e.g. the corpus's resolved YAML)."""
    from voders.config.loader import load_config

    profiles = list(load_config(path).augmentation_profiles)
    if not profiles:
        raise ValueError(f"{path} has no augmentation_profiles")
    return profiles


class DynamicAugmentor:
    """Apply a randomly-chosen, optionally-validated augmentation profile per sample.

    Reproducible from ``seed`` for a fixed worker layout and access order; decorrelated
    across DataLoader workers and varied across epochs via a per-worker RNG that advances
    every call.
    """

    def __init__(
        self,
        profiles,
        *,
        validate: bool = True,
        f0_method: str = "crepe_f0",
        f0_device: str = "auto",
        f0_model: str = "tiny",
        f0_decoder: str = "argmax",
        pitch_shift_semitones: float = 0.0,
        time_stretch_amount: float = 0.0,
        max_retries: int = 4,
        augment_prob: float = 1.0,
        seed: int = 0,
        sr: int = SAMPLE_RATE,
    ) -> None:
        from voders.render.augment import AugmentationChain

        # Pass the training sample rate so the chain designs its codec/reverb filters for the
        # audio actually being augmented (16 kHz here, not voders' 22.05 kHz corpus default).
        self.chain = AugmentationChain(profiles, sr=int(sr))
        self.profile_ids = self.chain.profile_ids()
        if not self.profile_ids:
            raise ValueError("DynamicAugmentor requires at least one augmentation profile")
        # Label-transforming augmentations (change the score, not just the audio):
        #   pitch_shift_semitones: max |semitones|; each sample draws an integer in [-p, p].
        #   time_stretch_amount:   rate drawn from [1-a, 1+a] (a>0 enables it).
        # When either fires, augment_chunk returns a transform the dataset applies to the
        # labels (pitch += semitones; note times /= rate) so audio and labels stay in lockstep.
        self.pitch_shift_semitones = float(pitch_shift_semitones)
        self.time_stretch_amount = float(time_stretch_amount)
        self.max_retries = max(1, int(max_retries))
        self.augment_prob = float(augment_prob)
        self.base_seed = int(seed)
        self.sr = sr
        self.validate = bool(validate)
        # crepe_f0 (torchcrepe, GPU) is the default and intended estimator; pyin_f0 is the
        # CPU fallback. The Validator (and its torchcrepe model + CUDA context) is created
        # lazily in _ensure_validator so it is built in whatever process actually runs
        # validation — never inherited across a fork (that is the fork-after-CUDA deadlock).
        self.f0_method = f0_method
        self.f0_device = f0_device
        self.f0_model = f0_model
        self.f0_decoder = f0_decoder
        self._rng: np.random.Generator | None = None
        self._validator = None

    @property
    def requires_main_process(self) -> bool:
        """True when augmentation must run in the main process (num_workers=0).

        Only GPU f0 validation (crepe_f0, or auto when torchcrepe/CUDA is present) needs this,
        to avoid the fork-after-CUDA deadlock. Without validation, or with CPU pyin, the
        augmentation is fork-safe and can run in parallel DataLoader workers.
        """
        return self.validate and self.f0_method in ("crepe_f0", "auto")

    def _ensure_rng(self) -> np.random.Generator:
        if self._rng is None:
            try:
                import torch

                info = torch.utils.data.get_worker_info()
                worker_id = info.id if info is not None else 0
            except Exception:
                worker_id = 0
            self._rng = np.random.default_rng(
                np.random.SeedSequence([self.base_seed, int(worker_id)])
            )
        return self._rng

    def _ensure_validator(self):
        """Lazily build the Validator in the current process (correct CUDA context)."""
        if self._validator is None:
            from voders.config.models import ValidatorConfig
            from voders.validate.validator import Validator

            # sr MUST be the training rate (16 kHz), not voders' 22.05 kHz default: the
            # chunk audio is already resampled to SAMPLE_RATE, and the validator's onset/
            # offset time axis is derived from sr. (16 kHz is also CREPE's native rate, so
            # crepe_f0 does no internal resample.)
            self._validator = Validator(
                ValidatorConfig(
                    f0_method=self.f0_method,
                    f0_device=self.f0_device,
                    f0_model=self.f0_model,
                    f0_decoder=self.f0_decoder,
                ),
                sr=self.sr,
            )
        return self._validator

    def _apply_pitch_time(self, audio: np.ndarray, semitones: int, rate: float) -> np.ndarray:
        """Pitch-shift (semitones) and time-stretch (rate) a 1-D waveform, kept at input length."""
        import librosa

        y = audio
        if semitones != 0:
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=float(semitones))
        if rate != 1.0:
            y = librosa.effects.time_stretch(y, rate=float(rate))
        n = audio.shape[0]
        y = y[:n] if y.shape[0] >= n else np.pad(y, (0, n - y.shape[0]))
        return np.ascontiguousarray(y, dtype=np.float32)

    @staticmethod
    def transform_notes(notes: np.ndarray, transform: dict) -> np.ndarray:
        """Apply a label transform to (onset_s, offset_s, pitch) rows: time /= rate, pitch += n."""
        if notes is None or len(notes) == 0:
            return notes
        out = np.array(notes, dtype=np.float64, copy=True)
        rate = float(transform.get("time_rate", 1.0))
        semitones = int(transform.get("semitones", 0))
        if rate != 1.0:
            out[:, 0] /= rate
            out[:, 1] /= rate
        out[:, 2] += semitones
        return out

    def _build_score(self, notes: np.ndarray):
        """Build a voders Score from chunk-relative (onset_s, offset_s, pitch_midi) rows."""
        from voders.scores.models import Note, Score

        out = []
        for row in notes:
            onset_s, offset_s, pitch = float(row[0]), float(row[1]), int(round(float(row[2])))
            if offset_s > onset_s:
                out.append(Note(onset_s=onset_s, offset_s=offset_s, pitch_midi=pitch))
        return Score(score_id="train_chunk", notes=out)

    def augment_chunk(self, audio: np.ndarray, notes: np.ndarray):
        """Return ``(augmented_audio, transform)`` for a 1-D float32 chunk.

        ``notes`` is an (N, 3) array of chunk-relative ``(onset_s, offset_s, pitch_midi)`` rows,
        used to validate each proposal. ``transform`` is ``None`` for audio-only (label-safe)
        augmentation, or ``{"semitones", "time_rate"}`` when a label-transforming pitch/time
        augmentation fired — the caller must apply it to the labels. Falls back to the clean
        audio (and ``None`` transform) if no proposal validates within the retry budget.
        """
        from voders.manifest.models import VerdictStatus

        rng = self._ensure_rng()
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        has_notes = notes is not None and len(notes) > 0

        if self.augment_prob < 1.0 and rng.random() >= self.augment_prob:
            return audio, None

        for _ in range(self.max_retries):
            # Label-transforming params (draw per proposal so rejection sampling explores them).
            semitones = 0
            if self.pitch_shift_semitones > 0:
                p = int(self.pitch_shift_semitones)
                semitones = int(rng.integers(-p, p + 1))
            rate = 1.0
            if self.time_stretch_amount > 0:
                a = self.time_stretch_amount
                rate = float(rng.uniform(1.0 - a, 1.0 + a))
            transform = (
                {"semitones": semitones, "time_rate": rate}
                if (semitones != 0 or rate != 1.0)
                else None
            )

            y = audio if transform is None else self._apply_pitch_time(audio, semitones, rate)

            # Label-safe chain (gain/reverb/codec/mix) on top.
            profile_id = self.profile_ids[int(rng.integers(len(self.profile_ids)))]
            seed = int(rng.integers(0, 2**63 - 1))
            augmented, snr_db = self.chain.apply(y, profile_id, seed)
            augmented = np.ascontiguousarray(augmented, dtype=np.float32)

            if not self.validate or not has_notes:
                return augmented, transform

            tnotes = self.transform_notes(notes, transform) if transform else notes
            score = self._build_score(tnotes)
            if score.is_empty:
                return augmented, transform

            verdict = self._ensure_validator().validate(augmented, score, snr_db=snr_db)
            if verdict.status == VerdictStatus.ACCEPTED:
                return augmented, transform

        return audio, None  # no proposal passed validation -> clean fallback
