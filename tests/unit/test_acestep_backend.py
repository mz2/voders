"""CPU unit tests for the ACE-Step backend orchestration (the model/separator seam is injected).

Verifies caption construction, 22.05 kHz↔48 kHz bridging, mode dispatch, and BackendOutput
invariants without a GPU or ACE-Step weights — only the two real library calls (generator,
separator) are stubbed out via injection.
"""

from __future__ import annotations

import numpy as np

from voders.constants import SAMPLE_RATE
from voders.render.accompaniment_backend import AccompanimentBackend, BackendOutput
from voders.render.backends.acestep import AceStepBackend, build_caption
from voders.scores.models import Note, Score

_NATIVE_SR = 48_000


def _vocal(seconds: float = 1.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _score() -> Score:
    return Score(score_id="s", notes=[Note(onset_s=0.2, offset_s=0.8, pitch_midi=60)])


def _fake_generator(ref_48k_stereo, caption, seed, duration_s):
    # Ignore the conditioning; emit a 48 kHz stereo sine of the requested duration.
    n = int(duration_s * _NATIVE_SR)
    t = np.arange(n) / _NATIVE_SR
    mono = (0.4 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    return np.stack([mono, mono], axis=-1)


def _fake_separator(mix_48k_stereo):
    return (mix_48k_stereo * 0.5).astype(np.float32)


def _backend() -> AceStepBackend:
    return AceStepBackend(generator=_fake_generator, separator=_fake_separator)


def test_backend_satisfies_protocol_and_caps():
    b = _backend()
    assert isinstance(b, AccompanimentBackend)
    assert b.requires_gpu() is True
    assert b.supports("lego") and b.supports("complete")
    assert not b.supports("bogus")
    assert b.model_license == "Apache-2.0"  # real ace_step v0.2.0 license


def test_complete_returns_mono_22050_mix():
    b = _backend()
    vocal = _vocal()
    out = b.generate(vocal, _score(), "complete", 7, {"free_time": True})
    assert isinstance(out, BackendOutput)
    assert out.accompaniment_stem is None
    assert out.mix is not None
    assert out.mix.dtype == np.float32 and out.mix.ndim == 1
    assert out.mix.size == vocal.size  # bridged to 22,050 mono and length-fit to the vocal
    assert out.native_sr == _NATIVE_SR


def test_lego_returns_separated_stem():
    b = _backend()
    vocal = _vocal()
    out = b.generate(vocal, _score(), "lego", 3, {"free_time": True})
    assert out.mix is None
    assert out.accompaniment_stem is not None
    assert out.accompaniment_stem.dtype == np.float32 and out.accompaniment_stem.ndim == 1
    assert out.accompaniment_stem.size == vocal.size


def test_unknown_mode_raises():
    b = _backend()
    try:
        b.generate(_vocal(), _score(), "bogus", 1, {})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


def test_caption_free_time_has_no_bpm():
    cap = build_caption(_score(), {"free_time": True, "target_instrument": "bowed metal pad"})
    assert "free time" in cap and "no fixed tempo" in cap
    assert "bpm" not in cap
    assert "bowed metal pad" in cap


def test_caption_metered_includes_bpm():
    cap = build_caption(_score(), {"free_time": False, "bpm": 90, "target_instrument": "strings"})
    assert "90 bpm" in cap
    assert "strings" in cap


def test_preflight_skips_when_backend_unavailable():
    # When torch/CUDA or the ACE-Step package is missing, the real backend reports a skip reason so
    # the lane degrades gracefully (FR-013) instead of crashing mid-run. preflight() must NEVER
    # raise — any torch import/init failure is caught and turned into a skip reason.
    reason = AceStepBackend().preflight("lego")
    assert reason is not None and isinstance(reason, str)


def test_build_accompanist_skips_acestep_when_unavailable():
    from voders.config.models import LaneToggle, RunConfig, ValidatorConfig
    from voders.render.registry import build_accompanist
    from voders.validate.validator import Validator

    config = RunConfig(
        run_id="t",
        master_seed=1,
        scores="x/*.tsv",
        output_root="out/t",
        lanes={"accompaniment": LaneToggle(enabled=True, backend="acestep")},
    )
    assert build_accompanist(config, Validator(ValidatorConfig())) is None
