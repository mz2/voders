"""Contract tests for the expressive neural-SVS lane (T035, FR-007, US4).

SVS = singing-voice synthesis. The expressive lane adds naturalistic timing, which can drift the
note boundaries outside the 50 ms onset tolerance and silently corrupt labels. The lane therefore
offers two safety modes (FR-007):

* ``force_score_f0`` — use the score's f0 directly (behaves like the deterministic P1 lane); the
  label score is the input score, exact and row-for-row.
* ``rederive_labels`` — render expressively (seeded timing humanization), then re-derive the note
  onsets/offsets from the actually-rendered audio (a forced-alignment / onset-detection analogue)
  and carry the re-derived score plus the measured onset deviation.

The CPU backend is an expressive WORLD-vocoder stand-in so US4 runs on a laptop; ``diffsinger`` /
``nnsvs`` are the production GPU backends and must not pull in ``torch`` at module load.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from voders.render.base import RenderRequest
from voders.render.svs import SvsLane
from voders.scores.models import Score
from voders.voices.models import Voice, VoiceKind


def _svs_voice(donor_ah: Voice) -> Voice:
    """An SVS voicebank voice that reuses a consented donor recording as its timbre source."""
    return Voice(
        voice_id="svs_ah",
        kind=VoiceKind.SVS_VOICEBANK,
        license="CC0 synthetic vowel",
        consent_verified=True,
        model_ref=donor_ah.model_ref,
    )


def test_force_score_f0_keeps_input_label_exactly(small_score: Score, donor_ah: Voice) -> None:
    """force_score_f0: label_score == input row-for-row; cpu backend needs no GPU."""
    lane = SvsLane({"mode": "force_score_f0", "backend": "cpu"})
    assert lane.requires_gpu() is False

    voice = _svs_voice(donor_ah)
    result = lane.render(RenderRequest(score=small_score, voice=voice, seed=11))

    assert result.label_score == small_score
    assert result.label_score.notes == small_score.notes
    assert result.audio.dtype.name == "float32"
    assert result.audio.ndim == 1
    assert result.audio.size > 0


def test_rederive_labels_carries_measured_score(small_score: Score, donor_ah: Voice) -> None:
    """rederive_labels: label is re-derived (not the input object) and deviation is numeric."""
    lane = SvsLane({"mode": "rederive_labels", "backend": "cpu", "humanize_ms": 15.0})
    assert lane.requires_gpu() is False

    voice = _svs_voice(donor_ah)
    result = lane.render(RenderRequest(score=small_score, voice=voice, seed=7))

    # The label is a re-derived score: same note count and pitches, but measured onsets/offsets.
    assert result.label_score is not small_score
    assert result.label_score != small_score
    assert len(result.label_score.notes) == len(small_score.notes)
    assert [n.pitch_midi for n in result.label_score.notes] == [
        n.pitch_midi for n in small_score.notes
    ]

    dev = result.notes.get("max_onset_dev_ms")
    assert isinstance(dev, float)
    assert dev >= 0.0


def test_diffsinger_backend_is_out_of_process(
    small_score: Score, donor_ah: Voice, monkeypatch: pytest.MonkeyPatch
) -> None:
    """diffsinger is now a wired out-of-process backend: requires_gpu() is False (it runs in its own
    uv project, not the core process) and the lane routes it to the diffsinger backend project."""
    import numpy as np

    import voders.render.backend_bridge as bb

    lane = SvsLane({"backend": "diffsinger"})
    assert lane.requires_gpu() is False

    captured: dict[str, str] = {}

    def fake_render_via_backend(project, module, *a, **k):  # noqa: ANN001, ANN202
        captured["project"] = project
        captured["module"] = module
        return np.zeros(256, dtype=np.float32)

    monkeypatch.setattr(bb, "render_via_backend", fake_render_via_backend)
    lane.render(RenderRequest(score=small_score, voice=_svs_voice(donor_ah), seed=1))
    assert captured == {
        "project": "diffsinger",
        "module": "voders_diffsinger_backend.worker",
    }


def test_unknown_backend_raises(small_score: Score, donor_ah: Voice) -> None:
    """An unrecognised backend is rejected with a clear error listing the valid ones."""
    lane = SvsLane({"backend": "nope"})
    with pytest.raises(ValueError, match="unknown SVS backend"):
        lane.render(RenderRequest(score=small_score, voice=_svs_voice(donor_ah), seed=1))


def test_importing_module_does_not_import_torch() -> None:
    """The CPU baseline must not pull in torch at module load (FR-009)."""
    sys.modules.pop("torch", None)
    importlib.reload(importlib.import_module("voders.render.svs"))
    assert "torch" not in sys.modules


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
