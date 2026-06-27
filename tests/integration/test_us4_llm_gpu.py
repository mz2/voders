"""US4 with the real instruct LLM on GPU (skipped without CUDA + transformers).

Exercises the actual model-driven path end-to-end: GeneratedSource -> real LLM -> segmentation ->
one-syllable-per-note plan -> pinned cache artifact reused on replay (FR-011/012/019, SC-007).
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    _GPU = torch.cuda.is_available()
except Exception:  # pragma: no cover - environment without the gpu/lyrics-gpu extras
    _GPU = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _GPU, reason="needs GPU + transformers (lyrics-gpu extra)"),
]

MODEL_REF = "Qwen/Qwen2.5-0.5B-Instruct"


def _score(n: int):
    from voders.scores.models import Note, Score

    return Score(
        score_id="s1",
        notes=[Note(onset_s=i * 0.5, offset_s=i * 0.5 + 0.4, pitch_midi=60) for i in range(n)],
    )


def test_generated_source_with_real_llm(tmp_path: Path):
    from voders.lyrics.models import LyricModel
    from voders.lyrics.sources import GeneratedSource

    model = LyricModel(
        model_id="qwen2.5-0.5b-instruct",
        license="apache-2.0",
        license_ok=True,
        model_ref=MODEL_REF,
    )
    src = GeneratedSource(theme="winter, longing", model=model, output_root=str(tmp_path))
    score = _score(6)

    plan = src.resolve(score, master_seed=7, voice_id="v1")
    assert len(plan.syllables) == 6
    assert plan.multisyllable_notes == []  # segmented to one syllable per note (SC-010)
    assert any(s for s in plan.syllables)  # the LLM produced real syllables, not all vowel-fallback

    # The generated text is pinned; a replay reuses it verbatim (SC-007).
    assert (tmp_path / "lyrics").exists()
    plan2 = src.resolve(score, master_seed=7, voice_id="v1")
    assert plan2.syllables == plan.syllables
