"""Neural (byte-level T5) G2P on GPU — the higher-quality alternative to espeak (US1, FR-006).

Skipped without CUDA + transformers. Verifies the neural backend returns IPA phoneme tokens that the
vowel-on-the-beat mapping can split, and is deterministic (greedy decode).
"""

from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    _GPU = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _GPU = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _GPU, reason="needs GPU + transformers (lyrics-gpu extra)"),
]


def test_neural_g2p_returns_phoneme_tokens():
    from voders.lyrics.g2p import is_vowel_phoneme, text_to_phonemes

    ph = text_to_phonemes("winter", backend="neural")
    assert len(ph) >= 4
    assert all(len(tok) == 1 for tok in ph)  # one IPA symbol per token
    assert any(is_vowel_phoneme(p) for p in ph)  # has a vowel nucleus
    assert ph == text_to_phonemes("winter", backend="neural")  # greedy => deterministic


def test_neural_g2p_maps_consonants_and_vowel():
    from voders.lyrics.g2p import map_syllable, text_to_phonemes

    run = map_syllable(text_to_phonemes("ski", backend="neural"), 0.5, 1.0, 0)
    assert run.nucleus_onset_s == 0.5
    assert "s" in run.lead_consonants  # leading consonant(s) before the nucleus
