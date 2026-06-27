"""Grapheme-to-phoneme (G2P) and phoneme->note mapping for the SVS lane (FR-006, Decision L3).

Turns a per-note syllable into phonemes and places them on the note's timeline with the
**vowel-on-the-beat** convention: the vowel (nucleus) carries the sustained pitch starting at the
note onset, any leading consonants are articulated in a short pre-onset window, and a trailing
consonant in a short pre-offset window (research Decision L3). This keeps the *labelled* onset (the
vowel nucleus) aligned to the score beat even though a consonant sounds slightly before it — the
alignment risk the SVS safety net then verifies (FR-007).

``phonemizer``/espeak-ng is imported **lazily** inside :func:`text_to_phonemes`, so importing this
module on the CPU baseline pulls in no heavy dependency (FR-005). The phoneme->note mapping is a
pure function and needs neither.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from voders.lyrics.models import PhonemeRun

if TYPE_CHECKING:
    from voders.scores.models import Score

# espeak-ng IPA vowel symbols used to find the nucleus. Approximate but sufficient to split a
# syllable into (leading consonants, vowel nucleus, trailing consonants).
_IPA_VOWELS = frozenset("aeiouɑɐɒæɛɜɝɪɔʊʌəɚɘɞyøœɶ")


def is_vowel_phoneme(phoneme: str) -> bool:
    """True if ``phoneme`` contains a vowel symbol (its first char is treated as the type)."""
    return any(ch in _IPA_VOWELS for ch in phoneme)


def map_syllable(
    phonemes: list[str], onset_s: float, offset_s: float, note_index: int
) -> PhonemeRun:
    """Place ``phonemes`` on one note: vowel nucleus on the beat, consonants around it (L3).

    Pure function (no G2P backend). ``nucleus_onset_s`` is the note onset — the labelled onset —
    regardless of any leading consonants, which ride in the pre-onset window.
    """
    first_vowel = next((i for i, p in enumerate(phonemes) if is_vowel_phoneme(p)), None)
    if first_vowel is None:
        # No vowel (e.g. an all-consonant token): treat the whole token as a lead-in; the note onset
        # still anchors the (silent) nucleus so timing labels are unaffected.
        return PhonemeRun(
            note_index=note_index,
            phonemes=list(phonemes),
            nucleus_onset_s=onset_s,
            lead_consonants=list(phonemes),
        )
    last_vowel = max(i for i, p in enumerate(phonemes) if is_vowel_phoneme(p))
    return PhonemeRun(
        note_index=note_index,
        phonemes=list(phonemes),
        nucleus_onset_s=onset_s,
        lead_consonants=phonemes[:first_vowel],
        tail_consonants=phonemes[last_vowel + 1 :],
    )


# A higher-quality, GPU-based alternative to espeak's rule G2P: a byte-level T5 trained for G2P.
# Selected with ``lyrics.g2p_backend: neural``; needs the gpu + lyrics-gpu extras.
DEFAULT_NEURAL_G2P_MODEL = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
# IPA stress/length/diacritic marks to drop when splitting a fused neural IPA string into tokens.
_IPA_DIACRITICS = set("ˈˌːˑ̃ʰʷ̩̯̪̥")


# espeak language code -> Charsiu byT5 language prefix, for the neural backend.
_CHARSIU_LANG = {
    "en-us": "eng-us", "en": "eng-us", "de": "deu", "fr-fr": "fra", "fr": "fra", "es": "spa",
    "it": "ita", "pt": "por", "ru": "rus", "nl": "nld", "pl": "pol", "ja": "jpn", "cmn": "cmn",
    "ko": "kor",
}


def text_to_phonemes(text: str, *, backend: str = "espeak", language: str = "en-us") -> list[str]:
    """Convert a syllable/word to a list of phonemes (one token each) in ``language``.

    ``backend='espeak'`` (default) uses ``phonemizer``/espeak-ng (rule-based, CPU, ~100 languages).
    ``backend='neural'`` uses a byte-level T5 G2P model on GPU. Heavy deps are imported lazily, so
    the CPU baseline loads neither (FR-005). ``language`` is an espeak code (en-us|de|fr-fr|ja|...).
    """
    if backend == "neural":
        return _neural_phonemes(text, language=language)
    return _espeak_phonemes(text, language=language)


def _espeak_phonemes(text: str, language: str = "en-us") -> list[str]:
    try:
        from phonemizer import phonemize  # lazy: not imported on the CPU baseline (FR-005)
        from phonemizer.separator import Separator
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise RuntimeError(
            "G2P requires the 'lyrics' extra (phonemizer) and espeak-ng; "
            "install with `uv sync --extra lyrics` and `apt install espeak-ng`"
        ) from exc
    # A phone separator makes espeak emit one phoneme per token ("la" -> "l|æ"), so the nucleus
    # split in :func:`map_syllable` sees individual phonemes rather than a single fused string.
    out = phonemize(
        [text],
        language=language,
        backend="espeak",
        separator=Separator(phone="|", word=" "),
        strip=True,
    )
    if not out:
        return []
    # espeak marks unknown spans like "(en)...(ru)"; drop those language tags.
    toks = str(out[0]).replace(" ", "|").split("|")
    return [p for p in toks if p and not (p.startswith("(") and p.endswith(")"))]


def _neural_phonemes(
    text: str, model_ref: str = DEFAULT_NEURAL_G2P_MODEL, language: str = "en-us"
) -> list[str]:
    import torch

    tokenizer, model, device = _load_neural_g2p_cached(model_ref)
    prefix = _CHARSIU_LANG.get(language, "eng-us")
    ids = tokenizer([f"<{prefix}>: {text}"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_length=48, num_beams=1)  # greedy => deterministic
    ipa = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    # The model emits a fused IPA string ("ˈwɪntɝ"); split into single-symbol tokens, dropping
    # stress/length diacritics so map_syllable sees individual consonant/vowel phonemes.
    return [c for c in ipa if c.strip() and c not in _IPA_DIACRITICS]


@functools.cache
def _load_neural_g2p_cached(model_ref: str):
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = T5ForConditionalGeneration.from_pretrained(model_ref).to(device).eval()
    return tokenizer, model, device


def phoneme_runs(
    plan_syllables: list[str | None],
    score: Score,
    *,
    backend: str = "espeak",
    language: str = "en-us",
) -> list[PhonemeRun]:
    """Map each note's syllable to a :class:`PhonemeRun` (skips ``None``/open-vowel notes).

    Calls the G2P backend (lazy) per non-empty syllable in ``language``, so it only runs when lyrics
    are present.
    """
    runs: list[PhonemeRun] = []
    for i, (syllable, note) in enumerate(zip(plan_syllables, score.notes, strict=True)):
        if not syllable:
            continue
        phonemes = text_to_phonemes(syllable, backend=backend, language=language)
        runs.append(map_syllable(phonemes, note.onset_s, note.offset_s, i))
    return runs
