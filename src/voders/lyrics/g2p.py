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

from typing import TYPE_CHECKING

from voders.lyrics.models import PhonemeRun

if TYPE_CHECKING:
    from voders.scores.models import Score

# espeak-ng IPA vowel symbols used to find the nucleus. Approximate but sufficient to split a
# syllable into (leading consonants, vowel nucleus, trailing consonants).
_IPA_VOWELS = frozenset("aeiouɑɐɒæɛɜɪɔʊʌəɘɞyøœɶ")


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


def text_to_phonemes(text: str, *, backend: str = "espeak") -> list[str]:
    """Convert a syllable/word to phonemes via ``phonemizer``/espeak-ng (lazy import, FR-005).

    Raises RuntimeError if the optional ``lyrics`` extra (``phonemizer``) / espeak-ng is missing.
    """
    try:
        from phonemizer import phonemize  # lazy: not imported on the CPU baseline (FR-005)
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise RuntimeError(
            "G2P requires the 'lyrics' extra (phonemizer) and espeak-ng; "
            "install with `uv sync --extra lyrics` and `apt install espeak-ng`"
        ) from exc
    out = phonemize([text], language="en-us", backend=backend, strip=True)
    return [p for p in str(out[0]).split() if p] if out else []


def phoneme_runs(plan_syllables: list[str | None], score: Score) -> list[PhonemeRun]:
    """Map each note's syllable to a :class:`PhonemeRun` (skips ``None``/open-vowel notes).

    Calls the G2P backend (lazy) per non-empty syllable, so it only runs when lyrics are present.
    """
    runs: list[PhonemeRun] = []
    for i, (syllable, note) in enumerate(zip(plan_syllables, score.notes, strict=True)):
        if not syllable:
            continue
        phonemes = text_to_phonemes(syllable)
        runs.append(map_syllable(phonemes, note.onset_s, note.offset_s, i))
    return runs
