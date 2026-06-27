"""Deterministic syllable segmentation (FR-019, SC-010, research Decision L7).

Two pure-CPU entry points with no heavy imports (FR-005):

- ``segment(text)`` turns words / free generated text into an ordered list of singable syllables,
  used by the ``generated`` source to align one syllable per note.
- ``syllable_count(text)`` reports how many syllables a token holds, used by the ``supplied`` source
  to *flag* (never alter) a cell that is not a single syllable.

The algorithm is a rule-based vowel-group splitter — English syllables need only be *singable* here,
not linguistically authoritative (spec Assumptions). It is a pure function of its input, so a
``generated`` replay over pinned text re-segments identically (FR-012).
"""

from __future__ import annotations

import re

_VOWELS = frozenset("aeiouy")
# A run of one or more alphabetic characters is a "word" token; everything else (digits, symbols,
# whitespace, punctuation) is dropped, so it cannot be sung as a syllable.
_WORD_RE = re.compile(r"[a-z]+", re.IGNORECASE)


def _vowel_groups(word: str) -> list[tuple[int, int]]:
    """Return [start, end) spans of each maximal vowel run in ``word`` (lowercased)."""
    groups: list[tuple[int, int]] = []
    i, n = 0, len(word)
    while i < n:
        if word[i] in _VOWELS:
            j = i
            while j < n and word[j] in _VOWELS:
                j += 1
            groups.append((i, j))
            i = j
        else:
            i += 1
    return groups


def _split_word(word: str) -> list[str]:
    """Split one lowercase word into syllables — one maximal vowel group each."""
    groups = _vowel_groups(word)
    if not groups:
        # A letter token with no vowel (e.g. "hmm") is one singable unit.
        return [word] if word else []

    # Boundary between two vowel groups separated by ``k`` consonants: keep ``k // 2`` consonants in
    # the left (coda) syllable, the rest open the next — VCV->V-CV, VCCV->VC-CV, VCCCV->VC-CCV.
    boundaries = [0]
    for (_, end), (nxt_start, _) in zip(groups, groups[1:], strict=False):
        k = nxt_start - end
        boundaries.append(end + k // 2)
    boundaries.append(len(word))

    syllables = [word[a:b] for a, b in zip(boundaries, boundaries[1:], strict=False) if word[a:b]]

    # Silent trailing ``e``: merge a final "...Ce" syllable into the previous one ("ma","ke" ->
    # "make"), EXCEPT the syllabic consonant+"le" ending which keeps its own syllable ("ta","ble").
    if len(syllables) >= 2:
        last = syllables[-1]
        last_groups = _vowel_groups(last)
        is_silent_e = (
            len(last_groups) == 1
            and last.endswith("e")
            and last_groups[0] == (len(last) - 1, len(last))
        )
        is_consonant_le = re.search(r"[^aeiouy]le$", word) is not None
        if is_silent_e and not is_consonant_le:
            tail = syllables.pop()
            syllables[-1] = syllables[-1] + tail

    return syllables


def segment(text: str) -> list[str]:
    """Split ``text`` into an ordered list of lowercase singable syllables (FR-019).

    Non-alphabetic tokens (digits, symbols, punctuation) are dropped — they cannot be sung.
    """
    syllables: list[str] = []
    for match in _WORD_RE.finditer(text):
        syllables.extend(_split_word(match.group(0).lower()))
    return syllables


def syllable_count(text: str) -> int:
    """Number of singable syllables in ``text`` (0 for a token that cannot be sung).

    Used to flag a non-single-syllable ``supplied`` cell (count != 1) without altering it.
    """
    return len(segment(text))


# Unicode vowels across the supported European languages (Latin + Cyrillic), for multilingual
# segmentation; CJK is split per character (each kana/hanzi/hangul block is ~one syllable).
_UNI_VOWELS = frozenset("aeiouy" "àáâãäåæ" "èéêë" "ìíîï" "òóôõöø" "ùúûü" "ýÿœ" "ąę" "аеёиоуыэюяі")
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_UNI_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_CJK_LANGS = frozenset({"ja", "cmn", "zh", "ko"})


def _split_word_unicode(word: str) -> list[str]:
    """Vowel-group split for a lowercase Latin/Cyrillic word (accent-aware); no silent-e rules."""
    groups: list[tuple[int, int]] = []
    i, n = 0, len(word)
    while i < n:
        if word[i] in _UNI_VOWELS:
            j = i
            while j < n and word[j] in _UNI_VOWELS:
                j += 1
            groups.append((i, j))
            i = j
        else:
            i += 1
    if not groups:
        return [word] if word else []
    bounds = [0]
    for (_, end), (nxt, _) in zip(groups, groups[1:], strict=False):
        bounds.append(end + (nxt - end) // 2)
    bounds.append(n)
    return [word[a:b] for a, b in zip(bounds, bounds[1:], strict=False) if word[a:b]]


def segment_multilingual(text: str, language: str = "en-us") -> list[str]:
    """Segment generated lyric text into one-syllable units, language-aware (FR-019).

    English uses the rule splitter; other Latin/Cyrillic languages use an accent-aware vowel-group
    split; CJK languages split per character (kana/hanzi/hangul ≈ one mora/syllable).
    """
    base = language.split("-")[0].lower()
    if base in _CJK_LANGS:
        return [c for c in text if _CJK.match(c)]
    if base in ("en", ""):
        return segment(text)
    out: list[str] = []
    for m in _UNI_WORD.finditer(text):
        out.extend(_split_word_unicode(m.group(0).lower()))
    return out
