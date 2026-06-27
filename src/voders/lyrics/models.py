"""Lyric data carriers (T007, data-model.md).

In-memory, CPU-only models for the lyric layer. No heavy imports at module load (FR-005):
only the standard library and pydantic are used here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

# Unit separator between syllables and the sentinel that represents a ``None`` (open-vowel) slot in
# the canonical hash input, so [None] and [""] hash distinctly and ("a","b") != ("ab",).
_SEP = "\x1f"
_NONE_SENTINEL = "\x00"


class LyricSource(StrEnum):
    """The origin of a sample's lyrics, recorded in provenance (data-model.md)."""

    VOWEL = "vowel"
    SUPPLIED = "supplied"
    AUTOMATIC = "automatic"
    GENERATED = "generated"


class LyricModel(BaseModel):
    """A lyric-generation model with the license gate, mirroring ``Voice`` (FR-010).

    A model whose ``license_ok`` is false or missing is refused by the ``generated`` source.
    """

    model_id: str = ""
    license: str = ""
    license_ok: bool = False
    model_ref: str = ""

    def is_usable(self) -> bool:
        """True only when the license is verified (FR-010), mirroring ``Voice.is_usable``."""
        return bool(self.license_ok)


class LyricPlan(BaseModel):
    """The resolved per-note lyric assignment a source produces (data-model.md).

    ``len(syllables) == len(score.notes)`` always; ``None`` entries fall back to the open vowel.
    """

    score_id: str
    source: LyricSource
    syllables: list[str | None]
    text_hash: str
    mismatch: bool = False
    # Note indices whose assigned text is not a single syllable. Empty for vowel/automatic/generated
    # (one syllable per note by construction/segmentation, FR-019); for ``supplied`` it lists cells
    # taken as authored and flagged (never re-segmented or rejected).
    multisyllable_notes: list[int] = Field(default_factory=list)
    model: LyricModel | None = None

    @staticmethod
    def hash_syllables(syllables: list[str | None]) -> str:
        """sha256 hexdigest over the canonical (note-ordered) syllable list (FR-009)."""
        h = hashlib.sha256()
        for syllable in syllables:
            token = _NONE_SENTINEL if syllable is None else syllable
            h.update(token.encode("utf-8"))
            h.update(_SEP.encode("utf-8"))
        return h.hexdigest()

    @classmethod
    def build(
        cls,
        score_id: str,
        source: LyricSource,
        syllables: list[str | None],
        *,
        model: LyricModel | None = None,
        mismatch: bool = False,
        multisyllable_notes: list[int] | None = None,
    ) -> LyricPlan:
        """Build a plan, computing the deterministic ``text_hash`` from ``syllables``."""
        return cls(
            score_id=score_id,
            source=source,
            syllables=list(syllables),
            text_hash=cls.hash_syllables(syllables),
            mismatch=mismatch,
            multisyllable_notes=list(multisyllable_notes or []),
            model=model,
        )


@dataclass
class PhonemeRun:
    """G2P output for one note: phonemes mapped to its onset/offset window (data-model.md)."""

    note_index: int
    phonemes: list[str]
    nucleus_onset_s: float
    lead_consonants: list[str] = field(default_factory=list)
    tail_consonants: list[str] = field(default_factory=list)
