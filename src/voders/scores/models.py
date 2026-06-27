"""Score and Note models (FR-001, data-model.md).

A Score is the authoritative ground truth for any sample derived from it: an ordered sequence of
``(onset_s, offset_s, pitch_midi)`` note rows.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class Note(BaseModel):
    """One note row: onset/offset in seconds and a MIDI pitch integer (FR-001)."""

    onset_s: float = Field(ge=0.0)
    offset_s: float
    pitch_midi: int = Field(ge=0, le=127)
    # Optional lyric/syllable for this note (None = lyric-free; the lanes default to an open vowel).
    # Lyric-driven SVS (G2P + a voicebank) is the work this `lyrics` branch builds on top of.
    lyric: str | None = None

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1000.0

    @model_validator(mode="after")
    def _check_positive_duration(self) -> Note:
        if self.offset_s <= self.onset_s:
            raise ValueError(
                f"note offset_s ({self.offset_s}) must be greater than onset_s ({self.onset_s})"
            )
        return self


class Score(BaseModel):
    """An ordered, monophonic sequence of notes (data-model.md).

    Monophony (no two notes overlap in time) is checked here; overlapping/polyphonic scores are
    rejected by the parser, which records the choice in provenance.
    """

    score_id: str
    source: str = ""
    notes: list[Note] = Field(default_factory=list)

    @field_validator("notes")
    @classmethod
    def _sorted_by_onset(cls, notes: list[Note]) -> list[Note]:
        return sorted(notes, key=lambda n: (n.onset_s, n.offset_s))

    @property
    def is_empty(self) -> bool:
        return len(self.notes) == 0

    @property
    def duration_s(self) -> float:
        return max((n.offset_s for n in self.notes), default=0.0)

    def is_monophonic(self) -> bool:
        """True when no two notes overlap in time (FR-001; singing is monophonic)."""
        ordered = sorted(self.notes, key=lambda n: n.onset_s)
        return all(
            ordered[i].offset_s <= ordered[i + 1].onset_s + 1e-9 for i in range(len(ordered) - 1)
        )

    def overlapping_pairs(self) -> list[tuple[int, int]]:
        """Indices of adjacent notes (in onset order) that overlap, for diagnostics."""
        ordered = sorted(range(len(self.notes)), key=lambda i: self.notes[i].onset_s)
        out: list[tuple[int, int]] = []
        for a, b in zip(ordered, ordered[1:], strict=False):
            if self.notes[b].onset_s + 1e-9 < self.notes[a].offset_s:
                out.append((a, b))
        return out
