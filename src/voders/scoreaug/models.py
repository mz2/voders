"""Score-augmentation data carriers: ``ScoreAxis`` and ``ScoreVariant`` (data-model.md).

Pure-CPU, dependency-light: a ``ScoreVariant`` wraps a rewritten :class:`Score` (whose note rows
*are* the variant's labels, FR-003) together with the provenance to record it. New types only — no
I/O, no model, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from voders.scores.models import Score

#: Reserved separator between a base score id and its transform descriptor in a variant's id.
TRANSFORM_SEP = "__"


class ScoreAxis(StrEnum):
    """The three score-augmentation axes (also the ``corpus/augmented/<axis>/`` subtree segment)."""

    TRANSPOSE = "transpose"
    HUMANIZE = "humanize"
    VOLUME = "volume"


@dataclass(frozen=True)
class ScoreVariant:
    """A base score expanded into one variant input score, plus the provenance to record it.

    ``score.score_id`` is ``f"{base_score_id}{TRANSFORM_SEP}{transform}"`` — self-describing, so a
    file's id alone classifies it original-vs-augmented (SC-009). The base score id MUST NOT itself
    contain the reserved ``__`` separator (guarded at construction).
    """

    score: Score
    base_score_id: str
    profile_id: str
    axis: ScoreAxis
    transform: str
    seed: int
    dynamics_applied: bool = False
    notes_meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if TRANSFORM_SEP in self.base_score_id:
            raise ValueError(
                f"base_score_id {self.base_score_id!r} must not contain the reserved "
                f"{TRANSFORM_SEP!r} separator (it delimits base id from transform)"
            )
        expected = f"{self.base_score_id}{TRANSFORM_SEP}{self.transform}"
        if self.score.score_id != expected:
            raise ValueError(
                f"variant score_id {self.score.score_id!r} must equal {expected!r} "
                "(base_score_id + transform)"
            )
