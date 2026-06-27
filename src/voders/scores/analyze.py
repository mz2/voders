"""Learned ``min_note_ms`` threshold (FR-018, FR-019).

Before rendering, an analysis pass over the input score set learns the minimum acceptable note
duration. The learned value is::

    min_note_ms = max(
        low percentile (default 1st) of the note-duration distribution,
        the largest analysis frame hop among active timing-affecting methods,
        the 50 ms onset tolerance,
    )

The learned value and the contributing method floors are recorded for audit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from voders.constants import ONSET_TOLERANCE_MS
from voders.scores.models import Score


@dataclass(frozen=True)
class LearnedThreshold:
    """The learned ``min_note_ms`` plus the values that produced it (recorded in stats/manifest)."""

    min_note_ms: float
    data_percentile_ms: float
    percentile: float
    largest_frame_hop_ms: float
    onset_tolerance_ms: float
    n_notes: int
    method_frame_hops_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "min_note_ms": self.min_note_ms,
            "data_percentile_ms": self.data_percentile_ms,
            "percentile": self.percentile,
            "largest_frame_hop_ms": self.largest_frame_hop_ms,
            "onset_tolerance_ms": self.onset_tolerance_ms,
            "n_notes": self.n_notes,
            "method_frame_hops_ms": dict(self.method_frame_hops_ms),
        }


def learn_min_note_ms(
    scores: Iterable[Score],
    *,
    percentile: float = 1.0,
    method_frame_hops_ms: dict[str, float] | None = None,
) -> LearnedThreshold:
    """Learn ``min_note_ms`` from the input score set and the active methods (FR-018, FR-019).

    Args:
        scores: the input score set.
        percentile: low percentile of the note-duration distribution to use (default 1st).
        method_frame_hops_ms: ``{method_name: frame_hop_ms}`` for active timing-affecting methods.
    """
    hops = method_frame_hops_ms or {}
    durations_ms = [n.duration_ms for s in scores for n in s.notes]

    if durations_ms:
        data_pct = float(np.percentile(np.asarray(durations_ms, dtype=float), percentile))
    else:
        data_pct = 0.0

    largest_hop = max(hops.values(), default=0.0)
    min_note_ms = max(data_pct, largest_hop, ONSET_TOLERANCE_MS)

    return LearnedThreshold(
        min_note_ms=min_note_ms,
        data_percentile_ms=data_pct,
        percentile=percentile,
        largest_frame_hop_ms=largest_hop,
        onset_tolerance_ms=ONSET_TOLERANCE_MS,
        n_notes=len(durations_ms),
        method_frame_hops_ms=dict(hops),
    )
