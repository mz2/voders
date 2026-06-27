"""Method timing-budget registry (FR-019).

Each timing-affecting method (f0 estimator, onset detector, forced aligner, codec/resampler)
contributes two documented numbers — its analysis frame hop and any constant group delay. The
validator (a) floors the learned ``min_note_ms`` at the largest active frame hop and (b) subtracts
a method's constant group delay from measured onsets/offsets before comparing to the score, so a
fixed delay does not cause spurious rejections. No timing constants are invented; unknown values
are measured on a click-train fixture, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

from voders.constants import SAMPLE_RATE

# pyin hop is 512 samples by default in librosa; at 22,050 Hz that is ~23.2 ms.
_PYIN_HOP_SAMPLES = 512


@dataclass(frozen=True)
class MethodTimingBudget:
    """Two documented numbers for one timing-affecting method (data-model.md)."""

    method: str
    frame_hop_ms: float
    group_delay_ms: float = 0.0


# Documented defaults. Sources:
#  - pyin: librosa default hop_length=512 → 512/22050 s ≈ 23.22 ms; pyin is centred (no net delay).
#  - crepe: CREPE uses a 10 ms analysis hop (Kim et al., 2018); centred → no constant delay.
#  - mfa_align: Montreal Forced Aligner uses 10 ms MFCC frames.
#  - mp3_codec: encoder/decoder add a constant algorithmic delay (~26 ms for MPEG-1 Layer III at
#    44.1 kHz, ~13 ms equivalent of frames); kept as a documented constant, refine by measurement.
_DEFAULT_BUDGETS: dict[str, MethodTimingBudget] = {
    "pyin_f0": MethodTimingBudget(
        "pyin_f0", frame_hop_ms=1000.0 * _PYIN_HOP_SAMPLES / SAMPLE_RATE, group_delay_ms=0.0
    ),
    "crepe_f0": MethodTimingBudget("crepe_f0", frame_hop_ms=10.0, group_delay_ms=0.0),
    "mfa_align": MethodTimingBudget("mfa_align", frame_hop_ms=10.0, group_delay_ms=0.0),
    "mp3_codec": MethodTimingBudget("mp3_codec", frame_hop_ms=0.0, group_delay_ms=0.0),
}


def default_budgets() -> dict[str, MethodTimingBudget]:
    """A fresh copy of the documented default timing budgets."""
    return dict(_DEFAULT_BUDGETS)


class TimingRegistry:
    """Holds the active methods' timing budgets and applies their corrections (FR-019)."""

    def __init__(self, budgets: dict[str, MethodTimingBudget] | None = None) -> None:
        self._budgets: dict[str, MethodTimingBudget] = budgets or default_budgets()
        self._active: set[str] = set()

    def register(self, budget: MethodTimingBudget) -> None:
        self._budgets[budget.method] = budget

    def activate(self, *methods: str) -> None:
        for m in methods:
            if m not in self._budgets:
                raise KeyError(f"no timing budget registered for method {m!r} (FR-019)")
            self._active.add(m)

    def get(self, method: str) -> MethodTimingBudget:
        return self._budgets[method]

    def active_frame_hops_ms(self) -> dict[str, float]:
        return {m: self._budgets[m].frame_hop_ms for m in self._active}

    def largest_active_frame_hop_ms(self) -> float:
        return max((self._budgets[m].frame_hop_ms for m in self._active), default=0.0)

    def total_active_group_delay_ms(self) -> float:
        """Sum of constant group delays across active methods, subtracted before comparison."""
        return sum(self._budgets[m].group_delay_ms for m in self._active)
