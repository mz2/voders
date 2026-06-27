"""Lyric source interface and the CPU ``vowel`` source (T009, contracts/lyric-source.md).

The orchestrator depends on ``LyricSourceProtocol`` to obtain per-note lyrics, independent of which
source produced them. ``supplied``/``automatic``/``generated`` sources land in their own story
phases; ``resolve_source`` is the registry seam they plug into. No heavy imports at module load
(FR-005): only the standard library and the in-package data carriers are used here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from voders.lyrics.models import LyricPlan, LyricSource
from voders.lyrics.sampler import load_inventory, sample_syllables
from voders.seeds import sample_seed

if TYPE_CHECKING:
    from voders.scores.models import Score


@runtime_checkable
class LyricSourceProtocol(Protocol):
    """The boundary the orchestrator depends on to obtain per-note lyrics (contract)."""

    name: str

    def requires_gpu(self) -> bool:
        """vowel/supplied/automatic MUST return False (FR-005); generated MAY return True."""
        ...

    def resolve(self, score: Score, *, master_seed: int, voice_id: str) -> LyricPlan:
        """Produce a plan whose ``syllables`` length == ``len(score.notes)`` (FR-008)."""
        ...


def reconcile(syllables: list[str | None], n_notes: int) -> tuple[list[str | None], bool]:
    """Reconcile a syllable list to ``n_notes`` by truncation / ``None``-padding (FR-008, SC-005).

    Never reorders, drops, or shifts a syllable's note position. Returns the reconciled list (of
    length ``n_notes``) and a ``mismatch`` flag, true when the input count differed from the count.
    """
    if len(syllables) == n_notes:
        return list(syllables), False
    if len(syllables) > n_notes:
        return list(syllables[:n_notes]), True
    padded: list[str | None] = list(syllables) + [None] * (n_notes - len(syllables))
    return padded, True


class VowelSource:
    """The default lyric-free source: an all-``None`` plan, byte-identity path (SC-001)."""

    name = "vowel"

    def requires_gpu(self) -> bool:
        return False

    def resolve(self, score: Score, *, master_seed: int, voice_id: str) -> LyricPlan:
        return LyricPlan.build(
            score.score_id,
            LyricSource.VOWEL,
            [None] * len(score.notes),
        )


class AutomaticSource:
    """Seed-deterministic CV-syllable source needing no input data (US2, FR-003/FR-004).

    Draws one coverage-biased singable syllable per note from the checked-in inventory, derived
    from ``sample_seed(master_seed, score_id, voice_id, "lyrics")`` so an assignment reproduces in
    isolation regardless of worker count or processing order (SC-004). CPU-only (FR-005).
    """

    name = "automatic"

    def __init__(self, inventory: str = "en_cv") -> None:
        self._inventory_name = inventory

    def requires_gpu(self) -> bool:
        return False

    def resolve(self, score: Score, *, master_seed: int, voice_id: str) -> LyricPlan:
        seed = sample_seed(master_seed, score.score_id, voice_id, "lyrics")
        inventory = load_inventory(self._inventory_name)
        syllables = sample_syllables(len(score.notes), seed, inventory)
        return LyricPlan.build(score.score_id, LyricSource.AUTOMATIC, list(syllables))


def resolve_source(config_source: str | LyricSource) -> LyricSourceProtocol:
    """Return the source implementation for a config selector (the registry seam).

    ``vowel`` and ``automatic`` are implemented here; ``supplied``/``generated`` land in later
    stories.
    """
    source = LyricSource(config_source)
    if source is LyricSource.VOWEL:
        return VowelSource()
    if source is LyricSource.AUTOMATIC:
        return AutomaticSource()
    raise NotImplementedError(f"lyric source {source.value} not implemented yet")
