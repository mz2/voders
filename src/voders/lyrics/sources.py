"""Lyric source interface and the CPU ``vowel`` source (T009, contracts/lyric-source.md).

The orchestrator depends on ``LyricSourceProtocol`` to obtain per-note lyrics, independent of which
source produced them. ``supplied``/``automatic``/``generated`` sources land in their own story
phases; ``resolve_source`` is the registry seam they plug into. No heavy imports at module load
(FR-005): only the standard library and the in-package data carriers are used here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from voders.lyrics.models import LyricModel, LyricPlan, LyricSource
from voders.lyrics.sampler import load_inventory, sample_syllables
from voders.lyrics.syllabify import syllable_count
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


class SuppliedSource:
    """Operator-supplied lyrics carried on the score's notes (US1, FR-002).

    Reads ``note.lyric`` for each note. Cells are taken **as authored** (FR-019): a cell that is not
    a single syllable is *flagged* in ``multisyllable_notes`` (surfaced in provenance/stats) but is
    never re-segmented, truncated, or rejected. CPU-only (FR-005).
    """

    name = "supplied"

    def __init__(self, syllabifier: str = "en_rule") -> None:
        self._syllabifier = syllabifier

    def requires_gpu(self) -> bool:
        return False

    def resolve(self, score: Score, *, master_seed: int, voice_id: str) -> LyricPlan:
        syllables: list[str | None] = [n.lyric for n in score.notes]
        multisyllable = [
            i for i, s in enumerate(syllables) if s and syllable_count(s) != 1
        ]
        return LyricPlan.build(
            score.score_id,
            LyricSource.SUPPLIED,
            syllables,
            multisyllable_notes=multisyllable,
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


def build_source(
    config_source: str | LyricSource,
    *,
    inventory: str = "en_cv",
    syllabifier: str = "en_rule",
    theme: str | None = None,
    model: LyricModel | None = None,
    cache_dir: str = "lyrics",
    output_root: str | None = None,
) -> LyricSourceProtocol:
    """Construct the source for a config selector, threading the relevant ``LyricsConfig`` params.

    ``vowel``/``supplied``/``automatic`` are CPU; ``generated`` lands in US4 (raises until then).
    """
    source = LyricSource(config_source)
    if source is LyricSource.VOWEL:
        return VowelSource()
    if source is LyricSource.SUPPLIED:
        return SuppliedSource(syllabifier=syllabifier)
    if source is LyricSource.AUTOMATIC:
        return AutomaticSource(inventory=inventory)
    raise NotImplementedError(f"lyric source {source.value} not implemented yet")


def resolve_source(config_source: str | LyricSource) -> LyricSourceProtocol:
    """Return the default-configured source for a selector (the registry seam, back-compat)."""
    return build_source(config_source)
