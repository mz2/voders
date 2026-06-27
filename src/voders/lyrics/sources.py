"""Lyric source interface and the CPU ``vowel`` source (T009, contracts/lyric-source.md).

The orchestrator depends on ``LyricSourceProtocol`` to obtain per-note lyrics, independent of which
source produced them. ``supplied``/``automatic``/``generated`` sources land in their own story
phases; ``resolve_source`` is the registry seam they plug into. No heavy imports at module load
(FR-005): only the standard library and the in-package data carriers are used here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from voders.lyrics.cache import LyricCache, request_key
from voders.lyrics.models import LyricModel, LyricPlan, LyricSource
from voders.lyrics.sampler import load_inventory, sample_syllables
from voders.lyrics.syllabify import segment, syllable_count
from voders.seeds import sample_seed

if TYPE_CHECKING:
    from voders.scores.models import Score

# A generator turns (theme, score, seed) into raw lyric text; the source then segments it into
# syllables (FR-019). Injectable so tests can avoid the out-of-process model backend.
GeneratorFn = Callable[[str, "Score", int], str]


class LyricLicenseRefused(PermissionError):
    """Raised when a ``generated`` source's model license is not verified (FR-010).

    Mirrors the donor-voice ``ConsentRefusedError``: the orchestrator turns it into a
    ``license_refused`` manifest record rather than producing a sample.
    """


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
        multisyllable = [i for i, s in enumerate(syllables) if s and syllable_count(s) != 1]
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


def _generate_via_backend(theme: str, score: Score, seed: int, model_ref: str) -> str:
    """Default ``generated`` generator: run the out-of-process lyrics model backend (FR-011).

    Lazily bridges to ``backends/lyrics`` so no model/GPU dependency reaches the CPU core. The
    backend returns lyric text, which the source then segments into one-syllable-per-note (FR-019).
    """
    from voders.render.backend_bridge import generate_lyrics_via_backend

    return generate_lyrics_via_backend(theme, score, seed, model_ref=model_ref)


class GeneratedSource:
    """Themed, model-driven lyric source, cached as a pinned artifact (US4, FR-011/012).

    Runs the model once per (score, theme, model, seed), segments the returned text into one
    syllable per note (FR-019), and pins it so a replay reuses the pinned text with zero model
    re-invocations (SC-007). A model whose license is unverified is refused (FR-010).
    """

    name = "generated"

    def __init__(
        self,
        *,
        theme: str | None = None,
        model: LyricModel | None = None,
        syllabifier: str = "en_rule",
        cache_dir: str = "lyrics",
        output_root: str | None = None,
        generator: GeneratorFn | None = None,
    ) -> None:
        self._theme = theme or ""
        self._model = model
        self._syllabifier = syllabifier
        self._cache_dir = cache_dir
        self._output_root = output_root
        self._generator = generator

    def requires_gpu(self) -> bool:
        # The model runs out-of-process (backends/lyrics); the core never imports it.
        return False

    def resolve(self, score: Score, *, master_seed: int, voice_id: str) -> LyricPlan:
        if self._model is None or not self._model.is_usable():
            model_id = self._model.model_id if self._model else "(none)"
            raise LyricLicenseRefused(
                f"lyric model {model_id!r} license not verified; refused (FR-010)"
            )
        seed = sample_seed(master_seed, score.score_id, voice_id, "lyrics")
        n = len(score.notes)
        key = request_key(self._theme, self._model.model_ref, score.score_id, seed, n)
        cache = LyricCache(self._output_root, self._cache_dir) if self._output_root else None

        cached = cache.load(key) if cache is not None else None
        if cached is not None:
            syllables, mismatch = reconcile(cached, n)
        else:
            raw_text = self._generate(score, seed)
            segmented: list[str | None] = list(segment(raw_text))
            syllables, mismatch = reconcile(segmented, n)
            if cache is not None:
                cache.store(
                    key,
                    score.score_id,
                    syllables,
                    source="generated",
                    model_id=self._model.model_id,
                )
        # Each non-None entry is one syllable from ``segment`` by construction (FR-019/SC-010).
        return LyricPlan.build(
            score.score_id,
            LyricSource.GENERATED,
            syllables,
            model=self._model,
            mismatch=mismatch,
            multisyllable_notes=[],
        )

    def _generate(self, score: Score, seed: int) -> str:
        if self._generator is not None:
            return self._generator(self._theme, score, seed)
        assert self._model is not None
        # Default: a real instruct LLM run in-process on GPU (Decision L1). The out-of-process
        # ``_generate_via_backend`` remains available for isolated/alternative model toolkits.
        from voders.lyrics.llm import generate_lyrics

        return generate_lyrics(
            self._theme, len(score.notes), seed, model_ref=self._model.model_ref
        )


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
    if source is LyricSource.GENERATED:
        return GeneratedSource(
            theme=theme,
            model=model,
            syllabifier=syllabifier,
            cache_dir=cache_dir,
            output_root=output_root,
        )
    raise NotImplementedError(f"lyric source {source.value} not implemented yet")


def resolve_source(config_source: str | LyricSource) -> LyricSourceProtocol:
    """Return the default-configured source for a selector (the registry seam, back-compat)."""
    return build_source(config_source)
