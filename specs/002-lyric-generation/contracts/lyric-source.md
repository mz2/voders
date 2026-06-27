# Contract: Lyric Source Interface

The boundary the orchestrator depends on to obtain per-note lyrics, independent of which source
produced them. Lives at `src/voders/lyrics/sources.py`. Glossary: **LyricPlan** = the per-note
syllable assignment for one score (see data-model.md).

## Protocol

```python
@runtime_checkable
class LyricSource(Protocol):
    name: str                      # "vowel" | "supplied" | "automatic" | "generated"

    def requires_gpu(self) -> bool:
        """vowel/supplied/automatic MUST return False (FR-005). generated MAY return True."""

    def resolve(self, score: Score, *, master_seed: int, voice_id: str) -> LyricPlan:
        """Produce a LyricPlan whose `syllables` length == len(score.notes) (FR-008)."""
```

## Guarantees every source MUST honor

1. **Length invariant**: `len(plan.syllables) == len(score.notes)`. Sources reconcile under/over-count
   by truncation / `None`-padding (open-vowel fallback) and set `plan.mismatch=True` (FR-008, SC-005).
   No note is dropped, added, or reordered.
2. **Determinism** (`automatic`): identical `(master_seed, score_id, voice_id)` ⇒ identical
   `syllables`, independent of worker count or call order, via
   `sample_seed(master_seed, score_id, voice_id, "lyrics")` (FR-004, SC-004).
3. **Byte-identity** (`vowel`): returns an all-`None` plan; the pipeline takes the unchanged
   open-vowel path and produces byte-identical output to the pre-feature pipeline (SC-001).
4. **Hashing**: `plan.text_hash` = sha256 over the canonical (note-ordered) syllable list; becomes the
   manifest `lyric_hash`.
4a. **One syllable per note** (`automatic`, `generated`, FR-019): every non-`None` entry in
   `plan.syllables` is exactly one singable syllable, so `plan.multisyllable_notes == []`. `automatic`
   satisfies this by construction; `generated` runs model text through `syllabify.segment` before 1:1
   alignment. Deterministic, so `generated` replay over pinned text re-segments identically. SC-010 is a
   structural assertion on the plan (no acoustic syllable-counting).
4b. **Supplied taken as authored** (`supplied`, FR-019): cells are never re-segmented, truncated, or
   rejected; `syllabify.syllable_count` only appends a non-single-syllable cell's index to
   `plan.multisyllable_notes`, surfaced as `lyric_multisyllable_supplied` in provenance + stats.
5. **License gate** (`generated`): a `LyricModel` with `license_ok` false/missing ⇒ `resolve` refuses
   (raises a refusal surfaced as a `license_refused` note), produces no plan (FR-010, SC-006).
6. **Caching** (`generated`): the model runs once per score set; output is pinned to
   `<output_root>/lyrics/<hash>.jsonl` and reused on replay with zero model re-invocations (FR-012,
   SC-007).

## SVS lane consumption (the only articulating lane, FR-006)

```python
# render/svs.py — when a LyricPlan with non-empty syllables is present:
#   1. g2p(plan, score) -> list[Phoneme run]   (lazy phonemizer/espeak-ng import)
#   2. articulate phonemes, vowel-on-the-beat (Decision L3)
#   3. label safety unchanged (FR-007): force_score_f0 (exact) | rederive_labels (gated, SC-002)
# Deterministic / voice-conversion lanes ignore the plan and set lyric_articulated=False (FR-006).
```

## Error modes

| Condition | Behavior |
|-----------|----------|
| Un-pronounceable / out-of-inventory syllable | that note falls back to open vowel; substitution logged (edge case) |
| `generated` model unavailable at runtime | affected samples skipped/refused and logged; no silent model substitution |
| `theme`/`model` set with `source != generated` | config validation error |
| syllable on sub-`min_note_ms` note | 001 short-note flag/reject policy wins |
