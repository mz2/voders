# Phase 0 Research: Score-Domain Augmentations

All decisions resolve to in-core, pure-CPU `numpy`; no NEEDS CLARIFICATION remained from the spec (the
four issue open-questions were resolved as documented defaults in the spec's Assumptions). This document
records the design decisions that ground the Phase 1 artifacts, each citing the 001 code it builds on.

## D1 — Where the stage runs (insertion point)

**Decision**: Implement `voders.scoreaug.expand(base: ParsedScore, profile, seed) -> list[ScoreVariant]`
and call it as a pre-pass inside `Orchestrator.run()` immediately after `_load_scores()`
(`corpus/orchestrator.py:101`) and before the `for ps in parsed` loop (`:123`). The orchestrator replaces
its `parsed` list with the flattened expansion (base score + its variants) and iterates exactly as today.

**Rationale**: The existing render loop is a cartesian product over `parsed × voices × lanes`, with the
audio-augmentation fan-out nested inside (`:145–164`). Expanding the *input* list is the smallest possible
change that makes score variants multiply with every downstream axis (FR-015) without touching the loop,
the lanes, the validator, or the seeding of render seeds. A variant is a `ParsedScore` with freshly
serialised `raw_bytes`, so the orchestrator's byte-identity path (`:277–280`) works for it unchanged.

**Alternatives considered**: (a) A separate CLI pre-stage that writes variant `.tsv` files to disk and
points a second run at them — rejected: doubles I/O, loses the single-run manifest/provenance, and breaks
the "one config, one run" model. (b) Expanding inside the inner loop per voice — rejected: would re-derive
identical variants per voice and risk voice-dependent variants, violating "variant is shared input".

## D2 — Determinism & seed convention

**Decision**: A variant's randomness derives from `derive_seed(master_seed, base_score_id, "score_aug",
profile_id, axis, draw_index)` (`seeds.py:17`), **independent of voice and lane**. Transposition needs no
RNG (pure offset). Humanisation and volume draw from `rng(variant_seed)` (`seeds.py:36`). The existing
per-sample *render* seed (`sample_seed(..., lane_name)`, `:232`) is unchanged and orthogonal.

**Rationale**: The variant score is a shared input consumed identically by every voice/lane, so its rows
must not depend on which voice renders it. Reusing 001's `derive_seed` keeps the whole feature inside the
existing reproducibility guarantee (FR-010, SC-005): same `(base, profile, seed)` ⇒ byte-identical
variant, regardless of worker count or order. No new seeding machinery.

**Alternatives considered**: Folding the variant into the render seed — rejected: couples the score rewrite
to the render stage and makes a variant's labels voice-dependent.

## D3 — Variant identity & physical/naming separation (FR-016, the operator ask)

**Decision**:
- A variant's `score_id` encodes base + transform with a reserved `__` separator and a compact axis tag:
  `"{base}__t+12"` (transpose), `"{base}__hum0"` / `"__hum1"` (humanise draw index), `"{base}__vol"`
  (volume). The base (un-augmented) score keeps its original `score_id`. So `sample_id`
  (`{score_id}_singer_{voice}`) and the `.wav`/`.tsv` filenames are self-describing (SC-006/SC-009).
- `CorpusStore` routes by a new record flag: **originals stay exactly where they are today**
  (`corpus/shard=NNN/`, `rejected/shard=NNN/`); **augmented variants** go to
  `corpus/augmented/{axis}/shard=NNN/` and `rejected/augmented/{axis}/shard=NNN/`, where `{axis}` ∈
  `{transpose, humanize, volume}`. No variant ever shares a path with an original or another variant.
- The chosen output location is recorded in provenance (the resolved `score_path`/`audio_path` already
  carry it; the `score_aug_*` fields make it queryable).

**Rationale**: Keeping originals in place means a feature-*off* run produces zero `augmented/` directories
and is byte-identical to today (SC-001), and even a feature-*on* run leaves the "real" corpus exactly
where consumers expect it while the variants live in an unmistakably separate, axis-labelled subtree
(FR-016). Encoding the transform in the id gives a second, manifest-free signal (SC-009). The `__`
separator is reserved and rejected in base `score_id`s by a guard so it cannot collide with a real score
stem.

**Alternatives considered**: (a) A flat `corpus/` with only id-encoded distinction — rejected: the
operator explicitly asked for *physical* separation (different folders). (b) Moving originals into
`corpus/original/` when the feature is on — rejected: needlessly relocates the canonical corpus and
complicates the SC-001 control; routing only the *new* files is simpler and collision-free.

## D4 — Time-humanisation constraint solver

**Decision**: For each note, draw onset and offset jitter from a Gaussian `N(0, σ)` (σ configurable),
**clamp each deviation to a max-deviation budget** `±max_dev_s`, then **project the jittered notes back to
a valid monophonic sequence** by a single left-to-right pass: each note's onset is clamped to be `≥`
the previous note's (jittered) offset (default `forbid_overlap` policy), and each note's offset is clamped
to keep duration `≥ min_dur_s > 0`. If projection would push a note past the budget, the note keeps its
base timing and a `constraint_hit` count is recorded; the variant is always emitted valid (FR-007).

**Rationale**: The `Note` model already rejects non-positive duration (`scores/models.py:30–36`) and
`parse_tsv` rejects overlap (`parse.py:70–75`); an emitted variant must satisfy both or it is unusable.
The clamp-then-project order keeps jitter musically bounded (max-dev budget) *and* structurally valid
(ordering, non-overlap, positive duration) without rejection-sampling loops that could be slow or
non-terminating. The overlap policy is a config knob defaulting to forbid (matches the monophony invariant
the corpus relies on); a `legato`/overlap policy is reserved for later and not in v1.

**Alternatives considered**: Rejection sampling (redraw until valid) — rejected: unbounded worst case and
harder to make deterministic. Per-note independent jitter with no projection — rejected: routinely
produces overlaps/reorderings the parser would reject.

## D5 — Transposition range guard

**Decision**: Default policy **`drop`** — if any note's shifted pitch leaves the configured singable MIDI
window `[lo, hi]` (default the full `0..127`, operator may narrow), the *whole variant* is dropped and a
`dropped: out_of_range` reason recorded; no out-of-range note is ever emitted. Optional policy **`clamp`**
— each offending pitch is clamped into the window and the clamp recorded, so a clamped variant is never
mistaken for a faithful transposition. The hard `0..127` bound is enforced regardless (the `Note` model
already validates `ge=0, le=127`, `scores/models.py:17`).

**Rationale**: Dropping yields cleaner labels (a transposition that no longer transposes some notes is a
lie); clamping is occasionally useful for narrow windows but must be explicit and logged. Both honour the
spec (FR-005) and the existing `Note` pitch validator gives a hard backstop.

## D6 — Per-note volume representation

**Decision**: Add an **optional in-memory `Note.gain: float | None`** (linear multiplier, `None` ⇒
nominal). It is **excluded from the label `.tsv`** — `serialize_score` does not emit it — so the
transcription label stays `(onset, offset, pitch[, lyric])` exactly (FR-009). The deterministic lane
multiplies each note's rendered samples by `note.gain` before mixing (`render/deterministic.py:137–140`;
the lane's single peak-normalise at `:152–154` preserves *relative* per-note levels). Model lanes (SVS,
voice-conversion) that cannot apply per-note gain ignore it and record `dynamics_applied: false` in
provenance (FR-008 edge case). The applied gains are reproducible from the variant seed (D2), so a
regenerated corpus reproduces them without persisting them in the label file.

**Rationale**: Amplitude is not part of Basic Pitch's onset/offset/pitch target (acoustic-only default per
spec), so it must not pollute the label file; an in-memory field threaded through the existing `Score`
object reaches the renderer with no schema change and no new sidecar file. The deterministic lane's
relative-amplitude-preserving normalisation makes the gain audible without per-note clipping.

**Alternatives considered**: (a) A 5th `.tsv` column — rejected: changes the label schema and risks a
consumer treating gain as a label. (b) A separate sidecar JSON per sample — rejected: extra files and I/O
for a value that is deterministically re-derivable from the seed; provenance already records enough to
audit it.

## D7 — Config schema

**Decision**: `RunConfig.score_augmentation: list[ScoreAugmentationProfile]` (default `[]`, like
`augmentation_profiles`, `config/models.py:104`). Each `ScoreAugmentationProfile` has `profile_id` and
three independent optional sub-blocks: `transpose` (list of semitone offsets + range policy/window),
`humanize_time` (σ, max-dev, N draws, overlap policy, min duration), `volume` (gain range/distribution).
An empty/absent profile list leaves runs unaffected (FR-001/013).

**Rationale**: Mirrors the existing seeded `AugmentationProfileConfig` pattern operators already know, and
the three sub-knobs map one-to-one to the three axes so a profile can enable any subset. A list (not a
single profile) lets a run sweep multiple score-aug recipes, composing as additional multiplier terms.

## D8 — Lyric preservation through transforms

**Decision**: All three transforms map notes **1:1** and carry each note's optional `lyric` (and the new
`gain`) onto the rewritten note. Transposition and humanisation never add/drop/reorder *which* note carries
which syllable, so syllable↔note and 002's melisma/tie alignment survive (FR-014). (Melisma = one syllable
sung across multiple notes; v1 score-aug never splits or merges notes, so it cannot break it.)

**Rationale**: Because every axis is a per-note value rewrite (pitch, timing, gain) with no structural
change to the note sequence, lyric alignment is preserved for free. A unit test pins it.

## D9 — Stats / coverage axis

**Decision**: Extend `build_stats` (`corpus/stats.py:36`) with a `score_augmentation` block: counts of
accepted samples per `score_aug_transform` (e.g. per transposition offset, per humanise draw, volume on/
off) and the share of accepted samples that are variants vs originals. Driven by the new
`base_score_id` / `score_aug_*` provenance fields, mirroring how `augmentation_coverage` is computed today.

**Rationale**: FR-012 requires the new axis surfaced alongside the existing voice/audio-aug axes; reusing
the manifest-scan pattern keeps stats a pure read over provenance.

## D10 — Evaluation suite

**Decision**: Add a `score_aug` suite to `evaluation.py` (criteria SC-001..SC-009 per the plan's
Evaluation Strategy) and two fixture configs (`score-aug-smoke.yaml`, `score-aug-off.yaml`). The
differential checks (SC-002 byte-identical onsets, SC-004 label byte-identity under volume, SC-005
determinism) compare a variant against its base score read from the same manifest/output; SC-001 compares
the off-control run against a pre-feature render.

**Rationale**: Constitution IV requires a single-command runnable verdict mapped to the spec's Success
Criteria before feature code. The suite extends the existing `EvalReport`/`CriterionResult` machinery
(`evaluation.py:24–46`) rather than introducing a parallel harness.

## Resolved unknowns

| Spec open question (issue #8) | Resolution |
|-------------------------------|------------|
| Volume → velocity label? | Acoustic-only; in-memory `Note.gain`, excluded from label `.tsv` (D6, FR-009) |
| Humanisation overlap policy | Default `forbid_overlap`; legato reserved (D4, FR-007) |
| Transposition range guard | Default `drop`; `clamp` opt-in, always logged; hard `0..127` (D5, FR-005) |
| Lyric (#4/002) interaction | 1:1 note mapping carries `lyric`; alignment preserved (D8, FR-014) |
| "Distinguishable from originals" | Physical `corpus/augmented/<axis>/` subtree + self-describing ids (D3, FR-016) |
