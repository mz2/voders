# Feature Specification: Score-Domain Augmentations (volume, time humanisation, transposition)

**Feature Branch**: `003-score-augmentations`
**Created**: 2026-06-27
**Status**: Draft
**Input**: GitHub issue mz2/voders#8 — "Score-domain augmentations: note volume variation, time humanisation, octave transpositions"

## Context

This feature extends the synthetic singing corpus generator (`001-synthetic-singing-corpus`) and is a
sibling of the lyric-generation follow-up (`002-lyric-generation`). Where 001's augmentation lane (US3)
multiplies the corpus along the **production / timbre axis**, this feature multiplies it along the
**score / label axis**: from each base score it fans out new *score variants* — and their already-correct
labels — along three axes.

1. **Octave transposition** — additional variants at a fixed pitch offset (±12, ±24 semitones).
2. **Time humanisation** — seeded jitter on note onsets and durations, producing new
   `(onset_s, offset_s)` combinations per variant.
3. **Note volume variation** — per-note dynamics (loud/soft, accents) so the corpus isn't uniformly
   leveled.

**The defining design choice is that this is an _input-data_ pre-processor, not an audio-domain lane.**
The stage is essentially a pure function `(base score, profile, seed) -> [variant scores]` that rewrites
the input score's note rows to produce new score variants **before anything renders**. Each variant is
then just another input score: it flows through the **entire existing 001 pipeline unchanged** — score
parsing, every renderer lane, and the f0/onset verification gate all operate on it identically, with no
lane-specific wiring. This keeps the surface tiny and the labels correct **by construction** — a
variant's labels *are* its modified note rows; there is nothing to re-derive.

**Why this is a new stage, not more of the existing audio augmentation lane.** The 001 augmentation lane
(`src/voders/render/augment.py`) is audio-domain and deliberately **label-safe**: its whole contract
(001 FR-005) is that reverb/codec/accompaniment **never move onsets/offsets or shift pitch**, so the
paired score stays byte-identical. The augmentations requested here are the **opposite category** — they
intentionally **rewrite the labels** at the input-data stage and let the pipeline render audio to match.

| Axis | Existing 001 audio lane (US3) | This stage (input score) |
|------|-------------------------------|--------------------------|
| Operates on | rendered audio | `Note` rows, before parsing/rendering |
| Labels | held byte-identical | **rewritten** into the variant |
| Correctness model | re-verify f0/onset on augmented audio | **correct by construction** (we own the transform) |
| Scales the | production / timbre axis | **score / label axis** |

**Governing constraint (opt-in, like 001/002):** the whole feature is optional. With no
score-augmentation configuration, a run emits exactly the base scores and reproduces existing corpora
byte-for-byte.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Octave transposition variants (Priority: P1)

A data engineer wants more pitch coverage in the corpus without hand-authoring new scores. They
configure a set of semitone offsets (e.g. `[-12, +12]`); for each base score the stage emits one
additional variant score per offset, with every note's pitch shifted by that offset and its onset/offset
labels unchanged. The variants flow through the existing renderers and verification gate untouched.

**Why this priority**: It is the cleanest of the three axes — a single integer offset, a pure note-row
rewrite, labels correct by construction — and it delivers the largest immediate corpus multiplier on the
pitch axis, which is the transcription model's primary target. It is the load-bearing MVP slice: shipping
only this already multiplies the corpus along a new axis.

**Independent Test**: Configure offsets on a fixture score set, run the stage, and confirm each base score
produces the expected number of variant scores whose pitches are shifted by exactly the offsets and whose
onset/offset rows are unchanged; out-of-range variants are handled by the configured guard; and each
variant renders and passes the existing 001 verification gate.

**Acceptance Scenarios**:

1. **Given** a base score and offsets `[-12, +12]`, **When** the stage runs, **Then** it emits two
   variant scores in which every note's `pitch_midi` is shifted by the offset and every `(onset_s,
   offset_s)` is byte-identical to the base.
2. **Given** an offset that would push any note outside the configured singable MIDI window (or outside
   the hard `0..127` bound), **When** the stage evaluates that variant, **Then** it applies the configured
   range guard (drop the whole variant by default) and records the decision in provenance, never emitting
   an out-of-range note.
3. **Given** the same `(base score, profile, seed)`, **When** the stage runs twice, **Then** the emitted
   variant scores are byte-identical (determinism).

---

### User Story 2 - Time humanisation variants (Priority: P2)

The engineer wants the corpus to contain realistic, non-quantised timing rather than perfectly
grid-aligned onsets. They configure onset/duration jitter (a standard deviation, a max-deviation budget,
and a number of draws N); for each base score the stage emits N variant scores, each with seeded jitter
applied to every note's onset and offset, producing new `(onset, length)` combinations while keeping the
score a valid monophonic sequence.

**Why this priority**: Timing diversity is the second-most valuable score-axis lever for a model that
predicts onsets and offsets, and it is materially more delicate than transposition because the rewrite
must preserve note ordering, non-overlap, and positive duration. It depends on the same pre-processor
machinery US1 establishes but adds the constraint-preservation logic.

**Independent Test**: Configure jitter on a fixture score with a fixed seed, render the N draws, and
confirm the onsets/offsets differ from the base within the configured budget, every variant is still a
valid monophonic score (ordered, non-overlapping, positive durations), and the draws are reproducible for
the seed.

**Acceptance Scenarios**:

1. **Given** a base score and a humanisation profile with N draws, **When** the stage runs, **Then** it
   emits N variant scores whose per-note onset/offset deviations from the base are non-zero and within the
   configured max-deviation budget.
2. **Given** any emitted humanised variant, **When** it is validated, **Then** it is a valid monophonic
   score — notes ordered by onset, no two notes overlapping (per the configured overlap policy), every
   duration strictly positive — so the existing parser accepts it unchanged.
3. **Given** the same `(base score, profile, seed)`, **When** the stage runs twice, **Then** the N variant
   draws are byte-identical.

---

### User Story 3 - Per-note volume / dynamics variation (Priority: P3)

The engineer wants the corpus to span a range of dynamics — accents, crescendi, loud and soft notes —
rather than every note at a uniform level. They enable per-note volume variation; the stage attaches a
seeded per-note gain to each note of a variant, and the renderers honour that gain when synthesising
audio. Because amplitude is **not** part of the `(onset, offset, pitch)` transcription label, this axis is
label-preserving for the model's target.

**Why this priority**: Dynamics add acoustic realism but, unlike the other two axes, change nothing in the
transcription labels, so they are the least load-bearing for label diversity and the most droppable. They
also require touching the renderers to honour a per-note gain, a slightly larger surface than the pure
note-row rewrites of US1/US2.

**Independent Test**: Enable volume variation on a fixture score, render with and without it for the same
seed, and confirm the rendered note levels vary as configured while the onset/offset/pitch labels are
byte-identical to the un-varied render and still pass the verification gate.

**Acceptance Scenarios**:

1. **Given** a base score with volume variation enabled, **When** a variant is produced, **Then** each note
   carries a seeded per-note gain within the configured range and the `(onset_s, offset_s, pitch_midi)`
   rows are unchanged from the base.
2. **Given** a variant carrying per-note gains, **When** a renderer synthesises it, **Then** the rendered
   per-note levels reflect those gains and the sample still passes the existing onset/offset/pitch
   verification gate.
3. **Given** a lane that does not support per-note gain, **When** it renders a gain-bearing variant,
   **Then** it renders at nominal level and the manifest records that the dynamics were not applied, rather
   than failing.

---

### User Story 4 - Originals and augmentations are unmistakably separated (Priority: P3)

A reviewer or downstream consumer must never confuse a base ("original") score and its rendered audio
with a score-augmented variant. The operator wants the two physically and namingly distinct: augmented
scores and their rendered audio land in their own clearly-labelled output locations (e.g. separate
directories), and each variant's identifier encodes the augmentation that produced it, so a file is
self-describing and an original is never overwritten or intermixed with a variant.

**Why this priority**: This feature deliberately *rewrites labels*, so a variant on disk looks exactly
like a legitimate score — the only thing preventing silent corruption of the "real" corpus is unambiguous
separation. Without it, an augmented score could be mistaken for an original, contaminating evaluation
splits or label audits. The minimal "write variants to a separate folder" guarantee should land with the
first augmentation axis (US1); the full distinguishability story here — self-describing identifiers,
recorded output location, and the provenance/coverage reporting it shares — is the P3 hardening that must
be complete before any augmented corpus is released.

**Independent Test**: Run end-to-end with score augmentations enabled and confirm (a) originals and
augmented variants occupy distinct, clearly-named output locations with no file collisions, (b) every
variant's identifier/path encodes its base score and the applied augmentation, and (c) the manifest reports
the base score id, profile, per-sample seed, and applied transform, and the aggregate statistics surface
score-augmentation coverage as its own axis.

**Acceptance Scenarios**:

1. **Given** a run with score augmentations enabled, **When** outputs are written, **Then** original
   (un-augmented) scores and their audio occupy a different output location from augmented variants, no
   variant file overwrites or shares a path with an original, and each location is labelled so a human can
   tell originals from augmentations at a glance.
2. **Given** any augmented variant file (score or audio), **When** its identifier/path is inspected,
   **Then** it encodes both the base score it derives from and the augmentation applied (e.g. the offset or
   draw), so the file is self-describing without consulting the manifest.
3. **Given** any sample derived from a variant, **When** the operator queries the manifest, **Then** it
   returns the base score id, the score-augmentation profile id, the per-sample seed, the applied transform
   (offset / jitter draw / dynamics), and the variant's output location.
4. **Given** a completed run, **When** the operator reads the aggregate statistics, **Then** the
   score-augmentation coverage (e.g. counts per transposition offset and per humanisation draw) is reported
   as its own axis alongside the existing voice and audio-augmentation axes.
5. **Given** a published manifest, **When** the corpus is regenerated from it, **Then** every variant is
   reproduced byte-identically from `(base score, profile, seed)` with no re-derivation, into the same
   separated layout.

---

### Edge Cases

- **Out-of-range transposition**: A semitone offset that pushes any note outside the configured singable
  MIDI window (or the hard `0..127` bound) causes the whole variant to be dropped by default, with the
  drop recorded in provenance. (An optional clamp policy is configurable; clamping is recorded so a
  clamped variant is never mistaken for a faithful transposition.)
- **Humanisation that breaks monophony**: A jitter draw that would make two notes overlap, reorder them,
  or drive a duration to zero/negative is re-drawn or shrunk within the max-deviation budget so the emitted
  variant is always a valid monophonic score; if no valid draw exists within budget, the note is left at
  its base timing and the constraint hit is recorded, never an invalid score emitted.
- **Max-deviation budget**: Humanisation jitter is bounded so labels stay musically valid; a configured σ
  large enough to routinely exceed the budget is surfaced rather than silently truncated everywhere.
- **Zero-variant / identity profile**: A profile with no offsets, no jitter, and no dynamics emits only the
  base score, byte-identical to a run with the feature off (opt-in safety).
- **Interaction with lyrics (#4 / 002)**: Transposition and humanisation rewrite note timing/pitch but MUST
  keep each note's optional `lyric` attached to that same note, so syllable↔note alignment (and any
  melisma/tie handling) survives the transform; a transposed or humanised variant carries the same lyrics
  as its base, re-aligned to the moved notes.
- **Empty / single-note scores**: A score with zero or one note transposes and humanises trivially (no
  ordering/overlap constraint to violate) and produces well-formed variants.
- **Corpus-multiplier blow-up**: Because variants multiply with voices and audio-augmentation profiles
  (`variants × voices × audio-profiles`), the effective sample count is the product; the run reports the
  multiplier so an operator is not surprised by the corpus size.
- **Volume on a non-supporting lane**: A lane that cannot honour per-note gain renders at nominal level and
  records that dynamics were not applied, rather than failing the sample.
- **Variant masquerading as an original**: Because augmentation rewrites labels, a variant file is
  structurally indistinguishable from a real score; the separation rule (distinct, labelled output
  locations + self-describing identifiers, FR-016) is what prevents a variant from being mistaken for an
  original or silently overwriting one. Two variants of the same base (e.g. `+12` and a humanisation draw)
  must also never collide with each other.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST treat score augmentation as fully optional. With no score-augmentation
  configuration, a run MUST emit exactly the base scores and reproduce existing corpora byte-for-byte.
- **FR-002**: The system MUST realise score augmentation as a **score pre-processor stage** that emits
  variant input scores **before** any renderer runs, so the existing parser, all renderer lanes, and the
  verification gate consume the already-varied score and labels with **no lane-specific changes**.
- **FR-003**: A variant's labels MUST be the modified note rows themselves (correct by construction); the
  system MUST NOT re-derive or relabel a variant's onsets, offsets, or pitches from rendered audio for the
  purpose of producing the variant. (The existing 001 audio verification gate still runs on the rendered
  variant as an independent backstop.)
- **FR-004**: The system MUST support **octave/semitone transposition**: for each configured semitone
  offset it emits a variant score whose every note pitch is shifted by that offset, with onsets and offsets
  unchanged.
- **FR-005**: Transposition MUST enforce a range guard: any note that a shift would push outside the
  configured singable MIDI window, or outside the hard MIDI `0..127` bound, MUST cause the configured guard
  to act (drop the whole variant by default, or clamp if explicitly configured), and the action MUST be
  recorded in provenance. The system MUST NEVER emit a note outside `0..127`.
- **FR-006**: The system MUST support **time humanisation**: seeded jitter applied to each note's onset and
  offset/duration, producing new `(onset_s, offset_s)` combinations, with a configurable jitter magnitude,
  a configurable max-deviation budget, and a configurable number of draws N per base score.
- **FR-007**: Every emitted humanised variant MUST be a valid monophonic score — notes ordered by onset, no
  two notes overlapping (per the configured overlap policy), and every duration strictly positive — so the
  existing parser accepts it unchanged. A draw that cannot satisfy these within the max-deviation budget
  MUST be corrected (re-drawn / shrunk / left at base timing) rather than emitted invalid.
- **FR-008**: The system MUST support **per-note volume variation**: a seeded per-note gain/dynamics value
  attached to each note of a variant, within a configurable range/distribution, which the renderers honour
  when synthesising audio.
- **FR-009**: Per-note volume variation MUST be **label-preserving for the transcription target**: it MUST
  NOT change a note's `(onset_s, offset_s, pitch_midi)`. Amplitude rides on the variant as a separate
  per-note field and is not part of the transcription label schema; whether it is also surfaced as a
  predicted label (velocity) is out of scope (default: acoustic-only).
- **FR-010**: Every score-augmentation transform MUST be **deterministic**: each variant's rows are a pure
  function of `(base score, profile, seed)`, so the same inputs yield byte-identical variants, independent
  of worker count or processing order (consistent with 001's seeding model).
- **FR-011**: The system MUST record, per sample, the score-augmentation provenance: the base score id, the
  score-augmentation profile, the per-sample seed, and the applied transform parameters (offset / jitter
  draw index / dynamics), appended to the existing provenance manifest, so each variant's lineage back to
  its base score is queryable without re-running the pipeline. This is recorded alongside, and independent
  of, the existing audio-augmentation profile.
- **FR-012**: The system MUST report the score-augmentation coverage as its own axis in the run's aggregate
  statistics (e.g. counts per transposition offset and per humanisation draw), alongside the existing voice
  and audio-augmentation axes.
- **FR-013**: The operator MUST be able to configure score augmentation per run via the existing
  declarative run config, as a seeded **score-augmentation profile** block with independent sub-knobs for
  transposition, time humanisation, and volume variation; the default (absent) configuration leaves runs
  unaffected.
- **FR-014**: Score augmentation MUST preserve each note's optional `lyric` association through the
  transform, so a transposed or humanised variant carries the same per-note lyrics as its base (re-aligned
  to the moved notes) and syllable↔note / melisma alignment from 002 is not broken.
- **FR-015**: Score augmentation MUST compose as a **corpus multiplier**: the effective sample count is
  `score variants × voices × audio-augmentation profiles`. The system MUST surface this multiplier (or the
  resulting projected sample count) so the operator can anticipate corpus size before a run.
- **FR-016**: Score-augmented variants MUST be kept **physically and namingly distinguishable from the
  originals**. The system MUST write augmented variant scores and their rendered audio to output locations
  (e.g. separate directories) that are clearly separated and labelled apart from the base/original scores
  and audio; MUST NOT overwrite, replace, or co-mingle a variant with an original (no path collisions); and
  MUST give each variant an identifier/path that encodes both its base score and the augmentation applied
  (offset / draw / dynamics), so a file is self-describing. The base ("original", un-augmented) scores MUST
  always be retrievable as a distinct set. The chosen output location for each variant MUST be recorded in
  provenance.

### Key Entities *(include if feature involves data)*

- **Score Variant**: A new input score derived from a base score by a score-augmentation transform. Its
  note rows *are* its labels. Carries the base score id lineage plus a variant discriminator, and any
  per-note `lyric` from the base, re-aligned to the moved notes.
- **Score-Augmentation Profile**: A seeded configuration block with independent sub-knobs — `transpose`
  (list of semitone offsets), `humanize_time` (jitter magnitude, max-deviation budget, number of draws,
  overlap policy), and `volume` (range/distribution) — selected per run, defaulting to absent (no
  augmentation).
- **Per-Note Gain / Dynamics**: An optional per-note amplitude value attached to a variant's notes that
  renderers honour; not part of the transcription label schema (default acoustic-only).
- **Score-Augmentation Provenance**: The per-sample record of base score id, profile, seed, and applied
  transform parameters, written to the manifest alongside the audio-augmentation provenance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of runs with no score-augmentation configuration emit exactly the base scores and
  reproduce existing corpora byte-for-byte (backward compatibility / opt-in safety).
- **SC-002**: For configured transposition offsets, 100% of emitted transposition variants have every note
  shifted by exactly the offset with onsets/offsets byte-identical to the base, and zero emitted notes fall
  outside MIDI `0..127`.
- **SC-003**: 100% of emitted time-humanisation variants are valid monophonic scores (ordered,
  non-overlapping per policy, strictly positive durations) accepted by the existing parser without
  modification, and 100% of per-note timing deviations fall within the configured max-deviation budget.
- **SC-004**: With volume variation enabled, the rendered corpus spans a measurably wider per-note level
  distribution than the un-varied baseline, while 100% of `(onset, offset, pitch)` labels remain
  byte-identical to the un-varied render of the same input.
- **SC-005**: Every score-augmentation transform is reproducible: for a fixed `(base score, profile,
  seed)`, 100% of emitted variants are byte-identical regardless of worker count or processing order.
- **SC-006**: Every sample derived from a variant records, in provenance, its base score id, profile,
  per-sample seed, and applied transform parameters; an operator can reconstruct any variant's lineage to
  its base score with zero pipeline re-runs.
- **SC-007**: The run's aggregate statistics report score-augmentation coverage as a distinct axis, and the
  reported effective sample count equals `variants × voices × audio-profiles` for the run.
- **SC-008**: For samples that flow through the existing 001 verification gate, score-augmentation variants
  pass at the same rate as equivalent base scores — i.e. the pre-processor introduces no new class of
  verification failure beyond what the (correct-by-construction) labels and existing tolerances already
  define.
- **SC-009**: In any run with score augmentations enabled, 100% of augmented variant files (scores and
  audio) reside in output locations distinct from the originals with zero path collisions, and a reviewer
  can correctly classify any file as original vs augmented — and, for an augmented file, name its base score
  and applied augmentation — from its location/identifier alone, without consulting the manifest.

## Assumptions

- The downstream consumer remains the 001 note/pitch transcription model; the transcription label schema is
  unchanged. Transposition and humanisation rewrite the existing `(onset, offset, pitch)` labels; volume
  variation adds a non-label per-note amplitude (acoustic-only by default).
- **Humanisation overlap policy** defaults to **forbidding overlaps** to keep monophonic-clean labels
  (matching the existing `Score` monophony invariant); a legato/overlap policy is a configurable option but
  not the default.
- **Transposition range guard** defaults to **dropping the whole out-of-range variant** (cleaner labels)
  rather than clamping; clamping is an explicit opt-in and is always recorded so a clamped variant is never
  mistaken for a faithful transposition.
- **Volume variation** is **acoustic-only** by default (Basic Pitch's target is onset/offset/pitch, not
  velocity); surfacing velocity as a predicted label is out of scope for v1.
- The stage runs **upstream of the renderers** as a pure score-to-scores function, so no renderer,
  voice-conversion, or audio-augmentation-lane code needs to change for the transposition and humanisation
  axes; only the volume axis requires renderers to honour a per-note gain, and lanes that cannot do so
  degrade gracefully (render at nominal level, record non-application).
- The existing 001 run config, seeding model, manifest, parser, and audio verification gate exist and are
  reused; this feature adds a score-augmentation profile selector and a score-augmentation provenance axis
  rather than new alignment or reproducibility machinery.
- Determinism is achieved by deriving each variant's randomness from `(base score, profile, seed)` exactly
  as 001 derives audio-augmentation randomness from a seed, so the existing reproducibility guarantees
  extend to score variants without new infrastructure.
- The supplied annotations contain no dynamics, tempo-feel, or transposition metadata; all three axes are
  operator-configured generators, never read from the score set.
- This feature is independent of, but composes cleanly with, 002 (lyric generation): lyrics travel with
  their note through every score-augmentation transform.
- "Clearly distinguishable from the originals" is interpreted as **physical + naming separation**:
  augmented variants live in their own labelled output locations (e.g. per-augmentation subdirectories),
  carry self-describing identifiers (base score + applied transform), and never overwrite or share a path
  with an original — so originals remain a cleanly retrievable set. The exact directory/naming scheme is an
  implementation choice for the plan; the spec requires only that the separation be unambiguous, collision-
  free, and recorded in provenance.
