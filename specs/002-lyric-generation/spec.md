# Feature Specification: Optional Lyric Generation & Phonetic Diversity

**Feature Branch**: `002-lyric-generation`
**Created**: 2026-06-27
**Status**: Draft
**Input**: User description: "Please specify a solution to mz2/voders/issues/4 taking the prototype on branch \"lyrics\" also into account"

## Context

This feature extends the synthetic singing corpus generator (feature `001-synthetic-singing-corpus`).
The downstream consumer of the corpus is a note/pitch transcription model (Basic Pitch-class) that
learns **onsets, pitches, and offsets — not words**. Lyrics therefore carry **no label value**; their
sole purpose here is **phonetic diversity** in the rendered audio — the consonants, plosives,
fricatives, and coarticulation that the corpus's current single open vowel ("ah") entirely lacks.
Lyric content is thus a **realism / domain-randomization lever**, a sibling of 001's timbre
multiplication (US2) and production augmentation (US3) — never a corpus label.

A foundation already exists on the `lyrics` branch (commit "Lyrics foundation"): a `Note` gained an
optional per-note `lyric` field, the score parser reads an optional 4th column as that lyric
(three-column scores parse exactly as before), and serialization emits the 4th column only when a
score carries lyrics, so lyric-free output stays byte-identical to today's. This feature builds on
that foundation: it makes lyrics **audible** (sung as phonemes by the expressive lane), adds an
**automatic** lyric source for corpus-scale use, and records lyric provenance.

**Two governing constraints (from the issue and the operator):**
1. The whole feature is **optional and opt-in**; with no lyric configuration the corpus stays
   vowel-only and every existing run reproduces byte-for-byte.
2. The supplied annotations carry **no lyric or theme data**. Any theming exists only as a generated
   input the operator opts into — it is never read from the score set.

## Clarifications

### Session 2026-06-27

- Q: What does "no timing or pitch shift from adding vocal synthesis" mean operationally? → A: A
  **differential (lyric-on vs lyric-off) null-difference guarantee**. For the same (score, voice,
  seed), rendering *with* lyrics MUST NOT move a note's pitch or its labeled onset/offset relative to
  rendering the *same* input *without* lyrics (the open-vowel render). The lyric layer is required to
  be **pitch-neutral and timing-neutral** with respect to the labels: it changes *what phonemes are
  sung*, not *where notes start/end or what pitch they are*. Any sample that violates this is rejected,
  never silently admitted or relabeled. This strengthens — does not replace — the absolute
  score-tolerance checks (SC-002) with a relative check against the lyric-free baseline.
- Q: How is the labeled onset defined once a note carries a leading consonant? → A: The **vowel
  nucleus on the score beat** is the note. A leading consonant is articulated in a short pre-onset
  window and is **not** the labeled onset; the validator's timing comparison targets the vowel onset
  (with any method group delay subtracted, 001 FR-019). A net residual shift the system cannot
  compensate causes rejection, not relabeling.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Sing supplied lyrics with the label safety net intact (Priority: P1)

A data engineer has a score whose notes carry syllables (a 4th column in the `.tsv`, or a `lyric`
on each note). They enable the expressive (neural singing-voice-synthesis) lane and ask it to
articulate those syllables instead of a flat "ah", so the rendered vocals contain real consonants
and vowel transitions. Crucially, the note onset/pitch/offset labels must stay correct — adding
consonants must not silently push onsets past tolerance.

**Why this priority**: This is the load-bearing slice. Without it, the prototype's stored lyric text
is inert — nothing sings it. It is also where lyric-driven neural voices finally earn their keep over
the deterministic vowel lane. It reuses 001's existing expressive-lane safety net (force-score-F0, or
re-derive onsets/offsets from the rendered audio and reject deviations over tolerance), so phonetic
realism is added without weakening label guarantees.

**Independent Test**: Take one score with syllables, render it through the expressive lane in both
safety modes, and confirm (a) the audio contains articulated consonants distinguishable from the
vowel baseline, and (b) every accepted sample still passes the onset/offset/pitch tolerance checks or
is rejected with the deviation logged — never silently admitted.

**Acceptance Scenarios**:

1. **Given** a score whose notes carry syllables, **When** the expressive lane renders it in
   force-score-F0 mode, **Then** the output carries the unchanged score labels and passes the same
   onset/offset/pitch tolerance checks the vowel render passes.
2. **Given** the same score rendered in re-derive mode, **When** an added pre-onset consonant shifts a
   measured onset beyond the 50 ms tolerance, **Then** that sample is rejected (or the labels
   re-derived within tolerance), with the deviation recorded in provenance — never silently relabeled.
3. **Given** a score with **no** lyrics, **When** any lane renders it, **Then** behavior and output
   are byte-identical to the pre-feature pipeline (open-vowel render, three-column score).
4. **Given** the same (score, voice, seed) rendered once **with** lyrics and once **without** (the
   open-vowel baseline), **When** both are measured, **Then** every accepted note's pitch matches the
   baseline (no pitch shift attributable to lyrics) and every accepted note's onset/offset matches the
   baseline within the validator's timing resolution (no timing shift attributable to lyrics); any
   note that does not is rejected, not admitted.

---

### User Story 2 - Automatic syllable source for corpus-scale phonetic diversity (Priority: P2)

Because the supplied annotations contain no lyrics, the engineer wants phonetic diversity across a
10,000–100,000-sample run **without authoring lyrics by hand**. They turn on an automatic syllable
source that assigns a singable syllable to each note, drawn to maximize phonetic coverage, derived
deterministically from the run's master seed so any sample is reproducible in isolation.

**Why this priority**: This is the source the corpus actually uses at scale, since no lyric input
data exists. It runs on commodity CPU and is fully deterministic, so it neither pulls a GPU
dependency into the baseline nor breaks 001's bit-exact reproducibility. It depends on US1 (something
must sing the syllables) but delivers the realism payoff the feature exists for.

**Independent Test**: Enable the automatic source on a fixture score set with a fixed master seed,
render twice with different worker counts, and confirm the assigned syllables (and resulting audio)
are identical across runs and that the corpus's phoneme inventory is materially broader than the
vowel-only baseline.

**Acceptance Scenarios**:

1. **Given** a score set with no lyrics and the automatic source enabled, **When** the run executes,
   **Then** every note receives a singable syllable and the corpus covers a documented minimum set of
   distinct phoneme classes well beyond a single vowel.
2. **Given** the same master seed and configuration, **When** the run is repeated with a different
   worker count or processing order, **Then** the assigned syllables for every sample are identical.
3. **Given** a note shorter than the learned `min_note_ms` threshold, **When** a syllable would be
   assigned to it, **Then** the note follows 001's existing short-note flag/reject policy rather than
   receiving an unsingable syllable.

---

### User Story 3 - Lyric provenance and source/license audit (Priority: P3)

A reviewer auditing a corpus needs to answer "where did this sample's lyrics come from?" and "was any
lyric-generation model used whose license forbids this use?" without re-running the pipeline.

**Why this priority**: Lyrics introduce a new provenance axis and, for model-based sources, a new
licensing surface. Without it the corpus's lyric composition is unauditable and the licensing
discipline 001 applies to donor voices would not extend to lyric models. It depends on a lyric source
existing (US1/US2) but is required before any corpus using non-default lyrics is released.

**Independent Test**: Run end-to-end with each lyric source, then query the manifest for any sample
and confirm it reports the lyric source, a stable reference to the exact lyric text used, and — for
model-based sources — the model identity and license; and that a model with an unacceptable license
is refused and the refusal surfaced.

**Acceptance Scenarios**:

1. **Given** any sample, **When** the operator queries the manifest, **Then** it returns the lyric
   source (vowel | supplied | automatic | generated) and a stable reference (hash) to the exact lyric
   text used for that sample.
2. **Given** a sample produced with a model-based lyric source, **When** the manifest is queried,
   **Then** it additionally records the model identity and license tag.
3. **Given** a lyric-generation model whose license is unacceptable for the run, **When** a stage
   tries to use it, **Then** the stage refuses, no sample is produced from it, and the refusal is
   surfaced in the manifest — mirroring the donor-voice consent refusal.

---

### User Story 4 - Themed generated lyrics, reproducible without re-running the model (Priority: P4)

An operator optionally wants coherent, themed lyrics (e.g., "winter, longing") rather than scattered
syllables, to make coarticulation statistics more song-like. They supply a theme string; a lyric
model generates count-matched lyrics once; the generated text is pinned as a replayable artifact so
the corpus reproduces from the pinned text rather than by re-running a non-deterministic model.

**Why this priority**: This is the furthest opt-in and the only path that pulls in a heavier
(model-based, possibly GPU) generator. Its marginal benefit over US2's automatic syllables is modest
for a pitch-only consumer, so it is last and fully droppable. The theme string is the **only** place
theming enters and is operator-supplied, never read from annotations.

**Independent Test**: Provide a theme, generate lyrics for a fixture score set, record the pinned
lyric artifact, then reproduce the corpus from the manifest and confirm the rendered samples use the
pinned text (identical lyrics) without invoking the model again.

**Acceptance Scenarios**:

1. **Given** a theme string and the generated source enabled, **When** the run executes, **Then** each
   score receives count-matched lyrics and the generated text is persisted as a pinned, hash-
   referenced artifact in/alongside the manifest.
2. **Given** a published manifest from such a run, **When** the corpus is regenerated, **Then** the
   pinned lyric text is reused verbatim and no lyric model is re-invoked, and the regenerated samples
   meet 001's reproducibility tolerance for the lane that rendered them.
3. **Given** a generated lyric whose syllable count does not match a score's note count, **When**
   rendering proceeds, **Then** the mismatch is resolved by the documented fallback (below) and
   surfaced in provenance, never by dropping or misaligning a note's label.

---

### Edge Cases

- **Syllable/note count mismatch**: When a lyric source yields fewer or more syllables than notes, the
  unmatched notes fall back to the neutral open vowel ("ah"); the mismatch is recorded in provenance.
  No note label is dropped, added, or shifted (consistent with 001's existing mismatch policy).
- **Melisma (one syllable across multiple notes)**: A single syllable sustained over tied/legato notes
  is articulated once and held; the choice is recorded so alignment remains explainable.
- **Empty 4th column / blank lyric**: An empty lyric cell is treated as "no lyric" (vowel fallback),
  identical to a three-column score; output stays byte-identical for fully lyric-free scores.
- **Un-pronounceable or out-of-inventory text**: Supplied text the phonetic step cannot render is
  flagged and the affected notes fall back to the open vowel, with the substitution logged.
- **Sub-tolerance notes carrying lyrics**: Notes below the learned `min_note_ms` threshold follow
  001's existing flag/reject policy regardless of any assigned syllable.
- **Lyric model unavailable or license-refused at runtime**: The run does not silently fall back to a
  different model; the affected samples are skipped/refused and logged (mirroring a missing donor
  voice).
- **Lyrics on a non-articulating lane**: Lyrics assigned to the deterministic vowel lane or carried
  through the timbre-only voice-conversion lane do not change their audio; only the expressive lane
  articulates them. This is recorded so the corpus is not assumed to contain consonants it does not.
- **Net lead-in / latency from articulation**: A leading consonant, voicebank lead-in silence, or a
  G2P/vocoder constant delay could shift the whole render in time. The labeled onset targets the vowel
  nucleus on the score beat, and any constant method delay is subtracted (001 FR-019); a residual
  per-note shift versus the lyric-free baseline that exceeds the timing resolution causes rejection,
  never a silent relabel.
- **Articulation perturbing pitch**: If a voicebank's phoneme model would bend the pitch of a note
  (e.g. consonant-induced micro-pitch), the system must keep the note's pitch equal to the lyric-free
  (score-derived) render; a note whose pitch deviates from that baseline is rejected.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST treat lyrics as fully optional. With no lyric configuration, every run
  MUST behave exactly as the pre-feature pipeline, producing byte-identical audio and three-column
  scores, and reproducing existing corpora bit-for-bit.
- **FR-002**: The system MUST accept operator-supplied per-note lyrics carried with the score (an
  optional 4th score column / optional per-note `lyric`), without breaking parsing or
  byte-identity of lyric-free scores.
- **FR-003**: The system MUST provide an automatic lyric source that assigns a singable syllable to
  each note, requiring no lyric input data, selected to broaden phonetic coverage beyond a single
  vowel.
- **FR-004**: The automatic lyric source MUST be deterministic with respect to the run's master seed
  and stable per-sample identifiers, so assigned syllables are reproducible in isolation, independent
  of worker count or processing order (consistent with 001's seeding model).
- **FR-005**: The automatic and operator-supplied lyric sources MUST run on commodity CPU without a
  GPU, so the corpus's baseline phonetic-diversity capability does not depend on accelerated hardware.
- **FR-006**: Only the expressive (neural SVS) lane is REQUIRED to articulate lyrics as phonemes. The
  deterministic and voice-conversion lanes MUST remain unchanged in audio output when lyrics are
  present, and the manifest MUST record that those lanes did not articulate the lyrics.
- **FR-007**: When the expressive lane articulates lyrics, the system MUST preserve label correctness
  using 001's existing expressive-lane safety net — either forcing the score-derived F0 contour, or
  re-deriving onsets/offsets from the rendered audio — and MUST reject (not silently relabel) any
  sample whose onsets deviate beyond the configured tolerance.
- **FR-008**: When a lyric source yields a syllable count that does not match a score's note count, the
  system MUST resolve the mismatch by a documented fallback (unmatched notes default to the neutral
  open vowel) and record the mismatch in provenance, never dropping, adding, or shifting a note label.
- **FR-009**: The system MUST record, per sample, the lyric source (vowel | supplied | automatic |
  generated) and a stable reference (content hash) to the exact lyric text used, appended to the
  existing provenance manifest.
- **FR-010**: For any model-based lyric source, the system MUST record the model identity and a license
  tag in provenance and MUST refuse to use a model whose license is unacceptable for the run,
  surfacing the refusal — consistent with 001's donor-voice consent discipline.
- **FR-011**: The system MUST support an optional generated lyric source driven by an operator-supplied
  theme string. The theme is the only channel through which theming enters; the system MUST NOT read
  themes or lyrics from the annotation/score set.
- **FR-012**: Generated lyrics from a non-deterministic model MUST be produced once and persisted as a
  pinned, hash-referenced artifact so a published corpus reproduces from the pinned text without
  re-invoking the model. Reproducibility of the resulting audio then follows 001's per-lane tolerance.
- **FR-013**: The system MUST let the operator select the lyric source per run via the existing
  declarative run config, defaulting to the vowel (lyric-free) source so existing configs are
  unaffected.
- **FR-014**: The system MUST report, in the run's aggregate statistics, the lyric-source breakdown
  and a phonetic-coverage summary, so an operator can confirm the corpus gained phonetic diversity.
- **FR-015 (pitch neutrality)**: Adding lyrics MUST NOT alter a note's pitch. The sung fundamental
  frequency MUST remain score-derived, so that for the same (score, voice, seed) each accepted note's
  measured pitch equals that of the lyric-free (open-vowel) render of the same input within the
  validator's pitch tolerance (±25 cents over ≥80% of the sustained interval). A note whose pitch
  deviates from that baseline because of articulation MUST be rejected, never admitted or relabeled.
- **FR-016 (timing neutrality)**: Adding lyrics MUST NOT move a note's labeled onset or offset. For the
  same (score, voice, seed), each accepted note's measured onset/offset MUST match the lyric-free
  render of the same input within the validator's timing resolution (≤10 ms), in addition to staying
  within the absolute score tolerance (FR-007). The labeled onset is the vowel nucleus on the score
  beat; a leading consonant is articulated in a pre-onset window and is not the labeled onset. A note
  that cannot meet this relative bound MUST be rejected.
- **FR-017**: The system MUST compensate any constant, documented method delay introduced by the
  articulation path (G2P/voicebank/vocoder lead-in) by subtracting it before the validator compares
  onsets/offsets (consistent with 001 FR-019). Only non-constant residual shift counts against
  FR-016; unknown delays are measured once, not guessed.
- **FR-018**: The evaluation harness MUST verify FR-015 and FR-016 as a differential check — rendering
  a fixture set both with and without lyrics for the same seeds and asserting the per-note pitch and
  onset/offset deltas between the two are within the tolerances above — so "no shift from adding vocal
  synthesis" is a gated, reproducible verdict rather than a claim.

### Key Entities *(include if feature involves data)*

- **Lyric**: An optional per-note syllable/text attached to a Note. Absent (the default) means the
  note is sung as a neutral open vowel. Lyrics are never a corpus label.
- **Lyric Source**: The origin of a sample's lyrics — one of vowel (default), supplied (carried with
  the score), automatic (seed-derived syllables), or generated (model + theme). Recorded in
  provenance.
- **Lyric Artifact**: The exact lyric text used for a score, content-hashed and — for generated
  lyrics — pinned alongside the manifest so the run replays without re-running a model.
- **Phonetic Rendering (expressive lane only)**: The articulation of a Lyric into sung phonemes by the
  expressive lane, governed by 001's force-score-F0 / re-derive safety net.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of lyric-free scores produce audio and score files byte-identical to the
  pre-feature pipeline, and existing corpora reproduce bit-for-bit (backward compatibility).
- **SC-002**: For samples where the expressive lane articulates lyrics, at least 99% of accepted
  samples still meet the onset tolerance (within 50 ms) and offset tolerance (within max(50 ms, 20% of
  note length)); samples that cannot are rejected, never silently admitted.
- **SC-003**: With the automatic lyric source enabled, the corpus's distinct sung-phoneme inventory is
  at least an order of magnitude larger than the vowel-only baseline (which is effectively one vowel).
- **SC-004**: Automatic-source lyrics are reproducible: for a fixed master seed, 100% of samples
  receive identical syllables regardless of worker count or processing order.
- **SC-005**: 100% of syllable/note count mismatches are resolved by the documented vowel fallback with
  the mismatch logged; zero note labels are dropped, added, or shifted by lyric handling.
- **SC-006**: Every sample's provenance records its lyric source and a stable reference to the exact
  lyric text; for model-based sources it also records model identity and license. An end-of-run audit
  finds zero samples using a license-refused lyric model.
- **SC-007**: For generated-lyric runs, the corpus regenerates from the manifest using the pinned
  lyric text with zero re-invocations of the lyric model, and the regenerated samples meet 001's
  per-lane reproducibility tolerance.
- **SC-008 (no pitch shift)**: In a differential test rendering the same fixture set with and without
  lyrics for identical seeds, at least 99% of accepted lyric notes have a measured pitch within ±25
  cents (over ≥80% of the sustained interval) of the lyric-free render of the same note; notes
  exceeding this are rejected, so zero pitch shift attributable to lyrics reaches the accepted corpus.
- **SC-009 (no timing shift)**: In the same differential test, at least 99% of accepted lyric notes
  have a measured onset and offset within 10 ms of the lyric-free render of the same note (after
  constant-delay compensation), and 100% of accepted notes additionally stay within the absolute score
  tolerance; notes exceeding the relative bound are rejected, so zero timing shift attributable to
  lyrics reaches the accepted corpus.

## Assumptions

- The downstream consumer remains the 001 note/pitch transcription model; lyrics are never emitted as
  a label and have no effect on the corpus's label schema beyond the optional, ignorable lyric field.
- The supplied score annotations contain no lyrics or themes; the automatic source needs no input
  data, and any theming is an operator-supplied string consumed only by the optional generated source.
- The automatic source aims for phonetic *coverage*, not lyrical meaning; syllables need only be
  singable and pronounceable, not coherent words.
- The default melisma policy assigns one syllable per note and sustains a single syllable across
  explicitly tied/legato notes; a different policy can be configured but is not required for v1.
- Only the expressive lane articulates lyrics; the deterministic and voice-conversion lanes are
  unaffected. Operators wanting consonant-bearing audio enable the expressive lane.
- The operator holds the rights to any lyric-generation model and any operator-supplied lyric text; the
  system enforces the manifest/license record but cannot verify upstream rights, as in 001.
- 001's run config, seeding, manifest, validator, and expressive-lane safety net exist and are reused;
  this feature adds a lyric-source selector and a lyric provenance axis rather than new alignment or
  reproducibility machinery.
- The generated (model-based) source is the lowest-priority, fully droppable slice; the feature
  delivers its core value (phonetic diversity) through the supplied and automatic sources alone.
- Pitch neutrality (FR-015) holds **by construction** in the force-score-F0 mode, where the sung f0 is
  the score's f0 regardless of phoneme content; the differential check (FR-018, SC-008) guards against
  a backend that ignores that contract. Timing neutrality (FR-016) is the load-bearing new guarantee,
  since articulation is where time can actually drift; it is enforced relative to the lyric-free
  baseline and gated by rejection.
- "No shift from adding vocal synthesis" is interpreted as a **relative (lyric-on vs lyric-off)**
  guarantee for the same seed, layered on top of the existing absolute score-tolerance checks — not as
  a claim that the open-vowel baseline itself is shift-free (that is 001's responsibility).
