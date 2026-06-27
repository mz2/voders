# Feature Specification: Vocal-Conditioned Accompaniment Lane

**Feature Branch**: `002-vocal-conditioned-accompaniment`
**Created**: 2026-06-27
**Status**: Draft
**Input**: User description: "Open-Weight Models for Vocal-Conditioned Accompaniment — Updated Report (June 2026). Relaxed requirement: the singing's *offsets must stay put*, but the vocal need not be bit-exact (mild re-encoding coloration is acceptable). Add instrumental accompaniment beneath an existing sung vocal without disturbing the labeled note timing, using open-weight models."

## Overview

The existing pipeline produces `(vocal audio, score)` pairs whose note labels — `(onset, offset, pitch)`
rows — are *correct by construction*. This feature adds a new, independently toggleable **accompaniment
lane** that places instrumental backing *underneath* such a vocal so the corpus also contains realistic
"voice-in-a-mix" material, **without invalidating the existing labels**.

The load-bearing constraint is timing. "Onset" is when a sung note begins; "offset" is when it ends. The
sung notes' onsets and offsets must stay within the project's existing alignment tolerance after
accompaniment is added, so the same score still describes the mix. The vocal itself need *not* be
bit-for-bit identical — mild coloration from re-encoding (a slightly "old-tape"/processed timbre) is
acceptable — as long as the note *timing* does not move.

## Clarifications

### Session 2026-06-27

- Q: How should the lane select among generated accompaniment takes (some come out irrelevant/incoherent)? → A: No separate relevance/coherence gate — the alignment validator is the only quality gate; among validator-passing takes, the best-aligned one is admitted.
- Q: How is the grid-imposition concern operationalized (FR-012 / SC-005)? → A: By direct measurement, not a perceptual beat classifier — the generated audio must not have shifted any original note in time; detected note onset/offset positions in the mix are compared to the original labels and must not have moved.
- Q: Per source vocal, how many accompaniment-augmented samples enter the corpus? → A: One best take per (source vocal × target instrument/mode).
- Q: Which vocals does the lane accept as input in v1? → A: Corpus-internal vocals only (from existing lanes); external user-supplied vocals are out of scope for v1.
- Q: What audio does the alignment validator run on for admission? → A: The final vocal+accompaniment mix — sung notes must remain detectable through the accompaniment; a masked/undetectable vocal is rejected (so the validator must track the voice within a mix, not only a solo vocal).
- Q: What does the corpus store per accompaniment-augmented sample? → A: The final mix plus the separately generated accompaniment stem (and, in vocal-preserving mode, a reference to the unchanged source vocal), enabling label-safe re-mixing.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Layer accompaniment under a vocal without moving note timing (Priority: P1)

A corpus operator takes an existing vocal sample and its score, and produces a mixed sample with an added
instrumental layer in which the vocal track is left **completely untouched** — a generated instrument
"stem" (a single isolated instrument track) is summed beneath the original vocal. Because the vocal
waveform is never re-encoded, every labeled onset and offset is preserved *by construction*. The mixed
sample is then run through the existing alignment validator and admitted to the corpus only if it passes.

**Why this priority**: This is the safest, highest-confidence path and the MVP. It guarantees the labels
remain valid (timing preserved by construction) and keeps the original voice perfectly intact, which is
exactly the relaxed requirement's strongest form. It can ship and deliver value before any other mode.

**Independent Test**: Feed a fixture vocal + score, generate one instrument stem, sum it under the
original vocal, run the alignment validator, and confirm (a) onsets within 50 ms and offsets within
max(50 ms, 20% of note length) of the original labels, and (b) the vocal channel content is byte-identical
to the input vocal.

**Acceptance Scenarios**:

1. **Given** a vocal sample with a valid score, **When** the lane generates an instrument stem and sums it
   beneath the unchanged vocal, **Then** the admitted mix's note onsets/offsets match the original labels
   within tolerance and the isolated vocal content is bit-identical to the input.
2. **Given** several generated takes for one vocal, **When** the lane evaluates them against the alignment
   validator, **Then** it admits exactly one best-aligned passing take per (vocal × target/mode) and discards
   the rest — with no separate relevance scoring — recording the number of takes tried.
3. **Given** the lane is disabled for a run, **When** the run executes, **Then** no accompaniment samples are
   produced and all other lanes behave unchanged.

---

### User Story 2 - One-pass full mix with locked onsets (Priority: P2)

A corpus operator wants the generator to also place rhythmic and structural elements (not only sustained
pads), so they use a one-pass mode that conditions on the vocal and emits a single combined vocal +
instrument mix. The vocal is re-encoded as part of generation (mild coloration accepted), but its labeled
note onsets and offsets stay within the alignment tolerance, so the original score still describes the mix.

**Why this priority**: Adds corpus diversity (full arrangements, rhythmic content) with a simpler one-pass
workflow, at the cost of slight vocal coloration. It is valuable but strictly secondary to the
vocal-preserving path because it accepts a fidelity trade-off and a higher gate-failure rate.

**Independent Test**: Run one-pass generation on a fixture vocal, validate the resulting mix's note timing
against the original labels, and confirm onsets/offsets remain within tolerance (labels still valid) even
though the vocal channel is not bit-identical.

**Acceptance Scenarios**:

1. **Given** a vocal sample and a style description, **When** one-pass generation runs, **Then** the emitted
   mix's note onsets/offsets remain within tolerance of the original labels and the sample is admitted.
2. **Given** a one-pass result whose re-encoding pushes an offset outside tolerance, **When** it is
   validated, **Then** it is routed to the rejected tree with a recorded reason and not admitted.

---

### User Story 3 - Free-time / rubato-safe accompaniment (Priority: P3)

A corpus operator generating accompaniment for free-time, rubato material (singing whose phrasing follows no
fixed beat — e.g., voice over handpan/gong/bowl textures) configures the lane to avoid imposing a metrical
pulse. The dominant real-world failure for this material is *grid imposition*: the generator implies a steady
beat that fights the singer's phrasing, so the *perceived* timing shifts even though the vocal samples never
moved. The operator selects sustained/textural target instruments, free-time descriptors, and no fixed tempo
("BPM" = beats per minute) to mitigate this.

**Why this priority**: Critical for the project's characteristic material but a refinement of the core modes
(P1/P2) rather than a standalone capability. It protects *perceived* timing quality on top of the
sample-timing guarantee the validator already enforces.

**Independent Test**: Generate accompaniment for a rubato fixture under free-time configuration; confirm no
fixed-tempo metadata is required or recorded, and that the alignment validator detects zero timing shift of
every original note (measured onset/offset in the mix versus the original labels).

**Acceptance Scenarios**:

1. **Given** a rubato vocal and a free-time configuration with no BPM, **When** accompaniment is generated,
   **Then** the admitted sample carries no fixed-tempo metadata and the measured timing shift of every original
   note is zero within the validator's tolerance.
2. **Given** a take whose accompaniment displaced any original note in time beyond tolerance, **When** it is
   validated, **Then** it is rejected with the per-note shift recorded and never admitted.

---

### User Story 4 - License and provenance audit for generated accompaniment (Priority: P4)

A corpus maintainer needs every accompaniment-augmented sample to be auditable and **commercially
redistributable**. For each such sample, the manifest records the model/weights identity and version, its
license (plus any required attribution text), the conditioning mode used, the source vocal sample id,
generation seed(s), and whether the vocal track is bit-exact. The license audit refuses non-commercial
(CC-BY-NC) or closed weights, and for attribution-required (CC-BY-class) weights it ensures the attribution
text is captured and propagated into the corpus's redistributable provenance.

**Why this priority**: Required for a trustworthy, license-clean corpus, but it layers onto whichever
generation modes exist. The modes can be demonstrated before the audit tooling is complete, so it is lowest
priority while still mandatory for release.

**Independent Test**: Generate a batch of accompaniment samples, run the audit, and confirm every record has
complete model/license/mode/source/seed provenance and that any disallowed-license weights are flagged and
their outputs not admitted.

**Acceptance Scenarios**:

1. **Given** a batch of admitted accompaniment samples, **When** the audit runs, **Then** 100% have complete
   provenance and license fields recorded in the manifest.
2. **Given** weights with a non-commercial (CC-BY-NC) or closed license, **When** the lane attempts to use
   them, **Then** their outputs are refused/flagged and excluded from the admitted corpus.
3. **Given** samples produced with an attribution-required (CC-BY-class) model, **When** the audit runs,
   **Then** each carries the required attribution text in the corpus's redistributable provenance.

---

### Edge Cases

- **Grid imposition with preserved samples**: accompaniment implies a beat that fights the vocal's phrasing,
  shifting *perceived* timing although the labeled samples are unmoved. Mitigated via Story 3 configuration;
  the residual risk is recorded per sample, not silently ignored.
- **Irrelevant / incoherent stem**: takes are gated only by the alignment validator, not by any relevance
  score, so an accompaniment that is musically unrelated but does not move the notes can be admitted.
  Mitigation is limited to generating multiple takes and admitting the best-aligned; musical relevance is not
  a v1 admission criterion (accepted limitation).
- **One-pass offset drift**: re-encoding nudges an onset/offset just outside tolerance — the sample is
  rejected with a recorded reason, never admitted with stale labels.
- **Format mismatch**: the source vocal's sample rate/channel layout differs from what the model expects.
  The vocal is resampled/reformatted *for conditioning only*; the original retained vocal used in the
  vocal-preserving mix is unchanged.
- **No eligible model**: no available open-weight model satisfies the configured license policy or required
  mode. The lane no-ops gracefully and the run continues with the remaining lanes, logging the skip.
- **Out-of-range duration**: vocals longer or shorter than the model can condition on in a single pass are
  handled (segmented or skipped with a recorded reason) rather than producing mistimed output.
- **Vocal masked by accompaniment**: the accompaniment is loud or dense enough that the sung notes are no
  longer detectable in the final mix. Because admission validates the mix, the sample fails note detection and
  is rejected with a recorded reason rather than admitted with unusable labels.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST accept a vocal audio sample together with its note score (onset/offset/pitch
  rows) as the conditioning input for accompaniment generation.
- **FR-002**: The system MUST provide a **vocal-preserving** mode that generates an instrument stem and sums
  it beneath the original vocal waveform unchanged, so every labeled onset/offset is preserved by
  construction and the isolated vocal content is bit-identical to the input.
- **FR-003**: The system MUST provide a **one-pass full-mix** mode that conditions on the vocal and emits a
  combined vocal+instrument mix whose labeled note onsets/offsets remain within the project's existing
  alignment tolerance (onset ≤ 50 ms; offset ≤ max(50 ms, 20% of note length)), while permitting mild vocal
  coloration.
- **FR-004**: The system MUST validate every accompaniment-augmented sample **on the final
  vocal+accompaniment mix** and admit only samples whose sung-note onsets/offsets remain within tolerance AND
  whose notes remain detectable through the accompaniment; a sample whose vocal is masked/undetectable in the
  mix MUST be rejected. Failing samples MUST be routed to the rejected tree with a recorded reason.
  (Validating the mix requires the validator to track the sung voice within accompaniment, not only a solo
  vocal.)
- **FR-005**: The system MUST support a free-time/rubato configuration that requires no fixed tempo (no BPM)
  and biases generation toward sustained/textural accompaniment, and MUST NOT require tempo metadata for such
  samples.
- **FR-006**: The system MUST use only open-weight models whose license permits commercial use with at most an
  attribution obligation (MIT, Apache-2.0, CC-BY-class) and MUST refuse or flag weights with non-commercial
  (e.g., CC-BY-NC) or closed terms. The accepted-license set MUST be configurable.
- **FR-006a**: For any admitted sample produced with an attribution-required (CC-BY-class) model, the system
  MUST capture the required attribution text and propagate it into the corpus's redistributable provenance so
  the corpus can satisfy the attribution obligation.
- **FR-007**: For every accompaniment-augmented sample, the system MUST record provenance in the manifest:
  source vocal sample id, model/weights identity and version, license (and any required attribution text),
  conditioning mode (vocal-preserving / one-pass), whether the vocal track is bit-exact, generation seed(s),
  and the free-time/target-instrument configuration used.
- **FR-008**: The accompaniment capability MUST be an independently toggleable lane that can be enabled or
  disabled per run without affecting other lanes, consistent with the existing lane architecture.
- **FR-009**: Per-sample generation MUST be reproducible from the run's master seed: the same conditioning and
  seed MUST yield the same admission verdict and timing (bit-exact where the lane is deterministic, otherwise
  the same validator verdict with f0/onset within the project's tolerances).
- **FR-010**: The alignment validator is the only admission gate; the system MUST NOT apply a separate
  relevance/coherence score to takes. The lane MAY generate multiple takes per sample; among the takes that
  pass the validator it MUST admit exactly one — the best-aligned (smallest detected note-timing shift) — per
  (source vocal × target instrument/mode), discard the rest, and record the number of takes tried and the
  reason when every take fails.
- **FR-011**: The system MUST resample/reformat the vocal as needed for model conditioning without altering
  the retained original vocal used in the vocal-preserving mix.
- **FR-012**: The system MUST verify that the generated audio has not shifted any original note in time: for
  each sample it MUST compare the note onset/offset positions detected in the generated mix against the
  original labels, record the per-note temporal shift, and reject any sample in which a note has shifted
  beyond the validator's measurement tolerance. (This is the operative form of the "grid-imposition" concern —
  measured note-timing displacement, not a perceptual beat/tempo classifier.)
- **FR-013**: When no available model satisfies the license policy or the requested mode, the lane MUST no-op
  gracefully and the run MUST continue with remaining lanes, logging the skip.
- **FR-014**: Accompaniment generation MUST stream and checkpoint within the project's existing run envelope
  so enabling the lane does not break large-run (10k–100k sample) execution.
- **FR-015**: For each admitted sample the corpus MUST store the final mix **and** the separately generated
  accompaniment stem; in vocal-preserving mode it MUST also retain a reference to the unchanged source vocal.
  These retained stems MUST be sufficient to re-mix the sample at a different vocal/accompaniment balance
  without regenerating.

### Key Entities *(include if feature involves data)*

- **Vocal Sample (input)**: the source voice audio plus its note score and sample id; the conditioning subject
  and, in vocal-preserving mode, the retained untouched track.
- **Accompaniment Model**: an open-weight generator identity — version, license, the conditioning modes it
  supports (vocal-preserving stem / one-pass mix), and its input-format requirements.
- **Accompaniment-Augmented Sample**: the produced sample with conditioning mode, vocal-bit-exact flag,
  seed(s), validation verdict (measured on the mix), per-note timing shift (detected vs original labels),
  takes-tried count, free-time/instrument configuration, and a link back to the source vocal sample. Stored
  artifacts are the final mix **and** the separate accompaniment stem (plus the retained source vocal in
  vocal-preserving mode).
- **Generation Config**: mode selection, target instrument class/descriptors, free-time flag, license policy,
  takes-per-sample, and seed-derivation settings for a run.
- **Validation Verdict**: per-sample onset/offset deviations from the original labels, pass/fail, and any
  rejection reason.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In vocal-preserving mode, 100% of admitted samples have an isolated vocal track bit-identical to
  the input (offsets unmoved by construction).
- **SC-002**: In one-pass mode, ≥ 99% of admitted samples have note onsets within 50 ms and offsets within
  max(50 ms, 20% of note length) of the original labels (labels remain valid on the mix).
- **SC-003**: At least 95% of vocal-preserving takes and at least 80% of one-pass takes pass the alignment gate
  on first attempt.
- **SC-004**: 100% of admitted accompaniment-augmented samples carry complete provenance (model, version,
  license, mode, source vocal id, seed, bit-exact flag) in the manifest; 0 admitted samples use
  non-commercial or closed-license weights; and 100% of samples from attribution-required (CC-BY-class)
  models carry the required attribution text in redistributable provenance.
- **SC-005**: 100% of admitted samples show no original note shifted in time — detected onset/offset in the
  generated mix versus the original label shows zero displacement within the validator's measurement
  tolerance — with the per-note shift recorded; and 100% of free-time-configured admitted samples additionally
  carry no fixed-tempo metadata.
- **SC-006**: Any admitted accompaniment sample is reproducible in isolation from its recorded seed within the
  project's reproducibility tolerance.
- **SC-007**: With the lane enabled, a run still completes a 10k–100k sample corpus by streaming/checkpointing
  without exhausting memory (no regression to the existing run envelope).
- **SC-008**: 100% of admitted samples store both the final mix and the accompaniment stem (plus a
  source-vocal reference in vocal-preserving mode), so any admitted sample can be re-mixed at a different
  vocal/accompaniment balance without regeneration.

## Assumptions

- This is a **new optional lane within the existing synthetic-singing-corpus pipeline**, matching its
  toggleable-lane architecture — not a standalone product.
- The project's **existing alignment validator and tolerances** (onset ≤ 50 ms; offset ≤ max(50 ms, 20% of
  note length)) are the admission gate; "offsets stay put" is judged against the original score labels using
  that validator.
- **License policy: commercial use is intended; attribution-required licenses are accepted.** The accepted set
  is MIT, Apache-2.0, and CC-BY-class (commercial use OK, attribution obligation OK); non-commercial
  (CC-BY-NC) and closed-weight models are excluded. *(Confirmed by the requester: CC-BY is acceptable.)* This
  admits attribution-licensed models (e.g., CC-BY 4.0 weights) provided their attribution text is captured and
  propagated (FR-006a); it excludes CC-BY-NC research-only models. The accepted-license set remains
  configurable.
- **Vocal coloration in one-pass mode is acceptable** as long as the alignment gate passes; no separate
  vocal-fidelity metric is gated. Bit-exact vocal is required only of the vocal-preserving mode.
- Vocal samples fed to this lane come **only from the existing corpus lanes** (deterministic / neural-SVS /
  voice-conversion), whose labels are correct-by-construction. **External / user-supplied real vocals are out
  of scope for v1** — their scores are not correct-by-construction and would need a separate validation/trust
  path.
- A **GPU is available** for generation; this lane is an optional GPU extra and does not affect the CPU
  baseline (deterministic lane + validator) which must keep running without it.
- Specific model selection (open-weight survey, e.g. the June 2026 vocal-conditioned-accompaniment report) is
  an **implementation/planning concern**, resolved in the plan/research phase, not in this spec.

## Dependencies

- Requires the existing **alignment validator**, **JSON Lines manifest writer**, **master-seed derivation**,
  and **corpus sharding (accepted vs rejected trees)** from the `001-synthetic-singing-corpus` pipeline.
- The alignment validator must operate on **voice-in-a-mix** (detect the sung notes through accompaniment),
  which may require a mix-robust f0 tracker or source separation; the existing solo-vocal validator is
  extended/configured for this rather than assumed sufficient as-is.
- Requires at least one **open-weight, vocal-conditioned music-generation model** that supports a
  vocal-preserving stem mode and/or a one-pass full-mix mode (concrete selection deferred to planning).
