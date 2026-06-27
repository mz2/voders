# Feature Specification: Synthetic Singing Corpus Generator

**Feature Branch**: `001-synthetic-singing-corpus`
**Created**: 2026-06-27
**Status**: Draft
**Input**: User description: "Specify the system described in the research/deep-research-report.md file"

## Clarifications

### Session 2026-06-27

- Q: What is the target total size of a single corpus run? → A: Medium — 10,000–100,000 samples per run, requiring a streaming/checkpointed pipeline and sharded on-disk layout (not an in-memory design). Two DGX Spark machines are available for the accelerated (voice-conversion / neural-SVS) lanes and downstream training.
- Q: How is the manifest/provenance stored? → A: JSON Lines — one provenance record per line, append-only during the run. Queries, license audits, and aggregate statistics are computed by scanning the JSONL log (no separate database service).
- Q: What happens to non-accepted samples (rejected, quarantined, flagged, license-refused)? → A: Full forensic retention — persist both the provenance record (with the rejection/quarantine reason and verdict) and the rendered audio for every non-accepted sample, stored separately from the training set so it is never trained on but remains fully auditable.
- Q: What implementation language/runtime? → A: Python — orchestrator and the ML/audio primitives (renderers, voice conversion, augmentation) share one runtime; GPU lanes run PyTorch. No separate primary language.
- Q: How is randomness seeded for reproducibility? → A: A single run-level master seed deterministically derives every per-sample and per-stage seed from stable identifiers (e.g., score ID + voice ID + stage name). Any individual sample is reproducible in isolation, independent of worker count or processing order.
- Q: How does an operator specify a run? → A: A single declarative config file (e.g., YAML) names the score set, voice pool, augmentation profiles, master seed, and lane toggles for a run. Run config files are version-controlled in the repository, and the manifest embeds (or hash-references) the fully-resolved config so a run is a one-command replay.
- Q: What of a run goes into version history? → A: The run's end-result record — the final manifest (JSONL) plus the resolved run config and the stats report — is committed to version history. Intermediate streaming checkpoints (the resume state) are NOT committed; they are scratch and git-ignored. The rendered audio corpus itself stays out of Git per the constitution's "generated corpora are never committed" rule and the local-disk / object-storage assumption.
- Q: How is a voice's license/consent represented for the FR-011 refusal and SC-008 audit? → A: Two-field model per voice — a free-text license string plus a required boolean `consent_verified` flag. The system refuses to use any voice whose `consent_verified` is false (or missing) and surfaces the refusal in the manifest; the audit checks the flag, not a parsed string.
- Q: What is the minimum note-duration cutoff for flag/reject? → A: It is not a fixed constant. The validator applies a configurable `min_note_ms` threshold whose default is calibrated from the input score annotation statistics (e.g., the shortest reliably-annotated note durations in the source dataset), rather than a hardcoded value. The earlier "~40 ms" and "50 ms × 2" mentions are illustrative, not normative.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Deterministic baseline corpus from scores (Priority: P1)

A data engineer takes a folder of singing scores — each score is a list of `(onset, offset, MIDI pitch)` note tuples — and produces a paired training corpus of synthetic vocal audio whose note boundaries match the score with mathematical certainty. The engineer can hand this corpus straight to the Klangio singing-transcription pipeline and trust that no label is wrong.

**Why this priority**: This is the minimum viable corpus. Without exact alignment between rendered audio and score labels, every later step is built on sand. The research report is explicit that *alignment drift, not audio quality, is the dominant risk*. A renderer whose pitch and timing are derived directly from the score eliminates that risk by construction and yields a known-good baseline before any naturalness or diversity work is attempted.

**Independent Test**: Point the system at a small set of scores (e.g., 10 scores from the public Klangio dataset), run the deterministic pipeline, and confirm that every rendered note's onset and offset fall inside the challenge's tolerance window (50 ms onset; max(50 ms, 20% of note length) offset). The resulting `(audio.wav, score.tsv)` pairs are valid training inputs even if nothing else in the system is implemented.

**Acceptance Scenarios**:

1. **Given** a `.tsv` score with 50 notes, **When** the deterministic pipeline renders it, **Then** the output contains a 22,050 Hz mono audio file paired with a `.tsv` whose note rows are byte-identical to the input.
2. **Given** the rendered audio for that score, **When** automated f0 verification compares the rendered pitch contour to the score, **Then** every note's measured f0 lies within ±25 cents of the score pitch for at least 80% of the note's sustained interval.
3. **Given** a score containing a note shorter than the configured `min_note_ms` threshold (calibrated from the input annotation statistics), **When** the pipeline renders it, **Then** the sample is either accepted with a flag or rejected with a clear reason logged to provenance.

---

### User Story 2 - Timbre multiplication through voice conversion (Priority: P2)

The same engineer wants 20 different "singers" performing the same score so the downstream transcription model learns to ignore timbre and focus on pitch. They configure a pool of voice-conversion voices and ask the system to fan every rendered take out across that pool, keeping the original alignment intact.

**Why this priority**: The published evidence (Sato & Akama 2024) shows that timbre diversity is the highest-ROI lever for vocal-like targets, beating per-sample realism. Multiplying timbres after deterministic rendering means alignment is preserved for free — the score-aligned f0 propagates through voice conversion unchanged when `auto_predict_f0` is disabled. This story unlocks the corpus-size scaling that the research recommends spending the budget on.

**Independent Test**: Take one accepted P1 sample, run it through N voice-conversion voices, and verify that the N output audio files share the same paired score file, that each output's f0 still tracks the score within tolerance, and that listener-level inspection confirms timbre actually differs across the N outputs.

**Acceptance Scenarios**:

1. **Given** an accepted P1 sample and a configured pool of 20 voice-conversion voices, **When** the timbre-fanout stage runs, **Then** 20 `score_NNN_singer_X` audio files are produced and all share the same source score row-for-row.
2. **Given** any singer variant from that fanout, **When** automated f0 verification runs against the shared score, **Then** the variant passes the same tolerance check that the P1 source passed.
3. **Given** a voice-conversion voice that lacks an upstream license entry, **When** the fanout stage encounters it, **Then** that voice is skipped and the donor-voice manifest records the omission.

---

### User Story 3 - Production-chain domain randomization (Priority: P3)

The engineer wants the synthetic corpus to look like real, in-the-mix pop singing — not dry studio takes — so the downstream model generalizes to the Klangio hidden test set. They turn on augmentation stages that mix vocals with backing tracks, convolve with room impulse responses, and apply MP3/Opus codec round-trips, all while leaving the score labels untouched.

**Why this priority**: Basic Pitch was trained in-the-mix on MAESTRO/Slakh/iKala-style data, so in-the-mix synthetic data closes the domain gap. The research report explicitly identifies this as the stage *"most likely to beat the real-data baseline"*. This story is the dial that converts raw synthesis into competitive training data — but it depends on P1 and P2 being in place first because broken alignment cannot be patched with augmentation.

**Independent Test**: Take a fully-rendered, timbre-fanned sample, apply the augmentation chain at varied settings (reverb on/off, codec on/off, accompaniment SNR sweep), and verify that the score-aligned labels still pass f0/onset verification on the augmented audio.

**Acceptance Scenarios**:

1. **Given** an accepted sample and an augmentation profile with reverb + codec + accompaniment, **When** the augmentation chain runs, **Then** the output audio carries the unchanged score and still passes onset/offset tolerance checks.
2. **Given** an augmentation chain that includes mixing in a real backing track, **When** the mix is rendered, **Then** the vocal remains the dominant harmonic source verifiable by a vocal-to-accompaniment level check.
3. **Given** an augmentation profile that produces clipping, **When** the chain runs, **Then** the sample is normalized or rejected — never silently distorted into the corpus.

---

### User Story 4 - Naturalistic neural SVS with label safety net (Priority: P4)

The engineer wants to fold in expressive neural-SVS samples (DiffSinger / NNSVS-class) to cover edge cases like vibrato, portamento, and pre-onset consonants — but only if the system can guarantee that the expressive timing does not silently corrupt the labels.

**Why this priority**: Expressive synthesis improves audio realism but actively threatens alignment because timing humanization can violate the 50 ms onset tolerance. The research report calls this out as the most dangerous failure mode. Including it as P4 means the system must already have a working baseline (P1) and a working validator (cross-cutting) before this lane is enabled, so any drift becomes a measurable, recoverable problem.

**Independent Test**: Render the same score through a neural SVS toolkit, run the label re-derivation pass (forced alignment + onset detection on the actually-rendered audio), and verify that either the re-derived labels stay within tolerance of the score or the sample is rejected with the deviation logged.

**Acceptance Scenarios**:

1. **Given** a neural SVS render of a score, **When** the label re-derivation stage runs, **Then** the output sample carries the *re-derived* labels rather than the original score, and the deviation between the two is recorded.
2. **Given** a neural SVS render whose re-derived onsets deviate by more than 50 ms on any note, **When** the validator runs, **Then** the sample is rejected from the corpus and the operator sees a flagged provenance entry.
3. **Given** the operator enables a "force score F0" mode on the neural SVS pipeline, **When** rendering proceeds, **Then** the system uses the score-derived F0 contour and skips label re-derivation, behaving like P1 with better timbre.

---

### User Story 5 - Provenance and license audit (Priority: P5)

A reviewer (or the same engineer auditing their own corpus) needs to answer questions like *"which donor voices appear in this corpus?"*, *"which augmentation profile produced this specific sample?"*, and *"are any of these voices unlicensed clones of real people?"* without re-running the pipeline.

**Why this priority**: Without a manifest, the corpus is unauditable and the licensing risk surfaced in the research report (RVC/so-vits-svc consent and SVS corpus licenses) cannot be controlled. This story makes corpus composition transparent and is a prerequisite for any external release of the data.

**Independent Test**: After running the pipeline end-to-end, query the manifest for any single sample's full provenance chain (score source, renderer, voice/timbre, augmentation profile, license tag) and verify the chain reproduces the sample if re-executed with the same seed.

**Acceptance Scenarios**:

1. **Given** any sample in the corpus, **When** the operator queries the manifest, **Then** the manifest returns the renderer, voice ID, augmentation profile, random seed, and donor-voice license tag.
2. **Given** a donor voice flagged "cloned identifiable person without consent", **When** any pipeline stage tries to use it, **Then** the stage refuses and surfaces the refusal in the manifest.
3. **Given** a published corpus manifest, **When** the operator re-runs the pipeline with the same seed and configuration, **Then** the regenerated audio is bit-equivalent (or within a documented tolerance) to the original.

---

### Edge Cases

- **Overlapping or polyphonic notes in a score**: Singing is monophonic; the system rejects scores with simultaneous pitches or splits them into separate monophonic tracks, with the choice surfaced in provenance.
- **Sub-tolerance note duration**: Notes shorter than a configurable `min_note_ms` threshold cannot be reliably bounded by the onset tolerance; the system flags or rejects them rather than silently emitting unlearnable samples. The threshold's default is calibrated from the input score annotation statistics (the shortest reliably-annotated durations in the source dataset), not a hardcoded constant.
- **Legato pairs at the same pitch**: Two adjacent notes at the same pitch still require a perceptible articulation between them; deterministic synthesis inserts an amplitude dip or consonant.
- **Augmentation that destroys the vocal**: If a reverb/codec/accompaniment combination drops the vocal SNR below a configurable floor, the sample is rejected.
- **Empty or near-empty scores**: A score with zero notes produces no sample; a score with one note still goes through validation and may be accepted.
- **Lyrics shorter or longer than the note count**: When lyrics are provided, mismatches default to a neutral open vowel ("ah") rather than dropping notes.
- **F0-verification false negatives** on highly breathy or whispered voices: The validator allows a configurable per-voice tolerance override, surfaced in provenance.
- **Donor voice missing at runtime**: The pipeline skips affected variants and logs them; it does not silently substitute a different voice.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST accept singing scores in a row-oriented format whose columns are (note onset in seconds, note offset in seconds, MIDI pitch as an integer), matching the Klangio challenge input format.
- **FR-002**: The system MUST emit, for each rendered sample, a `(audio, score)` pair where the audio is 22,050 Hz mono floating-point and the paired score is identical to the score used to drive (or re-derived from) the rendering.
- **FR-003**: The deterministic rendering path MUST construct the audio's fundamental frequency contour directly from the input score so that note onset, offset, and pitch are guaranteed by construction rather than by inference.
- **FR-004**: The system MUST allow generating multiple labelled variants per score, addressable as distinct "singer" identities, without re-creating the underlying score.
- **FR-005**: The system MUST provide a label-safe augmentation chain — covering pitch shift, time stretch, reverb, codec round-trips, and mixing with accompaniment — that never alters the paired score's note rows.
- **FR-006**: The system MUST run an automated alignment validator on every rendered sample before it is accepted into the corpus, comparing the audio's measured pitch and note boundaries against the paired score and rejecting samples that fall outside the configured tolerance.
- **FR-006a**: The system MUST persist, for every non-accepted sample (rejected, quarantined, flagged, or license-refused), both its provenance record — including the verdict and the reason — and its rendered audio, in a location separate from the accepted training set so the audio is never trained on but the full attempt history remains auditable.
- **FR-007**: Whenever a non-deterministic (expressive) renderer is used, the system MUST either (a) force the renderer to use the score-derived F0 contour, or (b) re-derive the note onsets/offsets from the actually-rendered audio and update the paired score accordingly.
- **FR-008**: The system MUST record, for every accepted sample, a provenance entry — appended as one JSON object per line to a JSON Lines manifest — capturing the source score, renderer, voice/timbre identity, augmentation profile, random seed, and license tag for any donor voice involved. The manifest MUST support per-sample lookup, license audits, and aggregate statistics by scanning this log.
- **FR-009**: The deterministic rendering and validation paths MUST run on commodity CPU hardware without requiring a GPU, so contributors can iterate locally before scaling on accelerated infrastructure.
- **FR-010**: The system MUST allow the operator to scale the corpus along the timbre axis (more voice-conversion voices, more augmentation profiles) independently of the underlying score set.
- **FR-011**: Every enrolled voice MUST carry a free-text license string and a required boolean `consent_verified` flag. The system MUST refuse to use any voice whose `consent_verified` is false or missing — covering unconsented clones of real identifiable people — and MUST surface that refusal in the provenance manifest.
- **FR-012**: The system MUST report aggregate corpus statistics — total samples, unique scores, unique voices, pitch distribution, duration distribution, augmentation coverage — before the corpus is exported for training.
- **FR-013**: The system MUST be deterministic with respect to its configuration and a single run-level master seed, from which every per-sample and per-stage seed is deterministically derived using stable identifiers (e.g., score ID + voice ID + stage name). Any individual sample MUST be reproducible in isolation, independent of worker count or processing order, so that a published corpus manifest is sufficient to reproduce the corpus.
- **FR-014**: The system MUST quarantine any sample whose vocal level after augmentation falls below the configured signal-to-accompaniment floor, rather than emit it into the training set.
- **FR-015**: The system MUST be modular at the renderer boundary so that the deterministic lane, expressive neural-SVS lane, voice-conversion lane, and augmentation lane can each be enabled, disabled, or replaced independently.
- **FR-016**: A corpus run MUST be specified by a single declarative config file (e.g., YAML) that names the score set, voice pool, augmentation profiles, master seed, and lane toggles. Run config files are intended to be version-controlled in the repository, and the manifest MUST embed or hash-reference the fully-resolved config so the run can be replayed with a single command.
- **FR-017**: The run's end-result record — the final manifest, the resolved run config, and the aggregate stats report — MUST be committable to version history as text artifacts. Intermediate streaming checkpoints (resume state) MUST be git-ignored scratch, and the rendered audio corpus MUST NOT be committed to Git (per the constitution's binary-assets rule); it lives on local disk or operator object storage.

### Key Entities *(include if feature involves data)*

- **Score**: An ordered sequence of `(onset, offset, MIDI pitch)` note tuples representing a melody; the authoritative ground truth for any sample derived from it.
- **Sample**: An `(audio, paired score, provenance)` triple representing one rendered training example.
- **Voice / Timbre**: A named singer identity — either an SVS voice bank, a voice-conversion voice, or a deterministic-vocoder donor — carrying a free-text license string and a required boolean `consent_verified` flag (a voice with `consent_verified` false or missing is refused).
- **Augmentation Profile**: An ordered, label-preserving sequence of transformations (e.g., reverb IR, codec, accompaniment mix, pitch/time perturbation) applied to a rendered sample.
- **Renderer Lane**: One of the four orthogonal rendering paths — deterministic F0-driven, expressive neural SVS, voice conversion (timbre-only), or accompaniment/mix — each with its own input/output contract.
- **Validation Verdict**: A per-sample alignment, level, and license check result that gates corpus admission and lives inside provenance.
- **Run Config**: A single declarative, version-controlled file (e.g., YAML) that fully specifies one corpus run — score set, voice pool, augmentation profiles, master seed, and lane toggles. The manifest embeds or hash-references the resolved Run Config so the run is replayable.
- **Corpus Manifest**: The complete, replayable description of which samples are in the corpus and how each one was produced, stored as a JSON Lines file (one provenance record per line, appended as samples are accepted).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: At least 99% of accepted samples have every rendered note onset within 50 ms of the paired score onset, measured by the automated validator.
- **SC-002**: At least 99% of accepted samples have every rendered note offset within the greater of 50 ms or 20% of the note length, measured by the automated validator.
- **SC-003**: A transcription model trained from random initialization on the produced corpus achieves a note-level F1 (correct-pitch + correct-onset, within the challenge's tolerance) of at least 0.15 on the held-out evaluation slice — clearing the published floor.
- **SC-004**: The corpus contains at least 1,000 distinct timbre identities (combinations of voice and augmentation profile) without requiring per-singer manual configuration beyond enrolling the voice once.
- **SC-005**: The deterministic rendering and validation lane sustains at least 100 score-singer pairs per hour on a laptop-class CPU (4 cores, no GPU).
- **SC-006**: At least 60% of accepted samples include at least one production-style augmentation stage (reverb, codec, or accompaniment mix) so the corpus is dominated by in-the-mix examples rather than dry studio renders.
- **SC-007**: At least 95% of synthesized samples pass the automated alignment validator on the first attempt; the remaining 5% are flagged or rejected, never silently admitted.
- **SC-008**: Zero samples in any exported corpus carry a donor voice whose `consent_verified` flag is false or missing, verified by an end-of-run audit of the manifest.
- **SC-009**: Any published corpus can be regenerated end-to-end from its manifest plus the source scores within a documented bit/sample tolerance, demonstrating reproducibility.
- **SC-010**: For samples produced by the expressive-renderer lane, the re-derived note onsets deviate from the original score by less than 50 ms in at least 90% of accepted notes; samples failing this bound are rejected, not relabelled silently.
- **SC-011**: A single corpus run produces between 10,000 and 100,000 samples without exhausting commodity-machine memory, by streaming and checkpointing progress so an interrupted run resumes from the last completed shard rather than restarting.

## Assumptions

- The downstream consumer of the corpus is a Basic Pitch-class note-level transcription model — i.e., a model that ingests 22,050 Hz mono audio and outputs onset/note/contour posteriorgrams over a fixed pitch grid. This fixes the audio format and the relevant tolerance metrics.
- The corpus' training-time use is owned by the consumer (Klangio's submission pipeline). The system's responsibility ends at producing the `(audio, score)` pairs and the manifest; it does *not* train the transcription model itself.
- Scores are supplied externally (e.g., the Klangio public dataset, mined from MIR-ST500-class corpora, or generated from MIDI/MusicXML). Score acquisition is out of scope; the system consumes whatever score set the operator provides.
- The operator has, or will obtain, rights to every donor voice, SVS voice bank, voice-conversion model, real backing track, and impulse response they enroll. The system enforces the *manifest*, but cannot verify upstream licenses.
- Real backing tracks for mixing, when used, are independently licensed by the operator; the system supports — but does not require — generative accompaniment as an alternative.
- "Singer X" variants for a single score are an acceptable training-data format for the downstream consumer, allowing aggressive timbre multiplication.
- Corpus storage is local-disk or operator-provided object storage; managed cloud storage is out of scope for the first version.
- The system orchestrates existing open-source synthesizers, vocoders, voice-conversion models, and augmentation libraries; it does not re-implement those primitives.
- Lyrics are optional. When omitted, the deterministic and neural-SVS lanes default to a neutral open vowel; when provided, the operator is responsible for syllable-count alignment to notes (the system surfaces mismatches but does not autocorrect them beyond the documented vowel fallback).
- Per-sample reproducibility uses a documented numerical tolerance because some renderers and vocoders are not strictly bit-deterministic across hardware; "regeneration" means audibly equivalent and within validator tolerances.
- The system is implemented in Python; the orchestrator and the rendering/voice-conversion/augmentation primitives share one Python runtime, and the GPU-accelerated lanes run on PyTorch. This fixes the linter/formatter/test-framework choices at planning time.
