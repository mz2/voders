# Phase 1 Data Model: Synthetic Singing Corpus Generator

Entities derive from the spec's Key Entities and Functional Requirements. Field types are Python /
JSON oriented (the manifest is JSON Lines; the config is YAML). "MUST" rules trace to FR/SC numbers.

## Score

The authoritative ground truth for any sample derived from it.

| Field | Type | Notes |
|-------|------|-------|
| `score_id` | str | Stable identifier (e.g., source filename stem); used in seed derivation (FR-013) |
| `source` | str | Provenance of the score (dataset name / path) |
| `notes` | list[Note] | Ordered note rows |

**Note**: `onset_s: float`, `offset_s: float`, `pitch_midi: int` (FR-001).

**Rules:**
- Monophonic: no two notes overlap in time. Overlapping/polyphonic scores are rejected or split into
  separate monophonic tracks, the choice recorded in provenance (Edge Cases).
- `offset_s > onset_s`; a note shorter than the configured `min_note_ms` is flagged or rejected
  (Edge Cases; `min_note_ms` learned from the annotation distribution + active method timing, FR-018/FR-019).
- Empty score → no sample; single-note score → still validated.

## Voice (Timbre)

A named singer identity used by a renderer lane.

| Field | Type | Notes |
|-------|------|-------|
| `voice_id` | str | Stable identifier; used in seed derivation and `score_NNN_singer_X` naming |
| `kind` | enum | `svs_voicebank` \| `voice_conversion` \| `deterministic_donor` |
| `license` | str | Free-text license string |
| `consent_verified` | bool | **Required.** If false or missing the voice is refused (FR-011) |
| `model_ref` | str | Path/id of the underlying voice bank / VC model / donor sample |

**Rules:**
- Any lane that tries to use a voice with `consent_verified` false/missing MUST refuse and record the
  refusal in the manifest (FR-011, SC-008).
- A voice missing at runtime → affected variants skipped and logged; never silently substituted
  (Edge Cases).

## RunConfig

A single declarative, version-controlled file fully specifying one run (FR-016).

| Field | Type | Notes |
|-------|------|-------|
| `run_id` | str | Stable run identifier |
| `master_seed` | int | Run-level seed; all per-sample/per-stage seeds derive from it (FR-013) |
| `scores` | path/glob | Score set to consume |
| `voices` | list[Voice] | Enrolled voice pool |
| `lanes` | map | Per-lane enable/disable + lane options (FR-015) |
| `augmentation_profiles` | list[AugmentationProfile] | Label-safe transforms to apply |
| `validator` | map | Tolerances: onset_ms=50, offset = max(50ms, 20%), `min_note_ms` (learned from annotation distribution per FR-018; null = learn), f0 cents/coverage, snr_floor_db |
| `reproduction` | map | SC-009 tolerance: bit-exact (deterministic/voice-conversion) vs. same-verdict + f0/onset bounds (neural) |
| `output_root` | path | Where corpus/manifest/checkpoints are written |

**Rules:**
- The resolved config MUST be embedded or hash-referenced by the manifest so a run replays from the
  manifest + source scores (FR-016, SC-009).
- The resolved config is a committable end-result artifact (FR-017).

## AugmentationProfile

An ordered, label-preserving sequence of transforms.

| Field | Type | Notes |
|-------|------|-------|
| `profile_id` | str | Stable identifier; part of a timbre identity (SC-004) |
| `steps` | list[step] | e.g., `pitch_shift`, `time_stretch`, `reverb_ir`, `codec`, `accompaniment_mix` |
| `params` | map | Per-step parameters (ranges sampled via the derived seed) |

**Rules:** Steps MUST NOT alter the paired score's note rows (FR-005). A profile that drops vocal
level below `snr_floor_db` causes quarantine (FR-014); clipping is normalized or the sample rejected
(US3 scenario 3).

## RendererLane (interface, not stored)

One of four orthogonal rendering paths behind a common contract (FR-015): `deterministic`, `svs`,
`voice_conversion`, `augmentation`. Contract in `contracts/lane-interface.md`. Each lane takes a
`(Score, Voice, seed, options)` and returns rendered audio + the score it is labelled by (the
original score, or — for the expressive lane in re-derive mode — a re-derived score, FR-007).

## ValidationVerdict

Per-sample gate result; lives inside the provenance record (FR-006).

| Field | Type | Notes |
|-------|------|-------|
| `status` | enum | `accepted` \| `rejected` \| `quarantined` \| `flagged` \| `license_refused` |
| `onset_ok` | bool | All onsets within 50 ms (SC-001) |
| `offset_ok` | bool | All offsets within max(50 ms, 20%) (SC-002) |
| `f0_ok` | bool | Measured f0 within ±25 cents over ≥80% of sustained interval |
| `snr_db` | float\|null | Vocal-to-accompaniment level (augmented samples) |
| `reason` | str\|null | Human-readable reason when not accepted |
| `max_onset_dev_ms` | float | Largest onset deviation (for SC-010 on the expressive lane) |

**Lifecycle:** rendered → validated → one terminal status. Accepted samples go to `corpus/`; all
non-accepted statuses retain provenance **and** audio under `rejected/` (FR-006a), never trained on.

## MethodTimingBudget (per timing-affecting method)

Two documented numbers per timing-affecting method (f0 estimator, onset detector, forced aligner,
codec/resampler). Values come from each method's own documentation, or a one-time measurement on a
click-train fixture — not invented (FR-019).

| Field | Type | Notes |
|-------|------|-------|
| `method` | str | e.g., `crepe_f0`, `pyin_f0`, `mfa_align`, `mp3_codec` |
| `frame_hop_ms` | float | The method's documented analysis hop (e.g., CREPE = 10 ms) |
| `group_delay_ms` | float | A codec/resampler's known constant latency, from its spec |

**Rules:** `min_note_ms` (FR-018) is floored by the largest `frame_hop_ms` among active methods; the
validator subtracts each method's `group_delay_ms` from measured onsets/offsets before comparing to
the score (FR-019). Active budgets are recorded in the run's stats/manifest for audit.

## ProvenanceRecord (one JSON Lines row)

The replayable description of one produced sample (FR-008).

| Field | Type | Notes |
|-------|------|-------|
| `sample_id` | str | e.g., `score_000123_singer_A` |
| `score_id` | str | Source score |
| `score_path` / `audio_path` | str | Paired output paths (`.tsv` byte-identical to label score) |
| `lane` | str | Renderer lane used |
| `voice_id` | str | Voice/timbre identity |
| `augmentation_profile` | str\|null | Profile applied, if any |
| `seed` | int | Derived per-sample seed |
| `voice_license` | str | Voice license string |
| `consent_verified` | bool | Copied from the voice for audit (SC-008) |
| `verdict` | ValidationVerdict | Embedded gate result |
| `config_hash` | str | Hash of the resolved RunConfig (FR-016) |

**Rules:** Append-only; one row per attempted sample (accepted and non-accepted). License audit
(FR-011/SC-008) and aggregate stats (FR-012) are computed by scanning the file.

## CorpusManifest / StatsReport

- **Manifest**: the JSON Lines file of ProvenanceRecords for the run (FR-008).
- **StatsReport** (`stats.json`, FR-012): total samples, unique scores, unique voices, unique timbre
  identities (voice × augmentation profile, target ≥1,000 / SC-004), pitch distribution, duration
  distribution, augmentation coverage (target ≥60% in-the-mix / SC-006), accept/reject counts
  (SC-007). Produced before export.

## Identity & seed derivation

- **Timbre identity** = (`voice_id`, `augmentation_profile`) — the unit counted for SC-004.
- **Per-sample seed** = deterministic function of (`master_seed`, `score_id`, `voice_id`,
  `stage_name`), so a sample reproduces in isolation regardless of worker count/order (FR-013).
