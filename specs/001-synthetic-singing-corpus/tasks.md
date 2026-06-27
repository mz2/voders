---
description: "Task list for Synthetic Singing Corpus Generator"
---

# Tasks: Synthetic Singing Corpus Generator

**Input**: Design documents from `/specs/001-synthetic-singing-corpus/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: REQUIRED. The project constitution (Principle I — TDD; Principle IV — eval-first) overrides
the template's "tests are optional" note. Every user story gets test tasks AND at least one
evaluation-harness task, ordered before that story's implementation tasks.

**Organization**: Tasks are grouped by user story (P1–P5) so each story is an independently testable
increment. Stories build on US1 (the deterministic baseline) but each remains independently verifiable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story the task belongs to (US1–US5)
- Exact file paths are included in every task

## Path Conventions

Single Python project: package at `src/voders/`, tests at `tests/`, eval harness/fixtures at `evals/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, toolchain, fixtures.

- [X] T001 Create the project tree per plan.md: `src/voders/{config,scores,render,validate,manifest,corpus,voices,cli}/__init__.py`, `tests/{contract,integration,unit}/`, `evals/fixtures/`
- [X] T002 Create `pyproject.toml` targeting Python 3.14 with `[cpu]` and `[gpu]` optional-dependency extras and pinned major versions from plan.md (numpy>=2.5, scipy>=1.18, soundfile>=0.14, librosa>=0.11, pyworld>=0.3.5, pretty_midi>=0.2.11, music21>=10.5, pyyaml>=6.0.3, pydantic>=2.13, torchcrepe>=0.0.24; gpu: torch>=2.12, torchaudio; dev: ruff>=0.15, mypy>=2.1, pytest>=9.1)
- [X] T003 [P] Configure `ruff` (lint + format) and `mypy` and `pytest` in `pyproject.toml` for zero-warning lint (Constitution Principle II)
- [X] T004 [P] Create LFS-tracked fixtures: ~10 tiny `.tsv` scores in `evals/fixtures/scores/`, one synthetic consented donor vowel in `evals/fixtures/voices/donor_ah_synth.wav`, and `evals/fixtures/smoke.yaml` run config (deterministic lane only)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Cross-cutting primitives every story needs. ⚠️ No user-story work begins until this phase is complete.

- [X] T005 Implement Run Config models + loader/validator/resolver + `config_hash` in `src/voders/config/` (pydantic; contract: `contracts/run-config.schema.yaml`; FR-016)
- [X] T006 [P] Implement `Score` and `Note` models in `src/voders/scores/models.py` (data-model.md; FR-001)
- [X] T007 [P] Implement master-seed → per-sample/per-stage seed derivation in `src/voders/seeds.py` (FR-013)
- [X] T008 [P] Implement `ProvenanceRecord` + `ValidationVerdict` models in `src/voders/manifest/models.py` (contract: `contracts/manifest-record.schema.json`; FR-008)
- [X] T009 Implement JSON Lines manifest writer/reader (append-only, scan queries) in `src/voders/manifest/io.py` (depends on T008; FR-008)
- [X] T010 [P] Implement `Voice` model + enrollment with `consent_verified` gate in `src/voders/voices/registry.py` (FR-011)
- [X] T011 [P] Define `RendererLane` Protocol + `RenderRequest`/`RenderResult` in `src/voders/render/base.py` (contract: `contracts/lane-interface.md`; FR-015)
- [X] T012 Implement sharded corpus layout (accepted `corpus/`, `rejected/`, git-ignored `checkpoints/`) with checkpoint/resume in `src/voders/corpus/store.py` (research.md Decision 7; FR-006a, SC-011)
- [X] T013 Implement CLI skeleton with `run`/`eval`/`audit`/`stats` subcommands wired to stubs in `src/voders/cli/__main__.py` (contract: `contracts/cli.md`)

**Checkpoint**: Foundation ready — user-story phases can begin.

---

## Phase 3: User Story 1 - Deterministic baseline corpus from scores (Priority: P1) 🎯 MVP

**Goal**: Render scores to score-aligned vocal audio whose `(audio.wav, score.tsv)` labels are correct by construction, gated by the alignment validator.

**Independent Test**: Run the deterministic pipeline on the ~10 fixture scores; every rendered note's onset/offset falls inside tolerance and the paired `.tsv` is byte-identical to the input.

### Tests for User Story 1 ⚠️ (write first, must FAIL before implementation)

- [X] T014 [P] [US1] Contract test: deterministic lane `RenderResult` invariants (`label_score == input`, `requires_gpu()==False`, 22,050 Hz mono float32) in `tests/contract/test_deterministic_lane.py`
- [X] T015 [P] [US1] Contract test: validator emits `ValidationVerdict` with onset/offset/f0 checks in `tests/contract/test_validator.py`
- [X] T016 [P] [US1] Integration test for US1 acceptance scenarios (50-note score → mono 22,050 Hz wav + byte-identical tsv; f0 within ±25 cents over ≥80%; sub-`min_note_ms` note flagged/rejected) in `tests/integration/test_us1_deterministic.py`

### Evaluation harness for User Story 1 ⚠️ (before implementation)

- [X] T017 [US1] Create eval harness `evals/run_eval.py` computing SC-001 (onset ≥99%), SC-002 (offset ≥99%), SC-007 (first-attempt pass ≥95%) over a produced manifest, with non-zero exit on failure (Constitution Principle IV)

### Implementation for User Story 1

- [X] T018 [P] [US1] Implement `.tsv` score parser + monophony check in `src/voders/scores/parse.py`, and a learned-threshold analysis pass in `src/voders/scores/analyze.py` that derives `min_note_ms` = max(low percentile (default 1st) of the input note-duration distribution, the active methods' temporal-resolution floor, 50 ms) and records the learned value + contributing method floors to stats/manifest (FR-001, FR-018, FR-019)
- [X] T019 [US1] Implement deterministic WORLD f0-driven renderer (score → step/glide f0 contour → resynthesis) in `src/voders/render/deterministic.py`, no GPU import at module load (FR-003, FR-009)
- [X] T020 [US1] Implement alignment validator (f0 via `pyin` CPU fallback; onset/offset vs score → verdict) in `src/voders/validate/validator.py`, with a method-timing-budget registry (`src/voders/validate/timing.py`) holding each method's documented frame hop + constant group delay, and compensating the group delay before comparison (FR-006, FR-019, SC-001/002)
- [X] T021 [US1] Wire `run` deterministic path: parse → render → validate → write accepted/rejected + append manifest in `src/voders/cli/run.py` + `src/voders/corpus/orchestrator.py` (FR-002, FR-006a)
- [X] T022 [US1] Implement `voders eval` command (reads manifest, prints SC pass/fail table, exit code) in `src/voders/cli/eval.py` wrapping `evals/run_eval.py`

**Checkpoint**: US1 is a fully functional, independently testable MVP.

---

## Phase 4: User Story 2 - Timbre multiplication through voice conversion (Priority: P2)

**Goal**: Fan one accepted render out across N enrolled voices, preserving alignment, producing `score_NNN_singer_X` pairs that share the source score.

**Independent Test**: Take one accepted US1 sample, run it through N voices; all N share the same score row-for-row, each passes the tolerance check, and timbre differs across outputs.

### Tests for User Story 2 ⚠️

- [X] T023 [P] [US2] Contract test: voice-conversion lane preserves score f0 (`auto_predict_f0=False`) and the consent gate refuses `consent_verified=false` in `tests/contract/test_voiceconv_lane.py`
- [X] T024 [P] [US2] Integration test: fan-out to N voices shares score row-for-row, each variant passes tolerance, unlicensed voice skipped + logged in `tests/integration/test_us2_voiceconv.py`

### Evaluation harness for User Story 2 ⚠️

- [X] T025 [US2] Extend `evals/run_eval.py` with SC-004 (≥1,000 timbre identities at scale) and per-variant tolerance over a multi-voice manifest

### Implementation for User Story 2

- [X] T026 [US2] Implement voice-conversion lane (RVC / so-vits-svc, `auto_predict_f0=False`, lazy GPU import) in `src/voders/render/voiceconv.py` (FR-004, research Decision 4)
- [X] T027 [US2] Implement timbre fan-out in `src/voders/corpus/orchestrator.py` (one render → N voices; consent refusal + missing-voice omission recorded) (FR-010, FR-011)
- [X] T028 [US2] Add donor-voice omission/refusal provenance records in `src/voders/manifest/io.py` (US2 scenario 3)

**Checkpoint**: US1 + US2 both independently functional.

---

## Phase 5: User Story 3 - Production-chain domain randomization (Priority: P3)

**Goal**: Apply label-safe augmentation (reverb, codec, accompaniment mix) so the corpus is in-the-mix, leaving score labels untouched.

**Independent Test**: Take a rendered sample, apply the augmentation chain at varied settings; score-aligned labels still pass onset/offset/f0 verification on the augmented audio.

### Tests for User Story 3 ⚠️

- [X] T029 [P] [US3] Contract test: augmentation never alters score note rows; SNR floor → quarantine in `tests/contract/test_augment_lane.py`
- [X] T030 [P] [US3] Integration test: reverb+codec+accompaniment preserves labels and passes tolerance; vocal stays dominant; clipping normalized/rejected in `tests/integration/test_us3_augment.py`

### Evaluation harness for User Story 3 ⚠️

- [X] T031 [US3] Extend `evals/run_eval.py` with SC-006 (≥60% of accepted samples carry ≥1 production-style augmentation)

### Implementation for User Story 3

- [X] T032 [P] [US3] Implement `AugmentationProfile` model + step registry in `src/voders/render/augment.py` (data-model.md; FR-005)
- [X] T033 [US3] Implement augmentation chain (reverb IR via `pedalboard`, MP3/Opus codec round-trip via `torchaudio`, accompaniment mix at SNR sweep) in `src/voders/render/augment.py` (research Decision 5)
- [X] T034 [US3] Implement SNR-floor quarantine + clipping normalize/reject gate in `src/voders/validate/validator.py` + orchestrator (FR-014, US3 scenario 3)

**Checkpoint**: US1–US3 independently functional.

---

## Phase 6: User Story 4 - Naturalistic neural SVS with label safety net (Priority: P4)

**Goal**: Add expressive neural-SVS samples with a guarantee that expressive timing does not silently corrupt labels (force-score-F0 OR re-derive-and-gate).

**Independent Test**: Render a score through neural SVS; either the re-derived labels stay within tolerance or the sample is rejected with the deviation logged.

### Tests for User Story 4 ⚠️

- [X] T035 [P] [US4] Contract test: SVS lane two-mode invariants (`force_score_f0` → `label_score==input`; `rederive_labels` → re-derived score + deviation reported) in `tests/contract/test_svs_lane.py`
- [X] T036 [P] [US4] Integration test: re-derived tsv carried + deviation recorded; >50 ms onset deviation → rejected/flagged; force-score-F0 behaves like P1 in `tests/integration/test_us4_svs.py`

### Evaluation harness for User Story 4 ⚠️

- [X] T037 [US4] Extend `evals/run_eval.py` with SC-010 (re-derived onset deviation <50 ms in ≥90% of accepted notes; failures rejected)

### Implementation for User Story 4

- [X] T038 [US4] Implement SVS lane `force_score_f0` mode (DiffSinger via OpenUTAU / NNSVS, lazy GPU import) in `src/voders/render/svs.py` (FR-007, research Decision 3)
- [X] T039 [US4] Implement `rederive_labels` mode: MFA forced alignment + onset detection → re-derived score + deviation in `src/voders/render/svs.py` + `src/voders/validate/rederive.py`, registering the aligner/onset-detector documented frame hop and group delay in the timing-budget registry and compensating before deviation is computed (FR-007, FR-019)
- [X] T040 [US4] Enforce rejection on >50 ms deviation with flagged provenance in orchestrator (SC-010)

**Checkpoint**: US1–US4 independently functional.

---

## Phase 7: User Story 5 - Provenance and license audit (Priority: P5)

**Goal**: Make corpus composition transparent and auditable — full per-sample provenance chain, license/consent audit, replayable from manifest.

**Independent Test**: Query the manifest for any sample's full provenance chain; re-run with the same seed and confirm the sample reproduces within documented tolerance.

### Tests for User Story 5 ⚠️

- [X] T041 [P] [US5] Contract test: `audit` refuses any `consent_verified=false` reaching the corpus; manifest records validate against schema in `tests/contract/test_audit.py`
- [X] T042 [P] [US5] Integration test: query full provenance chain for a sample; reproduce sample in isolation by seed and match within tolerance in `tests/integration/test_us5_provenance.py`

### Evaluation harness for User Story 5 ⚠️

- [X] T043 [US5] Extend `evals/run_eval.py` with SC-008 (zero unconsented voices in accepted corpus) and SC-009 (isolated re-render matches the documented tolerance: bit-exact for deterministic/voice-conversion lanes; same verdict + f0-within-±25-cents + onset-within-10 ms for neural/GPU lanes)

### Implementation for User Story 5

- [X] T044 [P] [US5] Implement `voders audit` command (license/consent audit over manifest) in `src/voders/cli/audit.py` (FR-011, SC-008)
- [X] T045 [P] [US5] Implement `voders stats` command writing `stats.json` (totals, unique scores/voices, timbre identities, pitch/duration distributions, augmentation coverage) in `src/voders/cli/stats.py` (FR-012)
- [X] T046 [US5] Implement per-sample isolated reproduction (re-run by derived seed) verification in `src/voders/corpus/reproduce.py`, asserting the SC-009 tolerance per lane (bit-exact vs. verdict+f0/onset) (FR-013, SC-009)

**Checkpoint**: All five user stories independently functional.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T047 [P] Update `README.md` and `specs/001-synthetic-singing-corpus/quickstart.md` to match shipped CLI (Constitution Principle III)
- [X] T048 [P] Add unit tests: seed determinism, score-parse edge cases (polyphony/empty/legato), manifest scan in `tests/unit/`
- [X] T049 [P] Add throughput benchmark asserting deterministic lane ≥100 score-singer pairs/hour on 4 CPU cores in `evals/bench.py` (SC-005)
- [X] T050 Validate streaming/checkpoint resume across a 10k+-sample run (resume from last shard) in `tests/integration/test_resume.py` (SC-011)
- [X] T051 Run `quickstart.md` end-to-end (`voders run` then `voders eval`) and confirm green
- [X] T052 Ensure `ruff` + `ruff format` + `mypy` clean and that importing CPU modules pulls in no `torch` at module load (FR-009, Principle II)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup; BLOCKS all user stories.
- **User Stories (Phase 3–7)**: all depend on Foundational. US1 is the MVP; US2–US5 build on US1's parser/renderer/validator/orchestrator but each is independently testable.
- **Polish (Phase 8)**: depends on the desired user stories being complete.

### User Story Dependencies

- **US1 (P1)**: after Foundational — no dependency on other stories.
- **US2 (P2)**: after US1 (consumes accepted US1 renders for fan-out).
- **US3 (P3)**: after US1 (augments rendered samples); independent of US2.
- **US4 (P4)**: after US1 + validator (re-derivation safety net); independent of US2/US3.
- **US5 (P5)**: after US1 (audits whatever manifest exists); richer once US2–US4 add provenance variety.

### Within Each User Story

- Tests + eval-harness task written FIRST and FAIL before implementation (Constitution Principle I/IV).
- Models → lane/service → orchestration wiring → CLI.

### Parallel Opportunities

- Setup: T003, T004 in parallel.
- Foundational: T006, T007, T008, T010, T011 in parallel (T009 after T008; T012/T013 after their deps).
- Within a story: all `[P]` test tasks run together; `[P]` implementation tasks touch different files.
- With staff: once Foundational is done, US3 and US4 can proceed alongside US2 (all build only on US1).

---

## Parallel Example: User Story 1

```bash
# Tests first (parallel):
Task: "Contract test for deterministic lane in tests/contract/test_deterministic_lane.py"   # T014
Task: "Contract test for validator in tests/contract/test_validator.py"                      # T015
Task: "Integration test for US1 scenarios in tests/integration/test_us1_deterministic.py"    # T016
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup → 2. Phase 2 Foundational → 3. Phase 3 US1 → 4. **STOP and validate**: run `voders run` + `voders eval` on the fixtures, confirm SC-001/002/007 green. This alone is a usable, label-correct corpus (the spec's "minimum viable corpus").

### Incremental Delivery

US1 (MVP) → US2 (timbre scale) → US3 (in-the-mix realism) → US4 (expressive + safety net) → US5 (audit/replay). Each adds value without breaking earlier stories.

---

## Notes

- `[P]` = different files, no incomplete dependencies.
- CPU baseline (US1) carries no `torch` import at module load so it installs/runs on a laptop (FR-009).
- Generated corpora are never committed; only fixtures (via Git LFS) and the end-result record (manifest + resolved config + stats) are version-controlled.
- Commit after each task or logical group; no AI-attribution trailers (Constitution).
