---
description: "Task list for Vocal-Conditioned Accompaniment Lane"
---

# Tasks: Vocal-Conditioned Accompaniment Lane

**Input**: Design documents from `/specs/002-vocal-conditioned-accompaniment/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: REQUIRED. The project constitution (Principle I — TDD; Principle IV — eval-first) overrides
the template's "tests are optional" note. Every user story gets test tasks AND at least one
evaluation-harness task, ordered before that story's implementation tasks.

**Organization**: Tasks are grouped by user story (US1–US4) so each is an independently testable
increment. All four stories share the Foundational plumbing (Phase 2); US1 (Lego) is the MVP.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story the task belongs to (US1–US4)
- Exact file paths are included in every task

## Path Conventions

Single Python project (extends `voders`): package at `src/voders/`, tests at `tests/`, eval
harness/fixtures at `evals/`. All commands run through `uv` (constitution: Python Tooling).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Optional GPU extra, module skeletons, fixture config.

- [X] T001 [P] Add the `accomp` optional-dependency extra (`torch>=2.12`, `torchaudio`, the ACE-Step 1.5 XL inference package, optional `demucs`-class separator) to `pyproject.toml` and regenerate `uv.lock` with `uv lock` (plan Technical Context; research.md Decision 8)
- [X] T002 [P] Create module skeletons with docstrings: `src/voders/render/accompaniment.py`, `src/voders/render/accompaniment_backend.py`, `src/voders/render/backends/__init__.py`, `src/voders/render/backends/acestep.py`, `src/voders/validate/mix.py`, and matching empty test files under `tests/{contract,integration,unit}/`
- [X] T003 [P] Add `evals/fixtures/accompaniment-smoke.yaml` enabling the accompaniment stage with `backend: fake` for both Lego and Complete over the existing donor vocals (quickstart.md; contract: run-config-accompaniment.md)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The backend boundary, config/manifest/store/validation plumbing every story needs.
⚠️ No user-story work begins until this phase is complete.

### Tests (write first, must FAIL before impl)

- [X] T004 [P] Contract test for `AccompanimentBackend` Protocol conformance + `BackendOutput` invariants (mode/return rules; output already 22,050 mono; torch-free fake) in `tests/contract/test_accompaniment_backend.py` (contract: accompaniment-backend.md)
- [X] T005 [P] Contract test for the manifest `accompaniment` provenance object shape + field rules (`vocal_bit_exact == (mode=="lego")`, license within policy, attribution for CC-BY) in `tests/contract/test_manifest_accompaniment.py` (contract: manifest-accompaniment.md)
- [X] T006 [P] Contract test for accompaniment run-config option parsing + validation (`free_time ⟹ bpm is null`, `takes ≥ 1`) in `tests/contract/test_config_accompaniment.py` (contract: run-config-accompaniment.md)

### Implementation

- [X] T007 Define `AccompanimentBackend` Protocol + `BackendOutput` dataclass in `src/voders/render/accompaniment_backend.py` (data-model.md §2; makes T004 pass)
- [X] T008 Implement the deterministic torch-free CPU **fake** backend (Lego: seeded, score-shaped band-limited pad as the stem; Complete: that pad summed under a vocal copy as the mix; `model_license="MIT"`, `requires_gpu()=False`, supports both modes) in `src/voders/render/accompaniment_backend.py` (contract: fake-backend section; depends on T007)
- [X] T009 [P] Add the `AccompanimentProvenance` typed sub-model on `ProvenanceRecord` and update `contracts/manifest-record.schema.json` in `src/voders/manifest/models.py` (FR-007/006a; makes T005 pass)
- [X] T010 [P] Add documented accompaniment option access + validation (`mode`, `backend`, `license_policy`, `target_instrument`, `free_time`, `bpm`, `takes`, `target_snr_db`; `free_time ⟹ bpm null`; `takes ≥ 1`) in `src/voders/config/models.py` (LaneToggle is `extra="allow"`; FR-005/008; makes T006 pass)
- [X] T011 [P] Implement the 48 kHz-stereo → 22,050-mono bridging helper (down-mix + resample) in `src/voders/audio.py` with a unit test in `tests/unit/test_accompaniment_bridge.py` (research.md Decision 3)
- [X] T012 Extend `CorpusStore` to write `mix.wav` + (Lego) `accompaniment_stem.wav` and record `source_vocal_sample_id` in `src/voders/corpus/store.py` with a unit test in `tests/unit/test_store_stems.py` (FR-015; data-model.md §4)
- [X] T013 Implement validate-on-mix + per-note timing-shift measurement (reuse `validate/rederive.py::derive_labels`; return `max_note_shift_ms`) in `src/voders/validate/mix.py` with a unit test in `tests/unit/test_mix_validate.py` (FR-004/012; research.md Decisions 2, 5)
- [X] T014 Implement the `Accompanist` stage scaffold (mode dispatch; take loop; best-aligned selection by `max_note_shift_ms` then SNR; license-policy pre-check no-op; backend/GPU-unavailable no-op with logged skip) in `src/voders/render/accompaniment.py` (FR-006/010/013; research.md Decision 6; depends on T007, T008, T010, T011, T013)
- [X] T015 Wire the orchestrator fan-out: for each `ACCEPTED` base render, fan through enabled accompaniment configs via `Accompanist`, validate, write, append to manifest — in `src/voders/corpus/orchestrator.py` (mirrors the existing augmentation fan-out; depends on T012, T014)

**Checkpoint**: Foundation ready — user-story phases can begin.

---

## Phase 3: User Story 1 - Layer accompaniment without moving note timing (Priority: P1) 🎯 MVP

**Goal**: Lego mode — generate an instrument stem, sum it beneath the untouched vocal; vocal stays
byte-identical, timing preserved by construction; validated on the mix and admitted best-take.

**Independent Test**: `uv run voders run --config evals/fixtures/accompaniment-smoke.yaml` (Lego) then
`uv run voders eval` — the admitted mix's vocal channel is byte-identical to the source vocal, onsets
≤ 50 ms / offsets within tolerance, and `accompaniment_stem.wav` is stored.

### Tests (write first, must FAIL before impl)

- [X] T016 [P] [US1] Integration test for the Lego path (mix vocal channel byte-identical to source vocal; onsets/offsets within tolerance; stem stored; `vocal_bit_exact=true`) in `tests/integration/test_us1_lego.py` (spec US1; SC-001/008)
- [X] T017 [P] [US1] Unit test for take selection (admits the best-aligned passing take, records `takes_tried`; all-fail → rejected with reason) in `tests/unit/test_take_selection.py` (FR-010)
- [X] T018 [US1] Extend the eval harness to report SC-001 (Lego vocal bit-exact vs source) and SC-008 (mix + stem stored), exiting non-zero on failure, in `evals/run_eval.py` and `src/voders/cli/eval.py` (constitution gate 5)

### Implementation

- [X] T019 [US1] Implement Lego mode in `Accompanist`: call backend (`mode=lego`) → bridge stem to 22,050 mono → sum under the untouched vocal at `target_snr_db` (reuse `render/augment.py` mixing/SNR) → return mix + retained stem; set `vocal_bit_exact=true` — in `src/voders/render/accompaniment.py` (SC-001; depends on T014, T019 uses T011)
- [X] T020 [US1] Populate `AccompanimentProvenance` for Lego (mode, model id/version/license, `source_vocal_sample_id`, `takes_tried`, `max_note_shift_ms`, `stem_available=true`) and persist via the orchestrator in `src/voders/render/accompaniment.py` + `src/voders/corpus/orchestrator.py` (FR-007)
- [X] T021 [US1] Run the Lego smoke + eval; make T016–T018 green (`uv run voders run … && uv run voders eval …`)

**Checkpoint**: US1 (MVP) fully functional and independently testable.

---

## Phase 4: User Story 2 - One-pass full mix with locked onsets (Priority: P2)

**Goal**: Complete mode — model emits a single mix (vocal re-encoded, mild coloration); admitted only
if sung notes stay within tolerance when measured on the mix.

**Independent Test**: run the smoke config in Complete mode then eval — admitted mixes keep onsets/
offsets within tolerance on the mix, `vocal_bit_exact=false`, `stem_available=false`.

### Tests (write first, must FAIL before impl)

- [X] T022 [P] [US2] Integration test for the Complete path (onsets/offsets within tolerance validated on the mix via separation; `vocal_bit_exact=false`; `stem_available=false`; one-pass offset-drift sample is rejected with reason) in `tests/integration/test_us2_complete.py` (spec US2; SC-002)
- [X] T023 [US2] Extend the eval harness to report SC-002 (one-pass onset/offset on the mix ≥ 99%) in `evals/run_eval.py` (constitution gate 5)

### Implementation

- [X] T024 [US2] Implement Complete mode in `Accompanist`: backend (`mode=complete`) → bridge mix to 22,050 mono → validate on the mix using the source-separation → `derive_labels` path; set `vocal_bit_exact=false`, `stem_available=false` — in `src/voders/render/accompaniment.py` + `src/voders/validate/mix.py` (research.md Decision 2; depends on T013, T014)
- [X] T025 [US2] Run the Complete smoke + eval; make T022–T023 green

**Checkpoint**: US1 and US2 both work independently.

---

## Phase 5: User Story 3 - Free-time / rubato-safe accompaniment (Priority: P3)

**Goal**: For free/rubato material, avoid a metrical pulse displacing the phrasing — no BPM, sustained
captions — and enforce the gate: any take that shifts a note is rejected; admitted free-time samples
carry no fixed-tempo metadata.

**Independent Test**: run the smoke config with `free_time: true` then eval — admitted samples carry
no fixed-tempo metadata and `max_note_shift_ms` is zero within tolerance; a shifted take is rejected
with the per-note shift recorded.

### Tests (write first, must FAIL before impl)

- [X] T026 [P] [US3] Integration test for free-time handling (no fixed-tempo metadata on admitted samples; `max_note_shift_ms` within tolerance; a note-shifting take rejected with per-note shift recorded) in `tests/integration/test_us3_freetime.py` (spec US3; FR-005/012, SC-005)
- [X] T027 [US3] Extend the eval harness to report SC-005 (zero note shift; no fixed-tempo metadata on free-time samples) in `evals/run_eval.py` (constitution gate 5)

### Implementation

- [X] T028 [US3] Implement free-time handling in `Accompanist`: pass `free_time`/`target_instrument` captions and null BPM to the backend; enforce the `bpm`-null invariant; ensure the note-shift gate rejects shifted takes and records `max_note_shift_ms`; omit tempo metadata from provenance — in `src/voders/render/accompaniment.py` (FR-005/012; depends on T013, T014)
- [X] T029 [US3] Run the free-time smoke + eval; make T026–T027 green

**Checkpoint**: US1–US3 all independently functional.

---

## Phase 6: User Story 4 - License & provenance audit (Priority: P4)

**Goal**: Only commercial-friendly (MIT/Apache/CC-BY-class) weights are used; CC-BY attribution is
captured and propagated; the audit flags any violation. Corpus stays redistributable.

**Independent Test**: `uv run voders audit --manifest <out>/manifest.jsonl` passes on the fixture
(complete provenance, in-policy licenses, attribution present); a disallowed-license config produces
no admitted samples and a logged skip.

### Tests (write first, must FAIL before impl)

- [X] T030 [P] [US4] Unit test for license-policy enforcement (disallowed license → stage no-ops with logged skip, nothing admitted; CC-BY-class → `attribution_text` propagated to every sample) in `tests/unit/test_license_policy.py` (FR-006/006a)
- [X] T031 [P] [US4] Integration test for `voders audit` (flags out-of-policy license + missing attribution; passes on the fixture) in `tests/integration/test_us4_audit.py` (spec US4; SC-004)
- [X] T032 [US4] Extend the eval harness to report SC-004 (provenance complete; 0 disallowed-license admitted; CC-BY-class attribution present) in `evals/run_eval.py` (constitution gate 5)

### Implementation

- [X] T033 [US4] Implement the license-policy gate in `Accompanist` (check `backend.model_license` against `config.license_policy` before `generate`; no-op + log on fail) and attribution propagation into provenance — in `src/voders/render/accompaniment.py` (FR-006/006a; depends on T014)
- [X] T034 [US4] Extend `voders audit` to check accompaniment license + attribution completeness in `src/voders/cli/audit.py` (FR-006a; depends on T009)
- [X] T035 [US4] Run the audit + eval; make T030–T032 green

**Checkpoint**: All user stories independently functional and license-clean.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: The real GPU backend, benchmark, docs, and the final gate sweep.

- [X] T036 [P] Implement the GPU **ACE-Step 1.5 XL** backend (lazy `torch`/ACE-Step import; Lego + Complete; 48 kHz-stereo internal; declares `model_license` + `attribution_text`) in `src/voders/render/backends/acestep.py` (research.md Decision 1; FR-009). **Verify at implementation**: exact `xl-base` LICENSE, that Lego/Complete are exposed by the pinned `generate_music.py`, and the package name / Python 3.14 wheel (research.md open items 1–3)
- [X] T037 [P] Add a GPU-gated throughput benchmark (target ≥ 60 admitted samples/hour per A6000-class GPU) in `evals/bench_accompaniment.py` (research.md Decision 7)
- [X] T038 [P] Update `README.md` run/test instructions and `specs/002-vocal-conditioned-accompaniment/quickstart.md` for the accompaniment stage (constitution III)
- [X] T039 [P] Update `.gitattributes` if any new fixture audio type is introduced; confirm generated mixes/stems are git-ignored, never committed (constitution: Binary Assets)
- [X] T040 Final gate sweep: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest`, and the quickstart run + eval — all green, zero-warning lint (constitution II/IV)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies — start immediately.
- **Foundational (Phase 2)**: depends on Setup — **BLOCKS all user stories**.
- **User Stories (Phases 3–6)**: all depend on Foundational. US1 is the MVP; US2/US3/US4 each build on
  the shared `Accompanist`/orchestrator but are independently testable and can proceed in parallel
  once Phase 2 is done.
- **Polish (Phase 7)**: depends on the desired user stories being complete (T036 real backend can be
  built any time after T007 but is validated last).

### User Story Dependencies

- **US1 (P1)**: after Foundational. No dependency on other stories.
- **US2 (P2)**: after Foundational. Independent of US1 (different mode path).
- **US3 (P3)**: after Foundational. Reuses the shared note-shift gate; independent of US1/US2.
- **US4 (P4)**: after Foundational. Cross-cuts provenance/audit; independent of US1–US3.

### Within Each User Story

- Tests + the eval-harness task are written and FAIL before implementation (constitution I/IV).
- Foundational plumbing before mode-specific behavior.
- Story complete (eval green) before moving to the next priority.

### Parallel Opportunities

- Setup: T001, T002, T003 in parallel.
- Foundational tests: T004, T005, T006 in parallel; impl T009, T010, T011 in parallel (distinct files)
  after the Protocol (T007) lands.
- Once Phase 2 completes, US1–US4 can be staffed in parallel; within a story all `[P]` test tasks run
  together.
- Polish: T036, T037, T038, T039 in parallel.

---

## Parallel Example: User Story 1

```bash
# Write the US1 tests first (parallel), confirm they fail:
Task: "Integration test for the Lego path in tests/integration/test_us1_lego.py"
Task: "Unit test for take selection in tests/unit/test_take_selection.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup.
2. Phase 2 Foundational (CRITICAL — blocks all stories).
3. Phase 3 US1 (Lego).
4. **STOP and VALIDATE**: `uv run voders run … && uv run voders eval …` — SC-001/008 green.
5. Demo the vocal-preserving path.

### Incremental Delivery

1. Setup + Foundational → plumbing ready.
2. US1 Lego → eval green → demo (MVP).
3. US2 Complete → eval green → demo.
4. US3 Free-time → eval green → demo.
5. US4 License/audit → eval green → corpus is redistributable.
6. Polish: real ACE-Step backend + benchmark + docs + gate sweep.

---

## Notes

- **Complete-mode stem**: Phase 4 follows the recommended resolution — Complete sets
  `stem_available=false` (no separable stem), so FR-015/SC-008 re-mixability is a **Lego-mode**
  guarantee (data-model.md design note). Revisit only if strict Complete-stem retention is required.
- The CPU **fake** backend keeps Phases 1–6 runnable in CI without a GPU; the real ACE-Step backend
  (T036) and the throughput benchmark (T037) are GPU-gated and validated in Polish.
- `[P]` tasks = different files, no incomplete-task dependencies.
- Generated mixes/stems are never committed (constitution: Binary Assets); only text manifest +
  resolved config are committable.
- Commit after each task or logical group; no AI-attribution trailers (constitution).
