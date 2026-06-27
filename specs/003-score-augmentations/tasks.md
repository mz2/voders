---
description: "Task list for Score-Domain Augmentations (volume, time humanisation, transposition)"
---

# Tasks: Score-Domain Augmentations

**Input**: Design documents from `/specs/003-score-augmentations/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: REQUIRED. The constitution (Principle I, NON-NEGOTIABLE) overrides the template's
"tests optional" note: every behavioural task starts as a **failing** test, and per the
`/speckit-tasks` gate each user story carries a test task **and** an evaluation-harness task ordered
before its implementation.

**Organization**: Tasks grouped by user story (US1=P1 transpose · US2=P2 humanise · US3=P3 volume ·
US4=P3 separation/provenance/coverage). MVP = Phase 1 + Phase 2 + US1.

## Path Conventions

Single Python project: `src/voders/`, `tests/`, `evals/`. No new dependency, no GPU, no `backends/`
project — the feature is pure-CPU `numpy`. All commands run via `uv` (constitution: Python Tooling).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: package scaffolding and eval fixtures. No dependency changes.

- [x] T001 Scaffold the pure-CPU package `src/voders/scoreaug/__init__.py`, and add a **failing** guard test `tests/unit/test_scoreaug_imports.py` asserting `import voders.scoreaug` pulls in neither `torch` nor any GPU module (the feature is `numpy`-only; mirrors `tests/unit/test_lyrics_imports.py`).
- [x] T002 [P] Create eval fixtures `evals/fixtures/score-aug-smoke.yaml` (deterministic lane; one profile enabling all three axes: `transpose: [-12, 12]`, `humanize_time` with `draws: 2`, `volume`) and `evals/fixtures/score-aug-off.yaml` (no `score_augmentation`, same seed as smoke — the SC-001 control), both pointing at the existing `evals/fixtures/scores`; document the new `score_augmentation` config key and the `corpus/augmented/<axis>/` layout in `README.md` run/test instructions (constitution gate 3).

**Checkpoint**: package imports clean with no new heavy deps; fixtures exist.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the shared carriers (config, provenance, variant model, per-note gain), the `expand()`
dispatch, and the generic orchestrator/store wiring every story rides on. **No user story may start
until this phase is complete.** Every delta defaults to inert so a run without `score_augmentation`
stays byte-identical (SC-001).

- [x] T003 [P] Failing contract test `tests/contract/test_score_aug_config_manifest.py` — `RunConfig` accepts `score_augmentation` defaulting to `[]` (absent ⇒ unchanged); profile validation (`transpose.policy ∈ {drop,clamp}`, `window` within `[0,127]` and `lo≤hi`, `humanize_time.draws ≥ 1`, `max_dev_s>0`, `min_dur_s>0`, `overlap == "forbid"` only in v1 else "not yet supported" error, unique `profile_id`); `ProvenanceRecord` accepts `base_score_id`/`score_aug_profile`/`score_aug_axis`/`score_aug_transform`/`score_aug_seed`/`dynamics_applied` with backward-compatible defaults and `extra="forbid"` intact (contracts/config-manifest-store.md).
- [x] T004 Implement `ScoreAugmentationProfile` + `TransposeKnob`/`HumanizeKnob`/`VolumeKnob` and `RunConfig.score_augmentation` in `src/voders/config/models.py`, and add the six score-aug fields to `ProvenanceRecord` in `src/voders/manifest/models.py` (FR-005/006/008/011/013; makes T003 pass).
- [x] T005 [P] Failing unit test `tests/unit/test_score_variant_models.py` — `ScoreAxis` enum values; `ScoreVariant` invariants; `score.score_id == f"{base}__{transform}"`; a base `score_id` containing the reserved `__` separator is rejected.
- [x] T006 [P] Implement `ScoreAxis` + `ScoreVariant` in `src/voders/scoreaug/models.py` per data-model.md (FR-003/011; makes T005 pass).
- [x] T007 Add optional `gain: float | None = None` to `Note` in `src/voders/scores/models.py`, and add a failing-then-passing unit test `tests/unit/test_score_gain_label.py` asserting `serialize_score` output is **byte-identical** whether or not `gain` is set (gain is excluded from the label `.tsv`, FR-009; `parse_tsv` never reads it).
- [x] T008 [P] Failing contract test `tests/contract/test_score_aug_expand.py` — the `expand()` guarantees G1–G5 from contracts/score-augmentation.md using its determinism test vector (byte-identical re-expansion; every pitch ∈ `[0,127]`; every variant a valid monophonic score the parser accepts; note count + per-note `lyric` preserved; labels are the rows).
- [x] T009 Implement the `expand(base, profile, seed) -> list[ScoreVariant]` dispatch skeleton in `src/voders/scoreaug/expand.py` — derive per-axis sub-seeds via `derive_seed(master_seed, base_id, "score_aug", profile_id, axis[, draw])` (`voders.seeds`), fixed axis order, returns `[]` when no knob set; per-axis transforms are filled in their story phases (FR-002/010; partially satisfies T008).
- [x] T010 Wire the pre-render expansion into `Orchestrator.run()` in `src/voders/corpus/orchestrator.py`: after `_load_scores()` build the work list as base scores **plus** `expand(ps, profile, seed)` flattened over `config.score_augmentation`; iterate exactly as today; populate `base_score_id` + `score_aug_*` (+ `score_aug_seed`) on each variant's `ProvenanceRecord`; when `score_augmentation == []` the list and outputs are byte-identical to today (FR-002, SC-001).
- [x] T011 Add `augmented/<axis>/` routing to `CorpusStore.paths_for` in `src/voders/corpus/store.py`: originals keep their exact current `corpus/`/`rejected/` path; a record with `score_aug_axis` set routes to `corpus/augmented/<axis>/shard=NNN/` (or `rejected/augmented/<axis>/`); guarantees no path collision (FR-016). Depends on T010 populating `score_aug_axis`.

**Checkpoint**: run `uv run voders run --config evals/fixtures/score-aug-off.yaml` and confirm
`corpus/` is byte-identical to a pre-feature run and **no** `corpus/augmented/` directory exists
(SC-001) before starting any user story.

---

## Phase 3: User Story 1 - Octave transposition variants (Priority: P1) 🎯 MVP

**Goal**: for each configured semitone offset, emit a variant score with every pitch shifted by exactly
that offset and onsets/offsets unchanged, range-guarded, landing in `corpus/augmented/transpose/`.

**Independent Test**: run the smoke fixture with `transpose: [-12, 12]`; confirm two variants per base
with pitches shifted exactly and onset/offset byte-identical to the base, an out-of-window offset
dropped (or clamped) and recorded, and every emitted note within MIDI `0..127`.

- [x] T012 [P] [US1] Failing unit test `tests/unit/test_transpose.py` — `transpose(score, offset, policy, window)`: every `pitch_midi += offset`; `policy="drop"` returns `None` when any shifted pitch leaves `window`; `policy="clamp"` clamps and records a `clamped` count; onsets/offsets/`lyric` unchanged; never emits a pitch outside `[0,127]` (FR-004/005).
- [x] T013 [P] [US1] Failing integration test `tests/integration/test_us1_transpose.py` — run the deterministic lane with `transpose: [-12, 12]` on a fixture; assert two variant scores per base, pitches shifted exactly, `(onset_s, offset_s)` byte-identical to the base, variant files under `corpus/augmented/transpose/shard=*/`, and an out-of-range offset on an extreme fixture is dropped with the reason in provenance (SC-002).
- [x] T014 [US1] Evaluation harness — add the gated **SC-002** criterion (transposition exactness + zero notes outside `0..127`, read by comparing each `t±N` variant against its `base_score_id`) to a new `score_aug` suite in `src/voders/evaluation.py`, wired into `voders eval --suite score_aug`; this is US1's runnable verdict (constitution gate 5), authored before the implementation below.
- [x] T015 [US1] Implement `transpose()` in `src/voders/scoreaug/transpose.py` and hook the transpose axis into `expand.py` (one variant per offset, `transform = f"t{offset:+d}"`, axis `transpose`); makes T012/T013 pass and turns T014 green.

**Checkpoint**: transposition variants are produced, physically separated, and SC-002 passes — MVP is
demonstrable end-to-end.

---

## Phase 4: User Story 2 - Time humanisation variants (Priority: P2)

**Goal**: emit N variant scores per base with seeded onset/duration jitter, each a valid monophonic
score within the max-deviation budget, landing in `corpus/augmented/humanize/`.

**Independent Test**: run with `humanize_time{draws: 2}`; confirm two variants per base with non-zero
per-note deviations within budget, each variant parses as a valid monophonic score, and the draws are
byte-identical on a repeat run.

- [x] T016 [P] [US2] Failing unit test `tests/unit/test_humanize.py` — `humanize(score, knob, sub_seed)`: onset/offset jitter present and every `|deviation| ≤ max_dev_s`; output is ordered, non-overlapping (`overlap="forbid"`), every duration `≥ min_dur_s`; a forced constraint keeps base timing and increments `constraint_hit`; identical output for a fixed `sub_seed`; pitch/`lyric` unchanged (FR-006/007).
- [x] T017 [P] [US2] Failing integration test `tests/integration/test_us2_humanize.py` — run with `draws: 2`; assert two variants per base, deviations non-zero and within budget, each variant accepted by `parse_tsv` unmodified, files under `corpus/augmented/humanize/shard=*/`, and byte-identical variants across a re-run with a different worker count/order (SC-003, SC-005).
- [x] T018 [US2] Evaluation harness — add gated **SC-003** (every humanise variant valid monophonic + deviations within budget) and **SC-005** (re-expansion byte-identical) criteria to the `score_aug` suite in `src/voders/evaluation.py`; authored before the implementation below.
- [x] T019 [US2] Implement the clamp-then-project constraint solver `humanize()` in `src/voders/scoreaug/humanize.py` (Gaussian jitter clamped to `max_dev_s`, left-to-right projection preserving order/non-overlap/positive duration) and hook the humanise axis into `expand.py` (one variant per draw, `transform = f"hum{draw}"`, axis `humanize`); makes T016/T017 pass and turns T018 green.

**Checkpoint**: humanisation variants are valid, bounded, deterministic, and separated.

---

## Phase 5: User Story 3 - Per-note volume / dynamics variation (Priority: P3)

**Goal**: attach a seeded per-note gain that the deterministic lane honours, widening the rendered
level distribution while keeping `(onset, offset, pitch)` labels byte-identical; lanes that cannot
honour gain record `dynamics_applied: false`.

**Independent Test**: render a fixture with volume on through the deterministic lane and compare to the
un-varied render; confirm wider per-note levels but byte-identical label `.tsv`, and that the SVS/voice-
conversion lanes record `dynamics_applied: false`.

- [x] T020 [P] [US3] Failing unit test `tests/unit/test_volume.py` — `volume(score, knob, sub_seed)` assigns each note a seeded `gain` within `gain_db_range` (linear); `(onset_s, offset_s, pitch_midi, lyric)` unchanged; identical for a fixed `sub_seed` (FR-008/009).
- [x] T021 [P] [US3] Failing integration test `tests/integration/test_us3_volume.py` — run volume on through the deterministic lane; assert the rendered per-note level distribution is measurably wider than the un-varied render, the label `.tsv` is byte-identical to the un-varied render, files land under `corpus/augmented/volume/shard=*/`, and an SVS/voice-conversion lane render of the same variant records `dynamics_applied: false` (SC-004, FR-008 edge case).
- [x] T022 [US3] Evaluation harness — add the gated **SC-004** criterion (volume widens the rendered level distribution AND `(onset,offset,pitch)` labels byte-identical to the un-varied render) to the `score_aug` suite in `src/voders/evaluation.py`; authored before the implementation below.
- [x] T023 [US3] Implement `volume()` in `src/voders/scoreaug/volume.py` (seeded per-note gain) and hook the volume axis into `expand.py` (`transform = "vol"`, axis `volume`); make the deterministic lane honour `note.gain` in `src/voders/render/deterministic.py` (multiply each note's samples before mixing; the existing peak-normalise preserves relative levels) and set `dynamics_applied=true`; model lanes leave it `false`; makes T020/T021 pass and turns T022 green.

**Checkpoint**: dynamics are audible, label-preserving, and lane-aware.

---

## Phase 6: User Story 4 - Originals and augmentations unmistakably separated + provenance & coverage (Priority: P3)

**Goal**: prove that originals and variants never collide on disk, every variant is self-describing and
fully traceable in the manifest, and the run surfaces score-augmentation coverage and the corpus
multiplier.

**Independent Test**: run all axes; confirm originals in `corpus/` and variants in
`corpus/augmented/<axis>/` with zero collisions, every variant record carries base id + profile + seed
+ transform + location, a file is classifiable original-vs-augmented from its path/id alone, the stats
report a `score_augmentation` coverage block, and the projected effective sample count is logged.

- [x] T024 [P] [US4] Failing integration test `tests/integration/test_us4_separation.py` — run all axes; assert originals live only under `corpus/shard=*/` and variants only under `corpus/augmented/<axis>/shard=*/` with zero path collisions; every variant record has `base_score_id`, `score_aug_profile`, `score_aug_seed`, `score_aug_transform`, and a resolved augmented `score_path`/`audio_path`; classify every file original-vs-augmented from its path/id alone; regenerating from the manifest reproduces variants byte-identically (SC-006, SC-009, FR-016).
- [x] T025 [P] [US4] Failing unit test `tests/unit/test_stats_score_aug.py` — `build_stats` returns a `score_augmentation` block (`enabled`, `variant_share`, `by_transform` counts, `effective_multiplier`) from a synthetic manifest with mixed originals/variants (FR-012).
- [x] T026 [US4] Evaluation harness — add gated **SC-001** (off-control run byte-identical to a pre-feature render, no `augmented/` subtree), **SC-006** (lineage fields present + path-classifiable), **SC-007** (coverage axis reported + effective count = `variants × voices × audio-profiles`), and **SC-009** (separation, zero collisions) criteria to the `score_aug` suite in `src/voders/evaluation.py`, using `score-aug-off.yaml` as the SC-001 control; authored before the implementation below.
- [x] T027 [US4] Implement the `score_augmentation` coverage block in `src/voders/corpus/stats.py` (`variant_share`, `by_transform`, `effective_multiplier`) from the new provenance fields (FR-012); makes T025 pass.
- [x] T028 [US4] Implement the projected effective-sample-count log in `src/voders/cli/run.py` before producing (`Σ_scores (1 + variants) × voices × (1 + audio_profiles)`, FR-015), and confirm the orchestrator records `dynamics_applied` and the resolved augmented location so T024's SC-006/SC-009 assertions pass; turns T026 green.

**Checkpoint**: full separation guarantee, queryable lineage, and coverage reporting — the augmented
corpus is releasable.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: docs, scope guard, and the green end-to-end verdict.

- [x] T029 [P] Finalise `README.md` and any 001 docs touched: the `score_augmentation` config block, the `corpus/augmented/<axis>/` layout, and the `uv run voders eval --suite score_aug` command (constitution gate 3).
- [x] T030 [P] Add a scope-guard unit test `tests/unit/test_score_aug_scope.py` asserting `expand()` emits **per-axis** variants only (no transpose×humanize cross-products in v1), locking the documented scope.
- [x] T031 Run the full gate via `uv`: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest`, `uv run mypy` (advisory) — all clean (constitution gates 1, 2).
- [x] T032 Run quickstart.md end-to-end: `uv run voders run --config evals/fixtures/score-aug-smoke.yaml` then `uv run voders eval --manifest <out>/manifest.jsonl --suite score_aug` and confirm SC-001..SC-009 all pass (constitution gate 5).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies — start immediately.
- **Foundational (Phase 2)**: depends on Setup — **blocks all user stories**. Within it: T003→T004,
  T005→T006, T007 standalone, T008→T009→T010→T011 (the wiring chain).
- **User Stories (Phase 3–6)**: each depends only on Foundational. US1/US2/US3 are independent axis
  functions (different files under `src/voders/scoreaug/`) and can proceed in parallel; US4 verifies the
  separation/coverage that US1–US3 produce, so it is most meaningful once ≥1 axis is implemented but its
  store/provenance mechanism is already foundational.
- **Polish (Phase 7)**: after the desired stories are complete.

### User Story Dependencies

- **US1 (P1)**: Foundational only — MVP.
- **US2 (P2)**: Foundational only — independent of US1 (`humanize.py` vs `transpose.py`).
- **US3 (P3)**: Foundational only — independent; additionally touches `render/deterministic.py`.
- **US4 (P3)**: Foundational only for its mechanism; its eval/integration assertions are strongest after
  US1–US3 land. Each `evaluation.py` criterion task (T014, T018, T022, T026) edits the same file, so
  those run sequentially across stories (naturally ordered by phase).

### Within Each User Story

- Failing unit + integration tests and the eval-harness task precede the implementation task.
- Axis transform before its `expand.py` hook (same task), before the eval turns green.

### Parallel Opportunities

- T002 ‖ T001 (different files).
- Foundational: {T003, T005, T008} author tests in parallel; {T006} ‖ after T005; T007 ‖ the config work.
- US1: T012 ‖ T013. US2: T016 ‖ T017. US3: T020 ‖ T021. US4: T024 ‖ T025.
- Across stories: US1, US2, US3 implementation can be staffed in parallel after Foundational.

---

## Parallel Example: User Story 1

```bash
# Author US1 tests together (they fail first):
Task: "Unit test transpose() in tests/unit/test_transpose.py"
Task: "Integration test transposition run in tests/integration/test_us1_transpose.py"
# Then the eval criterion, then the implementation:
Task: "Add SC-002 to the score_aug suite in src/voders/evaluation.py"
Task: "Implement transpose() + expand hook in src/voders/scoreaug/transpose.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup → 2. Phase 2 Foundational (verify the SC-001 off-control checkpoint) →
3. Phase 3 US1 → **STOP and validate**: `uv run voders eval --suite score_aug` shows SC-002 green and
the `corpus/augmented/transpose/` subtree is populated and collision-free.

### Incremental Delivery

1. Setup + Foundational → off-control byte-identical (SC-001).
2. US1 transpose → SC-002 → demo (MVP).
3. US2 humanise → SC-003/SC-005 → demo.
4. US3 volume → SC-004 → demo.
5. US4 separation/coverage → SC-001/006/007/009 → releasable augmented corpus.

### Notes

- `[P]` = different files, no dependency on an incomplete task.
- Every behavioural task is red first (constitution Principle I); verify the failing state before
  implementing.
- No new dependency: do not add to `pyproject.toml`/`uv.lock` for this feature.
- Keep originals' on-disk location unchanged throughout — it is the SC-001 backstop.
- Commit after each task or logical group (no AI-attribution trailers, per the constitution).
