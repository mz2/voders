---
description: "Task list for Optional Lyric Generation & Phonetic Diversity"
---

# Tasks: Optional Lyric Generation & Phonetic Diversity

**Input**: Design documents from `/specs/002-lyric-generation/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: REQUIRED. The constitution (Principle I, NON-NEGOTIABLE) overrides the template's
"tests optional" note: every behavioural task starts as a failing test, and per the `/speckit-tasks`
gate each user story carries a test task **and** an evaluation-harness task ordered before its
implementation.

**Organization**: Tasks grouped by user story (US1=P1 … US4=P4). MVP = Phase 1 + 2 + US1.

## Implementation status (2026-06-27)

Implemented on the `002-lyric-generation` branch (full pytest suite + ruff green):

- **Foundational**: `lyrics` extra (phonemizer) added; `voders.lyrics` package (models, sources,
  sampler, coverage, **syllabify**, **g2p**, **cache**); `Note.lyric` + 4th-column parse/serialize;
  `LyricsConfig` (incl. `syllabifier`) + `ProvenanceRecord` lyric axis (incl.
  `lyric_multisyllable_supplied`); FR-005 import-isolation tests.
- **US1**: lyric layer wired through the orchestrator (per-(score,voice) plan → render request +
  provenance); `SuppliedSource` (cells as authored, multi-syllable flagged); SVS lane
  `lyric_articulated` for articulating backends; G2P vowel-on-the-beat mapping; differential
  no-shift (label-level) test (SC-008/009). Real neural articulation rides the out-of-process SVS
  backend (not run on CPU here).
- **FR-019/SC-010**: deterministic `syllabify` (segment + syllable_count); one-syllable-per-note
  structural guarantee for automatic/generated; supplied multi-syllable flagging.
- **US3**: provenance population + stats lyric axis (by_source, multisyllable total, phonetic
  coverage). License gate folded into US4 (runtime, manifest-surfaced).
- **US4**: `GeneratedSource` + cache-as-artifact (replay with zero model re-invocation) + license
  refusal; lyrics backend bridge + deterministic placeholder worker.
- **Styles**: automatic source `inventory` selects the style — `en_cv` (neutral CV) and **`scat`**
  (jazz scat-singing, Scatman "ski-ba-bop-ba-dop-bop"). Fixtures `lyrics-smoke.yaml` /
  `lyrics-scat.yaml` run end-to-end on CPU.

Remaining (environment-bound / optional): a dedicated `voders eval --suite lyrics` CLI (SC gates are
currently covered by the pytest suite); real neural SVS phoneme articulation + real espeak G2P +
real LLM generation (the out-of-process/GPU backends, plumbed with placeholders).

**Amended 2026-06-27** (post-`/speckit-clarify`, FR-019 + SC-010): added the deterministic `syllabify`
module (T011a/T011b) and the one-syllable-per-note work — `supplied` multi-syllable flagging (T017,
provenance `lyric_multisyllable_supplied` in T010/T011, stats in T034), structural SC-010 assertions
for `automatic` (T023/T025) and `generated` segmentation (T035/T037/T039). New tasks use letter
suffixes to keep existing IDs stable.

## Path Conventions

Single Python project: `src/voders/`, `tests/`, `evals/`, optional out-of-process `backends/lyrics/`.
All commands run via `uv` (constitution: Python Tooling).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: dependencies and package scaffolding for the lyric layer.

- [ ] T001 Add a `lyrics` optional extra (`phonemizer>=3.3`) to `[project.optional-dependencies]` in `pyproject.toml`, regenerate `uv.lock` via `uv lock`, and record the espeak-ng system prerequisite in `README.md` (run/test instructions, constitution gate 3).
- [ ] T002 [P] Scaffold the CPU lyric package `src/voders/lyrics/__init__.py` (no torch/phonemizer import at module load — FR-005), and add a **failing** guard test `tests/unit/test_lyrics_imports.py` asserting `import voders.lyrics` loads neither `torch` nor `phonemizer` (mirrors `tests/unit/test_device.py`; makes FR-005 a gated test, not just a lint guard).
- [ ] T003 [P] Scaffold the optional out-of-process generated backend: `backends/lyrics/pyproject.toml`, `backends/lyrics/.python-version`, `backends/lyrics/src/voders_lyrics_backend/__init__.py` (own uv project; not synced by default).

**Checkpoint**: package + extra exist; CPU baseline still imports with no new heavy deps.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the shared data carriers, config selector, manifest axis, and source interface every
story depends on. **No user story may start until this phase is complete.** All preserve SC-001
byte-identity for lyric-free runs.

- [ ] T004 [P] Failing unit test landing the prototype: `tests/unit/test_lyrics.py` — 3-column score parses with `lyric=None`; 4th column parsed as lyric; `serialize_score` stays 3-column when lyric-free, 4-column otherwise (from the `lyrics` branch; copy/cherry-pick).
- [ ] T005 Add optional `lyric: str | None = None` to `Note` in `src/voders/scores/models.py`; make `parse_tsv` read an optional 4th column and `serialize_score` emit it only when a score carries lyrics, in `src/voders/scores/parse.py` (FR-002, SC-001; makes T004 pass).
- [ ] T006 [P] Failing unit test for lyric models: `tests/unit/test_lyric_models.py` — `LyricPlan` length invariant (== note count), `text_hash` determinism, `LyricModel.license_ok` gate (FR-008/010).
- [ ] T007 [P] Implement `LyricSource` enum, `LyricPlan`, `LyricModel`, and `Phoneme run` dataclasses in `src/voders/lyrics/models.py` per data-model.md (makes T006 pass).
- [ ] T008 [P] Failing contract test for the source interface: `tests/contract/test_lyric_source.py` — every source returns a `LyricPlan` with `len(syllables)==len(notes)`; `vowel` returns all-`None`; count reconciliation sets `mismatch=True` and never drops/shifts notes (contracts/lyric-source.md, FR-008/SC-005).
- [ ] T009 Implement the `LyricSource` Protocol + a `vowel` source + a `resolve_source(config)` factory and the count-reconciliation helper in `src/voders/lyrics/sources.py` (makes T008 pass; `automatic`/`supplied`/`generated` land in their story phases).
- [ ] T010 [P] Failing contract test for config + manifest deltas: `tests/contract/test_lyrics_config_manifest.py` — `RunConfig` accepts a `lyrics` block defaulting to `source: vowel` (incl. `syllabifier: en_rule`); `theme`/`model` with `source!=generated` is a validation error; `melisma: sustain_ties` raises a "not yet supported" validation error in v1 (only `per_note` accepted); `ProvenanceRecord` accepts the 6 new lyric fields (incl. `lyric_multisyllable_supplied: int = 0`) with backward-compatible defaults (contracts/manifest-and-config-deltas.md).
- [ ] T011 Add `LyricsConfig` to `src/voders/config/models.py` (default `source: vowel`; `syllabifier: "en_rule"`; `theme`/`model` require `source: generated` else validation error; `melisma` accepts only `per_note` in v1 — `sustain_ties` is **reserved** and raises a "not yet supported" validation error, per the spec Assumptions), and add the lyric provenance fields (`lyric_source`, `lyric_hash`, `lyric_model`, `lyric_model_license`, `lyric_articulated`, `lyric_multisyllable_supplied`) to `ProvenanceRecord` in `src/voders/manifest/models.py` (FR-009/010/013/019; makes T010 pass).
- [ ] T011a [P] Failing unit test for the syllable segmenter: `tests/unit/test_syllabify.py` — `segment(text)` splits words/free text into an ordered list of singable syllables **deterministically** (same input → same output; `"winter"` → 2 syllables); `syllable_count(text)` returns 1 for a single syllable and >1 for a multi-syllable word; an align helper maps a syllable list 1:1 onto notes leaving `multisyllable_notes` empty (FR-019, structural SC-010).
- [ ] T011b Implement `src/voders/lyrics/syllabify.py` — `segment` (used by `generated`) and `syllable_count` (flag for `supplied`), via a dictionary lookup + vowel-group rule fallback; pure-CPU, no new heavy dependency, no `torch`/`phonemizer` import (makes T011a pass; FR-019, FR-005).

**Checkpoint**: data carriers, config selector, manifest axis, and source interface exist; a
`source: vowel` run is byte-identical to pre-feature (verify SC-001 before proceeding).

---

## Phase 3: User Story 1 - Sing supplied lyrics with the label safety net + no pitch/timing shift (Priority: P1) 🎯 MVP

**Goal**: the expressive (SVS) lane articulates supplied syllables as phonemes instead of a flat
vowel, under 001's force-score-F0 / re-derive safety net, with a **gated guarantee that adding vocal
synthesis shifts neither pitch nor timing** versus the lyric-free render of the same (score, voice,
seed) (FR-006/007/015/016/017/018, SC-002/008/009).

**Independent Test**: render a syllable-bearing score through the SVS lane; confirm consonants are
present and audible, labels pass tolerance or the sample is rejected, and the differential (lyric-on
vs lyric-off, same seed) pitch/onset deltas are within ±25 cents / 10 ms.

### Evaluation harness for User Story 1 (constitution Principle IV — before impl)

- [ ] T012 [US1] Add the `lyrics` eval suite + differential mode to `src/voders/cli/eval.py` and `evals/run_eval.py`: `uv run voders eval --manifest <out> --suite lyrics [--differential <lyric_free_manifest>]` prints a pass/fail table for SC-001/002/005/008/009 (SC-005: zero dropped/added/shifted note labels across count mismatches) and exits non-zero on any gated failure (FR-018).
- [ ] T013 [P] [US1] Add the supplied-source fixtures (canonical names per quickstart.md §3): `evals/fixtures/lyrics-supplied.yaml` (SVS lane, supplied source) and a paired lyric-free `evals/fixtures/lyrics-offbaseline.yaml` sharing the same master seed (for the differential check), plus a 4-column supplied-lyric score under `evals/fixtures/scores/` that includes **at least one multi-syllable cell** (e.g. `winter`) to exercise the supplied-as-authored flag (FR-019).

### Tests for User Story 1 (write first, must fail)

- [ ] T014 [P] [US1] Integration test `tests/integration/test_us1_lyrics_svs.py`: SVS lane articulates a supplied syllable (phoneme energy/consonant present vs the vowel render) and, in force-score-F0 mode, `label_score == score` and passes onset/offset/pitch tolerance (SC-002). Also assert a **multi-syllable supplied cell is sung as authored** (not re-segmented/truncated/rejected) and its note index is flagged in `LyricPlan.multisyllable_notes` with `lyric_multisyllable_supplied` incremented in the record (FR-019).
- [ ] T015 [P] [US1] Integration test `tests/integration/test_us1_no_shift.py`: render the same (score, voice, seed) with and without lyrics; assert per-note pitch within ±25 cents (SC-008) and onset/offset within 10 ms after constant-delay compensation (SC-009); a deliberately mis-timed phoneme stub is rejected, not admitted.
- [ ] T016 [P] [US1] Unit test `tests/unit/test_g2p.py`: syllable → phonemes maps the vowel nucleus onset to the note onset, leading consonants into a pre-onset window, trailing into a pre-offset window (Decision L3); out-of-inventory text falls back to the open vowel.

### Implementation for User Story 1

- [ ] T017 [US1] Implement the `supplied` source (read `Note.lyric` → `LyricPlan`) in `src/voders/lyrics/sources.py` (FR-002). Take supplied cells **as authored** and use `syllabify.syllable_count` to append any non-single-syllable cell's index to `LyricPlan.multisyllable_notes` (no re-segmentation/truncation/rejection) so the count surfaces in provenance/stats (FR-019).
- [ ] T018 [P] [US1] Implement `src/voders/lyrics/g2p.py`: lazy `phonemizer`/espeak-ng import, syllables → `Phoneme run`s mapped to note durations, vowel-on-the-beat, vowel fallback for un-pronounceable text (FR-006; makes T016 pass).
- [ ] T019 [US1] Register an `svs_articulation` method timing budget (frame hop + constant lead-in group delay) in `src/voders/validate/timing.py` so the validator subtracts the articulation path's constant delay before comparing onsets (FR-017).
- [ ] T020 [US1] Wire the SVS lane to articulate a `LyricPlan` when present: pass phonemes through `render_via_backend` (extend the JSON request) and the in-process CPU path in `src/voders/render/svs.py`; keep `force_score_f0` pitch score-derived (FR-015 by construction) and `rederive_labels` gated (FR-007). Set `lyric_articulated=True` only here.
- [ ] T021 [US1] Implement the differential no-shift check used by T015 and the eval suite: compare a lyric render's per-note measured pitch/onset/offset to the lyric-free baseline render of the same seed; record the largest deltas in the verdict `notes`; force-reject notes exceeding ±25 cents / 10 ms (FR-016/018, SC-008/009).
- [ ] T022 [US1] Ensure the deterministic and voice-conversion lanes leave audio unchanged when a lyric is present and set `lyric_articulated=False` (FR-006 edge case) — guard in `src/voders/render/deterministic.py` and `src/voders/render/voiceconv.py`.

**Checkpoint**: `uv run voders eval --suite lyrics --differential …` is green for SC-001/002/008/009;
the SVS lane sings consonants with no pitch/timing shift. **MVP deliverable.**

---

## Phase 4: User Story 2 - Automatic syllable source for corpus-scale phonetic diversity (Priority: P2)

**Goal**: a seeded, CPU, dependency-free syllable source that needs no lyric input data and broadens
phonetic coverage, reproducible independent of worker count (FR-003/004/005, SC-003/004).

**Independent Test**: enable `source: automatic` with a fixed seed; re-run at different worker counts
→ identical syllables; corpus phoneme inventory ≥10× the vowel baseline.

### Evaluation harness for User Story 2

- [ ] T023 [US2] Extend the `lyrics` eval suite to assert SC-003 (distinct sung-phoneme inventory ≥10× vowel baseline), SC-004 (identical syllables across a 1-worker vs N-worker re-run), and **SC-010 structural** (every `automatic` `LyricPlan` has exactly one syllable token per note, `multisyllable_notes` empty — no acoustic counting) in `evals/run_eval.py` + `src/voders/cli/eval.py`.

### Tests for User Story 2 (write first, must fail)

- [ ] T024 [P] [US2] Unit test `tests/unit/test_sampler.py`: `sample_seed(master, score_id, voice_id, "lyrics")`-seeded sampler is deterministic and order/worker-independent; coverage-biased draws span the inventory (SC-004).
- [ ] T025 [P] [US2] Integration test `tests/integration/test_us2_automatic.py`: an `automatic` run assigns **exactly one singable syllable per note** (structural SC-010: `len(plan.syllables)==len(notes)`, `multisyllable_notes` empty) and the phoneme inventory exceeds the vowel baseline by ≥10× (SC-003); sub-`min_note_ms` notes follow 001's short-note policy.

### Implementation for User Story 2

- [ ] T026 [P] [US2] Add the checked-in consonant–vowel inventory `src/voders/lyrics/data/en_cv.txt` (small text; no binary) and the automatic-source run config `evals/fixtures/lyrics-smoke.yaml` (SVS lane, `source: automatic` — the canonical harness fixture named in plan.md §Evaluation Strategy and quickstart.md §2).
- [ ] T027 [US2] Implement `src/voders/lyrics/sampler.py`: seeded, coverage-biased CV-syllable sampler (makes T024 pass), and wire the `automatic` source into `src/voders/lyrics/sources.py` (FR-003/004).
- [ ] T028 [P] [US2] Implement `src/voders/lyrics/coverage.py`: distinct sung-phoneme count for a corpus vs the vowel baseline (used by T023; FR-014/SC-003).

**Checkpoint**: `source: automatic` produces reproducible, phonetically-diverse corpora on CPU.

---

## Phase 5: User Story 3 - Lyric provenance and source/license audit (Priority: P3)

**Goal**: every sample records its lyric source + text hash; model sources record id + license and are
refused when the license is unacceptable; stats report the lyric breakdown (FR-009/010/014, SC-006).

**Independent Test**: run each source, query the manifest for the lyric axis, and confirm a
license-refused model produces no samples and is surfaced.

### Evaluation harness for User Story 3

- [ ] T029 [US3] Extend the `lyrics` eval suite / `src/voders/cli/audit.py` to assert SC-006: every record has `lyric_source` + (non-vowel) `lyric_hash`; zero samples carry a license-refused lyric model.

### Tests for User Story 3 (write first, must fail)

- [ ] T030 [P] [US3] Integration test `tests/integration/test_us3_lyric_provenance.py`: records carry the correct `lyric_source`/`lyric_hash` per source; a `LyricModel` with `license_ok=false` yields a `license_refused` outcome and no sample (FR-010/SC-006).
- [ ] T031 [P] [US3] Unit test `tests/unit/test_lyric_stats.py`: the stats report includes the lyric-source breakdown, phonetic-coverage summary, and the supplied multi-syllable count (FR-014/019).

### Implementation for User Story 3

- [ ] T032 [US3] Populate the lyric provenance fields when writing records in the orchestrator/`src/voders/cli/run.py` (source, hash, model id/license, articulated flag, and `lyric_multisyllable_supplied` = `len(plan.multisyllable_notes)`) (FR-009/019).
- [ ] T033 [US3] Enforce the `LyricModel` license gate in `src/voders/lyrics/sources.py` (refuse + surface, mirroring `Voice.consent_verified`) (FR-010).
- [ ] T034 [US3] Add the lyric-source breakdown + phonetic-coverage summary + **supplied multi-syllable count** (sum of `lyric_multisyllable_supplied`) to the stats output in `src/voders/cli/stats.py` (FR-014/019; makes T031 pass).

**Checkpoint**: lyric composition is fully auditable; license discipline extends to lyric models.

---

## Phase 6: User Story 4 - Themed generated lyrics, reproducible without re-running the model (Priority: P4)

**Goal**: an optional, opt-in model-driven source from a theme string; generated text is produced once
and pinned so replay never re-invokes the model (FR-011/012, SC-007). Fully droppable.

**Independent Test**: generate with a theme, record the pinned artifact, regenerate from the manifest,
confirm the pinned text is reused verbatim with zero model re-invocations.

### Evaluation harness for User Story 4

- [ ] T035 [US4] Extend the `lyrics` eval suite to assert SC-007 (a generated run replays from `<output_root>/lyrics/<hash>.jsonl` with zero model calls and meets 001's per-lane reproduction tolerance) and **SC-010 structural** for `generated` (every plan has exactly one syllable token per note after segmentation, `multisyllable_notes` empty) in `evals/run_eval.py`.

### Tests for User Story 4 (write first, must fail)

- [ ] T036 [P] [US4] Unit test `tests/unit/test_lyric_cache.py`: pin/lookup of generated text by sha256; replay reads the pinned file and never invokes the model (FR-012).
- [ ] T037 [P] [US4] Integration test `tests/integration/test_us4_generated.py` plus its fixture `evals/fixtures/lyrics-generated.yaml` (generated source, stub/mock model emitting multi-syllable words — canonical name per quickstart.md §4): a `generated` run **segments model text into one syllable per note** via `syllabify.segment` (structural SC-010, `multisyllable_notes` empty), pins text, count-matches notes (vowel fallback on mismatch), and regenerates identically from the manifest (SC-007/019).

### Implementation for User Story 4

- [ ] T038 [P] [US4] Implement `src/voders/lyrics/cache.py`: write/read `<output_root>/<cache_dir>/<sha256>.jsonl` pinned artifacts (FR-012; makes T036 pass).
- [ ] T039 [US4] Implement the `generated` source in `src/voders/lyrics/sources.py`: run-once pre-pass via the out-of-process `backends/lyrics` worker, **run the returned text through `syllabify.segment` and align one syllable per note** (FR-019; overflow/underflow via the FR-008 vowel fallback), cache + pin the segmented syllables, theme/model validation (FR-011).
- [ ] T040 [P] [US4] Implement the generated backend worker `backends/lyrics/src/voders_lyrics_backend/worker.py` (constrained-decoding default / SongComposer alt per research Decision L1) and the `render_via_backend` lyrics request in `src/voders/render/backend_bridge.py`.

**Checkpoint**: optional themed generation works and replays deterministically from pinned text.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T041 [P] Update `README.md` (espeak-ng prereq, `uv sync --extra lyrics`, lyric source usage) and confirm `specs/002-lyric-generation/quickstart.md` runs end-to-end on a clean checkout (constitution gate 3).
- [ ] T042 Run `uv run ruff check` + `uv run ruff format --check` + `uv run mypy` clean across the new `src/voders/lyrics/` and changed files (constitution gate 2); guard the lazy `phonemizer` import.
- [ ] T043 [P] Run the full `uv run voders eval --suite lyrics --differential …` against all four sources and record the SC-001..SC-010 pass table in the PR (constitution gate 5).

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (P1: T001–T003)** → no deps.
- **Foundational (P2: T004–T011, T011a–T011b)** → depends on Setup; **BLOCKS all user stories**. `syllabify` (T011a/T011b) is consumed by the `supplied` flag (US1) and `generated` segmentation (US4).
- **US1 (P3)** → after Foundational. The MVP; also lands the no-shift guarantee.
- **US2 (P4)** → after Foundational. Independent of US1 (different files: sampler/coverage/inventory), but its SVS articulation rides US1's g2p/svs wiring at run time.
- **US3 (P5)** → after Foundational; reads provenance written by any source (lightest with US1/US2 present).
- **US4 (P6)** → after Foundational; independent, fully droppable.
- **Polish (P7)** → after the desired stories.

### Within each story

Eval-harness task → tests (must fail) → implementation. Models before sources before lane wiring.

### Parallel opportunities

- Setup: T002, T003 in parallel.
- Foundational: T004, T006, T008, T010, T011a (tests) in parallel; then T005/T007/T009/T011/T011b (note T009 depends on T007, T011 on its test, T011b on T011a).
- US1 tests T014/T015/T016 in parallel; impl T018 parallel to T017/T019.
- US2 T024/T025 in parallel; T026/T028 parallel to T027.
- After Foundational, US1–US4 can be staffed in parallel by different developers.

---

## Implementation Strategy

### MVP (Phase 1 + 2 + US1)

1. Setup → Foundational → verify SC-001 byte-identity.
2. US1: eval harness + differential check first, then SVS articulation.
3. **STOP and VALIDATE**: `uv run voders eval --suite lyrics --differential …` green for
   SC-001/002/008/009 — lyrics are sung with no pitch/timing shift. Ship.

### Incremental delivery

US1 (MVP) → US2 (automatic diversity at scale) → US3 (audit before any release) → US4 (optional
themed generation). Each story is independently testable and adds value without breaking the prior.

---

## Notes

- [P] = different files, no incomplete-task dependency. [US#] maps to spec.md user stories.
- Constitution: tests fail before impl; every Python action via `uv run`; no AI-attribution trailers;
  no new binaries (inventory + fixtures are text; model weights live in the out-of-process backend).
- The no-shift guarantee (FR-015/016/017/018) is enforced **relative to the lyric-free baseline** and
  gated by rejection — it is implemented across T015/T019/T020/T021 inside US1.
- One-syllable-per-note (FR-019, SC-010) is a **structural** invariant on the `LyricPlan`, not an
  acoustic check: `automatic` satisfies it by construction (T025), `generated` via `syllabify.segment`
  (T039), and `supplied` cells are taken as authored with non-single-syllable cells flagged
  (`lyric_multisyllable_supplied`, T017). The shared segmenter is T011a/T011b.
