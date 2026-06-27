# Implementation Plan: Score-Domain Augmentations (volume, time humanisation, transposition)

**Branch**: `003-score-augmentations` | **Date**: 2026-06-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-score-augmentations/spec.md`

## Summary

Add an **optional, opt-in score pre-processor stage** to the 001 corpus pipeline that fans out new
*score variants* from each base score along three axes — **octave/semitone transposition**, **time
humanisation** (seeded onset/duration jitter), and **per-note volume variation** — *before* anything
renders. The stage is a pure function `expand(base_score, profile, seed) → [variant scores]`. Each
variant is just another input score: it flows through the **entire existing 001 pipeline unchanged**
(parse → render → validate → write), so labels are correct **by construction** (a variant's note rows
*are* its labels) and no renderer, voice-conversion, or audio-augmentation-lane code needs to change for
the two label-rewriting axes.

This is the **opposite category** from 001's audio augmentation lane (`render/augment.py`), which is
deliberately label-safe (001 FR-005: never move onsets/offsets or pitch). Here we *intentionally rewrite
the labels* at the input-data stage and let the pipeline render audio to match.

Two design choices carry the feature:

1. **Insertion point** — the stage sits between `Orchestrator._load_scores()` and the existing
   `(score × voice × lane × audio-profile)` loop (`corpus/orchestrator.py:101,123`). Expanding the
   parsed-score list there makes score variants compose as a corpus multiplier for free (FR-015) with no
   change to the render loop.
2. **Physical + naming separation** (FR-016, the operator's explicit ask) — augmented variants are
   written to a distinct `corpus/augmented/<axis>/` subtree (and `rejected/augmented/<axis>/`), never
   co-mingled with or overwriting the originals in `corpus/`; each variant's `score_id` (hence its
   `sample_id` and filenames) encodes its base score and applied transform, so a file is self-describing.
   With no score-augmentation config the `augmented/` subtree never appears and originals stay byte-for-
   byte where they are today (SC-001).

The default is **off**: `RunConfig.score_augmentation` defaults to empty, so every existing run
reproduces bit-for-bit. The whole feature is pure-CPU `numpy` — no new dependency, no GPU, no
out-of-process backend (unlike 002's generated source).

## Technical Context

**Language/Version**: Python 3.14 (the 001 core), via uv. The entire feature lives in the core — it is a
deterministic score-row transform plus a per-note gain the deterministic lane honours; no model, no GPU,
no out-of-process backend.
**Primary Dependencies**: **None new.** Transposition is integer arithmetic; humanisation uses the
existing seeded `numpy` RNG (`voders.seeds.rng`); volume is a per-note linear gain. No change to
`pyproject.toml` / `uv.lock` beyond version bump if any.
**Storage**: Reuses 001's output root and sharded `CorpusStore`. Adds an `augmented/<axis>/` subtree
under `corpus/` and `rejected/` for variant audio + label `.tsv`. Variant labels are ordinary 3/4-column
score `.tsv` (transposition/humanisation rewrite onset/offset/pitch; the optional per-note gain is **not**
serialised into the label file — it rides in-memory to the renderer and is recorded in provenance, so the
label `.tsv` stays a pure transcription label). No new database.
**Testing**: `pytest` via `uv run pytest` (001's harness). Lint/format: `ruff` + `ruff format`,
zero-warning. Type-check: `mypy` (advisory).
**Target Platform**: Linux. Entirely commodity 4-core CPU; no GPU anywhere in this feature.
**Project Type**: Single Python project — extends the 001 library + CLI with a `voders.scoreaug` package,
a `RunConfig.score_augmentation` block, a score-augmentation provenance axis, and a one-line per-note gain
hook in the deterministic lane.
**Performance Goals**: The transform is O(notes) integer/float arithmetic per variant, negligible against
rendering. It MUST NOT regress 001's deterministic-lane throughput (≥100 score-singer pairs/hour, 4-core
CPU). Expansion is a cheap pre-pass over the (small) score set, off the per-sample hot path.
**Constraints**: Audio stays 22,050 Hz mono float32. **Feature-off runs MUST be byte-identical to the
pre-feature pipeline (SC-001).** Every emitted note stays within MIDI `0..127` (FR-005). Every humanised
variant is a valid monophonic score — ordered, non-overlapping (default policy), strictly positive
durations — accepted by the existing parser without modification (FR-007, SC-003). Every transform is a
pure function of `(base score, profile, seed)`, reproducible independent of worker count (FR-010, SC-005).
Augmented variants are physically separated from originals with zero path collisions (FR-016, SC-009).
**Scale/Scope**: Same 10k–100k samples/run as 001; this multiplies sample count by `Σ variants` per base
score (`1 + |transpose offsets| + N humanise draws`, optionally × a volume flag), composing with voices
and audio-augmentation profiles. The run logs the projected multiplier before producing.

### Evaluation Strategy *(Constitution Principle IV — eval-first)*

- **How to run it (single command):**
  `uv run voders run --config evals/fixtures/score-aug-smoke.yaml` renders the checked-in fixture scores
  through the deterministic lane with a score-augmentation profile enabling all three axes (transpose
  `[-12, +12]`, a 2-draw humanisation, volume on), writing originals to `corpus/` and variants to
  `corpus/augmented/<axis>/`, plus the manifest and stats.
- **How to evaluate it (single command, yes/no verdict):**
  `uv run voders eval --manifest <out>/manifest.jsonl --suite score_aug` prints a pass/fail table mapped
  to this spec's Success Criteria and exits non-zero on any gated failure:
  - feature-off control run is byte-identical to a pre-feature render (SC-001),
  - every transposition variant note is shifted by exactly the offset with onsets/offsets byte-identical
    to the base, and zero notes fall outside MIDI `0..127` (SC-002),
  - every humanisation variant is a valid monophonic score and every per-note deviation is within the
    configured max-deviation budget (SC-003),
  - with volume on, the rendered per-note level distribution is measurably wider than the un-varied
    baseline while `(onset, offset, pitch)` labels are byte-identical (SC-004),
  - re-running expansion for a fixed `(base, profile, seed)` yields byte-identical variants regardless of
    worker count / order (SC-005),
  - every variant-derived record carries base score id + profile + seed + applied transform, and a
    file's path/id alone classifies it original-vs-augmented (SC-006, SC-009),
  - stats report score-augmentation coverage as its own axis and the reported effective sample count
    equals `variants × voices × audio-profiles` (SC-007),
  - augmented variants pass the existing 001 verification gate at the same rate as base scores (SC-008).
- **Fixture:** the 001 fixture scores plus `score-aug-smoke.yaml` (all three axes on, deterministic lane)
  and `score-aug-off.yaml` (feature off, shares a seed with the smoke config for the SC-001 control).
  All small text; no new binaries (Git-LFS unaffected).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|-----------|------|--------|
| I. Red-Green-Refactor TDD | `/speckit-tasks` emits a failing test per story first: transposition offset + range guard, humanisation validity + budget, per-note gain honoured + label byte-identity, provenance/separation/coverage | PASS — planned |
| II. Zero-Warning Linting | `ruff` + `ruff format` clean; no new deps to lint around | PASS — toolchain reused |
| III. Docs current with repo | spec/plan/research/data-model/quickstart + README (new `score_augmentation` config key + `corpus/augmented/` layout) updated in the same change set; CLAUDE.md plan pointer updated | PASS |
| IV. Evaluation-First | Runnable harness + runnable eval defined above, mapped to SC-001..SC-009, before feature code | PASS — strategy captured |
| V. Clear, Audience-Aware Writing | "humanisation", "monophonic", "MIDI 0..127", "melisma" glossed on first use; no marketing language | PASS |
| Binary Assets & Large Files | No new binaries; only small text fixtures/configs; generated corpora never committed | PASS |
| Commit & Attribution Discipline | No AI-attribution trailers | PASS — followed |
| Python Tooling: uv | No new dependency; if any, declared in `pyproject.toml` with `uv.lock` regenerated; all actions via `uv run` | PASS |

No violations. Complexity Tracking table left empty.

## Project Structure

### Documentation (this feature)

```text
specs/003-score-augmentations/
├── plan.md              # This file
├── research.md          # Phase 0 — insertion point, determinism, separation layout, constraint solver
├── data-model.md        # Phase 1 — ScoreVariant, profile/config, Note.gain, provenance/store deltas
├── quickstart.md        # Phase 1 — single-command run + eval with score augmentations
├── contracts/
│   ├── score-augmentation.md   # expand() interface + transform semantics + ScoreVariant
│   └── config-manifest-store.md# RunConfig.score_augmentation, provenance fields, augmented/ layout
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/voders/
├── scoreaug/                 # NEW package — the score pre-processor (pure CPU)
│   ├── __init__.py
│   ├── models.py             # ScoreVariant (variant Score + provenance descriptor); axis enums
│   ├── expand.py             # expand(base, profile, seed) -> [ScoreVariant]; the multiplier (FR-002/015)
│   ├── transpose.py          # semitone shift + range guard (drop/clamp), MIDI 0..127 (FR-004/005)
│   ├── humanize.py           # seeded onset/duration jitter + monophonic constraint solver (FR-006/007)
│   └── volume.py             # seeded per-note gain within range; label-preserving (FR-008/009)
├── scores/
│   └── models.py             # Note gains optional `gain: float | None` (linear, label-excluded) (FR-008/009)
├── corpus/
│   ├── orchestrator.py       # call expand() after _load_scores; route variant writes; log multiplier (FR-002/015)
│   ├── store.py              # augmented/<axis>/ subtree under corpus/ & rejected/; no-collision paths (FR-016)
│   └── stats.py              # score-augmentation coverage axis (per offset / per draw / volume) (FR-012)
├── config/
│   └── models.py             # ScoreAugmentationProfile + RunConfig.score_augmentation (default []) (FR-013)
├── manifest/
│   └── models.py             # ProvenanceRecord gains base_score_id, score_aug_profile, score_aug_transform (FR-011)
├── render/
│   └── deterministic.py      # honour optional per-note gain; model lanes record non-application (FR-008)
├── evaluation.py             # SC-001..SC-009 score-aug suite (differential, range, validity, coverage)
└── cli/run.py                # log projected effective-sample multiplier before producing (FR-015)

tests/
├── contract/                 # expand() interface; config schema; provenance + store-layout deltas
├── integration/              # US1 transposition; US2 humanisation validity+determinism; US3 volume; US4 separation/provenance
└── unit/                     # transpose range guard; humanize constraint solver; volume gain; multiplier count

evals/
└── fixtures/                 # score-aug-smoke.yaml (all axes) + score-aug-off.yaml (SC-001 control)
```

**Structure Decision**: Single Python project. The new `voders.scoreaug` package is a pure score-to-scores
function consumed at exactly one point — the orchestrator's pre-render expansion — so the render loop,
lanes, validator, and seeding model are reused unchanged for the transposition and humanisation axes. Only
two existing modules gain real behaviour: `CorpusStore` (the `augmented/` subtree for physical separation,
FR-016) and the deterministic lane (one optional per-note gain multiply, FR-008); both are additive and
gated on score augmentation being active, so feature-off runs are byte-identical (SC-001). No
`backends/` project is needed — this feature pulls in no model or GPU dependency.

## Complexity Tracking

> No constitution violations. No entries.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | | |
