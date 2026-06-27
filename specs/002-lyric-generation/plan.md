# Implementation Plan: Optional Lyric Generation & Phonetic Diversity

**Branch**: `002-lyric-generation` | **Date**: 2026-06-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-lyric-generation/spec.md`

## Summary

Add an **optional, opt-in lyric layer** to the 001 corpus pipeline whose only purpose is **phonetic
diversity** in the rendered audio — consonants, plosives, fricatives, and vowel transitions the
current single open vowel ("ah") lacks. Lyrics are never a corpus label (the downstream consumer
transcribes pitch, not words). The feature has four lyric *sources* selected per run — `vowel`
(default, unchanged), `supplied` (a per-note `lyric` carried in an optional 4th score column,
already prototyped on the `lyrics` branch), `automatic` (a seeded, CPU syllable sampler that needs
no input data), and `generated` (an optional, opt-in model driven by an operator-supplied theme
string, cached as a pinned artifact). Only the expressive neural singing-voice-synthesis ("SVS")
lane articulates lyrics into phonemes; it does so behind 001's existing force-score-F0 / re-derive
safety net (FR-007), so phonetic realism is added without weakening label correctness. A new lyric
provenance axis (source, text hash, and for model sources the model id + license) rides 001's
manifest and consent discipline.

The default is `vowel`: with no lyric configuration every existing run reproduces byte-for-byte
(SC-001). The supplied annotations carry no lyric/theme data, so the `automatic` source is what the
corpus uses at scale; theming exists only through the optional `generated` source's theme string and
is never read from the score set.

## Technical Context

**Language/Version**: Python 3.14 (the 001 core), via uv. The new lyric sources (`vowel`, `supplied`,
`automatic`) are pure-CPU and live in the 3.14 core. The optional `generated` model source runs
**out of process** in its own uv project under `backends/lyrics/` (mirroring `backends/svs`,
`backends/rvc`, `backends/seedvc`) so a model toolkit that pins older Python / GPU deps never
constrains the core.
**Primary Dependencies** (added as bounded extras; core CPU path stays light):
- **G2P** (grapheme-to-phoneme — turning letters/syllables into the phonemes a voicebank sings):
  `phonemizer>=3.3` driving the **espeak-ng** system library. `phonemizer` is imported lazily, only
  when `lyrics.source != vowel` and the SVS lane articulates; the CPU baseline and the deterministic
  lane never import it. espeak-ng is a system package (e.g. `apt install espeak-ng`) recorded in the
  README/quickstart prerequisites.
- **Automatic syllable inventory**: a small, checked-in consonant–vowel (CV) syllable/phoneme
  inventory (text data, no new heavy dep). Optional `pronouncing>=0.2` (CMUdict wrapper) is **not**
  required for v1; the curated inventory keeps the sampler deterministic and dependency-free.
- **Generated source (optional, `backends/lyrics/`)**: a melody→lyric model (research Decision L1
  picks SongComposer or a constrained-decoding LLM). Its weights/toolkit are **never** committed
  (Binary-Assets rule) and never pulled into the core; the backend project owns them.
**Storage**: Reuses 001's output root. Generated-lyric text is persisted as a pinned, hash-named
artifact under `<output_root>/lyrics/<hash>.jsonl` and hash-referenced from each provenance record,
so a run replays from pinned text without re-invoking a model (FR-012). Supplied lyrics live in the
score `.tsv` 4th column. No new database.
**Testing**: `pytest` via `uv run pytest` (001's harness). Lint/format: `ruff` + `ruff format`,
zero-warning. Type-check: `mypy` (advisory).
**Target Platform**: Linux. All non-generated lyric work runs on a commodity 4-core CPU, no GPU
(FR-005). The `generated` source may use a DGX Spark GPU through its out-of-process backend; it is
fully optional.
**Project Type**: Single Python project — extends the 001 library + CLI with a `voders.lyrics`
package and a lyric-source hook in the SVS lane.
**Performance Goals**: Lyric assignment (`automatic`/`supplied`) and G2P add negligible cost and
MUST NOT regress 001's deterministic-lane throughput target (≥100 score-singer pairs/hour on a
4-core CPU). The `generated` source is a one-time, cached pre-pass per score set, off the per-sample
hot path.
**Constraints**: Audio stays 22,050 Hz mono float32. Lyric-free runs MUST be byte-identical to the
pre-feature pipeline (SC-001). Lyric-driven SVS samples MUST meet the same onset (50 ms) / offset
(max(50 ms, 20%)) tolerances or be rejected (SC-002). Automatic-source lyrics MUST be seed-reproducible
independent of worker count (SC-004).
**Scale/Scope**: Same 10k–100k samples/run as 001; the lyric layer adds one per-note string and one
per-sample provenance axis.

### Evaluation Strategy *(Constitution Principle IV — eval-first)*

- **How to run it (single command):**
  `uv run voders run --config evals/fixtures/lyrics-smoke.yaml` renders the checked-in fixture scores
  with the `automatic` lyric source through the SVS lane (force-score-F0 and re-derive modes), plus a
  lyric-free control, writing audio + a JSONL manifest + the pinned lyric artifact to an output dir.
- **How to evaluate it (single command, yes/no verdict):**
  `uv run voders eval --manifest <out>/manifest.jsonl --suite lyrics` prints a pass/fail table mapped
  to this spec's Success Criteria and exits non-zero on any gated failure:
  - lyric-free scores byte-identical to a pre-feature render (SC-001),
  - onset ≤50 ms / offset within tolerance on ≥99% of accepted lyric-driven SVS samples (SC-002),
  - distinct sung-phoneme inventory ≥10× the vowel baseline (SC-003),
  - identical assigned syllables across a 1-worker vs N-worker re-run (SC-004),
  - zero dropped/added/shifted note labels across all count mismatches (SC-005),
  - every record carries `lyric_source` + `lyric_hash`; zero license-refused lyric models (SC-006),
  - generated-lyric run replays from the pinned artifact with zero model re-invocations (SC-007).
- **Fixture:** the 001 fixture scores, plus (a) one 3-column lyric-free score, (b) one 4-column
  supplied-lyric score, and (c) a tiny CV inventory — all small text, Git-LFS unaffected (no new
  binaries).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|-----------|------|--------|
| I. Red-Green-Refactor TDD | `/speckit-tasks` emits a failing test per story before impl (parse 4th column, automatic-source determinism, SVS phoneme articulation + safety net, provenance/license, generated replay) | PASS — planned |
| II. Zero-Warning Linting | `ruff` + `ruff format` clean; lazy `phonemizer` import guarded | PASS — toolchain reused |
| III. Docs current with repo | spec/plan/quickstart + README prerequisites (espeak-ng) updated in the same change set; CLAUDE.md plan pointer updated | PASS |
| IV. Evaluation-First | Runnable harness + runnable eval defined above, mapped to SC-001..SC-007, before feature code | PASS — strategy captured |
| V. Clear, Audience-Aware Writing | G2P, phoneme, espeak-ng, melisma, SVS glossed on first use; no marketing language | PASS |
| Binary Assets & Large Files | No new binaries; lyric model weights never committed (out-of-process backend); generated corpora never committed | PASS |
| Commit & Attribution Discipline | No AI-attribution trailers | PASS — followed |
| Python Tooling: uv | `phonemizer` added to a `lyrics` optional extra in `pyproject.toml` with `uv.lock` regenerated; `backends/lyrics/` is its own uv project; all actions via `uv run` | PASS |

No violations. Complexity Tracking table left empty.

## Project Structure

### Documentation (this feature)

```text
specs/002-lyric-generation/
├── plan.md              # This file
├── research.md          # Phase 0 — resolves G2P toolchain, automatic-source design, generated model
├── data-model.md        # Phase 1 — Lyric entities, manifest/config additions, lifecycle
├── quickstart.md        # Phase 1 — single-command run + eval with lyrics
├── contracts/           # Phase 1 — lyric-source interface, manifest/config deltas
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/voders/
├── scores/
│   ├── models.py        # Note gains optional `lyric: str | None` (prototype, FR-002)
│   └── parse.py         # parse_tsv reads optional 4th column; serialize_score stays 3-col when lyric-free (FR-001/002)
├── lyrics/              # NEW package — the lyric layer (CPU)
│   ├── __init__.py
│   ├── models.py        # LyricSource enum, LyricPlan (per-note syllables), LyricModel (license), Phoneme run
│   ├── sources.py       # resolve a Score → LyricPlan for vowel | supplied | automatic | generated (FR-003/011/013)
│   ├── sampler.py       # seeded CV-syllable sampler; deterministic from sample_seed(stage="lyrics") (FR-004)
│   ├── g2p.py           # syllables → phonemes (lazy phonemizer/espeak-ng), mapped to note durations (FR-006)
│   ├── cache.py         # pin/lookup generated lyric text as <output_root>/lyrics/<hash>.jsonl (FR-012)
│   └── coverage.py      # phonetic-coverage statistic for the stats report (FR-014, SC-003)
├── render/
│   ├── svs.py           # articulate a LyricPlan's phonemes when present; vowel otherwise (FR-006/007)
│   └── backend_bridge.py# render_via_backend gains optional phonemes+durations in its JSON request
├── config/
│   └── models.py        # RunConfig gains LyricsConfig (source selector, default `vowel`) (FR-013)
└── manifest/
    └── models.py        # ProvenanceRecord gains lyric_source, lyric_hash, lyric_model, lyric_model_license (FR-009/010)

backends/
└── lyrics/              # NEW optional out-of-process uv project for the generated source (own .python-version/uv.lock)
    ├── pyproject.toml
    └── src/voders_lyrics_backend/worker.py

tests/
├── contract/            # lyric-source interface; manifest record + config schema deltas
├── integration/         # US1 SVS-sings-lyrics; US2 automatic determinism; US3 provenance/license; US4 generated replay
└── unit/                # parse 4th column (from prototype test_lyrics.py); sampler; g2p mapping; cache; coverage

evals/
└── fixtures/            # lyrics-smoke.yaml + lyric-free, supplied-lyric scores + CV inventory
```

**Structure Decision**: Single Python project. The lyric layer is a new CPU-only `voders.lyrics`
package consumed by exactly one renderer (the SVS lane) plus the config/manifest models; the
deterministic and voice-conversion lanes are untouched (FR-006). The optional `generated` model is
isolated in its own `backends/lyrics/` uv project so no model/GPU dependency reaches the core
(mirroring the existing `backends/svs|rvc|seedvc` pattern). The `lyric` field and 4th-column parsing
land the `lyrics`-branch prototype onto this branch as the `supplied` source's foundation.

## Complexity Tracking

> No constitution violations. No entries.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | | |
