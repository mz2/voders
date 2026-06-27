# Implementation Plan: Synthetic Singing Corpus Generator

**Branch**: `001-synthetic-singing-corpus` | **Date**: 2026-06-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-synthetic-singing-corpus/spec.md`

## Summary

Build a Python pipeline that turns folders of singing scores — each a list of
`(onset, offset, MIDI pitch)` note rows — into a large training corpus of `(audio.wav, score.tsv)`
pairs whose labels are correct by construction. The pipeline has four independently-toggled
renderer lanes (deterministic f0-driven, expressive neural singing-voice-synthesis, voice
conversion for timbre, and augmentation/mix), an alignment validator that gates every sample, and
a JSON Lines manifest that records full provenance for replay and license audit. The dominant risk
is *alignment drift* — rendered note boundaries straying outside the challenge's tolerance — so the
deterministic lane (pitch derived directly from the score) is the load-bearing baseline and every
other lane is validated against the same tolerance before admission.

A single declarative YAML config file specifies a run (score set, voice pool, augmentation
profiles, master seed, lane toggles); a single run-level master seed derives every per-sample seed
so any one sample is reproducible in isolation. Runs stream and checkpoint to handle
10,000–100,000 samples without exhausting memory.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**:
- Core (CPU): `numpy`, `scipy`, `soundfile`, `librosa`, `pyworld` (WORLD vocoder — a
  classic analysis/resynthesis vocoder that lets us replace a voice's pitch with a score-derived
  contour), `pretty_midi` / `music21` (convert `.tsv` scores to MIDI/MusicXML), `pyyaml`,
  `pydantic` (config + manifest schema validation).
- Validator: `torchcrepe` (CREPE — a neural fundamental-frequency / "f0" estimator) with a
  `pyin` (`librosa`) CPU fallback.
- Augmentation: `pedalboard` (Spotify's DSP plugin host — convenient since the downstream judge is
  Spotify's model), `audiomentations`, `torchaudio` (codec round-trips).
- Optional GPU lanes (PyTorch): DiffSinger via OpenUTAU, NNSVS (neural singing-voice synthesis);
  RVC and so-vits-svc (voice conversion); Montreal Forced Aligner / "MFA" (forced alignment — lines
  text/phonemes up against audio to re-derive note boundaries).

**Storage**: Local disk. Accepted corpus written to a sharded directory tree; provenance to a JSON
Lines manifest (one record per line, append-only). Non-accepted samples (provenance + audio)
retained in a separate `rejected/` tree. The rendered corpus audio is never committed to Git; the
run's end-result record (final manifest + resolved config + stats report) is text and is committable.
**Testing**: `pytest`. Lint/format: `ruff` (lint) + `ruff format`. Type-check: `mypy` (advisory).
**Target Platform**: Linux. Deterministic lane + validator run on commodity 4-core CPU, no GPU;
the neural-SVS and voice-conversion lanes run on GPU (two DGX Spark machines available).
**Project Type**: Single Python project — a CLI-driven data pipeline (library + thin CLI).
**Performance Goals**: Deterministic render+validate ≥100 score-singer pairs/hour on a 4-core CPU
(SC-005). A run produces 10,000–100,000 samples while streaming/checkpointing (SC-011).
**Constraints**: Audio is 22,050 Hz mono float32 (Basic Pitch's required input format). Onset
within 50 ms; offset within max(50 ms, 20% of note length). Deterministic lane MUST need no GPU.
**Scale/Scope**: 10k–100k samples per run; ≥1,000 distinct timbre identities (SC-004).

### Evaluation Strategy *(Constitution Principle IV — eval-first)*

- **How to run it (single command):** `python -m voders.cli run --config evals/fixtures/smoke.yaml`
  renders the checked-in fixture scores end-to-end through the enabled lanes and writes audio + a
  JSONL manifest to an output dir.
- **How to evaluate it (single command, yes/no verdict):**
  `python -m voders.cli eval --manifest <out>/manifest.jsonl` runs the alignment validator over the
  produced corpus and prints a pass/fail table mapped to the spec's Success Criteria:
  - onset within 50 ms ≥99% (SC-001), offset within tolerance ≥99% (SC-002),
  - first-attempt validator pass ≥95% (SC-007), zero `consent_verified=false` voices (SC-008),
  - per-sample isolated reproducibility within tolerance (SC-009).
  Exit code is non-zero if any gated criterion fails. SC-003 (downstream note-F1 ≥0.15) is a
  consumer-side metric (training is out of scope per the spec) and is reported as an *informational*
  number when a Basic Pitch eval harness is available, not gated here.
- **Fixture:** ~10 small public scores under `evals/fixtures/scores/`, plus one synthetic donor
  voice with `consent_verified=true`, kept small and tracked via Git LFS.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|-----------|------|--------|
| I. Red-Green-Refactor TDD | Every behavioural change starts as a failing test; `/speckit-tasks` emits test tasks per story before impl | PASS — planned |
| II. Zero-Warning Linting | `ruff` + `ruff format` clean on every commit/CI | PASS — toolchain chosen |
| III. Docs current with repo | Spec/plan/quickstart updated in same change set as code | PASS |
| IV. Evaluation-First | Runnable harness + runnable eval defined above, before feature code, mapped to SC-001/002/007/008/009 | PASS — Evaluation Strategy captured |
| V. Clear, Audience-Aware Writing | ML jargon glossed on first use (f0, WORLD, CREPE, MFA, SVS); no marketing language | PASS |
| Binary Assets & Large Files | Fixtures via Git LFS; generated corpora never committed; `.gitattributes` already tracks `.wav`/weights | PASS |
| Commit & Attribution Discipline | No AI-attribution trailers in commits/PRs | PASS — followed |

No violations. Complexity Tracking table left empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-synthetic-singing-corpus/
├── plan.md              # This file
├── research.md          # Phase 0 output — resolves deferred toolkit/method choices
├── data-model.md        # Phase 1 output — entities, fields, lifecycle
├── quickstart.md        # Phase 1 output — single-command run + eval
├── contracts/           # Phase 1 output — config schema, manifest record, lane interface, CLI
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/voders/
├── __init__.py
├── config/              # Run Config: load/validate YAML, resolve, hash (FR-016)
├── scores/              # parse .tsv scores, validate monophony, convert to MIDI/MusicXML (FR-001)
├── seeds.py             # master-seed → per-sample/per-stage derivation (FR-013)
├── render/
│   ├── base.py          # RendererLane interface/contract (FR-015)
│   ├── deterministic.py # score-f0 → WORLD/NSF/DDSP vocal excitation (FR-003) — CPU, no GPU
│   ├── svs.py           # neural SVS lane (DiffSinger/NNSVS) with force-score-F0 mode (FR-007)
│   ├── voiceconv.py     # RVC / so-vits-svc timbre fan-out, auto_predict_f0=False (FR-004)
│   └── augment.py       # label-safe augmentation chain (FR-005, FR-014)
├── validate/            # alignment validator: f0 + onset/offset vs score; verdicts (FR-006)
├── manifest/            # JSONL provenance writer/reader, license audit, stats (FR-008/011/012)
├── corpus/              # sharded on-disk layout, accepted vs rejected trees, export (FR-006a)
├── voices/              # voice enrollment: license string + consent_verified flag (FR-011)
└── cli/                 # `run`, `eval`, `audit`, `stats` subcommands

tests/
├── contract/            # config schema, manifest record, lane interface, CLI contracts
├── integration/         # per-user-story end-to-end (P1..P5)
└── unit/                # scores, seeds, validator, manifest

evals/
├── fixtures/            # small LFS-tracked scores + one synthetic consented donor voice
└── run_eval.py          # SC-mapped pass/fail harness (wrapped by `voders.cli eval`)
```

**Structure Decision**: Single Python project (library `src/voders/` + thin CLI). The four
renderer lanes live behind one `RendererLane` interface (`render/base.py`) so each lane is
enabled/disabled/replaced independently (FR-015). CPU-only modules (`scores`, `seeds`,
`render/deterministic`, `validate`, `manifest`, `corpus`) carry no GPU/PyTorch import at module load
so the deterministic lane and validator install and run on a laptop (FR-009); GPU lane dependencies
are optional extras imported lazily.

## Complexity Tracking

> No constitution violations. No entries.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | | |
