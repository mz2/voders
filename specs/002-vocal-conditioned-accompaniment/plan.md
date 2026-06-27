# Implementation Plan: Vocal-Conditioned Accompaniment Lane

**Branch**: `002-vocal-conditioned-accompaniment` | **Date**: 2026-06-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-vocal-conditioned-accompaniment/spec.md`

## Summary

Add an optional **accompaniment stage** to the existing corpus pipeline that takes an *accepted*
corpus vocal (audio + its correct-by-construction score) and lays instrumental backing under it using
an open-weight, vocal-conditioned music-generation model, **without moving the sung-note timing**.
Two modes:

- **Lego (vocal-preserving)** — the model generates an isolated accompaniment *stem* (a single
  instrument track) conditioned on the vocal; the stage sums it beneath the **untouched** vocal at a
  target vocal-to-accompaniment ratio. Timing is preserved by construction (the vocal waveform is
  never re-encoded). This reuses the machinery the existing `augmentation` lane already has for its
  synthetic `accompaniment_mix` step.
- **Complete (one-pass)** — the model emits a single full mix (vocal re-encoded with mild
  coloration). Admitted only if the sung notes still land within tolerance when measured on the mix.

The dominant integration insight: **the codebase already does the vocal-preserving pattern**.
`render/augment.py` carries accompaniment as a separate stem, mixes it under an untouched vocal at a
target signal-to-accompaniment ratio ("SNR"), and hands the *mix* plus that SNR to the validator,
which already gates SNR (masked vocal → `QUARANTINED`) and pitch/onset (→ `REJECTED`). So
"validate on the mix" and the masking-rejection behaviour from the spec are **existing mechanisms**;
this feature replaces the synthetic noise pad with a real neural generator behind a GPU-gated,
lazily-imported backend, and adds the provenance/stem-retention the spec requires.

The load-bearing risk is unchanged from the spec: a pop-trained generator imposing a metrical pulse
that displaces the sung notes. The mitigation is enforced, not hoped for — every admitted sample is
re-measured on the mix and rejected if any note moved (FR-004, FR-012).

## Technical Context

**Language/Version**: Python 3.14 (matches `001`; `requires-python >=3.14`).
**Primary Dependencies**:
- Reuse the `001` core (`numpy`, `scipy`, `soundfile`, `librosa`, `pydantic`, `pyyaml`) and the
  existing `validate` / `manifest` / `corpus` / `config` / `render` modules unchanged where possible.
- New **optional `accomp` extra** (GPU, lazily imported — the CPU baseline stays torch-free, FR-009):
  `torch>=2.12`, `torchaudio` (resampling + codec), and the ACE-Step 1.5 XL inference package /
  repo (the open-weight vocal-conditioned model — see research.md Decision 1). A source separator
  (`demucs`-class) is an *optional* sub-dependency used only for Complete-mode stem extraction and
  voice-in-mix validation fallback (research.md Decision 2).
- A **CPU fake backend** (torch-free, deterministic) implementing the same backend Protocol so unit
  tests, integration tests, and the smoke eval run in CI without a GPU.

**Storage**: Corpus audio stays **22,050 Hz mono float32** (`voders.constants.SAMPLE_RATE`,
Basic Pitch's required input format). The model runs at its native **48 kHz stereo**; output is
down-mixed + resampled to 22,050 mono before mixing/validation/storage (research.md Decision 3).
Per admitted sample the store keeps `mix.wav` and (Lego only) `accompaniment_stem.wav`, plus a
manifest reference to the source vocal sample id. Generated corpora and stems are **never committed**
(constitution: Binary Assets); fixtures go through Git LFS.

**Testing**: `pytest>=9.1` via `uv run pytest`. Lint/format: `ruff` (`uv run ruff check` /
`uv run ruff format`). Type-check: `mypy` (advisory). All Python actions go through `uv` (constitution
Python-Tooling gate).

**Target Platform**: Linux. The accompaniment stage runs on GPU (A6000 native; 3090 with CPU offload
+ quantization; the two DGX Spark machines). The CPU baseline (deterministic lane + validator +
fake accompaniment backend) needs no GPU.

**Project Type**: Single Python project — extends the existing `voders` library + CLI. No new service.

**Performance Goals**: With the stage enabled, a run still streams/checkpoints a 10k–100k corpus
without memory exhaustion (SC-007 — no regression). Accompaniment throughput target: **≥ 60 admitted
samples/hour per A6000-class GPU** for ≤ 30 s clips (to be confirmed by the benchmark in research.md
Decision 7; this is informational, not a merge gate).

**Constraints**: 22,050 Hz mono float32 corpus audio; onset within 50 ms, offset within
max(50 ms, 20% of note length); **no sung note may shift in time** (measured on the mix);
license policy = MIT / Apache-2.0 / CC-BY-class only (CC-BY-NC and closed excluded), attribution text
captured and propagated.

**Scale/Scope**: Inherits the 10k–100k run envelope; accompaniment is an opt-in subset fanned from
*accepted* base renders (corpus-internal vocals only — external vocals are out of scope, per spec).

### Evaluation Strategy *(Constitution Principle IV — eval-first)*

- **How to run it (single command):**
  `uv run voders run --config evals/fixtures/accompaniment-smoke.yaml`
  enables the accompaniment stage (both modes, using the **CPU fake backend** so CI needs no GPU) on
  the checked-in fixture vocals, and writes `mix.wav` + `accompaniment_stem.wav` + a JSONL manifest.
- **How to evaluate it (single command, yes/no verdict):**
  `uv run voders eval --manifest <out>/manifest.jsonl` — the eval harness is extended to report the
  accompaniment Success Criteria and exit non-zero on any gated failure:
  - **SC-001** Lego vocal byte-identical to the source vocal (retained vocal vs base render).
  - **SC-002** one-pass onsets ≤ 50 ms / offsets within tolerance on the mix ≥ 99%.
  - **SC-004** every admitted sample has complete model/license/attribution/mode/source/seed
    provenance; 0 disallowed-license samples; CC-BY-class samples carry attribution text.
  - **SC-005** zero measured note-timing shift on admitted samples; free-time samples carry no
    fixed-tempo metadata.
  - **SC-008** every admitted sample stores the mix (+ stem for Lego) sufficient to re-mix.
- **Fixture:** the consented donor vocal(s) + small scores already under `evals/fixtures/` (from
  `001`), plus one accompaniment smoke config. The **real ACE-Step backend** is exercised by a
  separate GPU-gated benchmark (research.md Decision 7), not in CPU CI.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|-----------|------|--------|
| I. Red-Green-Refactor TDD | `/speckit-tasks` emits a failing test per behaviour before impl; backend Protocol + CPU fake make the stage unit-testable without GPU | PASS — planned |
| II. Zero-Warning Linting | `ruff` + `ruff format` clean; new torch import is lazy/guarded, no blanket ignores | PASS — toolchain reused |
| III. Docs current with repo (+ README run/test) | Spec/plan/quickstart and `README.md` run/test sections updated in the same change set; new config keys documented in contracts | PASS — planned |
| IV. Evaluation-First | Runnable harness + runnable eval defined above, mapped to SC-001/002/004/005/008, before feature code | PASS — Evaluation Strategy captured |
| V. Clear, Audience-Aware Writing | Jargon glossed on first use (stem, Lego/Complete, DiT, VAE, source separation, SNR, rubato); no marketing language | PASS |
| Commit & Attribution Discipline | No AI-attribution trailers in commits/PRs | PASS — followed |
| Binary Assets & Large Files | Fixtures via LFS; generated mixes/stems never committed; model weights downloaded at runtime, not committed; `.gitattributes` already tracks `.wav`/weights | PASS |
| Python Tooling: uv | `accomp` extra added to `pyproject.toml`; `uv.lock` regenerated; all commands shown as `uv run …` | PASS — planned |

No violations. Complexity Tracking left empty.

## Project Structure

### Documentation (this feature)

```text
specs/002-vocal-conditioned-accompaniment/
├── plan.md              # This file
├── research.md          # Phase 0 — model selection + validation/format/repro decisions
├── data-model.md        # Phase 1 — entities, fields, lifecycle
├── quickstart.md        # Phase 1 — single-command run + eval
├── contracts/           # Phase 1 — backend Protocol, config + manifest additions
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/voders/
├── render/
│   ├── accompaniment.py          # NEW — Accompanist stage: orchestrates mode (Lego/Complete),
│   │                             #       takes, 48k↔22.05k bridging, stem retention, SNR mixing
│   ├── accompaniment_backend.py  # NEW — AccompanimentBackend Protocol + deterministic CPU fake
│   │                             #       (torch-free; the load-bearing testable baseline)
│   └── backends/
│       └── acestep.py            # NEW — GPU ACE-Step 1.5 XL backend (lazy torch; `accomp` extra)
├── validate/
│   └── mix.py                    # NEW — voice-in-mix note-shift measurement (FR-012) + the
│                                 #       vocal-preserving shortcut; wraps existing Validator
├── corpus/store.py               # EXTEND — write mix + accompaniment stem + source-vocal ref
├── manifest/models.py            # EXTEND — AccompanimentProvenance on ProvenanceRecord (FR-007/006a)
├── config/models.py              # EXTEND — documented accompaniment options (LaneToggle is extra=allow)
└── corpus/orchestrator.py        # EXTEND — fan accepted base renders through the accompaniment stage
                                  #          (mirrors the existing augmentation fan-out block)

tests/
├── contract/                     # backend Protocol conformance; manifest/config schema additions
├── integration/                  # end-to-end per user story (P1 Lego, P2 Complete, P3 free-time, P4 audit)
└── unit/                         # accompaniment stage, mix note-shift, format bridging, take selection

evals/
└── fixtures/
    └── accompaniment-smoke.yaml  # NEW — enables the stage (fake backend) on existing donor vocals
```

**Structure Decision**: Single Python project, extending `voders`. The accompaniment capability is a
**post-acceptance stage**, not a from-score `RendererLane`: it consumes an *accepted base render*
(audio + score) and emits new samples — exactly the shape of the existing augmentation fan-out in
`orchestrator.run()`. It therefore plugs in beside the augmentor rather than into
`LANE_VOICE_KINDS`. The neural model sits behind an `AccompanimentBackend` Protocol with a torch-free
CPU fake (CI/tests) and a GPU ACE-Step implementation (lazy import, `accomp` extra) so the CPU
baseline never imports torch (FR-009), mirroring how `001` isolates its neural lanes.

## Complexity Tracking

> No constitution violations. No entries.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | | |
