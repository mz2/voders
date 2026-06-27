# Phase 1 Data Model: Vocal-Conditioned Accompaniment Lane

Extends the `001` data model. New/changed types only; everything else is reused unchanged
(`Score`, `Note`, `Voice`, `ValidationVerdict`, `CorpusStore` layout, `ManifestWriter`).

---

## 1. AccompanimentConfig (run config)

Carried as free-form options on the existing `LaneToggle` (which is `extra="allow"`), so enabling the
stage is **non-breaking**. Documented shape (contract: `contracts/run-config-accompaniment.md`):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `enabled` | bool | `false` | Standard `LaneToggle` flag (FR-008). |
| `mode` | `"lego" \| "complete"` | `"lego"` | Vocal-preserving stem-sum vs one-pass full mix. |
| `backend` | `"fake" \| "acestep"` | `"fake"` | CPU deterministic fake (CI/tests) vs GPU ACE-Step. |
| `model_id` | str | `""` | Backend resolves the concrete checkpoint (e.g. `ace-step-1.5-xl-base`). |
| `license_policy` | list[str] | `["MIT","Apache-2.0","CC-BY-4.0"]` | Allow-set; disallowed → stage no-ops (FR-006). |
| `target_instrument` | str | `"sustained pad"` | Caption hint; sustained/textural preferred (FR-005). |
| `free_time` | bool | `true` | No BPM emitted; free/rubato captioning (FR-005). |
| `bpm` | float \| null | `null` | MUST stay null when `free_time` (FR-005). |
| `takes` | int | `1` | Max takes generated; best-aligned admitted (FR-010, Decision 6). |
| `target_snr_db` | float \| list[float] | `12.0` | Vocal-to-accompaniment ratio for Lego mixing. |

**Validation rules**: `free_time=true ⟹ bpm is null`; `takes ≥ 1`; `mode="complete" ⟹ backend
supports Complete` (else stage no-ops with a logged skip).

---

## 2. AccompanimentBackend (interface)

Protocol (contract: `contracts/accompaniment-backend.md`), implemented by the CPU **fake** and the GPU
**ACE-Step** backend.

```text
AccompanimentBackend (Protocol)
  name: str
  model_license: str          # declared license string, checked against license_policy (FR-006)
  attribution_text: str | None # required CC-BY-class credit, propagated to provenance (FR-006a)
  requires_gpu() -> bool       # fake -> False; acestep -> True
  supports(mode) -> bool       # "lego" and/or "complete"
  generate(vocal, score, mode, seed, options) -> BackendOutput
```

`BackendOutput`:
| Field | Type | Notes |
|-------|------|-------|
| `accompaniment_stem` | ndarray \| null | 22,050 mono float32 (Lego). `null` for Complete (no separable stem — research.md Decision 1/§2). |
| `mix` | ndarray \| null | 22,050 mono float32 full mix (Complete). `null` for Lego (the stage mixes). |
| `native_sr` | int | e.g. 48000 — for provenance only; output already bridged to 22,050 (Decision 3). |

---

## 3. AccompanimentProvenance (manifest)

New optional sub-model on `ProvenanceRecord` (which is `extra="forbid"`, so the field is added
explicitly). Present iff `lane == "accompaniment"` (contract:
`contracts/manifest-accompaniment.md`). Satisfies FR-007 / FR-006a.

| Field | Type | Maps to |
|-------|------|---------|
| `mode` | `"lego" \| "complete"` | conditioning mode |
| `model_id` | str | model/weights identity |
| `model_version` | str | model/weights version |
| `model_license` | str | license (FR-006) |
| `attribution_text` | str \| null | required credit for CC-BY-class (FR-006a) |
| `vocal_bit_exact` | bool | `true` for Lego, `false` for Complete (SC-001) |
| `source_vocal_sample_id` | str | the accepted base render this was layered on |
| `takes_tried` | int | FR-010 / Decision 6 |
| `max_note_shift_ms` | float | measured note displacement (FR-012 / SC-005) |
| `free_time` | bool | FR-005 |
| `target_instrument` | str | FR-005 |
| `stem_available` | bool | `true` Lego, `false` Complete; SC-008 re-mixability is Lego-scoped (resolved) |

`ValidationVerdict` is reused as-is; `max_onset_dev_ms` already carries onset deviation, and
`snr_db` already carries the vocal-to-accompaniment ratio. `max_note_shift_ms` lives on the
provenance sub-model (it is a property of the accompaniment step, not the generic verdict).

---

## 4. Accompaniment-Augmented Sample (stored artifacts)

Per admitted sample, `CorpusStore` is extended to write, under the sample's shard:
- `mix.wav` — 22,050 mono float32 (the corpus audio; what the score labels).
- `accompaniment_stem.wav` — **Lego only**; enables label-safe re-mixing (FR-015, SC-008).
- `score.tsv` — **byte-identical** to the source vocal's score (labels unchanged; the `.tsv` is the
  source render's `raw_bytes`).
- manifest `source_vocal_sample_id` — reference to the unchanged source vocal sample (FR-015).

**Sample id**: `f"{base_sample_id}_accomp_{mode}"` (mirrors the augmentation `_aug_{profile}` scheme),
keeping ids unique and traceable to the base render.

---

## 5. Lifecycle / state

```text
accepted base render (audio + score)         # corpus-internal vocal only (spec scope)
   └─ for each enabled accompaniment config:
        generate up to N takes (backend)      # Lego: stem | Complete: mix  (Decision 1)
        bridge 48k stereo -> 22,050 mono      # Decision 3
        Lego: mix stem under untouched vocal at target SNR   # reuse augment.py machinery
        validate on the mix (+ snr) -> verdict; measure note shift (Decisions 2, 5)
        admit best-aligned passing take       # Decision 6
          ├─ ACCEPTED  -> write mix (+stem) + provenance to accepted tree + manifest
          └─ none pass -> REJECTED/QUARANTINED -> rejected tree + manifest (with reason)
```

Reproducibility: neural tier — same verdict on replay (Decision 4). License disallowed → stage
no-ops for the run with a logged skip (FR-006/FR-013), nothing admitted.

---

## Design note — resolved (2026-06-27)

FR-015 / SC-008 require storing "the separately generated accompaniment stem". In **Complete**
(one-pass) mode the model emits a single fused mix with **no separable stem**, so those clauses are
satisfiable only for **Lego** mode. **Resolution (spec owner): re-mixability is a Lego-mode
guarantee.** Complete samples store the mix only and record `stem_available = false`; FR-015 and SC-008
are scoped to Lego accordingly (see the spec Clarifications). The lossy source-separation fallback on
Complete output (Decision 2) is intentionally *not* applied. The eval (`acc_sc008_stems`) gates SC-008
over Lego samples only, matching this scope.
