# Phase 0 Research: Vocal-Conditioned Accompaniment Lane

Resolves the deferred (plan-level) decisions from the spec: concrete model selection, the
voice-in-mix validation mechanism, sample-rate/channel bridging, reproducibility tier, note-shift
measurement, take selection, and the runtime/packaging story. Source for the model survey is the
user's June 2026 "Open-Weight Models for Vocal-Conditioned Accompaniment" report; claims below are
attributed to it and **must be re-verified against the model's repo/model-card and `generate_music.py`
at implementation time** (the report flags some conclusions as partly inferential).

---

## Decision 1 — Accompaniment model: ACE-Step 1.5 XL (`xl-base`)

**Decision**: Use **ACE-Step 1.5 XL-Base** (4B-parameter Diffusion Transformer, "DiT" — a
transformer that denoises audio latents — over a 48 kHz stereo 1-D VAE). Use the **Base** checkpoint
(`xl-base`), which exposes the editing tasks we need (**Lego** = generate one isolated stem
conditioned on the audio; **Complete** = condition on the vocal and decode a full mix). Turbo/SFT
checkpoints do not expose these modes.

**Rationale**:
- It is the only surveyed open-weight model that **conditions on the vocal as temporal input** and so
  holds the sung-note timing in place — Lego by construction (vocal untouched), Complete by decoding
  the vocal latent in place.
- License fits the policy (Decision 8): the 1.5 repo is reported **MIT**, the original ACE-Step
  **Apache-2.0** — both commercial-OK with at most attribution. (Verify the LICENSE of the exact
  checkpoint before shipping — report caveat.)

**Alternatives considered (rejected)**:
- **Stable Audio 3** (open Small/Medium, May 2026) — wrong paradigm: audio-to-audio *transforms* the
  upload (would corrupt the vocal), inpainting regenerates a *masked time region*, and it does not
  model vocals or accept stem conditioning. Only usable as a hand-aligned bed generator → no temporal
  lock → reintroduces grid risk. Rejected.
- **Magenta RealTime / RT2** — CC-BY-class weights (license OK) but **style-embedding only**, no
  external vocal-stem conditioning → cannot hold the vocal. Rejected on capability.
- **MusicGen-melody / JASCO** — **CC-BY-NC** (non-commercial) → excluded by the license policy; also
  chroma/timbre-blurred conditioning (not onset-locked). Rejected on license + capability.
- **YuE / DiffRhythm** — regenerate vocals from lyrics/style; not vocal-preserving. Rejected.
- **SingSong / Diff-A-Riff** — ideal design (instrumental-from-vocal) but weights never released.
  Rejected (not open-weight).

**Mode mapping**:
- **Lego → vocal-preserving (P1, primary)**: model emits an accompaniment stem; we sum it under the
  untouched vocal. Best fidelity; caveat — ACE-Step issue #369 reports occasional stems "unrelated to
  the source", so generate several takes and admit the best-aligned (Decision 6).
- **Complete → one-pass (P2)**: model emits the full mix; the vocal is re-encoded through the VAE
  (mild coloration), onsets held. Report notes the "Complete re-decodes the vocal" behaviour is
  *partly inferential* — verify in `generate_music.py` for the pinned release.

---

## Decision 2 — Voice-in-mix validation mechanism

**Decision**: Validate **per mode**, both ending on the existing `Validator` and its SNR gate:
- **Lego**: the vocal is byte-identical to the accepted base render, so its note timing is already
  validated. The new check is **audibility in the mix**: compute the vocal-to-accompaniment ratio
  (the augmentor already reports this `snr_db`) and pass the *mix* + `snr_db` to `Validator.validate`.
  A masked vocal trips the SNR floor (`QUARANTINED`) or fails pitch coverage (`REJECTED`).
- **Complete**: there is no separate vocal stem, so run a **source separator** (a `demucs`-class model
  that splits a mix into vocal/accompaniment stems) to recover an estimated vocal, then run the
  existing `Validator` (pyin/CREPE pitch + onset/offset) on that estimate and measure note shift
  (Decision 5). Direct pyin on the raw polyphonic mix is the fallback when separation is unavailable,
  accepting a higher false-reject rate.

**Rationale**: Reuses the entire existing validator (pitch/onset/offset/SNR/clipping gates) rather
than inventing a parallel gate. The Lego path needs *no* new pitch tracking — only the SNR the
augmentor already computes. Separation is confined to Complete, where it is unavoidable.

**Alternatives considered**: (a) run pyin/CREPE directly on every mix — simplest but pyin tracks the
loudest periodicity and is unreliable on dense accompaniment → false rejects; kept only as fallback.
(b) Train a mix-robust tracker — out of scope.

---

## Decision 3 — Sample-rate / channel bridging (48 kHz stereo ↔ 22,050 mono)

**Decision**: Generate at the model's native **48 kHz stereo**, then **down-mix to mono and resample
to 22,050 Hz** (`librosa.resample` / `torchaudio`) *before* mixing, validation, and storage. Stored
artifacts (`mix.wav`, `accompaniment_stem.wav`) are 22,050 Hz mono float32, matching the corpus
contract and Basic Pitch's required input. For **Lego**, only the *generated stem* is resampled; the
original vocal is used at its stored 22,050 mono rate and never re-encoded (preserves SC-001
bit-exactness).

**Rationale**: The whole corpus and the downstream judge are 22,050 mono; keeping the corpus contract
unchanged means the existing store/validator/eval work without modification. Resampling the stem (not
the vocal) keeps the vocal byte-identical in Lego mode.

**Alternatives considered**: storing 48 kHz stereo corpus audio — rejected: breaks the Basic Pitch
input contract and the existing store/validator assumptions for a benefit (stereo) the transcription
task does not use.

---

## Decision 4 — Reproducibility tier

**Decision**: The accompaniment stage is **neural/GPU**, so it uses the existing **`NeuralReproduction`
tier** (`config.models.ReproductionConfig.neural`): same admission verdict + f0/onset within tolerance
on replay, **not** bit-exact (GPU diffusion is non-deterministic). The seed derivation reuses
`voders.seeds.sample_seed(master_seed, score_id, voice_id, f"accompaniment_{mode}_{take}")`.

**Rationale**: Matches how `001` already classifies its neural SVS / voice-conversion lanes (SC-009);
no new reproducibility concept. The CPU fake backend *is* bit-exact (deterministic from seed), so
unit tests can assert exact arrays.

---

## Decision 5 — Measuring note-timing shift (FR-012 / SC-005)

**Decision**: Reuse the existing **label re-derivation** approach (`validate/rederive.py`,
`derive_labels`) — which already measures where each note actually lands in rendered audio and returns
`max_onset_dev_ms` — applied to the **validated audio** (the mix for Lego audibility; the separated
vocal estimate for Complete). The per-note shift is `|rederived_onset − original_onset|`; admission
requires it to be within the validator tolerance (target: zero). Record `max_note_shift_ms` per
sample.

**Rationale**: `derive_labels` already exists and returns exactly this quantity; the SVS lane already
uses it. No new DSP. Free-time mitigation (no BPM, sustained instrument captions) is an *input* to the
backend (Decision 1 captions), but the *gate* is this measured shift, per the clarification.

---

## Decision 6 — Take selection (FR-010)

**Decision**: Generate up to **N takes** (config `takes`, default 1) per (vocal × mode × target
instrument). Each take is validated; among the **validator-passing** takes, admit the **single
best-aligned** one — smallest `max_note_shift_ms`, tie-broken by highest mix SNR. No separate
relevance/coherence scoring (per clarification). Record `takes_tried`. If every take fails, the
sample is rejected with the reason and routed to the rejected tree.

**Rationale**: Directly encodes the two clarifications (validator-only gate + one-best-take) using
quantities the validator already produces (`max_onset_dev_ms`, `snr_db`).

---

## Decision 7 — Runtime, VRAM, throughput

**Decision**: Run `xl-base` (4B DiT) **native on the A6000** (≥ 20 GB VRAM) and with **CPU offload +
quantization on the 3090** (≥ 12 GB with offload), per the report. Benchmark throughput on a GPU-gated
script (not CI) and record it; the planning target is **≥ 60 admitted samples/hour per A6000-class
GPU** for ≤ 30 s clips. This is informational (SC-007 only gates "no streaming/memory regression",
which the stage satisfies because it processes one accepted render at a time and writes incrementally
through the existing `CorpusStore`/`ManifestWriter`).

**Rationale**: The report's hardware notes; the corpus already streams/checkpoints, so the stage
inherits that envelope. Throughput is a benchmark, not a merge gate, to avoid coupling CI to GPU.

---

## Decision 8 — Packaging & license enforcement

**Decision**:
- Add an optional **`accomp`** extra to `pyproject.toml` (`torch`, `torchaudio`, the ACE-Step package,
  optional `demucs`-class separator), regenerate `uv.lock`, and **lazily import** it only inside the
  GPU backend so `uv sync` (no extra) keeps the CPU baseline torch-free (FR-009, FR-013).
- Enforce the **license policy** in code: a configured allow-set (default `{MIT, Apache-2.0,
  CC-BY-4.0}` and CC-BY variants) is checked against the backend's declared `model_license` *before*
  generation; a disallowed license makes the stage **no-op for that run with a logged skip** (FR-006,
  FR-013), and never admits output. For CC-BY-class models the backend supplies `attribution_text`,
  which is copied into every sample's provenance (FR-006a) and surfaced by `voders audit`.

**Rationale**: Mirrors the `001` `cpu`/`gpu` extras split and the consent-gate-before-render pattern
in the orchestrator. License is checked before spending GPU time.

---

## Open verification items (GPU run — backend is implemented, not yet hardware-verified)

The `AceStepBackend` (`render/backends/acestep.py`) is a **full implementation** against the real
ACE-Step + Demucs APIs (no stub / `NotImplementedError`): caption construction, 22.05↔48 kHz
bridging, mode dispatch, and the Demucs stem-extraction path for Lego are concrete and CPU-unit-
tested via injected fakes; only the two library calls need a GPU box to validate. Resolve these on
that run and adjust the two adapter call-sites if a signature differs:

1. Confirm the exact `xl-base` checkpoint **LICENSE** (the backend declares `model_license="MIT"`)
   and that the `ACEStepPipeline(...)` call args (`audio2audio_enable` / `ref_audio_strength` /
   `manual_seeds`) match the pinned release's `generate_music.py`.
2. **Complete** emits a fused mix with no separable stem, so FR-015 stem retention / SC-008
   re-mixability are **Lego-mode properties** — Complete samples set `stem_available=false`
   (implemented). Revisit only if strict Complete-stem retention is required.
3. Confirm the ACE-Step package name / `pipeline_ace_step` import path and the Python 3.14 wheel for
   `acestep` + `demucs`; build from source if no wheel (same risk noted for `001` niche audio deps).
   `demucs>=4.0` is pinned in the `accomp` extra; `acestep` is installed out-of-band.
