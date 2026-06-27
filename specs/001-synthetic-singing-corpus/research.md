# Phase 0 Research: Synthetic Singing Corpus Generator

Resolves the items the spec deferred to planning: per-lane toolkit selection, the validator's f0
estimation method, label re-derivation for the expressive lane, the augmentation toolset, lyrics
handling, and the corpus export layout. Choices are grounded in `research/deep-research-report.md`
(cited once here).

## Decision 1 — Deterministic lane synthesizer (the load-bearing baseline)

**Decision:** Build the deterministic lane on the **WORLD vocoder** via `pyworld`: decompose a donor
vowel/voice into pitch (f0), spectral envelope, and aperiodicity; replace f0 with a step/glide
contour constructed directly from the score; resynthesize. Swapping the spectral envelope across
donor vowels/voices multiplies timbre for free.

**Rationale:** The ground truth is the score itself — pitch and onsets are exact by construction, so
this lane cannot drift. It runs fast on CPU (meets FR-009 and SC-005) and is the most defensible
lane under time pressure (deep-research report, Approach B / Stage 1).

**Alternatives considered:** Neural Source-Filter (NSF) / DDSP vocoders are also f0-conditioned and
higher quality but pull in PyTorch and GPU; kept as optional drop-in replacements behind the same
lane interface, not the default. Pure sine/vowel excitation is the cheapest fallback for crude but
perfectly-labeled timbres.

## Decision 2 — Validator f0 / onset estimation

**Decision:** Measure rendered pitch with **`torchcrepe`** (CREPE neural f0 estimator) and fall back
to **`pyin`** (`librosa`) when no GPU is present. Compare measured f0 to the score per note
(±25 cents over ≥80% of the sustained interval, per acceptance scenario) and estimate onset/offset
from the audio for the expressive lane's re-derivation.

**Rationale:** CREPE is the standard accurate f0 estimator; `pyin` keeps the validator CPU-only so
it ships with the deterministic lane. The validator is the single gate every sample passes (FR-006),
so it must run without a GPU.

**Alternatives considered:** `pyworld`'s f0 (Harvest/DIO) — fast but less accurate at note
boundaries; usable as a second fallback. `praat-parselmouth` — accurate but a heavier dependency.

## Decision 3 — Expressive neural-SVS lane + label re-derivation

**Decision:** Use **DiffSinger** (batch-driven through **OpenUTAU**) as the primary expressive lane,
with **NNSVS** as an alternative behind the same interface. Offer two safety modes (FR-007):
(a) **force-score-F0** — feed the score f0 into the vocoder, behaving like the deterministic lane
with better timbre; (b) **re-derive labels** — run **Montreal Forced Aligner (MFA)** plus onset
detection on the actually-rendered audio and carry the re-derived `.tsv`, rejecting any sample whose
onsets deviate >50 ms (SC-010).

**Rationale:** Expressive synthesis adds vibrato/portamento that threatens the 50 ms tolerance; the
two modes make drift either impossible (mode a) or measured-and-gated (mode b). This lane is P4 and
optional, so it depends on GPU infrastructure being available.

**Alternatives considered:** VISinger 2 / Sinsy (Sinsy takes MusicXML directly and is deterministic
but older). Kept as future lane plug-ins.

## Decision 4 — Voice-conversion (timbre) lane

**Decision:** Use **RVC** and **so-vits-svc** for timbre fan-out, always with
**`auto_predict_f0=False`** so the score-aligned f0 propagates unchanged through conversion. Fan one
accepted render out to N enrolled voices, each producing a `score_NNN_singer_X` pair sharing the
source score row-for-row (FR-004, US2).

**Rationale:** Timbre diversity is the highest-ROI lever for vocal targets (Sato & Akama 2024), and
disabling f0 re-prediction preserves alignment for free. GPU lane.

**Alternatives considered:** None preferred; both are validated against the same tolerance after
conversion, so a failed conversion is caught by the gate rather than trusted.

## Decision 5 — Augmentation / domain-randomization toolset

**Decision:** Build the label-safe augmentation chain on **`pedalboard`** (reverb, EQ, compression,
codec-like effects), **`audiomentations`** (pitch/time perturbation, noise), and **`torchaudio`**
(MP3/Opus codec round-trips). Mixing with operator-supplied backing tracks at swept
signal-to-accompaniment ratios; convolution with room impulse responses. All transforms leave the
score's note rows untouched (FR-005); samples whose vocal level drops below the configured floor are
quarantined (FR-014).

**Rationale:** Basic Pitch is instrument-agnostic and trained in-the-mix, so in-the-mix augmentation
is the highest-value domain-gap closer (deep-research report, Stage 3). `pedalboard` is Spotify's own
DSP host — apt since the judge is Spotify's model.

**Alternatives considered:** `torch-audiomentations` (GPU batch) — optional speed-up; generative
accompaniment (ACE-Step) supported but not required (operator supplies licensed backing tracks).

## Decision 6 — Lyrics handling

**Decision:** Lyrics are optional. When omitted, the deterministic and SVS lanes default to a neutral
open vowel ("ah"). When provided, the operator owns syllable-to-note alignment; the system surfaces
count mismatches in provenance but does not autocorrect beyond the vowel fallback. Phonemization for
the SVS lane uses `phonemizer` (espeak-ng) / OpenUTAU's built-in grapheme-to-phoneme.

**Rationale:** Matches the spec's lyrics assumption; keeps the deterministic baseline lyric-free and
trivially reproducible.

## Decision 7 — Corpus export layout (the one remaining interface detail)

**Decision:** A run writes to a single output root:

```text
<out>/
├── config.resolved.yaml      # fully-resolved Run Config (committable end-result record)
├── manifest.jsonl            # accepted-sample provenance, one JSON object per line
├── stats.json                # aggregate corpus statistics (FR-012)
├── corpus/                   # accepted training pairs, sharded
│   └── shard=000/ … shard=NNN/
│       ├── score_000123_singer_A.wav
│       └── score_000123_singer_A.tsv
├── rejected/                 # non-accepted samples (audio + per-sample reason), not for training
│   └── shard=000/ …
└── checkpoints/              # streaming resume state — git-ignored scratch
```

Audio is 22,050 Hz mono float32 WAV; each `.wav` has a sibling `.tsv` with byte-identical note rows
to the driving (or re-derived) score. Shards cap the file count per directory for the 10k–100k-sample
scale (SC-011). The downstream consumer reads `corpus/**/*.{wav,tsv}` directly; `manifest.jsonl` +
`config.resolved.yaml` are the replayable record.

**Rationale:** A flat-within-shards `(wav, tsv)` convention is exactly the challenge's expected input
and the simplest thing the consumer can glob. Separating `corpus/` from `rejected/` keeps forensic
retention (FR-006a) out of the training path. Checkpoints are scratch, so they are git-ignored and
never committed.

**Alternatives considered:** WebDataset/tar shards or Parquet — better for remote object storage but
add packaging the consumer doesn't need for the first version (managed cloud storage is out of scope).

## Resolved unknowns

| Deferred item (from spec) | Resolution |
|---------------------------|------------|
| Deterministic lane synthesizer | WORLD/`pyworld`, score-derived f0 (NSF/DDSP optional) |
| Validator f0 method | `torchcrepe` (GPU) / `pyin` (CPU fallback) |
| Expressive SVS toolkit + re-derivation | DiffSinger via OpenUTAU + MFA; NNSVS alternative |
| Voice-conversion toolkit | RVC + so-vits-svc, `auto_predict_f0=False` |
| Augmentation libraries | `pedalboard` + `audiomentations` + `torchaudio` |
| Lyrics / G2P | optional; open-vowel default; `phonemizer`/OpenUTAU G2P |
| Corpus export layout | sharded `corpus/` + `rejected/`, `manifest.jsonl`, resolved config, stats |

No NEEDS CLARIFICATION remain.
