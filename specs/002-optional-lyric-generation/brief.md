# Feature Brief: Optional Lyric Generation & Phonetic Diversity

**Feature**: `002-optional-lyric-generation`
**Status**: Brief (pre-spec) — scoping document, not yet a `/speckit-specify` spec
**Depends on**: `001-synthetic-singing-corpus` (consumes its existing lanes, manifest, validator)
**Created**: 2026-06-27

> This is a **detailed brief of *how* lyric generation would factor into the pipeline**, not a
> committed spec. It exists so the design can be reviewed before any code or `Note`-model change is
> made. Everything here is **strictly optional and opt-in**; the default corpus stays vowel-only.

---

## 1. Why this is a separate feature, not a 001 change

001's job is *label-correct* `(audio, score)` pairs for a **note/pitch transcription model**
(Basic Pitch-class). That consumer learns **onsets, pitches, offsets — not words**. So lyrics carry
**zero label value** and are deliberately out of 001's core:

- The `Note` model is purely `(onset_s, offset_s, pitch_midi)`; the `.tsv` parser reads exactly
  those three columns. There is **no lyric/text field today**, so lyrics cannot even be *ingested*.
- Every lane synthesizes a fixed open vowel ("ah") from the donor timbre: the deterministic WORLD
  lane, the voice-conversion lane (operates on that audio), and both SVS backends (the CPU WORLD
  stand-in and the `nnsvs` worker). No phonemes, no words.
- No G2P/phonemizer is wired. 001 research **Decision 6** named `phonemizer`/espeak-ng + OpenUTAU
  G2P as the *future* path; none of it is implemented.

Lyric generation is therefore a **realism / domain-randomization lever** — a sibling of 001's P2
(timbre multiplication) and P3 (production augmentation) — that *produces an optional input* the
pipeline can consume. Keeping it in 002 preserves 001's two load-bearing guarantees:

- **CPU-only baseline (FR-009)** — no GPU/LLM dependency creeps into the deterministic lane.
- **Bit-exact reproducibility (FR-013, SC-009)** — no non-deterministic generator in the core path.

**Hard constraint from the operator:** there is **no theme/lyric data in the annotations**. Any
theming is generated, never ingested. 002 must never assume the score set carries lyrics or themes.

---

## 2. What "lyric generation" actually has to deliver here

Because the consumer is pitch-only, the goal is **phonetic diversity in the rendered audio**, not
literary quality. Real singing has consonants, plosives, fricatives, and coarticulation that the
single "ah" vowel entirely lacks; the value is purely acoustic domain coverage. This sharply lowers
the bar versus the `research/lyric-generation.md` report, which studies *coherent, theme-controlled,
melody-aligned* lyrics (a partially-unsolved, GPU/LLM-heavy problem). **We do not need that for
label-correct phonetic diversity.** We need, per note (or per phrase):

> a **singable syllable / phoneme group** such that the syllable count aligns to the note count.

This gives a **tiered design** — ship the cheap tier first, treat the expensive tier as optional:

| Tier | Producer | Output | Cost / deps | Determinism |
|------|----------|--------|-------------|-------------|
| **T0 (default, unchanged)** | none | open vowel "ah" | CPU, none | bit-exact |
| **T1 (recommended first)** | seeded syllable sampler over a pronunciation dict (CMUdict/g2p) | phoneme/syllable per note, count-matched | CPU, deterministic | bit-exact (seed → text) |
| **T2 (optional, themed)** | LLM melody→lyric (report's SongComposer-FT or constrained-decoding path) | coherent themed lyric, syllable-aligned | GPU/LLM, license-bearing | **cache-as-artifact** (not re-run) |

T1 delivers ~most of the phonetic-coverage benefit with none of T2's weight. T2 buys *coherent
themed words* whose only marginal benefit over T1 is more natural coarticulation statistics — a
diminishing return for a pitch model. **Recommend: build T1; gate T2 behind explicit opt-in.**

---

## 3. Where lyrics enter the pipeline (narrow surface)

Of 001's four lanes, **only the neural SVS lane (P4) can articulate text**:

- **Deterministic WORLD lane** — phonemes come from the donor recording's spectral envelope, not
  from text. Feeding it words cannot add consonants. **No benefit; stays vowel-only.**
- **Voice-conversion lane (P2)** — preserves the source's phonemes; VC changes timbre, not words.
  Whatever phonemes were in its input audio pass through. **No new wiring.**
- **Augmentation lane (P3)** — operates on rendered audio. **N/A.**
- **Neural SVS lane (P4: `svs.py`, DiffSinger/NNSVS)** — text→singing. **The sole consumer of
  generated lyrics**, and *this is where lyric-driven neural voices finally earn their keep over the
  WORLD vowel lane.*

Consequently, lyric generation rides **entirely on the lane 001 already flags as the most dangerous
for alignment** (FR-007, SC-010): real words add pre-onset consonants (the "b" in "bee" sounds
before the vowel lands on the beat) that shift perceived onsets past the 50 ms tolerance. **002 adds
no new alignment machinery** — it reuses 001's existing P4 safety net:

- **force-score-F0 mode** keeps pitch correct by construction, OR
- **re-derive-labels mode** runs MFA + onset detection on the actually-rendered audio and carries
  the re-derived `.tsv`, rejecting any sample with onset deviation > 50 ms (SC-010).

Phonemes give MFA *more* to align against than a bare vowel, so re-derivation mode is the natural
default whenever real lyrics are used.

---

## 4. Concrete changes 002 would introduce

### 4.1 Data model (smallest possible footprint)

- Add an **optional** `lyric` payload alongside `Note` — **not** a 4th mandatory `.tsv` column, to
  keep 001's 3-column parser and all existing scores valid. Two viable shapes:
  - **Sidecar file** `score.lyrics.jsonl`/`.txt` keyed by note index (preferred — zero change to the
    `.tsv` contract, lyrics are a detachable artifact), or
  - an optional `lyric: str | None` / `phonemes: list[str] | None` field on `Note`, populated only
    when a lyric source is active.
- The lyric payload is **derived/generated**, so it is treated like a score input artifact: pinned
  in the manifest and hash-referenced for replay (mirrors how the resolved config is embedded).

### 4.2 A `LyricSource` abstraction (config + provenance)

Add a lyric-source selector to the Run Config, defaulting to `vowel` so existing runs are unchanged:

```yaml
lyrics:
  source: vowel            # vowel (default) | sampler | operator | generated
  # sampler (T1):
  dict: cmudict            # pronunciation dictionary
  phoneme_inventory: en-us
  melisma: split           # how multi-note syllables map (1 syllable : N notes)
  # generated (T2, optional):
  model: songcomposer-sft  # or constrained-decoding:qwen2.5
  theme: "winter, longing" # the ONLY place theming enters — generated, never ingested
  cache: lyrics_cache/     # generated text is cached & pinned, not re-run for determinism
```

- `LyricSource.generate(score, seed) -> per-note syllables/phonemes`, count-matched to notes, with
  the same **count-mismatch surfacing + vowel fallback** 001 already documents (Edge Cases:
  "Lyrics shorter or longer than the note count" → neutral "ah").
- New **provenance axis** in the JSONL manifest (fits FR-008): `lyric_source`, `lyric_hash`, and for
  T2 the `model id` + `model license` + `theme` + generation `seed`. T2 model licenses
  (InternLM2/Llama/Qwen) drop straight into 001's existing **FR-011 license/consent discipline** —
  a generated-lyric model with an unacceptable license is refused exactly like a non-consented voice.

### 4.3 G2P / phonemization step (only when a real lyric source is active)

- Wire `phonemizer` (espeak-ng) and/or OpenUTAU G2P (001 research Decision 6) **lazily**, as an
  optional extra, imported only by the SVS lane when `lyrics.source != vowel`. The CPU baseline and
  deterministic lane never import it.
- Map phonemes → note durations for the SVS backend (NNSVS/DiffSinger ingest phoneme + duration +
  pitch), then validate via the existing P4 path.

---

## 5. Determinism & reproducibility handling (the one real friction)

001 guarantees bit-exact replay from the manifest (SC-009). LLM generation (T2) is non-deterministic
and GPU-bound, which would break that. Resolution, by tier:

- **T1 sampler** — seeded from 001's existing master-seed derivation (`score ID + stage name`);
  pure-CPU and **bit-exact**, no special handling.
- **T2 LLM** — **cache-as-artifact**: generate the lyric text *once*, pin it in the manifest
  (hash-referenced like the resolved config), and feed it into rendering like operator-supplied
  lyrics. Reproducibility then means *"replay the pinned lyric text,"* **not** *"re-run the LLM
  deterministically."* The neural SVS render that consumes it already falls under SC-009's
  *non-bit-deterministic* clause (same verdict + f0/onset within validator tolerances), so no new
  reproducibility category is needed.

---

## 6. Scope boundaries for the eventual 002 spec

**In scope:** optional lyric *source* (vowel/sampler/operator/generated); G2P for the SVS lane;
lyric provenance + license axis in the manifest; count-mismatch surfacing with vowel fallback;
cache-as-artifact for any non-deterministic generator.

**Out of scope:** changing the downstream task (still pitch-only — lyrics are never a label);
ingesting themes/lyrics from annotations (operator provides a theme string for T2, or nothing);
any lyric quality metric (no literary eval — phonetic coverage is the only relevant signal);
adding lyrics to the deterministic or voice-conversion lanes.

**Open questions for `/speckit-specify`:**
1. Sidecar lyric file vs. optional `Note.lyric` field — which keeps the contracts cleanest?
2. Is T1 (sampler) phonetic diversity alone enough to move a pitch model, making T2 unnecessary?
3. Melisma policy — how does one syllable map across N tied/legato notes for count-matching?
4. Should a phonetic-coverage statistic be added to the stats report (FR-012) as the success signal,
   in lieu of any lyric-quality metric?

---

## 7. One-paragraph summary

Lyric generation factors in as an **optional, opt-in phonetic-diversity lever** that produces a
*derived input artifact* consumed **only by the neural SVS lane**, leaving 001's CPU baseline,
deterministic lanes, and bit-exact reproducibility untouched. Ship a **CPU, seeded syllable sampler
(T1)** as the cheap default-off realism source; treat the research report's **LLM melody→lyric
machinery (T2)** as a further opt-in whose non-determinism is handled by **caching generated text as
a pinned manifest artifact**. Theming exists only as a generated `theme` string — it is never read
from annotations, which carry no lyric data. All lyric provenance and any model license ride 001's
existing manifest and consent discipline; all alignment risk rides 001's existing P4 re-derivation
safety net. **No change to the downstream pitch-only task; lyrics are never a label.**
