# Phase 0 Research: Optional Lyric Generation & Phonetic Diversity

Resolves the open toolchain/design choices for the lyric layer. Each decision lists what was chosen,
why, and what was rejected. Inputs: the feature spec, GitHub issue #4, the `research/lyric-generation.md`
report, and the existing 001 code (`src/voders/render/svs.py`, `scores/parse.py`, `seeds.py`,
`manifest/models.py`, `config/models.py`, `backends/`).

## Decision L1 — Generated melody→lyric model (the optional `generated` source)

**Decision:** Make `generated` an **optional, out-of-process backend** under `backends/lyrics/`, with
the model unselected by default. Target a **constrained-decoding pipeline** (a permissive open LLM —
e.g. Qwen2.5/Llama-3 the operator already self-hosts — plus a syllable/note-count constraint
compiler) as the reference path, with **SongComposer** (symbolic melody→aligned lyrics, Apache-2.0
code, weights stated commercially usable) as the alternative behind the same backend interface.

**Rationale:** The `research/lyric-generation.md` report is explicit that **no open-weight model
natively takes a note sequence + theme prompt → aligned lyrics**; the two viable paths are a
SongComposer QLoRA fine-tune (1–3 wk) or a constrained-decoding pipeline (2–4 wk). For *this* feature
the lyric need is weak — count-matched singable syllables for phonetic diversity, not literary
quality — so neither heavy path is on the critical path; both are encapsulated behind the backend so
the feature ships its value via `automatic` alone. Constrained decoding is the cleaner default
(any permissive LLM + a CMUdict/g2p syllable counter, no model-license entanglement); SongComposer is
the alternative when symbolic alignment is wanted.

**Alternatives considered:** Bundling a model into the 3.14 core — rejected (pulls GPU/older-Python
deps into the CPU baseline, violates FR-005 isolation). YuE/ACE-Step/DiffRhythm/SongGen — rejected as
the inverse direction (lyrics+style → audio; cannot emit standalone lyrics). REFFLY — rejected (no
released weights located). Lyra — design reference only (CC-BY-NC, non-commercial).

## Decision L2 — Automatic syllable source (the corpus's real workhorse)

**Decision:** A **seeded consonant–vowel (CV) syllable sampler** over a small checked-in phoneme
inventory. For each note it draws one singable syllable, seeded by
`sample_seed(master_seed, score_id, voice_id, stage="lyrics")` (the existing 001 derivation), so the
assignment is reproducible in isolation and independent of worker count (FR-004, SC-004). The sampler
biases draws to **broaden phoneme coverage** across a score rather than uniform random.

**Rationale:** Annotations carry no lyrics, so the corpus needs a source that needs no input data.
A CV sampler is pure-CPU, dependency-free, fully deterministic, and directly maximizes the only signal
that matters to a pitch model — phonetic coverage (SC-003). It reuses 001's seed derivation verbatim,
so it inherits the bit-exact reproducibility guarantee with no new machinery.

**Alternatives considered:** Sampling real words from CMUdict — rejected for v1 (adds a dependency and
word-frequency bias without improving phoneme coverage for a pitch model; a curated inventory is
simpler and more controllable). An LLM for every sample — rejected (non-deterministic, GPU, and
overkill; that is the `generated` opt-in, used once and cached).

## Decision L3 — G2P (grapheme-to-phoneme) and phoneme→note mapping

**Decision:** Use **`phonemizer`** with the **espeak-ng** backend to convert a syllable/word string
into phonemes, imported **lazily** only when `lyrics.source != vowel` and the SVS lane articulates.
Map phonemes to a note's duration with the convention: the **vowel (nucleus) carries the sustained
pitch over the full note**, any leading consonant(s) are placed in a short pre-onset window, and a
trailing consonant in a short pre-offset window. espeak-ng is a system package recorded as a
prerequisite.

**Rationale:** `phonemizer`/espeak-ng is the path 001 research Decision 6 already named, is CPU-only,
multilingual, and widely used by SVS toolkits. Putting the vowel on the beat keeps the *perceived*
onset aligned to the score even though a consonant sounds slightly before it — which is exactly the
alignment risk the SVS safety net (Decision L4) then verifies, not assumes.

**Alternatives considered:** OpenUTAU's built-in G2P — viable but couples G2P to one SVS toolkit;
kept as a fallback inside the `nnsvs`/`diffsinger` backends. A hand-rolled CMU phoneme table —
rejected (reinvents espeak-ng, English-only).

## Decision L4 — Label safety when the SVS lane sings real phonemes

**Decision:** **Reuse 001's existing FR-007 safety net unchanged.** Lyric articulation feeds the same
two modes: `force_score_f0` (pitch/onset/offset correct by construction; `label_score == score`) and
`rederive_labels` (humanize, render, re-derive onsets/offsets from the rendered audio via
`voders.validate.rederive`, reject if any onset drifts >50 ms, SC-010/SC-002). No new alignment code.

**Rationale:** Pre-onset consonants are the one real alignment threat lyrics add, and 001 already has
the exactly-right gate for "expressive timing might break the label." The re-derivation envelope
analysis operates on audio energy and is agnostic to whether the audio is a vowel or a word, so it
extends to phonemes for free. Adding lyrics therefore changes *what the SVS lane sings*, not *how
labels are guaranteed*.

**Alternatives considered:** A new lyric-specific aligner (e.g. forced alignment with a phoneme
dictionary) — deferred; the energy-based re-derivation already gates onset drift, and a phoneme-level
aligner is a later precision upgrade, not required for SC-002.

**No-shift (pitch/timing neutrality) addendum (FR-015–FR-018, SC-008/SC-009):** Beyond the absolute
score-tolerance gate, the lyric layer must introduce **no pitch or timing shift relative to the
lyric-free render of the same (score, voice, seed)**. Pitch neutrality is free in `force_score_f0`
mode (the f0 is the score's, independent of phonemes); a differential test guards a misbehaving
backend. Timing neutrality is the real new guarantee: the labeled onset is the vowel nucleus on the
score beat, leading consonants live in a pre-onset window, any constant G2P/vocoder lead-in delay is
subtracted (001 FR-019), and a residual per-note onset/offset delta versus the lyric-free baseline
beyond the validator's 10 ms resolution forces rejection. The evaluation harness verifies this by
rendering each fixture twice — lyric-on and lyric-off at the same seed — and asserting near-zero
per-note pitch and onset/offset deltas (the gated "no shift from adding vocal synthesis" verdict).

## Decision L5 — Reproducibility of the non-deterministic `generated` source

**Decision:** **Cache-as-artifact.** The `generated` model runs once per score set as a pre-pass; its
output is written to `<output_root>/lyrics/<sha256>.jsonl` (one record per score: score_id → ordered
syllables) and each provenance record stores `lyric_hash` = that sha256. Replay reads the pinned file;
the model is never re-invoked (FR-012, SC-007). The resulting audio then reproduces under 001's
existing per-lane SC-009 tolerance (bit-exact for deterministic/VC lanes; same-verdict + f0/onset
bounds for neural).

**Rationale:** An LLM is non-deterministic and GPU-bound, which would otherwise break SC-009. Pinning
the generated text turns a non-reproducible step into a reproducible input artifact (the same trick
001 uses for the resolved config), without forcing deterministic LLM inference.

**Alternatives considered:** Requiring temperature-0/greedy deterministic generation — rejected
(brittle across hardware/toolkit versions; the report flags vendor-specific nondeterminism).
Re-generating on replay — rejected (not reproducible, wastes GPU).

## Decision L6 — Lyric storage shape (landing the prototype)

**Decision:** Adopt the `lyrics`-branch prototype: an **optional `Note.lyric: str | None`** field and
an **optional 4th `.tsv` column** read by `parse_tsv`; `serialize_score` emits the 4th column only
when a score carries lyrics, so lyric-free scores stay byte-identical (3-column). This is the
`supplied` source's on-disk form and the in-memory carrier for all sources.

**Rationale:** The prototype is already written and tested (`tests/unit/test_lyrics.py`), is
backward-compatible by construction (SC-001), and keeps lyrics attached to the note they align to. It
resolves issue #4's open question (sidecar vs. field) in favor of the field/column.

**Alternatives considered:** A sidecar lyric file keyed by note index (issue #4's earlier preference)
— rejected now that the prototype exists; the 4th column is simpler, keeps one artifact per score, and
preserves byte-identity for lyric-free scores anyway.

## Resolved choices summary

| Topic | Choice |
|-------|--------|
| Generated model (optional) | constrained-decoding LLM (default) / SongComposer (alt), out-of-process `backends/lyrics/` |
| Automatic source | seeded CV-syllable sampler, coverage-biased, `stage="lyrics"` |
| G2P | `phonemizer` + espeak-ng, lazy import; vowel-on-the-beat phoneme→duration mapping |
| Label safety | reuse 001 FR-007 force-score-F0 / rederive_labels unchanged |
| Generated reproducibility | cache-as-artifact `<output_root>/lyrics/<hash>.jsonl`, pinned in manifest |
| Storage shape | `Note.lyric` + optional 4th `.tsv` column (the prototype) |

No NEEDS CLARIFICATION remain.
