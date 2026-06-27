# Phase 1 Data Model: Optional Lyric Generation & Phonetic Diversity

Entities and field/lifecycle changes for the lyric layer. Builds on 001's data model; only the
deltas are described here. Glossary: **G2P** = grapheme-to-phoneme (letters/syllables → phonemes);
**phoneme** = a unit of speech sound; **melisma** = one syllable sung across several notes.

## Note (modified — the prototype foundation)

001's `Note` is `(onset_s, offset_s, pitch_midi)`. This feature adds one optional field.

| Field | Type | Notes |
|-------|------|-------|
| `onset_s` | float | unchanged (FR-001) |
| `offset_s` | float | unchanged |
| `pitch_midi` | int | unchanged |
| `lyric` | str \| None | **new.** Optional per-note syllable/word. `None` (default) = lyric-free → sung as the neutral open vowel. Never a corpus label (FR-002). |

**Rules:**
- `lyric=None` for every note ⇒ the score is lyric-free and serializes to the original 3 columns
  (byte-identity preserved, SC-001).
- A blank/whitespace 4th column parses as `None` (treated as lyric-free).
- `lyric` does not affect monophony, `min_note_ms`, or any existing validation.

## Score serialization (modified)

- `parse_tsv` reads an optional **4th column** as `lyric`; rows with 3 columns parse exactly as
  before. (Prototype on the `lyrics` branch; `tests/unit/test_lyrics.py`.)
- `serialize_score` emits a 4th column **only when at least one note carries a lyric**, so re-derived
  labels for lyric-free scores stay byte-identical (FR-001).

## LyricSource (new — enum)

The origin of a sample's lyrics, recorded in provenance.

| Value | Meaning | Hardware |
|-------|---------|----------|
| `vowel` | default; no lyric, open-vowel render | CPU |
| `supplied` | operator-supplied `lyric` carried in the score (4th column) | CPU |
| `automatic` | seeded CV-syllable sampler; needs no input data (FR-003) | CPU |
| `generated` | model-driven from a theme string; cached as a pinned artifact (FR-011) | optional GPU (out-of-process) |

## LyricPlan (new — in-memory, not stored)

The resolved per-note lyric assignment a source produces and the SVS lane consumes.

| Field | Type | Notes |
|-------|------|-------|
| `score_id` | str | the score this plan labels |
| `source` | LyricSource | which source produced it |
| `syllables` | list[str \| None] | one entry per note; `None` ⇒ open-vowel fallback for that note (FR-008) |
| `text_hash` | str | sha256 of the canonical syllable list — the manifest `lyric_hash` |
| `mismatch` | bool | true when the source's syllable count ≠ note count (resolved by vowel fallback, logged) |
| `multisyllable_notes` | list[int] | **new.** Note indices whose assigned text is not a single syllable. For `automatic`/`generated` this MUST be empty (FR-019 holds by construction/segmentation); for `supplied` it lists operator cells taken as-authored and flagged (never re-segmented). |
| `model` | LyricModel \| None | set only for `generated` |

**Rules:**
- `len(syllables) == len(score.notes)` always; a source that yields too few/many is reconciled to the
  note count by truncation/`None`-padding (vowel fallback), with `mismatch=True` (FR-008, SC-005). No
  note is dropped, added, or shifted.
- **One syllable per note (FR-019)**: for `automatic` and `generated`, every non-`None` entry in
  `syllables` is exactly one singable syllable — the CV sampler emits one per note by construction, and
  `generated` free text passes through `syllabify.segment` before 1:1 alignment — so
  `multisyllable_notes` is empty. The structural SC-010 check asserts this directly on the plan.
- **Supplied cells are taken as authored**: for `supplied`, a cell is never re-segmented, truncated, or
  rejected; `syllabify.syllable_count` only *flags* a non-single-syllable cell by appending its index
  to `multisyllable_notes` (surfaced in provenance + stats).
- A syllable on a note shorter than the learned `min_note_ms` does not override 001's short-note
  flag/reject policy.
- `text_hash` is deterministic over the canonical (note-ordered) syllable list.

## LyricModel (new — for the `generated` source)

Mirrors `Voice`'s license/consent discipline for lyric-generation models.

| Field | Type | Notes |
|-------|------|-------|
| `model_id` | str | model identity recorded in provenance (FR-010) |
| `license` | str | free-text license tag |
| `license_ok` | bool | required; a model with `license_ok` false/missing is **refused** (FR-010), mirroring `Voice.consent_verified` |
| `model_ref` | str | toolkit/weight reference resolved by `backends/lyrics/` |

**Rules:**
- `license_ok` false/missing ⇒ the stage refuses, produces no sample from it, and surfaces a
  `license_refused`-style note in provenance (FR-010, SC-006).

## Phoneme run (new — in-memory, G2P output)

Produced by `voders.lyrics.g2p` from a `LyricPlan` + the score's note timings; consumed by the SVS
lane. Not persisted.

| Field | Type | Notes |
|-------|------|-------|
| `note_index` | int | the note this run sings |
| `phonemes` | list[str] | espeak-ng phonemes for the syllable |
| `nucleus_onset_s` | float | the vowel start = the note onset (vowel-on-the-beat, Decision L3) |
| `lead_consonants` | list[str] | placed in a short pre-onset window |
| `tail_consonants` | list[str] | placed in a short pre-offset window |

## Syllable segmentation (new — `voders.lyrics.syllabify`, FR-019)

A pure, deterministic helper (no model, no I/O) the sources use to honor one-syllable-per-note:

| Function | Signature | Used by |
|----------|-----------|---------|
| `segment` | `segment(text: str) -> list[str]` | `generated` — splits model words/lines into ordered singable syllables before 1:1 note alignment |
| `syllable_count` | `syllable_count(text: str) -> int` | `supplied` — flags a cell whose count ≠ 1 into `LyricPlan.multisyllable_notes` (no transformation) |

**Rules:**
- Deterministic: a pure function of `text` (dictionary lookup for known words + a rule-based fallback
  for the rest), so `generated` replay over the pinned text re-segments identically (FR-012, FR-019).
- The `automatic` sampler does **not** call `segment` — it already emits one CV syllable per note.
- English-rule sufficiency is assumed for v1 (syllables need only be singable, not linguistically
  authoritative — spec Assumptions).

## LyricsConfig (new — RunConfig addition, FR-013)

Added to the existing `RunConfig` (`src/voders/config/models.py`). Default keeps runs unchanged.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `source` | LyricSource | `vowel` | per-run lyric source selector (FR-013) |
| `inventory` | str | `"en_cv"` | checked-in syllable inventory / **style** for `automatic`: `en_cv` (neutral CV) or `scat` (jazz scat-singing, Scatman-style) |
| `g2p_backend` | str | `"espeak"` | G2P backend for articulation: `espeak` (CPU rules) or `neural` (byte-level T5 on GPU, higher quality) |
| `syllabifier` | str | `"en_rule"` | segmenter used by `generated` (split words→syllables) and to flag `supplied` multi-syllable cells (FR-019); deterministic, CPU |
| `melisma` | str | `"per_note"` | v1 accepts only `per_note` (one syllable/note); `sustain_ties` (hold across tied notes) is **reserved** and raises a "not yet supported" validation error until implemented |
| `theme` | str \| None | `None` | **only** for `generated`; operator-supplied; never read from annotations (FR-011) |
| `model` | LyricModel \| None | `None` | only for `generated` |
| `cache_dir` | str | `"lyrics"` | subdir of `output_root` for pinned generated text (FR-012) |

**Rules:**
- `source=vowel` (default) ⇒ no lyric module is imported and output is byte-identical to pre-feature
  (SC-001).
- `theme`/`model` are valid only with `source=generated`; set otherwise ⇒ config validation error.

## ProvenanceRecord (modified — new lyric axis, FR-009/010)

001's `ProvenanceRecord` has `model_config = ConfigDict(extra="forbid")`, so new fields are added
explicitly. All default such that existing lyric-free runs serialize compatibly.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `lyric_source` | str | `"vowel"` | which source produced the sample's lyrics (FR-009) |
| `lyric_hash` | str \| None | `None` | sha256 of the syllable list / pinned artifact key (FR-009, FR-012) |
| `lyric_model` | str \| None | `None` | model id for `generated` (FR-010) |
| `lyric_model_license` | str \| None | `None` | license tag for `generated` (FR-010) |
| `lyric_articulated` | bool | `False` | true only when the lane actually sang phonemes (the SVS lane); false for the deterministic/VC lanes even if a lyric was present (FR-006 edge case) |
| `lyric_multisyllable_supplied` | int | `0` | count of operator-`supplied` cells on this sample that were not a single syllable and were sung as authored (FR-019 flag; 0 for vowel/automatic/generated) |

The **no-shift** guarantee (FR-015/FR-016, SC-008/SC-009) needs no new stored field: the per-note
pitch and onset/offset deltas of a lyric render versus the lyric-free baseline are computed by the
differential eval harness (FR-018) and, when an SVS sample is articulated, the largest such deltas are
recorded in the existing free-form `notes` map (alongside 001's `max_onset_dev_ms`) for audit.

## Lyric cache artifact (new — on disk, FR-012)

`<output_root>/lyrics/<sha256>.jsonl` — one JSON record per score (`score_id`, ordered `syllables`,
`source`, optional `model`). Written once by the `generated` pre-pass; read on replay so the model is
never re-invoked (SC-007). Text, not binary (no LFS implication).

## Aggregate stats (modified — FR-014)

The run stats report gains:
- **lyric-source breakdown** — sample counts per `lyric_source`.
- **phonetic-coverage summary** — count of distinct sung phonemes across the corpus vs. the
  vowel-only baseline (SC-003), produced by `voders.lyrics.coverage`.
- **supplied multi-syllable count** — total operator-supplied cells sung as authored that were not a
  single syllable (sum of `lyric_multisyllable_supplied`), so the operator sees how many supplied cells
  fell outside the one-syllable-per-note convention (FR-019).

## Lifecycle

```text
RunConfig.lyrics.source
   │
   ├─ vowel ──────────────► (no lyric module) ─► lanes render open vowel  [byte-identical, SC-001]
   │
   ├─ supplied ─► Note.lyric (4th column) ─┐
   ├─ automatic ─► sampler(sample_seed) ───┼─► LyricPlan (count-reconciled, hashed) ──┐
   └─ generated ─► backends/lyrics (once) ─┘        │ pin <hash>.jsonl (FR-012)        │
                                                    ▼                                  ▼
                                         SVS lane only: g2p → phonemes → articulate    other lanes:
                                         under FR-007 force_score_f0 / rederive        unchanged audio,
                                         (label safety, SC-002)                        lyric_articulated=False
                                                    │
                                                    ▼
                                    ValidationVerdict (001, unchanged) ─► ProvenanceRecord(+lyric axis)
```
