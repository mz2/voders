# Contract: Manifest & Run-Config Deltas

Additive changes to the 001 contracts. All new fields default so that lyric-free runs serialize
identically to today (SC-001). The 001 schemas remain authoritative
(`specs/001-synthetic-singing-corpus/contracts/manifest-record.schema.json`,
`run-config.schema.yaml`); these are the 002 additions.

## ProvenanceRecord additions (`manifest.jsonl`)

`ProvenanceRecord` uses `extra="forbid"`, so each new field is declared explicitly.

| Field | JSON type | Default | Requirement |
|-------|-----------|---------|-------------|
| `lyric_source` | string (`vowel`\|`supplied`\|`automatic`\|`generated`) | `"vowel"` | FR-009 |
| `lyric_hash` | string \| null | `null` | FR-009, FR-012 |
| `lyric_model` | string \| null | `null` | FR-010 (generated only) |
| `lyric_model_license` | string \| null | `null` | FR-010 (generated only) |
| `lyric_articulated` | boolean | `false` | FR-006 — true only when the lane sang phonemes |
| `lyric_multisyllable_supplied` | integer | `0` | FR-019 — count of supplied cells sung as authored that were not a single syllable (0 for vowel/automatic/generated) |

**Audit query (SC-006):** an end-of-run scan asserts every record has a `lyric_source`, every
non-`vowel` record has a `lyric_hash`, and zero records carry a `generated` source whose
`lyric_model_license` was refused.

## RunConfig addition: `lyrics` block

```yaml
# Optional. Omit entirely → source defaults to `vowel`, run is byte-identical to pre-feature.
lyrics:
  source: vowel            # vowel | supplied | automatic | generated   (FR-013, default vowel)
  inventory: en_cv         # checked-in syllable/phoneme inventory for `automatic`
  g2p_backend: espeak      # articulation backend (lazy phonemizer/espeak-ng)
  syllabifier: en_rule     # deterministic segmenter: split generated words→syllables, flag supplied multi-syllable cells (FR-019)
  melisma: per_note        # per_note | sustain_ties
  # generated-only (validation error if set with another source):
  theme: null              # operator-supplied theme string — the ONLY theming channel (FR-011)
  model:                   # LyricModel — license-gated like a Voice (FR-010)
    model_id: ""
    license: ""
    license_ok: false      # false/missing ⇒ refused
    model_ref: ""
  cache_dir: lyrics        # subdir of output_root for pinned generated text (FR-012)
```

**Validation rules:**
- `source: vowel` ⇒ no other `lyrics` field has any effect; no lyric module imported.
- `theme` or `model` present with `source != generated` ⇒ config error.
- `source: generated` ⇒ `model.license_ok` must be true or the run refuses that source (FR-010).
- The resolved `lyrics` block is hashed into the existing `config_hash`, so a lyric run replays from
  the manifest + pinned artifact (FR-012, SC-007).

## Backend request delta (`render_via_backend`, out-of-process SVS)

The JSON request to `backends/svs` gains optional fields; absent ⇒ vowel render (unchanged):

```json
{ "score": {...}, "seed": 123, "model_ref": "...",
  "phonemes": [ { "note_index": 0, "phonemes": ["s"," a"], "lead": ["s"], "tail": [] } ] }
```

The `backends/lyrics` worker (generated source) request/response:

```json
// request:  { "score": {...}, "theme": "winter, longing", "model_ref": "...", "seed": 123 }
// response: { "score_id": "s001", "syllables": ["win","ter","long","ing", ...] }
```
