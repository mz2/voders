# Contract: Config, Manifest & Store Deltas

Additive surface changes to the 001 declarative run config, provenance manifest, and on-disk store. Every
delta defaults to inert so a config without `score_augmentation` round-trips and reproduces exactly as
today (FR-001, SC-001).

## Run config (`RunConfig.score_augmentation`)

```yaml
# evals/fixtures/score-aug-smoke.yaml (excerpt)
run_id: score-aug-smoke
master_seed: 42
scores: evals/fixtures/scores
output_root: /tmp/score-aug-smoke
voices: [ ... ]                 # unchanged
lanes:
  deterministic: { enabled: true }
score_augmentation:             # NEW — default [] when absent
  - profile_id: all-axes
    transpose:
      offsets: [-12, 12]
      policy: drop              # drop | clamp
      window: [0, 127]
    humanize_time:
      onset_sigma_s: 0.02
      duration_sigma_s: 0.02
      max_dev_s: 0.05
      draws: 2
      overlap: forbid
      min_dur_s: 0.01
    volume:
      gain_db_range: [-6.0, 6.0]
      distribution: uniform
```

- Absent `score_augmentation` ⇒ `[]` ⇒ no expansion (FR-013). Existing configs are valid unchanged.
- Each knob is independently omittable; an omitted knob disables its axis.
- `config_hash` includes the block, so a score-aug run hashes distinctly from its off-control.

### Config validation

- `transpose.policy ∈ {drop, clamp}`; `transpose.window` within `[0,127]` and `lo ≤ hi`.
- `humanize_time.draws ≥ 1`; `max_dev_s > 0`; `min_dur_s > 0`; `overlap == "forbid"` in v1 (others
  rejected with a "not yet supported" error, mirroring 002's melisma guard).
- `volume.gain_db_range` low ≤ high.
- `profile_id` unique within the list.

## Manifest (`ProvenanceRecord` additions)

```jsonc
{
  "sample_id": "t1__t+12_singer_alice",
  "score_id": "t1__t+12",          // the variant's id (self-describing)
  "base_score_id": "t1",            // NEW — lineage (null for an original)
  "score_aug_profile": "all-axes",  // NEW
  "score_aug_axis": "transpose",    // NEW — transpose | humanize | volume
  "score_aug_transform": "t+12",    // NEW
  "score_aug_seed": 7234123,        // NEW
  "dynamics_applied": false,         // NEW — true only when a lane honoured per-note gain
  "score_path": "corpus/augmented/transpose/shard=000/t1__t+12_singer_alice.tsv",
  "audio_path": "corpus/augmented/transpose/shard=000/t1__t+12_singer_alice.wav"
  // ...all existing fields unchanged...
}
```

- An **original** record has `base_score_id=null` and all `score_aug_*` fields null/false.
- The record is queryable for full lineage without re-running (FR-011): base → profile → seed → transform
  → output location.

## Store layout (`CorpusStore.paths_for` routing)

| Record | Accepted path | Rejected path |
|--------|---------------|---------------|
| original (`score_aug_axis` null) | `corpus/shard=NNN/<id>.{wav,tsv}` | `rejected/shard=NNN/...` |
| variant (axis set) | `corpus/augmented/<axis>/shard=NNN/<id>.{wav,tsv}` | `rejected/augmented/<axis>/shard=NNN/...` |

- Originals keep their **exact current location** (SC-001 backstop): a feature-off run creates no
  `augmented/` directory.
- `<id>` encodes base + transform, so no two files collide and every file is classifiable as
  original-vs-augmented from its path alone (FR-016, SC-009).

## Stats (`build_stats` addition)

```jsonc
"score_augmentation": {
  "enabled": true,
  "variant_share": 0.75,
  "by_transform": { "t+12": 12, "t-12": 12, "hum0": 12, "hum1": 12, "vol": 12 },
  "effective_multiplier": 4.0
}
```

Reports the new coverage axis alongside the existing `timbre_identities` / `augmentation_coverage` axes
(FR-012, SC-007).

## CLI (`voders run`) projected-count log

Before producing, log the projected effective sample count
`= Σ_scores (1 + variants(score)) × voices(lane) × (1 + audio_profiles)` so the operator anticipates
corpus size (FR-015). Non-fatal; informational line only.

## Eval (`voders eval --suite score_aug`)

Adds gated criteria SC-001..SC-009 (see plan Evaluation Strategy). Differential criteria read each variant
record's `base_score_id` to locate its base score in the same manifest and compare label rows.
