# Phase 1 Data Model: Score-Domain Augmentations

Entities and field-level deltas to existing 001 models. New types live in `voders.scoreaug`; deltas are
additive and default to inert so feature-off runs are byte-identical (SC-001).

## New entities (`voders.scoreaug.models`)

### `ScoreAxis` (enum)

`transpose | humanize | volume` — the three augmentation axes. Used as the `score_aug` directory segment
(`corpus/augmented/{axis}/`) and the transform-descriptor tag.

### `ScoreVariant`

A base score expanded into one variant input score, plus the provenance to record it.

| Field | Type | Notes |
|-------|------|-------|
| `score` | `Score` | the rewritten note rows — *these are the labels* (FR-003) |
| `base_score_id` | `str` | the originating base score's id (lineage, FR-011) |
| `profile_id` | `str` | the `ScoreAugmentationProfile` that produced it |
| `axis` | `ScoreAxis` | which axis (drives the output subtree, FR-016) |
| `transform` | `str` | compact descriptor: `t+12`, `t-24`, `hum0`, `vol` (in the `score_id`, SC-009) |
| `seed` | `int` | the variant seed `derive_seed(master_seed, base, "score_aug", profile_id, axis, draw)` |
| `dynamics_applied` | `bool` | volume axis only; lanes that honour per-note gain set true (FR-008) |
| `notes_meta` | `dict` | optional diagnostics: `dropped`/`clamped`/`constraint_hit` counts |

`score.score_id` = `f"{base_score_id}__{transform}"`. The base score is emitted as an ordinary
`ParsedScore` (no `ScoreVariant` wrapper); `expand()` returns the base plus its variants.

**Validation / invariants**
- Every `score.notes[i].pitch_midi ∈ [0, 127]` (transpose, FR-005) — enforced by the existing `Note`
  validator (`scores/models.py:17`) as a hard backstop.
- `score` is a valid monophonic sequence — ordered, non-overlapping per policy, strictly positive
  durations (humanise, FR-007) — accepted by `parse_tsv` without modification.
- `base_score_id` MUST NOT contain the reserved `__` separator (guarded at expansion).
- Note count and ordering equal the base's; each note's `lyric` is carried 1:1 (FR-014).

## Existing-model deltas

### `Note` (`scores/models.py`) — add per-note gain

```text
gain: float | None = None   # linear amplitude multiplier; None => nominal (FR-008/009)
```

- **Excluded from the label `.tsv`**: `serialize_score` is unchanged — it emits only
  `onset/offset/pitch[/lyric]`, so the transcription label schema is untouched (FR-009).
- `parse_tsv` never reads a gain (no 5th column); a parsed score always has `gain=None`.
- Carried in-memory to the renderer via the existing `Score` on `RenderRequest.score`.

### `ScoreAugmentationProfile` (NEW, `config/models.py`) — seeded profile block

```text
profile_id: str
transpose:      TransposeKnob | None = None
humanize_time:  HumanizeKnob | None  = None
volume:         VolumeKnob | None     = None
```

- `TransposeKnob`: `offsets: list[int]` (semitones), `policy: "drop" | "clamp" = "drop"`,
  `window: tuple[int, int] = (0, 127)` (singable MIDI range) (FR-004/005).
- `HumanizeKnob`: `onset_sigma_s: float`, `duration_sigma_s: float`, `max_dev_s: float`,
  `draws: int`, `overlap: "forbid" = "forbid"`, `min_dur_s: float = 0.01` (FR-006/007).
- `VolumeKnob`: `gain_db_range: tuple[float, float]` (or linear range) + distribution `"uniform"`
  default (FR-008).
- A knob left `None` disables that axis. A profile with all three `None` emits only the base score.

### `RunConfig` (`config/models.py`) — add the selector

```text
score_augmentation: list[ScoreAugmentationProfile] = Field(default_factory=list)
```

Default `[]` ⇒ no expansion ⇒ feature off ⇒ byte-identical runs (FR-001/013, SC-001). Included in
`config_hash` so a score-aug run hashes distinctly. `RunConfig.model_config` is `extra="forbid"`, so the
key is added explicitly (existing configs without it stay valid via the default).

### `ProvenanceRecord` (`manifest/models.py`) — add the score-aug axis

```text
base_score_id: str | None = None        # lineage to the base score (None => an original) (FR-011)
score_aug_profile: str | None = None     # ScoreAugmentationProfile id (None => original)
score_aug_axis: str | None = None        # transpose | humanize | volume
score_aug_transform: str | None = None   # t+12 | hum0 | vol — the applied transform
score_aug_seed: int | None = None        # the variant seed (reproducibility audit, FR-010)
dynamics_applied: bool = False           # did the lane honour per-note gain (FR-008)
```

All default to "original/none", so a record for a base score (or any feature-off run) serialises with the
new fields at their inert defaults — existing manifest consumers are unaffected. `model_config` stays
`extra="forbid"`.

## On-disk layout delta (`corpus/store.py`)

```text
<output_root>/
  corpus/
    shard=NNN/            <base_id>_singer_<voice>.wav + .tsv      # ORIGINALS (unchanged location)
    augmented/
      transpose/shard=NNN/ <base>__t+12_singer_<voice>.wav + .tsv  # variants, axis-foldered (FR-016)
      humanize/shard=NNN/  <base>__hum0_singer_<voice>.wav + .tsv
      volume/shard=NNN/    <base>__vol_singer_<voice>.wav + .tsv
  rejected/
    shard=NNN/             …                                       # original rejects
    augmented/<axis>/shard=NNN/ …                                  # variant rejects
```

- `CorpusStore.paths_for` gains routing: when `record.score_aug_axis` is set, the shard base becomes
  `corpus/augmented/<axis>` (or `rejected/augmented/<axis>`); otherwise the existing `corpus/`/`rejected/`
  path is returned **unchanged** (SC-001 backstop).
- No variant path equals an original path or another variant path (distinct `score_id` + distinct axis
  subtree) — zero collisions (FR-016, SC-009).

## Stats delta (`corpus/stats.py`)

`build_stats` returns a new `score_augmentation` block:

```text
score_augmentation: {
  enabled: bool,                       # any accepted record has score_aug_profile
  variant_share: float,                # accepted variants / accepted total
  by_transform: { "t+12": n, "hum0": n, "vol": n, ... },   # FR-012 coverage axis
  effective_multiplier: float,         # accepted total / accepted originals (≈ variants×… , SC-007)
}
```

## Lifecycle

```text
load_config → RunConfig.score_augmentation
  └─ Orchestrator.run:
       parsed = _load_scores()
       expanded = flatten( expand(ps, profile, derive_seed(...)) for ps in parsed for profile in cfg.score_augmentation )
                  ++ parsed            # base scores always present
       log projected effective sample count (FR-015)
       for ps in expanded:            # existing loop, unchanged
         for voice, for lane, for audio-profile: render → validate → write
           store routes by ps' score_aug_axis (original vs augmented/<axis>)
           ProvenanceRecord carries base_score_id + score_aug_* (FR-011)
  → build_stats adds score_augmentation block (FR-012)
  → voders eval --suite score_aug asserts SC-001..SC-009
```
