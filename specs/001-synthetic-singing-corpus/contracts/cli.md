# CLI Contract

Single entry point: `python -m voders.cli <subcommand>`. Exit code 0 on success, non-zero on a
gated failure. All subcommands are non-interactive (config-driven) for reproducibility.

## `run`

Render a corpus from a Run Config.

```
voders run --config <path.yaml> [--output-root <dir>] [--resume] [--lanes deterministic,voice_conversion]
```

- `--config` (required): path to the declarative Run Config (FR-016).
- `--output-root`: overrides `output_root` in the config.
- `--resume`: resume from the last completed shard checkpoint (SC-011); default starts fresh.
- `--lanes`: comma-list overriding which lanes are enabled (FR-015).
- **Output:** `corpus/`, `rejected/`, `manifest.jsonl`, `config.resolved.yaml`, `stats.json` under
  the output root (see `research.md` Decision 7).
- **Determinism:** identical config + `master_seed` reproduce the run within documented tolerance
  (SC-009), independent of worker count/order (FR-013).

## `eval`

Validate a produced corpus and print a Success-Criteria pass/fail table.

```
voders eval --manifest <out>/manifest.jsonl [--strict]
```

- Recomputes the alignment validator over accepted samples and reports: SC-001 (onset ≥99%),
  SC-002 (offset ≥99%), SC-007 (first-attempt pass ≥95%), SC-008 (zero unconsented voices),
  SC-009 (isolated reproducibility).
- Exit non-zero if any gated criterion fails (`--strict` also fails on warnings).

## `audit`

License/consent audit over the manifest (FR-011, SC-008).

```
voders audit --manifest <out>/manifest.jsonl
```

- Lists donor voices present, flags any `consent_verified=false` record, exits non-zero if any
  unconsented voice reached the accepted corpus.

## `stats`

Emit/refresh aggregate corpus statistics (FR-012).

```
voders stats --manifest <out>/manifest.jsonl [--out stats.json]
```

- Reports total samples, unique scores/voices, timbre identities (SC-004), pitch & duration
  distributions, augmentation coverage (SC-006), accept/reject counts.

## `splits`

Write source-stratified train/validation splits over accepted samples (issue #7), so a downstream
training run can attribute errors to specific generators.

```
voders splits --manifest <out>/manifest.jsonl [--val-fraction 0.2] [--seed 0] \
              [--stratify lane,voice_id,augmentation_profile] [--out splits.json]
```

- Holds out `--val-fraction` *within each source group* (lane/voice/augmentation profile), so train
  and val both carry a proportional slice of every source. Deterministic in `--seed`. Writes
  `splits.json` (train/val entry lists with `sample_id`/`audio_path`/`score_path`/`source`, plus
  per-source counts).
