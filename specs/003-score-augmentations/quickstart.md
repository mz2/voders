# Quickstart: Score-Domain Augmentations

Single-command run + single-command verdict, per Constitution Principle IV. All commands go through `uv`.

## Prerequisites

- The 001 environment: `uv sync` (no new dependency — this feature is pure-CPU `numpy`).
- No GPU, no system libraries, no model backend.

## 1. Run the feature (fans out score variants)

```bash
uv run voders run --config evals/fixtures/score-aug-smoke.yaml --output-root /tmp/score-aug
```

This expands each fixture score along all three axes (transpose `[-12, +12]`, a 2-draw humanisation,
volume on), renders every base score + variant through the deterministic lane, and writes:

```text
/tmp/score-aug/
  corpus/shard=000/                       # ORIGINALS (un-augmented base scores)
    fixt1_singer_alice.wav + .tsv
  corpus/augmented/transpose/shard=000/   # variants — physically separate (FR-016)
    fixt1__t+12_singer_alice.wav + .tsv
    fixt1__t-12_singer_alice.wav + .tsv
  corpus/augmented/humanize/shard=000/
    fixt1__hum0_singer_alice.wav + .tsv
    fixt1__hum1_singer_alice.wav + .tsv
  corpus/augmented/volume/shard=000/
    fixt1__vol_singer_alice.wav + .tsv
  manifest.jsonl   stats.json   config.resolved.yaml
```

The run logs the projected effective sample count before producing (FR-015). Every variant file's
name/path tells you its base score and the augmentation applied, with no need to open the manifest.

## 2. Evaluate it (pass/fail against Success Criteria)

```bash
uv run voders eval --manifest /tmp/score-aug/manifest.jsonl --suite score_aug
```

Prints a table and exits non-zero on any gated failure:

| SC | Check |
|----|-------|
| SC-001 | feature-off control run byte-identical to a pre-feature render |
| SC-002 | every transposition note shifted by exactly the offset; zero notes outside `0..127` |
| SC-003 | every humanisation variant valid monophonic; deviations within the max-dev budget |
| SC-004 | volume widens the rendered level distribution; labels byte-identical to un-varied render |
| SC-005 | re-expanding a fixed `(base, profile, seed)` is byte-identical |
| SC-006 | every variant record carries base id + profile + seed + transform; path-classifiable |
| SC-007 | stats report score-aug coverage; effective count = `variants × voices × audio-profiles` |
| SC-008 | variants pass the 001 verification gate at the base-score rate |
| SC-009 | 100% of variant files in `augmented/`, zero collisions; original-vs-augmented from path alone |

## 3. Confirm the SC-001 backward-compatibility control

```bash
uv run voders run --config evals/fixtures/score-aug-off.yaml --output-root /tmp/score-aug-off
# /tmp/score-aug-off has NO corpus/augmented/ subtree; corpus/ is byte-identical to a pre-feature run.
```

## 4. Inspect coverage

```bash
uv run voders stats --manifest /tmp/score-aug/manifest.jsonl   # score_augmentation block in stats.json
```

## Notes

- **Transposition out of range**: a `[-24]` offset on a low fixture drops the whole variant (default
  `policy: drop`); set `policy: clamp` to clamp into the window instead — either way it is recorded.
- **Lyrics**: if a fixture score carries 4th-column syllables, every variant keeps each note's syllable
  (FR-014) — transposition/humanisation never break syllable↔note alignment.
- **Other lanes**: the SVS/voice-conversion lanes render variants too, but ignore per-note gain and record
  `dynamics_applied: false` (FR-008); transposition and humanisation labels apply to them unchanged.
