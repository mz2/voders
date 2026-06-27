# Contract: Score-Augmentation Pre-Processor

The pure score-to-scores transform consumed by the orchestrator before rendering. All functions are
deterministic given their inputs (FR-010) and pure-CPU (no I/O, no model, no GPU).

## `expand(base, profile, seed) -> list[ScoreVariant]`

```text
expand(base: ParsedScore, profile: ScoreAugmentationProfile, seed: int) -> list[ScoreVariant]
```

- Returns **only the variants** the profile's enabled axes produce (the base score is emitted separately
  by the orchestrator, never wrapped as a `ScoreVariant`).
- `seed` is `derive_seed(master_seed, base.score.score_id, "score_aug", profile.profile_id)`; each axis
  derives its own sub-seed from it (`+ axis [+ draw_index]`) so axes and draws are independent and
  reproducible regardless of worker count or order.
- Axis order is fixed (`transpose`, then `humanize`, then `volume`) so the output list is deterministic.
- Each returned `ScoreVariant.score` is a **valid input score**: monophonic, ordered, strictly positive
  durations, every `pitch_midi ∈ [0,127]`, note count = base, each note's `lyric` carried 1:1.
- Composition in v1 is **per-axis** (each axis emits its own variants from the base); cross-axis products
  (e.g. transpose×humanize on one variant) are out of scope and noted as a future extension.

### Guarantees

| # | Guarantee | Spec |
|---|-----------|------|
| G1 | Same `(base, profile, seed)` ⇒ byte-identical variant list | FR-010, SC-005 |
| G2 | Every emitted note pitch ∈ `[0,127]`; out-of-window handled by policy | FR-005, SC-002 |
| G3 | Every emitted score is valid monophonic (parser accepts unmodified) | FR-007, SC-003 |
| G4 | Note count, ordering, and per-note `lyric` preserved | FR-014 |
| G5 | A variant's labels are its note rows; no audio re-derivation to form it | FR-003 |

## Axis functions

### `transpose(score, offset, *, policy, window) -> Score | None`

- Shifts every `pitch_midi` by `offset` semitones.
- `policy="drop"` (default): returns `None` if any shifted pitch leaves `window` (default `(0,127)`) —
  the variant is dropped; reason recorded by the caller.
- `policy="clamp"`: clamps offending pitches into `window`; records a `clamped` count.
- Onsets/offsets/lyrics unchanged; one variant per offset, `transform = f"t{offset:+d}"`.

### `humanize(score, knob, sub_seed) -> Score`

- Draws onset/offset jitter `~ N(0, sigma)`, clamps each |deviation| ≤ `max_dev_s`.
- Projects left-to-right to preserve order + non-overlap (`overlap="forbid"`) + `duration ≥ min_dur_s`.
- A note that cannot satisfy projection within budget keeps base timing; increments `constraint_hit`.
- One variant per draw `0..draws-1`, `transform = f"hum{draw}"`. Pitch/lyric unchanged.

### `volume(score, knob, sub_seed) -> Score`

- Assigns each note a seeded `gain` within `knob.gain_db_range` (converted to linear).
- `(onset, offset, pitch, lyric)` unchanged — label-preserving (FR-009). `transform = "vol"`.

## Determinism test vector (contract test)

```text
base: 3 notes, score_id="t1"
profile: transpose.offsets=[+12], humanize{sigma=0.02,max_dev=0.05,draws=2}, volume{range=[-6,+6] dB}
seed: derive_seed(42, "t1", "score_aug", "p")
assert expand(base, profile, seed) == expand(base, profile, seed)   # G1 byte-identical
assert all(0 <= n.pitch_midi <= 127 for v in out for n in v.score.notes)            # G2
assert all(parse accepts v.score for v in out)                                      # G3
assert all(len(v.score.notes) == 3 and [n.lyric] preserved for v in out)            # G4
```
