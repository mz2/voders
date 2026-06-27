---
name: world-render-determinism
description: WORLD-vocoder audio is bit-exact in-process but NOT across separate process runs; test determinism via labels/in-process re-render
metadata:
  type: project
---

The deterministic WORLD/pyworld renderer (`src/voders/render/deterministic.py`) produces
**bit-exact audio when re-rendered within the same Python process** (this is what
`reproduce_sample` + `test_reproduce_accepted_sample_is_bit_exact` assert via `np.array_equal`),
but **two separate `voders run` invocations of the same config produce differing `.wav` bytes** —
pyworld/FFT threading is not reproducible across processes. The `.tsv` label files ARE byte-identical
across runs.

**Why:** matters for any "byte-identical" success criterion (e.g. feature 003 SC-001 feature-off
control, SC-005 re-expansion determinism).

**How to apply:** Verify byte-identity at the **label (`.tsv`) / score-row level** and via
**in-process** re-render or re-expansion (like `reproduce_sample`). Do NOT diff `.wav` files produced
by two separate CLI runs and expect equality. The pure `voders.scoreaug.expand()` output is fully
byte-identical (numpy/integer), so determinism criteria should lean on the expanded scores/labels.
