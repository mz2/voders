# Quickstart: Vocal-Conditioned Accompaniment Lane

Lay instrumental backing under existing corpus vocals without moving the sung-note timing. All
commands go through `uv` (constitution: Python Tooling). The default path uses the **CPU fake
backend**, so it runs with no GPU.

## 1. Install

```bash
uv sync                       # CPU baseline — torch-free, fake accompaniment backend works
uv sync --extra accomp        # add the GPU ACE-Step backend (torch + ACE-Step 1.5 XL)
```

## 2. Run the stage (single command)

```bash
uv run voders run --config evals/fixtures/accompaniment-smoke.yaml
```

This enables the accompaniment stage on the checked-in donor vocals. For each **accepted** base
render it generates accompaniment (Lego or Complete per the config), validates the result **on the
mix**, and writes `mix.wav` (+ `accompaniment_stem.wav` for Lego) and a JSONL manifest line to the
output dir. The smoke config uses `backend: fake` so it needs no GPU.

### Real ACE-Step backend (GPU)

ACE-Step (`ACE-Step/ACE-Step-v1-3.5B`, Apache-2.0) pins deps that conflict with this project and lack
Python 3.14 / aarch64 wheels, so it lives as a **standalone uv project** under `tools/acestep/`
(Python 3.12, its own `uv.lock`) and the backend drives it via subprocess. Build it exactly like the
rest of the project's uv envs:

```bash
cd tools/acestep && uv sync && cd -      # ACE-Step env (weights auto-download from HF on first run)
uv sync --extra cpu --extra accomp       # Demucs (Lego separation) in this project's accomp extra
```

That's it — the backend **auto-detects** `tools/acestep/.venv/bin/python`, so no env var is needed
(override with `VODERS_ACESTEP_PYTHON` if you keep the ACE-Step env elsewhere). Weights cache to
`~/.cache/ace-step`; set `VODERS_ACESTEP_CHECKPOINT` to pin a local checkpoint dir.

> On a host whose torchaudio routes I/O through `torchcodec` without a matching FFmpeg, the runner
> falls back to `soundfile` for audio load/save — no system FFmpeg needed.

To use the real model, set `backend: acestep` in the run config (GPU required):

```yaml
lanes:
  deterministic: { enabled: true }
  accompaniment:
    enabled: true
    mode: lego
    backend: acestep
    model_id: ace-step-1.5-xl-base
    target_instrument: "sustained bowed metal pad"
    free_time: true
    takes: 3
    target_snr_db: [9, 12, 15]
```

## 3. Evaluate it (single command, yes/no verdict)

```bash
uv run voders eval --manifest <out>/manifest.jsonl
```

Prints a pass/fail table mapped to this feature's Success Criteria and exits non-zero on any gated
failure:

| Criterion | Check |
|-----------|-------|
| SC-001 | Lego mix's vocal is byte-identical to the source vocal |
| SC-002 | one-pass onsets ≤ 50 ms / offsets within tolerance on the mix ≥ 99% |
| SC-004 | every admitted sample has complete model/license/attribution/mode/source/seed provenance; 0 disallowed-license; CC-BY-class carries attribution |
| SC-005 | zero measured note-timing shift; free-time samples carry no fixed-tempo metadata |
| SC-008 | mix (+ stem for Lego) stored, sufficient to re-mix |

## 4. Audit licensing

```bash
uv run voders audit --manifest <out>/manifest.jsonl
```

Confirms every accompaniment sample's `model_license` is within policy and that CC-BY-class samples
carry `attribution_text` (FR-006a).

## Notes

- **Free time matters.** For handpan/gong/bowl-style rubato material, keep `free_time: true`, leave
  `bpm` null, and prefer sustained/textural `target_instrument`s — this is what keeps a metrical
  pulse from displacing the phrasing. The gate still enforces it: any sample whose notes shifted is
  rejected.
- **Lego vs Complete.** Lego keeps your vocal bit-exact and retains a re-mixable stem; Complete is
  one pass (mild vocal coloration, no separable stem). Start with Lego.
- Generated mixes/stems are **never committed** to Git (constitution: Binary Assets); only the text
  manifest + resolved config are committable.
