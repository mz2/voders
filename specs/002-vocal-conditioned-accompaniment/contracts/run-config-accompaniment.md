# Contract: Accompaniment run-config additions

Extends `contracts/run-config.schema.yaml` (from `001`). The accompaniment stage is enabled via the
existing `lanes` map; `LaneToggle` is `extra="allow"`, so these options need **no schema-breaking
change** — they are documented free-form keys read by `config.lane_options("accompaniment")`.

```yaml
# Run config excerpt — enable the accompaniment stage.
lanes:
  deterministic:
    enabled: true            # produces the base vocals the stage layers under (corpus-internal)
  accompaniment:
    enabled: true
    mode: lego               # lego (vocal-preserving) | complete (one-pass)
    backend: fake            # fake (CPU, CI) | acestep (GPU, `accomp` extra)
    model_id: ace-step-1.5-xl-base
    license_policy: [MIT, Apache-2.0, CC-BY-4.0]
    target_instrument: "sustained bowed metal pad"
    free_time: true          # no BPM; free/rubato captioning
    bpm: null                # MUST be null when free_time: true
    takes: 3                 # generate N, admit the best-aligned passing take
    target_snr_db: [9, 12, 15]   # vocal-to-accompaniment ratio(s) for Lego mixing
```

## Rules

- The stage runs **after** a base render is `ACCEPTED`, fanning that accepted vocal through each
  enabled accompaniment config (mirrors the augmentation fan-out in `orchestrator.run()`).
- `enabled: false` (or absent) → stage does nothing; all other lanes unaffected (FR-008).
- `free_time: true` ⟹ `bpm` MUST be null; the validator/manifest record no fixed-tempo metadata
  (FR-005, SC-005).
- `backend: acestep` requires the `accomp` extra installed and a GPU; otherwise the stage no-ops with
  a logged skip (FR-009/FR-013). `backend: fake` always runs (CPU).
- `model_license` outside `license_policy` → no-op with a logged skip; nothing admitted (FR-006).
