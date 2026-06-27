# Contract: Accompaniment manifest additions

Extends `contracts/manifest-record.schema.json` (from `001`). One new optional object,
`accompaniment`, on `ProvenanceRecord`, present iff `lane == "accompaniment"`. Satisfies FR-007 and
FR-006a. `ProvenanceRecord` is `extra="forbid"`, so this is an explicit typed addition.

```json
{
  "sample_id": "score007_singer_donor3_accomp_lego",
  "score_id": "score007",
  "lane": "accompaniment",
  "voice_id": "donor3",
  "seed": 184467,
  "config_hash": "…",
  "verdict": {
    "status": "accepted",
    "onset_ok": true, "offset_ok": true, "f0_ok": true,
    "snr_db": 12.3,
    "max_onset_dev_ms": 7.0
  },
  "accompaniment": {
    "mode": "lego",
    "model_id": "ace-step-1.5-xl-base",
    "model_version": "1.5-xl-2026-04-02",
    "model_license": "MIT",
    "attribution_text": null,
    "vocal_bit_exact": true,
    "source_vocal_sample_id": "score007_singer_donor3",
    "takes_tried": 3,
    "max_note_shift_ms": 0.0,
    "free_time": true,
    "target_instrument": "sustained bowed metal pad",
    "stem_available": true
  }
}
```

## Field rules

- `mode` ∈ {`lego`,`complete`}. `vocal_bit_exact = (mode == "lego")` (SC-001).
- `model_license` MUST be within the run's `license_policy` for any **admitted** sample (SC-004); a
  disallowed license means the sample was never generated.
- `attribution_text` MUST be non-null for CC-BY-class models and is surfaced by `voders audit`
  (FR-006a, SC-004).
- `max_note_shift_ms` ≤ validator tolerance for any admitted sample; target 0 (SC-005).
- `stem_available = (mode == "lego")` until research open-item 2 is resolved; gates SC-008
  re-mixability to Lego samples (see data-model.md design note).
- `source_vocal_sample_id` MUST reference an existing accepted base render (FR-015).

The existing `audit` / `stats` / `eval` CLI commands read these fields; no new subcommand is added
(`contracts/cli.md` addendum: `eval` reports SC-001/002/004/005/008; `audit` checks license +
attribution completeness).
