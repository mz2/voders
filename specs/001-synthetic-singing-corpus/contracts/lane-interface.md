# RendererLane Interface Contract

The renderer boundary that makes the four lanes independently enable/disable/replaceable (FR-015).
Every lane implements this contract; the orchestrator depends only on it.

```python
class RendererLane(Protocol):
    name: str  # "deterministic" | "svs" | "voice_conversion" | "augmentation"

    def requires_gpu(self) -> bool:
        """deterministic and the validator MUST return False (FR-009)."""

    def render(self, req: RenderRequest) -> RenderResult:
        ...
```

## RenderRequest

| Field | Type | Notes |
|-------|------|-------|
| `score` | Score | Driving score (FR-001) |
| `voice` | Voice | Must have `consent_verified=true`, else orchestrator refuses before calling (FR-011) |
| `seed` | int | Per-sample seed derived from master seed (FR-013) |
| `options` | map | Lane-specific options from RunConfig.lanes |

## RenderResult

| Field | Type | Notes |
|-------|------|-------|
| `audio` | float32 ndarray | 22,050 Hz mono (FR-002) |
| `label_score` | Score | Score the audio is labelled by — the input score, or a re-derived score for the expressive `rederive_labels` mode (FR-007) |
| `notes` | map | Lane diagnostics (e.g., `max_onset_dev_ms`) merged into the verdict |

## Invariants

- **Deterministic lane:** `label_score == input score` (pitch/onset/offset exact by construction,
  FR-003); `requires_gpu() == False`.
- **Voice-conversion lane:** preserves the input score's f0 (`auto_predict_f0=False`); `label_score
  == input score` (FR-004).
- **Augmentation lane:** input is a rendered sample; never alters `label_score` note rows (FR-005);
  may set `snr_db` for the quarantine gate (FR-014).
- **SVS lane:** in `force_score_f0` mode `label_score == input score`; in `rederive_labels` mode
  `label_score` is re-derived from rendered audio and the deviation reported (FR-007, SC-010).
- Every `RenderResult` is passed to the validator before admission; no lane writes to `corpus/`
  directly (FR-006).
