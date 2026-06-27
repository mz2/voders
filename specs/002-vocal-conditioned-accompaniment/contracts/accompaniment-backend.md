# Contract: AccompanimentBackend Protocol

The boundary between the accompaniment stage (`render/accompaniment.py`) and a concrete generator.
Two implementations: the torch-free CPU **fake** (deterministic, CI/tests) and the GPU **ACE-Step**
backend (lazy torch, `accomp` extra). The stage depends only on this Protocol — backends are swapped
by config (`backend: fake | acestep`).

```python
from typing import Protocol, runtime_checkable
import numpy as np
from dataclasses import dataclass

@dataclass
class BackendOutput:
    accompaniment_stem: np.ndarray | None  # 22,050 mono float32 (Lego); None for Complete
    mix: np.ndarray | None                 # 22,050 mono float32 (Complete); None for Lego
    native_sr: int                         # model's native rate (e.g. 48000), provenance only

@runtime_checkable
class AccompanimentBackend(Protocol):
    name: str
    model_id: str
    model_version: str
    model_license: str            # checked against license_policy BEFORE generate() (FR-006)
    attribution_text: str | None  # CC-BY-class credit, propagated to provenance (FR-006a)

    def requires_gpu(self) -> bool: ...
    def supports(self, mode: str) -> bool: ...        # "lego" / "complete"
    def generate(
        self,
        vocal: np.ndarray,        # 22,050 mono float32 (the conditioning vocal)
        score: "Score",           # the vocal's correct-by-construction labels
        mode: str,                # "lego" | "complete"
        seed: int,                # derived per (sample, mode, take); reproducibility tier per Decision 4
        options: dict,            # target_instrument, free_time, bpm, target_snr_db, ...
    ) -> BackendOutput: ...
```

## Conformance rules

1. **Output already bridged**: `generate` returns 22,050 Hz mono float32 (Decision 3). The backend
   does the 48 kHz-stereo → 22,050-mono down-mix/resample internally; the stage never sees 48 kHz.
2. **Mode/return invariants**: `mode="lego"` → `accompaniment_stem` set, `mix is None`;
   `mode="complete"` → `mix` set, `accompaniment_stem is None`. The stage asserts this.
3. **Determinism**: the **fake** backend is bit-exact from `seed` (unit tests assert exact arrays).
   The **acestep** backend need only satisfy the neural reproducibility tier (same verdict on replay).
4. **No torch at module load** for the CPU baseline: `acestep.py` imports torch/ACE-Step **inside**
   methods, never at module top level (FR-009). `requires_gpu()` returns `True`.
5. **License honesty**: `model_license` / `attribution_text` are declared truthfully; the stage
   refuses to call `generate` when `model_license` is outside `license_policy` (FR-006).
6. **`supports(mode)`**: a backend that cannot do a requested mode returns `False`; the stage no-ops
   that config with a logged skip (FR-013).

## Fake backend (test baseline)

Deterministic, torch-free. Lego: a seeded band-limited noise pad shaped to the score's note spans
(so it is plausibly "related" and onset-aligned), returned as the stem. Complete: that pad summed
under a copy of the input vocal as the `mix`. `model_license="MIT"`, `attribution_text=None`,
`requires_gpu()=False`, `supports("lego")=supports("complete")=True`. This lets every Success
Criterion except real-model fidelity be exercised in CPU CI.
