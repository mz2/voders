"""ACE-Step accompaniment worker — invoked by the 3.14 core across a process boundary.

ACE-Step (real package ``ace_step`` v0.2.0, Apache-2.0) pins a stack that conflicts with the 3.14
core and has no Python 3.14 / aarch64 wheels, so it runs here in this backend's Python 3.12 env. The
core calls ``uv run --project backends/acestep python -m voders_acestep_backend.worker <spec.json>``
and reads back the written wav (the same out-of-process pattern as the SVS / RVC / Seed-VC workers).

Protocol:
  argv[1] = JSON spec {"ref", "out", "prompt", "seed", "duration", "strength", "checkpoint_dir",
                       "device_id", "cpu_offload", "infer_step", "guidance_scale"}
  On success: writes ``out`` (48 kHz stereo wav) and exits 0.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys


def _patch_torchaudio_io() -> None:
    """Route torchaudio.load/save through soundfile to avoid the torchcodec/FFmpeg backend.

    torchaudio 2.11 dispatches load/save to ``torchcodec``, which needs a matching FFmpeg shared
    library that may be absent. soundfile (libsndfile) covers the wav I/O ACE-Step needs, so swap
    the two functions for soundfile-backed equivalents.
    """
    import soundfile as sf
    import torch
    import torchaudio

    def _load(path, *args, **kwargs):  # noqa: ANN001, ANN202
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (N, C)
        return torch.from_numpy(data.T).contiguous(), sr  # torchaudio returns (C, N)

    def _save(path, tensor, sample_rate, *args, **kwargs):  # noqa: ANN001, ANN202
        import numpy as np

        arr = np.asarray(tensor.detach().cpu().numpy(), dtype="float32")
        if arr.ndim == 2:
            arr = arr.T  # (C, N) -> (N, C)
        sf.write(str(path), arr, int(sample_rate))

    torchaudio.load = _load
    torchaudio.save = _save


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as fh:
        spec = json.load(fh)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.get("device_id", 0))

    _patch_torchaudio_io()
    from acestep.pipeline_ace_step import ACEStepPipeline

    pipe = ACEStepPipeline(
        checkpoint_dir=spec.get("checkpoint_dir") or "",
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=bool(spec.get("cpu_offload", False)),
    )

    out = spec["out"]
    out_dir = os.path.dirname(out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # audio2audio: a non-null ``ref_audio_input`` makes ACE-Step set task="audio2audio" (verified in
    # pipeline_ace_step.__call__). ``manual_seeds`` uses ACE-Step's comma-joined string form.
    pipe(
        format="wav",
        audio_duration=float(spec["duration"]),
        prompt=spec["prompt"],
        lyrics="",
        infer_step=int(spec.get("infer_step", 60)),
        guidance_scale=float(spec.get("guidance_scale", 15.0)),
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10.0,
        manual_seeds=str(int(spec["seed"])),
        audio2audio_enable=True,
        ref_audio_strength=float(spec["strength"]),
        ref_audio_input=spec["ref"],
        save_path=out,
        batch_size=1,
    )

    # ACE-Step may save to ``out`` or to a derived name in its directory; normalize to ``out``.
    if not os.path.exists(out):
        candidates = sorted(
            (p for p in glob.glob(os.path.join(out_dir, "*.wav")) if p != out),
            key=os.path.getmtime,
        )
        if not candidates:
            raise RuntimeError(f"ACE-Step produced no wav in {out_dir}")
        shutil.move(candidates[-1], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
