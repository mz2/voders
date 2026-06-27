# voders

`voders` turns folders of singing scores into a training corpus of `(audio.wav, score.tsv)`
pairs whose labels are **correct by construction**. Each score is a list of
`(onset_s, offset_s, pitch_midi)` note rows; the pipeline renders singing audio from it and
keeps the score as the ground-truth label, so the label never has to be guessed back from the
audio. An alignment validator gates every sample, and a JSON Lines manifest records full
provenance (which voice, seed, license, and config produced each pair) for replay and audit.

The intended consumer is a transcription model that ingests 22,050 Hz mono float32 audio, so
every rendered sample uses that format.

## The four renderer lanes

Each lane is enabled or disabled independently in the run config:

1. **deterministic** — drives the pitch directly from the score with the WORLD vocoder (a classic
   analysis/resynthesis vocoder that lets us replace a donor voice's pitch with a score-derived
   contour). This is the load-bearing CPU baseline: f0 (the fundamental frequency, i.e. the sung
   pitch) comes straight from the label, so alignment is exact.
2. **voice_conversion** — keeps the score-accurate timing and pitch but converts the timbre toward
   a target singer, for variety in voice identity.
3. **svs** — expressive neural singing-voice synthesis (SVS), which sings the score with natural
   phrasing; labels are re-derived and re-validated so expressive timing stays within tolerance.
4. **augmentation** — label-preserving production-style effects (noise, room, codec) applied to an
   accepted base render, multiplying the corpus without changing the score.

The deterministic lane and validator run on a laptop CPU with no GPU. The neural lanes
(`svs`, `voice_conversion`) use the `gpu` extra.

## Quickstart (uv)

`voders` targets Python 3.14 and is managed with [uv](https://docs.astral.sh/uv/). All commands
run through `uv run`.

```bash
# install the CPU core (deterministic lane + validator + manifest)
uv sync --extra cpu

# generate the checked-in test fixtures (scores + a synthetic consented donor voice)
uv run python evals/make_fixtures.py

# render the deterministic baseline over the fixtures
uv run voders run --config evals/fixtures/smoke.yaml

# evaluate the produced corpus against the Success Criteria
uv run voders eval --manifest out/smoke/manifest.jsonl

# license/consent audit and aggregate statistics
uv run voders audit --manifest out/smoke/manifest.jsonl
uv run voders stats --manifest out/smoke/manifest.jsonl
```

`uv run voders run` writes under `out/smoke/`: `config.resolved.yaml` (the committable end-result
record), `manifest.jsonl` (one provenance row per attempted sample), `stats.json`, `corpus/`
(accepted `wav`+`tsv` pairs), and `rejected/` (non-accepted samples, never trained on). A run
streams and checkpoints, so an interrupted large run resumes with `--resume`.

### GPU and neural backends

`uv sync --extra gpu` installs `torch`, `torchaudio`, and `torchcrepe` — these have Python 3.14
wheels and run on this hardware (verified: CUDA available on an NVIDIA GB10). `torchcrepe` is the
GPU CREPE pitch estimator the validator can use in place of the CPU `pyin` fallback.

The expressive lanes' production model toolkits install with varying ease and are therefore wired
as lazily-imported backends (selected per lane in the run config), not core dependencies:

| Toolkit | Lane | Install on this box |
|---------|------|---------------------|
| Montreal Forced Aligner (MFA) | svs re-derive | `uv pip install montreal-forced-aligner` — works on Python 3.14 |
| NNSVS | svs | pip-installable, but its `numba`/`llvmlite` pins need Python ≤3.11 |
| rvc-python | voice_conversion | pip-installable, but its `numpy` pin needs Python ≤3.11 |
| so-vits-svc, DiffSinger | voice_conversion / svs | GitHub repos + model weights (DiffSinger via the OpenUTAU app), not PyPI packages |

The CPU backends (`backend: world` / `backend: cpu`) reproduce each lane's contract without a GPU,
so the whole pipeline and its evaluation run on a laptop; swap in a GPU backend on a machine where
its toolkit is installed.

## Running tests / development

Everything runs through uv; no extra environment setup is needed on Ubuntu. A [`just`](https://github.com/casey/just)
task runner wraps the common actions — `just` lists them, and every action depends on `setup`
(`uv sync`, a fast no-op when already current), so a fresh checkout needs no manual prep:

```bash
just            # list all actions
just setup      # create/update the CPU environment (Python 3.14 + CPU deps)
just smoke      # render the fixtures, then evaluate them end-to-end
just test       # full test suite
just lint       # ruff check + ruff format --check + mypy (zero-warning gate)
just bench      # deterministic-lane throughput vs the SC-005 floor
just check      # lint + test (what CI runs)
just run config=path.yaml          # render a corpus from a run config
just eval manifest=out/.../manifest.jsonl
```

The equivalent raw commands (if you prefer not to use `just`) are `uv sync --extra cpu`,
`uv run pytest`, `uv run ruff check .`, `uv run mypy`, `uv run voders run --config …`, etc.

Building from source needs a C/C++ toolchain for the `pyworld` wheel (`sudo apt install
build-essential`); the resulting system-linked wheel needs no library-path tweaks at runtime.

### Out-of-process backends (`backends/`)

Lane toolkits whose dependency chains conflict with the 3.14 core live in their own uv projects
under `backends/` (own `pyproject.toml`, `.python-version`, `uv.lock`). The core invokes them with
`uv run --project backends/<name>` and exchanges a JSON request plus a WAV, so the incompatible
chains never share an interpreter. `backends/svs` is the NNSVS backend (Python 3.11); run the
demo with `just demo-svs-nnsvs`.

## Reproducibility

A single run-level `master_seed` derives every per-sample seed, so any one sample reproduces in
isolation from the manifest plus the source scores. The deterministic and voice-conversion lanes
reproduce bit-for-bit; the neural lanes reproduce to within the validator's tolerances.
