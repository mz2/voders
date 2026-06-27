# voders task runner. Every action goes through uv (Constitution: Python Tooling — uv).
# Run `just` (or `just --list`) to see all actions.
#
# `setup` is the prerequisite action; recipes that need the CPU environment depend on it.
# `uv sync` is a fast no-op when the environment is already current, so depending on it is cheap.

set shell := ["bash", "-uc"]

config := "evals/fixtures/smoke.yaml"
manifest := "out/smoke/manifest.jsonl"

# List available actions.
default:
    @just --list

# Prerequisite: create/update the CPU environment (Python 3.14 + CPU deps). Fast no-op if current.
setup:
    uv sync --extra cpu

# Add the GPU extra (torch / torchaudio / torchcrepe) on a GPU machine.
setup-gpu:
    uv sync --extra cpu --extra gpu

# Prerequisite for the out-of-process NNSVS backend: sync its standalone uv project (Python 3.11).
setup-backends:
    uv sync --project backends/svs

# Generate the checked-in fixtures (scores + synthetic consented donor voices).
fixtures: setup
    uv run python evals/make_fixtures.py

# Render a corpus from a run config (override: `just run config=path.yaml`).
run config=config: setup
    uv run voders run --config {{config}}

# Evaluate a produced corpus against the Success Criteria (exits non-zero on a gated failure).
eval manifest=manifest: setup
    uv run voders eval --manifest {{manifest}}

# License/consent audit over the manifest.
audit manifest=manifest: setup
    uv run voders audit --manifest {{manifest}}

# Aggregate corpus statistics.
stats manifest=manifest: setup
    uv run voders stats --manifest {{manifest}}

# Render the smoke fixtures then evaluate them end-to-end.
smoke: fixtures
    uv run voders run --config {{config}}
    uv run voders eval --manifest {{manifest}}

# Run the full test suite.
test: setup
    uv run pytest

# Lint + format check + type check (zero-warning gate).
lint: setup
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy

# Auto-format the codebase.
fmt: setup
    uv run ruff format .

# Deterministic-lane throughput benchmark against the SC-005 floor.
bench: setup
    uv run python evals/bench.py

# Demonstrate the out-of-process NNSVS backend (its own uv project / Python 3.11).
demo-svs-nnsvs: setup setup-backends
    uv run voders run --config evals/fixtures/svs_nnsvs.yaml

# Download the RVC base model weights (HuBERT + RMVPE) into models/ (git-ignored).
download-rvc-models: setup
    uv run python evals/download_models.py rvc

# Sync the out-of-process RVC voice-conversion backend (its own uv project, Python 3.10).
setup-rvc-backend:
    uv sync --project backends/rvc

# Download a consented RVC target voice (VCTK p231; Apache-2.0 model, CC BY 4.0 dataset).
download-rvc-voice: setup
    uv run python evals/download_models.py vctk-p231

# Stream real donor voices (VocalSet + VCTK, CC BY 4.0) into models/donors/ (git-ignored).
download-donors:
    uv sync --extra cpu --extra donors
    uv run --extra cpu --extra donors python evals/download_donors.py

# Render the deterministic lane with a real VocalSet donor voice, then evaluate.
demo-real-donor: download-donors
    uv run voders run --config evals/fixtures/real_donor.yaml
    uv run voders eval --manifest out/real_donor/manifest.jsonl

# Real end-to-end RVC voice conversion with the consented VCTK p231 voice.
demo-rvc: setup setup-rvc-backend download-rvc-models download-rvc-voice
    uv run voders run --config evals/fixtures/rvc_vctk.yaml
    uv run voders eval --manifest out/rvc_vctk/manifest.jsonl

# Render + validate with the CREPE neural f0 estimator on the GPU (needs the `gpu` extra).
smoke-gpu: setup-gpu
    uv run --extra cpu --extra gpu python evals/make_fixtures.py
    uv run --extra cpu --extra gpu voders run --config evals/fixtures/smoke_gpu.yaml
    uv run --extra cpu --extra gpu voders eval --manifest out/smoke_gpu/manifest.jsonl

# Everything CI checks: lint, type, tests.
check: lint test
