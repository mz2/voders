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

# Add the accompaniment GPU extra (Demucs, for Lego-mode source separation).
setup-accomp:
    uv sync --extra cpu --extra accomp

# Prerequisite for the out-of-process NNSVS backend: sync its standalone uv project (Python 3.11).
setup-backends:
    uv sync --project backends/svs

# Sync the out-of-process ACE-Step accompaniment backend (its own uv project, Python 3.12).
setup-acestep-backend:
    uv sync --project backends/acestep

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

# Source-stratified train/val splits (issue #7): hold out per lane/voice/aug profile.
splits manifest=manifest: setup
    uv run voders splits --manifest {{manifest}}

# Render the smoke fixtures then evaluate them end-to-end.
smoke: fixtures
    uv run voders run --config {{config}}
    uv run voders eval --manifest {{manifest}}

# Regenerate the unified donor-pool config from every donor on disk (all data-source methods).
build-pool: setup
    uv run --extra cpu python evals/build_donor_pool.py

# Whole data-augmentation pipeline (what the data-augmentation Action runs). Enrolls donors from
# EVERY source method (synthetic fixtures + VocalSet/VCTK + Freesound; mic recordings join if
# present), unifies them into one pool, renders the deterministic lane, fans each accepted render
# through the augmentation profiles, then audits + evaluates + aggregates. FETCH=0 skips the network
# fetches and runs purely on the checked-in donors (this is what the CI action uses).
augment FETCH="1" FREESOUND_COUNT="5" VOCALSET_SINGERS="18" pool="evals/fixtures/donor_pool.yaml" manifest="out/donor_pool/manifest.jsonl": fixtures
    {{ if FETCH == "1" { "-uv run --extra cpu --extra donors python evals/download_donors.py --vocalset-singers " + VOCALSET_SINGERS } else { "echo 'FETCH=0: using checked-in donors (no dataset download)'" } }}
    {{ if FETCH == "1" { "-uv run --extra cpu python evals/download_freesound.py --count " + FREESOUND_COUNT + " --license cc0" } else { "echo 'FETCH=0: using checked-in Freesound donors (no download)'" } }}
    rm -rf out/donor_pool
    uv run --extra cpu python evals/build_donor_pool.py --out {{pool}}
    @just list-donors
    uv run voders run --config {{pool}}
    uv run voders audit --manifest {{manifest}}
    uv run voders eval --manifest {{manifest}}
    uv run voders stats --manifest {{manifest}}

# Like `augment`, but ALSO lays vocal-conditioned accompaniment (spec 002) under each accepted donor
# render — including the real VocalSet singer — as a training augmentation. BACKEND=fake is CPU/CI;
# BACKEND=acestep uses the real GPU model (run `just setup-acestep-backend` first).
augment-accomp BACKEND="fake" FETCH="1" FREESOUND_COUNT="5" VOCALSET_SINGERS="18" pool="evals/fixtures/donor_pool_accomp.yaml" manifest="out/donor_pool_accomp/manifest.jsonl": fixtures setup-accomp
    {{ if FETCH == "1" { "-uv run --extra cpu --extra donors python evals/download_donors.py --vocalset-singers " + VOCALSET_SINGERS } else { "echo 'FETCH=0: using checked-in donors (no dataset download)'" } }}
    {{ if FETCH == "1" { "-uv run --extra cpu python evals/download_freesound.py --count " + FREESOUND_COUNT + " --license cc0" } else { "echo 'FETCH=0: using checked-in Freesound donors (no download)'" } }}
    rm -rf out/donor_pool_accomp
    uv run --extra cpu python evals/build_donor_pool.py --accompaniment {{BACKEND}} --run-id donor_pool_accomp --out {{pool}}
    @just list-donors
    uv run --extra cpu --extra accomp voders run --config {{pool}}
    uv run --extra cpu --extra accomp voders audit --manifest {{manifest}}
    uv run --extra cpu --extra accomp voders eval --manifest {{manifest}}
    uv run --extra cpu --extra accomp voders stats --manifest {{manifest}}

# Pack the rendered corpus into a committable OGG archive (~17x smaller, LFS) under datasets/<run_id>/.
pack-corpus RUN_ID="donor_pool": setup
    uv run --extra cpu python evals/corpus_archive.py pack --run-id {{RUN_ID}}

# Decompress the committed OGG archive back to WAV at out/<run_id>/corpus/ for the training task.
unpack-corpus RUN_ID="donor_pool": setup
    uv run --extra cpu python evals/corpus_archive.py unpack --run-id {{RUN_ID}}

# Train the SoulX Basic Pitch baseline and stream metrics and media to Weights & Biases.
# Stages the committed datasets — the deterministic donor_pool AND the lyric/SVS lyrics_pool — into
# the trainer's flat input dir, so the current vocal-synthesis-with-lyrics audio is trained on.
train RUN_IDS="donor_pool lyrics_pool":
    uv sync --extra cpu --extra gpu --extra training
    uv run --extra cpu python evals/corpus_archive.py stage --run-ids {{RUN_IDS}} --out syntheticdataset_soulx
    bash training/train_soulx.sh

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

# Stream real donor voices (VocalSet + VCTK, CC BY 4.0) into models/donors/. SINGERS=N enrolls N
# distinct VocalSet singers (~20 available); the vocalset_*.wav files are versioned via LFS.
download-donors SINGERS="1":
    uv sync --extra cpu --extra donors
    uv run --extra cpu --extra donors python evals/download_donors.py --vocalset-singers {{SINGERS}}

# Fetch CC0/CC-BY donor vowels from Freesound (needs FREESOUND_API_TOKEN). QUERY=/COUNT=/LICENSE= optional.
download-freesound QUERY="sung vowel" COUNT="3" LICENSE="cc0": setup
    uv run --extra cpu python evals/download_freesound.py \
        --query {{quote(QUERY)}} --count {{COUNT}} --license {{LICENSE}}

# Show the donor voices fetched/enrolled under models/donors/ (counts, sizes, licenses).
list-donors:
    @echo "== donor voices under models/donors/ (git-ignored) =="
    @find models/donors -name '*.wav' 2>/dev/null | sort | sed 's|models/donors/|  |' || true
    @echo "  ($(find models/donors -name '*.wav' 2>/dev/null | wc -l | tr -d ' ') WAV(s), $(du -sh models/donors 2>/dev/null | cut -f1 || echo 0) on disk)"
    @if [ -f models/donors/freesound/ATTRIBUTION.txt ]; then \
        echo ""; echo "== Freesound attribution / licenses =="; \
        column -t -s$'\t' models/donors/freesound/ATTRIBUTION.txt; \
    fi

# Enroll your own consented donor vowel. Import a WAV (FILE=...) or record from the mic (RECORD=1).
record-donor VOICE_ID FILE="" RECORD="": setup
    uv run --extra cpu {{ if RECORD != "" { "--extra record" } else { "" } }} \
        python evals/record_donor.py --voice-id {{VOICE_ID}} \
        {{ if RECORD != "" { "--record" } else { "--input " + FILE } }}

# Render the deterministic lane with a real VocalSet donor voice, then evaluate.
demo-real-donor: download-donors
    uv run voders run --config evals/fixtures/real_donor.yaml
    uv run voders eval --manifest out/real_donor/manifest.jsonl

# Real end-to-end RVC voice conversion with the consented VCTK p231 voice.
demo-rvc: setup setup-rvc-backend download-rvc-models download-rvc-voice
    uv run voders run --config evals/fixtures/rvc_vctk.yaml
    uv run voders eval --manifest out/rvc_vctk/manifest.jsonl

# Sync the out-of-process Seed-VC backend (zero-shot VC; its own uv project, Python 3.10).
setup-seedvc-backend:
    uv sync --project backends/seedvc

# Zero-shot Seed-VC voice conversion using a real VocalSet reference clip.
demo-seedvc: setup setup-seedvc-backend download-donors
    uv run voders run --config evals/fixtures/seedvc.yaml
    uv run voders eval --manifest out/seedvc/manifest.jsonl

# EXPERIMENTAL: deterministic lane re-vocoded through Vocos (needs the gpu extra).
# Note: mel Vocos is not f0-conditioned, so the alignment gate currently rejects its output —
# see src/voders/render/vocos_enhance.py. Kept as a runnable experiment, not a default.
demo-vocos: setup-gpu
    uv run --extra cpu --extra gpu voders run --config evals/fixtures/vocos.yaml
    uv run --extra cpu --extra gpu voders eval --manifest out/vocos/manifest.jsonl

# Sync the consumer-side Basic Pitch eval backend (its own uv project, Python 3.11).
setup-basicpitch-backend:
    uv sync --project backends/basicpitch

# Consumer-side eval: run Basic Pitch on a produced corpus, report note-F1 per source
# (does the drift actually matter to the downstream transcriber?). Override: `just basic-pitch-eval manifest=out/<run>/manifest.jsonl`.
basic-pitch-eval manifest=manifest: setup-basicpitch-backend
    uv run --project backends/basicpitch python -m voders_basicpitch_backend.worker {{manifest}}

# Render + validate with the CREPE neural f0 estimator on the GPU (needs the `gpu` extra).
smoke-gpu: setup-gpu
    uv run --extra cpu --extra gpu python evals/make_fixtures.py
    uv run --extra cpu --extra gpu voders run --config evals/fixtures/smoke_gpu.yaml
    uv run --extra cpu --extra gpu voders eval --manifest out/smoke_gpu/manifest.jsonl

# Lay accompaniment under the deterministic vocals with the CPU fake backend, then evaluate.
accompaniment-smoke: setup
    uv run voders run --config evals/fixtures/accompaniment-smoke.yaml
    uv run voders eval --manifest out/accompaniment_smoke/manifest.jsonl

# Real ACE-Step accompaniment on a vocal fixture (GPU; downloads ACE-Step weights on first run).
demo-acestep: setup-accomp setup-acestep-backend
    uv run --extra cpu --extra accomp voders run --config evals/fixtures/accompaniment_acestep.yaml
    uv run --extra cpu --extra accomp voders eval --manifest out/accompaniment_acestep/manifest.jsonl

# Everything CI checks: lint, type, tests.
check: lint test
