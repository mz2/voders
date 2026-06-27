# Quickstart: Synthetic Singing Corpus Generator

The single-command run + single-command eval required by Constitution Principle IV (eval-first).
The deterministic lane and validator run on a laptop CPU — no GPU needed.

## Install

`voders` targets Python 3.14 and is managed with [uv](https://docs.astral.sh/uv/); every command
runs through `uv run`.

```bash
# core (CPU): deterministic lane + validator + manifest
uv sync --extra cpu

# optional GPU lanes (neural SVS, voice conversion) — on the DGX Spark machines
uv sync --extra gpu
```

Git LFS must be installed so the fixture audio resolves (`git lfs install` once per clone).

## Generate the fixtures

```bash
uv run python evals/make_fixtures.py
```

## Run the deterministic baseline on the fixtures

```bash
uv run voders run --config evals/fixtures/smoke.yaml --output-root out/smoke
```

Produces under `out/smoke/`:

```text
config.resolved.yaml   # committable end-result record
manifest.jsonl         # one provenance row per attempted sample
stats.json             # aggregate corpus statistics
corpus/shard=000/      # score_*.wav + byte-identical score_*.tsv (accepted)
rejected/shard=000/    # non-accepted samples (audio + reason), never trained on
checkpoints/           # resume state (git-ignored)
```

## Evaluate against the Success Criteria

```bash
uv run voders eval --manifest out/smoke/manifest.jsonl
```

Prints a pass/fail table and exits non-zero on any gated failure:

| Criterion | Check |
|-----------|-------|
| SC-001 | onset within 50 ms ≥ 99% |
| SC-002 | offset within max(50 ms, 20%) ≥ 99% |
| SC-007 | first-attempt validator pass ≥ 95% |
| SC-008 | zero `consent_verified=false` voices in accepted corpus |
| SC-009 | re-render a sampled record in isolation, matches within tolerance |

SC-003 (downstream note-F1 ≥ 0.15) is a consumer-side training metric and is reported
informationally when a Basic Pitch eval harness is wired up; it is not gated here.

## Audit licensing

```bash
uv run voders audit --manifest out/smoke/manifest.jsonl
```

## Scale up

Point `scores:` at a larger set, enable the `voice_conversion` and `augmentation` lanes in the
config, raise the voice pool, and re-run. Use `--resume` to continue an interrupted 10k–100k-sample
run from the last completed shard.
