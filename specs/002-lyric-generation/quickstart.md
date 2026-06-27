# Quickstart: Optional Lyric Generation & Phonetic Diversity

Single-command run + single-command evaluation for the lyric layer, per Constitution Principle IV.
All commands go through `uv` (Constitution: Python Tooling). The default pipeline is unchanged — you
only get lyrics when you opt in.

## Prerequisites

- The 001 setup (`uv sync`).
- **espeak-ng** system library for G2P (grapheme-to-phoneme), used only when articulating lyrics:

  ```bash
  sudo apt install espeak-ng        # Debian/Ubuntu
  uv sync --extra lyrics            # adds `phonemizer` (the lyrics extra)
  ```

  The CPU baseline and lyric-free runs need neither; espeak-ng/phonemizer import lazily.

## 1. Lyric-free run is unchanged (backward compatibility, SC-001)

```bash
uv run voders run --config evals/fixtures/smoke.yaml --out /tmp/vowel
# No `lyrics` block → source defaults to `vowel`; audio + 3-column scores are byte-identical
# to the pre-feature pipeline.
```

## 2. Automatic phonetic diversity (the corpus workhorse, US2)

```yaml
# evals/fixtures/lyrics-smoke.yaml  (excerpt)
lyrics:
  source: automatic        # seeded CV-syllable sampler; needs no lyric input data
  inventory: en_cv
lanes:
  svs: { enabled: true, backend: cpu, mode: rederive_labels }   # only the SVS lane articulates
```

```bash
uv run voders run --config evals/fixtures/lyrics-smoke.yaml --out /tmp/lyrics
# Every note gets a singable syllable; the SVS lane sings phonemes under the FR-007 safety net.
```

## 3. Supplied lyrics (US1)

Add a 4th tab-separated column to a score (`onset  offset  pitch  syllable`); 3-column scores still
parse unchanged:

```text
0.20	0.70	60	la
0.80	1.30	64	di
```

```bash
uv run voders run --config evals/fixtures/lyrics-supplied.yaml --out /tmp/supplied
```

## 4. Generated themed lyrics (optional, US4)

```yaml
lyrics:
  source: generated
  theme: "winter, longing"      # the only place theming enters; never read from annotations
  model: { model_id: songcomposer-sft, license: "apache-2.0", license_ok: true, model_ref: "..." }
```

```bash
uv run voders run --config evals/fixtures/lyrics-generated.yaml --out /tmp/gen
# The model runs once; generated text is pinned to /tmp/gen/lyrics/<hash>.jsonl and reused on replay.
```

## 5. Evaluate (single-command yes/no verdict)

```bash
uv run voders eval --manifest /tmp/lyrics/manifest.jsonl --suite lyrics
```

Prints a pass/fail table mapped to the spec's Success Criteria and exits non-zero on any gated
failure:

| Check | Criterion |
|-------|-----------|
| lyric-free byte-identity | SC-001 |
| onset ≤50 ms / offset within tolerance on ≥99% of accepted lyric SVS samples | SC-002 |
| distinct sung-phoneme inventory ≥10× the vowel baseline | SC-003 |
| identical syllables across a 1-worker vs N-worker re-run | SC-004 |
| zero dropped/added/shifted note labels across count mismatches | SC-005 |
| every record has `lyric_source` + `lyric_hash`; zero license-refused models | SC-006 |
| generated run replays from the pinned artifact, zero model re-invocations | SC-007 |
| no pitch shift vs lyric-free baseline (same seed): within ±25 cents | SC-008 |
| no timing shift vs lyric-free baseline (same seed): onset/offset within 10 ms | SC-009 |

The last two are a **differential** check: the harness renders each fixture score twice for the same
seed — once with lyrics, once lyric-free — and asserts adding vocal synthesis moved neither the pitch
nor the labeled onset/offset of any accepted note (FR-018). It runs with:

```bash
uv run voders eval --manifest /tmp/lyrics/manifest.jsonl --suite lyrics --differential /tmp/vowel/manifest.jsonl
```

## What changed vs. 001

- `Note` gains an optional `lyric`; `.tsv` gains an optional 4th column (lyric-free output stays
  3-column).
- New `voders.lyrics` package (sources, sampler, G2P, cache, coverage) — CPU only.
- The SVS lane articulates phonemes when a lyric plan is present; **the deterministic and
  voice-conversion lanes are untouched**.
- The manifest gains a lyric provenance axis; `RunConfig` gains an optional `lyrics` block.
- An optional `backends/lyrics/` uv project hosts the generated model out of process.
