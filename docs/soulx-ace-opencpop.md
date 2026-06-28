# Generate a SoulX synthetic singing dataset from ACE-Opencpop

This workflow turns ACE-Opencpop's symbolic Mandarin singing annotations into a new synthetic
training corpus rendered by SoulX-Singer. The final samples use the same pair expected by the
MML26 Basic Pitch training script:

```text
ACE-Opencpop note metadata
        -> onset/offset/MIDI score.tsv
        -> SoulX-Singer audio generation
        -> f0-based timing re-alignment and validation
        -> accepted audio.wav + score.tsv pairs
        -> optional OGG archive and syntheticdataset_soulx staging
```

The importer reads only projected Parquet metadata columns. It does **not** download the roughly
43 GB ACE-Opencpop audio column, because this workflow needs ACE's notes and timing—not its source
recordings. SoulX creates the audio from scratch.

## Licensing and responsible use

- [ACE-Opencpop](https://huggingface.co/datasets/espnet/ace-opencpop-segments) is **CC BY-NC 4.0**.
  Its score annotations are source material for this corpus. This workflow conservatively treats
  generated and packaged output as non-commercial; preserve attribution and obtain legal guidance
  before relying on a different interpretation. The importer requires `--accept-license`.
- [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) code and weights are Apache-2.0. Its
  upstream usage policy prohibits unauthorized impersonation and deceptive audio.
- The default uses SoulX's bundled Mandarin prompt. Do not replace it with a person's voice unless
  you have explicit consent and the right to create and redistribute derived recordings.
- Generated audio and downloaded weights are intentionally gitignored. Review your institution's
  policies before publishing a corpus or model trained from it.

## What the workflow uses

- `evals/prepare_ace_opencpop.py`: range-reads ACE Parquet shards and writes compatible scores.
- `configs/soulx_ace_opencpop.yaml`: a safe SoulX render configuration with audio-derived labels.
- `zh_cv`: an authored Mandarin syllable inventory used instead of ACE's tone-less pinyin.
- `just` recipes for backend setup, score preparation, rendering, and corpus packing.

ACE's `note_lyrics` values are phonetic fragments without Mandarin tones. SoulX's frontend expects
Chinese characters or tone-bearing pinyin, so feeding those values directly would silently produce
incorrect or empty phonemes. The checked-in `zh_cv` inventory supplies deterministic, one-character
Mandarin syllables. ACE still provides melody, note timing, phrasing, and rests; the lyric content is
newly authored synthetic material.

## Prerequisites

You need Git, Git LFS, `uv`, `just`, an NVIDIA CUDA GPU, space for the model and output, and network
access to GitHub and Hugging Face. Set `HF_TOKEN` in `.env` or the environment for better Hugging
Face range-read limits.

The repository core uses Python 3.14 while the isolated SoulX backend uses Python 3.10. `uv` keeps
those incompatible dependency stacks separate.

## 1. Set up SoulX

Clone SoulX into the ignored backend directory and create its environment:

```bash
just setup-soulx-backend
```

Download the model weights to the exact path the worker expects:

```bash
just download-soulx-model
```

The resulting local-only layout is:

```text
backends/soulx/
├── .venv/
└── SoulX-Singer/
    ├── example/audio/zh_prompt.*
    ├── soulxsinger/
    └── pretrained_models/SoulX-Singer/model.pt
```

Verify CUDA before a long run:

```bash
uv run --project backends/soulx python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

## 2. Prepare an ACE smoke set

Start with a tiny slice. The production config reads the train directory:

```bash
just prepare-ace-opencpop train 10
```

The equivalent direct command is:

```bash
uv run --extra donors python evals/prepare_ace_opencpop.py \
  --split train \
  --limit 10 \
  --out generated/ace_opencpop_scores \
  --accept-license
```

Use `--clean` when the output directory should contain exactly the newly requested slice. Without
it, existing files are kept and counted, making preparation resumable. Every split also gets
`_source.json` and `_ingest_report.json`; the former stamps ACE attribution and license onto every
generated manifest record.

Prepared scores look like this:

```text
generated/ace_opencpop_scores/train/
├── _source.json
├── _ingest_report.json
└── acesinger_1_2001000004-<digest>.tsv
```

Each TSV row is `onset_seconds<TAB>offset_seconds<TAB>MIDI_pitch`. MIDI 0 rests become gaps. A
contiguous same-pitch `—` slur is merged to avoid a false onset; a pitch-changing slur remains a
distinct transcription note.

## 3. Render with SoulX

Render prepared train scores with a model-resident worker:

```bash
just render-soulx-ace
```

`VODERS_SVS_PERSISTENT=1` matters: one SoulX process loads the model once and serves the full batch.
The recipe passes `--resume`, so rerunning continues an interrupted corpus. To deliberately start a
non-resume run:

```bash
just render-soulx-ace 0
```

SoulX is expressive, so sung boundaries do not exactly follow the rigid source grid. The config
uses `mode: rederive`: after synthesis, pitch tracking aligns ACE's note sequence to rendered f0,
writes timings derived from the actual audio, and validates the pair. A sample that cannot be
verified is rejected rather than entering training with a guessed label.

Outputs are written under:

```text
out/soulx_ace_opencpop/
├── corpus/shard=000/<sample-id>/
│   ├── audio.wav
│   └── score.tsv
├── rejected/
├── manifest.jsonl
└── config.resolved.yaml
```

Only `corpus/` contains accepted training pairs. Rejected samples remain diagnostic artifacts and
are excluded from packing and training.

## 4. Inspect and validate

Run the license audit, quality evaluation, and aggregate statistics:

```bash
uv run voders audit --manifest out/soulx_ace_opencpop/manifest.jsonl
uv run voders eval --manifest out/soulx_ace_opencpop/manifest.jsonl
uv run voders stats --manifest out/soulx_ace_opencpop/manifest.jsonl
```

Before scaling, inspect manifest counts and listen to accepted examples. A low accepted yield is
actionable: do not bypass the validator. Prefer removing pathological phrases or improving
alignment over relaxing tolerances until labels no longer describe the audio.

## 5. Scale up safely

ACE has 100,510 train segments. Increase `LIMIT` in stages to estimate speed, acceptance, and disk
growth:

```bash
just prepare-ace-opencpop train 100
just render-soulx-ace

just prepare-ace-opencpop train 1000
just render-soulx-ace
```

For the entire train split, `LIMIT=0` means unlimited:

```bash
just prepare-ace-opencpop train 0
just render-soulx-ace
```

The importer uses Parquet column projection and HTTP range reads. It still opens many shards for a
full split, so set `HF_TOKEN`, expect network latency, and retain prepared scores between runs.
Rendering is the expensive stage and may take days depending on the GPU.

## 6. Pack and train

Pack accepted pairs into the repository's OGG corpus format:

```bash
just pack-soulx-ace
```

This creates `datasets/soulx_ace_opencpop/`. Keep ACE attribution and its non-commercial license
with any distributed archive. Stage it into the flat directory read by `SyntheticDataset`:

```bash
uv run --extra cpu python evals/corpus_archive.py stage \
  --clean \
  --run-ids soulx_ace_opencpop \
  --out syntheticdataset_soulx
```

The MML26 reference `experiments_sample.sh` trains `Synthetic` and validates on `Klangio`. This repo
keeps that contract in `training/train_soulx.sh`; once staged, run:

```bash
uv sync --extra cpu --extra gpu --extra training
bash training/train_soulx.sh
```

To mix the SoulX corpus with existing sources, stage all desired IDs together:

```bash
uv run --extra cpu python evals/corpus_archive.py stage \
  --clean \
  --run-ids soulx_ace_opencpop donor_pool:40 lyrics_pool_words \
  --out syntheticdataset_soulx
```

## Reproducibility

- Score filenames include a digest of the ACE segment ID and are stable.
- `master_seed: 20260628` makes Mandarin syllable selection deterministic per score and voice.
- `_source.json`, `_ingest_report.json`, `config.resolved.yaml`, and `manifest.jsonl` record source,
  settings, seeds, license, validation verdicts, and paths.
- SoulX GPU inference is not expected to be bit-exact across hardware. Preserve the archive and
  manifest when exact reuse matters.

## Troubleshooting

`no train Parquet shards found`
: Confirm network access, set a current `HF_TOKEN`, and retry. Anonymous range requests are
  rate-limited.

`model.pt` is missing
: Run `just download-soulx-model` and verify the setup tree above.

`persistent backend soulx closed its output`
: Check CUDA memory and the isolated backend environment. Use a one-score smoke run before scaling.

Most samples are rejected
: Inspect reasons and audio. Very short notes, expressive transitions, or f0 failures can make a
  phrase unverifiable. Keep the safety gate; curate inputs or fix alignment before changing limits.

Rendering starts from zero after interruption
: Keep `out/soulx_ace_opencpop/`, use the default `RESUME=1`, and do not rename prepared scores.

Training finds zero samples
: Pack and stage the corpus, then confirm every staged sample has both `audio.wav` and `score.tsv`.
