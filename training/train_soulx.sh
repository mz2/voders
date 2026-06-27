#!/usr/bin/env bash

set -euo pipefail

exec uv run --no-sync python -m training.train \
    --train-dataset Synthetic \
    --train-path out/donor_pool/corpus \
    --val-dataset Klangio \
    --sequence-length 8 \
    --batch-size 32 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --frame-weight 9.0 \
    --onset-weight 18.0 \
    --learning-rate 1e-4 \
    --max-epochs 100 \
    --patience 30 \
    --eval-metric COnPOff_f1 \
    --precision bf16 \
    --torch-compile \
    --torch-compile-mode reduce-overhead \
    --media-log-interval-steps 50 \
    --output-dir BASIC_PITCH_CHALLENGE \
    --experiment-name soulx_baseline \
    --logger wandb \
    --wandb-entity krishnakalyan \
    --wandb-project MMLHackathon \
    --wandb-mode online \
    --wandb-name soulx-30epoch-compiled-bf16 \
    --wandb-tags baseline soulx huggingface bf16 media
