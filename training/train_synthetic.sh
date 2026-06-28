#!/usr/bin/env bash
#
# Synthetic-only training: train on the hand-made synthetic short sequences (donor_pool + its
# augmentations, and the other authored-score pools) and IGNORE the Klangio-transcribed training
# renders entirely. Validation still runs on the held-out Klangio set under
# external/MML26-singing-synthesis/klangiodataset, so this is an apples-to-apples A/B against
# train_soulx.sh: the ONLY difference is the training data (--train-path), letting us test whether
# the Klangio-transcription annotations in the training mix were holding the scores down.
#
# Stage the data first with:  just train-synthetic   (or stage syntheticdataset_synth yourself)

set -euo pipefail

exec uv run --no-sync python -m training.train \
    --train-dataset Synthetic \
    --train-path syntheticdataset_synth \
    --val-dataset Klangio \
    --sequence-length 8 \
    --batch-size 32 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --frame-weight "${FRAME_WEIGHT:-16}" \
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
    --experiment-name synthetic_only \
    --logger wandb \
    --wandb-entity krishnakalyan \
    --wandb-project MMLHackathon \
    --wandb-mode online \
    --wandb-name synthetic-only-klangio-val \
    --wandb-tags synthetic-only no-klangio-train klangio-val bf16
