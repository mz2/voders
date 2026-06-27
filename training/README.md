# Basic Pitch training

This directory vendors the files needed to train the MML26 Basic Pitch model on a
voders corpus. The runtime is based on
Klangio/MML26-singing-synthesis@6d95b7b108f7daabe07e99f12a3ab9d2b9dc2535
and includes the training changes from the local MML26 checkout on 2026-06-27.
Inference-only code and generated checkpoints are intentionally excluded.

The synthetic loader supports both the original challenge layout
(sample/audio.wav plus sample/score.tsv) and the native voders layout
(corpus/shard/sample.wav plus a same-stem TSV). Klangio evaluation data is
checked out at the pinned commit during the Action rather than duplicated here.

## GitHub Action setup

Configure these repository secrets:

- WANDB_API_KEY
- HF_TOKEN

Optionally set repository variables:

- WANDB_PROJECT (defaults to basic_pitch+_MML-hackaton2026)
- WANDB_ENTITY
- HF_MODEL_REPO (defaults to token-owner/voders-basic-pitch)

Adding the train label to an issue runs augmentation and GPU training. The
workflow comments the W&B URL on that issue as soon as the run is created, then
comments the private Hugging Face checkpoint URL after training succeeds.
