"""
Training script for Basic Pitch.

Default layout: train on Synthetic, validate and test on Klangio.

Usage:
    python -m training.train --batch-size 8 --max-epochs 50

Run `python -m training.train --help` for all options.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional
from socket import gethostname

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

from .transcription_utils.evaluation import METRICS_REGISTRY
from .lightning_module import LightningModuleSingingVoice
from .constants import SAMPLE_RATE, WANDB_PROJECT
from .dataloading import get_dataset_registry

TRAIN_DATASET_NAME = "Synthetic"
VAL_DATASET_NAME = "Klangio"
TEST_DATASET_NAME = "Klangio"


class NsysCaptureCallback(pl.Callback):
    """Capture a bounded set of training batches via the CUDA Profiler API."""

    def __init__(
        self,
        warmup_steps: int,
        capture_steps: int,
        stop_after_capture: bool = False,
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.capture_steps = capture_steps
        self.stop_after_capture = stop_after_capture
        self.batches_seen = 0
        self.active = False
        self.finished = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.finished:
            return
        if self.batches_seen == self.warmup_steps:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            self.active = True
            print(
                f"Nsight Systems capture started at global step {trainer.global_step} "
                f"for {self.capture_steps} batches"
            )
        if self.active:
            torch.cuda.nvtx.range_push(f"train_step_{trainer.global_step}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.finished:
            return
        if self.active:
            torch.cuda.nvtx.range_pop()
        self.batches_seen += 1
        if self.active and self.batches_seen >= self.warmup_steps + self.capture_steps:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            self.active = False
            self.finished = True
            print(f"Nsight Systems capture stopped at global step {trainer.global_step}")
            if self.stop_after_capture:
                trainer.should_stop = True
                print("Stopping training after the requested Nsight capture")

    def on_exception(self, trainer, pl_module, exception):
        if self.active:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            self.active = False


class TorchCompileCallback(pl.Callback):
    """Compile the transcriber after checkpoint state has been restored."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.compiled = False

    def on_train_start(self, trainer, pl_module):
        if self.compiled:
            return
        print(f"Compiling BasicPitchTranscriber with torch.compile(mode={self.mode!r})")
        # Bypass nn.Module registration for the optimized wrapper. The original
        # eager module remains the source of checkpoint state and is used for
        # variable-length validation/inference; only the fixed-shape training
        # path dispatches through this compiled callable.
        object.__setattr__(
            pl_module,
            "_compiled_training_model",
            torch.compile(pl_module.model, mode=self.mode),
        )
        self.compiled = True


def create_datasets(
    groups: List[str],
    sequence_length: Optional[int],
    seed: int,
    device: str,
    dataset_name: str = TRAIN_DATASET_NAME,
    data_path: Optional[str] = None,
):
    registry = get_dataset_registry()
    dataset_meta = registry[dataset_name]
    print(f"\nLoading {dataset_name} (default path)")
    print(f"  Groups: {groups}")
    dataset_kwargs = {}
    if data_path is not None:
        dataset_kwargs["path"] = data_path
    return dataset_meta["class"](
        groups=groups,
        sequence_length=sequence_length,
        seed=seed,
        device=device,
        **dataset_kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Basic Pitch (Synthetic train, Klangio val/test by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--train-dataset",
        type=str,
        default=TRAIN_DATASET_NAME,
        choices=list(get_dataset_registry().keys()),
        help="Dataset to use for training",
    )
    data_group.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Override the selected training dataset's default path",
    )
    data_group.add_argument(
        "--val-dataset",
        type=str,
        default=VAL_DATASET_NAME,
        choices=list(get_dataset_registry().keys()),
        help="Dataset to use for validation",
    )
    data_group.add_argument(
        "--train-groups",
        type=str,
        nargs="+",
        default=["train"],
        help="Dataset splits to use for training",
    )
    data_group.add_argument(
        "--val-groups",
        type=str,
        nargs="+",
        default=["test"],
        help="Dataset splits to use for validation",
    )
    data_group.add_argument(
        "--test-dataset",
        type=str,
        default=TEST_DATASET_NAME,
        choices=list(get_dataset_registry().keys()),
        help="Dataset to use for post-training testing",
    )
    data_group.add_argument(
        "--test-groups",
        type=str,
        nargs="+",
        default=["test"],
        help="Dataset splits to use for post-training testing",
    )
    data_group.add_argument(
        "--eval-metric",
        type=str,
        default="COnPOff_f1",
        choices=list(METRICS_REGISTRY.keys()),
        help="Metric to track to select the best checkpoint. Defaults to the offset-aware "
        "COnPOff_f1; COnP_f1 ignores offsets (offset_ratio=None), so checkpoint and "
        "early-stopping selection would be blind to note end-times.",
    )

    audio_group = parser.add_argument_group("Audio Processing")
    audio_group.add_argument(
        "--sequence-length",
        type=float,
        default=8.0,
        help="Length of audio chunks in seconds (None for full audio)",
    )
    audio_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    audio_group.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() // 3,
        help="Number of data loading workers",
    )
    audio_group.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches prefetched by each data loading worker",
    )

    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    train_group.add_argument(
        "--accumulate-gradients",
        type=int,
        default=1,
        help="Number of gradients to accumulate",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use",
    )
    train_group.add_argument(
        "--onset-weight",
        type=float,
        default=1.0,
        help="Weight for onset loss (positive class weight)",
    )
    train_group.add_argument(
        "--frame-weight",
        type=float,
        default=1.0,
        help="Weight for frame loss (positive class weight)",
    )
    train_group.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )
    train_group.add_argument(
        "--val-check-interval-hours",
        type=float,
        default=None,
        help="Validate every X hours of audio. If None, validates every epoch.",
    )
    train_group.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Limit training batches per epoch",
    )
    train_group.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Limit validation batches per epoch",
    )
    train_group.add_argument(
        "--media-log-interval-steps",
        type=int,
        default=0,
        help="Log input audio and decoded predictions to WandB every N optimizer steps (0 disables)",
    )
    train_group.add_argument(
        "--num-sanity-val-steps",
        type=int,
        default=2,
        help="Number of sanity check validation steps (0 to disable)",
    )
    train_group.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the transcriber with torch.compile before training",
    )
    train_group.add_argument(
        "--torch-compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Optimization mode passed to torch.compile",
    )

    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "mps"],
        help="Hardware accelerator to use",
    )
    system_group.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16", "bf16", "32"],
        help="Training precision",
    )
    system_group.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    system_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    system_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save logs",
    )
    system_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (defaults to output-dir)",
    )
    system_group.add_argument(
        "--experiment-name",
        type=str,
        default="basic_pitch_synthetic_data",
        help="Name for this experiment",
    )
    system_group.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="Logger backend to use",
    )
    system_group.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    system_group.add_argument(
        "--wandb-project",
        type=str,
        default=WANDB_PROJECT,
        help="Weights & Biases project name",
    )
    system_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases user or team entity",
    )
    system_group.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        help="Weights & Biases tags",
    )
    system_group.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )
    system_group.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    system_group.add_argument(
        "--nsys-warmup-steps",
        type=int,
        default=0,
        help="Training batches to warm up before a cudaProfilerApi capture",
    )
    system_group.add_argument(
        "--nsys-capture-steps",
        type=int,
        default=0,
        help="Training batches to include in a cudaProfilerApi capture (0 disables)",
    )
    system_group.add_argument(
        "--nsys-exit-after-capture",
        action="store_true",
        help="Stop training and skip final evaluation once the Nsight capture finishes",
    )

    args = parser.parse_args()

    if args.sequence_length is None:
        sequence_length_samples = None
    else:
        sequence_length_samples = int(args.sequence_length * SAMPLE_RATE)

    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: BasicPitchTranscriber")
    print(f"Train dataset: {args.train_dataset}")
    print(f"Val dataset: {args.val_dataset}")
    print(f"Test dataset: {args.test_dataset}")
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
    print(f"Train groups: {args.train_groups}")
    print(f"Val groups: {args.val_groups}")
    print(f"Test groups: {args.test_groups}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length (s): {args.sequence_length}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Epochs: {args.max_epochs}")
    print("=" * 70)
    print()

    print("Loading training dataset...")
    train_dataset = create_datasets(
        groups=args.train_groups,
        sequence_length=sequence_length_samples,
        seed=args.seed,
        device="cpu",
        dataset_name=args.train_dataset,
        data_path=args.train_path,
    )
    print(f"Training dataset: {len(train_dataset)} samples")

    print("\nLoading validation dataset...")
    val_dataset = create_datasets(
        groups=args.val_groups,
        sequence_length=None,
        seed=args.seed,
        device="cpu",
        dataset_name=args.val_dataset,
    )
    print(f"Validation dataset: {len(val_dataset)} samples")

    loader_worker_kwargs = {}
    if args.num_workers > 0:
        loader_worker_kwargs = {
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": True,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.accelerator == "gpu",
        **loader_worker_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.accelerator == "gpu",
        **loader_worker_kwargs,
    )
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print(f"\nLoading test dataset ({args.test_dataset})...")
    test_dataset = create_datasets(
        groups=args.test_groups,
        sequence_length=None,
        seed=args.seed,
        device="cpu",
        dataset_name=args.test_dataset,
    )
    print(f"Test dataset: {len(test_dataset)} samples")

    # num_workers=0: the test loader is iterated by trainer.test() only after fit()
    # has fully initialized CUDA. Forking worker processes at that point (the old
    # multiprocessing_context='fork') inherits a broken CUDA context and deadlocks
    # at "STARTING TESTING". The test set is tiny (batch_size=1), so in-process
    # loading costs nothing and sidesteps the fork-after-CUDA hazard entirely.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print(f"Test batches: {len(test_loader)}")

    print("\nInitializing Basic Pitch model...")
    model = LightningModuleSingingVoice(
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        onset_weight=args.onset_weight,
        frame_weight=args.frame_weight,
        media_log_interval_steps=args.media_log_interval_steps,
    )

    base_dir = args.checkpoint_dir if args.checkpoint_dir else args.output_dir

    # W&B is optional and gated purely on WANDB_API_KEY: with a key present we log
    # online to Weights & Biases; without one we fall back to local TensorBoard
    # logging so the run never blocks on an interactive login prompt.
    logger = None
    using_wandb = False
    if args.logger == "wandb":
        if not os.environ.get("WANDB_API_KEY"):
            print(
                "W&B: WANDB_API_KEY not set — using TensorBoard logging instead "
                "(set WANDB_API_KEY to log to Weights & Biases)."
            )
        else:
            try:
                from pytorch_lightning.loggers import WandbLogger
            except ImportError as exc:
                raise ImportError(
                    "WandB logger requested but not available. Install wandb with: pip install wandb"
                ) from exc
            try:
                wandb_tags = [] if args.wandb_tags is None else list(args.wandb_tags)
                logger = WandbLogger(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_name or args.experiment_name,
                    tags=wandb_tags + [gethostname()],
                    save_dir=base_dir,
                    mode=args.wandb_mode,
                    log_model=False,
                )
                config_dict = {
                    k: v for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                }
                logger.experiment.config.update(config_dict, allow_val_change=True)
                using_wandb = True
            except Exception as e:
                print(f"Warning: W&B logging unavailable ({e}); falling back to TensorBoard.")
                logger = None

    if logger is None:
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
        except ImportError as exc:
            raise ImportError(
                "TensorBoard logger requested but not available. Install tensorboard."
            ) from exc
        logger = TensorBoardLogger(
            save_dir=base_dir,
            name=args.experiment_name,
            default_hp_metric=False,
        )

    output_dir = Path(base_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if using_wandb and logger.version is not None:
        checkpoint_dir = output_dir / str(logger.version) / "checkpoints"
    elif hasattr(logger, "log_dir") and logger.log_dir is not None:
        checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    else:
        checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = f"eval/{args.eval_metric}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        # Built-in {epoch}/{step} tokens only: the monitored metric name contains a
        # "/" which would leak into the path as a subdirectory, and the previous
        # template referenced non-existent keys (global_step / eval_COnP_f1) that
        # always rendered as 0. The best score is preserved in the checkpoint's
        # best_model_score and in the logger.
        filename="best-epoch{epoch:02d}-step{step:06d}",
        auto_insert_metric_name=False,
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode="max",
        patience=args.patience,
        verbose=True,
    )

    precision_map = {"16": "16-mixed", "bf16": "bf16-mixed", "32": "32-true"}
    callbacks = [checkpoint_callback, lr_monitor, early_stopping]
    if args.torch_compile:
        callbacks.append(TorchCompileCallback(args.torch_compile_mode))
    if args.nsys_capture_steps > 0:
        if args.accelerator != "gpu":
            raise ValueError("Nsight Systems cudaProfilerApi capture requires --accelerator gpu")
        callbacks.append(
            NsysCaptureCallback(
                args.nsys_warmup_steps,
                args.nsys_capture_steps,
                stop_after_capture=args.nsys_exit_after_capture,
            )
        )

    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": precision_map[args.precision],
        "logger": logger,
        "callbacks": callbacks,
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 10,
        "deterministic": False,
        "num_sanity_val_steps": args.num_sanity_val_steps,
        "accumulate_grad_batches": args.accumulate_gradients,
        "enable_model_summary": True,
    }

    if args.val_check_interval_hours is not None:
        samples_per_hour = 3600 * SAMPLE_RATE
        samples_per_batch = args.batch_size * sequence_length_samples
        batches_per_hour = samples_per_hour / samples_per_batch
        val_check_batches = int(batches_per_hour * args.val_check_interval_hours)
        trainer_kwargs["val_check_interval"] = max(1, val_check_batches)
    else:
        trainer_kwargs["check_val_every_n_epoch"] = 1

    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Checkpoints will be saved to: {checkpoint_callback.dirpath}")
    print("=" * 70)
    print()

    # Locally produced checkpoints include NumPy scalar metadata from threshold
    # optimization, which PyTorch's restricted weights-only loader rejects.
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=args.resume_from,
        weights_only=False,
    )

    if args.nsys_capture_steps > 0 and args.nsys_exit_after_capture:
        print("Nsight profiling run complete; skipping checkpoint reload and testing")
        return

    print("\n" + "=" * 70)
    print("LOADING BEST CHECKPOINT FOR TESTING")
    print("=" * 70)
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading checkpoint: {best_model_path}")
    # This checkpoint is produced locally by the training run and includes
    # NumPy threshold metadata, which PyTorch's restricted weights-only loader
    # cannot deserialize as of PyTorch 2.6.
    model = LightningModuleSingingVoice.load_from_checkpoint(
        best_model_path, weights_only=False
    )

    print("\n" + "=" * 70)
    print("STARTING TESTING")
    print("=" * 70)
    trainer.test(model, test_loader)

    if logger is not None:
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            metrics[key] = value.item() if hasattr(value, "item") else value
        if using_wandb:
            logger.log_hyperparams(vars(args))
        else:
            logger.log_hyperparams(vars(args), metrics)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best {args.eval_metric}: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
