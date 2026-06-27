"""Upload a run's best checkpoint to a private Hugging Face model repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def set_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--repo-id", default=os.environ.get("HF_MODEL_REPO"))
    parser.add_argument("--run-id", default=os.environ.get("WANDB_RUN_ID", "unknown-run"))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    api = HfApi(token=token)
    repo_id = args.repo_id
    if not repo_id:
        username = api.whoami()["name"]
        repo_id = f"{username}/voders-basic-pitch"

    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    run_path = f"runs/{args.run_id}/{args.checkpoint.name}"
    api.upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo=run_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload best checkpoint for W&B run {args.run_id}",
    )
    api.upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo="best.ckpt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update latest best checkpoint from {args.run_id}",
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    checkpoint_url = f"{repo_url}/blob/main/{run_path}"
    print(f"Hugging Face checkpoint: {checkpoint_url}")
    set_github_output("repo_id", repo_id)
    set_github_output("repo_url", repo_url)
    set_github_output("checkpoint_url", checkpoint_url)


if __name__ == "__main__":
    main()
