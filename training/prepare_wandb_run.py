"""Create the W&B run before training so CI can publish its URL immediately."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import wandb


def set_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--name", required=True)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--issue", type=int)
    args = parser.parse_args()

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        tags=args.tag,
        config={
            "github_repository": os.environ.get("GITHUB_REPOSITORY"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "github_issue": args.issue,
        },
    )
    if run is None:
        raise RuntimeError("wandb.init() did not create a run")

    print(f"W&B run: {run.url}")
    set_github_output("run_id", run.id)
    set_github_output("run_url", run.url)
    set_github_output("entity", run.entity)
    run.finish()


if __name__ == "__main__":
    main()
