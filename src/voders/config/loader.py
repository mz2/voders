"""Run Config loader, resolver, and ``config_hash`` (FR-016)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from voders.config.models import RunConfig


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a Run Config YAML file (FR-016)."""
    raw = Path(path).read_text(encoding="utf-8")
    data: Any = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"run config {path!r} must be a YAML mapping")
    return RunConfig.model_validate(data)


def resolved_dict(config: RunConfig) -> dict[str, Any]:
    """The fully-resolved config as a plain dict (defaults filled in), for embedding/hashing."""
    return config.model_dump(mode="json")


def config_hash(config: RunConfig) -> str:
    """Stable SHA-256 of the resolved config (FR-016).

    Keys are sorted so the hash depends only on content, not on field order.
    """
    canonical = json.dumps(resolved_dict(config), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_resolved(config: RunConfig, path: str | Path) -> None:
    """Write ``config.resolved.yaml`` — the committable end-result record (FR-017)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(resolved_dict(config), sort_keys=True), encoding="utf-8")
