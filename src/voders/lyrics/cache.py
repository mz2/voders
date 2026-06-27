"""Cache-as-artifact for generated lyric text (FR-012, SC-007, research Decision L5).

A non-deterministic ``generated`` model is run once per (score, theme, model, seed); its output is
pinned to ``<output_root>/<cache_dir>/<key>.jsonl`` and reused on replay, so a published corpus
reproduces from the pinned text with zero model re-invocations. CPU-only, stdlib-only (FR-005).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_SEP = "\x1f"


def request_key(theme: str, model_ref: str, score_id: str, seed: int, n_notes: int) -> str:
    """Deterministic cache key from the generation inputs, so replay finds the pinned artifact."""
    h = hashlib.sha256()
    h.update(_SEP.join([theme, model_ref, score_id, str(seed), str(n_notes)]).encode("utf-8"))
    return h.hexdigest()


class LyricCache:
    """Read/write pinned generated-lyric artifacts under ``<output_root>/<cache_dir>``."""

    def __init__(self, output_root: str | Path, cache_dir: str = "lyrics") -> None:
        self.dir = Path(output_root) / cache_dir

    def path_for(self, key: str) -> Path:
        return self.dir / f"{key}.jsonl"

    def load(self, key: str) -> list[str | None] | None:
        """Return the pinned syllables for ``key``, or ``None`` if not yet cached."""
        path = self.path_for(key)
        if not path.exists():
            return None
        line = path.read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(line)
        return list(record["syllables"])

    def store(
        self,
        key: str,
        score_id: str,
        syllables: list[str | None],
        *,
        source: str = "generated",
        model_id: str | None = None,
    ) -> Path:
        """Pin ``syllables`` for ``key`` as a one-record JSONL artifact and return its path."""
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {
            "score_id": score_id,
            "syllables": list(syllables),
            "source": source,
            "model": model_id,
        }
        path = self.path_for(key)
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return path
