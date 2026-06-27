"""Lyric generation worker — out-of-process backend for the ``generated`` source (US4, FR-011).

Invoked by the 3.14 core across a process boundary (``uv run --project backends/lyrics python -m
voders_lyrics_backend.worker <request.json>``) so a model toolkit pinning older Python / GPU deps
never constrains the CPU core.

Protocol:
  argv[1] = path to a JSON request: ``{"theme", "score_id", "notes": [[on,off,pitch],...], "seed",
            "model_ref"}``.
  On success: prints a JSON result ``{"score_id", "text"}`` (lyric text; the core segments it into
            one syllable per note, FR-019) to stdout; exit 0.

This default implementation is a **deterministic placeholder**, not a real melody->lyric model: it
cycles the theme's words to cover the note count so the out-of-process contract is exercised and
runs reproducibly. Replace ``_generate`` with the constrained-decoding / SongComposer model
(research Decision L1); the core caches whatever text it returns (FR-012), so determinism of the
real model is handled by the cache, not here.
"""

from __future__ import annotations

import json
import sys


def _generate(theme: str, n_notes: int, seed: int) -> str:
    words = [w for w in theme.replace(",", " ").split() if w] or ["la"]
    count = max(n_notes, len(words))
    return " ".join(words[i % len(words)] for i in range(count))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: worker.py <request.json>", file=sys.stderr)
        return 2
    with open(args[0], encoding="utf-8") as fh:
        request = json.load(fh)
    theme = str(request.get("theme", "")).strip()
    n_notes = len(request.get("notes", []))
    seed = int(request.get("seed", 0))
    text = _generate(theme, n_notes, seed)
    print(json.dumps({"score_id": request.get("score_id", ""), "text": text}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
