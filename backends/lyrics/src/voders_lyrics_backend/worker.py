"""Lyric generation worker — a stub CLI for the future model-driven lyric source (US4).

When implemented, this worker will be invoked by the 3.14 core across a process boundary (via
``uv run --project backends/lyrics python -m voders_lyrics_backend.worker <request.json>``). It will
load the constrained-decoding / SongComposer melody->lyric model, run it against the requested
melody, and emit a syllables JSON result (one syllable per note, aligned to the score) for the core
to consume.

Protocol (to be finalised when implemented):
  argv[1] = path to a JSON request describing the melody (notes, seed, constraints, ...).
  On success: prints a JSON result with the generated syllables to stdout; exit 0.

This is currently a placeholder following the existing out-of-process backend pattern; it is not
yet wired into the core and raises ``NotImplementedError`` if invoked.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: worker.py <request.json>", file=sys.stderr)
        return 2
    raise NotImplementedError("lyrics generation backend not yet implemented (US4)")


if __name__ == "__main__":
    main()
