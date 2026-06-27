"""Out-of-process lyric generation backend for voders (placeholder for US4).

This package lives in its own uv project (its own ``pyproject.toml`` and ``.python-version``)
because the model-driven "generated" lyric source (US4) will depend on a constrained-decoding /
SongComposer lyric model whose toolkit does not track the 3.14 core's interpreter. The core would
invoke it across a process boundary via ``uv run --project backends/lyrics``, exchanging a small
JSON request for syllables JSON, so the two dependency chains never share an interpreter. It is
intentionally not synced or imported by the core yet.
"""
