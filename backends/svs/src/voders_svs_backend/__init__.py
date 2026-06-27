"""Out-of-process SVS render backend for voders.

This package lives in its own uv project (its own ``pyproject.toml``, ``.python-version``, and
``uv.lock``) because its toolkit (NNSVS) pins dependencies that have no Python 3.14 wheels. The
3.14 core invokes it across a process boundary via ``uv run --project backends/svs`` and exchanges
a small JSON request plus a WAV file, so the two incompatible dependency chains never share an
interpreter.
"""
