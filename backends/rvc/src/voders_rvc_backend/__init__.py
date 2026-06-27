"""Out-of-process RVC voice-conversion backend for voders.

Runs the real RVC (Retrieval-based Voice Conversion) toolkit in its own uv project (Python 3.11)
because its dependency chain has no Python 3.14 wheels. The 3.14 core sends already-rendered,
score-aligned audio plus a *consented* target voice model and gets back timbre-converted audio;
RVC takes the pitch (f0) from the input audio, so the score-aligned f0 — and thus the labels — are
preserved (FR-004).
"""
