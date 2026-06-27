"""``expand(base, profile, seed)`` — the pure score-to-scores multiplier (FR-002/010/015).

Fans a base score out into per-axis variant scores. Each variant is a valid input score (monophonic,
ordered, positive durations, pitches in ``0..127``, note count + per-note ``lyric`` preserved), so
it flows through the pipeline unchanged and its labels are correct by construction (FR-003).

Determinism (G1/SC-005): every axis/draw derives its own sub-seed from the profile seed via
``derive_seed`` so axes and draws are independent and reproducible regardless of worker count or
order. The axis order is fixed (transpose → humanize → volume), so the output list is deterministic.

Composition in v1 is **per-axis**: each axis emits its own variants from the base. Cross-axis
products (e.g. transpose×humanize on one variant) are out of scope (a documented future extension).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from voders.config.models import ScoreAugmentationProfile
from voders.scoreaug.humanize import _humanize
from voders.scoreaug.models import ScoreAxis, ScoreVariant
from voders.scoreaug.transpose import _transpose
from voders.scoreaug.volume import volume
from voders.scores.parse import ParsedScore
from voders.seeds import derive_seed


@dataclass(frozen=True)
class DroppedVariant:
    """A variant the range guard discarded before rendering (FR-005); recorded in provenance."""

    base_score_id: str
    profile_id: str
    axis: ScoreAxis
    transform: str
    seed: int
    reason: str


@dataclass(frozen=True)
class ExpansionResult:
    """The full outcome of expanding one base score under one profile."""

    variants: list[ScoreVariant] = field(default_factory=list)
    drops: list[DroppedVariant] = field(default_factory=list)


def expand(base: ParsedScore, profile: ScoreAugmentationProfile, seed: int) -> list[ScoreVariant]:
    """Return only the valid variants the profile's enabled axes produce (contract surface)."""
    return expand_full(base, profile, seed).variants


def expand_full(base: ParsedScore, profile: ScoreAugmentationProfile, seed: int) -> ExpansionResult:
    """Return both the emitted variants and any range-guard drops (orchestrator surface)."""
    base_id = base.score.score_id
    variants: list[ScoreVariant] = []
    drops: list[DroppedVariant] = []

    # Axis order is fixed for determinism: transpose → humanize → volume.
    if profile.transpose is not None:
        tknob = profile.transpose
        axis_seed = derive_seed(seed, ScoreAxis.TRANSPOSE.value)
        for offset in tknob.offsets:
            transform = f"t{offset:+d}"
            shifted, clamped = _transpose(
                base.score, offset, policy=tknob.policy, window=tknob.window
            )
            if shifted is None:
                drops.append(
                    DroppedVariant(
                        base_score_id=base_id,
                        profile_id=profile.profile_id,
                        axis=ScoreAxis.TRANSPOSE,
                        transform=transform,
                        seed=axis_seed,
                        reason=(
                            f"offset {offset:+d} pushes a note outside the "
                            f"window {tknob.window} (policy=drop)"
                        ),
                    )
                )
                continue
            meta: dict[str, object] = {}
            if clamped:
                meta["clamped"] = clamped
            variants.append(
                _variant(
                    base_id,
                    profile.profile_id,
                    ScoreAxis.TRANSPOSE,
                    transform,
                    axis_seed,
                    shifted,
                    meta,
                )
            )

    if profile.humanize_time is not None:
        hknob = profile.humanize_time
        axis_seed = derive_seed(seed, ScoreAxis.HUMANIZE.value)
        for draw in range(hknob.draws):
            transform = f"hum{draw}"
            draw_seed = derive_seed(axis_seed, draw)
            humanised, hmeta = _humanize(base.score, hknob, draw_seed)
            variants.append(
                _variant(
                    base_id,
                    profile.profile_id,
                    ScoreAxis.HUMANIZE,
                    transform,
                    draw_seed,
                    humanised,
                    dict(hmeta),
                )
            )

    if profile.volume is not None:
        axis_seed = derive_seed(seed, ScoreAxis.VOLUME.value)
        varied = volume(base.score, profile.volume, axis_seed)
        variants.append(
            _variant(base_id, profile.profile_id, ScoreAxis.VOLUME, "vol", axis_seed, varied, {})
        )

    return ExpansionResult(variants=variants, drops=drops)


def _variant(
    base_id: str,
    profile_id: str,
    axis: ScoreAxis,
    transform: str,
    seed: int,
    score: object,
    meta: dict[str, object],
) -> ScoreVariant:
    from voders.scores.models import Score

    assert isinstance(score, Score)
    variant_score = score.model_copy(update={"score_id": f"{base_id}__{transform}"})
    return ScoreVariant(
        score=variant_score,
        base_score_id=base_id,
        profile_id=profile_id,
        axis=axis,
        transform=transform,
        seed=seed,
        dynamics_applied=False,
        notes_meta=meta,
    )
