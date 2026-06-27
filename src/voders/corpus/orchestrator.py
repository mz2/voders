"""Run orchestration: parse → render → validate → write accepted/rejected + manifest.

The orchestrator depends only on the ``RendererLane`` contract (FR-015) and the consent gate
(FR-011). Lanes are injected, so the deterministic baseline (US1) runs without importing any GPU
lane, and the voice-conversion (US2), augmentation (US3), and SVS (US4) lanes plug in unchanged.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

from voders.config.loader import config_hash, write_resolved
from voders.config.models import RunConfig
from voders.corpus.store import CorpusStore
from voders.manifest.io import ManifestWriter
from voders.manifest.models import ProvenanceRecord, ValidationVerdict, VerdictStatus
from voders.render.augmentor import Augmentor
from voders.render.base import RendererLane, RenderRequest, RenderResult
from voders.scoreaug.expand import DroppedVariant, expand_full
from voders.scoreaug.models import ScoreVariant
from voders.scores.analyze import LearnedThreshold, learn_min_note_ms
from voders.scores.parse import ParsedScore, PolyphonyError, parse_tsv, serialize_score
from voders.seeds import derive_seed, sample_seed
from voders.validate.timing import TimingRegistry
from voders.validate.validator import Validator
from voders.voices.models import Voice, VoiceKind
from voders.voices.registry import ConsentRefusedError, VoiceRegistry

LANE_VOICE_KINDS: dict[str, VoiceKind] = {
    "deterministic": VoiceKind.DETERMINISTIC_DONOR,
    "voice_conversion": VoiceKind.VOICE_CONVERSION,
    "svs": VoiceKind.SVS_VOICEBANK,
}


@dataclass
class RunSummary:
    run_id: str
    output_root: str
    attempted: int = 0
    accepted: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    learned_threshold: dict[str, object] = field(default_factory=dict)


class Orchestrator:
    def __init__(
        self,
        config: RunConfig,
        lanes: dict[str, RendererLane],
        *,
        timing: TimingRegistry | None = None,
        augmentor: Augmentor | None = None,
    ) -> None:
        self.config = config
        self.lanes = lanes
        self.store = CorpusStore(config.output_root)
        self.voices = VoiceRegistry(config.voices)
        self.timing = timing or self._default_timing(config, lanes)
        self.augmentor = augmentor
        self.config_hash = config_hash(config)

    @staticmethod
    def _default_timing(config: RunConfig, lanes: dict[str, RendererLane]) -> TimingRegistry:
        reg = TimingRegistry()
        method = config.validator.f0_method
        # Activate the validator's f0 estimator so its frame hop / group delay feed FR-018/FR-019.
        reg.activate("crepe_f0" if method in {"crepe_f0", "auto"} else "pyin_f0")
        return reg

    def _load_scores(self) -> tuple[list[ParsedScore], list[tuple[str, str]]]:
        """Parse all scores; return (parsed, refusals) where refusals are (score_id, reason)."""
        pattern = self.config.scores
        paths = sorted(glob.glob(pattern)) or sorted(glob.glob(str(Path(pattern) / "*.tsv")))
        parsed: list[ParsedScore] = []
        refusals: list[tuple[str, str]] = []
        for path in paths:
            try:
                parsed.append(parse_tsv(path))
            except PolyphonyError as exc:
                refusals.append((Path(path).stem, str(exc)))
        return parsed, refusals

    def _learn_threshold(self, parsed: list[ParsedScore]) -> LearnedThreshold:
        return learn_min_note_ms(
            (p.score for p in parsed),
            percentile=self.config.validator.min_note_percentile,
            method_frame_hops_ms=self.timing.active_frame_hops_ms(),
        )

    def _voices_for_lane(self, lane: str) -> list[Voice]:
        kind = LANE_VOICE_KINDS.get(lane)
        if kind is None:
            return []
        return [v for v in self.voices.all() if v.kind == kind]

    def run(self, *, resume: bool = False) -> RunSummary:
        self.store.ensure_dirs()
        write_resolved(self.config, self.store.config_resolved_path)

        parsed, refusals = self._load_scores()
        # Threshold is learned over the BASE scores only: derived variants must not shift the
        # learned annotation distribution (and a feature-off run is unaffected — SC-001).
        learned = self._learn_threshold(parsed)
        work_items, drops = self._expand_work(parsed)
        if self.config.validator.min_note_ms is not None:
            min_note_ms = self.config.validator.min_note_ms
        else:
            min_note_ms = learned.min_note_ms
        validator = Validator(self.config.validator, self.timing, min_note_ms)

        summary = RunSummary(
            run_id=self.config.run_id,
            output_root=str(self.store.root),
            learned_threshold=learned.to_dict(),
        )
        completed = self.store.completed_ids() if resume else set()
        index = 0

        with ManifestWriter(self.store.manifest_path) as manifest:
            # Record range-guard drops up front so the manifest carries every variant's lineage,
            # including the ones never rendered (FR-005). No audio is written for a drop.
            for drop in drops:
                drop_record = self._record_drop(drop)
                manifest.append(drop_record)
                self._tally(summary, drop_record)

            for lane_name in self.config.enabled_lanes():
                lane = self.lanes.get(lane_name)
                if lane is None or lane_name not in LANE_VOICE_KINDS:
                    continue
                options = self.config.lane_options(lane_name)
                for ps, variant in work_items:
                    for voice in self._voices_for_lane(lane_name):
                        sample_id = f"{ps.score.score_id}_singer_{voice.voice_id}"
                        if lane_name != "deterministic":
                            sample_id = f"{sample_id}_{lane_name}"
                        if sample_id in completed:
                            index += 1
                            continue
                        record, result = self._produce(
                            lane=lane,
                            lane_name=lane_name,
                            ps=ps,
                            voice=voice,
                            validator=validator,
                            options=options,
                            index=index,
                            manifest=manifest,
                            variant=variant,
                        )
                        self._tally(summary, record)
                        self.store.mark_completed(sample_id)
                        index += 1

                        # US3: fan an accepted base render through the augmentation profiles.
                        if (
                            self.augmentor is not None
                            and result is not None
                            and record.verdict.status == VerdictStatus.ACCEPTED
                        ):
                            for prof in self.config.augmentation_profiles:
                                aug_record = self._augment(
                                    base=record,
                                    result=result,
                                    ps=ps,
                                    voice=voice,
                                    profile_id=prof.profile_id,
                                    validator=validator,
                                    index=index,
                                    manifest=manifest,
                                )
                                self._tally(summary, aug_record)
                                self.store.mark_completed(aug_record.sample_id)
                                index += 1
        return summary

    def _expand_work(
        self, parsed: list[ParsedScore]
    ) -> tuple[list[tuple[ParsedScore, ScoreVariant | None]], list[DroppedVariant]]:
        """Build the render work list: each base score plus its per-profile score variants (FR-002).

        With no ``score_augmentation`` the list is exactly ``[(ps, None) for ps in parsed]`` in the
        original order, so a feature-off run is byte-identical to today (SC-001).
        """
        work: list[tuple[ParsedScore, ScoreVariant | None]] = []
        drops: list[DroppedVariant] = []
        for ps in parsed:
            work.append((ps, None))
            for profile in self.config.score_augmentation:
                profile_seed = derive_seed(
                    self.config.master_seed, ps.score.score_id, "score_aug", profile.profile_id
                )
                result = expand_full(ps, profile, profile_seed)
                for variant in result.variants:
                    variant_ps = ParsedScore(
                        score=variant.score,
                        source_path=ps.source_path,
                        raw_bytes=serialize_score(variant.score),
                    )
                    work.append((variant_ps, variant))
                drops.extend(result.drops)
        return work, drops

    @staticmethod
    def _aug_lineage(variant: ScoreVariant | None, *, dynamics_applied: bool = False) -> dict:
        """Score-augmentation provenance fields for a record (empty dict for an original)."""
        if variant is None:
            return {}
        return {
            "base_score_id": variant.base_score_id,
            "score_aug_profile": variant.profile_id,
            "score_aug_axis": variant.axis.value,
            "score_aug_transform": variant.transform,
            "score_aug_seed": variant.seed,
            "dynamics_applied": dynamics_applied,
        }

    def _record_drop(self, drop: DroppedVariant) -> ProvenanceRecord:
        """A rejected, audio-less provenance row for a range-guard drop (FR-005)."""
        return ProvenanceRecord(
            sample_id=f"{drop.base_score_id}__{drop.transform}",
            score_id=f"{drop.base_score_id}__{drop.transform}",
            score_path="",
            audio_path="",
            lane="score_augmentation",
            voice_id="",
            seed=drop.seed,
            config_hash=self.config_hash,
            verdict=ValidationVerdict(status=VerdictStatus.REJECTED, reason=drop.reason),
            base_score_id=drop.base_score_id,
            score_aug_profile=drop.profile_id,
            score_aug_axis=drop.axis.value,
            score_aug_transform=drop.transform,
            score_aug_seed=drop.seed,
        )

    @staticmethod
    def _tally(summary: RunSummary, record: ProvenanceRecord) -> None:
        summary.attempted += 1
        key = record.verdict.status.value
        summary.by_status[key] = summary.by_status.get(key, 0) + 1
        if record.verdict.status == VerdictStatus.ACCEPTED:
            summary.accepted += 1

    def _augment(
        self,
        *,
        base: ProvenanceRecord,
        result: RenderResult,
        ps: ParsedScore,
        voice: Voice,
        profile_id: str,
        validator: Validator,
        index: int,
        manifest: ManifestWriter,
    ) -> ProvenanceRecord:
        """Apply one label-preserving augmentation profile to an accepted base render (FR-005)."""
        assert self.augmentor is not None
        sample_id = f"{base.sample_id}_aug_{profile_id}"
        seed = sample_seed(
            self.config.master_seed, ps.score.score_id, voice.voice_id, f"augment_{profile_id}"
        )
        audio, snr_db = self.augmentor.apply(result.audio, profile_id, seed)
        verdict = validator.validate(audio, result.label_score, snr_db=snr_db)
        # Labels are unchanged by augmentation (FR-005): keep the byte-identical .tsv.
        score_tsv = (
            ps.raw_bytes if result.label_score == ps.score else serialize_score(result.label_score)
        )
        record = ProvenanceRecord(
            sample_id=sample_id,
            score_id=ps.score.score_id,
            score_path="",
            audio_path="",
            lane="augmentation",
            voice_id=voice.voice_id,
            augmentation_profile=profile_id,
            seed=seed,
            voice_license=voice.license,
            consent_verified=voice.consent_verified,
            config_hash=self.config_hash,
            verdict=verdict,
            # Keep an audio-augmented variant in its score-augmentation subtree (FR-016).
            base_score_id=base.base_score_id,
            score_aug_profile=base.score_aug_profile,
            score_aug_axis=base.score_aug_axis,
            score_aug_transform=base.score_aug_transform,
            score_aug_seed=base.score_aug_seed,
            dynamics_applied=base.dynamics_applied,
        )
        record = self.store.write_sample(record, audio, score_tsv, index)
        manifest.append(record)
        return record

    def _produce(
        self,
        *,
        lane: RendererLane,
        lane_name: str,
        ps: ParsedScore,
        voice: Voice,
        validator: Validator,
        options: dict[str, object],
        index: int,
        manifest: ManifestWriter,
        variant: ScoreVariant | None = None,
    ) -> tuple[ProvenanceRecord, RenderResult | None]:
        sample_id = f"{ps.score.score_id}_singer_{voice.voice_id}"
        if lane_name != "deterministic":
            sample_id = f"{sample_id}_{lane_name}"
        seed = sample_seed(self.config.master_seed, ps.score.score_id, voice.voice_id, lane_name)

        # Consent gate (FR-011): refuse before rendering; record license_refused with no audio.
        try:
            self.voices.require_usable(voice.voice_id)
        except ConsentRefusedError as exc:
            record = ProvenanceRecord(
                sample_id=sample_id,
                score_id=ps.score.score_id,
                score_path="",
                audio_path="",
                lane=lane_name,
                voice_id=voice.voice_id,
                seed=seed,
                voice_license=voice.license,
                consent_verified=voice.consent_verified,
                config_hash=self.config_hash,
                verdict=ValidationVerdict(status=VerdictStatus.LICENSE_REFUSED, reason=str(exc)),
                **self._aug_lineage(variant),
            )
            manifest.append(record)
            return record, None

        result = lane.render(RenderRequest(score=ps.score, voice=voice, seed=seed, options=options))
        verdict = validator.validate(result.audio, result.label_score)
        lane_dev = result.notes.get("max_onset_dev_ms", 0.0)
        verdict = verdict.model_copy(
            update={
                "max_onset_dev_ms": max(
                    verdict.max_onset_dev_ms,
                    float(lane_dev) if isinstance(lane_dev, int | float) else 0.0,
                )
            }
        )
        # A lane may force a terminal status (e.g. SVS re-derive onset deviation >50 ms, SC-010).
        forced = result.notes.get("force_status")
        if isinstance(forced, str):
            reason = result.notes.get("reject_reason")
            verdict = verdict.model_copy(
                update={
                    "status": VerdictStatus(forced),
                    "reason": (reason if isinstance(reason, str) else verdict.reason),
                }
            )

        # Byte-identical .tsv for non-rederive lanes; serialized for re-derived labels (FR-002/007).
        if result.label_score is ps.score or result.label_score == ps.score:
            score_tsv = ps.raw_bytes
        else:
            score_tsv = serialize_score(result.label_score)

        dynamics_applied = bool(result.notes.get("dynamics_applied", False))
        record = ProvenanceRecord(
            sample_id=sample_id,
            score_id=ps.score.score_id,
            score_path="",
            audio_path="",
            lane=lane_name,
            voice_id=voice.voice_id,
            seed=seed,
            voice_license=voice.license,
            consent_verified=voice.consent_verified,
            config_hash=self.config_hash,
            verdict=verdict,
            notes=result.notes,
            **self._aug_lineage(variant, dynamics_applied=dynamics_applied),
        )
        record = self.store.write_sample(record, result.audio, score_tsv, index)
        manifest.append(record)
        return record, result
