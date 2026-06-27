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
from voders.lyrics.models import LyricPlan, LyricSource
from voders.lyrics.sources import LyricLicenseRefused, LyricSourceProtocol, build_source
from voders.manifest.io import ManifestWriter
from voders.manifest.models import ProvenanceRecord, ValidationVerdict, VerdictStatus
from voders.render.augmentor import Augmentor
from voders.render.base import RendererLane, RenderRequest, RenderResult
from voders.scores.analyze import LearnedThreshold, learn_min_note_ms
from voders.scores.parse import ParsedScore, PolyphonyError, parse_tsv, serialize_score
from voders.seeds import sample_seed
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
        accompanist: object | None = None,
    ) -> None:
        self.config = config
        self.lanes = lanes
        self.store = CorpusStore(config.output_root)
        self.voices = VoiceRegistry(config.voices)
        self.timing = timing or self._default_timing(config, lanes)
        self.augmentor = augmentor
        self.accompanist = accompanist
        self.config_hash = config_hash(config)
        self._lyric_source = self._build_lyric_source(config)

    @staticmethod
    def _build_lyric_source(config: RunConfig) -> LyricSourceProtocol | None:
        """Build the lyric source once, or ``None`` for the default ``vowel`` (lyric-free) run.

        A ``vowel`` run never resolves lyrics, so output stays byte-identical to pre-feature
        (SC-001) and the manifest's lyric axis keeps its inert defaults.
        """
        if config.lyrics.source == LyricSource.VOWEL:
            return None
        return build_source(
            config.lyrics.source,
            inventory=config.lyrics.inventory,
            syllabifier=config.lyrics.syllabifier,
            theme=config.lyrics.theme,
            model=config.lyrics.model,
            cache_dir=config.lyrics.cache_dir,
            output_root=config.output_root,
        )

    def _lyric_plan(self, score: ParsedScore | None, voice: Voice) -> LyricPlan | None:
        """Resolve the per-(score, voice) lyric plan, or ``None`` for a ``vowel`` run."""
        if self._lyric_source is None or score is None:
            return None
        return self._lyric_source.resolve(
            score.score, master_seed=self.config.master_seed, voice_id=voice.voice_id
        )

    @staticmethod
    def _lyric_fields(plan: LyricPlan | None, result: RenderResult | None) -> dict[str, object]:
        """Provenance lyric axis for a plan (empty => the record keeps vowel-default fields)."""
        if plan is None:
            return {}
        articulated = bool(result.notes.get("lyric_articulated", False)) if result else False
        fields: dict[str, object] = {
            "lyric_source": plan.source.value,
            "lyric_hash": plan.text_hash,
            "lyric_articulated": articulated,
            "lyric_multisyllable_supplied": len(plan.multisyllable_notes),
        }
        if plan.model is not None:
            fields["lyric_model"] = plan.model.model_id
            fields["lyric_model_license"] = plan.model.license
        return fields

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
        learned = self._learn_threshold(parsed)
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
            for lane_name in self.config.enabled_lanes():
                lane = self.lanes.get(lane_name)
                if lane is None or lane_name not in LANE_VOICE_KINDS:
                    continue
                options = self.config.lane_options(lane_name)
                for ps in parsed:
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

                        # spec 002: lay accompaniment under an accepted vocal (corpus-internal).
                        if (
                            self.accompanist is not None
                            and result is not None
                            and record.verdict.status == VerdictStatus.ACCEPTED
                        ):
                            acc_record = self._accompany(
                                base=record,
                                result=result,
                                ps=ps,
                                voice=voice,
                                index=index,
                                manifest=manifest,
                            )
                            self._tally(summary, acc_record)
                            self.store.mark_completed(acc_record.sample_id)
                            index += 1
        return summary

    def _accompany(
        self,
        *,
        base: ProvenanceRecord,
        result: RenderResult,
        ps: ParsedScore,
        voice: Voice,
        index: int,
        manifest: ManifestWriter,
    ) -> ProvenanceRecord:
        """Lay accompaniment under an accepted base render and write the new sample (spec 002)."""
        acc = self.accompanist
        assert acc is not None
        mode = acc.options.mode
        sample_id = f"{base.sample_id}_accomp_{mode}"
        base_seed = sample_seed(
            self.config.master_seed, ps.score.score_id, voice.voice_id, f"accompaniment_{mode}"
        )
        out = acc.generate(
            result.audio,
            result.label_score,
            source_vocal_sample_id=base.sample_id,
            base_seed=base_seed,
        )
        # Labels are unchanged (the score still describes the mix): keep the byte-identical .tsv.
        score_tsv = (
            ps.raw_bytes if result.label_score == ps.score else serialize_score(result.label_score)
        )
        record = ProvenanceRecord(
            sample_id=sample_id,
            score_id=ps.score.score_id,
            score_path="",
            audio_path="",
            lane="accompaniment",
            voice_id=voice.voice_id,
            seed=base_seed,
            voice_license=voice.license,
            consent_verified=voice.consent_verified,
            config_hash=self.config_hash,
            verdict=out.verdict,
            accompaniment=out.provenance,
        )
        record = self.store.write_accompaniment_sample(
            record, out.mix, score_tsv, index, stem=out.stem
        )
        manifest.append(record)
        return record

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
            lyric_source=base.lyric_source,
            lyric_hash=base.lyric_hash,
            lyric_model=base.lyric_model,
            lyric_model_license=base.lyric_model_license,
            lyric_articulated=base.lyric_articulated,
            lyric_multisyllable_supplied=base.lyric_multisyllable_supplied,
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
            )
            manifest.append(record)
            return record, None

        # Lyric model license gate (FR-010), mirroring the donor-voice consent gate above: a refused
        # generated model produces a license_refused record and no audio, surfaced in the manifest.
        try:
            plan = self._lyric_plan(ps, voice)
        except LyricLicenseRefused as exc:
            model = self.config.lyrics.model
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
                lyric_source=LyricSource.GENERATED.value,
                lyric_model=model.model_id if model else None,
                lyric_model_license=model.license if model else None,
            )
            manifest.append(record)
            return record, None
        lyrics = plan.syllables if plan is not None else None
        result = lane.render(
            RenderRequest(
                score=ps.score,
                voice=voice,
                seed=seed,
                options=options,
                lyrics=lyrics,
                g2p_backend=self.config.lyrics.g2p_backend,
            )
        )
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
            **self._lyric_fields(plan, result),
        )
        record = self.store.write_sample(record, result.audio, score_tsv, index)
        manifest.append(record)
        return record, result
