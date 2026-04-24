"""Creative draft planning built on top of matched source segments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models.project import Project
from models.segment import Segment, SegmentType, compute_segment_duration


TIME_SCALE = 1_000_000


@dataclass(slots=True)
class CreativeUnit:
    """One visual/text atom inside a recomposed segment."""

    unit_type: str
    timeline_start_us: int
    duration_us: int
    source_start: Optional[float] = None
    source_end: Optional[float] = None
    text: str = ""
    label: str = ""


@dataclass(slots=True)
class CreativeSegmentPlan:
    """Recomposed plan for one narration segment."""

    segment_id: str
    timeline_start_us: int
    duration_us: int
    match_type: str
    summary_text: str = ""
    units: list[CreativeUnit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CreativePlan:
    """Full creative timeline."""

    segments: list[CreativeSegmentPlan]
    total_duration_us: int


class CreativePlanner:
    """Turn aligned source matches into a more editorial timeline."""

    _CARD_MIN_S = 0.55
    _CARD_MAX_S = 1.10

    def __init__(self, audio_source: str = "original", template: str = "story_mix"):
        self.audio_source = audio_source
        self.template = template

    def build(self, project: Project) -> CreativePlan:
        timeline_cursor_us = 0
        plans: list[CreativeSegmentPlan] = []

        for segment in project.segments:
            if not segment.use_segment:
                continue

            duration_s = self._playback_duration(segment)
            if duration_s <= 0.05:
                continue

            plan = self._plan_segment(segment, timeline_cursor_us, duration_s)
            if not plan:
                continue

            plans.append(plan)
            timeline_cursor_us += plan.duration_us

        return CreativePlan(segments=plans, total_duration_us=timeline_cursor_us)

    def _plan_segment(
        self,
        segment: Segment,
        timeline_start_us: int,
        duration_s: float,
    ) -> Optional[CreativeSegmentPlan]:
        duration_us = max(int(round(duration_s * TIME_SCALE)), 1)
        summary_text = self._summary_text(segment)
        match_type = self._match_type(segment)

        plan = CreativeSegmentPlan(
            segment_id=segment.id,
            timeline_start_us=timeline_start_us,
            duration_us=duration_us,
            match_type=match_type,
            summary_text=summary_text,
        )

        if match_type == "fallback":
            plan.units.append(
                CreativeUnit(
                    unit_type="narration_clip",
                    timeline_start_us=timeline_start_us,
                    duration_us=duration_us,
                    source_start=segment.narration_start,
                    source_end=segment.narration_end,
                    label="fallback_narration",
                )
            )
            if summary_text:
                plan.units.append(
                    CreativeUnit(
                        unit_type="text_card",
                        timeline_start_us=timeline_start_us,
                        duration_us=duration_us,
                        text=f"待替换画面\n{summary_text}",
                        label="fallback_text",
                    )
                )
            plan.notes.extend(["fallback", "narration_reference"])
            return plan

        source_start = float(segment.movie_start or 0.0)
        source_end = float(segment.movie_end or source_start)
        for clip in self._movie_clips(source_start, source_end, timeline_start_us, duration_us):
            plan.units.append(clip)

        card_duration_us = self._card_duration_us(duration_s, match_type, bool(summary_text))
        if summary_text and card_duration_us > 0:
            prefix = "剧情重点" if match_type == "exact" else "可改写画面"
            plan.units.append(
                CreativeUnit(
                    unit_type="text_card",
                    timeline_start_us=timeline_start_us,
                    duration_us=card_duration_us,
                    text=f"{prefix}\n{summary_text}",
                    label="story_card",
                )
            )
            plan.notes.append("story_card")

        if len([unit for unit in plan.units if unit.unit_type == "movie_clip"]) > 1:
            plan.notes.append("multi_shot_recut")
        plan.notes.append(match_type)
        return plan

    def _movie_clips(
        self,
        movie_start: float,
        movie_end: float,
        timeline_start_us: int,
        duration_us: int,
    ) -> list[CreativeUnit]:
        source_duration = max(movie_end - movie_start, 0.1)
        playback_s = duration_us / TIME_SCALE

        if playback_s < 2.4 or source_duration < 1.6:
            ratios = [1.0]
            windows = [(0.0, 1.0)]
        elif playback_s < 5.4 or source_duration < 3.8:
            ratios = [0.46, 0.54]
            windows = [(0.0, 0.48), (0.55, 1.0)]
        else:
            ratios = [0.24, 0.34, 0.42]
            windows = [(0.0, 0.28), (0.34, 0.68), (0.72, 1.0)]

        clip_durations = self._split_duration(duration_us, ratios)
        clips: list[CreativeUnit] = []
        cursor_us = timeline_start_us

        for idx, (clip_duration_us, (start_ratio, end_ratio)) in enumerate(zip(clip_durations, windows), start=1):
            clip_start = movie_start + source_duration * start_ratio
            clip_end = movie_start + source_duration * end_ratio
            if clip_end - clip_start < 0.25:
                clip_start = movie_start
                clip_end = movie_end

            clips.append(
                CreativeUnit(
                    unit_type="movie_clip",
                    timeline_start_us=cursor_us,
                    duration_us=clip_duration_us,
                    source_start=clip_start,
                    source_end=clip_end,
                    label=f"shot_{idx}",
                )
            )
            cursor_us += clip_duration_us

        return clips

    @staticmethod
    def _split_duration(duration_us: int, ratios: list[float]) -> list[int]:
        raw = [max(int(duration_us * ratio), 1) for ratio in ratios]
        delta = duration_us - sum(raw)
        raw[-1] += delta
        return raw

    def _playback_duration(self, segment: Segment) -> float:
        if self.audio_source == "original":
            return max(0.0, segment.narration_end - segment.narration_start)
        if segment.segment_type == SegmentType.NO_NARRATION or not segment.tts_audio_path:
            return max(0.0, segment.narration_end - segment.narration_start)
        return max(0.0, compute_segment_duration(segment, self.audio_source))

    @staticmethod
    def _match_type(segment: Segment) -> str:
        if segment.movie_start is None or segment.movie_end is None:
            return "fallback"
        if segment.review_required or segment.match_confidence < 0.78:
            return "inferred"
        return getattr(segment, "match_type", "") or "exact"

    def _card_duration_us(self, duration_s: float, match_type: str, has_text: bool) -> int:
        if not has_text:
            return 0
        if match_type == "fallback":
            return int(duration_s * TIME_SCALE)
        card_s = max(self._CARD_MIN_S, min(self._CARD_MAX_S, duration_s * 0.18))
        if duration_s < 1.8:
            return 0
        return int(card_s * TIME_SCALE)

    @staticmethod
    def _summary_text(segment: Segment) -> str:
        raw_text = (segment.polished_text if segment.use_polished_text and segment.polished_text else segment.original_text).strip()
        if not raw_text:
            return ""
        text = " ".join(raw_text.split())
        if len(text) <= 28:
            return text
        return f"{text[:28].rstrip()}..."
