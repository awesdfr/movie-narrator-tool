"""Processing API routes and orchestration."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from api.routes.project import load_project, save_project
from api.websocket import manager
from models.project import ProcessingProgress, ProjectStatus
from models.segment import (
    AlignmentStatus,
    MatchCandidate,
    Segment,
    SegmentBatchUpdate,
    SegmentType,
    SegmentUpdate,
    TTSStatus,
)

router = APIRouter()
_processing_tasks: dict[str, asyncio.Task] = {}


class StartPolishRequest(BaseModel):
    style_preset: str = Field(default="movie_pro")


class ResegmentRequest(BaseModel):
    preserve_manual_matches: bool = Field(default=True)


class RematchRequest(BaseModel):
    candidate_top_k: Optional[int] = Field(default=None)


class RematchProjectRequest(BaseModel):
    preserve_manual_matches: bool = Field(default=True)


def _upsert_project(project):
    save_project(project)


def _segment_duration(segment: Segment) -> float:
    return max(0.0, segment.narration_end - segment.narration_start)


def _find_temporal_outliers(segments: list[Segment], deviation_threshold: float = 90.0) -> list[Segment]:
    """Find AUTO_ACCEPTED segments whose movie position deviates significantly from local timeline trend.

    Uses local neighbor interpolation: for each matched segment, predicts its movie position
    from its 2 nearest matched neighbors on each side.  Segments deviating by more than
    `deviation_threshold` seconds from the prediction are flagged for re-matching.
    """
    matched = [
        s for s in segments
        if s.movie_start is not None
        and s.segment_type != SegmentType.NON_MOVIE
        and not s.skip_matching
        and not s.review_required
        and (s.match_confidence or 0.0) < 0.96  # never disturb very high confidence
    ]
    if len(matched) < 7:
        return []

    outliers: list[Segment] = []
    for i in range(2, len(matched) - 2):
        seg = matched[i]
        prev2, prev1 = matched[i - 2], matched[i - 1]
        next1, next2 = matched[i + 1], matched[i + 2]

        # Rate estimated from left neighbors
        nd_left = prev1.narration_start - prev2.narration_start
        if nd_left < 0.5:
            continue
        md_left = prev1.movie_start - prev2.movie_start
        rate_left = md_left / nd_left
        expected_left = prev1.movie_start + rate_left * (seg.narration_start - prev1.narration_start)

        # Rate estimated from right neighbors
        nd_right = next2.narration_start - next1.narration_start
        if nd_right < 0.5:
            continue
        md_right = next2.movie_start - next1.movie_start
        rate_right = md_right / nd_right
        expected_right = next1.movie_start - rate_right * (next1.narration_start - seg.narration_start)

        expected = (expected_left + expected_right) / 2.0
        if abs(seg.movie_start - expected) > deviation_threshold:
            outliers.append(seg)

    return outliers


def _calculate_expected_movie_time(project, segment: Segment) -> Optional[float]:
    if not project.movie_duration or not project.narration_duration or project.narration_duration <= 0:
        return None
    return (segment.narration_start / project.narration_duration) * project.movie_duration


def _audio_weight_for_segment(segment: Segment) -> float:
    if segment.segment_type == SegmentType.NO_NARRATION:
        return 0.32
    if segment.audio_activity_label == "silent":
        return 0.18
    if segment.audio_activity_label == "weak":
        return 0.24
    speech_likelihood = float(segment.speech_likelihood or 0.0)
    if speech_likelihood >= 0.72:
        return 0.08
    if speech_likelihood >= 0.42:
        return 0.12
    return 0.18


def _combine_candidate_confidence(
    segment: Segment,
    visual_confidence: float,
    temporal_confidence: float,
    duration_gap: float,
    audio_confidence: float = 0.0,
    stability_score: float = 0.0,
) -> float:
    narr_duration = max(_segment_duration(segment), 1.0)
    duration_confidence = max(0.0, 1.0 - duration_gap / narr_duration)
    # 视觉为核心权重（90%）；temporal_confidence 基于线性时间估计，对非线性解说惩罚正确匹配，故不参与评分
    score = (
        visual_confidence * 0.90
        + duration_confidence * 0.04
        + stability_score * 0.06
    )
    # 音频仅在有可靠音频时加成，无音频不惩罚
    if audio_confidence > 0.15:
        aw = _audio_weight_for_segment(segment)
        score = score * (1.0 - aw * 0.5) + audio_confidence * aw * 0.5
    # 非线性校准：正确匹配（≥0.85）向1.0收敛
    # f(x) = 1-(1-x)^1.5: 0.85→0.942, 0.90→0.968, 0.93→0.983, 0.95→0.989
    if score >= 0.85:
        score = 1.0 - (1.0 - score) ** 1.5
    return min(1.0, max(0.0, score))


def _inject_gap_segments(raw_segments: list[dict], total_duration: Optional[float], min_gap_duration: float) -> list[dict]:
    if total_duration is None:
        total_duration = raw_segments[-1]["end"] if raw_segments else 0.0
    ordered = sorted(raw_segments, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    if not ordered:
        return [{"start": 0.0, "end": float(total_duration), "text": "", "type": SegmentType.NO_NARRATION}] if total_duration > min_gap_duration else []

    enriched: list[dict] = []
    cursor = 0.0
    for item in ordered:
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if start - cursor >= min_gap_duration:
            enriched.append(
                {
                    "start": cursor,
                    "end": start,
                    "text": "",
                    "type": SegmentType.NO_NARRATION,
                    "words": [],
                }
            )
        enriched.append(item)
        cursor = max(cursor, end)
    if float(total_duration) - cursor >= min_gap_duration:
        enriched.append(
            {
                "start": cursor,
                "end": float(total_duration),
                "text": "",
                "type": SegmentType.NO_NARRATION,
                "words": [],
            }
        )
    return enriched


def _candidate_from_result(
    segment: Segment,
    result: dict,
    rank: int,
    source: str,
    expected_movie_time: Optional[float],
    reason_prefix: str,
    audio_confidence: float = 0.0,
    audio_note: str = "",
    stability_score: float = 0.0,
    candidate_quality: float = 0.0,
    query_quality: float = 0.0,
    low_info_ratio: float = 0.0,
) -> MatchCandidate:
    narr_duration = _segment_duration(segment)
    movie_duration = max(0.0, float(result["end"]) - float(result["start"]))
    duration_gap = abs(movie_duration - narr_duration)
    center = (float(result["start"]) + float(result["end"])) / 2
    if expected_movie_time is None:
        temporal_confidence = 1.0
    else:
        temporal_confidence = max(0.0, 1.0 - abs(center - expected_movie_time) / max(120.0, narr_duration * 15.0, 1.0))
    visual_confidence = float(result.get("confidence", 0.0))
    combined = _combine_candidate_confidence(
        segment,
        visual_confidence=visual_confidence,
        temporal_confidence=temporal_confidence,
        duration_gap=duration_gap,
        audio_confidence=audio_confidence,
        stability_score=stability_score,
    )
    confidence_level = result.get("confidence_level", "")
    reason = (
        f"{reason_prefix}; visual={visual_confidence:.2f}, temporal={temporal_confidence:.2f}, "
        f"stability={stability_score:.2f}, duration_gap={duration_gap:.2f}s"
    )
    if audio_confidence > 0 or audio_note:
        reason += f", audio={audio_confidence:.2f}"
    if candidate_quality > 0 or query_quality > 0:
        reason += f", q={query_quality:.2f}/{candidate_quality:.2f}"
    if low_info_ratio > 0:
        reason += f", low_info={low_info_ratio:.2f}"
    if confidence_level:
        reason += f", level={confidence_level}"
    if audio_note:
        reason += f", audio_note={audio_note}"
    return MatchCandidate(
        id=f"{segment.id}_cand_{rank}",
        start=float(result["start"]),
        end=float(result["end"]),
        score=combined,
        confidence=combined,
        visual_confidence=visual_confidence,
        audio_confidence=audio_confidence,
        temporal_confidence=temporal_confidence,
        stability_score=stability_score,
        candidate_quality=candidate_quality,
        query_quality=query_quality,
        low_info_ratio=low_info_ratio,
        duration_gap=duration_gap,
        match_count=int(result.get("match_count", 0)),
        reason=reason,
        source=source,
        rank=rank,
    )


def _dedupe_candidates(candidates: list[MatchCandidate], tolerance: float = 12.0) -> list[MatchCandidate]:
    deduped: list[MatchCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(abs(existing.start - candidate.start) <= tolerance for existing in deduped):
            continue
        deduped.append(candidate)
    for idx, candidate in enumerate(deduped, start=1):
        candidate.rank = idx
        candidate.id = f"{candidate.id.split('_cand_')[0]}_cand_{idx}"
    return deduped


async def _rescore_candidates_with_audio(audio_scorer, segment: Segment, candidates: list[MatchCandidate]) -> list[MatchCandidate]:
    if not audio_scorer or not candidates:
        return candidates

    for candidate in candidates:
        try:
            audio_result = await audio_scorer.score_pair(
                narration_start=segment.narration_start,
                narration_end=segment.narration_end,
                movie_start=candidate.start,
                movie_end=candidate.end,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Audio rerank failed for {segment.id}/{candidate.id}: {exc}")
            continue

        candidate.audio_confidence = audio_result.confidence
        candidate.score = _combine_candidate_confidence(
            segment,
            visual_confidence=candidate.visual_confidence,
            temporal_confidence=candidate.temporal_confidence,
            duration_gap=candidate.duration_gap,
            audio_confidence=candidate.audio_confidence,
            stability_score=candidate.stability_score,
        )
        candidate.confidence = candidate.score
        if "audio=" not in candidate.reason:
            candidate.reason += f"; audio={candidate.audio_confidence:.2f}"
        if audio_result.note:
            candidate.reason += f", audio_note={audio_result.note}"

    candidates.sort(key=lambda item: item.score, reverse=True)
    for idx, candidate in enumerate(candidates, start=1):
        candidate.rank = idx
        candidate.id = f"{candidate.id.split('_cand_')[0]}_cand_{idx}"
    return candidates


def _clear_match(segment: Segment, status: AlignmentStatus, reason: str) -> None:
    segment.movie_start = None
    segment.movie_end = None
    segment.match_confidence = 0.0
    segment.visual_confidence = 0.0
    segment.audio_confidence = 0.0
    segment.temporal_confidence = 0.0
    segment.stability_score = 0.0
    segment.duration_gap = 0.0
    segment.match_reason = reason
    segment.alignment_status = status
    segment.review_required = status in {AlignmentStatus.NEEDS_REVIEW, AlignmentStatus.UNMATCHED}
    segment.selected_candidate_id = None
    segment.estimated_boundary_error = None


def _mark_segment_skipped(segment: Segment, reason: str = "Skipped from matching by user") -> None:
    _clear_match(segment, AlignmentStatus.SKIPPED, reason)
    segment.match_candidates = []
    segment.skip_matching = True
    segment.is_manual_match = False


def _apply_selected_candidate(segment: Segment, candidate: MatchCandidate, status: AlignmentStatus, review_required: bool) -> None:
    segment.movie_start = candidate.start
    segment.movie_end = candidate.end
    segment.match_confidence = candidate.confidence
    segment.visual_confidence = candidate.visual_confidence
    segment.audio_confidence = candidate.audio_confidence
    segment.temporal_confidence = candidate.temporal_confidence
    segment.stability_score = candidate.stability_score
    segment.duration_gap = candidate.duration_gap
    segment.match_reason = candidate.reason
    segment.alignment_status = status
    segment.review_required = review_required
    segment.selected_candidate_id = candidate.id
    segment.estimated_boundary_error = max(candidate.duration_gap / 2.0, 0.0)


def _compute_stats(segments: list[Segment]) -> dict:
    matched = [segment for segment in segments if segment.movie_start is not None]
    auto_accepted = [segment for segment in segments if segment.alignment_status == AlignmentStatus.AUTO_ACCEPTED]
    review_required = [segment for segment in segments if segment.review_required]
    boundary_errors = [segment.estimated_boundary_error for segment in matched if segment.estimated_boundary_error is not None]
    return {
        "total": len(segments),
        "matched": len(matched),
        "auto_accepted": len(auto_accepted),
        "review_required": len(review_required),
        "skipped": len([segment for segment in segments if segment.skip_matching]),
        "avg_confidence": sum(segment.match_confidence for segment in matched) / len(matched) if matched else 0.0,
        "avg_duration_gap": sum(segment.duration_gap for segment in matched) / len(matched) if matched else 0.0,
        "avg_boundary_error": sum(boundary_errors) / len(boundary_errors) if boundary_errors else None,
    }


def _manual_overlap(old_segment: Segment, new_segment: Segment) -> float:
    overlap = max(
        0.0,
        min(old_segment.narration_end, new_segment.narration_end)
        - max(old_segment.narration_start, new_segment.narration_start),
    )
    union = max(old_segment.narration_end, new_segment.narration_end) - min(
        old_segment.narration_start,
        new_segment.narration_start,
    )
    return overlap / union if union > 0 else 0.0


def _preserve_manual_segment_edits(old_segments: list[Segment], new_segments: list[Segment]) -> None:
    manual_segments = [segment for segment in old_segments if segment.is_manual_match or segment.polished_text or segment.skip_matching]
    for new_segment in new_segments:
        best = None
        best_score = 0.0
        for old_segment in manual_segments:
            score = _manual_overlap(old_segment, new_segment)
            if score > best_score:
                best = old_segment
                best_score = score
        if not best or best_score < 0.45:
            continue
        if best.skip_matching:
            _mark_segment_skipped(new_segment, "Preserved skip-matching selection after resegment")
        if best.is_manual_match and best.movie_start is not None and best.movie_end is not None:
            new_segment.movie_start = best.movie_start
            new_segment.movie_end = best.movie_end
            new_segment.match_confidence = best.match_confidence
            new_segment.visual_confidence = best.visual_confidence
            new_segment.audio_confidence = best.audio_confidence
            new_segment.temporal_confidence = best.temporal_confidence
            new_segment.stability_score = best.stability_score
            new_segment.duration_gap = best.duration_gap
            new_segment.match_reason = best.match_reason or "Preserved manual match after resegment"
            new_segment.alignment_status = AlignmentStatus.MANUAL
            new_segment.review_required = False
            new_segment.is_manual_match = True
            new_segment.selected_candidate_id = best.selected_candidate_id
            new_segment.match_candidates = list(best.match_candidates)
            new_segment.estimated_boundary_error = best.estimated_boundary_error
        if best.polished_text:
            new_segment.polished_text = best.polished_text


def _is_content_filter_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "content_filter" in message
        or "content management policy" in message
        or "response was filtered" in message
    )


def _neighbor_candidate_hint(segments: list[Segment], index: int, direction: int) -> Optional[float]:
    cursor = index + direction
    while 0 <= cursor < len(segments):
        neighbor = segments[cursor]
        if neighbor.match_candidates:
            top_candidate = max(neighbor.match_candidates, key=lambda item: item.score)
            return top_candidate.end if direction < 0 else top_candidate.start
        if neighbor.movie_start is not None and neighbor.movie_end is not None:
            return neighbor.movie_end if direction < 0 else neighbor.movie_start
        cursor += direction
    return None


def _build_llm_rerank_prompt(segments: list[Segment], index: int, segment: Segment) -> str:
    prev_movie_end = _neighbor_candidate_hint(segments, index, -1)
    next_movie_start = _neighbor_candidate_hint(segments, index, 1)
    candidate_lines = [
        (
            f"- id={candidate.id}; movie={candidate.start:.1f}-{candidate.end:.1f}; "
            f"score={candidate.score:.3f}; visual={candidate.visual_confidence:.2f}; "
            f"audio={candidate.audio_confidence:.2f}; temporal={candidate.temporal_confidence:.2f}; "
            f"stability={candidate.stability_score:.2f}; gap={candidate.duration_gap:.2f}s; "
            f"matches={candidate.match_count}; source={candidate.source}"
        )
        for candidate in segment.match_candidates
    ]
    return (
        "Choose the best candidate id using timing consistency and confidence metrics only. "
        "Return exactly one candidate id. Do not add explanation.\n\n"
        f"Segment duration: {_segment_duration(segment):.1f}s\n"
        f"Segment type: {segment.segment_type}\n"
        f"Speech activity: {segment.audio_activity_label}\n"
        f"Previous movie end hint: {prev_movie_end if prev_movie_end is not None else 'unknown'}\n"
        f"Next movie start hint: {next_movie_start if next_movie_start is not None else 'unknown'}\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
    )

async def update_progress(
    project_id: str,
    stage: str,
    progress: float,
    message: str,
    parallel_tasks: list | None = None,
    total_tasks: int | None = None,
    completed_tasks: int | None = None,
):
    """Persist and broadcast processing progress."""

    project = load_project(project_id)
    if project:
        project.progress = ProcessingProgress(stage=stage, progress=progress, message=message)
        save_project(project)

    payload = {"type": "progress", "stage": stage, "progress": progress, "message": message}
    if parallel_tasks is not None:
        payload["parallel_tasks"] = parallel_tasks
    if total_tasks is not None:
        payload["total_tasks"] = total_tasks
    if completed_tasks is not None:
        payload["completed_tasks"] = completed_tasks
    await manager.broadcast_to_project(project_id, payload)


async def _load_processing_dependencies():
    from api.routes.settings import load_settings
    from core.audio_processor.background_audio_scorer import AudioSimilarityScorer
    from core.audio_processor.segment_refiner import SegmentRefiner, SegmenterConfig
    from core.audio_processor.subtitle_parser import SubtitleParser
    from core.matcher.global_aligner import GlobalAlignmentOptimizer
    from core.video_processor.frame_extractor import FrameExtractor
    from core.video_processor.frame_matcher import FrameMatcher
    from core.video_processor.non_movie_detector import NonMovieDetector
    from core.video_processor.scene_detector import SceneDetector

    return {
        "load_settings": load_settings,
        "AudioSimilarityScorer": AudioSimilarityScorer,
        "FrameExtractor": FrameExtractor,
        "SceneDetector": SceneDetector,
        "SubtitleParser": SubtitleParser,
        "SegmentRefiner": SegmentRefiner,
        "SegmenterConfig": SegmenterConfig,
        "FrameMatcher": FrameMatcher,
        "NonMovieDetector": NonMovieDetector,
        "GlobalAlignmentOptimizer": GlobalAlignmentOptimizer,
    }


async def _build_refined_segments(project_id: str, project, app_settings, scenes: list[dict]) -> list[Segment]:
    deps = await _load_processing_dependencies()
    SubtitleParser = deps["SubtitleParser"]
    SegmentRefiner = deps["SegmentRefiner"]
    SegmenterConfig = deps["SegmenterConfig"]

    if project.subtitle_path:
        await update_progress(project_id, "recognizing", 0, "Parsing subtitles...")
        parser = SubtitleParser()
        transcription = parser.parse_srt(project.subtitle_path)
    else:
        from core.audio_processor.speech_recognizer import SpeechRecognizer

        await update_progress(project_id, "recognizing", 0, "Running speech recognition...")
        recognizer = SpeechRecognizer(word_timestamps=app_settings.whisper.word_timestamps)
        transcription = await recognizer.transcribe(project.narration_path)

    from core.audio_processor.voiceprint import VoiceprintRecognizer

    await update_progress(project_id, "recognizing", 45, "Separating narration voice...")
    voiceprint = VoiceprintRecognizer(threshold=app_settings.voiceprint.threshold)
    if project.reference_audio_path:
        await voiceprint.load_reference(project.reference_audio_path)
    narration_segments = await voiceprint.identify_narrator(project.narration_path, transcription)
    narration_segments = _inject_gap_segments(
        narration_segments,
        total_duration=project.narration_duration,
        min_gap_duration=max(1.0, app_settings.segmentation.min_segment_duration),
    )

    refiner = SegmentRefiner(
        SegmenterConfig(
            min_segment_duration=app_settings.segmentation.min_segment_duration,
            max_segment_duration=app_settings.segmentation.max_segment_duration,
            split_pause_seconds=app_settings.segmentation.split_pause_seconds,
            merge_gap_seconds=app_settings.segmentation.merge_gap_seconds,
            sentence_snap_tolerance=app_settings.segmentation.sentence_snap_tolerance,
            enable_scene_snap=app_settings.segmentation.enable_scene_snap,
            prefer_word_timestamps=app_settings.segmentation.prefer_word_timestamps,
        )
    )
    refined = refiner.refine(narration_segments, scenes=scenes)

    segments: list[Segment] = []
    for idx, item in enumerate(refined):
        segments.append(
            Segment(
                id=f"seg_{idx:04d}",
                index=idx,
                narration_start=float(item["start"]),
                narration_end=float(item["end"]),
                movie_start=None,
                movie_end=None,
                segment_type=item.get("type", SegmentType.HAS_NARRATION),
                original_text=item.get("text", ""),
                speech_likelihood=float(item.get("speech_likelihood") or 0.0),
                audio_activity_label=str(item.get("audio_activity_label") or "unknown"),
                voiceprint_similarity=item.get("voiceprint_similarity"),
                match_reason=(
                    "No narration voice detected in this segment"
                    if item.get("type") == SegmentType.NO_NARRATION
                    else ""
                ),
            )
        )

    await update_progress(project_id, "recognizing", 100, f"Segmentation complete: {len(segments)} segments")
    return segments


async def _mark_non_movie_segments(project, segments: list[Segment]) -> None:
    deps = await _load_processing_dependencies()
    detector = deps["NonMovieDetector"]()
    target_segments = [segment for segment in segments if segment.segment_type != SegmentType.HAS_NARRATION or not segment.original_text.strip()]
    if not target_segments:
        return
    batch = [(segment.narration_start, segment.narration_end) for segment in target_segments]
    results = await detector.detect_batch(project.narration_path, batch)
    for segment, is_non_movie in zip(target_segments, results):
        if is_non_movie:
            segment.segment_type = SegmentType.NON_MOVIE
            segment.match_reason = "Detected as non-movie segment"
            segment.alignment_status = AlignmentStatus.NON_MOVIE
            segment.review_required = False


async def _collect_candidates_for_segment(
    frame_matcher,
    project,
    segment: Segment,
    app_settings,
    base_result: Optional[dict],
    neighbor_hint: Optional[float] = None,
    audio_scorer=None,
) -> list[MatchCandidate]:
    if segment.segment_type == SegmentType.NON_MOVIE or segment.skip_matching:
        return []

    expected_movie_time = _calculate_expected_movie_time(project, segment)
    candidate_top_k = app_settings.match.candidate_top_k
    candidates: list[MatchCandidate] = []

    if base_result and base_result.get("success"):
        candidates.append(
            _candidate_from_result(
                segment,
                base_result,
                rank=1,
                source="batch_fast",
                expected_movie_time=expected_movie_time,
                reason_prefix="Batch cluster match",
                stability_score=float(base_result.get("stability_score", 0.0)),
                candidate_quality=float(base_result.get("candidate_quality", 0.0)),
                query_quality=float(base_result.get("query_quality", 0.0)),
                low_info_ratio=float(base_result.get("low_info_ratio", 0.0)),
            )
        )

    # 高置信度批次结果：仅在附近做少量精细搜索，跳过耗时的全局搜索
    batch_confidence = float(base_result.get("confidence", 0.0)) if base_result and base_result.get("success") else 0.0
    skip_global_search = batch_confidence >= 0.72

    hints: list[Optional[float]] = []
    if neighbor_hint is not None:
        hints.append(neighbor_hint)
    if expected_movie_time is not None:
        hints.append(expected_movie_time)
    if base_result and base_result.get("success"):
        hints.append((float(base_result["start"]) + float(base_result["end"])) / 2)
        seg_duration = _segment_duration(segment)
        hints.append(float(base_result["start"]) - seg_duration)
        hints.append(float(base_result["start"]) + seg_duration)
        hints.append(float(base_result["start"]) - 30.0)
        hints.append(float(base_result["start"]) + 30.0)
    if not skip_global_search:
        hints.append(None)

    unique_hints: list[Optional[float]] = []
    for hint in hints:
        if hint is None:
            if None not in unique_hints:
                unique_hints.append(None)
            continue
        if not any(existing is not None and abs(existing - hint) <= 6.0 for existing in unique_hints):
            unique_hints.append(max(0.0, hint))

    # 预提取特征一次，在所有 hint 的 match_segment 调用中复用（避免重复 I/O）
    precomputed = await frame_matcher._extract_segment_features(
        project.narration_path, segment.narration_start, segment.narration_end
    )

    for hint in unique_hints[: candidate_top_k + 2]:
        for relaxed in (False, True):
            result = await frame_matcher.match_segment(
                project.narration_path,
                segment.narration_start,
                segment.narration_end,
                time_hint=hint,
                relaxed=relaxed,
                strict_window=hint is not None,
                precomputed_features=precomputed,
            )
            if not result:
                continue
            candidates.append(
                _candidate_from_result(
                    segment,
                    result,
                    rank=len(candidates) + 1,
                    source="local_refine_relaxed" if relaxed else "local_refine",
                    expected_movie_time=expected_movie_time,
                    reason_prefix="Local rerank match",
                    stability_score=float(result.get("stability_score", 0.0)),
                    candidate_quality=float(result.get("candidate_quality", 0.0)),
                    query_quality=float(result.get("query_quality", 0.0)),
                    low_info_ratio=float(result.get("low_info_ratio", 0.0)),
                )
            )
            if len(candidates) >= candidate_top_k * 2:
                break
        if len(candidates) >= candidate_top_k * 2:
            break

    deduped = _dedupe_candidates(candidates)
    deduped = await _rescore_candidates_with_audio(audio_scorer, segment, deduped)
    return deduped[:candidate_top_k]

async def _maybe_llm_rerank(project, segments: list[Segment], app_settings) -> None:
    if not app_settings.match.use_multimodal_rerank:
        return
    if not app_settings.ai.api_key:
        return

    from core.ai_service.api_manager import APIManager

    api = APIManager(api_base=app_settings.ai.api_base, api_key=app_settings.ai.api_key, model=app_settings.ai.model)
    disable_remaining_rerank = False
    for idx, segment in enumerate(segments):
        if disable_remaining_rerank:
            break
        if segment.skip_matching:
            continue
        if not segment.original_text.strip():
            continue
        if len(segment.match_candidates) < 2 or segment.match_confidence >= app_settings.match.medium_confidence_threshold:
            continue
        prompt = _build_llm_rerank_prompt(segments, idx, segment)
        try:
            choice = (
                await api.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You rerank movie match candidates from numeric metadata only. "
                                "Reply with one candidate id exactly as listed."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=12,
                )
            ).strip()
        except Exception as exc:  # pragma: no cover
            if _is_content_filter_error(exc):
                logger.warning(
                    f"LLM rerank disabled for remaining segments after content filter on {segment.id}"
                )
                disable_remaining_rerank = True
                continue
            logger.warning(f"LLM rerank failed for {segment.id}: {exc}")
            continue
        choice = choice.splitlines()[0].strip()
        for candidate in segment.match_candidates:
            if candidate.id == choice or candidate.id in choice:
                candidate.score = min(1.0, candidate.score + 0.05)
                candidate.confidence = min(1.0, candidate.confidence + 0.05)
                candidate.reason += " | LLM rerank preferred"
                break
        segment.match_candidates.sort(key=lambda item: item.score, reverse=True)
        for rank, candidate in enumerate(segment.match_candidates, start=1):
            candidate.rank = rank
    await api.close()


async def _match_segments(
    project_id: str,
    project,
    segments: list[Segment],
    app_settings,
    preserve_manual_matches: bool = False,
) -> None:
    deps = await _load_processing_dependencies()
    AudioSimilarityScorer = deps["AudioSimilarityScorer"]
    FrameMatcher = deps["FrameMatcher"]
    GlobalAlignmentOptimizer = deps["GlobalAlignmentOptimizer"]

    active_segments: list[Segment] = []
    for segment in segments:
        if segment.skip_matching:
            _mark_segment_skipped(segment)
            continue
        if preserve_manual_matches and segment.is_manual_match and segment.movie_start is not None and segment.movie_end is not None:
            segment.alignment_status = AlignmentStatus.MANUAL
            segment.review_required = False
            continue
        active_segments.append(segment)

    if not active_segments:
        await update_progress(project_id, "matching", 100, "No eligible segments need matching")
        return

    frame_matcher = FrameMatcher(
        phash_threshold=app_settings.match.phash_threshold,
        match_threshold=app_settings.match.frame_match_threshold,
        use_deep_learning=app_settings.match.use_deep_learning,
        index_sample_fps=app_settings.match.index_sample_fps,
        fast_mode=app_settings.match.fast_mode,
        subtitle_mask_mode=getattr(project, "subtitle_mask_mode", "hybrid"),
        movie_subtitle_regions=[region.model_dump() for region in getattr(project, "movie_subtitle_regions", [])],
        narration_subtitle_regions=[region.model_dump() for region in getattr(project, "narration_subtitle_regions", [])],
    )

    cache_dir = Path(__file__).resolve().parents[2] / "temp" / "match_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    movie_name = Path(project.movie_path).stem
    frame_cache_path = cache_dir / f"{movie_name}_frame.pkl"
    await update_progress(project_id, "matching", 1, "Preparing movie frame index...")

    last_index_progress = -1

    def on_index_progress(progress_pct: float, message: str):
        nonlocal last_index_progress
        mapped_progress = min(12.0, max(1.0, round(progress_pct * 12 / 100, 1)))
        if mapped_progress <= last_index_progress and progress_pct < 100:
            return
        last_index_progress = mapped_progress
        asyncio.create_task(update_progress(project_id, "matching", mapped_progress, message))

    await frame_matcher.build_index(
        project.movie_path,
        sample_interval=app_settings.match.sample_interval,
        cache_path=frame_cache_path,
        progress_callback=on_index_progress,
    )
    await update_progress(project_id, "matching", 12, "Preparing background audio references...")
    audio_scorer = AudioSimilarityScorer()
    audio_cache_dir = Path(__file__).resolve().parents[2] / "temp" / "audio_cache"
    try:
        await audio_scorer.prepare(project.movie_path, project.narration_path, output_dir=audio_cache_dir)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Audio scorer preparation failed for {project_id}: {exc}")
        audio_scorer = None
    await update_progress(project_id, "matching", 15, "Matching narration to the movie...")

    segment_tasks = [{"id": segment.id, "start": segment.narration_start, "end": segment.narration_end} for segment in active_segments]

    last_batch_progress = -1

    def on_progress(stage, progress_pct, message):
        nonlocal last_batch_progress
        mapped_progress = min(55.0, max(15.0, round(15 + 40 * progress_pct / 100, 1)))
        if mapped_progress <= last_batch_progress and progress_pct < 100:
            return
        last_batch_progress = mapped_progress
        logger.info(f"match[{stage}] {mapped_progress}% {message}")
        asyncio.create_task(update_progress(project_id, "matching", mapped_progress, message))

    batch_results = await frame_matcher.match_all_segments_fast(
        narration_path=project.narration_path,
        segments=segment_tasks,
        sample_fps=4.0,
        progress_callback=on_progress,
        movie_duration=project.movie_duration,
        narration_duration=project.narration_duration,
        allow_non_sequential=app_settings.match.allow_non_sequential,
        max_concurrent=app_settings.concurrency.match_concurrency,
    )
    batch_map = {item["id"]: item for item in batch_results}

    # 预计算邻居提示：为每个片段提供前后邻居的批量匹配结果作为时序锚点
    neighbor_hint_map: dict[str, Optional[float]] = {}
    for i, seg in enumerate(active_segments):
        hints_near: list[float] = []
        for delta in (-1, 1):
            j = i + delta
            if 0 <= j < len(active_segments):
                nb = batch_map.get(active_segments[j].id)
                if nb and nb.get("success") and nb.get("start") is not None:
                    hints_near.append(float(nb["start"]))
        neighbor_hint_map[seg.id] = float(sum(hints_near) / len(hints_near)) if hints_near else None

    semaphore = asyncio.Semaphore(max(1, app_settings.concurrency.match_concurrency))
    completed = 0

    async def collect(segment: Segment):
        nonlocal completed
        async with semaphore:
            segment.match_candidates = await _collect_candidates_for_segment(
                frame_matcher,
                project,
                segment,
                app_settings,
                batch_map.get(segment.id),
                neighbor_hint=neighbor_hint_map.get(segment.id),
                audio_scorer=audio_scorer,
            )
            completed += 1
            await update_progress(
                project_id,
                "matching",
                55 + int(25 * completed / max(1, len(active_segments))),
                f"Collected candidates {completed}/{len(active_segments)}",
            )

    await asyncio.gather(*(collect(segment) for segment in active_segments))
    await _maybe_llm_rerank(project, active_segments, app_settings)

    optimizer = GlobalAlignmentOptimizer(
        auto_accept_threshold=app_settings.match.high_confidence_threshold,
        review_threshold=app_settings.match.medium_confidence_threshold,
        backtrack_penalty=app_settings.match.global_backtrack_penalty,
        duplicate_scene_penalty=app_settings.match.duplicate_scene_penalty,
    )

    def apply_optimizer_results():
        results = optimizer.optimize(active_segments, allow_non_sequential=app_settings.match.allow_non_sequential)
        for segment, result in zip(active_segments, results):
            candidate = result["candidate"]
            status = result["alignment_status"]
            review_required = result["review_required"]
            if candidate is None:
                _clear_match(segment, status, segment.match_reason or "No stable candidate selected")
                continue
            _apply_selected_candidate(segment, candidate, status, review_required)

    apply_optimizer_results()

    # 时序异常检测：将位置偏离邻居预测的 AUTO_ACCEPTED 片段标记为需复查，纳入二次匹配
    if app_settings.match.rerank_low_confidence:
        temporal_outliers = _find_temporal_outliers(active_segments)
        for seg in temporal_outliers:
            seg.review_required = True
            logger.debug(f"Temporal outlier flagged for rerank: segment {seg.id} movie_start={seg.movie_start:.1f}")

    if app_settings.match.rerank_low_confidence:
        low_confidence_segments = [
            segment
            for segment in active_segments
            if segment.review_required and segment.segment_type != SegmentType.NON_MOVIE and not segment.skip_matching
        ]
        if low_confidence_segments:
            completed = 0

            async def enrich(segment: Segment):
                nonlocal completed
                async with semaphore:
                    neighbor_hint = None
                    idx = active_segments.index(segment)
                    prev_segment = next((item for item in reversed(active_segments[:idx]) if item.movie_end is not None), None)
                    next_segment = next((item for item in active_segments[idx + 1 :] if item.movie_start is not None), None)
                    if prev_segment and next_segment and next_segment.narration_start > prev_segment.narration_start:
                        ratio = (segment.narration_start - prev_segment.narration_start) / (
                            next_segment.narration_start - prev_segment.narration_start
                        )
                        neighbor_hint = prev_segment.movie_start + ratio * (next_segment.movie_start - prev_segment.movie_start)
                    elif prev_segment:
                        neighbor_hint = prev_segment.movie_end
                    elif next_segment:
                        neighbor_hint = max(0.0, next_segment.movie_start - _segment_duration(segment))
                    extra_candidates = await _collect_candidates_for_segment(
                        frame_matcher,
                        project,
                        segment,
                        app_settings,
                        batch_map.get(segment.id),
                        neighbor_hint=neighbor_hint,
                        audio_scorer=audio_scorer,
                    )
                    segment.match_candidates = _dedupe_candidates(list(segment.match_candidates) + extra_candidates)[: app_settings.match.candidate_top_k]
                    completed += 1
                    await update_progress(
                        project_id,
                        "matching",
                        82 + int(12 * completed / max(1, len(low_confidence_segments))),
                        f"Reranking low-confidence segments {completed}/{len(low_confidence_segments)}",
                    )

            await asyncio.gather(*(enrich(segment) for segment in low_confidence_segments))
            await _maybe_llm_rerank(project, low_confidence_segments, app_settings)
            apply_optimizer_results()


async def process_project_task(project_id: str):
    """Main project processing task."""

    project = load_project(project_id)
    if not project:
        return

    try:
        deps = await _load_processing_dependencies()
        load_settings = deps["load_settings"]
        FrameExtractor = deps["FrameExtractor"]
        SceneDetector = deps["SceneDetector"]
        app_settings = load_settings()

        project.status = ProjectStatus.ANALYZING
        _upsert_project(project)
        await update_progress(project_id, "analyzing", 0, "Reading video metadata...")

        frame_extractor = FrameExtractor()
        scene_detector = SceneDetector()
        narration_info = await frame_extractor.get_video_info(project.narration_path)
        movie_info = await frame_extractor.get_video_info(project.movie_path)
        project.narration_duration = narration_info["duration"]
        project.narration_fps = narration_info["fps"]
        project.movie_duration = movie_info["duration"]
        project.movie_fps = movie_info["fps"]
        project.movie_resolution = (movie_info["width"], movie_info["height"])
        _upsert_project(project)

        await update_progress(project_id, "analyzing", 25, "Detecting scene changes...")

        def report_scene_detection_progress(progress_value: float, message: str) -> None:
            asyncio.create_task(update_progress(project_id, "analyzing", progress_value, message))

        scenes = await scene_detector.detect_scenes(
            project.narration_path,
            progress_callback=report_scene_detection_progress,
        )
        await update_progress(project_id, "analyzing", 40, f"Detected {len(scenes)} scene boundaries")

        project.status = ProjectStatus.RECOGNIZING
        _upsert_project(project)
        segments = await _build_refined_segments(project_id, project, app_settings, scenes)
        await _mark_non_movie_segments(project, segments)
        project.segments = segments
        _upsert_project(project)

        project.status = ProjectStatus.MATCHING
        _upsert_project(project)
        await update_progress(project_id, "matching", 0, "Matching narration to the movie...")
        await _match_segments(project_id, project, segments, app_settings)

        stats = _compute_stats(segments)
        project.status = ProjectStatus.READY_FOR_POLISH
        _upsert_project(project)
        await update_progress(
            project_id,
            "ready_for_polish",
            100,
            "Frame matching complete: {matched}/{total} matched, {auto} auto accepted, {review} need review, {skipped} skipped".format(
                matched=stats["matched"],
                total=stats["total"],
                auto=stats["auto_accepted"],
                review=stats["review_required"],
                skipped=stats["skipped"],
            ),
        )
    except Exception as exc:  # pragma: no cover
        logger.exception(f"Project processing failed: {project_id}")
        project = load_project(project_id)
        if project:
            project.status = ProjectStatus.ERROR
            project.progress.message = str(exc)
            _upsert_project(project)
        await update_progress(project_id, "error", 0, f"Processing failed: {exc}")
    finally:
        _processing_tasks.pop(project_id, None)


async def rematch_project_task(project_id: str, preserve_manual_matches: bool = True):
    """Re-run matching for current segments while respecting skip flags."""

    project = load_project(project_id)
    if not project:
        return

    try:
        deps = await _load_processing_dependencies()
        load_settings = deps["load_settings"]
        if not project.segments:
            raise RuntimeError("Project has no segments to rematch")

        app_settings = load_settings()
        project.status = ProjectStatus.MATCHING
        _upsert_project(project)
        await update_progress(project_id, "matching", 0, "Rematching unskipped segments...")
        await _match_segments(
            project_id,
            project,
            project.segments,
            app_settings,
            preserve_manual_matches=preserve_manual_matches,
        )

        stats = _compute_stats(project.segments)
        project.status = ProjectStatus.READY_FOR_POLISH
        _upsert_project(project)
        await update_progress(
            project_id,
            "ready_for_polish",
            100,
            "Rematch complete: {matched}/{total} matched, {auto} auto accepted, {review} need review, {skipped} skipped".format(
                matched=stats["matched"],
                total=stats["total"],
                auto=stats["auto_accepted"],
                review=stats["review_required"],
                skipped=stats["skipped"],
            ),
        )
    except Exception as exc:  # pragma: no cover
        logger.exception(f"Project rematch failed: {project_id}")
        project = load_project(project_id)
        if project:
            project.status = ProjectStatus.ERROR
            project.progress.message = str(exc)
            _upsert_project(project)
        await update_progress(project_id, "error", 0, f"Rematch failed: {exc}")
    finally:
        _processing_tasks.pop(project_id, None)

@router.post("/{project_id}/start")
async def start_processing(project_id: str, background_tasks: BackgroundTasks):  # noqa: ARG001
    """Start the full processing pipeline."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project_id in _processing_tasks and not _processing_tasks[project_id].done():
        raise HTTPException(status_code=400, detail="Project is already processing")

    if project.status in {
        ProjectStatus.READY_FOR_POLISH,
        ProjectStatus.READY_FOR_TTS,
        ProjectStatus.COMPLETED,
        ProjectStatus.ERROR,
    }:
        project.segments = []
        project.progress = ProcessingProgress()
        tts_dir = Path(__file__).resolve().parents[2] / "temp" / project_id / "tts"
        if tts_dir.exists():
            import shutil

            shutil.rmtree(tts_dir, ignore_errors=True)

    task = asyncio.create_task(process_project_task(project_id))
    _processing_tasks[project_id] = task
    project.status = ProjectStatus.ANALYZING
    project.progress = ProcessingProgress(stage="analyzing", progress=0, message="Preparing processing dependencies...")
    _upsert_project(project)
    return {"message": "Processing started", "project_id": project_id}


@router.post("/{project_id}/stop")
async def stop_processing(project_id: str):
    """Stop a running processing job."""

    if project_id in _processing_tasks:
        task = _processing_tasks[project_id]
        if not task.done():
            task.cancel()
        _processing_tasks.pop(project_id, None)

    project = load_project(project_id)
    if project:
        project.status = ProjectStatus.ERROR
        project.progress.message = "Cancelled by user"
        _upsert_project(project)
    return {"message": "Processing stopped"}


@router.post("/{project_id}/rematch")
async def rematch_project(project_id: str, request: RematchProjectRequest = RematchProjectRequest()):
    """Re-run matching on existing segments and skip user-excluded ones."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.segments:
        raise HTTPException(status_code=400, detail="Project has no segments")
    if project_id in _processing_tasks and not _processing_tasks[project_id].done():
        raise HTTPException(status_code=400, detail="Project is already processing")

    task = asyncio.create_task(rematch_project_task(project_id, request.preserve_manual_matches))
    _processing_tasks[project_id] = task
    project.status = ProjectStatus.MATCHING
    project.progress = ProcessingProgress(stage="matching", progress=0, message="Preparing rematch dependencies...")
    _upsert_project(project)
    return {
        "message": "Project rematch started",
        "project_id": project_id,
        "preserve_manual_matches": request.preserve_manual_matches,
    }


@router.get("/{project_id}/progress")
async def get_progress(project_id: str):
    """Get processing progress."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": project.status, "progress": project.progress}


@router.get("/{project_id}/segments")
async def get_segments(project_id: str):
    """Get project segments with candidate metadata."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.segments


@router.put("/{project_id}/segments/{segment_id}")
async def update_segment(project_id: str, segment_id: str, update: SegmentUpdate):
    """Update a single segment."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    segment = next((item for item in project.segments if item.id == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    update_data = update.model_dump(exclude_unset=True)
    movie_changed = any(key in update_data for key in ["movie_start", "movie_end"])
    for key, value in update_data.items():
        setattr(segment, key, value)

    if "skip_matching" in update_data:
        if segment.skip_matching:
            _mark_segment_skipped(segment)
        else:
            if segment.alignment_status == AlignmentStatus.SKIPPED:
                segment.alignment_status = AlignmentStatus.PENDING
                segment.match_reason = "Ready to match"
            if segment.match_reason == "Skipped from matching by user":
                segment.match_reason = "Ready to match"

    if movie_changed:
        segment.is_manual_match = True
        segment.skip_matching = False
        segment.alignment_status = AlignmentStatus.MANUAL
        segment.review_required = False
        if "match_confidence" not in update_data:
            segment.match_confidence = 0.0
        if "visual_confidence" not in update_data:
            segment.visual_confidence = 0.0
        if "audio_confidence" not in update_data:
            segment.audio_confidence = 0.0
        if "temporal_confidence" not in update_data:
            segment.temporal_confidence = 0.0
        if "stability_score" not in update_data:
            segment.stability_score = 0.0
        if "duration_gap" not in update_data:
            segment.duration_gap = abs(
                max(0.0, (segment.movie_end or 0.0) - (segment.movie_start or 0.0)) - _segment_duration(segment)
            )
        if "selected_candidate_id" not in update_data:
            segment.selected_candidate_id = None
        if segment.movie_start is not None and segment.movie_end is not None and not segment.match_reason:
            segment.match_reason = "Manually adjusted in editor"

    _upsert_project(project)
    return segment


@router.post("/{project_id}/segments/batch")
async def batch_update_segments(project_id: str, batch: SegmentBatchUpdate):
    """Batch update segments."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = batch.model_dump(exclude={"segment_ids"}, exclude_unset=True)
    updated = 0
    updated_segments: list[Segment] = []
    for segment in project.segments:
        if segment.id not in batch.segment_ids:
            continue
        for key, value in update_data.items():
            if value is not None:
                setattr(segment, key, value)
        if update_data.get("skip_matching") is True:
            _mark_segment_skipped(segment)
        elif update_data.get("skip_matching") is False and segment.alignment_status == AlignmentStatus.SKIPPED:
            segment.alignment_status = AlignmentStatus.PENDING
            segment.match_reason = "Ready to match"
        updated += 1
        updated_segments.append(segment)

    _upsert_project(project)
    return {"message": f"Updated {updated} segments", "segments": updated_segments}


@router.post("/{project_id}/resegment")
async def resegment_project(project_id: str, request: ResegmentRequest = ResegmentRequest()):
    """Re-run segmentation without recreating the project."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project_id in _processing_tasks and not _processing_tasks[project_id].done():
        raise HTTPException(status_code=400, detail="Project is already processing")

    deps = await _load_processing_dependencies()
    FrameExtractor = deps["FrameExtractor"]
    SceneDetector = deps["SceneDetector"]
    load_settings = deps["load_settings"]
    app_settings = load_settings()
    frame_extractor = FrameExtractor()
    scene_detector = SceneDetector()
    if not project.narration_duration or not project.narration_fps:
        narration_info = await frame_extractor.get_video_info(project.narration_path)
        project.narration_duration = narration_info["duration"]
        project.narration_fps = narration_info["fps"]
    scenes = await scene_detector.detect_scenes(project.narration_path)
    old_segments = list(project.segments)
    new_segments = await _build_refined_segments(project_id, project, app_settings, scenes)
    await _mark_non_movie_segments(project, new_segments)
    if request.preserve_manual_matches:
        _preserve_manual_segment_edits(old_segments, new_segments)
    project.segments = new_segments
    _upsert_project(project)
    return {"message": "Segments regenerated", "segments": project.segments}


@router.post("/{project_id}/segments/{segment_id}/rematch")
async def rematch_segment(project_id: str, segment_id: str, request: RematchRequest = RematchRequest()):
    """Re-run matching for a single segment."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    segment = next((item for item in project.segments if item.id == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    if segment.skip_matching:
        raise HTTPException(status_code=400, detail="This segment is marked to skip matching")

    from api.routes.settings import load_settings
    from core.audio_processor.background_audio_scorer import AudioSimilarityScorer
    from core.matcher.global_aligner import GlobalAlignmentOptimizer
    from core.video_processor.frame_matcher import FrameMatcher

    app_settings = load_settings()
    if request.candidate_top_k is not None:
        app_settings.match.candidate_top_k = request.candidate_top_k

    frame_matcher = FrameMatcher(
        phash_threshold=app_settings.match.phash_threshold,
        match_threshold=app_settings.match.frame_match_threshold,
        use_deep_learning=app_settings.match.use_deep_learning,
        index_sample_fps=app_settings.match.index_sample_fps,
        fast_mode=app_settings.match.fast_mode,
        subtitle_mask_mode=getattr(project, "subtitle_mask_mode", "hybrid"),
        movie_subtitle_regions=[region.model_dump() for region in getattr(project, "movie_subtitle_regions", [])],
        narration_subtitle_regions=[region.model_dump() for region in getattr(project, "narration_subtitle_regions", [])],
    )
    cache_dir = Path(__file__).resolve().parents[2] / "temp" / "match_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    movie_name = Path(project.movie_path).stem
    frame_cache_path = cache_dir / f"{movie_name}_frame.pkl"
    await frame_matcher.build_index(project.movie_path, sample_interval=app_settings.match.sample_interval, cache_path=frame_cache_path)
    audio_scorer = AudioSimilarityScorer()
    audio_cache_dir = Path(__file__).resolve().parents[2] / "temp" / "audio_cache"
    try:
        await audio_scorer.prepare(project.movie_path, project.narration_path, output_dir=audio_cache_dir)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Audio scorer preparation failed for segment rematch {project_id}/{segment_id}: {exc}")
        audio_scorer = None

    idx = project.segments.index(segment)
    neighbor_hint = None
    prev_segment = next((item for item in reversed(project.segments[:idx]) if item.movie_end is not None), None)
    next_segment = next((item for item in project.segments[idx + 1 :] if item.movie_start is not None), None)
    if prev_segment and next_segment and next_segment.narration_start > prev_segment.narration_start:
        ratio = (segment.narration_start - prev_segment.narration_start) / (next_segment.narration_start - prev_segment.narration_start)
        neighbor_hint = prev_segment.movie_start + ratio * (next_segment.movie_start - prev_segment.movie_start)
    elif prev_segment:
        neighbor_hint = prev_segment.movie_end
    elif next_segment:
        neighbor_hint = max(0.0, next_segment.movie_start - _segment_duration(segment))

    candidate_results = await _collect_candidates_for_segment(
        frame_matcher,
        project,
        segment,
        app_settings,
        None,
        neighbor_hint=neighbor_hint,
        audio_scorer=audio_scorer,
    )
    segment.match_candidates = candidate_results
    optimizer = GlobalAlignmentOptimizer(
        auto_accept_threshold=app_settings.match.high_confidence_threshold,
        review_threshold=app_settings.match.medium_confidence_threshold,
        backtrack_penalty=app_settings.match.global_backtrack_penalty,
        duplicate_scene_penalty=app_settings.match.duplicate_scene_penalty,
    )
    temp_segments = []
    for item in project.segments[max(0, idx - 1) : idx + 2]:
        clone = item.model_copy(deep=True)
        if clone.id == segment.id:
            clone.match_candidates = candidate_results
        temp_segments.append(clone)
    result = optimizer.optimize(temp_segments, allow_non_sequential=app_settings.match.allow_non_sequential)[min(idx, 1)]
    candidate = result["candidate"]
    if candidate is None:
        _clear_match(segment, AlignmentStatus.UNMATCHED, "No stable candidate after rematch")
    else:
        _apply_selected_candidate(segment, candidate, AlignmentStatus.REMATCHED, candidate.confidence < app_settings.match.high_confidence_threshold)
        segment.is_manual_match = False
    _upsert_project(project)
    return segment


@router.post("/{project_id}/segments/{segment_id}/regenerate-tts")
async def regenerate_segment_tts(project_id: str, segment_id: str):
    """Generate TTS for one segment."""

    from api.routes.settings import load_settings
    from core.tts_service.tts_client import TTSClient

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    segment = next((item for item in project.segments if item.id == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    text = segment.polished_text if segment.use_polished_text and segment.polished_text else segment.original_text
    if not text:
        raise HTTPException(status_code=400, detail="Segment has no narration text")

    app_settings = load_settings()
    tts_client = TTSClient(
        api_base=app_settings.tts.api_base,
        api_endpoint=app_settings.tts.api_endpoint,
        reference_audio=app_settings.tts.reference_audio,
        infer_mode=app_settings.tts.infer_mode,
    )
    try:
        audio_path = await tts_client.generate(text, f"{project_id}_{segment_id}")
        segment.tts_audio_path = str(audio_path)
        segment.tts_duration = await tts_client.get_duration(str(audio_path))
        segment.tts_status = TTSStatus.GENERATED
        segment.tts_error = None
    except Exception as exc:  # pragma: no cover
        segment.tts_status = TTSStatus.FAILED
        segment.tts_error = str(exc)
        _upsert_project(project)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}")

    _upsert_project(project)
    return {"audio_path": audio_path, "duration": segment.tts_duration, "tts_status": segment.tts_status}

async def _polish_project_task(project_id: str, style_preset: str = "movie_pro"):
    """Background task for contextual narration polishing."""

    from api.routes.settings import load_settings
    from core.ai_service.api_manager import APIManager
    from core.ai_service.text_polisher import TextPolisher

    project = load_project(project_id)
    if not project:
        return

    app_settings = load_settings()
    api_manager = APIManager(api_base=app_settings.ai.api_base, api_key=app_settings.ai.api_key, model=app_settings.ai.model)
    polisher = TextPolisher(
        api_manager=api_manager,
        template=app_settings.ai.polish_template,
        temperature=app_settings.ai.temperature,
        max_tokens=app_settings.ai.max_tokens,
        default_style_preset=style_preset or app_settings.ai.polish_style_preset,
        enable_de_ai_pass=app_settings.ai.enable_de_ai_pass,
        enable_self_review=app_settings.ai.enable_self_review,
    )

    try:
        await update_progress(project_id, "polishing", 0, "Polishing narration text...")
        segments_with_text = [segment for segment in project.segments if segment.original_text]
        total = len(segments_with_text)
        if total == 0:
            project.status = ProjectStatus.READY_FOR_TTS
            _upsert_project(project)
            await update_progress(project_id, "ready_for_tts", 100, "No segments need polishing")
            return

        semaphore = asyncio.Semaphore(max(1, app_settings.concurrency.polish_concurrency))
        completed = 0

        async def polish_segment(idx: int, segment: Segment):
            nonlocal completed
            async with semaphore:
                prev_segment = project.segments[idx - 1] if idx > 0 else None
                next_segment = project.segments[idx + 1] if idx + 1 < len(project.segments) else None
                try:
                    segment.polished_text = await polisher.rewrite_segment(
                        text=segment.original_text,
                        target_duration=_segment_duration(segment),
                        prev_text=prev_segment.original_text if prev_segment else "",
                        next_text=next_segment.original_text if next_segment else "",
                        match_reason=segment.match_reason,
                        style_preset=style_preset or app_settings.ai.polish_style_preset,
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning(f"Polish failed for {segment.id}: {exc}")
                    segment.polished_text = segment.original_text
                completed += 1
                await update_progress(project_id, "polishing", int(100 * completed / total), f"Polished {completed}/{total}")

        await asyncio.gather(*(polish_segment(idx, segment) for idx, segment in enumerate(project.segments) if segment.original_text))
        project.status = ProjectStatus.READY_FOR_TTS
        _upsert_project(project)
        await update_progress(project_id, "ready_for_tts", 100, "Polishing complete, ready for TTS")
    except Exception as exc:  # pragma: no cover
        logger.exception(f"Polish task failed: {project_id}")
        project = load_project(project_id)
        if project:
            project.status = ProjectStatus.ERROR
            project.progress.message = str(exc)
            _upsert_project(project)
        await update_progress(project_id, "error", 0, f"Polishing failed: {exc}")
    finally:
        _processing_tasks.pop(project_id, None)
        await polisher.close()


@router.post("/{project_id}/segments/{segment_id}/repolish")
async def repolish_segment(project_id: str, segment_id: str, request: StartPolishRequest = StartPolishRequest()):
    """Re-polish a single segment with contextual prompts."""

    from api.routes.settings import load_settings
    from core.ai_service.api_manager import APIManager
    from core.ai_service.text_polisher import TextPolisher

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    idx = next((i for i, item in enumerate(project.segments) if item.id == segment_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Segment not found")
    segment = project.segments[idx]
    if not segment.original_text:
        raise HTTPException(status_code=400, detail="Segment has no original text")

    app_settings = load_settings()
    api_manager = APIManager(api_base=app_settings.ai.api_base, api_key=app_settings.ai.api_key, model=app_settings.ai.model)
    polisher = TextPolisher(
        api_manager=api_manager,
        template=app_settings.ai.polish_template,
        temperature=app_settings.ai.temperature,
        max_tokens=app_settings.ai.max_tokens,
        default_style_preset=request.style_preset or app_settings.ai.polish_style_preset,
        enable_de_ai_pass=app_settings.ai.enable_de_ai_pass,
        enable_self_review=app_settings.ai.enable_self_review,
    )

    prev_text = project.segments[idx - 1].original_text if idx > 0 else ""
    next_text = project.segments[idx + 1].original_text if idx + 1 < len(project.segments) else ""
    try:
        segment.polished_text = await polisher.rewrite_segment(
            text=segment.original_text,
            target_duration=_segment_duration(segment),
            prev_text=prev_text,
            next_text=next_text,
            match_reason=segment.match_reason,
            style_preset=request.style_preset or app_settings.ai.polish_style_preset,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Polishing failed: {exc}")
    finally:
        await polisher.close()

    _upsert_project(project)
    return {"polished_text": segment.polished_text}


@router.post("/{project_id}/start-polish")
async def start_polishing(project_id: str, request: StartPolishRequest = StartPolishRequest()):
    """Start the contextual polishing phase."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project_id in _processing_tasks and not _processing_tasks[project_id].done():
        raise HTTPException(status_code=400, detail="Project is already processing")
    if not project.segments:
        raise HTTPException(status_code=400, detail="Project has no segments")

    task = asyncio.create_task(_polish_project_task(project_id, request.style_preset))
    _processing_tasks[project_id] = task
    project.status = ProjectStatus.POLISHING
    _upsert_project(project)
    return {"message": "Polishing started", "project_id": project_id, "style_preset": request.style_preset}


@router.post("/{project_id}/generate-tts")
async def batch_generate_tts(project_id: str):
    """Start batch TTS generation."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project_id in _processing_tasks and not _processing_tasks[project_id].done():
        raise HTTPException(status_code=400, detail="Project is already processing")

    task = asyncio.create_task(_batch_generate_tts_task(project_id))
    _processing_tasks[project_id] = task
    project.status = ProjectStatus.GENERATING_TTS
    _upsert_project(project)
    return {"message": "TTS generation started", "project_id": project_id}


async def _batch_generate_tts_task(project_id: str):
    """Background batch TTS generation."""

    from api.routes.settings import load_settings
    from core.tts_service.tts_client import TTSClient

    project = load_project(project_id)
    if not project:
        return

    app_settings = load_settings()
    tts_client = TTSClient(
        api_base=app_settings.tts.api_base,
        api_endpoint=app_settings.tts.api_endpoint,
        reference_audio=app_settings.tts.reference_audio,
        infer_mode=app_settings.tts.infer_mode,
    )
    tts_output_dir = Path(__file__).resolve().parents[2] / "temp" / project_id / "tts"

    try:
        await update_progress(project_id, "generating_tts", 0, "Generating TTS audio...")
        segments_with_text = [segment for segment in project.segments if segment.original_text or segment.polished_text]
        total = len(segments_with_text)
        completed = 0
        semaphore = asyncio.Semaphore(max(1, app_settings.concurrency.tts_concurrency))

        async def generate_one(segment: Segment):
            nonlocal completed
            async with semaphore:
                text = segment.polished_text if segment.use_polished_text and segment.polished_text else segment.original_text
                if not text:
                    return
                try:
                    ref_audio = project.tts_reference_audio_path or project.reference_audio_path
                    old_path = tts_output_dir / f"{project_id}_{segment.id}.wav"
                    if old_path.exists():
                        old_path.unlink()
                    audio_path = await tts_client.generate(
                        text,
                        output_name=f"{project_id}_{segment.id}",
                        output_dir=tts_output_dir,
                        reference_audio=ref_audio,
                    )
                    segment.tts_audio_path = str(audio_path)
                    segment.tts_duration = await tts_client.get_duration(str(audio_path))
                    segment.tts_status = TTSStatus.GENERATED
                    segment.tts_error = None
                except Exception as exc:  # pragma: no cover
                    segment.tts_duration = await tts_client.estimate_duration(text)
                    segment.tts_status = TTSStatus.FAILED
                    segment.tts_error = str(exc)
                completed += 1
                await update_progress(project_id, "generating_tts", int(100 * completed / max(1, total)), f"Generated TTS {completed}/{total}")

        await asyncio.gather(*(generate_one(segment) for segment in segments_with_text))
        project = load_project(project_id)
        if project:
            project.status = ProjectStatus.COMPLETED
            _upsert_project(project)
        await update_progress(project_id, "completed", 100, "TTS generation complete")
    except Exception as exc:  # pragma: no cover
        logger.exception(f"Batch TTS failed: {project_id}")
        project = load_project(project_id)
        if project:
            project.status = ProjectStatus.ERROR
            project.progress.message = str(exc)
            _upsert_project(project)
        await update_progress(project_id, "error", 0, f"TTS generation failed: {exc}")
    finally:
        _processing_tasks.pop(project_id, None)
