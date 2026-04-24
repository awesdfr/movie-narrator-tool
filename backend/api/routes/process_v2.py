"""Processing API routes and orchestration."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field

from api.routes.project import load_project, recover_stale_project, save_project
from api.websocket import manager
from models.project import ProcessingProgress, ProjectStatus
from models.segment import (
    AlignmentStatus,
    MatchCandidate,
    Segment,
    SegmentBatchUpdate,
    SegmentType,
    SegmentUpdate,
)

router = APIRouter()
_processing_tasks: dict[str, asyncio.Task] = {}
_VISUAL_SPLIT_ID_RE = re.compile(r"^(?P<base>.+)_(?:v|c)(?P<part>\d{2,})$")


class ResegmentRequest(BaseModel):
    preserve_manual_matches: bool = Field(default=True)


class RematchRequest(BaseModel):
    candidate_top_k: Optional[int] = Field(default=None)


class RematchProjectRequest(BaseModel):
    preserve_manual_matches: bool = Field(default=True)


class RematchWeakSegmentsRequest(BaseModel):
    preserve_manual_matches: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    visual_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    include_inferred: bool = Field(default=True)
    include_review_required: bool = Field(default=True)
    max_segments: Optional[int] = Field(default=None, ge=1)
    deep_search_fallback: bool = Field(default=False)


def _upsert_project(project):
    save_project(project)


def _segment_duration(segment: Segment) -> float:
    return max(0.0, segment.narration_end - segment.narration_start)


def _is_visual_cut_piece(segment: Segment) -> bool:
    return bool(re.search(r"_c\d{2,}$", str(segment.id))) or "visual_cut parent=" in str(segment.evidence_summary or "")


def _detect_visual_boundary_breaks(project, segments: list[Segment]) -> set[str]:
    """Return segment ids whose start is a real visual cut in the narration.

    ASR/subtitle segment boundaries are not visual boundaries. If we split on
    them, similar shots from the same scene can independently jump to different
    movie phases. This detector only marks boundaries where the picture itself
    changes strongly; continuous boundaries are allowed to merge into one match
    chunk.
    """
    if not segments or not getattr(project, "narration_path", None):
        return set()

    def _detect() -> set[str]:
        import cv2

        capture = cv2.VideoCapture(str(project.narration_path))
        if not capture.isOpened():
            return set()

        cache: dict[float, np.ndarray] = {}

        def read_gray(timestamp: float) -> Optional[np.ndarray]:
            key = round(max(0.0, float(timestamp)), 3)
            cached = cache.get(key)
            if cached is not None:
                return cached
            capture.set(cv2.CAP_PROP_POS_MSEC, key * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                return None
            # Ignore the subtitle band and most sticker overlays near bottom.
            frame = frame[: max(1, int(height * 0.78)), :]
            frame = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cache[key] = gray
            return gray

        def cut_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
            left_hist = cv2.calcHist([left], [0], None, [32], [0, 256]).astype("float32")
            right_hist = cv2.calcHist([right], [0], None, [32], [0, 256]).astype("float32")
            left_hist = cv2.normalize(left_hist, left_hist).flatten()
            right_hist = cv2.normalize(right_hist, right_hist).flatten()
            hist_diff = float(cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CHISQR))
            mean_abs = float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))) / 255.0
            edge_diff = float(
                np.mean(
                    np.abs(
                        cv2.Canny(left, 60, 140).astype(np.float32)
                        - cv2.Canny(right, 60, 140).astype(np.float32)
                    )
                )
            ) / 255.0
            return hist_diff, mean_abs, edge_diff

        breaks: set[str] = set()
        try:
            previous: Optional[Segment] = None
            for segment in segments:
                if segment.skip_matching or segment.segment_type == SegmentType.NON_MOVIE:
                    previous = None
                    continue
                if previous is None:
                    previous = segment
                    continue
                gap = float(segment.narration_start) - float(previous.narration_end)
                if gap > 0.45:
                    breaks.add(str(segment.id))
                    previous = segment
                    continue
                boundary = max(float(segment.narration_start), float(previous.narration_end))
                left_time = max(
                    float(previous.narration_start) + 0.03,
                    min(float(previous.narration_end) - 0.08, boundary - 0.08),
                )
                right_time = min(
                    float(segment.narration_end) - 0.03,
                    max(float(segment.narration_start) + 0.08, boundary + 0.08),
                )
                left = read_gray(left_time)
                right = read_gray(right_time)
                if left is not None and right is not None:
                    hist_diff, mean_abs, edge_diff = cut_metrics(left, right)
                    hard_cut = (
                        (mean_abs >= 0.145 and edge_diff >= 0.095)
                        or mean_abs >= 0.220
                        or (edge_diff >= 0.280 and mean_abs >= 0.070)
                        or (edge_diff >= 0.200 and mean_abs >= 0.115)
                        or (edge_diff >= 0.250 and mean_abs >= 0.100)
                        or (hist_diff >= 18.0 and mean_abs >= 0.080)
                    )
                    if hard_cut:
                        breaks.add(str(segment.id))
                previous = segment
        finally:
            capture.release()
        return breaks

    try:
        return _detect()
    except Exception as exc:
        logger.debug("Visual boundary detection failed: {}", exc)
        return set()


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


def _monotonic_ratio(movie_starts: list[float], slack_seconds: float = 3.0) -> float:
    if len(movie_starts) < 2:
        return 0.0
    non_backtracking = sum(
        1 for prev, current in zip(movie_starts, movie_starts[1:]) if current + slack_seconds >= prev
    )
    return non_backtracking / max(1, len(movie_starts) - 1)


def _infer_sequence_mode_from_batch(active_segments: list[Segment], batch_map: dict[str, dict], allow_non_sequential: bool) -> tuple[bool, float]:
    if not allow_non_sequential:
        return True, 1.0

    anchors: list[tuple[float, float, float]] = []
    for segment in active_segments:
        batch_item = batch_map.get(segment.id)
        if not batch_item or not batch_item.get("success") or batch_item.get("start") is None:
            continue
        anchors.append(
            (
                float(batch_item["start"]),
                float(batch_item.get("confidence", 0.0)),
                float(batch_item.get("stability_score", 0.0)),
            )
        )

    if len(anchors) < 10:
        return False, _monotonic_ratio([item[0] for item in anchors])

    movie_starts = [item[0] for item in anchors]
    monotonicity = _monotonic_ratio(movie_starts)
    strong_anchor_ratio = sum(1 for _, confidence, stability in anchors if confidence >= 0.76 and stability >= 0.45) / len(anchors)
    prefer_sequential = monotonicity >= 0.74 and strong_anchor_ratio >= 0.45
    return prefer_sequential, monotonicity


def _infer_sequence_mode_from_segments(segments: list[Segment], allow_non_sequential: bool) -> tuple[bool, float]:
    if not allow_non_sequential:
        return True, 1.0

    movie_starts = [float(segment.movie_start) for segment in segments if segment.movie_start is not None]
    monotonicity = _monotonic_ratio(movie_starts)
    prefer_sequential = len(movie_starts) >= 10 and monotonicity >= 0.74
    return prefer_sequential, monotonicity


def _select_sparse_anchor_segments(active_segments: list[Segment]) -> list[Segment]:
    eligible = [
        segment
        for segment in active_segments
        if segment.segment_type != SegmentType.NON_MOVIE
        and not segment.skip_matching
        and _segment_duration(segment) >= 1.0
    ]
    if not eligible:
        return []

    target_count = min(36, max(12, len(eligible) // 16))
    step = max(1, len(eligible) // target_count)
    chosen: list[Segment] = []
    for idx, segment in enumerate(eligible):
        if idx == 0 or idx == len(eligible) - 1 or idx % step == 0:
            chosen.append(segment)

    deduped: list[Segment] = []
    seen: set[str] = set()
    for segment in chosen:
        if segment.id in seen:
            continue
        seen.add(segment.id)
        deduped.append(segment)
    return deduped


def _weighted_monotonic_anchor_subset(anchor_hits: list[dict]) -> tuple[list[dict], float]:
    if not anchor_hits:
        return [], 0.0

    ordered = sorted(anchor_hits, key=lambda item: float(item["narration_start"]))
    n = len(ordered)
    dp = [float(item["confidence"]) for item in ordered]
    prev = [-1] * n
    for i in range(n):
        current_movie = float(ordered[i]["movie_start"])
        for j in range(i):
            prev_movie = float(ordered[j]["movie_start"])
            if current_movie + 5.0 < prev_movie:
                continue
            candidate_score = dp[j] + float(ordered[i]["confidence"])
            if candidate_score > dp[i]:
                dp[i] = candidate_score
                prev[i] = j

    best_idx = max(range(n), key=lambda idx: dp[idx])
    chain: list[dict] = []
    while best_idx != -1:
        chain.append(ordered[best_idx])
        best_idx = prev[best_idx]
    chain.reverse()
    return chain, _monotonic_ratio([float(item["movie_start"]) for item in chain], slack_seconds=5.0)


def _estimate_movie_time_from_anchors(project, narration_time: float, anchors: list[dict]) -> Optional[float]:
    if not project.movie_duration or not project.narration_duration or project.narration_duration <= 0:
        return None

    linear = (narration_time / project.narration_duration) * project.movie_duration
    if not anchors:
        return linear
    
    # 确保至少有 2 个锚点用于插值
    if len(anchors) < 2:
        anchor = anchors[0]
        slope = project.movie_duration / max(project.narration_duration, 1.0)
        estimate = float(anchor["movie_start"]) + (narration_time - float(anchor["narration_start"])) * slope
        return max(0.0, min(project.movie_duration, estimate))
    
    if len(anchors) == 2:
        left, right = anchors[0], anchors[1]
    elif narration_time <= float(anchors[0]["narration_start"]):
        left, right = anchors[0], anchors[1]
    elif narration_time >= float(anchors[-1]["narration_start"]):
        left, right = anchors[-2], anchors[-1]
    else:
        left, right = anchors[0], anchors[-1]
        for idx in range(1, len(anchors)):
            if narration_time <= float(anchors[idx]["narration_start"]):
                left = anchors[idx - 1]
                right = anchors[idx]
                break

    left_narr = float(left["narration_start"])
    right_narr = float(right["narration_start"])
    left_movie = float(left["movie_start"])
    right_movie = float(right["movie_start"])
    narr_span = max(1.0, right_narr - left_narr)
    movie_span = right_movie - left_movie
    local_slope = movie_span / narr_span if abs(movie_span) > 1e-6 else (project.movie_duration / max(project.narration_duration, 1.0))
    estimate = left_movie + (narration_time - left_narr) * local_slope

    if narr_span > 180.0:
        estimate = estimate * 0.85 + linear * 0.15

    return max(0.0, min(project.movie_duration, estimate))


async def _build_anchor_time_map(
    frame_matcher,
    project,
    active_segments: list[Segment],
    narration_feature_map: dict[str, list[dict]],
    app_settings,
) -> tuple[dict[str, float], list[dict], float]:
    anchor_segments = _select_sparse_anchor_segments(active_segments)
    if not anchor_segments:
        return {}, [], 0.0

    semaphore = asyncio.Semaphore(min(4, max(1, app_settings.concurrency.match_concurrency)))
    anchor_hits: list[dict] = []

    async def match_anchor(segment: Segment) -> None:
        async with semaphore:
            result = await frame_matcher.match_segment(
                project.narration_path,
                segment.narration_start,
                segment.narration_end,
                time_hint=None,
                relaxed=True,
                strict_window=False,
                precomputed_features=narration_feature_map.get(segment.id),
            )
            if not result:
                return
            confidence = float(result.get("confidence", 0.0))
            stability = float(result.get("stability_score", 0.0))
            low_info = float(result.get("low_info_ratio", 1.0))
            if confidence < max(0.76, app_settings.match.medium_confidence_threshold) or stability < 0.40 or low_info > 0.65:
                return
            anchor_hits.append(
                {
                    "segment_id": segment.id,
                    "narration_start": float(segment.narration_start),
                    "movie_start": float(result["start"]),
                    "confidence": confidence,
                }
            )

    await asyncio.gather(*(match_anchor(segment) for segment in anchor_segments))
    anchors, monotonicity = _weighted_monotonic_anchor_subset(anchor_hits)
    if len(anchors) < 3:
        return {}, anchors, monotonicity

    expected_time_map = {
        segment.id: _estimate_movie_time_from_anchors(project, float(segment.narration_start), anchors)
        for segment in active_segments
    }
    return {key: value for key, value in expected_time_map.items() if value is not None}, anchors, monotonicity


def _combine_candidate_confidence(
    segment: Segment,
    visual_confidence: float,
    temporal_confidence: float,
    duration_gap: float,
    rank_gap: float = 0.0,
    audio_confidence: float = 0.0,
    stability_score: float = 0.0,
) -> float:
    narr_duration = max(_segment_duration(segment), 1.0)
    duration_confidence = max(0.0, 1.0 - duration_gap / narr_duration)
    # Visual evidence remains primary, but candidates that are both visually weak
    # and far off the expected timeline should no longer rank near the top.
    score = (
        visual_confidence * 0.84
        + duration_confidence * 0.04
        + stability_score * 0.08
        + temporal_confidence * 0.04
    )
    if temporal_confidence < 0.35 and visual_confidence < 0.90:
        score -= min(0.10, (0.35 - temporal_confidence) * 0.18)
    elif temporal_confidence > 0.82:
        score += min(0.04, (temporal_confidence - 0.82) * 0.20)
    if rank_gap < 0.015 and visual_confidence < 0.92:
        score -= min(0.10, (0.015 - rank_gap) * 4.0)
    elif rank_gap > 0.08:
        score += min(0.04, (rank_gap - 0.08) * 0.5)
    # 音频仅在有可靠音频时加成，无音频不惩罚
    if audio_confidence > 0.15:
        aw = _audio_weight_for_segment(segment)
        score = score * (1.0 - aw * 0.5) + audio_confidence * aw * 0.5
    score = float(np.clip(score, 0.0, 1.0))
    # 非线性校准：正确匹配（≥0.85）向1.0收敛
    # f(x) = 1-(1-x)^1.5: 0.85→0.942, 0.90→0.968, 0.93→0.983, 0.95→0.989
    if score >= 0.85:
        score = 1.0 - (1.0 - score) ** 1.5
    return min(1.0, max(0.0, score))


def _temporal_confidence_for_result(segment: Segment, result: dict, expected_movie_time: Optional[float]) -> float:
    if expected_movie_time is None:
        return 1.0
    center = (float(result["start"]) + float(result["end"])) / 2
    return max(
        0.0,
        1.0 - abs(center - expected_movie_time) / max(120.0, _segment_duration(segment) * 15.0, 1.0),
    )


def _can_use_batch_direct(
    segment: Segment,
    base_result: Optional[dict],
    expected_movie_time: Optional[float],
    prefer_sequential: bool,
    app_settings,
) -> bool:
    if not base_result or not base_result.get("success"):
        return False

    confidence = float(base_result.get("confidence", 0.0))
    stability = float(base_result.get("stability_score", 0.0))
    low_info = float(base_result.get("low_info_ratio", 1.0))
    temporal_confidence = _temporal_confidence_for_result(segment, base_result, expected_movie_time)

    if prefer_sequential:
        return (
            confidence >= max(0.80, app_settings.match.medium_confidence_threshold + 0.04)
            and stability >= 0.48
            and low_info <= 0.55
            and temporal_confidence >= 0.22
        )

    return (
        confidence >= max(0.88, app_settings.match.high_confidence_threshold)
        and stability >= 0.60
        and low_info <= 0.40
        and temporal_confidence >= 0.50
    )


def _inject_gap_segments(raw_segments: list[dict], total_duration: Optional[float], min_gap_duration: float) -> list[dict]:
    if total_duration is None:
        total_duration = raw_segments[-1]["end"] if raw_segments else 0.0
    duration_limit = max(0.0, float(total_duration or 0.0))
    ordered = sorted(raw_segments, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
    if not ordered:
        return [{"start": 0.0, "end": duration_limit, "text": "", "type": SegmentType.NO_NARRATION}] if duration_limit > min_gap_duration else []

    enriched: list[dict] = []
    cursor = 0.0
    for item in ordered:
        start = max(0.0, min(duration_limit, float(item.get("start", 0.0))))
        end = max(start, min(duration_limit, float(item.get("end", start))))
        if end <= start:
            continue
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
        normalized_item = dict(item)
        normalized_item["start"] = start
        normalized_item["end"] = end
        enriched.append(normalized_item)
        cursor = max(cursor, end)
    if duration_limit - cursor >= min_gap_duration:
        enriched.append(
            {
                "start": cursor,
                "end": duration_limit,
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
    rank_gap = float(result.get("rank_gap", 0.0))
    combined = _combine_candidate_confidence(
        segment,
        visual_confidence=visual_confidence,
        temporal_confidence=temporal_confidence,
        duration_gap=duration_gap,
        rank_gap=rank_gap,
        audio_confidence=audio_confidence,
        stability_score=stability_score,
    )
    confidence_level = result.get("confidence_level", "")
    reason = (
        f"{reason_prefix}; visual={visual_confidence:.2f}, temporal={temporal_confidence:.2f}, "
        f"stability={stability_score:.2f}, gap={rank_gap:.3f}, duration_gap={duration_gap:.2f}s"
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
        rank_gap=rank_gap,
        verification_score=0.0,
        geometric_inliers=0,
        geometric_inlier_ratio=0.0,
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

    candidates.sort(key=lambda item: item.score, reverse=True)
    best_score = candidates[0].score if candidates else 0.0
    if segment.segment_type == SegmentType.NO_NARRATION or segment.audio_activity_label in {"silent", "weak"}:
        rerank_limit = 4
    else:
        rerank_limit = 2
    rerank_targets = [
        candidate for candidate in candidates
        if candidate.score >= best_score - 0.06
    ][:rerank_limit]

    for candidate in rerank_targets:
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
            rank_gap=candidate.rank_gap,
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


async def _verify_ambiguous_candidates(frame_matcher, project, segment: Segment, candidates: list[MatchCandidate]) -> list[MatchCandidate]:
    if not candidates or len(candidates) < 2:
        return candidates

    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    top_gap = ordered[0].score - ordered[1].score
    needs_verification = (
        (top_gap < 0.009 and abs(ordered[0].start - ordered[1].start) > 24.0)
        or ordered[0].temporal_confidence < 0.05
        or (
            ordered[0].source == "batch_fast"
            and ordered[1].score >= ordered[0].score - 0.010
            and abs(ordered[0].start - ordered[1].start) > 32.0
        )
    )
    if not needs_verification:
        return ordered

    verify_targets = ordered[: min(2, len(ordered))]
    verification_results = await frame_matcher.verify_candidates(
        project.narration_path,
        segment.narration_start,
        segment.narration_end,
        [{"start": candidate.start, "end": candidate.end} for candidate in verify_targets],
    )
    if not verification_results:
        return ordered

    for candidate in verify_targets:
        metrics = verification_results.get((candidate.start, candidate.end))
        if not metrics:
            continue
        candidate.verification_score = float(metrics.get("verification_score", 0.0))
        candidate.geometric_inliers = int(metrics.get("geometric_inliers", 0))
        candidate.geometric_inlier_ratio = float(metrics.get("geometric_inlier_ratio", 0.0))
        score_delta = (candidate.verification_score - 0.50) * 0.28
        if candidate.geometric_inlier_ratio >= 0.22:
            score_delta += 0.04
        elif candidate.geometric_inlier_ratio <= 0.05:
            score_delta -= 0.04
        candidate.score = min(1.0, max(0.0, candidate.score + score_delta))
        candidate.confidence = candidate.score
        candidate.reason += (
            f"; verify={candidate.verification_score:.2f}, "
            f"inliers={candidate.geometric_inliers}, "
            f"inlier_ratio={candidate.geometric_inlier_ratio:.2f}"
        )

    ordered.sort(key=lambda item: item.score, reverse=True)
    for idx, candidate in enumerate(ordered, start=1):
        candidate.rank = idx
        candidate.id = f"{candidate.id.split('_cand_')[0]}_cand_{idx}"
    return ordered


async def _phase_lock_candidates(
    frame_matcher,
    project,
    segment: Segment,
    candidates: list[MatchCandidate],
    *,
    max_candidates: int = 5,
    precomputed_features: Optional[list[dict]] = None,
    verify_frames: bool = True,
) -> list[MatchCandidate]:
    """Strictly rerank candidates by exact visual phase after masking overlays.

    Retrieval can find the right scene while still being 0.5-1.0s off in actor
    motion. This pass tests nearby starts and conservative speed ratios, then
    rewards only candidates that match the narration frame sequence itself.
    """
    if not candidates or not hasattr(frame_matcher, "verify_segment_matches"):
        return candidates

    duration = _segment_duration(segment)
    if duration <= 0.12:
        return candidates

    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    targets = ordered[: max(1, min(int(max_candidates), len(ordered)))]
    movie_duration = float(getattr(frame_matcher, "_movie_duration", 0.0) or getattr(project, "movie_duration", 0.0) or 0.0)

    if duration < 0.70:
        offsets = (0.0, -0.22, 0.22)
        variants_per_candidate = 1
    elif duration < 1.60:
        offsets = (0.0, -0.42, 0.42)
        variants_per_candidate = 2
    else:
        offsets = (0.0, -0.36, 0.36, -0.72, 0.72)
        variants_per_candidate = 2

    scorer = getattr(frame_matcher, "score_precomputed_segment_at", None)
    can_fast_prefilter = scorer is not None and bool(precomputed_features)

    def _fast_variant_score(source_start: float, scale: float) -> float:
        if not can_fast_prefilter:
            return 0.0
        try:
            payload = scorer(
                precomputed_features,
                source_start,
                search_radius=0.55 if duration < 1.60 else 0.75,
                time_scales=(float(scale),),
            )
        except Exception:
            return 0.0
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        stability = float(payload.get("stability_score", 0.0) or 0.0)
        low_info = float(payload.get("low_info_ratio", 1.0) or 1.0)
        return max(0.0, min(1.0, confidence * 0.82 + stability * 0.12 + max(0.0, 1.0 - low_info) * 0.06))

    payload: list[dict] = []
    variant_map: dict[str, tuple[MatchCandidate, float, float, float, float]] = {}
    fast_metrics_by_id: dict[str, dict] = {}

    for candidate in targets:
        base_start = max(0.0, float(candidate.start))
        base_duration = max(0.05, float(candidate.end) - base_start)
        base_ratio = base_duration / max(duration, 0.05)
        scale_options = [1.0]
        if 0.72 <= base_ratio <= 1.36 and abs(base_ratio - 1.0) >= 0.045:
            scale_options.append(base_ratio)
        if duration >= 0.85:
            scale_options.extend([0.92, 1.08])
        if duration >= 1.80:
            scale_options.extend([0.86, 1.16])

        unique_scales: list[float] = []
        for scale in scale_options:
            scale = max(0.70, min(1.40, float(scale)))
            if not any(abs(existing - scale) <= 0.018 for existing in unique_scales):
                unique_scales.append(scale)

        scored_variants: list[tuple[float, float, float, float, float]] = []
        for scale in unique_scales:
            source_duration = max(0.05, duration * scale)
            for offset in offsets:
                source_start = max(0.0, base_start + float(offset))
                source_end = source_start + source_duration
                if movie_duration > 0.0 and source_end > movie_duration:
                    continue
                fast_score = _fast_variant_score(source_start, scale)
                if not can_fast_prefilter:
                    fast_score = 1.0 if abs(offset) <= 0.001 and abs(scale - 1.0) <= 0.018 else 0.5
                scored_variants.append((fast_score, source_start, source_end, float(offset), float(scale)))

        if not scored_variants:
            continue
        scored_variants.sort(key=lambda item: item[0], reverse=True)
        kept_variants = scored_variants[:variants_per_candidate]
        base_variant = next(
            (
                item
                for item in scored_variants
                if abs(item[3]) <= 0.001 and abs(item[4] - 1.0) <= 0.018
            ),
            None,
        )
        if base_variant is not None and all(abs(item[1] - base_variant[1]) > 0.04 for item in kept_variants):
            kept_variants.append(base_variant)

        for _, source_start, source_end, offset, scale in kept_variants:
            variant_id = f"{candidate.id}|phase|{len(payload)}"
            payload.append(
                {
                    "id": variant_id,
                    "narration_start": segment.narration_start,
                    "narration_end": segment.narration_end,
                    "movie_start": source_start,
                    "movie_end": source_end,
                }
            )
            variant_map[variant_id] = (candidate, source_start, source_end, float(offset), float(scale))
            fast_metrics_by_id[variant_id] = {
                "verification_score": float(_fast_variant_score(source_start, scale)),
                "geometric_inliers": 0,
                "geometric_inlier_ratio": 0.0,
                "sample_count": len(precomputed_features or []),
            }

    if not payload:
        return ordered

    if verify_frames:
        try:
            metrics_by_id = await frame_matcher.verify_segment_matches(project.narration_path, payload)
        except Exception as exc:  # pragma: no cover
            logger.debug("Phase-lock verification failed for segment {}: {}", segment.id, exc)
            return ordered
    else:
        metrics_by_id = fast_metrics_by_id
    if not metrics_by_id:
        return ordered

    best_by_candidate: dict[str, tuple[dict, float, float, float, float]] = {}
    base_score_by_candidate: dict[str, float] = {}
    for variant_id, metrics in metrics_by_id.items():
        mapped = variant_map.get(variant_id)
        if mapped is None:
            continue
        candidate, source_start, source_end, offset, scale = mapped
        score = float(metrics.get("verification_score", 0.0) or 0.0)
        if abs(offset) <= 0.001 and abs(scale - 1.0) <= 0.018:
            base_score_by_candidate[candidate.id] = score
        current = best_by_candidate.get(candidate.id)
        if current is None or score > float(current[0].get("verification_score", 0.0) or 0.0):
            best_by_candidate[candidate.id] = (metrics, source_start, source_end, offset, scale)

    for candidate in targets:
        best = best_by_candidate.get(candidate.id)
        if best is None:
            continue

        metrics, source_start, source_end, offset, scale = best
        score = float(metrics.get("verification_score", 0.0) or 0.0)
        base_score = float(base_score_by_candidate.get(candidate.id, score))

        # Do not retime unless the shifted/scaled sequence clearly beats the
        # original timing. This prevents "fixes" that jump to another similar shot.
        if abs(scale - 1.0) >= 0.045 and score < base_score + 0.045:
            scale = 1.0
            source_end = source_start + duration
            score = base_score
        if abs(offset) >= 0.08 and score < base_score + 0.030:
            source_start = max(0.0, float(candidate.start))
            source_end = source_start + max(0.05, duration * scale)
            offset = 0.0
            score = max(score, base_score)

        candidate.start = source_start
        candidate.end = source_end
        candidate.verification_score = score
        candidate.geometric_inliers = int(metrics.get("geometric_inliers", 0) or 0)
        candidate.geometric_inlier_ratio = float(metrics.get("geometric_inlier_ratio", 0.0) or 0.0)

        sample_count = int(metrics.get("sample_count", 0) or 0)
        inlier_ratio = float(candidate.geometric_inlier_ratio or 0.0)
        score_delta = (score - 0.62) * 0.56
        if score >= 0.86 and inlier_ratio >= 0.12:
            score_delta += 0.08
        elif score >= 0.78:
            score_delta += 0.035
        elif score < 0.50:
            score_delta -= 0.12
        if duration >= 1.40 and sample_count < 2:
            score_delta -= 0.05

        candidate.score = min(1.0, max(0.0, float(candidate.score or 0.0) + score_delta))
        candidate.confidence = candidate.score
        if "phase_lock=" not in candidate.reason:
            candidate.reason += (
                f"; phase_lock={score:.2f}, offset={offset:+.2f}s, "
                f"scale={scale:.3f}, samples={sample_count}"
            )

    ordered.sort(key=lambda item: item.score, reverse=True)
    for idx, candidate in enumerate(ordered, start=1):
        candidate.rank = idx
        candidate.id = f"{candidate.id.split('_cand_')[0]}_cand_{idx}"
    return ordered


def _apply_sequence_bias_to_candidates(candidates: list[MatchCandidate], prefer_sequential: bool) -> list[MatchCandidate]:
    if not candidates:
        return candidates

    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    if not prefer_sequential:
        return ordered

    for candidate in ordered:
        temporal = float(candidate.temporal_confidence or 0.0)
        visual = float(candidate.visual_confidence or 0.0)
        verification = float(candidate.verification_score or 0.0)
        rank_gap = float(candidate.rank_gap or 0.0)
        score_delta = 0.0

        if temporal >= 0.65:
            score_delta += 0.04
        elif temporal >= 0.35:
            score_delta += 0.02

        if candidate.source == "batch_fast" and temporal >= 0.22:
            score_delta += 0.03

        if temporal < 0.08 and verification < 0.42:
            score_delta -= 0.14 if visual < 0.93 else 0.08
        elif temporal < 0.16 and rank_gap < 0.01 and verification < 0.40:
            score_delta -= 0.05

        if rank_gap >= 0.04 and temporal >= 0.25:
            score_delta += 0.02

        if score_delta != 0.0:
            candidate.score = min(1.0, max(0.0, candidate.score + score_delta))
            candidate.confidence = candidate.score
            candidate.reason += f"; timeline_bias={score_delta:+.2f}"

    ordered.sort(key=lambda item: item.score, reverse=True)
    top_candidate = ordered[0]
    if len(ordered) > 1 and top_candidate.temporal_confidence < 0.08 and top_candidate.verification_score < 0.42:
        better_timed = next(
            (
                candidate
                for candidate in ordered[1:]
                if candidate.temporal_confidence >= 0.25 and candidate.score >= top_candidate.score - 0.04
            ),
            None,
        )
        if better_timed is not None:
            better_timed.score = min(1.0, better_timed.score + 0.05)
            better_timed.confidence = better_timed.score
            better_timed.reason += "; promoted_for_timeline"
            ordered.sort(key=lambda item: item.score, reverse=True)

    for idx, candidate in enumerate(ordered, start=1):
        candidate.rank = idx
        candidate.id = f"{candidate.id.split('_cand_')[0]}_cand_{idx}"
    return ordered


def _needs_second_pass_rerank(segment: Segment, high_confidence_threshold: float) -> bool:
    if segment.segment_type == SegmentType.NON_MOVIE or segment.skip_matching:
        return False
    if segment.movie_start is None or not segment.match_candidates:
        return True

    ordered = sorted(segment.match_candidates, key=lambda item: item.score, reverse=True)
    top_candidate = ordered[0]
    second_candidate = ordered[1] if len(ordered) > 1 else None

    if len(ordered) == 1:
        return top_candidate.confidence < max(0.76, high_confidence_threshold - 0.08) and top_candidate.temporal_confidence < 0.24

    if top_candidate.confidence >= max(0.82, high_confidence_threshold - 0.04):
        if top_candidate.verification_score >= 0.85:
            return False

    if top_candidate.confidence < max(0.78, high_confidence_threshold - 0.04):
        return True
    if top_candidate.temporal_confidence < 0.10 and top_candidate.verification_score < 0.42:
        return True
    if second_candidate and top_candidate.score - second_candidate.score < 0.012:
        return True
    if (
        second_candidate
        and abs(second_candidate.start - top_candidate.start) > 45.0
        and top_candidate.score - second_candidate.score < 0.025
        and top_candidate.temporal_confidence < 0.25
    ):
        return True
    return False


def _segment_rerank_priority(segment: Segment) -> float:
    priority = 0.0
    if segment.movie_start is None or segment.movie_end is None:
        priority += 2.4
    if segment.review_required:
        priority += 1.4
    if segment.match_type == "fallback":
        priority += 1.0

    ordered = sorted(segment.match_candidates, key=lambda item: item.score, reverse=True)
    top_candidate = ordered[0] if ordered else None
    second_candidate = ordered[1] if len(ordered) > 1 else None
    if top_candidate is None:
        priority += 1.8
    else:
        priority += max(0.0, 0.84 - float(top_candidate.confidence or 0.0)) * 3.0
        priority += max(0.0, 0.18 - float(top_candidate.temporal_confidence or 0.0)) * 2.0
        priority += max(0.0, 0.36 - float(top_candidate.verification_score or 0.0)) * 0.8
        if second_candidate is not None:
            priority += max(0.0, 0.022 - (float(top_candidate.score) - float(second_candidate.score))) * 18.0

    if _segment_duration(segment) <= 2.4:
        priority += 0.2
    return priority


def _cap_rerank_segments(
    segments: list[Segment],
    active_count: int,
    phase: str,
    prefer_sequential: bool,
) -> list[Segment]:
    if len(segments) <= 1:
        return segments

    ordered = sorted(segments, key=_segment_rerank_priority, reverse=True)
    if active_count >= 500:
        if phase == "low_confidence":
            ratio = 0.07 if prefer_sequential else 0.10
            limit = min(max(24, int(active_count * ratio)), 64)
        else:
            ratio = 0.09 if prefer_sequential else 0.12
            limit = min(max(36, int(active_count * ratio)), 80)
        return ordered[: min(len(ordered), limit)]

    if phase == "low_confidence":
        ratio = 0.12 if prefer_sequential else 0.18
        limit = min(max(32, int(active_count * ratio)), 96)
    else:
        ratio = 0.16 if prefer_sequential else 0.24
        limit = min(max(64, int(active_count * ratio)), 128)
    return ordered[: min(len(ordered), limit)]


def _compute_neighbor_hint_for_segment(segments: list[Segment], idx: int, segment: Segment) -> Optional[float]:
    prev_segment = next((item for item in reversed(segments[:idx]) if item.movie_end is not None), None)
    next_segment = next((item for item in segments[idx + 1 :] if item.movie_start is not None), None)
    if prev_segment and next_segment and next_segment.narration_start > prev_segment.narration_start:
        ratio = (segment.narration_start - prev_segment.narration_start) / (
            next_segment.narration_start - prev_segment.narration_start
        )
        return prev_segment.movie_start + ratio * (next_segment.movie_start - prev_segment.movie_start)
    if prev_segment:
        return prev_segment.movie_end
    if next_segment:
        return max(0.0, next_segment.movie_start - _segment_duration(segment))
    return None


def _collect_recheck_hints(segment: Segment, neighbor_hint: Optional[float]) -> list[float]:
    hints: list[float] = []

    def _add(value: Optional[float]) -> None:
        if value is None:
            return
        value = float(max(0.0, value))
        if not any(abs(existing - value) <= 3.0 for existing in hints):
            hints.append(value)

    duration = max(1.0, _segment_duration(segment))
    _add(neighbor_hint)
    _add(segment.movie_start)
    if segment.movie_start is not None and segment.movie_end is not None:
        _add((float(segment.movie_start) + float(segment.movie_end)) / 2.0)

    ordered_candidates = sorted(segment.match_candidates, key=lambda item: item.score, reverse=True)
    for candidate in ordered_candidates[:4]:
        _add(candidate.start)
        _add((float(candidate.start) + float(candidate.end)) / 2.0)

    seeds = list(hints)
    for seed in seeds[:6]:
        for offset in (-45.0, -15.0, -duration, duration, 15.0, 45.0):
            _add(seed + offset)
    return hints[:12]


def _build_context_window(segments: list[Segment], idx: int, min_duration: float = 5.5, max_neighbors: int = 2) -> Optional[tuple[float, float]]:
    center = segments[idx]
    context_start = float(center.narration_start)
    context_end = float(center.narration_end)
    left = idx - 1
    right = idx + 1
    used_neighbors = 0

    def _eligible(seg: Segment) -> bool:
        return seg.segment_type != SegmentType.NON_MOVIE and not seg.skip_matching

    while context_end - context_start < min_duration and used_neighbors < max_neighbors * 2:
        expanded = False
        if left >= 0:
            seg = segments[left]
            left -= 1
            if _eligible(seg):
                context_start = float(seg.narration_start)
                used_neighbors += 1
                expanded = True
        if context_end - context_start >= min_duration or used_neighbors >= max_neighbors * 2:
            break
        if right < len(segments):
            seg = segments[right]
            right += 1
            if _eligible(seg):
                context_end = float(seg.narration_end)
                used_neighbors += 1
                expanded = True
        if not expanded:
            break

    if context_end - context_start <= _segment_duration(center) + 0.8:
        return None
    return context_start, context_end


def _context_params_for_segment(segment: Segment, base_result: Optional[dict]) -> tuple[float, int]:
    duration = _segment_duration(segment)
    batch_confidence = float(base_result.get("confidence", 0.0)) if base_result and base_result.get("success") else 0.0
    if duration <= 1.4:
        return 7.2, 3
    if duration <= 2.2:
        return 6.4, 3
    if duration <= 3.0 and batch_confidence < 0.80:
        return 5.8, 2
    return 5.5, 2


def _should_use_context_first(segment: Segment, base_result: Optional[dict], app_settings) -> bool:
    duration = _segment_duration(segment)
    if duration <= 2.2:
        return True
    batch_confidence = float(base_result.get("confidence", 0.0)) if base_result and base_result.get("success") else 0.0
    if duration <= 3.0 and batch_confidence < max(0.80, app_settings.match.high_confidence_threshold - 0.03):
        return True
    if duration <= 3.5 and segment.audio_activity_label in {"silent", "weak"} and batch_confidence < 0.86:
        return True
    return False


def _context_candidates_sufficient(candidates: list[MatchCandidate], high_confidence_threshold: float) -> bool:
    if not candidates:
        return False
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    rank_gap = top.score - second.score if second is not None else 0.03
    return (
        top.confidence >= max(0.80, high_confidence_threshold - 0.04)
        and (
            rank_gap >= 0.012
            or top.temporal_confidence >= 0.22
            or top.verification_score >= 0.38
        )
    )


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
    segment.match_type = "fallback" if status in {AlignmentStatus.UNMATCHED, AlignmentStatus.SKIPPED} else "inferred"
    segment.evidence_summary = reason
    segment.speed_changed = False
    segment.source_speed_ratio = 1.0
    segment.speed_change_confidence = 0.0


def _fill_non_narration_segments(project, segments: list[Segment]) -> None:
    movie_duration = float(project.movie_duration or 0.0)
    for idx, segment in enumerate(segments):
        if segment.segment_type != SegmentType.NO_NARRATION or segment.skip_matching:
            continue
        segment.movie_start = None
        segment.movie_end = None

        prev_segment = next(
            (
                item
                for item in reversed(segments[:idx])
                if item.movie_start is not None and item.movie_end is not None and item.segment_type != SegmentType.NON_MOVIE
            ),
            None,
        )
        next_segment = next(
            (
                item
                for item in segments[idx + 1 :]
                if item.movie_start is not None and item.movie_end is not None and item.segment_type != SegmentType.NON_MOVIE
            ),
            None,
        )

        duration = _segment_duration(segment)
        inferred_start: Optional[float] = None
        inferred_end: Optional[float] = None

        if prev_segment and next_segment:
            prev_narr = float(prev_segment.narration_end)
            next_narr = float(next_segment.narration_start)
            prev_movie = float(prev_segment.movie_end)
            next_movie = float(next_segment.movie_start)
            narr_span = max(0.2, next_narr - prev_narr)
            ratio = (float(segment.narration_start) - prev_narr) / narr_span
            ratio = max(0.0, min(1.0, ratio))
            bridge_span = max(0.0, next_movie - prev_movie - duration)
            inferred_start = prev_movie + ratio * bridge_span
            inferred_end = inferred_start + duration
        elif prev_segment:
            inferred_start = float(prev_segment.movie_end)
            inferred_end = inferred_start + duration
        elif next_segment:
            inferred_end = float(next_segment.movie_start)
            inferred_start = inferred_end - duration

        if inferred_start is None or inferred_end is None:
            continue

        inferred_start = max(0.0, inferred_start)
        inferred_end = max(inferred_start, inferred_end)
        if movie_duration > 0.0 and inferred_end > movie_duration:
            inferred_end = movie_duration
            inferred_start = max(0.0, inferred_end - duration)

        segment.movie_start = inferred_start
        segment.movie_end = inferred_end
        segment.match_confidence = 0.74
        segment.visual_confidence = 0.0
        segment.audio_confidence = 0.0
        segment.temporal_confidence = 0.82 if prev_segment or next_segment else 0.55
        segment.stability_score = 0.55
        segment.duration_gap = 0.0
        segment.match_reason = "Filled from neighboring matched segments for continuity"
        segment.alignment_status = AlignmentStatus.AUTO_ACCEPTED
        segment.review_required = False
        segment.selected_candidate_id = None
        segment.estimated_boundary_error = max(0.25, duration / 2.0)
        segment.match_type = "inferred"
        segment.evidence_summary = "continuity_fill"


def _enforce_segment_source_continuity(project, segments: list[Segment]) -> int:
    movie_duration = float(project.movie_duration or 0.0)
    last_source_end: Optional[float] = None
    repaired = 0

    for segment in sorted(segments, key=lambda item: float(item.narration_start)):
        if segment.skip_matching or segment.segment_type == SegmentType.NON_MOVIE:
            continue
        if segment.movie_start is None or segment.movie_end is None:
            continue

        duration = _segment_duration(segment)
        if duration <= 0.0:
            continue

        source_start = float(segment.movie_start)
        source_end = float(segment.movie_end)
        if last_source_end is not None and source_start < last_source_end - 0.05:
            source_start = last_source_end
            source_end = source_start + duration
            if movie_duration > 0.0 and source_end > movie_duration:
                source_end = movie_duration
                source_start = max(last_source_end, source_end - duration)

            segment.movie_start = source_start
            segment.movie_end = max(source_start, source_end)
            segment.match_confidence = min(float(segment.match_confidence or 0.0), 0.70)
            segment.visual_confidence = min(float(segment.visual_confidence or 0.0), 0.70)
            segment.match_type = "inferred"
            segment.evidence_summary = f"{segment.evidence_summary or ''}; source_continuity_repair".strip("; ")
            repaired += 1
        elif last_source_end is not None and source_start < last_source_end:
            source_start = last_source_end
            source_end = source_start + duration
            segment.movie_start = source_start
            segment.movie_end = source_end

        last_source_end = float(segment.movie_end)

    return repaired


def _is_strong_alignment_anchor(segment: Segment) -> bool:
    if segment.skip_matching or segment.segment_type == SegmentType.NON_MOVIE:
        return False
    if segment.movie_start is None or segment.movie_end is None:
        return False
    if segment.review_required:
        return False
    if segment.alignment_status not in {
        AlignmentStatus.AUTO_ACCEPTED,
        AlignmentStatus.MANUAL,
        AlignmentStatus.REMATCHED,
    }:
        return False
    if segment.match_type != "exact":
        return False
    return (
        float(segment.match_confidence or 0.0) >= 0.82
        and float(segment.stability_score or 0.0) >= 0.45
    )


def _fill_short_unmatched_segments(project, segments: list[Segment]) -> None:
    movie_duration = float(project.movie_duration or 0.0)
    for idx, segment in enumerate(segments):
        if segment.skip_matching or segment.segment_type == SegmentType.NON_MOVIE:
            continue
        if segment.movie_start is not None and segment.movie_end is not None:
            continue

        duration = _segment_duration(segment)
        prev_anchor = next(
            (item for item in reversed(segments[:idx]) if _is_strong_alignment_anchor(item)),
            None,
        )
        next_anchor = next(
            (item for item in segments[idx + 1 :] if _is_strong_alignment_anchor(item)),
            None,
        )

        inferred_start: Optional[float] = None
        inferred_end: Optional[float] = None
        review_required = True
        evidence = "continuity_fill_edge"

        if prev_anchor and next_anchor:
            prev_narr = float(prev_anchor.narration_end)
            next_narr = float(next_anchor.narration_start)
            prev_movie = float(prev_anchor.movie_end)
            next_movie = float(next_anchor.movie_start)
            narr_span = max(0.2, next_narr - prev_narr)
            movie_span = max(0.0, next_movie - prev_movie)
            if movie_span <= max(90.0, narr_span * 5.0):
                ratio = (float(segment.narration_start) - prev_narr) / narr_span
                ratio = max(0.0, min(1.0, ratio))
                bridge_span = max(0.0, next_movie - prev_movie - duration)
                inferred_start = prev_movie + ratio * bridge_span
                inferred_end = inferred_start + duration
                review_required = duration > 2.6 or movie_span > max(40.0, narr_span * 3.0)
                evidence = "continuity_fill_strong"
        elif segment.segment_type == SegmentType.NO_NARRATION or duration <= 1.6:
            if prev_anchor:
                inferred_start = float(prev_anchor.movie_end)
                inferred_end = inferred_start + duration
                review_required = segment.segment_type != SegmentType.NO_NARRATION
            elif next_anchor:
                inferred_end = float(next_anchor.movie_start)
                inferred_start = inferred_end - duration
                review_required = segment.segment_type != SegmentType.NO_NARRATION

        if inferred_start is None or inferred_end is None:
            continue

        inferred_start = max(0.0, inferred_start)
        inferred_end = max(inferred_start, inferred_end)
        if movie_duration > 0.0 and inferred_end > movie_duration:
            inferred_end = movie_duration
            inferred_start = max(0.0, inferred_end - duration)

        segment.movie_start = inferred_start
        segment.movie_end = inferred_end
        segment.match_confidence = 0.70 if evidence == "continuity_fill_strong" else 0.62
        segment.visual_confidence = 0.0
        segment.audio_confidence = 0.0
        segment.temporal_confidence = 0.88 if evidence == "continuity_fill_strong" else 0.72
        segment.stability_score = 0.60 if evidence == "continuity_fill_strong" else 0.48
        segment.duration_gap = 0.0
        segment.match_reason = "Filled from neighboring strong anchors for continuity"
        segment.alignment_status = (
            AlignmentStatus.NEEDS_REVIEW if review_required else AlignmentStatus.AUTO_ACCEPTED
        )
        segment.review_required = review_required
        segment.selected_candidate_id = None
        segment.estimated_boundary_error = max(0.35, duration / 2.0)
        segment.match_type = "inferred"
        segment.evidence_summary = evidence


def _mark_segment_skipped(segment: Segment, reason: str = "Skipped from matching by user") -> None:
    _clear_match(segment, AlignmentStatus.SKIPPED, reason)
    segment.match_candidates = []
    segment.skip_matching = True
    segment.is_manual_match = False


def _split_segments_for_visual_matching(
    segments: list[Segment],
    preserve_manual_matches: bool,
    max_duration: float = 8.0,
) -> tuple[list[Segment], int]:
    """Split only truly long narration spans.

    Earlier versions split every narration segment into ~0.7s micro-segments.
    That improves coverage numbers, but it makes the exported movie timeline
    jump between similar frames. Keep chunks long enough to carry motion.
    """
    result: list[Segment] = []
    split_count = 0
    next_index = 1

    for segment in segments:
        duration = _segment_duration(segment)
        should_keep = (
            duration <= max_duration + 0.08
            or segment.skip_matching
            or segment.segment_type == SegmentType.NON_MOVIE
            or (preserve_manual_matches and segment.is_manual_match)
        )
        if should_keep:
            kept = segment.model_copy(deep=True)
            kept.index = next_index
            result.append(kept)
            next_index += 1
            continue

        part_count = max(2, int(np.ceil(duration / max_duration)))
        part_duration = duration / part_count
        parent_id = segment.id
        for part_index in range(part_count):
            part_start = float(segment.narration_start) + part_duration * part_index
            part_end = float(segment.narration_start) + part_duration * (part_index + 1)
            if part_index == part_count - 1:
                part_end = float(segment.narration_end)

            clone = segment.model_copy(deep=True)
            clone.id = f"{parent_id}_v{part_index + 1:02d}"
            clone.index = next_index
            clone.narration_start = part_start
            clone.narration_end = part_end
            clone.movie_start = None
            clone.movie_end = None
            clone.match_confidence = 0.0
            clone.visual_confidence = 0.0
            clone.audio_confidence = 0.0
            clone.temporal_confidence = 0.0
            clone.stability_score = 0.0
            clone.duration_gap = 0.0
            clone.match_reason = ""
            clone.match_type = "exact"
            clone.evidence_summary = f"visual_split parent={parent_id}"
            clone.alignment_status = AlignmentStatus.PENDING
            clone.review_required = False
            clone.is_manual_match = False
            clone.selected_candidate_id = None
            clone.match_candidates = []
            clone.estimated_boundary_error = None
            if part_index > 0:
                clone.original_text = ""
                clone.polished_text = ""
            result.append(clone)
            next_index += 1
        split_count += part_count - 1

    return result, split_count


def _reset_segment_match_state(segment: Segment) -> None:
    segment.movie_start = None
    segment.movie_end = None
    segment.match_confidence = 0.0
    segment.visual_confidence = 0.0
    segment.audio_confidence = 0.0
    segment.temporal_confidence = 0.0
    segment.stability_score = 0.0
    segment.duration_gap = 0.0
    segment.match_reason = ""
    segment.match_type = "exact"
    segment.evidence_summary = ""
    segment.speed_changed = False
    segment.source_speed_ratio = 1.0
    segment.speed_change_confidence = 0.0
    segment.alignment_status = AlignmentStatus.PENDING
    segment.review_required = False
    segment.is_manual_match = False
    segment.selected_candidate_id = None
    segment.match_candidates = []
    segment.estimated_boundary_error = None


def _coalesce_visual_micro_segments(
    segments: list[Segment],
    preserve_manual_matches: bool,
) -> tuple[list[Segment], int]:
    """Merge legacy *_vNN visual micro-segments back into their parent segment."""
    result: list[Segment] = []
    merged_count = 0
    index = 0
    next_index = 1

    while index < len(segments):
        segment = segments[index]
        match = _VISUAL_SPLIT_ID_RE.match(str(segment.id))
        if not match:
            kept = segment.model_copy(deep=True)
            kept.index = next_index
            result.append(kept)
            next_index += 1
            index += 1
            continue

        base_id = match.group("base")
        group: list[Segment] = []
        cursor = index
        while cursor < len(segments):
            candidate = segments[cursor]
            candidate_match = _VISUAL_SPLIT_ID_RE.match(str(candidate.id))
            if not candidate_match or candidate_match.group("base") != base_id:
                break
            if group and float(candidate.narration_start) - float(group[-1].narration_end) > 0.20:
                break
            group.append(candidate)
            cursor += 1

        if len(group) <= 1 or (preserve_manual_matches and any(item.is_manual_match for item in group)):
            kept = segment.model_copy(deep=True)
            kept.index = next_index
            result.append(kept)
            next_index += 1
            index += 1
            continue

        merged = group[0].model_copy(deep=True)
        merged.id = base_id
        merged.index = next_index
        merged.narration_start = float(group[0].narration_start)
        merged.narration_end = float(group[-1].narration_end)
        merged.original_text = next((item.original_text for item in group if item.original_text), merged.original_text)
        merged.polished_text = next((item.polished_text for item in group if item.polished_text), merged.polished_text)
        merged.thumbnail_path = next((item.thumbnail_path for item in group if item.thumbnail_path), merged.thumbnail_path)
        _reset_segment_match_state(merged)
        merged.evidence_summary = f"coalesced_visual_micro_segments parts={len(group)}"
        result.append(merged)
        next_index += 1
        merged_count += len(group) - 1
        index = cursor

    return result, merged_count


async def _split_segments_on_visual_cuts(
    project,
    segments: list[Segment],
    preserve_manual_matches: bool,
    min_part_duration: float = 0.55,
    sample_step: float = 0.25,
) -> tuple[list[Segment], int]:
    """Split narration spans only where the narration video visibly cuts.

    This is the middle ground between two bad extremes:
    fixed 0.7s micro-splitting causes jitter, while whole ASR segments can hide
    several rapid movie shots. We cut only on strong frame-discontinuities.
    """
    if not segments or not getattr(project, "narration_path", None):
        return segments, 0

    def _detect_and_split() -> tuple[list[Segment], int]:
        import cv2
        import subprocess

        try:
            from core.video_processor.analysis_video import _resolve_ffmpeg_path
        except Exception:
            _resolve_ffmpeg_path = lambda: "ffmpeg"

        frame_width = 192
        frame_height = 108
        frame_bytes = frame_width * frame_height
        ffmpeg_bin = _resolve_ffmpeg_path()
        cmd = [
            ffmpeg_bin,
            "-i",
            str(project.narration_path),
            "-an",
            "-vf",
            f"fps={1.0 / sample_step:.6f},scale={frame_width}:{frame_height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        frames: list[np.ndarray] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_bytes * 64,
            )
            while True:
                raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
                if not raw or len(raw) < frame_bytes:
                    break
                frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(frame_height, frame_width).copy())
            if proc.stdout:
                proc.stdout.close()
            proc.wait(timeout=30)
        except Exception as exc:
            logger.warning("Visual cut sampling failed: {}", exc)
            return segments, 0

        if not frames:
            return segments, 0

        def read_frame(timestamp: float) -> Optional[np.ndarray]:
            frame_index = int(round(max(0.0, float(timestamp)) / sample_step))
            if frame_index < 0 or frame_index >= len(frames):
                return None
            return frames[frame_index]

        def visual_cut_score(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
            left_hist = cv2.calcHist([left], [0], None, [32], [0, 256]).astype("float32")
            right_hist = cv2.calcHist([right], [0], None, [32], [0, 256]).astype("float32")
            left_hist = cv2.normalize(left_hist, left_hist).flatten()
            right_hist = cv2.normalize(right_hist, right_hist).flatten()
            hist_diff = float(cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CHISQR))
            mean_abs = float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))) / 255.0
            left_edges = cv2.Canny(left, 60, 140)
            right_edges = cv2.Canny(right, 60, 140)
            edge_diff = float(np.mean(np.abs(left_edges.astype(np.float32) - right_edges.astype(np.float32)))) / 255.0
            return hist_diff, mean_abs, edge_diff

        result: list[Segment] = []
        added = 0
        next_index = 1

        for segment in segments:
            duration = _segment_duration(segment)
            should_keep = (
                duration <= min_part_duration * 1.35
                or segment.skip_matching
                or segment.segment_type == SegmentType.NON_MOVIE
                or (preserve_manual_matches and segment.is_manual_match)
            )
            if should_keep:
                kept = segment.model_copy(deep=True)
                kept.index = next_index
                result.append(kept)
                next_index += 1
                continue

            start = float(segment.narration_start)
            end = float(segment.narration_end)
            sample_times: list[float] = []
            cursor = start
            while cursor < end - 0.05:
                sample_times.append(cursor)
                cursor += sample_step
            if not sample_times or sample_times[-1] < end:
                sample_times.append(end)

            cuts: list[float] = []
            previous_frame: Optional[np.ndarray] = None
            previous_time: Optional[float] = None
            last_cut = start
            for timestamp in sample_times:
                frame = read_frame(timestamp)
                if frame is None:
                    previous_frame = None
                    previous_time = None
                    continue
                if previous_frame is not None and previous_time is not None:
                    hist_diff, mean_abs, edge_diff = visual_cut_score(previous_frame, frame)
                    base_strong_cut = (
                        hist_diff >= 80.0
                        or (hist_diff >= 24.0 and mean_abs >= 0.12)
                        or mean_abs >= 0.55
                        or (edge_diff >= 0.30 and mean_abs >= 0.08)
                        or (mean_abs >= 0.46 and edge_diff >= 0.22)
                    )
                    moderate_long_cut = duration >= 2.4 and (
                        mean_abs >= 0.22
                        or (mean_abs >= 0.18 and edge_diff >= 0.20)
                    )
                    strong_cut = base_strong_cut or moderate_long_cut
                    cut_time = float(timestamp)
                    if end - cut_time < min_part_duration and previous_time is not None:
                        cut_time = float(previous_time)
                    if strong_cut and cut_time - last_cut >= min_part_duration and end - cut_time >= min_part_duration:
                        cuts.append(cut_time)
                        last_cut = cut_time
                previous_frame = frame
                previous_time = timestamp

            boundaries = [start] + cuts + [end]
            if len(boundaries) <= 2:
                kept = segment.model_copy(deep=True)
                kept.index = next_index
                result.append(kept)
                next_index += 1
                continue

            parent_id = segment.id
            for part_index, (part_start, part_end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
                if part_end - part_start <= 0.05:
                    continue
                clone = segment.model_copy(deep=True)
                clone.id = f"{parent_id}_c{part_index:02d}"
                clone.index = next_index
                clone.narration_start = float(part_start)
                clone.narration_end = float(part_end)
                _reset_segment_match_state(clone)
                clone.evidence_summary = f"visual_cut parent={parent_id}"
                if part_index > 1:
                    clone.original_text = ""
                    clone.polished_text = ""
                result.append(clone)
                next_index += 1
            added += len(boundaries) - 2

        return result, added

    return await asyncio.to_thread(_detect_and_split)


def _apply_selected_candidate(segment: Segment, candidate: MatchCandidate, status: AlignmentStatus, review_required: bool) -> None:
    narration_duration = max(0.05, _segment_duration(segment))
    source_duration = max(0.05, float(candidate.end) - float(candidate.start))
    source_speed_ratio = source_duration / narration_duration
    speed_evidence = max(
        float(candidate.verification_score or 0.0),
        min(float(candidate.visual_confidence or 0.0), float(candidate.stability_score or 0.0) + 0.10),
    )
    speed_changed = (
        0.50 <= source_speed_ratio <= 2.00
        and abs(source_speed_ratio - 1.0) >= 0.08
        and speed_evidence >= 0.70
        and (
            float(candidate.verification_score or 0.0) >= 0.68
            or (float(candidate.stability_score or 0.0) >= 0.68 and int(candidate.match_count or 0) >= 3)
        )
    )

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
    if review_required or candidate.confidence < 0.78:
        segment.match_type = "inferred"
    elif speed_changed:
        segment.match_type = "speed_changed"
    else:
        segment.match_type = "exact"
    segment.speed_changed = bool(speed_changed)
    segment.source_speed_ratio = float(source_speed_ratio if speed_changed else 1.0)
    segment.speed_change_confidence = float(speed_evidence if speed_changed else 0.0)
    segment.evidence_summary = (
        f"visual={candidate.visual_confidence:.2f}, "
        f"audio={candidate.audio_confidence:.2f}, "
        f"temporal={candidate.temporal_confidence:.2f}, "
        f"stability={candidate.stability_score:.2f}, "
        f"gap={candidate.rank_gap:.3f}, "
        f"verify={candidate.verification_score:.2f}"
        + (f", speed_ratio={segment.source_speed_ratio:.3f}" if speed_changed else "")
    )


def _build_visual_match_chunks(
    active_segments: list[Segment],
    boundary_break_ids: Optional[set[str]] = None,
    min_duration: float = 2.2,
    max_duration: float = 3.2,
    max_segments: int = 8,
) -> list[dict]:
    chunks: list[dict] = []
    current: list[Segment] = []
    chunk_start = 0.0
    chunk_end = 0.0
    chunk_boundary_break = False
    boundary_break_ids = boundary_break_ids or set()

    def flush() -> None:
        nonlocal current, chunk_start, chunk_end, chunk_boundary_break
        if not current:
            return
        chunks.append(
            {
                "id": f"vchunk_{len(chunks):04d}",
                "start": float(chunk_start),
                "end": float(chunk_end),
                "segments": list(current),
                "boundary_break": bool(chunk_boundary_break),
            }
        )
        current = []
        chunk_start = 0.0
        chunk_end = 0.0
        chunk_boundary_break = False

    for segment in active_segments:
        if segment.skip_matching or segment.segment_type == SegmentType.NON_MOVIE:
            flush()
            continue
        duration = _segment_duration(segment)
        if duration <= 0.0:
            continue
        starts_after_visual_break = str(segment.id) in boundary_break_ids
        if current and starts_after_visual_break:
            flush()
        if _is_visual_cut_piece(segment):
            flush()
            current = [segment]
            chunk_start = float(segment.narration_start)
            chunk_end = float(segment.narration_end)
            chunk_boundary_break = True
            flush()
            continue
        if not current:
            current = [segment]
            chunk_start = float(segment.narration_start)
            chunk_end = float(segment.narration_end)
            chunk_boundary_break = starts_after_visual_break
            continue

        gap = float(segment.narration_start) - chunk_end
        expanded_duration = float(segment.narration_end) - chunk_start
        should_flush = (
            gap > 0.35
            or (len(current) >= max_segments and chunk_end - chunk_start >= min_duration)
            or expanded_duration > max_duration
        )
        if should_flush:
            flush()
            current = [segment]
            chunk_start = float(segment.narration_start)
            chunk_end = float(segment.narration_end)
            chunk_boundary_break = starts_after_visual_break
            continue
        current.append(segment)
        chunk_end = max(chunk_end, float(segment.narration_end))

    flush()
    return chunks


def _apply_visual_chunk_result(
    project,
    chunk: dict,
    chunk_source_start: float,
    confidence: float,
    visual_confidence: float,
    stability_score: float,
    inferred: bool,
    reason: str,
    chunk_source_duration: Optional[float] = None,
    speed_changed: bool = False,
    source_speed_ratio: float = 1.0,
    speed_change_confidence: float = 0.0,
) -> int:
    movie_duration = float(project.movie_duration or 0.0)
    chunk_target_duration = max(0.05, float(chunk["end"]) - float(chunk["start"]))
    total_source_duration = (
        max(0.05, float(chunk_source_duration))
        if chunk_source_duration is not None and float(chunk_source_duration) > 0.0
        else chunk_target_duration
    )
    applied = 0
    for segment in chunk["segments"]:
        duration = _segment_duration(segment)
        if duration <= 0.0:
            continue
        offset = max(0.0, float(segment.narration_start) - float(chunk["start"]))
        source_offset = total_source_duration * (offset / chunk_target_duration)
        source_duration = max(0.05, total_source_duration * (duration / chunk_target_duration))
        source_start = max(0.0, float(chunk_source_start) + source_offset)
        source_end = source_start + source_duration
        if movie_duration > 0.0 and source_end > movie_duration:
            source_end = movie_duration
            source_start = max(0.0, source_end - source_duration)
        if source_end <= source_start:
            continue

        duration_gap = abs((source_end - source_start) - duration)
        candidate = MatchCandidate(
            id=f"{segment.id}_cand_1",
            start=source_start,
            end=source_end,
            score=float(confidence),
            confidence=float(confidence),
            visual_confidence=float(visual_confidence),
            temporal_confidence=0.92 if not inferred else 0.72,
            stability_score=float(stability_score),
            duration_gap=duration_gap,
            match_count=max(1, len(chunk["segments"])),
            reason=reason,
            source="visual_chunk",
            rank=1,
        )
        segment.match_candidates = [candidate]
        status = AlignmentStatus.AUTO_ACCEPTED if inferred or confidence >= 0.74 else AlignmentStatus.NEEDS_REVIEW
        review_required = False if inferred else confidence < 0.74
        _apply_selected_candidate(segment, candidate, status, review_required)
        segment.match_type = "inferred" if inferred or review_required else ("speed_changed" if speed_changed else "exact")
        segment.speed_changed = bool(speed_changed)
        segment.source_speed_ratio = float(source_speed_ratio if speed_changed else 1.0)
        segment.speed_change_confidence = float(speed_change_confidence if speed_changed else 0.0)
        speed_note = (
            f"; speed_changed ratio={segment.source_speed_ratio:.3f} conf={segment.speed_change_confidence:.2f}"
            if segment.speed_changed
            else ""
        )
        segment.evidence_summary = f"visual_chunk conf={confidence:.2f}; inferred={inferred}{speed_note}"
        applied += 1
    return applied


def _repair_visual_chunk_continuity(
    frame_matcher,
    chunks: list[dict],
    results: list[Optional[dict]],
    feature_map: dict[str, list[dict]],
    movie_duration: float,
) -> tuple[int, int]:
    """Prefer local source continuity when an adjacent micro-match jumps to a similar far-away frame."""
    scorer = getattr(frame_matcher, "score_precomputed_segment_at", None)
    if scorer is None:
        return 0, 0

    repaired = 0
    filled = 0
    previous_target_end: Optional[float] = None
    previous_source_end: Optional[float] = None
    previous_chunk_id: Optional[str] = None
    previous_chunk_duration: Optional[float] = None
    previous_confidence: float = 0.0

    def boundary_visual_score(left_id: Optional[str], right_id: str) -> float:
        if not left_id:
            return 0.0
        feature_score = getattr(frame_matcher, "_feature_score", None)
        if feature_score is None:
            return 0.0
        left_features = feature_map.get(str(left_id), [])
        right_features = feature_map.get(str(right_id), [])
        best = 0.0
        for left in left_features:
            for right in right_features:
                try:
                    best = max(best, float(feature_score(left, right)))
                except Exception:
                    continue
        return best

    for index, chunk in enumerate(chunks):
        chunk_start = float(chunk["start"])
        chunk_end = float(chunk["end"])
        chunk_duration = max(0.2, chunk_end - chunk_start)
        result = results[index]

        if (
            previous_target_end is not None
            and previous_source_end is not None
            and not bool(chunk.get("boundary_break", False))
        ):
            target_gap = chunk_start - previous_target_end
            if -0.05 <= target_gap <= 0.35:
                expected_start = max(0.0, previous_source_end + max(0.0, target_gap))
                expected_end = expected_start + chunk_duration
                if movie_duration > 0.0 and expected_end <= movie_duration:
                    features = feature_map.get(str(chunk["id"]), [])
                    expected_score = scorer(features, expected_start, search_radius=0.95)
                    expected_confidence = float(expected_score.get("confidence", 0.0))
                    best_expected_start = expected_start
                    best_expected_end = expected_end
                    best_expected_score = expected_score
                    best_expected_confidence = expected_confidence
                    best_expected_final = expected_confidence
                    for offset in (-6.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 6.0):
                        candidate_start = max(0.0, expected_start + offset)
                        candidate_end = candidate_start + chunk_duration
                        if movie_duration > 0.0 and candidate_end > movie_duration:
                            continue
                        candidate_score = scorer(features, candidate_start, search_radius=0.55)
                        candidate_confidence = float(candidate_score.get("confidence", 0.0))
                        continuity_penalty = min(0.05, abs(offset) * 0.006)
                        candidate_final = candidate_confidence - continuity_penalty
                        if candidate_final > best_expected_final:
                            best_expected_start = candidate_start
                            best_expected_end = candidate_end
                            best_expected_score = candidate_score
                            best_expected_confidence = candidate_confidence
                            best_expected_final = candidate_final
                    expected_start = best_expected_start
                    expected_end = best_expected_end
                    expected_score = best_expected_score
                    expected_confidence = best_expected_confidence
                    actual_confidence = float(result.get("confidence", 0.0)) if result else 0.0
                    actual_start = float(result.get("start", -999999.0)) if result else -999999.0
                    source_gap = actual_start - previous_source_end if result else 999999.0
                    jump_mismatch = abs(source_gap - max(0.0, target_gap))
                    boundary_score = boundary_visual_score(previous_chunk_id, str(chunk["id"]))
                    visual_is_continuous = boundary_score >= 0.58
                    strong_actual_anchor = (
                        result is not None
                        and actual_confidence >= 0.86
                        and float(result.get("stability_score", 0.0)) >= 0.48
                        and float(result.get("low_info_ratio", 1.0)) <= 0.48
                    )
                    weak_previous_anchor = (
                        previous_chunk_duration is not None
                        and previous_chunk_duration < 0.85
                        and previous_confidence < 0.78
                    )
                    repeat_or_backtrack = source_gap < -0.12
                    protected_identity_match = (
                        result is not None
                        and bool(result.get("dino_corrected", False))
                        and float(result.get("identity_score", 0.0)) >= 0.90
                        and actual_confidence >= 0.94
                        and not repeat_or_backtrack
                        and jump_mismatch <= max(2.0, chunk_duration * 2.5)
                    )
                    tolerated_margin = 0.16 if visual_is_continuous else 0.08
                    required_expected = 0.70 if visual_is_continuous else 0.76
                    if repeat_or_backtrack and visual_is_continuous:
                        tolerated_margin = 0.22
                        required_expected = 0.68
                    should_repair_jump = (
                        result is not None
                        and not weak_previous_anchor
                        and not protected_identity_match
                        and not (
                            strong_actual_anchor
                            and expected_confidence < actual_confidence + 0.035
                        )
                        and jump_mismatch > max(0.38, chunk_duration * 0.48)
                        and expected_confidence >= required_expected
                        and expected_confidence + tolerated_margin >= actual_confidence
                    )
                    should_fill_missing = (
                        result is None
                        and not weak_previous_anchor
                        and expected_confidence >= 0.70
                    )
                    if should_repair_jump or should_fill_missing:
                        replacement = {
                            "success": True,
                            "start": expected_start,
                            "end": expected_end,
                            "confidence": expected_confidence,
                            "rank_gap": 0.0,
                            "match_count": int(expected_score.get("match_count", 0)),
                            "stability_score": float(expected_score.get("stability_score", 0.0)),
                            "candidate_quality": float(expected_score.get("candidate_quality", 0.0)),
                            "query_quality": float(expected_score.get("query_quality", 0.0)),
                            "low_info_ratio": float(expected_score.get("low_info_ratio", 1.0)),
                            "continuity_repaired": should_repair_jump,
                            "continuity_filled": should_fill_missing,
                        }
                        results[index] = replacement
                        result = replacement
                        if should_repair_jump:
                            repaired += 1
                        else:
                            filled += 1

        if result and result.get("success"):
            source_start = max(0.0, float(result["start"]))
            source_end = max(source_start + 0.05, float(result.get("end", source_start + chunk_duration)))
            if movie_duration <= 0.0 or source_end <= movie_duration:
                previous_target_end = chunk_end
                previous_source_end = source_end
                previous_chunk_id = str(chunk["id"])
                previous_chunk_duration = chunk_duration
                previous_confidence = float(result.get("confidence", 0.0))
            else:
                previous_target_end = None
                previous_source_end = None
                previous_chunk_id = None
                previous_chunk_duration = None
                previous_confidence = 0.0
        else:
            previous_target_end = None
            previous_source_end = None
            previous_chunk_id = None
            previous_chunk_duration = None
            previous_confidence = 0.0

    return repaired, filled


def _repair_visual_chunk_source_islands(
    frame_matcher,
    chunks: list[dict],
    results: list[Optional[dict]],
    feature_map: dict[str, list[dict]],
    movie_duration: float,
) -> int:
    """Fix isolated source jumps surrounded by locally continuous neighbors."""
    scorer = getattr(frame_matcher, "score_precomputed_segment_at", None)
    if scorer is None or len(chunks) < 3:
        return 0

    repaired = 0
    for index in range(1, len(chunks) - 1):
        previous = results[index - 1]
        current = results[index]
        following = results[index + 1]
        if not (previous and current and following):
            continue
        if not (previous.get("success") and current.get("success") and following.get("success")):
            continue

        chunk = chunks[index]
        prev_chunk = chunks[index - 1]
        next_chunk = chunks[index + 1]
        if bool(chunk.get("boundary_break", False)) or bool(next_chunk.get("boundary_break", False)):
            continue
        target_gap_prev = float(chunk["start"]) - float(prev_chunk["end"])
        target_gap_next = float(next_chunk["start"]) - float(chunk["end"])
        if not (-0.05 <= target_gap_prev <= 0.45 and -0.05 <= target_gap_next <= 0.45):
            continue

        duration = max(0.2, float(chunk["end"]) - float(chunk["start"]))
        prev_source_end = max(float(previous["start"]) + 0.05, float(previous.get("end", float(previous["start"]) + duration)))
        next_source_start = float(following["start"])
        current_start = float(current["start"])
        current_source_duration = max(0.05, float(current.get("end", current_start + duration)) - current_start)
        expected_start = max(0.0, prev_source_end + max(0.0, target_gap_prev))
        expected_next_start = expected_start + current_source_duration + max(0.0, target_gap_next)
        if movie_duration > 0.0 and expected_start + current_source_duration > movie_duration:
            continue

        current_jump = abs(current_start - expected_start)
        next_consistent = abs(next_source_start - expected_next_start) <= max(1.20, duration * 0.85)
        if current_jump <= max(1.10, duration * 0.80) or not next_consistent:
            continue

        features = feature_map.get(str(chunk["id"]), [])
        if not features:
            continue

        best_start = expected_start
        best_payload = scorer(features, expected_start, search_radius=0.58)
        best_score = float(best_payload.get("confidence", 0.0))
        for offset in (-0.75, -0.50, -0.25, 0.25, 0.50, 0.75):
            candidate_start = max(0.0, expected_start + offset)
            if movie_duration > 0.0 and candidate_start + current_source_duration > movie_duration:
                continue
            payload = scorer(features, candidate_start, search_radius=0.50)
            score = float(payload.get("confidence", 0.0))
            if score > best_score:
                best_start = candidate_start
                best_payload = payload
                best_score = score

        current_confidence = float(current.get("confidence", 0.0))
        if current_confidence >= 0.86 and best_score < current_confidence + 0.035:
            continue
        if best_score < 0.66 or best_score + 0.13 < current_confidence:
            continue

        replacement = dict(current)
        replacement["start"] = best_start
        replacement["end"] = best_start + current_source_duration
        replacement["confidence"] = min(float(current.get("confidence", 0.0)), max(0.70, best_score))
        replacement["match_count"] = int(best_payload.get("match_count", current.get("match_count", 0)))
        replacement["stability_score"] = max(
            float(current.get("stability_score", 0.0)),
            float(best_payload.get("stability_score", 0.0)),
        )
        replacement["candidate_quality"] = float(best_payload.get("candidate_quality", current.get("candidate_quality", 0.0)))
        replacement["query_quality"] = float(best_payload.get("query_quality", current.get("query_quality", 0.0)))
        replacement["low_info_ratio"] = float(best_payload.get("low_info_ratio", current.get("low_info_ratio", 1.0)))
        replacement["source_island_repaired"] = True
        results[index] = replacement
        repaired += 1

    return repaired


def _repair_low_confidence_neighbor_bridges(
    frame_matcher,
    chunks: list[dict],
    results: list[Optional[dict]],
    feature_map: dict[str, list[dict]],
    movie_duration: float,
) -> int:
    """Pull weak isolated matches back between two locally consistent neighbors."""
    scorer = getattr(frame_matcher, "score_precomputed_segment_at", None)
    if scorer is None or len(chunks) < 3:
        return 0

    repaired = 0
    for index in range(1, len(chunks) - 1):
        previous = results[index - 1]
        current = results[index]
        following = results[index + 1]
        if not (previous and current and following):
            continue
        if not (previous.get("success") and current.get("success") and following.get("success")):
            continue

        confidence = float(current.get("confidence", 0.0))
        if confidence >= 0.72:
            continue

        chunk = chunks[index]
        prev_chunk = chunks[index - 1]
        next_chunk = chunks[index + 1]
        target_gap_prev = float(chunk["start"]) - float(prev_chunk["end"])
        target_gap_next = float(next_chunk["start"]) - float(chunk["end"])
        if not (-0.08 <= target_gap_prev <= 0.55 and -0.08 <= target_gap_next <= 0.55):
            continue

        duration = max(0.2, float(chunk["end"]) - float(chunk["start"]))
        prev_source_end = max(
            float(previous["start"]) + 0.05,
            float(previous.get("end", float(previous["start"]) + duration)),
        )
        next_source_start = float(following["start"])
        current_start = float(current["start"])
        local_span = next_source_start - prev_source_end
        if local_span < -0.30 or local_span > max(14.0, duration * 9.0):
            continue

        expected_start = max(0.0, prev_source_end + max(0.0, target_gap_prev))
        latest_start = max(0.0, next_source_start - duration - max(0.0, target_gap_next))
        if latest_start + duration > 0.0:
            expected_start = min(expected_start, latest_start) if latest_start >= expected_start else expected_start
        source_jump = min(abs(current_start - expected_start), abs(current_start - latest_start))
        if source_jump < max(8.0, duration * 5.0):
            continue

        features = feature_map.get(str(chunk["id"]), [])
        if not features:
            continue

        candidate_starts = {round(expected_start, 3), round(max(0.0, latest_start), 3)}
        midpoint = max(0.0, (expected_start + max(expected_start, latest_start)) * 0.5)
        candidate_starts.add(round(midpoint, 3))
        for base in (expected_start, latest_start, midpoint):
            for offset in (-1.0, -0.5, 0.5, 1.0, 2.0, 3.5):
                candidate_starts.add(round(max(0.0, base + offset), 3))

        best_start = current_start
        best_payload: Optional[dict] = None
        best_score = 0.0
        for candidate_start in sorted(candidate_starts):
            if movie_duration > 0.0 and candidate_start + duration > movie_duration:
                continue
            payload = scorer(features, candidate_start, search_radius=0.70)
            score = float(payload.get("confidence", 0.0))
            if score > best_score:
                best_score = score
                best_start = candidate_start
                best_payload = payload

        if best_payload is None:
            continue
        if best_score < 0.56 or (best_score + 0.08 < confidence and source_jump < 60.0):
            continue

        replacement = dict(current)
        replacement["start"] = best_start
        replacement["end"] = best_start + duration
        replacement["confidence"] = max(confidence, min(0.74, best_score))
        replacement["match_count"] = int(best_payload.get("match_count", current.get("match_count", 0)))
        replacement["stability_score"] = max(
            float(current.get("stability_score", 0.0)),
            float(best_payload.get("stability_score", 0.0)),
        )
        replacement["candidate_quality"] = float(best_payload.get("candidate_quality", current.get("candidate_quality", 0.0)))
        replacement["query_quality"] = float(best_payload.get("query_quality", current.get("query_quality", 0.0)))
        replacement["low_info_ratio"] = float(best_payload.get("low_info_ratio", current.get("low_info_ratio", 1.0)))
        replacement["neighbor_bridge_repaired"] = True
        results[index] = replacement
        repaired += 1

    return repaired


def _refine_visual_chunk_source_offsets(
    frame_matcher,
    chunks: list[dict],
    results: list[Optional[dict]],
    feature_map: dict[str, list[dict]],
    movie_duration: float,
) -> int:
    """Fine-tune sub-second source offsets after coarse retrieval."""
    scorer = getattr(frame_matcher, "score_precomputed_segment_at", None)
    if scorer is None:
        return 0

    offsets = (-1.25, -1.0, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
    refined = 0
    for index, chunk in enumerate(chunks):
        result = results[index]
        if not result or not result.get("success"):
            continue
        features = feature_map.get(str(chunk["id"]), [])
        if not features:
            continue

        chunk_duration = max(0.2, float(chunk["end"]) - float(chunk["start"]))
        current_start = max(0.0, float(result["start"]))
        current_score = float(scorer(features, current_start, search_radius=0.45).get("confidence", 0.0))
        best_start = current_start
        best_score = current_score
        for offset in offsets:
            candidate_start = max(0.0, current_start + offset)
            if movie_duration > 0.0 and candidate_start + chunk_duration > movie_duration:
                continue
            score = float(scorer(features, candidate_start, search_radius=0.45).get("confidence", 0.0))
            if score > best_score:
                best_score = score
                best_start = candidate_start

        should_refine = (
            abs(best_start - current_start) >= 0.08
            and best_score >= 0.58
            and (
                best_score >= current_score + 0.035
                or (current_score < 0.68 and best_score >= current_score + 0.020)
            )
        )
        if not should_refine:
            continue

        replacement = dict(result)
        replacement["start"] = best_start
        replacement["end"] = best_start + chunk_duration
        replacement["confidence"] = max(float(result.get("confidence", 0.0)), best_score)
        replacement["source_offset_refined"] = True
        results[index] = replacement
        refined += 1

    return refined


def _refine_visual_chunk_warped_ranges(
    frame_matcher,
    chunks: list[dict],
    results: list[Optional[dict]],
    feature_map: dict[str, list[dict]],
    movie_duration: float,
    max_targets: int = 60,
) -> int:
    """Refit source in/out and playback scale using indexed structure features.

    DINO is good at saying "this is the same scene"; it is not precise enough
    to say "this is the same motion phase". This pass keeps the retrieved scene
    but searches nearby starts and several source-duration scales against the
    hash/edge/layout index, which is more sensitive to action timing and more
    tolerant of color grading/subtitles than raw pixels.
    """
    scorer = getattr(frame_matcher, "score_precomputed_segment_at", None)
    if scorer is None or not chunks or not results:
        return 0

    def score_at(features: list[dict], start: float, *, radius: float, scales: tuple[float, ...]) -> dict:
        try:
            payload = scorer(features, start, search_radius=radius, time_scales=scales)
        except TypeError:
            payload = scorer(features, start, search_radius=radius)
            payload["time_scale"] = 1.0
        return payload

    def source_duration_for(chunk: dict, result: dict) -> float:
        target_duration = max(0.05, float(chunk["end"]) - float(chunk["start"]))
        start = float(result.get("start", 0.0))
        end = float(result.get("end", start + target_duration))
        return max(0.05, end - start)

    previous_source_end_by_index: dict[int, float] = {}
    previous_target_end_by_index: dict[int, float] = {}
    previous_source_end: Optional[float] = None
    previous_target_end: Optional[float] = None
    for index, (chunk, result) in enumerate(zip(chunks, results)):
        if bool(chunk.get("boundary_break", False)):
            previous_source_end = None
            previous_target_end = None
        if previous_source_end is not None and previous_target_end is not None:
            previous_source_end_by_index[index] = previous_source_end
            previous_target_end_by_index[index] = previous_target_end
        if result and result.get("success"):
            previous_source_end = float(result["start"]) + source_duration_for(chunk, result)
            previous_target_end = float(chunk["end"])
        else:
            previous_source_end = None
            previous_target_end = None

    priorities: list[tuple[float, int]] = []
    for index, (chunk, result) in enumerate(zip(chunks, results)):
        if not result or not result.get("success"):
            continue
        features = feature_map.get(str(chunk["id"]), [])
        if not features:
            continue
        duration = max(0.2, float(chunk["end"]) - float(chunk["start"]))
        confidence = float(result.get("confidence", 0.0))
        priority = 0.0
        if confidence < 0.82:
            priority += 2.0
        if float(result.get("identity_score", 1.0)) < 0.74:
            priority += 1.4
        if bool(result.get("continuity_repaired", False)) or bool(result.get("continuity_filled", False)):
            priority += 1.2
        if float(result.get("rank_gap", 1.0)) < 0.035:
            priority += 0.6
        if priority > 0.0:
            priorities.append((priority, index))

    if not priorities:
        return 0
    target_indices = {
        index
        for _, index in sorted(priorities, key=lambda item: item[0], reverse=True)[: max(1, int(max_targets))]
    }

    refined = 0
    for index in sorted(target_indices):
        chunk = chunks[index]
        result = results[index]
        if not result or not result.get("success"):
            continue
        features = feature_map.get(str(chunk["id"]), [])
        if not features:
            continue

        duration = max(0.2, float(chunk["end"]) - float(chunk["start"]))
        current_start = max(0.0, float(result["start"]))
        current_source_duration = source_duration_for(chunk, result)
        current_scale = max(0.35, min(2.20, current_source_duration / duration))
        if duration < 0.75:
            scales = (0.82, 0.92, 1.0, 1.10, 1.22)
            radius = 0.42
            start_offsets = (-0.90, -0.65, -0.42, -0.25, -0.12, 0.0, 0.12, 0.25, 0.42, 0.65, 0.90)
        elif duration < 1.45:
            scales = (0.72, 0.82, 0.92, 1.0, 1.10, 1.22, 1.38)
            radius = 0.50
            start_offsets = (
                -2.4, -1.6, -1.0, -0.65, -0.42, -0.25, -0.12,
                0.0,
                0.12, 0.25, 0.42, 0.65, 1.0, 1.6, 2.4,
            )
        else:
            scales = (0.62, 0.72, 0.84, 0.94, 1.0, 1.10, 1.24, 1.42, 1.62)
            radius = 0.58
            start_offsets = (
                -12.0, -8.0, -5.0, -3.2, -2.0, -1.2, -0.65, -0.35, -0.15,
                0.0,
                0.15, 0.35, 0.65, 1.2, 2.0, 3.2, 5.0, 8.0, 12.0,
            )

        current_payload = score_at(features, current_start, radius=radius, scales=(current_scale,))
        current_score = float(current_payload.get("confidence", 0.0))
        current_final = current_score
        best_start = current_start
        best_scale = current_scale
        best_score = current_score
        best_final = current_final
        best_payload = current_payload

        candidate_starts: dict[float, str] = {}
        for offset in start_offsets:
            candidate_starts[round(max(0.0, current_start + offset), 3)] = "local"

        prev_source_end = previous_source_end_by_index.get(index)
        prev_target_end = previous_target_end_by_index.get(index)
        if prev_source_end is not None and prev_target_end is not None:
            target_gap = float(chunk["start"]) - prev_target_end
            if -0.05 <= target_gap <= 0.45:
                expected_start = max(0.0, prev_source_end + max(0.0, target_gap))
                for offset in (-0.42, -0.25, -0.12, 0.0, 0.12, 0.25, 0.42):
                    candidate_starts[round(max(0.0, expected_start + offset), 3)] = "continuity"

        for dino_candidate in result.get("_dino_candidates") or []:
            try:
                dino_start = float(dino_candidate.get("start", 0.0))
            except Exception:
                continue
            for offset in (-0.42, -0.20, 0.0, 0.20, 0.42):
                candidate_starts[round(max(0.0, dino_start + offset), 3)] = "dino"

        for candidate_start, source_label in candidate_starts.items():
            if candidate_start < 0.0:
                continue
            payload = score_at(features, candidate_start, radius=radius, scales=scales)
            score = float(payload.get("confidence", 0.0))
            scale = max(0.35, min(2.20, float(payload.get("time_scale", 1.0))))
            source_duration = duration * scale
            if movie_duration > 0.0 and candidate_start + source_duration > movie_duration:
                continue

            final_score = score
            if source_label == "continuity":
                final_score += 0.010
            if prev_source_end is not None and prev_target_end is not None:
                target_gap = float(chunk["start"]) - prev_target_end
                if -0.05 <= target_gap <= 0.45:
                    source_gap = candidate_start - prev_source_end
                    if source_gap < -0.08:
                        final_score -= min(0.10, abs(source_gap) * 0.10)
                    elif abs(source_gap - max(0.0, target_gap)) > max(2.4, duration * 2.6):
                        final_score -= 0.025
            if abs(scale - 1.0) > 0.45:
                final_score -= 0.010

            if final_score > best_final:
                best_start = float(candidate_start)
                best_scale = scale
                best_score = score
                best_final = final_score
                best_payload = payload

        source_duration = duration * best_scale
        if movie_duration > 0.0 and best_start + source_duration > movie_duration:
            continue
        start_changed = abs(best_start - current_start) >= 0.08
        scale_changed = abs(best_scale - current_scale) >= 0.08
        if not (start_changed or scale_changed):
            continue
        large_shift = abs(best_start - current_start) >= 1.50
        required_improvement = 0.050 if large_shift else 0.032
        if scale_changed and not start_changed:
            required_improvement = 0.040
        if current_score < 0.58:
            required_improvement = min(required_improvement, 0.026)
        should_refine = (
            best_score >= 0.64
            and best_final >= current_final + required_improvement
        )
        if not should_refine:
            continue

        replacement = dict(result)
        replacement["start"] = best_start
        replacement["end"] = best_start + source_duration
        replacement["confidence"] = max(float(result.get("confidence", 0.0)), min(0.96, best_score))
        replacement["visual_warp_refined"] = True
        replacement["visual_warp_score"] = best_score
        replacement["visual_warp_score_before"] = current_score
        replacement["source_speed_ratio"] = best_scale
        replacement["match_count"] = int(best_payload.get("match_count", result.get("match_count", 0)))
        replacement["stability_score"] = max(
            float(result.get("stability_score", 0.0)),
            float(best_payload.get("stability_score", 0.0)),
        )
        replacement["candidate_quality"] = float(best_payload.get("candidate_quality", result.get("candidate_quality", 0.0)))
        replacement["query_quality"] = float(best_payload.get("query_quality", result.get("query_quality", 0.0)))
        replacement["low_info_ratio"] = float(best_payload.get("low_info_ratio", result.get("low_info_ratio", 1.0)))
        results[index] = replacement
        refined += 1

    return refined


def _refine_visual_chunk_phase_offsets(
    project,
    chunks: list[dict],
    results: list[Optional[dict]],
    movie_duration: float,
    max_targets: int = 24,
) -> int:
    """Use direct frame structure to fix local phase drift around matched starts."""
    if not chunks or not results:
        return 0

    prioritized: list[tuple[float, int]] = []
    for index, (chunk, result) in enumerate(zip(chunks, results)):
        if not result or not result.get("success"):
            continue
        if bool(result.get("visual_warp_refined", False)):
            continue
        duration = max(0.2, float(chunk["end"]) - float(chunk["start"]))
        confidence = float(result.get("confidence", 0.0))
        priority = 0.0
        if bool(result.get("continuity_repaired", False)) or bool(result.get("continuity_filled", False)):
            priority += 3.0
        if confidence < 0.82:
            priority += 2.0
        identity_score = float(result.get("identity_score", 1.0))
        if identity_score < 0.78:
            priority += 1.5
        if duration >= 2.3 and confidence < 0.88:
            priority += 1.0
        if float(result.get("rank_gap", 1.0)) < 0.025:
            priority += 0.5
        if priority > 0.0:
            prioritized.append((priority, index))

    if not prioritized:
        return 0
    target_indices = {
        index
        for _, index in sorted(prioritized, key=lambda item: item[0], reverse=True)[: max(1, int(max_targets))]
    }

    try:
        import cv2
    except Exception:
        return 0

    narration_capture = cv2.VideoCapture(str(project.narration_path))
    movie_capture = cv2.VideoCapture(str(project.movie_path))
    if not narration_capture.isOpened() or not movie_capture.isOpened():
        narration_capture.release()
        movie_capture.release()
        return 0

    width = 160
    crop_ratio = 0.76
    narration_cache: dict[float, np.ndarray] = {}
    movie_cache: dict[float, np.ndarray] = {}

    def read_gray(capture, cache: dict[float, np.ndarray], timestamp: float) -> Optional[np.ndarray]:
        key = round(max(0.0, float(timestamp)), 2)
        cached = cache.get(key)
        if cached is not None:
            return cached
        capture.set(cv2.CAP_PROP_POS_MSEC, key * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        height, frame_width = frame.shape[:2]
        if height <= 0 or frame_width <= 0:
            return None
        frame = frame[: max(1, int(height * crop_ratio)), :]
        resized_height = max(24, int(frame.shape[0] * width / max(1, frame.shape[1])))
        frame = cv2.resize(frame, (width, resized_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cache[key] = gray
        if len(cache) > 512:
            cache.pop(next(iter(cache)))
        return gray

    def center_crop(frame: np.ndarray, ratio: float) -> np.ndarray:
        height, frame_width = frame.shape[:2]
        crop_h = max(8, int(height * ratio))
        crop_w = max(8, int(frame_width * ratio))
        y = max(0, (height - crop_h) // 2)
        x = max(0, (frame_width - crop_w) // 2)
        return frame[y : y + crop_h, x : x + crop_w]

    def score_pair(query_gray: Optional[np.ndarray], movie_gray: Optional[np.ndarray]) -> float:
        if query_gray is None or movie_gray is None:
            return 0.0
        if query_gray.shape != movie_gray.shape:
            movie_gray = cv2.resize(movie_gray, (query_gray.shape[1], query_gray.shape[0]), interpolation=cv2.INTER_AREA)

        def score_one(query: np.ndarray, movie: np.ndarray) -> float:
            if query.shape != movie.shape:
                movie = cv2.resize(movie, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA)
            query_hist = cv2.calcHist([query], [0], None, [48], [0, 256]).astype("float32")
            movie_hist = cv2.calcHist([movie], [0], None, [48], [0, 256]).astype("float32")
            query_hist /= max(float(query_hist.sum()), 1.0)
            movie_hist /= max(float(movie_hist.sum()), 1.0)
            hist = (float(cv2.compareHist(query_hist, movie_hist, cv2.HISTCMP_CORREL)) + 1.0) / 2.0
            hist = max(0.0, min(1.0, hist))
            query_edges = cv2.Canny(query, 60, 140)
            movie_edges = cv2.Canny(movie, 60, 140)
            edge = 1.0 - float(np.mean(np.abs(query_edges.astype(np.float32) - movie_edges.astype(np.float32)))) / 255.0
            edge = max(0.0, min(1.0, edge))
            return float(hist * 0.45 + edge * 0.55)

        return max(
            score_one(query_gray, movie_gray),
            score_one(center_crop(query_gray, 0.82), center_crop(movie_gray, 0.82)),
            score_one(center_crop(query_gray, 0.68), center_crop(movie_gray, 0.68)),
        )

    def sample_offsets(duration: float) -> list[float]:
        if duration <= 0.75:
            return [duration * 0.30, duration * 0.70]
        if duration <= 1.6:
            return [duration * 0.24, duration * 0.54, duration * 0.82]
        if duration <= 3.2:
            return [duration * 0.18, duration * 0.42, duration * 0.68, duration * 0.88]
        return [duration * 0.15, duration * 0.35, duration * 0.58, duration * 0.78, duration * 0.92]

    def candidate_score(chunk_start: float, candidate_start: float, offsets: list[float], source_scale: float) -> float:
        scores: list[float] = []
        for offset in offsets:
            query = read_gray(narration_capture, narration_cache, chunk_start + offset)
            movie = read_gray(movie_capture, movie_cache, candidate_start + offset * source_scale)
            scores.append(score_pair(query, movie))
        if not scores:
            return 0.0
        return float(np.mean(scores) * 0.72 + min(scores) * 0.28)

    refined = 0
    started_at = time.perf_counter()
    try:
        for index, (chunk, result) in enumerate(zip(chunks, results)):
            if time.perf_counter() - started_at > 45.0:
                break
            if index not in target_indices:
                continue
            if not result or not result.get("success"):
                continue
            chunk_start = float(chunk["start"])
            chunk_end = float(chunk["end"])
            duration = max(0.2, chunk_end - chunk_start)
            current_start = max(0.0, float(result["start"]))
            current_source_duration = max(0.05, float(result.get("end", current_start + duration)) - current_start)
            source_scale = max(0.35, min(2.20, current_source_duration / duration))
            offsets = sample_offsets(duration)
            current_score = candidate_score(chunk_start, current_start, offsets, source_scale)
            best_start = current_start
            best_scale = source_scale
            best_score = current_score
            confidence = float(result.get("confidence", 0.0))
            identity_score = float(result.get("identity_score", 1.0))
            offset_limit = 2.0 if confidence < 0.82 or identity_score < 0.78 else 1.0
            offsets_to_try = [step * 0.50 for step in range(int(-offset_limit / 0.50), int(offset_limit / 0.50) + 1)]
            scale_options = [source_scale]
            for scale in (0.90, 1.0, 1.10):
                if not any(abs(scale - existing) <= 0.03 for existing in scale_options):
                    scale_options.append(scale)
            for scale in scale_options:
                candidate_duration = duration * max(0.35, min(2.20, scale))
                for delta in offsets_to_try:
                    candidate_start = max(0.0, current_start + delta)
                    if movie_duration > 0.0 and candidate_start + candidate_duration > movie_duration:
                        continue
                    score = candidate_score(chunk_start, candidate_start, offsets, scale)
                    if abs(scale - source_scale) > 0.08:
                        score -= 0.018
                    if score > best_score:
                        best_score = score
                        best_start = candidate_start
                        best_scale = scale
            if abs(best_start - current_start) < 0.08:
                if abs(best_scale - source_scale) < 0.08:
                    continue
            if best_score < 0.62 or best_score < current_score + 0.045:
                continue
            replacement = dict(result)
            replacement["start"] = best_start
            replacement["end"] = best_start + duration * max(0.35, min(2.20, best_scale))
            replacement["confidence"] = max(float(result.get("confidence", 0.0)), min(0.94, best_score))
            replacement["phase_offset_refined"] = True
            replacement["phase_score"] = best_score
            replacement["phase_score_before"] = current_score
            replacement["source_speed_ratio"] = best_scale
            results[index] = replacement
            refined += 1
    finally:
        narration_capture.release()
        movie_capture.release()

    return refined


async def _repair_segment_visual_mismatches(
    project_id: str,
    frame_matcher,
    project,
    active_segments: list[Segment],
    movie_duration: float,
    *,
    max_targets: int = 42,
    max_concurrent: int = 3,
) -> int:
    """Second-pass repair for per-segment phase errors inside a matched scene.

    Chunk matching is intentionally monotonic and stable, but a narration edit can
    insert a short close-up/remote shot inside the same movie scene. In that case
    the chunk is globally right while a single segment is visually wrong. This
    pass verifies selected segment timelines, then rematches only the weak ones
    in a local source window.
    """
    verifier = getattr(frame_matcher, "verify_segment_matches", None)
    if verifier is None or not active_segments:
        return 0

    eligible: list[Segment] = []
    previous: Optional[Segment] = None
    risky_ids: set[str] = set()
    for segment in active_segments:
        if (
            segment.skip_matching
            or segment.segment_type == SegmentType.NON_MOVIE
            or segment.movie_start is None
            or segment.movie_end is None
            or _segment_duration(segment) < 0.50
        ):
            previous = None if segment.segment_type == SegmentType.NON_MOVIE else previous
            continue

        eligible.append(segment)
        if previous and previous.movie_start is not None and previous.movie_end is not None:
            target_gap = float(segment.narration_start) - float(previous.narration_end)
            source_gap = float(segment.movie_start) - float(previous.movie_end)
            if abs(source_gap - max(0.0, target_gap)) > max(5.0, _segment_duration(segment) * 2.8):
                risky_ids.add(str(segment.id))
                risky_ids.add(str(previous.id))
        previous = segment

    if not eligible:
        return 0

    verify_payload = [
        {
            "id": str(segment.id),
            "narration_start": float(segment.narration_start),
            "narration_end": float(segment.narration_end),
            "movie_start": float(segment.movie_start),
            "movie_end": float(segment.movie_end),
        }
        for segment in eligible
        if (
            str(segment.id) in risky_ids
            or bool(segment.review_required)
            or str(getattr(segment, "match_type", "")) != "exact"
            or float(segment.match_confidence or 0.0) < 0.82
            or "Repaired" in str(segment.match_reason or "")
            or "DINO" in str(segment.match_reason or "")
        )
    ]
    if not verify_payload:
        return 0

    try:
        current_scores = await verifier(str(project.narration_path), verify_payload)
    except Exception as exc:
        logger.debug("Segment visual mismatch verification failed project={}: {}", project_id, exc)
        return 0

    priorities: list[tuple[float, Segment]] = []
    for segment in eligible:
        score_info = current_scores.get(str(segment.id))
        if not score_info:
            continue
        score = float(score_info.get("verification_score", 0.0))
        edge_score = float(score_info.get("edge_score", 0.0))
        confidence = float(segment.match_confidence or 0.0)
        risky = str(segment.id) in risky_ids
        weak_segment = bool(segment.review_required) or str(getattr(segment, "match_type", "")) != "exact"
        target = (
            score < 0.58
            or (score < 0.68 and edge_score < 0.82)
            or (score < 0.74 and (confidence < 0.84 or risky or weak_segment))
        )
        if not target:
            continue
        priority = (0.76 - score) * 4.0 + max(0.0, 0.86 - confidence)
        if risky:
            priority += 0.7
        if weak_segment:
            priority += 0.5
        priorities.append((priority, segment))

    if not priorities:
        return 0

    targets = [
        segment
        for _, segment in sorted(priorities, key=lambda item: item[0], reverse=True)[: max(1, int(max_targets))]
    ]
    semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

    async def propose(segment: Segment) -> Optional[tuple[Segment, dict]]:
        async with semaphore:
            try:
                result = await frame_matcher.match_segment(
                    str(project.narration_path),
                    float(segment.narration_start),
                    float(segment.narration_end),
                    time_hint=float(segment.movie_start or 0.0),
                    relaxed=True,
                    strict_window=True,
                )
            except Exception as exc:
                logger.debug("Segment visual rematch failed project={} segment={}: {}", project_id, segment.id, exc)
                return None
            if not result or not result.get("success"):
                return None
            start = float(result.get("start", 0.0))
            end = float(result.get("end", start + _segment_duration(segment)))
            if movie_duration > 0.0 and end > movie_duration:
                duration = max(0.05, end - start)
                end = movie_duration
                start = max(0.0, end - duration)
                result = dict(result)
                result["start"] = start
                result["end"] = end
            if abs(start - float(segment.movie_start or 0.0)) < 0.08 and abs(end - float(segment.movie_end or end)) < 0.08:
                return None
            return segment, result

    proposals = [item for item in await asyncio.gather(*(propose(segment) for segment in targets)) if item]
    if not proposals:
        return 0

    candidate_payload = [
        {
            "id": str(segment.id),
            "narration_start": float(segment.narration_start),
            "narration_end": float(segment.narration_end),
            "movie_start": float(result["start"]),
            "movie_end": float(result["end"]),
        }
        for segment, result in proposals
    ]
    try:
        candidate_scores = await verifier(str(project.narration_path), candidate_payload)
    except Exception as exc:
        logger.debug("Segment visual candidate verification failed project={}: {}", project_id, exc)
        return 0

    repaired = 0
    for segment, result in proposals:
        current_info = current_scores.get(str(segment.id), {})
        candidate_info = candidate_scores.get(str(segment.id), {})
        current_score = float(current_info.get("verification_score", 0.0))
        candidate_score = float(candidate_info.get("verification_score", 0.0))
        confidence = float(result.get("confidence", 0.0))
        source_shift = abs(float(result["start"]) - float(segment.movie_start or 0.0))
        weak_segment = bool(segment.review_required) or str(getattr(segment, "match_type", "")) != "exact"

        if candidate_score < 0.62:
            continue
        required_gain = 0.045 if weak_segment else 0.080
        if current_score < 0.54:
            required_gain = min(required_gain, 0.035)
        if candidate_score < current_score + required_gain and not (current_score < 0.50 and candidate_score >= 0.70):
            continue
        if source_shift > 12.0 and candidate_score < max(0.76, current_score + 0.14):
            continue
        if source_shift > 30.0 and not weak_segment:
            continue
        if source_shift > 45.0 and candidate_score < max(0.82, current_score + 0.18):
            continue

        segment.movie_start = float(result["start"])
        segment.movie_end = float(result["end"])
        blended_confidence = confidence * 0.68 + candidate_score * 0.32
        segment.match_confidence = max(float(segment.match_confidence or 0.0), min(0.94, blended_confidence))
        segment.visual_confidence = max(float(segment.visual_confidence or 0.0), min(0.94, blended_confidence))
        segment.stability_score = max(float(segment.stability_score or 0.0), float(result.get("stability_score", 0.0)))
        segment.duration_gap = abs(_segment_duration(segment) - max(0.0, float(segment.movie_end) - float(segment.movie_start)))
        segment.estimated_boundary_error = min(0.35, max(0.0, segment.duration_gap / 2.0))
        segment.match_reason = (
            f"{segment.match_reason}; repaired segment visual phase"
            if segment.match_reason
            else "Repaired segment visual phase"
        )
        segment.evidence_summary = (
            f"segment_verify={candidate_score:.2f}, before={current_score:.2f}, "
            f"shift={source_shift:.2f}s"
        )
        if candidate_score >= 0.72 and segment.match_confidence >= 0.80:
            segment.review_required = False
            segment.alignment_status = AlignmentStatus.AUTO_ACCEPTED
            segment.match_type = "exact"
        else:
            segment.review_required = True
            segment.alignment_status = AlignmentStatus.NEEDS_REVIEW
            segment.match_type = "inferred"
        repaired += 1

    return repaired


async def _refine_segment_direct_phase_offsets(
    project_id: str,
    project,
    active_segments: list[Segment],
    movie_duration: float,
    *,
    max_targets: int = 24,
) -> int:
    """Shift low-confidence short segments by a small local offset using frames.

    This is deliberately not a retrieval pass. It only tests nearby source
    offsets, so it cannot jump to a different scene; it fixes the common case
    where the right shot was found but the action phase is off by ~0.5-1.0s.
    """
    if not active_segments or not getattr(project, "narration_path", None) or not getattr(project, "movie_path", None):
        return 0

    targets: list[tuple[float, Segment]] = []
    for segment in active_segments:
        if (
            segment.skip_matching
            or segment.segment_type == SegmentType.NON_MOVIE
            or segment.movie_start is None
            or segment.movie_end is None
        ):
            continue
        duration = _segment_duration(segment)
        source_duration = max(0.0, float(segment.movie_end) - float(segment.movie_start))
        confidence = float(segment.match_confidence or 0.0)
        candidate = _selected_candidate_for_segment(segment)
        verification_score = float(candidate.verification_score or 0.0) if candidate is not None else 0.0
        if duration < 0.50 or duration > 3.20 or source_duration <= 0.05:
            continue
        if confidence >= 0.88 and verification_score >= 0.82 and "Repaired" not in str(segment.match_reason or ""):
            continue
        priority = max(0.0, 0.88 - confidence) + (0.25 if duration <= 1.50 else 0.0)
        if verification_score <= 0.0:
            priority += 0.55
        elif verification_score < 0.72:
            priority += 0.35
        targets.append((priority, segment))

    if not targets:
        return 0

    payload = [
        {
            "id": str(segment.id),
            "narration_start": float(segment.narration_start),
            "narration_end": float(segment.narration_end),
            "movie_start": float(segment.movie_start),
            "movie_end": float(segment.movie_end),
            "confidence": float(segment.match_confidence or 0.0),
        }
        for _, segment in sorted(targets, key=lambda item: item[0], reverse=True)[: max(1, int(max_targets))]
    ]

    def _run() -> dict[str, dict]:
        import cv2

        narration_capture = cv2.VideoCapture(str(project.narration_path))
        movie_capture = cv2.VideoCapture(str(project.movie_path))
        if not narration_capture.isOpened() or not movie_capture.isOpened():
            narration_capture.release()
            movie_capture.release()
            return {}

        cache: dict[tuple[str, float], np.ndarray] = {}

        def read_gray(capture, role: str, timestamp: float) -> Optional[np.ndarray]:
            key = (role, round(max(0.0, float(timestamp)), 2))
            cached = cache.get(key)
            if cached is not None:
                return cached
            capture.set(cv2.CAP_PROP_POS_MSEC, key[1] * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                return None
            frame = frame[: max(1, int(height * 0.78)), int(width * 0.04) : int(width * 0.96)]
            resized_height = max(36, int(frame.shape[0] * 192 / max(1, frame.shape[1])))
            frame = cv2.resize(frame, (192, resized_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            cache[key] = gray
            if len(cache) > 768:
                cache.pop(next(iter(cache)))
            return gray

        def score_pair(query: Optional[np.ndarray], movie: Optional[np.ndarray]) -> float:
            if query is None or movie is None:
                return 0.0
            if query.shape != movie.shape:
                movie = cv2.resize(movie, (query.shape[1], query.shape[0]), interpolation=cv2.INTER_AREA)
            diff_score = 1.0 - float(np.mean(np.abs(query.astype(np.float32) - movie.astype(np.float32)))) / 255.0
            query_edges = cv2.Canny(query, 80, 160)
            movie_edges = cv2.Canny(movie, 80, 160)
            edge_score = 1.0 - float(
                np.mean(np.abs(query_edges.astype(np.float32) - movie_edges.astype(np.float32)))
            ) / 255.0
            return max(0.0, min(1.0, diff_score * 0.62 + edge_score * 0.38))

        def sample_positions(duration: float) -> tuple[float, ...]:
            if duration < 0.80:
                return (0.50,)
            if duration < 1.60:
                return (0.35, 0.70)
            return (0.25, 0.55, 0.82)

        def score_segment(item: dict, candidate_start: float, source_duration: float) -> float:
            duration = max(0.05, float(item["narration_end"]) - float(item["narration_start"]))
            scores: list[float] = []
            for pos in sample_positions(duration):
                query_time = float(item["narration_start"]) + duration * pos
                movie_time = candidate_start + source_duration * pos
                scores.append(
                    score_pair(
                        read_gray(narration_capture, "n", query_time),
                        read_gray(movie_capture, "m", movie_time),
                    )
                )
            if not scores:
                return 0.0
            return float(np.mean(scores) * 0.75 + min(scores) * 0.25)

        repairs: dict[str, dict] = {}
        try:
            offsets = (-1.40, -1.05, -0.70, -0.35, -0.18, 0.18, 0.35, 0.70, 1.05, 1.40)
            for item in payload:
                current_start = max(0.0, float(item["movie_start"]))
                source_duration = max(0.05, float(item["movie_end"]) - current_start)
                current_score = score_segment(item, current_start, source_duration)
                best_start = current_start
                best_score = current_score
                for offset in offsets:
                    candidate_start = max(0.0, current_start + offset)
                    if movie_duration > 0.0 and candidate_start + source_duration > movie_duration:
                        continue
                    score = score_segment(item, candidate_start, source_duration)
                    if score > best_score:
                        best_score = score
                        best_start = candidate_start
                required_gain = 0.050 if current_score >= 0.70 else 0.038
                if abs(best_start - current_start) < 0.08:
                    continue
                if best_score < 0.70 or best_score < current_score + required_gain:
                    continue
                repairs[str(item["id"])] = {
                    "start": best_start,
                    "end": best_start + source_duration,
                    "score": best_score,
                    "before": current_score,
                }
        finally:
            narration_capture.release()
            movie_capture.release()
        return repairs

    try:
        repairs = await asyncio.to_thread(_run)
    except Exception as exc:
        logger.debug("Direct phase refinement failed project={}: {}", project_id, exc)
        return 0
    if not repairs:
        return 0

    repaired = 0
    for segment in active_segments:
        repair = repairs.get(str(segment.id))
        if not repair:
            continue
        segment.movie_start = float(repair["start"])
        segment.movie_end = float(repair["end"])
        score = float(repair["score"])
        before = float(repair["before"])
        segment.match_confidence = max(float(segment.match_confidence or 0.0), min(0.91, score))
        segment.visual_confidence = max(float(segment.visual_confidence or 0.0), min(0.91, score))
        segment.match_reason = (
            f"{segment.match_reason}; refined direct phase"
            if segment.match_reason
            else "Refined direct phase"
        )
        segment.evidence_summary = f"direct_phase={score:.2f}, before={before:.2f}"
        if score >= 0.76 and segment.match_confidence >= 0.78:
            segment.review_required = False
            segment.alignment_status = AlignmentStatus.AUTO_ACCEPTED
            segment.match_type = "exact"
        repaired += 1

    return repaired


async def _apply_dino_visual_corrections(
    project_id: str,
    dino_matcher,
    project,
    chunks: list[dict],
    results: list[Optional[dict]],
    cache_dir: Path,
    narration_name: str,
    dino_cache_tag: str,
    max_concurrent: int,
    ) -> int:
    if dino_matcher is None or not chunks:
        return 0
    movie_duration = float(project.movie_duration or 0.0)

    target_indices = [
        index
        for index, result in enumerate(results)
        if (
            result is None
            or float(result.get("confidence", 0.0)) < 0.72
            or (
                float(result.get("confidence", 0.0)) < 0.78
                and float(result.get("rank_gap", 1.0)) < 0.015
            )
            or (
                float(result.get("confidence", 0.0)) < 0.84
                and float(result.get("rank_gap", 1.0)) < 0.030
            )
        )
    ]
    if not target_indices:
        return 0
    target_index_set = set(target_indices)

    await update_progress(project_id, "matching", 72, "Preparing DINO correction features...")
    loop = asyncio.get_running_loop()
    chunk_tasks = [
        {"id": chunk["id"], "start": float(chunk["start"]), "end": float(chunk["end"])}
        for index, chunk in enumerate(chunks)
        if index in target_index_set
    ]
    feature_cache_path = cache_dir / f"{narration_name}_dino{dino_cache_tag}_visual_chunks_v2.pkl"

    def on_feature_progress(done: int, total: int) -> None:
        mapped = 72 + int(5 * done / max(1, total))
        asyncio.run_coroutine_threadsafe(
            update_progress(project_id, "matching", mapped, f"Preparing DINO correction features {done}/{total}"),
            loop,
        )

    try:
        feature_map = await dino_matcher.precompute_segment_features_batch(
            project.narration_path,
            chunk_tasks,
            progress_callback=on_feature_progress,
            cache_path=feature_cache_path,
        )
    except Exception as exc:
        logger.warning("DINO correction feature preparation failed project={}: {}", project_id, exc)
        return 0

    await update_progress(project_id, "matching", 77, "Running DINO correction pass...")
    dino_results: list[Optional[dict]] = [None] * len(chunks)
    semaphore = asyncio.Semaphore(max(1, min(max_concurrent, 4)))
    completed = 0

    async def match_one(index: int, chunk: dict) -> None:
        nonlocal completed
        async with semaphore:
            try:
                candidate_getter = getattr(dino_matcher, "match_segment_candidates", None)
                if candidate_getter is not None:
                    candidates = await candidate_getter(
                        project.narration_path,
                        float(chunk["start"]),
                        float(chunk["end"]),
                        time_hint=None,
                        relaxed=False,
                        strict_window=False,
                        precomputed_features=feature_map.get(chunk["id"]),
                        limit=8,
                    )
                    result = dict(candidates[0]) if candidates else None
                    if result is not None:
                        result["_dino_candidates"] = candidates
                else:
                    result = await dino_matcher.match_segment(
                        project.narration_path,
                        float(chunk["start"]),
                        float(chunk["end"]),
                        time_hint=None,
                        relaxed=False,
                        strict_window=False,
                        precomputed_features=feature_map.get(chunk["id"]),
                    )
            except Exception as exc:
                logger.debug("DINO correction failed for chunk {}: {}", chunk.get("id"), exc)
                result = None
            dino_results[index] = result if result and result.get("success") else None
            completed += 1
            if completed % 10 == 0 or completed == len(target_indices):
                await update_progress(
                    project_id,
                    "matching",
                    77 + int(10 * completed / max(1, len(target_indices))),
                    f"DINO corrected chunks {completed}/{len(target_indices)}",
                )

    await asyncio.gather(*(match_one(index, chunks[index]) for index in target_indices))

    replaced = 0
    previous_target_end: Optional[float] = None
    previous_source_end: Optional[float] = None
    identity_scorer = getattr(dino_matcher, "score_precomputed_segment_identity_at", None)

    for index, chunk in enumerate(chunks):
        chunk_start = float(chunk["start"])
        chunk_end = float(chunk["end"])
        chunk_duration = max(0.2, chunk_end - chunk_start)
        if bool(chunk.get("boundary_break", False)):
            previous_target_end = None
            previous_source_end = None
        current = results[index]
        dino = dino_results[index]
        features = feature_map.get(str(chunk["id"]), [])
        if current and dino and dino.get("_dino_candidates"):
            current["_dino_candidates"] = dino.get("_dino_candidates")
        if dino and current:
            target_gap = 0.0 if previous_target_end is None else chunk_start - previous_target_end
            current_conf = float(current.get("confidence", 0.0))
            current_start = float(current.get("start", -999999.0))
            dino_conf = float(dino.get("confidence", 0.0))
            dino_gap = float(dino.get("rank_gap", 0.0))
            current_identity_score = 0.0
            dino_identity_score = float(dino.get("identity_score", 0.0))
            if identity_scorer is not None and features:
                try:
                    current_identity = identity_scorer(features, current_start)
                    dino_identity = identity_scorer(features, float(dino["start"]))
                    current_identity_score = float(current_identity.get("identity_score", 0.0))
                    dino_identity_score = float(dino_identity.get("identity_score", dino_identity_score))
                    current["identity_score"] = current_identity_score
                    current["identity_similarity"] = float(current_identity.get("identity_similarity", 0.0))
                    dino["identity_score"] = dino_identity_score
                    dino["identity_similarity"] = float(dino_identity.get("identity_similarity", 0.0))
                except Exception as exc:
                    logger.debug("DINO identity verification failed for chunk {}: {}", chunk.get("id"), exc)
            exact_mismatch = False
            dino_continuous = False
            dino_breaks_continuity = False
            current_continuity_error = 0.0
            dino_continuity_error = 0.0
            if previous_source_end is not None and -0.05 <= target_gap <= 0.50:
                expected_gap = max(0.0, target_gap)
                current_gap = current_start - previous_source_end
                dino_source_gap = float(dino["start"]) - previous_source_end
                current_continuity_error = abs(current_gap - expected_gap)
                dino_continuity_error = abs(dino_source_gap - expected_gap)
                exact_mismatch = current_continuity_error > max(2.0, chunk_duration * 3.0)
                dino_continuous = dino_continuity_error <= max(2.5, chunk_duration * 3.0)
                dino_breaks_continuity = dino_continuity_error > max(2.5, chunk_duration * 3.2)

            continuity_improves = dino_continuity_error + max(0.80, chunk_duration * 1.5) < current_continuity_error

            strong_dino = dino_conf >= 0.90 and dino_gap >= 0.060
            sequence_dino = dino_continuous and dino_conf >= 0.78 and exact_mismatch and current_conf < 0.74
            confidence_dino = strong_dino and current_conf < 0.72 and not dino_breaks_continuity
            identity_dino = (
                dino_identity_score >= 0.86
                and (
                    current_identity_score > 0.0
                    and current_identity_score < 0.52
                    and dino_identity_score >= current_identity_score + 0.16
                )
                and dino_conf >= 0.84
                and (
                    not dino_breaks_continuity
                    or (
                        dino_identity_score >= 0.92
                        and dino_conf >= 0.92
                        and dino_gap >= 0.08
                        and current_identity_score < 0.50
                    )
                )
            )
            anchor_identity_dino = (
                dino_identity_score >= 0.985
                and dino_conf >= 0.90
                and current_identity_score > 0.0
                and dino_identity_score >= current_identity_score + 0.08
                and (
                    not dino_breaks_continuity
                    or continuity_improves
                    or current_conf < 0.78
                )
                and (
                    current_conf < 0.88
                    or current_identity_score < 0.90
                )
            )
            if confidence_dino or sequence_dino or identity_dino or anchor_identity_dino:
                replacement = dict(dino)
                replacement["dino_corrected"] = True
                if identity_dino or anchor_identity_dino:
                    replacement["identity_corrected"] = True
                results[index] = replacement
                current = replacement
                replaced += 1
        elif dino and current is None and identity_scorer is not None and features:
            try:
                dino_identity = identity_scorer(features, float(dino["start"]))
                dino_identity_score = float(dino_identity.get("identity_score", 0.0))
            except Exception:
                dino_identity_score = float(dino.get("identity_score", 0.0))
            if dino_identity_score >= 0.82 and float(dino.get("confidence", 0.0)) >= 0.78:
                replacement = dict(dino)
                replacement["dino_corrected"] = True
                replacement["identity_corrected"] = True
                replacement["identity_score"] = dino_identity_score
                results[index] = replacement
                current = replacement
                replaced += 1

        if current and current.get("success"):
            previous_target_end = chunk_end
            previous_source_end = max(
                float(current["start"]) + 0.05,
                float(current.get("end", float(current["start"]) + chunk_duration)),
            )
        else:
            previous_target_end = None
            previous_source_end = None

    if replaced:
        logger.info("DINO correction replaced {} visual chunks for project={}", replaced, project_id)
    return replaced

async def _match_segments_with_visual_chunks(
    project_id: str,
    frame_matcher,
    project,
    active_segments: list[Segment],
    cache_dir: Path,
    narration_name: str,
    cache_suffix: str,
    cache_tag: str,
    max_concurrent: int,
    dino_matcher=None,
    dino_cache_tag: str = "",
) -> tuple[int, int]:
    boundary_break_ids = _detect_visual_boundary_breaks(project, active_segments)
    chunks = _build_visual_match_chunks(active_segments, boundary_break_ids=boundary_break_ids)
    if len(chunks) < 8:
        return 0, 0

    await update_progress(project_id, "matching", 16, f"Preparing visual chunk cache ({len(chunks)} chunks)...")
    loop = asyncio.get_running_loop()
    chunk_tasks = [
        {"id": chunk["id"], "start": float(chunk["start"]), "end": float(chunk["end"])}
        for chunk in chunks
    ]
    chunk_cache_path = cache_dir / f"{narration_name}_{cache_suffix}{cache_tag}_visual_chunks.pkl"

    def on_chunk_feature_progress(done: int, total: int) -> None:
        mapped = 16 + int(8 * done / max(1, total))
        asyncio.run_coroutine_threadsafe(
            update_progress(project_id, "matching", mapped, f"Preparing visual chunk features {done}/{total}"),
            loop,
        )

    feature_map = await frame_matcher.precompute_segment_features_batch(
        project.narration_path,
        chunk_tasks,
        progress_callback=on_chunk_feature_progress,
        cache_path=chunk_cache_path,
    )

    await update_progress(project_id, "matching", 24, f"Matching visual chunks {len(chunks)}...")
    semaphore = asyncio.Semaphore(max(1, min(max_concurrent, 6)))
    results: list[Optional[dict]] = [None] * len(chunks)
    completed = 0

    async def match_chunk(index: int, chunk: dict) -> None:
        nonlocal completed
        async with semaphore:
            result = await frame_matcher.match_segment(
                project.narration_path,
                float(chunk["start"]),
                float(chunk["end"]),
                time_hint=None,
                relaxed=False,
                strict_window=False,
                precomputed_features=feature_map.get(chunk["id"]),
            )
            if result and float(result.get("confidence", 0.0)) < 0.58:
                result = None
            results[index] = result if result and result.get("success") else None
            completed += 1
            if completed % 5 == 0 or completed == len(chunks):
                await update_progress(
                    project_id,
                    "matching",
                    24 + int(48 * completed / max(1, len(chunks))),
                    f"Matched visual chunks {completed}/{len(chunks)}",
                )

    await asyncio.gather(*(match_chunk(index, chunk) for index, chunk in enumerate(chunks)))

    movie_duration = float(project.movie_duration or 0.0)
    dino_replaced = await _apply_dino_visual_corrections(
        project_id,
        dino_matcher,
        project,
        chunks,
        results,
        cache_dir,
        narration_name,
        dino_cache_tag,
        max_concurrent,
    )
    offset_refined = _refine_visual_chunk_source_offsets(
        frame_matcher,
        chunks,
        results,
        feature_map,
        movie_duration,
    )
    repaired_chunks, filled_chunks = _repair_visual_chunk_continuity(
        frame_matcher,
        chunks,
        results,
        feature_map,
        movie_duration,
    )
    island_repaired = _repair_visual_chunk_source_islands(
        frame_matcher,
        chunks,
        results,
        feature_map,
        movie_duration,
    )
    # Keep this as an opt-in repair only. On narration videos with real cutaway
    # inserts, forcing a weak chunk between local neighbors can make a correct
    # far jump worse.
    bridge_repaired = 0
    if island_repaired:
        repaired_again, filled_again = _repair_visual_chunk_continuity(
            frame_matcher,
            chunks,
            results,
            feature_map,
            movie_duration,
        )
        repaired_chunks += repaired_again
        filled_chunks += filled_again
    # Disabled as a default pass: wide speed/phase retiming fixed a few clips
    # but lowered the aggregate visual audit on this project by accepting
    # semantically similar wrong phases. Keep the implementation available for
    # a future opt-in "slow repair" mode, but do not mutate the main pipeline.
    warp_refined = 0
    phase_refined = _refine_visual_chunk_phase_offsets(
        project,
        chunks,
        results,
        movie_duration,
    )
    applied = 0
    accepted_chunks = 0
    inferred_chunks = 0

    for chunk, result in zip(chunks, results):
        chunk_duration = max(0.0, float(chunk["end"]) - float(chunk["start"]))
        inferred = False
        if result is not None:
            source_start = float(result["start"])
            source_end = max(source_start + 0.05, float(result.get("end", source_start + chunk_duration)))
            confidence = float(result.get("confidence", 0.0))
            inferred = bool(result.get("inferred_timeline", False))
            visual_confidence = 0.0 if inferred else confidence
            stability = float(result.get("stability_score", 0.0))
            accepted_chunks += 1
        else:
            continue

        source_duration_for_apply = max(0.05, source_end - source_start)
        if movie_duration > 0.0 and source_end > movie_duration:
            source_end = movie_duration
            source_start = max(0.0, source_end - source_duration_for_apply)
        source_duration_for_apply = max(0.05, source_end - source_start)
        source_speed_ratio = float(
            result.get("source_speed_ratio", source_duration_for_apply / max(chunk_duration, 0.05))
            or 1.0
        )
        speed_change_confidence = float(result.get("visual_warp_score", confidence) or 0.0)
        speed_changed = (
            bool(result.get("visual_warp_refined", False))
            and abs(source_speed_ratio - 1.0) >= 0.08
            and speed_change_confidence >= 0.70
        )
        reason = (
            "Matched by continuous visual chunk"
            if not inferred
            else "Filled from previous visual chunk to keep playback continuous"
        )
        if bool(result.get("dino_corrected", False)):
            reason = "Corrected by DINO visual retrieval"
        if bool(result.get("identity_corrected", False)):
            reason = "Corrected by DINO identity verification"
        if bool(result.get("source_offset_refined", False)):
            reason += "; refined source offset"
        if bool(result.get("phase_offset_refined", False)):
            reason += "; refined visual phase"
        if bool(result.get("source_island_repaired", False)):
            reason = "Repaired isolated source jump by neighbor continuity"
        if bool(result.get("neighbor_bridge_repaired", False)):
            reason = "Repaired low-confidence source jump between neighbor anchors"
        if bool(result.get("visual_warp_refined", False)):
            if speed_changed:
                reason += f"; refit visual speed/phase ratio={source_speed_ratio:.3f}"
            else:
                reason += "; refit visual phase"
        elif bool(result.get("continuity_repaired", False)):
            reason = "Repaired to previous visual chunk continuity"
        elif bool(result.get("continuity_filled", False)):
            reason = "Filled by previous visual chunk continuity"
        applied += _apply_visual_chunk_result(
            project,
            chunk,
            source_start,
            confidence,
            visual_confidence,
            stability,
            inferred,
            reason,
            chunk_source_duration=max(0.05, source_end - source_start),
            speed_changed=speed_changed,
            source_speed_ratio=source_speed_ratio,
            speed_change_confidence=speed_change_confidence,
        )

    # Disabled as a default pass. It repairs a few isolated segments, but on the
    # current Shawshank test it added several minutes for only ~0.1% audit gain.
    # Keep it available for a future explicit "deep repair" button.
    segment_repaired = await _repair_segment_visual_mismatches(
        project_id,
        frame_matcher,
        project,
        active_segments,
        movie_duration,
        max_targets=min(max(36, int(len(active_segments) * 0.06)), 72),
        max_concurrent=2,
    )
    await update_progress(project_id, "matching", 88, "Refining short segment phases...")
    direct_phase_refined = await _refine_segment_direct_phase_offsets(
        project_id,
        project,
        active_segments,
        movie_duration,
        max_targets=min(max(120, int(len(active_segments) * 0.30)), 260),
    )

    logger.info(
        "visual chunk matching project={} chunks={} visual_breaks={} accepted={} inferred={} dino_replaced={} offset_refined={} repaired={} island_repaired={} bridge_repaired={} filled={} warp_refined={} phase_refined={} segment_repaired={} direct_phase_refined={} applied_segments={}",
        project_id,
        len(chunks),
        len(boundary_break_ids),
        accepted_chunks,
        inferred_chunks,
        dino_replaced,
        offset_refined,
        repaired_chunks,
        island_repaired,
        bridge_repaired,
        filled_chunks,
        warp_refined,
        phase_refined,
        segment_repaired,
        direct_phase_refined,
        applied,
    )
    return applied, len(chunks)


def _selected_candidate_for_segment(segment: Segment) -> Optional[MatchCandidate]:
    if not segment.match_candidates:
        return None
    if segment.selected_candidate_id:
        for candidate in segment.match_candidates:
            if candidate.id == segment.selected_candidate_id:
                return candidate
    if segment.movie_start is not None and segment.movie_end is not None:
        for candidate in segment.match_candidates:
            if abs(candidate.start - segment.movie_start) <= 0.08 and abs(candidate.end - segment.movie_end) <= 0.08:
                return candidate
    return sorted(segment.match_candidates, key=lambda item: item.score or item.confidence or 0.0, reverse=True)[0]


def _segment_validation_priority(segment: Segment) -> float:
    candidate = _selected_candidate_for_segment(segment)
    priority = 0.0
    if segment.alignment_status == AlignmentStatus.AUTO_ACCEPTED:
        priority += 2.0
    if segment.match_type == "exact":
        priority += 1.2
    if float(segment.match_confidence or 0.0) >= 0.96:
        priority += 0.8
    if candidate is None:
        return priority
    if float(candidate.verification_score or 0.0) <= 0.0:
        priority += 2.0
    if float(candidate.temporal_confidence or 0.0) < 0.35:
        priority += 0.8
    if float(candidate.rank_gap or 0.0) < 0.02:
        priority += 0.6
    if float(candidate.stability_score or 0.0) < 0.55:
        priority += 0.5
    if _segment_duration(segment) <= 2.0:
        priority += 0.3
    return priority


def _needs_visual_validation(segment: Segment) -> bool:
    if segment.skip_matching or segment.is_manual_match or segment.segment_type == SegmentType.NON_MOVIE:
        return False
    if segment.movie_start is None or segment.movie_end is None or segment.movie_end <= segment.movie_start:
        return False
    if segment.alignment_status not in {
        AlignmentStatus.AUTO_ACCEPTED,
        AlignmentStatus.REMATCHED,
        AlignmentStatus.NEEDS_REVIEW,
    } and segment.match_type != "exact":
        return False

    candidate = _selected_candidate_for_segment(segment)
    if candidate is None:
        return segment.alignment_status == AlignmentStatus.AUTO_ACCEPTED

    if (
        candidate.source == "shot_continuity"
        and float(candidate.temporal_confidence or 0.0) >= 0.85
        and float(candidate.confidence or 0.0) >= 0.70
    ):
        return False

    verification_score = float(candidate.verification_score or 0.0)
    if verification_score >= 0.85:
        return False
    return True


def _validation_passes(segment: Segment, metrics: dict) -> bool:
    score = float(metrics.get("verification_score", 0.0))
    inliers = int(metrics.get("geometric_inliers", 0))
    inlier_ratio = float(metrics.get("geometric_inlier_ratio", 0.0))
    hist_score = float(metrics.get("hist_score", 0.0))
    edge_score = float(metrics.get("edge_score", 0.0))
    duration = _segment_duration(segment)

    if score >= 0.85:
        return True
    if duration <= 1.35 and score >= 0.82 and hist_score >= 0.86 and edge_score >= 0.86:
        return True
    return False


def _apply_validation_metrics(segment: Segment, metrics: dict) -> None:
    candidate = _selected_candidate_for_segment(segment)
    score = float(metrics.get("verification_score", 0.0))
    inliers = int(metrics.get("geometric_inliers", 0))
    inlier_ratio = float(metrics.get("geometric_inlier_ratio", 0.0))
    if candidate is not None:
        candidate.verification_score = score
        candidate.geometric_inliers = inliers
        candidate.geometric_inlier_ratio = inlier_ratio
        if "post_verify=" not in candidate.reason:
            candidate.reason += f"; post_verify={score:.2f}, inliers={inliers}, ratio={inlier_ratio:.2f}"
    segment.evidence_summary = (
        f"{segment.evidence_summary}; post_verify={score:.2f}, "
        f"inliers={inliers}, ratio={inlier_ratio:.2f}"
    ).lstrip("; ")


def _demote_failed_visual_validation(segment: Segment, metrics: dict) -> None:
    score = float(metrics.get("verification_score", 0.0))
    inliers = int(metrics.get("geometric_inliers", 0))
    inlier_ratio = float(metrics.get("geometric_inlier_ratio", 0.0))
    candidate = _selected_candidate_for_segment(segment)
    if candidate is not None:
        candidate.score = min(float(candidate.score or 0.0), max(0.0, score))
        candidate.confidence = min(float(candidate.confidence or 0.0), max(0.0, score))
        if "validation_demoted" not in candidate.reason:
            candidate.reason += "; validation_demoted"
    segment.alignment_status = AlignmentStatus.NEEDS_REVIEW
    segment.review_required = True
    segment.match_type = "inferred"
    segment.match_confidence = min(float(segment.match_confidence or 0.0), max(0.0, score))
    segment.visual_confidence = min(float(segment.visual_confidence or 0.0), max(0.0, score))
    segment.estimated_boundary_error = max(float(segment.estimated_boundary_error or 0.0), 1.0)
    segment.match_reason = (
        f"{segment.match_reason}; visual validation failed "
        f"(score={score:.2f}, inliers={inliers}, ratio={inlier_ratio:.2f})"
    )


async def _post_validate_matches(
    project_id: str,
    frame_matcher,
    project,
    segments: list[Segment],
    *,
    stage: str,
    progress_start: int,
    progress_width: int,
    max_targets: Optional[int] = None,
) -> int:
    if not hasattr(frame_matcher, "verify_segment_matches"):
        return 0
    if max_targets is not None and max_targets <= 0:
        return 0

    targets = [segment for segment in segments if _needs_visual_validation(segment)]
    if not targets:
        return 0
    targets.sort(key=_segment_validation_priority, reverse=True)
    if max_targets is not None:
        targets = targets[:max_targets]

    await update_progress(
        project_id,
        "matching",
        progress_start,
        f"Visual verification {stage}: 0/{len(targets)}",
    )
    payload = [
        {
            "id": segment.id,
            "narration_start": segment.narration_start,
            "narration_end": segment.narration_end,
            "movie_start": segment.movie_start,
            "movie_end": segment.movie_end,
        }
        for segment in targets
    ]
    try:
        verification_results = await frame_matcher.verify_segment_matches(project.narration_path, payload)
    except Exception as exc:  # pragma: no cover
        logger.warning("Visual post-validation failed for project {}: {}", project_id, exc)
        return 0

    failed = 0
    for segment in targets:
        metrics = verification_results.get(segment.id)
        if not metrics:
            continue
        _apply_validation_metrics(segment, metrics)
        if not _validation_passes(segment, metrics):
            _demote_failed_visual_validation(segment, metrics)
            failed += 1

    await update_progress(
        project_id,
        "matching",
        min(99, progress_start + progress_width),
        f"Visual verification {stage}: failed {failed}/{len(targets)}",
    )
    if failed:
        logger.info(
            "Visual post-validation demoted matches project={} stage={} failed={}/{}",
            project_id,
            stage,
            failed,
            len(targets),
        )
    return failed


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
    from core.video_processor.dinov2_faiss_matcher import DinoFaissMatcher
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
        "DinoFaissMatcher": DinoFaissMatcher,
        "FrameMatcher": FrameMatcher,
        "NonMovieDetector": NonMovieDetector,
        "GlobalAlignmentOptimizer": GlobalAlignmentOptimizer,
    }


def _speech_cache_path(narration_path: str, app_settings) -> Optional[Path]:
    try:
        source = Path(narration_path)
        stat = source.stat()
        signature = {
            "version": 1,
            "path": str(source.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "model": app_settings.whisper.model,
            "device": app_settings.whisper.device,
            "language": app_settings.whisper.language,
            "word_timestamps": app_settings.whisper.word_timestamps,
        }
        digest = hashlib.sha1(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        cache_dir = Path(__file__).resolve().parents[2] / "temp" / "speech_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{digest}.json"
    except Exception as exc:
        logger.warning("Speech cache signature failed for {}: {}", narration_path, exc)
        return None


async def _load_or_transcribe_narration(project, app_settings) -> list[dict]:
    from core.audio_processor.speech_recognizer import SpeechRecognizer

    cache_path = _speech_cache_path(project.narration_path, app_settings)
    if cache_path and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("version") == 1 and isinstance(payload.get("transcription"), list):
                logger.info("Loaded speech recognition cache: {}", cache_path)
                return payload["transcription"]
        except Exception as exc:
            logger.warning("Failed to load speech cache {}: {}", cache_path, exc)

    recognizer = SpeechRecognizer(
        model_name=app_settings.whisper.model,
        device=app_settings.whisper.device,
        language=app_settings.whisper.language,
        word_timestamps=app_settings.whisper.word_timestamps,
    )
    transcription = await recognizer.transcribe(project.narration_path)
    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "transcription": transcription}, handle, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to write speech cache {}: {}", cache_path, exc)
    return transcription


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
        await update_progress(project_id, "recognizing", 0, "Running speech recognition...")
        transcription = await _load_or_transcribe_narration(project, app_settings)

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
    if len(target_segments) > 120:
        logger.info(
            "Skip non-movie detection for large project: {} target segments",
            len(target_segments),
        )
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
    precomputed_features: Optional[list[dict]] = None,
    prefer_sequential: bool = False,
    expected_movie_time_override: Optional[float] = None,
    aggressive: bool = False,
    extra_hints: Optional[list[float]] = None,
) -> list[MatchCandidate]:
    if segment.segment_type == SegmentType.NON_MOVIE or segment.skip_matching:
        return []

    expected_movie_time = (
        expected_movie_time_override
        if expected_movie_time_override is not None
        else _calculate_expected_movie_time(project, segment)
    )
    candidate_top_k = max(app_settings.match.candidate_top_k, 8) if aggressive else app_settings.match.candidate_top_k
    candidates: list[MatchCandidate] = []
    batch_direct = _can_use_batch_direct(
        segment,
        base_result,
        expected_movie_time=expected_movie_time,
        prefer_sequential=prefer_sequential,
        app_settings=app_settings,
    )
    if aggressive:
        batch_direct = False

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
        if batch_direct:
            direct_candidates = _apply_sequence_bias_to_candidates(candidates, prefer_sequential=prefer_sequential)
            return direct_candidates[:1]

    # 高置信度批次结果：仅在附近做少量精细搜索，跳过耗时的全局搜索
    batch_confidence = float(base_result.get("confidence", 0.0)) if base_result and base_result.get("success") else 0.0
    batch_stability = float(base_result.get("stability_score", 0.0)) if base_result and base_result.get("success") else 0.0
    batch_low_info = float(base_result.get("low_info_ratio", 1.0)) if base_result and base_result.get("success") else 1.0
    batch_temporal_confidence = 0.0
    if base_result and base_result.get("success"):
        movie_duration = max(0.0, float(base_result["end"]) - float(base_result["start"]))
        duration_gap = abs(movie_duration - _segment_duration(segment))
        center = (float(base_result["start"]) + float(base_result["end"])) / 2
        if expected_movie_time is None:
            batch_temporal_confidence = 1.0
        else:
            batch_temporal_confidence = max(
                0.0,
                1.0 - abs(center - expected_movie_time) / max(120.0, _segment_duration(segment) * 15.0, 1.0),
            )
    high_anchor = (
        batch_confidence >= max(0.78, app_settings.match.high_confidence_threshold - 0.04)
        and batch_stability >= 0.55
        and batch_low_info <= 0.45
    )
    skip_global_search = (
        high_anchor
        or (
            batch_confidence >= max(0.70, app_settings.match.medium_confidence_threshold)
            and batch_stability >= 0.42
            and batch_low_info <= 0.60
        )
    )
    if aggressive:
        skip_global_search = False
    strong_batch_lock = (
        prefer_sequential
        and base_result
        and base_result.get("success")
        and batch_confidence >= max(0.90, app_settings.match.high_confidence_threshold)
        and batch_stability >= 0.62
        and batch_low_info <= 0.35
        and batch_temporal_confidence >= 0.55
    )
    if strong_batch_lock:
        locked = _apply_sequence_bias_to_candidates(candidates, prefer_sequential=True)
        return locked[:1]

    hints: list[Optional[float]] = []
    if neighbor_hint is not None:
        hints.append(neighbor_hint)
    if expected_movie_time is not None:
        hints.append(expected_movie_time)
    if base_result and base_result.get("success"):
        hints.append((float(base_result["start"]) + float(base_result["end"])) / 2)
        if not prefer_sequential:
            seg_duration = _segment_duration(segment)
            hints.append(float(base_result["start"]) - seg_duration)
            hints.append(float(base_result["start"]) + seg_duration)
            hints.append(float(base_result["start"]) - 30.0)
            hints.append(float(base_result["start"]) + 30.0)
    if extra_hints:
        hints.extend(extra_hints)
    segment_duration = _segment_duration(segment)
    has_local_hints = any(hint is not None for hint in hints)
    if not skip_global_search and not prefer_sequential and (segment_duration > 1.6 or not has_local_hints):
        hints.append(None)
    if aggressive and None not in hints and (segment_duration > 2.0 or not has_local_hints):
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
    precomputed = precomputed_features
    if precomputed is None:
        precomputed = await frame_matcher._extract_segment_features(
            project.narration_path, segment.narration_start, segment.narration_end
        )

    max_hint_count = min(
        len(unique_hints),
        6
        if aggressive
        else 1
        if prefer_sequential and high_anchor
        else 2
        if prefer_sequential and skip_global_search
        else 3
        if prefer_sequential
        else 2
        if high_anchor
        else 4
        if skip_global_search
        else 4,
    )

    search_plan: list[tuple[Optional[float], bool]] = []
    for idx, hint in enumerate(unique_hints[:max_hint_count]):
        search_plan.append((hint, False))
        need_relaxed = aggressive or (not high_anchor and (batch_confidence < app_settings.match.medium_confidence_threshold or idx == 0))
        if prefer_sequential:
            need_relaxed = need_relaxed and idx == 0 and batch_confidence < app_settings.match.medium_confidence_threshold
            if aggressive:
                need_relaxed = True
        if need_relaxed:
            search_plan.append((hint, True))

    for hint, relaxed in search_plan:
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
        if len(candidates) >= max(3, candidate_top_k):
            break
        if len(candidates) >= max(3, candidate_top_k):
            break

    deduped = _dedupe_candidates(candidates)
    deduped = await _rescore_candidates_with_audio(audio_scorer, segment, deduped)
    deduped = await _verify_ambiguous_candidates(frame_matcher, project, segment, deduped)
    if aggressive:
        deduped = await _phase_lock_candidates(
            frame_matcher,
            project,
            segment,
            deduped,
            max_candidates=min(4, max(3, candidate_top_k)),
            precomputed_features=precomputed,
        )
    deduped = _apply_sequence_bias_to_candidates(deduped, prefer_sequential=prefer_sequential)
    return deduped[:candidate_top_k]


async def _collect_context_candidates_for_segment(
    frame_matcher,
    project,
    segments: list[Segment],
    idx: int,
    segment: Segment,
    app_settings,
    neighbor_hint: Optional[float],
    expected_movie_time: Optional[float],
    audio_scorer=None,
    context_window: Optional[tuple[float, float]] = None,
    precomputed_features: Optional[list[dict]] = None,
) -> list[MatchCandidate]:
    if context_window is None:
        min_duration, max_neighbors = _context_params_for_segment(segment, None)
        context_window = _build_context_window(segments, idx, min_duration=min_duration, max_neighbors=max_neighbors)
    if context_window is None:
        return []

    context_start, context_end = context_window
    offset_start = max(0.0, float(segment.narration_start) - context_start)
    seg_duration = _segment_duration(segment)
    precomputed = precomputed_features
    if precomputed is None:
        precomputed = await frame_matcher._extract_segment_features(project.narration_path, context_start, context_end)
    if not precomputed:
        return []

    hints = _collect_recheck_hints(segment, neighbor_hint)
    search_hints: list[Optional[float]] = []
    if neighbor_hint is not None:
        if expected_movie_time is not None:
            search_hints.append(max(0.0, float(expected_movie_time) - offset_start))
        search_hints.append(max(0.0, float(neighbor_hint) - offset_start))
    for hint in hints[:4]:
        shifted = max(0.0, float(hint) - offset_start)
        if not any(existing is not None and abs(existing - shifted) <= 6.0 for existing in search_hints):
            search_hints.append(shifted)
    if None not in search_hints:
        search_hints.append(None)

    candidates: list[MatchCandidate] = []
    for rank, hint in enumerate(search_hints[:6], start=1):
        result = await frame_matcher.match_segment(
            project.narration_path,
            context_start,
            context_end,
            time_hint=hint,
            relaxed=True,
            strict_window=hint is not None,
            precomputed_features=precomputed,
        )
        if not result:
            continue
        adjusted_result = dict(result)
        adjusted_start = float(result["start"]) + offset_start
        adjusted_result["start"] = adjusted_start
        adjusted_result["end"] = adjusted_start + seg_duration
        candidate = _candidate_from_result(
            segment,
            adjusted_result,
            rank=rank,
            source="context_refine",
            expected_movie_time=expected_movie_time,
            reason_prefix=f"Context-window match ({context_end - context_start:.1f}s)",
            stability_score=float(result.get("stability_score", 0.0)),
            candidate_quality=float(result.get("candidate_quality", 0.0)),
            query_quality=float(result.get("query_quality", 0.0)),
            low_info_ratio=float(result.get("low_info_ratio", 0.0)),
        )
        candidates.append(candidate)

    deduped = _dedupe_candidates(candidates)
    deduped = await _rescore_candidates_with_audio(audio_scorer, segment, deduped)
    deduped = await _verify_ambiguous_candidates(frame_matcher, project, segment, deduped)
    return deduped[:4]

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
    main_loop = asyncio.get_running_loop()
    deps = await _load_processing_dependencies()
    AudioSimilarityScorer = deps["AudioSimilarityScorer"]
    DinoFaissMatcher = deps["DinoFaissMatcher"]
    FrameMatcher = deps["FrameMatcher"]
    GlobalAlignmentOptimizer = deps["GlobalAlignmentOptimizer"]

    coalesced_segments, merged = _coalesce_visual_micro_segments(
        segments,
        preserve_manual_matches=preserve_manual_matches,
    )
    if merged > 0:
        project.segments = coalesced_segments
        segments = project.segments
        logger.info(
            "Coalesced legacy visual micro-segments project={} merged={} total={}",
            project_id,
            merged,
            len(segments),
        )

    if len(segments) >= 220:
        visual_cut_segments, visual_added = await _split_segments_on_visual_cuts(
            project,
            segments,
            preserve_manual_matches=preserve_manual_matches,
        )
        if visual_added > 0:
            project.segments = visual_cut_segments
            segments = project.segments
            logger.info(
                "Split narration on detected visual cuts project={} added={} total={}",
                project_id,
                visual_added,
                len(segments),
            )

    if len(segments) >= 220:
        split_segments, added = _split_segments_for_visual_matching(
            segments,
            preserve_manual_matches=preserve_manual_matches,
            max_duration=3.2,
        )
        if added > 0:
            project.segments = split_segments
            segments = project.segments
            logger.info(
                "Split long narration spans for visual matching project={} added={} total={}",
                project_id,
                added,
                len(segments),
            )

    active_segments: list[Segment] = []
    narration_duration_limit = float(project.narration_duration or 0.0)
    for segment in segments:
        if narration_duration_limit > 0.0:
            if float(segment.narration_start) >= narration_duration_limit:
                segment.use_segment = False
                _mark_segment_skipped(segment, "Segment starts after narration video duration")
                continue
            if float(segment.narration_end) > narration_duration_limit:
                segment.narration_end = narration_duration_limit
        if segment.skip_matching:
            _mark_segment_skipped(segment)
            continue
        if segment.segment_type == SegmentType.NO_NARRATION:
            segment.alignment_status = AlignmentStatus.PENDING
            segment.review_required = False
            segment.match_type = "inferred"
            segment.evidence_summary = "continuity_fill_pending"
            continue
        if preserve_manual_matches and segment.is_manual_match and segment.movie_start is not None and segment.movie_end is not None:
            segment.alignment_status = AlignmentStatus.MANUAL
            segment.review_required = False
            continue
        active_segments.append(segment)

    if not active_segments:
        await update_progress(project_id, "matching", 100, "No eligible segments need matching")
        return

    matcher_kwargs = dict(
        phash_threshold=app_settings.match.phash_threshold,
        match_threshold=app_settings.match.frame_match_threshold,
        use_deep_learning=app_settings.match.use_deep_learning,
        index_sample_fps=app_settings.match.index_sample_fps,
        fast_mode=app_settings.match.fast_mode,
        subtitle_mask_mode=getattr(project, "subtitle_mask_mode", "hybrid"),
        movie_subtitle_regions=[region.model_dump() for region in getattr(project, "movie_subtitle_regions", [])],
        narration_subtitle_regions=[region.model_dump() for region in getattr(project, "narration_subtitle_regions", [])],
    )
    use_dino = bool(getattr(app_settings.match, "use_dinov2_retrieval", False))
    prefer_exact_visual_chunks = len(segments) >= 220
    if prefer_exact_visual_chunks and use_dino:
        logger.info(
            "Using exact frame matcher as primary for long visual project={} segments={}",
            project_id,
            len(segments),
        )
        use_dino = False
    matcher_label = "movie frame index"
    cache_suffix = "frame"
    sample_interval = app_settings.match.sample_interval

    try:
        if use_dino:
            frame_matcher = DinoFaissMatcher(
                dino_model_name=app_settings.match.dino_model_name,
                dino_batch_size=app_settings.match.dino_batch_size,
                dino_top_k=app_settings.match.dino_top_k,
                dino_index_interval=app_settings.match.dino_index_interval,
                **matcher_kwargs,
            )
            matcher_label = "DINO movie index"
            cache_suffix = "dino"
            sample_interval = app_settings.match.dino_index_interval
        else:
            frame_matcher = FrameMatcher(**matcher_kwargs)
    except Exception as exc:
        logger.warning("Failed to initialize DINO matcher, falling back to legacy matcher: {}", exc)
        frame_matcher = FrameMatcher(**matcher_kwargs)
        use_dino = False

    dino_uses_cuda = bool(use_dino and getattr(frame_matcher, "_use_cuda", False))
    if dino_uses_cuda:
        if sample_interval > 1.0:
            sample_interval = 1.0
        if hasattr(frame_matcher, "dino_index_interval"):
            frame_matcher.dino_index_interval = min(float(getattr(frame_matcher, "dino_index_interval", sample_interval)), 1.0)
        if hasattr(frame_matcher, "dino_top_k"):
            frame_matcher.dino_top_k = max(int(getattr(frame_matcher, "dino_top_k", 64)), 64)
        logger.info(
            "Using high-quality CUDA DINO settings: interval={:.1f}s top_k={} project={}",
            sample_interval,
            getattr(frame_matcher, "dino_top_k", "n/a"),
            project_id,
        )
    elif use_dino and project.movie_duration and project.movie_duration >= 7200 and sample_interval < 3.0:
        sample_interval = 3.0
        logger.info(
            "Using relaxed DINO movie index interval {:.1f}s for long-movie project={}",
            sample_interval,
            project_id,
        )
    if use_dino and not getattr(frame_matcher, "_use_cuda", False) and project.movie_duration and project.movie_duration >= 3600:
        sample_interval = max(float(sample_interval), 5.0)
        if hasattr(frame_matcher, "dino_top_k"):
            frame_matcher.dino_top_k = min(int(frame_matcher.dino_top_k), 16)
        logger.warning(
            "DINO is running on CPU; using faster interval {:.1f}s/top_k={} for project={}",
            sample_interval,
            getattr(frame_matcher, "dino_top_k", "n/a"),
            project_id,
        )

    cache_dir = Path(__file__).resolve().parents[2] / "temp" / "match_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    movie_name = Path(project.movie_path).stem
    narration_name = Path(project.narration_path).stem
    model_tag = getattr(frame_matcher, "dino_model_name", "")
    model_tag = model_tag.replace("/", "_").replace(":", "_")
    cache_tag = f"_{model_tag}" if model_tag else ""
    frame_cache_path = cache_dir / f"{movie_name}_{cache_suffix}{cache_tag}.pkl"
    narration_feature_cache_path = cache_dir / f"{narration_name}_{cache_suffix}{cache_tag}_segment_features.pkl"
    await update_progress(project_id, "matching", 1, f"Preparing {matcher_label}...")

    last_index_progress = -1

    def on_index_progress(progress_pct: float, message: str):
        nonlocal last_index_progress
        mapped_progress = min(12.0, max(1.0, round(progress_pct * 12 / 100, 1)))
        if mapped_progress <= last_index_progress and progress_pct < 100:
            return
        last_index_progress = mapped_progress
        asyncio.run_coroutine_threadsafe(
            update_progress(project_id, "matching", mapped_progress, message),
            main_loop,
        )

    await frame_matcher.build_index(
        project.movie_path,
        sample_interval=sample_interval,
        cache_path=frame_cache_path,
        progress_callback=on_index_progress,
    )
    dino_corrector = None
    dino_cache_tag = ""
    if (
        not use_dino
        and len(active_segments) >= 220
        and bool(getattr(app_settings.match, "use_dinov2_retrieval", False))
    ):
        try:
            dino_corrector = DinoFaissMatcher(
                dino_model_name=app_settings.match.dino_model_name,
                dino_batch_size=app_settings.match.dino_batch_size,
                dino_top_k=max(64, int(getattr(app_settings.match, "dino_top_k", 64))),
                dino_index_interval=1.0,
                **matcher_kwargs,
            )
            dino_model_tag = app_settings.match.dino_model_name.replace("/", "_").replace(":", "_")
            dino_cache_tag = f"_{dino_model_tag}" if dino_model_tag else ""
            dino_cache_path = cache_dir / f"{movie_name}_dino{dino_cache_tag}.pkl"
            await update_progress(project_id, "matching", 13, "Preparing DINO correction index...")
            await dino_corrector.build_index(
                project.movie_path,
                sample_interval=1.0,
                cache_path=dino_cache_path,
            )
        except Exception as exc:
            logger.warning("DINO correction disabled for project={}: {}", project_id, exc)
            dino_corrector = None

    if len(active_segments) >= 220:
        applied_count, chunk_count = await _match_segments_with_visual_chunks(
            project_id,
            frame_matcher,
            project,
            active_segments,
            cache_dir,
            narration_name,
            cache_suffix,
            cache_tag,
            app_settings.concurrency.match_concurrency,
            dino_matcher=dino_corrector,
            dino_cache_tag=dino_cache_tag,
        )
        if applied_count >= max(1, int(len(active_segments) * 0.72)):
            _fill_non_narration_segments(project, segments)
            _fill_short_unmatched_segments(project, segments)
            failed_count = 0
            if use_dino:
                failed_count = await _post_validate_matches(
                    project_id,
                    frame_matcher,
                    project,
                    active_segments,
                    stage="visual-chunk",
                    progress_start=91,
                    progress_width=4,
                    max_targets=None,
                )
            allowed_failures = max(1, int(applied_count * 0.005))
            if failed_count <= allowed_failures:
                await update_progress(
                    project_id,
                    "matching",
                    99,
                    f"Visual chunk matching complete: {applied_count}/{len(active_segments)} segments via {chunk_count} chunks",
                )
                return
            logger.warning(
                "Visual chunk validation failed project={} failed={}/{}; falling back to targeted rerank",
                project_id,
                failed_count,
                applied_count,
            )
        logger.warning(
            "Visual chunk fast path not accepted project={} applied={}/{} chunks={}; falling back to per-segment matching",
            project_id,
            applied_count,
            len(active_segments),
            chunk_count,
        )
    audio_scorer = None
    enable_audio_rerank = bool(getattr(app_settings.match, "use_audio_rerank", False)) and len(active_segments) < 260
    if enable_audio_rerank:
        await update_progress(project_id, "matching", 12, "Preparing background audio references...")
        audio_scorer = AudioSimilarityScorer()
        audio_cache_dir = Path(__file__).resolve().parents[2] / "temp" / "audio_cache"
        try:
            await audio_scorer.prepare(project.movie_path, project.narration_path, output_dir=audio_cache_dir)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Audio scorer preparation failed for {project_id}: {exc}")
            audio_scorer = None
    else:
        await update_progress(project_id, "matching", 12, "Skipping audio rerank for visual-first fast matching...")
    await update_progress(project_id, "matching", 15, "Matching narration to the movie...")

    segment_tasks = [{"id": segment.id, "start": segment.narration_start, "end": segment.narration_end} for segment in active_segments]
    await update_progress(project_id, "matching", 16, f"Preparing {'DINO narration' if use_dino else 'narration'} feature cache...")
    narration_feature_map = await frame_matcher.precompute_segment_features_batch(
        project.narration_path,
        segment_tasks,
        cache_path=narration_feature_cache_path,
    )
    await update_progress(project_id, "matching", 17, "Searching sparse anchor segments...")
    anchor_time_map, sparse_anchors, anchor_monotonicity = await _build_anchor_time_map(
        frame_matcher,
        project,
        active_segments,
        narration_feature_map,
        app_settings,
    )
    logger.info(
        "anchor search project={} anchors={} monotonicity={:.3f}",
        project_id,
        len(sparse_anchors),
        anchor_monotonicity,
    )
    await update_progress(project_id, "matching", 18, "Running batch matching...")

    last_batch_progress = -1

    def on_progress(stage, progress_pct, message):
        nonlocal last_batch_progress
        mapped_progress = min(55.0, max(15.0, round(15 + 40 * progress_pct / 100, 1)))
        if mapped_progress <= last_batch_progress and progress_pct < 100:
            return
        last_batch_progress = mapped_progress
        logger.info(f"match[{stage}] {mapped_progress}% {message}")
        asyncio.run_coroutine_threadsafe(
            update_progress(project_id, "matching", mapped_progress, message),
            main_loop,
        )

    batch_results = await frame_matcher.match_all_segments_fast(
        narration_path=project.narration_path,
        segments=segment_tasks,
        sample_fps=2.5 if len(active_segments) >= 500 else 4.0,
        progress_callback=on_progress,
        movie_duration=project.movie_duration,
        narration_duration=project.narration_duration,
        allow_non_sequential=app_settings.match.allow_non_sequential,
        max_concurrent=app_settings.concurrency.match_concurrency,
        precomputed_features_map=narration_feature_map,
        expected_time_map=anchor_time_map,
    )
    batch_map = {item["id"]: item for item in batch_results}
    prefer_sequential, monotonicity = _infer_sequence_mode_from_batch(
        active_segments,
        batch_map,
        allow_non_sequential=app_settings.match.allow_non_sequential,
    )
    if sparse_anchors and anchor_monotonicity >= 0.72:
        prefer_sequential = True
        monotonicity = max(monotonicity, anchor_monotonicity)
    allow_non_sequential_effective = app_settings.match.allow_non_sequential and not prefer_sequential
    logger.info(
        "match sequence mode project={} prefer_sequential={} monotonicity={:.3f} allow_non_sequential_effective={}",
        project_id,
        prefer_sequential,
        monotonicity,
        allow_non_sequential_effective,
    )

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

    segment_index_map = {segment.id: idx for idx, segment in enumerate(active_segments)}
    context_window_cache: dict[str, Optional[tuple[float, float]]] = {}
    context_feature_cache: dict[tuple[float, float], list[dict]] = {}
    context_feature_tasks: dict[tuple[float, float], asyncio.Task] = {}

    async def get_context_payload(segment: Segment) -> tuple[Optional[tuple[float, float]], list[dict]]:
        idx = segment_index_map[segment.id]
        cached_window = context_window_cache.get(segment.id)
        if cached_window is None and segment.id not in context_window_cache:
            min_duration, max_neighbors = _context_params_for_segment(segment, batch_map.get(segment.id))
            cached_window = _build_context_window(
                active_segments,
                idx,
                min_duration=min_duration,
                max_neighbors=max_neighbors,
            )
            context_window_cache[segment.id] = cached_window
        context_window = context_window_cache.get(segment.id)
        if context_window is None:
            return None, []

        cache_key = (round(float(context_window[0]), 3), round(float(context_window[1]), 3))
        cached_features = context_feature_cache.get(cache_key)
        if cached_features is not None:
            return context_window, cached_features

        task = context_feature_tasks.get(cache_key)
        if task is None:
            async def _extract_context_features():
                return await frame_matcher._extract_segment_features(
                    project.narration_path,
                    context_window[0],
                    context_window[1],
                )

            task = asyncio.create_task(_extract_context_features())
            context_feature_tasks[cache_key] = task

        try:
            features = await task
        except Exception:
            context_feature_tasks.pop(cache_key, None)
            raise

        if features:
            context_feature_cache[cache_key] = features
        return context_window, features or []

    semaphore = asyncio.Semaphore(max(1, app_settings.concurrency.match_concurrency))
    completed = 0

    async def collect(segment: Segment):
        nonlocal completed
        async with semaphore:
            base_result = batch_map.get(segment.id)
            idx = segment_index_map[segment.id]
            context_first = _should_use_context_first(segment, base_result, app_settings)
            context_candidates: list[MatchCandidate] = []
            if context_first:
                context_window, context_features = await get_context_payload(segment)
                if context_window and context_features:
                    context_candidates = await _collect_context_candidates_for_segment(
                        frame_matcher,
                        project,
                        active_segments,
                        idx,
                        segment,
                        app_settings,
                        neighbor_hint=neighbor_hint_map.get(segment.id),
                        expected_movie_time=anchor_time_map.get(segment.id),
                        audio_scorer=audio_scorer,
                        context_window=context_window,
                        precomputed_features=context_features,
                    )
            if context_first and _context_candidates_sufficient(context_candidates, app_settings.match.high_confidence_threshold):
                segment.match_candidates = _apply_sequence_bias_to_candidates(
                    context_candidates,
                    prefer_sequential=prefer_sequential,
                )[: max(app_settings.match.candidate_top_k, 6)]
            else:
                segment.match_candidates = await _collect_candidates_for_segment(
                    frame_matcher,
                    project,
                    segment,
                    app_settings,
                    base_result,
                    neighbor_hint=neighbor_hint_map.get(segment.id),
                    audio_scorer=audio_scorer,
                    precomputed_features=narration_feature_map.get(segment.id),
                    prefer_sequential=prefer_sequential,
                    expected_movie_time_override=anchor_time_map.get(segment.id),
                )
                if context_candidates:
                    merged_candidates = _dedupe_candidates(list(segment.match_candidates) + context_candidates)
                    merged_candidates = await _rescore_candidates_with_audio(audio_scorer, segment, merged_candidates)
                    merged_candidates = await _verify_ambiguous_candidates(
                        frame_matcher,
                        project,
                        segment,
                        merged_candidates,
                    )
                    segment.match_candidates = _apply_sequence_bias_to_candidates(
                        merged_candidates,
                        prefer_sequential=prefer_sequential,
                    )[: max(app_settings.match.candidate_top_k, 6)]
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
        results = optimizer.optimize(active_segments, allow_non_sequential=allow_non_sequential_effective)
        for segment, result in zip(active_segments, results):
            candidate = result["candidate"]
            status = result["alignment_status"]
            review_required = result["review_required"]
            if candidate is None:
                _clear_match(segment, status, segment.match_reason or "No stable candidate selected")
                continue
            _apply_selected_candidate(segment, candidate, status, review_required)

    apply_optimizer_results()
    await _post_validate_matches(
        project_id,
        frame_matcher,
        project,
        active_segments,
        stage="pre-rerank",
        progress_start=80,
        progress_width=2,
        max_targets=min(max(80, int(len(active_segments) * 0.18)), 160),
    )

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
            if _needs_second_pass_rerank(segment, app_settings.match.high_confidence_threshold)
        ]
        low_confidence_total = len(low_confidence_segments)
        low_confidence_segments = _cap_rerank_segments(
            low_confidence_segments,
            len(active_segments),
            "low_confidence",
            prefer_sequential,
        )
        if low_confidence_segments:
            if len(low_confidence_segments) < low_confidence_total:
                logger.info(
                    "Reranking capped low-confidence segments project={} selected={}/{}",
                    project_id,
                    len(low_confidence_segments),
                    low_confidence_total,
                )
            completed = 0

            async def enrich(segment: Segment):
                nonlocal completed
                async with semaphore:
                    idx = segment_index_map[segment.id]
                    neighbor_hint = _compute_neighbor_hint_for_segment(active_segments, idx, segment)
                    base_result = batch_map.get(segment.id)
                    context_first = _should_use_context_first(segment, base_result, app_settings)
                    context_candidates: list[MatchCandidate] = []
                    if context_first or segment.movie_start is None:
                        context_window, context_features = await get_context_payload(segment)
                        if context_window and context_features:
                            context_candidates = await _collect_context_candidates_for_segment(
                                frame_matcher,
                                project,
                                active_segments,
                                idx,
                                segment,
                                app_settings,
                                neighbor_hint=neighbor_hint,
                                expected_movie_time=anchor_time_map.get(segment.id),
                                audio_scorer=audio_scorer,
                                context_window=context_window,
                                precomputed_features=context_features,
                            )
                    extra_candidates: list[MatchCandidate] = []
                    if not (context_first and _context_candidates_sufficient(context_candidates, app_settings.match.high_confidence_threshold)):
                        extra_candidates = await _collect_candidates_for_segment(
                            frame_matcher,
                            project,
                            segment,
                            app_settings,
                            base_result,
                            neighbor_hint=neighbor_hint,
                            audio_scorer=audio_scorer,
                            precomputed_features=narration_feature_map.get(segment.id),
                            prefer_sequential=prefer_sequential,
                            expected_movie_time_override=anchor_time_map.get(segment.id),
                        )
                    merged_candidates = _dedupe_candidates(list(segment.match_candidates) + extra_candidates + context_candidates)
                    merged_candidates = await _rescore_candidates_with_audio(audio_scorer, segment, merged_candidates)
                    merged_candidates = await _verify_ambiguous_candidates(frame_matcher, project, segment, merged_candidates)
                    segment.match_candidates = _apply_sequence_bias_to_candidates(
                        merged_candidates,
                        prefer_sequential=prefer_sequential,
                    )[: max(app_settings.match.candidate_top_k, 6)]
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

    aggressive_review_segments = [
        segment
        for segment in active_segments
        if segment.review_required
        and segment.segment_type != SegmentType.NON_MOVIE
        and not segment.skip_matching
    ]
    aggressive_review_total = len(aggressive_review_segments)
    aggressive_review_segments = _cap_rerank_segments(
        aggressive_review_segments,
        len(active_segments),
        "aggressive_review",
        prefer_sequential,
    )
    if aggressive_review_segments:
        if len(aggressive_review_segments) < aggressive_review_total:
            logger.info(
                "Aggressive recheck capped review segments project={} selected={}/{}",
                project_id,
                len(aggressive_review_segments),
                aggressive_review_total,
            )
        completed = 0
        keep_limit = max(app_settings.match.candidate_top_k + 4, 8)

        async def aggressive_recheck(segment: Segment):
            nonlocal completed
            async with semaphore:
                idx = segment_index_map[segment.id]
                neighbor_hint = _compute_neighbor_hint_for_segment(active_segments, idx, segment)
                extra_hints = _collect_recheck_hints(segment, neighbor_hint)
                base_result = batch_map.get(segment.id)
                context_first = _should_use_context_first(segment, base_result, app_settings)
                context_candidates: list[MatchCandidate] = []
                if _segment_duration(segment) <= 2.4 or segment.movie_start is None or segment.match_confidence < 0.82:
                    context_window, context_features = await get_context_payload(segment)
                    if context_window and context_features:
                        context_candidates = await _collect_context_candidates_for_segment(
                            frame_matcher,
                            project,
                            active_segments,
                            idx,
                            segment,
                            app_settings,
                            neighbor_hint=neighbor_hint,
                            expected_movie_time=anchor_time_map.get(segment.id),
                            audio_scorer=audio_scorer,
                            context_window=context_window,
                            precomputed_features=context_features,
                        )
                extra_candidates: list[MatchCandidate] = []
                if not (context_first and _context_candidates_sufficient(context_candidates, app_settings.match.high_confidence_threshold)):
                    extra_candidates = await _collect_candidates_for_segment(
                        frame_matcher,
                        project,
                        segment,
                        app_settings,
                        base_result,
                        neighbor_hint=neighbor_hint,
                        audio_scorer=audio_scorer,
                        precomputed_features=narration_feature_map.get(segment.id),
                        prefer_sequential=prefer_sequential,
                        expected_movie_time_override=anchor_time_map.get(segment.id),
                        aggressive=True,
                        extra_hints=extra_hints,
                    )
                merged_candidates = _dedupe_candidates(list(segment.match_candidates) + extra_candidates + context_candidates)
                merged_candidates = await _rescore_candidates_with_audio(audio_scorer, segment, merged_candidates)
                merged_candidates = await _verify_ambiguous_candidates(frame_matcher, project, segment, merged_candidates)
                segment.match_candidates = merged_candidates[:keep_limit]
                completed += 1
                await update_progress(
                    project_id,
                    "matching",
                    94 + int(5 * completed / max(1, len(aggressive_review_segments))),
                    f"Rechecking review segments {completed}/{len(aggressive_review_segments)}",
                )

        await asyncio.gather(*(aggressive_recheck(segment) for segment in aggressive_review_segments))
        await _maybe_llm_rerank(project, aggressive_review_segments, app_settings)
        apply_optimizer_results()

    _fill_non_narration_segments(project, segments)
    _fill_short_unmatched_segments(project, segments)
    # Do not rewrite selected matches from neighboring continuity alone. If the
    # selected anchors are wrong, smoothing them makes the whole draft wrong.
    # Export-time visual restore now validates against the narration frames.
    await _post_validate_matches(
        project_id,
        frame_matcher,
        project,
        active_segments,
        stage="final",
        progress_start=98,
        progress_width=1,
        max_targets=None,
    )


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
        project.segments = segments
        _upsert_project(project)
        await update_progress(project_id, "recognizing", 100, f"Saved {len(segments)} segments, detecting non-movie gaps...")
        await _mark_non_movie_segments(project, segments)
        project.segments = segments
        _upsert_project(project)

        project.status = ProjectStatus.MATCHING
        _upsert_project(project)
        await update_progress(project_id, "matching", 0, "Matching narration to the movie...")
        await _match_segments(project_id, project, segments, app_settings)

        stats = _compute_stats(segments)
        project.status = ProjectStatus.COMPLETED
        _upsert_project(project)
        await update_progress(
            project_id,
            "completed",
            100,
            "Video matching complete: {matched}/{total} matched, {auto} auto accepted, {review} need review, {skipped} skipped".format(
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
        project.status = ProjectStatus.COMPLETED
        _upsert_project(project)
        await update_progress(
            project_id,
            "completed",
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


def _is_weak_match_segment(
    segment: Segment,
    *,
    confidence_threshold: float = 0.78,
    visual_threshold: float = 0.70,
    include_inferred: bool = True,
    include_review_required: bool = True,
    preserve_manual_matches: bool = True,
) -> bool:
    if not segment.use_segment or segment.skip_matching:
        return False
    if segment.segment_type == SegmentType.NON_MOVIE:
        return False
    if preserve_manual_matches and segment.is_manual_match:
        return False
    if segment.movie_start is None or segment.movie_end is None:
        return True
    if include_review_required and segment.review_required:
        return True
    if segment.alignment_status in {AlignmentStatus.NEEDS_REVIEW, AlignmentStatus.UNMATCHED}:
        return True
    if include_inferred and segment.match_type in {"inferred", "fallback"}:
        return True
    if float(segment.match_confidence or 0.0) < confidence_threshold:
        return True
    visual_confidence = float(segment.visual_confidence or 0.0)
    if visual_confidence > 0.0 and visual_confidence < visual_threshold:
        return True
    return False


def _select_weak_match_segments(
    project,
    request: RematchWeakSegmentsRequest,
) -> list[Segment]:
    targets = [
        segment
        for segment in project.segments
        if _is_weak_match_segment(
            segment,
            confidence_threshold=request.confidence_threshold,
            visual_threshold=request.visual_threshold,
            include_inferred=request.include_inferred,
            include_review_required=request.include_review_required,
            preserve_manual_matches=request.preserve_manual_matches,
        )
    ]
    targets.sort(key=_segment_rerank_priority, reverse=True)
    if request.max_segments is not None:
        targets = targets[: request.max_segments]
    return targets


def _apply_targeted_rematch_result(
    project,
    segment: Segment,
    candidate_results: list[MatchCandidate],
    app_settings,
) -> bool:
    from core.matcher.global_aligner import GlobalAlignmentOptimizer

    segment.match_candidates = candidate_results
    optimizer = GlobalAlignmentOptimizer(
        auto_accept_threshold=app_settings.match.high_confidence_threshold,
        review_threshold=app_settings.match.medium_confidence_threshold,
        backtrack_penalty=app_settings.match.global_backtrack_penalty,
        duplicate_scene_penalty=app_settings.match.duplicate_scene_penalty,
    )
    idx = project.segments.index(segment)
    window_start = max(0, idx - 2)
    window_end = min(len(project.segments), idx + 3)
    target_pos = idx - window_start
    temp_segments = []
    for item in project.segments[window_start:window_end]:
        clone = item.model_copy(deep=True)
        if clone.id == segment.id:
            clone.match_candidates = candidate_results
        temp_segments.append(clone)

    prefer_sequential, _ = _infer_sequence_mode_from_segments(
        project.segments,
        app_settings.match.allow_non_sequential,
    )
    results = optimizer.optimize(
        temp_segments,
        allow_non_sequential=app_settings.match.allow_non_sequential and not prefer_sequential,
    )
    result = results[target_pos]
    candidate = result["candidate"]
    old_start = segment.movie_start
    old_end = segment.movie_end
    if candidate is None:
        _clear_match(segment, AlignmentStatus.UNMATCHED, "No stable candidate after weak-segment rematch")
        return old_start is not None or old_end is not None

    phase_locked = "phase_lock=" in str(candidate.reason or "")
    phase_failed = phase_locked and float(candidate.verification_score or 0.0) < 0.62
    if phase_failed:
        capped = max(0.0, min(float(candidate.confidence or 0.0), float(candidate.verification_score or 0.0) + 0.05))
        candidate.confidence = capped
        candidate.score = capped
        if "phase_lock_failed" not in candidate.reason:
            candidate.reason += "; phase_lock_failed"

    _apply_selected_candidate(
        segment,
        candidate,
        AlignmentStatus.REMATCHED,
        result["review_required"] or phase_failed or candidate.confidence < app_settings.match.high_confidence_threshold,
    )
    segment.is_manual_match = False
    return (
        old_start is None
        or old_end is None
        or abs(float(segment.movie_start or 0.0) - float(old_start)) > 0.20
        or abs(float(segment.movie_end or 0.0) - float(old_end)) > 0.20
    )


async def rematch_weak_segments_task(project_id: str, request: RematchWeakSegmentsRequest):
    """Re-run matching only for dynamically selected weak segments."""

    project = load_project(project_id)
    if not project:
        return

    try:
        from api.routes.settings import load_settings
        from core.audio_processor.background_audio_scorer import AudioSimilarityScorer
        from core.video_processor.frame_matcher import FrameMatcher

        if not project.segments:
            raise RuntimeError("Project has no segments to rematch")

        app_settings = load_settings()
        targets = _select_weak_match_segments(project, request)
        project.status = ProjectStatus.MATCHING
        _upsert_project(project)
        if not targets:
            stats = _compute_stats(project.segments)
            project.status = ProjectStatus.COMPLETED
            _upsert_project(project)
            await update_progress(
                project_id,
                "completed",
                100,
                "No weak segments need targeted rematch: {matched}/{total} matched, {review} need review".format(
                    matched=stats["matched"],
                    total=stats["total"],
                    review=stats["review_required"],
                ),
            )
            return

        await update_progress(project_id, "matching", 0, f"Preparing targeted weak-segment rematch for {len(targets)} segments...")
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
        await frame_matcher.build_index(
            project.movie_path,
            sample_interval=app_settings.match.sample_interval,
            cache_path=frame_cache_path,
        )

        audio_scorer = AudioSimilarityScorer()
        audio_cache_dir = Path(__file__).resolve().parents[2] / "temp" / "audio_cache"
        try:
            await audio_scorer.prepare(project.movie_path, project.narration_path, output_dir=audio_cache_dir)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Audio scorer preparation failed for weak rematch {project_id}: {exc}")
            audio_scorer = None

        prefer_sequential, _ = _infer_sequence_mode_from_segments(
            project.segments,
            app_settings.match.allow_non_sequential,
        )
        completed = 0
        changed = 0
        phase_only_accepted = 0
        for segment in targets:
            idx = project.segments.index(segment)
            neighbor_hint = _compute_neighbor_hint_for_segment(project.segments, idx, segment)
            extra_hints = _collect_recheck_hints(segment, neighbor_hint)
            precomputed = await frame_matcher._extract_segment_features(
                project.narration_path,
                segment.narration_start,
                segment.narration_end,
            )
            candidate_results: list[MatchCandidate] = []
            used_phase_only = False
            if segment.match_candidates:
                candidate_results = await _phase_lock_candidates(
                    frame_matcher,
                    project,
                    segment,
                    list(segment.match_candidates),
                    max_candidates=3,
                    precomputed_features=precomputed,
                    verify_frames=False,
                )
                candidate_results = _apply_sequence_bias_to_candidates(
                    candidate_results,
                    prefer_sequential=prefer_sequential,
                )
                top = candidate_results[0] if candidate_results else None
                if top is not None:
                    phase_score = float(top.verification_score or 0.0)
                    used_phase_only = (
                        phase_score >= 0.76
                        or (
                            phase_score >= 0.68
                            and float(top.confidence or 0.0) >= app_settings.match.high_confidence_threshold
                        )
                    )
            if used_phase_only:
                phase_only_accepted += 1
            elif request.deep_search_fallback:
                candidate_results = await _collect_candidates_for_segment(
                    frame_matcher,
                    project,
                    segment,
                    app_settings,
                    None,
                    neighbor_hint=neighbor_hint,
                    audio_scorer=audio_scorer,
                    precomputed_features=precomputed,
                    prefer_sequential=prefer_sequential,
                    aggressive=True,
                    extra_hints=extra_hints,
                )
            elif not candidate_results:
                completed += 1
                await update_progress(
                    project_id,
                    "matching",
                    min(98, int(5 + 90 * completed / max(1, len(targets)))),
                    f"Targeted weak rematch {completed}/{len(targets)}",
                )
                continue
            if _apply_targeted_rematch_result(project, segment, candidate_results, app_settings):
                changed += 1
            completed += 1
            await update_progress(
                project_id,
                "matching",
                min(98, int(5 + 90 * completed / max(1, len(targets)))),
                f"Targeted weak rematch {completed}/{len(targets)}",
            )

        validation_failed = await _post_validate_matches(
            project_id,
            frame_matcher,
            project,
            targets,
            stage="weak-rematch",
            progress_start=96,
            progress_width=2,
            max_targets=min(120, len(targets)),
        )
        stats = _compute_stats(project.segments)
        project.status = ProjectStatus.COMPLETED
        _upsert_project(project)
        await update_progress(
            project_id,
            "completed",
            100,
            "Weak-segment rematch complete: checked {checked}, changed {changed}, validation_failed {failed}, {matched}/{total} matched, {review} need review".format(
                checked=len(targets),
                changed=changed,
                failed=validation_failed,
                matched=stats["matched"],
                total=stats["total"],
                review=stats["review_required"],
            ),
        )
        logger.info(
            "Weak-segment rematch project={} checked={} changed={} phase_only_accepted={} validation_failed={}",
            project_id,
            len(targets),
            changed,
            phase_only_accepted,
            validation_failed,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception(f"Weak-segment rematch failed: {project_id}")
        project = load_project(project_id)
        if project:
            project.status = ProjectStatus.ERROR
            project.progress.message = str(exc)
            _upsert_project(project)
        await update_progress(project_id, "error", 0, f"Weak-segment rematch failed: {exc}")
    finally:
        _processing_tasks.pop(project_id, None)


@router.post("/{project_id}/start")
async def start_processing(project_id: str, background_tasks: BackgroundTasks):  # noqa: ARG001
    """Start the video matching pipeline."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project_id in _processing_tasks and not _processing_tasks[project_id].done():
        raise HTTPException(status_code=400, detail="Project is already processing")

    # 全新开始或出错后重试 → 清空重来
    if project.status in {ProjectStatus.COMPLETED, ProjectStatus.ERROR, ProjectStatus.READY_FOR_POLISH, ProjectStatus.READY_FOR_TTS}:
        project.segments = []
        project.progress = ProcessingProgress()

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
        # 如果匹配已完成（有段落带 movie_start），保留结果而不是丢弃
        matched_count = sum(1 for s in project.segments if s.movie_start is not None)
        if matched_count > 0:
            project.status = ProjectStatus.COMPLETED
            project.progress.message = f"已停止。匹配结果已保留（{matched_count} 个片段已匹配）。"
        else:
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


@router.post("/{project_id}/rematch-weak")
async def rematch_weak_segments(
    project_id: str,
    request: RematchWeakSegmentsRequest = RematchWeakSegmentsRequest(),
):
    """Re-run matching for weak/review segments in any project."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.segments:
        raise HTTPException(status_code=400, detail="Project has no segments")
    if project_id in _processing_tasks and not _processing_tasks[project_id].done():
        raise HTTPException(status_code=400, detail="Project is already processing")

    targets = _select_weak_match_segments(project, request)
    task = asyncio.create_task(rematch_weak_segments_task(project_id, request))
    _processing_tasks[project_id] = task
    project.status = ProjectStatus.MATCHING
    project.progress = ProcessingProgress(
        stage="matching",
        progress=0,
        message=f"Preparing weak-segment rematch for {len(targets)} segments...",
    )
    _upsert_project(project)
    return {
        "message": "Weak-segment rematch started",
        "project_id": project_id,
        "target_count": len(targets),
        "confidence_threshold": request.confidence_threshold,
        "visual_threshold": request.visual_threshold,
    }


@router.get("/{project_id}/progress")
async def get_progress(project_id: str):
    """Get processing progress."""

    project = load_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    task = _processing_tasks.get(project_id)
    if task is None and recover_stale_project(
        project,
        reason="检测到上次任务已中断，请重新点击开始处理或重匹配。",
    ):
        project = load_project(project_id) or project
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
        prefer_sequential=_infer_sequence_mode_from_segments(project.segments, app_settings.match.allow_non_sequential)[0],
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
    prefer_sequential, _ = _infer_sequence_mode_from_segments(project.segments, app_settings.match.allow_non_sequential)
    result = optimizer.optimize(temp_segments, allow_non_sequential=app_settings.match.allow_non_sequential and not prefer_sequential)[min(idx, 1)]
    candidate = result["candidate"]
    if candidate is None:
        _clear_match(segment, AlignmentStatus.UNMATCHED, "No stable candidate after rematch")
    else:
        _apply_selected_candidate(segment, candidate, AlignmentStatus.REMATCHED, candidate.confidence < app_settings.match.high_confidence_threshold)
        segment.is_manual_match = False
    _upsert_project(project)
    return segment
