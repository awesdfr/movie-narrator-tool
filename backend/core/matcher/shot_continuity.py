"""Shot-level continuity repair for movie/narration visual matching.

The matcher may produce high-scoring but locally inconsistent segment matches.
This module treats the narration video as the timeline authority: adjacent
speech/subtitle segments that are not separated by a real visual cut are forced
onto one continuous movie source timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
from typing import Optional

import numpy as np
from loguru import logger

from models.segment import AlignmentStatus, MatchCandidate, Segment, SegmentType

_CUT_MAP_CACHE: dict[tuple, dict[str, bool]] = {}


@dataclass
class ShotAnchor:
    segment: Segment
    target_offset: float
    source_start: float
    weight: float
    source: str


@dataclass
class ShotModel:
    intercept: float
    speed: float
    score: float
    inliers: int
    anchor_count: int


@dataclass
class ShotContinuityPlan:
    segment_ranges: dict[str, tuple[float, float, bool, float]] = field(default_factory=dict)
    cut_map: dict[str, bool] = field(default_factory=dict)
    groups: int = 0
    repaired_segments: int = 0
    inferred_segments: int = 0
    max_change_seconds: float = 0.0


def _segment_duration(segment: Segment) -> float:
    return max(0.0, float(segment.narration_end) - float(segment.narration_start))


def _status_value(segment: Segment) -> str:
    return str(getattr(segment.alignment_status, "value", segment.alignment_status) or "")


def _usable_segment(segment: Segment) -> bool:
    return (
        bool(segment.use_segment)
        and not bool(segment.skip_matching)
        and segment.segment_type != SegmentType.NON_MOVIE
    )


def _build_preview_requests(usable: list[Segment]) -> tuple[str, list[tuple[str, float]]]:
    first_end_key = f"end:{usable[0].id}"
    requests = [(first_end_key, max(0.0, float(usable[0].narration_end) - 0.05))]
    for segment in usable[1:]:
        requests.append((f"start:{segment.id}", max(0.0, float(segment.narration_start) + 0.05)))
        requests.append((f"end:{segment.id}", max(0.0, float(segment.narration_end) - 0.05)))
    return first_end_key, requests


def _segment_source_range(segment: Segment) -> Optional[tuple[float, float]]:
    if segment.movie_start is not None and segment.movie_end is not None and segment.movie_end > segment.movie_start:
        return float(segment.movie_start), float(segment.movie_end)
    candidates = sorted(
        segment.match_candidates or [],
        key=lambda item: item.score or item.confidence or 0.0,
        reverse=True,
    )
    for candidate in candidates[:3]:
        if candidate.end > candidate.start and max(float(candidate.score or 0.0), float(candidate.confidence or 0.0)) >= 0.58:
            return float(candidate.start), float(candidate.end)
    return None


def _source_jump_cut_map(usable: list[Segment]) -> dict[str, bool]:
    cuts: dict[str, bool] = {}
    previous_segment: Optional[Segment] = None
    previous_range: Optional[tuple[float, float]] = None

    for segment in usable:
        current_range = _segment_source_range(segment)
        if previous_segment is None:
            previous_segment = segment
            previous_range = current_range
            continue

        is_cut = False
        if current_range is not None and previous_range is not None:
            target_gap = float(segment.narration_start) - float(previous_segment.narration_end)
            source_gap = current_range[0] - previous_range[1]
            large_jump = source_gap > max(8.0, target_gap + 6.0)
            large_backtrack = source_gap < -6.0
            gap_mismatch = abs(source_gap - max(0.0, target_gap)) > max(10.0, abs(target_gap) * 4.0 + 6.0)
            is_cut = bool(large_jump or large_backtrack or gap_mismatch)
        cuts[segment.id] = is_cut

        previous_segment = segment
        previous_range = current_range
    return cuts


def _score_preview_cut(cv2_module, previous_frame, current_frame) -> bool:
    if previous_frame is None or current_frame is None:
        return True
    prev_hist = cv2_module.calcHist([previous_frame], [0], None, [40], [0, 256]).astype("float32")
    curr_hist = cv2_module.calcHist([current_frame], [0], None, [40], [0, 256]).astype("float32")
    prev_hist /= max(float(prev_hist.sum()), 1.0)
    curr_hist /= max(float(curr_hist.sum()), 1.0)
    hist_corr = float((cv2_module.compareHist(prev_hist, curr_hist, cv2_module.HISTCMP_CORREL) + 1.0) / 2.0)
    mean_diff = float(
        np.mean(np.abs(previous_frame.astype("float32") - current_frame.astype("float32")))
    ) / 255.0
    # Prefer false negatives over false positives. A missed cut is later
    # guarded by the anchor model; a false cut creates stutter.
    return bool(
        (hist_corr < 0.58 and mean_diff > 0.075)
        or (hist_corr < 0.70 and mean_diff > 0.125)
        or mean_diff > 0.235
    )


def _resolve_ffmpeg_bin() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[3]
    local_tools = repo_root / ".tools"
    local_candidates = sorted(local_tools.glob("ffmpeg-*essentials_build/bin/ffmpeg.exe"))
    if local_candidates:
        return str(local_candidates[-1])
    return shutil.which("ffmpeg")


def _decode_preview_series_ffmpeg(narration_path: str, preview_fps: float = 4.0) -> list[np.ndarray]:
    ffmpeg_bin = _resolve_ffmpeg_bin()
    if not ffmpeg_bin:
        return []

    width = 192
    height = 108
    crop_height = max(1, int(height * 0.72))
    frame_bytes = width * height

    def run_decode(hwaccel: Optional[str]) -> list[np.ndarray]:
        cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if hwaccel:
            cmd += ["-hwaccel", hwaccel]
        cmd += [
            "-i",
            narration_path,
            "-an",
            "-vf",
            f"fps={preview_fps:.3f},scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=frame_bytes * 64,
        )
        frames: list[np.ndarray] = []
        try:
            while True:
                raw = proc.stdout.read(frame_bytes) if proc.stdout else b""
                if not raw or len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width).copy()
                frames.append(frame[:crop_height, :])
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait(timeout=30)
        return frames if proc.returncode == 0 else []

    for hwaccel in ("cuda", "d3d11va", "auto", None):
        try:
            frames = run_decode(hwaccel)
            if frames:
                return frames
        except Exception as exc:
            logger.debug("FFmpeg preview decode failed (hwaccel={}): {}", hwaccel, exc)
    return []


def _candidate_weight(segment: Segment, candidate: Optional[MatchCandidate]) -> float:
    if candidate is None:
        confidence = float(segment.match_confidence or 0.0)
        visual = float(segment.visual_confidence or 0.0)
        verify = 0.0
        stability = float(segment.stability_score or 0.0)
        rank_gap = 0.0
    else:
        confidence = float(candidate.confidence or candidate.score or 0.0)
        visual = float(candidate.visual_confidence or 0.0)
        verify = float(candidate.verification_score or 0.0)
        stability = float(candidate.stability_score or 0.0)
        rank_gap = float(candidate.rank_gap or 0.0)

    weight = confidence * 1.5 + visual * 1.1 + verify * 1.4 + stability * 0.7
    weight += min(0.25, max(0.0, rank_gap) * 3.0)

    status = _status_value(segment)
    if segment.is_manual_match:
        weight += 1.4
    elif status in {"auto_accepted", "rematched"} and not segment.review_required:
        weight += 0.5
    if segment.review_required:
        weight *= 0.75
    if segment.match_type == "inferred":
        weight *= 0.72
    if segment.segment_type == SegmentType.NO_NARRATION:
        weight *= 0.60
    return max(0.0, weight)


def _anchors_for_group(group: list[Segment], group_start: float) -> list[ShotAnchor]:
    anchors: list[ShotAnchor] = []
    seen: set[tuple[str, int]] = set()

    for segment in group:
        target_offset = float(segment.narration_start) - group_start
        if segment.movie_start is not None and segment.movie_end is not None:
            key = (segment.id, int(round(float(segment.movie_start) * 100)))
            if key not in seen:
                seen.add(key)
                anchors.append(
                    ShotAnchor(
                        segment=segment,
                        target_offset=target_offset,
                        source_start=float(segment.movie_start),
                        weight=_candidate_weight(segment, None),
                        source="selected",
                    )
                )

        candidates = sorted(
            segment.match_candidates or [],
            key=lambda item: item.score or item.confidence or 0.0,
            reverse=True,
        )[:5]
        for candidate in candidates:
            score = max(float(candidate.score or 0.0), float(candidate.confidence or 0.0))
            if (
                score < 0.68
                and float(candidate.visual_confidence or 0.0) < 0.76
                and float(candidate.verification_score or 0.0) < 0.55
            ):
                continue
            key = (segment.id, int(round(float(candidate.start) * 100)))
            if key in seen:
                continue
            seen.add(key)
            anchors.append(
                ShotAnchor(
                    segment=segment,
                    target_offset=target_offset,
                    source_start=float(candidate.start),
                    weight=_candidate_weight(segment, candidate) * 0.92,
                    source=candidate.source or "candidate",
                )
            )

    return [anchor for anchor in anchors if anchor.weight > 0.1]


def _weighted_median(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(max(0.0, weight) for _value, weight in ordered)
    if total <= 0.0:
        return float(np.median([value for value, _weight in ordered]))
    cursor = 0.0
    half = total / 2.0
    for value, weight in ordered:
        cursor += max(0.0, weight)
        if cursor >= half:
            return float(value)
    return float(ordered[-1][0])


def _fit_model_for_anchors(
    anchors: list[ShotAnchor],
    group_duration: float,
    movie_duration: float,
) -> Optional[ShotModel]:
    if not anchors:
        return None

    anchors = sorted(anchors, key=lambda item: item.weight, reverse=True)[:24]
    tolerance = max(0.45, min(1.8, group_duration * 0.16))
    models: list[tuple[float, float]] = []

    for anchor in anchors:
        models.append((anchor.source_start - anchor.target_offset, 1.0))

    for left_index, left in enumerate(anchors):
        for right in anchors[left_index + 1 :]:
            dx = right.target_offset - left.target_offset
            if abs(dx) < 0.65:
                continue
            speed = (right.source_start - left.source_start) / dx
            if 0.55 <= speed <= 1.85:
                intercept = left.source_start - speed * left.target_offset
                models.append((intercept, speed))

    best: Optional[ShotModel] = None
    for intercept, speed in models:
        residuals: list[tuple[float, float, ShotAnchor]] = []
        score = 0.0
        inliers = 0
        for anchor in anchors:
            predicted = intercept + speed * anchor.target_offset
            residual = abs(predicted - anchor.source_start)
            residuals.append((residual, anchor.weight, anchor))
            if residual <= tolerance:
                inliers += 1
                score += anchor.weight * (1.0 - residual / max(tolerance, 1e-6))
        score -= abs(speed - 1.0) * 0.30
        if best is None or score > best.score:
            inlier_values = [
                (anchor.source_start - speed * anchor.target_offset, weight)
                for residual, weight, anchor in residuals
                if residual <= tolerance
            ]
            refined_intercept = _weighted_median(inlier_values) if inlier_values else intercept
            best = ShotModel(
                intercept=float(refined_intercept),
                speed=float(speed),
                score=float(score),
                inliers=int(inliers),
                anchor_count=len(anchors),
            )

    if best is None:
        return None

    min_inliers = 2 if len(anchors) >= 2 else 1
    if best.inliers < min_inliers:
        return None
    if len(anchors) >= 3 and best.score < 1.4:
        return None
    if len(anchors) <= 2 and best.score < 0.75:
        return None

    max_source_end = best.intercept + best.speed * group_duration
    if movie_duration > 0.0:
        if max_source_end > movie_duration:
            best.intercept = max(0.0, movie_duration - best.speed * group_duration)
        if best.intercept < 0.0:
            best.intercept = 0.0
    return best


def detect_narration_cut_map(narration_path: str | None, segments: list[Segment]) -> dict[str, bool]:
    usable = [segment for segment in segments if _usable_segment(segment)]
    if not narration_path or len(usable) < 2:
        return {}

    try:
        import cv2
    except Exception:
        return _source_jump_cut_map(usable)

    path = Path(str(narration_path))
    try:
        stat = path.stat()
        file_sig = (str(path), int(stat.st_mtime), int(stat.st_size))
    except OSError:
        file_sig = (str(path), 0, 0)
    segment_sig = tuple(
        (segment.id, round(float(segment.narration_start), 3), round(float(segment.narration_end), 3))
        for segment in usable
    )
    cache_key = (file_sig, segment_sig)
    cached = _CUT_MAP_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    first_end_key, preview_requests = _build_preview_requests(usable)
    preview_fps = 4.0
    preview_frames = _decode_preview_series_ffmpeg(str(path), preview_fps=preview_fps)
    if preview_frames:
        frames = {
            key: preview_frames[max(0, min(len(preview_frames) - 1, int(round(timestamp * preview_fps))))]
            for key, timestamp in preview_requests
        }
        cuts: dict[str, bool] = {}
        previous_frame = frames.get(first_end_key)
        for segment in usable[1:]:
            current_frame = frames.get(f"start:{segment.id}")
            cuts[segment.id] = _score_preview_cut(cv2, previous_frame, current_frame)
            previous_frame = frames.get(f"end:{segment.id}")
        _CUT_MAP_CACHE[cache_key] = dict(cuts)
        return cuts

    capture = cv2.VideoCapture(str(narration_path))
    if not capture.isOpened():
        fallback = _source_jump_cut_map(usable)
        _CUT_MAP_CACHE[cache_key] = dict(fallback)
        return fallback

    def preprocess(frame):
        if frame is None:
            return None
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return None
        frame = frame[: max(1, int(height * 0.72)), :]
        frame = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def score_cut(previous_frame, current_frame) -> bool:
        return _score_preview_cut(cv2, previous_frame, current_frame)

    requests: list[tuple[int, float, str]] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 24.0
        for key, timestamp in preview_requests:
            requests.append((max(0, int(round(timestamp * fps))), timestamp, key))

        frames: dict[str, np.ndarray | None] = {}
        requests.sort(key=lambda item: item[0])
        seek_gap_frames = max(1, int(fps * 4.0))
        current_frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        for target_frame, _timestamp, key in requests:
            if target_frame < current_frame_index or target_frame - current_frame_index > seek_gap_frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_frame_index = target_frame

            frame = None
            ok = False
            while current_frame_index <= target_frame:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                current_frame_index += 1
            frames[key] = preprocess(frame) if ok and frame is not None else None

        cuts: dict[str, bool] = {}
        previous_frame = frames.get(first_end_key)
        for segment in usable[1:]:
            current_frame = frames.get(f"start:{segment.id}")
            cuts[segment.id] = score_cut(previous_frame, current_frame)
            previous_frame = frames.get(f"end:{segment.id}")
    finally:
        capture.release()
    _CUT_MAP_CACHE[cache_key] = dict(cuts)
    return cuts


def _build_groups(segments: list[Segment], cut_map: dict[str, bool]) -> list[list[Segment]]:
    usable = sorted(
        [segment for segment in segments if _usable_segment(segment)],
        key=lambda item: (float(item.narration_start), float(item.narration_end)),
    )
    groups: list[list[Segment]] = []
    for segment in usable:
        starts_new_group = not groups or bool(cut_map.get(segment.id, False))
        if starts_new_group:
            groups.append([segment])
        else:
            groups[-1].append(segment)
    return groups


def plan_shot_continuity(project, segments: list[Segment]) -> ShotContinuityPlan:
    cut_map = detect_narration_cut_map(getattr(project, "narration_path", None), segments)
    groups = _build_groups(segments, cut_map)
    movie_duration = float(getattr(project, "movie_duration", 0.0) or 0.0)
    plan = ShotContinuityPlan(cut_map=cut_map, groups=len(groups))

    for group in groups:
        if not group:
            continue
        group_start = float(group[0].narration_start)
        group_end = max(float(segment.narration_end) for segment in group)
        group_duration = max(0.0, group_end - group_start)
        if group_duration <= 0.0:
            continue

        anchors = _anchors_for_group(group, group_start)
        model = _fit_model_for_anchors(anchors, group_duration, movie_duration)
        if model is None:
            continue

        for segment in group:
            offset_start = float(segment.narration_start) - group_start
            offset_end = float(segment.narration_end) - group_start
            source_start = model.intercept + model.speed * offset_start
            source_end = model.intercept + model.speed * offset_end
            if movie_duration > 0.0 and source_end > movie_duration:
                overflow = source_end - movie_duration
                source_start = max(0.0, source_start - overflow)
                source_end = movie_duration
            if source_start < 0.0:
                source_end -= source_start
                source_start = 0.0
            old_start = float(segment.movie_start) if segment.movie_start is not None else None
            changed = old_start is None or abs(old_start - source_start) > 0.18
            is_inferred = changed or segment.match_type == "inferred" or segment.review_required
            plan.segment_ranges[segment.id] = (
                float(source_start),
                float(max(source_start, source_end)),
                bool(is_inferred),
                float(max(0.0, min(1.0, model.score / max(1.0, model.anchor_count * 2.0)))),
            )
            if changed:
                plan.repaired_segments += 1
                if old_start is not None:
                    plan.max_change_seconds = max(plan.max_change_seconds, abs(old_start - source_start))
            if is_inferred:
                plan.inferred_segments += 1

    return plan


def apply_shot_continuity(project, segments: list[Segment]) -> dict[str, float | int]:
    plan = plan_shot_continuity(project, segments)
    if not plan.segment_ranges:
        return {
            "groups": plan.groups,
            "repaired_segments": 0,
            "inferred_segments": 0,
            "max_change_seconds": 0.0,
        }

    for segment in segments:
        item = plan.segment_ranges.get(segment.id)
        if item is None or segment.is_manual_match:
            continue
        source_start, source_end, is_inferred, score = item
        old_start = float(segment.movie_start) if segment.movie_start is not None else None
        old_end = float(segment.movie_end) if segment.movie_end is not None else None
        changed = (
            old_start is None
            or old_end is None
            or abs(old_start - source_start) > 0.18
            or abs(old_end - source_end) > 0.18
        )
        if not changed:
            continue

        duration_gap = abs((source_end - source_start) - _segment_duration(segment))
        previous_confidence = float(segment.match_confidence or 0.0)
        continuity_confidence = max(float(score), min(previous_confidence, 0.92))
        if score < 0.55:
            continuity_confidence = min(continuity_confidence, 0.78)
        candidate = MatchCandidate(
            id=f"{segment.id}_shot_continuity",
            start=source_start,
            end=source_end,
            score=continuity_confidence,
            confidence=continuity_confidence,
            visual_confidence=max(min(float(segment.visual_confidence or 0.0), 0.92), score),
            audio_confidence=float(segment.audio_confidence or 0.0),
            temporal_confidence=max(float(segment.temporal_confidence or 0.0), 0.90),
            rank_gap=max(0.02, float(score) * 0.04),
            verification_score=float(score),
            stability_score=max(float(segment.stability_score or 0.0), 0.75),
            duration_gap=duration_gap,
            match_count=max(1, int(round(_segment_duration(segment)))),
            reason="Shot continuity retime from narration visual group",
            source="shot_continuity",
            rank=1,
        )
        existing = [item for item in segment.match_candidates if item.id != candidate.id]
        segment.match_candidates = [candidate] + existing[:7]
        segment.selected_candidate_id = candidate.id
        segment.movie_start = source_start
        segment.movie_end = source_end
        segment.match_confidence = candidate.confidence
        segment.visual_confidence = candidate.visual_confidence
        segment.temporal_confidence = candidate.temporal_confidence
        segment.stability_score = candidate.stability_score
        segment.duration_gap = duration_gap
        if segment.alignment_status != AlignmentStatus.MANUAL:
            segment.alignment_status = AlignmentStatus.AUTO_ACCEPTED if score >= 0.55 else AlignmentStatus.NEEDS_REVIEW
        segment.review_required = bool(segment.review_required and score < 0.72)
        segment.match_type = "inferred" if is_inferred else "exact"
        marker = f"shot_continuity=1, score={score:.2f}"
        segment.match_reason = f"{segment.match_reason}; {marker}" if segment.match_reason else marker
        segment.evidence_summary = f"{segment.evidence_summary}; {marker}" if segment.evidence_summary else marker
        segment.estimated_boundary_error = min(float(segment.estimated_boundary_error or 1.0), 0.35)

    logger.info(
        "Shot continuity repair: groups={}, repaired={}, inferred={}, max_change={:.2f}s",
        plan.groups,
        plan.repaired_segments,
        plan.inferred_segments,
        plan.max_change_seconds,
    )
    return {
        "groups": plan.groups,
        "repaired_segments": plan.repaired_segments,
        "inferred_segments": plan.inferred_segments,
        "max_change_seconds": plan.max_change_seconds,
    }
